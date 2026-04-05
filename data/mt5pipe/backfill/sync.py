"""Synchronized dual-broker tick backfill orchestration."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from mt5pipe.backfill.engine import BackfillEngine
from mt5pipe.config.models import BackfillConfig, BrokerConfig
from mt5pipe.mt5.connection import MT5Connection
from mt5pipe.quality.merge_qa import (
    RawTickRangeStats,
    assert_synchronized_raw_tick_coverage,
    collect_raw_tick_range_stats,
    iter_utc_dates,
)
from mt5pipe.storage.checkpoint_db import CheckpointDB
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths


@dataclass(frozen=True)
class BrokerBackfillRunSummary:
    broker_id: str
    symbol: str
    days_requested: int
    days_written: int
    total_ticks_written: int
    first_timestamp: dt.datetime | None
    last_timestamp: dt.datetime | None
    rows_added_this_run: int
    newly_covered_days: int

    @classmethod
    def from_stats(
        cls,
        before: RawTickRangeStats,
        after: RawTickRangeStats,
    ) -> "BrokerBackfillRunSummary":
        return cls(
            broker_id=after.broker_id,
            symbol=after.symbol,
            days_requested=after.days_requested,
            days_written=after.days_written,
            total_ticks_written=after.total_ticks_written,
            first_timestamp=after.first_timestamp,
            last_timestamp=after.last_timestamp,
            rows_added_this_run=max(after.total_ticks_written - before.total_ticks_written, 0),
            newly_covered_days=max(after.days_written - before.days_written, 0),
        )


def run_synchronized_tick_backfill(
    broker_a_cfg: BrokerConfig,
    broker_b_cfg: BrokerConfig,
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
    *,
    paths: StoragePaths,
    store: ParquetStore,
    checkpoint_db_path: Path,
    backfill_cfg: BackfillConfig,
    hours_start_utc: dt.time | None = None,
    hours_end_utc: dt.time | None = None,
) -> dict[str, BrokerBackfillRunSummary]:
    """Backfill the same UTC date range for both brokers and verify parity."""
    before = {
        broker_a_cfg.broker_id: collect_raw_tick_range_stats(
            paths,
            store,
            broker_a_cfg.broker_id,
            symbol,
            start_date,
            end_date,
            hours_start_utc=hours_start_utc,
            hours_end_utc=hours_end_utc,
        ),
        broker_b_cfg.broker_id: collect_raw_tick_range_stats(
            paths,
            store,
            broker_b_cfg.broker_id,
            symbol,
            start_date,
            end_date,
            hours_start_utc=hours_start_utc,
            hours_end_utc=hours_end_utc,
        ),
    }

    dates = iter_utc_dates(start_date, end_date)
    for broker_cfg in (broker_a_cfg, broker_b_cfg):
        conn = MT5Connection(broker_cfg)
        db = CheckpointDB(checkpoint_db_path)
        try:
            with conn.connect():
                engine = BackfillEngine(conn, paths, store, db, backfill_cfg)
                for date in dates:
                    engine.backfill_ticks_for_utc_day(
                        symbol,
                        date,
                        hours_start_utc=hours_start_utc,
                        hours_end_utc=hours_end_utc,
                    )
        finally:
            db.close()

    after_a, after_b = assert_synchronized_raw_tick_coverage(
        paths,
        store,
        broker_a_cfg.broker_id,
        broker_b_cfg.broker_id,
        symbol,
        start_date,
        end_date,
        hours_start_utc=hours_start_utc,
        hours_end_utc=hours_end_utc,
    )

    return {
        broker_a_cfg.broker_id: BrokerBackfillRunSummary.from_stats(before[broker_a_cfg.broker_id], after_a),
        broker_b_cfg.broker_id: BrokerBackfillRunSummary.from_stats(before[broker_b_cfg.broker_id], after_b),
    }


def format_synchronized_backfill_summary(
    summaries: dict[str, BrokerBackfillRunSummary],
    *,
    start_date: dt.date,
    end_date: dt.date,
    hours_start_utc: dt.time | None = None,
    hours_end_utc: dt.time | None = None,
) -> str:
    """Format per-broker synchronized backfill coverage for terminal output."""
    if hours_start_utc is None:
        window_label = "full-day UTC"
    else:
        window_label = f"{hours_start_utc.strftime('%H:%M')}-{hours_end_utc.strftime('%H:%M')} UTC"

    lines = [
        f"Synchronized tick backfill {start_date.isoformat()} -> {end_date.isoformat()} ({window_label})",
        "broker      days_req   days_written   ticks_in_range   rows_added   new_days   first_timestamp              last_timestamp",
    ]

    for broker_id in sorted(summaries):
        summary = summaries[broker_id]
        first_ts = summary.first_timestamp.isoformat() if summary.first_timestamp else "-"
        last_ts = summary.last_timestamp.isoformat() if summary.last_timestamp else "-"
        lines.append(
            f"{broker_id:<11} "
            f"{summary.days_requested:>8} "
            f"{summary.days_written:>14} "
            f"{summary.total_ticks_written:>16,} "
            f"{summary.rows_added_this_run:>11,} "
            f"{summary.newly_covered_days:>9} "
            f"{first_ts:<28} "
            f"{last_ts}"
        )

    return "\n".join(lines)
