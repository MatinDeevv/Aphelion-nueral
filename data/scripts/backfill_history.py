"""Backfill synchronized dual-broker history in chunks and persist daily QA coverage."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys
from typing import Iterable


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent
DATA_ROOT = SCRIPT_PATH.parents[1]
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mt5pipe.backfill.sync import (
    BrokerBackfillRunSummary,
    format_synchronized_backfill_summary,
    run_synchronized_tick_backfill,
)
from mt5pipe.config.loader import load_config
from mt5pipe.merge.canonical import merge_canonical_date_range
from mt5pipe.quality.merge_qa import (
    build_daily_merge_qa_report,
    write_daily_merge_qa_report,
)
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths
from mt5pipe.utils.logging import setup_logging
from validate_coverage import CoverageDay, load_coverage_days


def iter_chunks(start_date: dt.date, end_date: dt.date, chunk_days: int) -> Iterable[tuple[dt.date, dt.date]]:
    """Yield inclusive date chunks of at most ``chunk_days`` days."""

    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive")
    current = start_date
    while current <= end_date:
        chunk_end = min(end_date, current + dt.timedelta(days=chunk_days - 1))
        yield current, chunk_end
        current = chunk_end + dt.timedelta(days=1)


def classify_history_day(row: CoverageDay) -> str:
    """Return the daily status label for the backfill summary table."""

    if row.dual_source_ratio < 0.05:
        return "SPARSE"
    if row.dual_source_ratio < 0.10:
        return "WARN"
    return "OK"


def format_history_table(rows: list[CoverageDay]) -> str:
    """Render the per-date coverage table requested by the operator prompt."""

    lines = [
        "date         broker_a_ticks   broker_b_ticks   canonical_rows   dual_source_ratio   status",
    ]
    for row in rows:
        lines.append(
            f"{row.date.isoformat()} "
            f"{row.broker_a_ticks:>16,} "
            f"{row.broker_b_ticks:>16,} "
            f"{row.canonical_rows:>16,} "
            f"{row.dual_source_ratio:>19.4f} "
            f"{classify_history_day(row)}"
        )
    return "\n".join(lines)


def print_chunk_summary(
    chunk_start: dt.date,
    chunk_end: dt.date,
    broker_summaries: dict[str, BrokerBackfillRunSummary],
    report_df,
) -> None:
    """Print a concise per-chunk progress summary after backfill, merge, and QA."""

    chunk_canonical_rows = int(report_df["canonical_rows"].sum()) if not report_df.is_empty() else 0
    chunk_canonical_dual_rows = (
        int(report_df["canonical_dual_rows"].sum()) if not report_df.is_empty() else 0
    )
    chunk_dual_ratio = (
        (chunk_canonical_dual_rows / chunk_canonical_rows) if chunk_canonical_rows else 0.0
    )
    print("")
    print(
        format_synchronized_backfill_summary(
            broker_summaries,
            start_date=chunk_start,
            end_date=chunk_end,
        )
    )
    print(
        "chunk_summary: "
        f"canonical_rows={chunk_canonical_rows:,} "
        f"canonical_dual_rows={chunk_canonical_dual_rows:,} "
        f"dual_source_ratio={chunk_dual_ratio:.4f}"
    )
    if chunk_dual_ratio < 0.05:
        print(
            "warning: "
            f"chunk {chunk_start.isoformat()} -> {chunk_end.isoformat()} is sparse "
            f"(dual_source_ratio={chunk_dual_ratio:.4f})"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="date_from", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--symbol", required=True, help="Canonical symbol, e.g. XAUUSD")
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=7,
        help="Inclusive chunk size in UTC days for MT5 backfill",
    )
    parser.add_argument(
        "--config",
        default=str(DATA_ROOT / "config" / "pipeline.yaml"),
        help="Path to pipeline.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    start_date = dt.date.fromisoformat(args.date_from)
    end_date = dt.date.fromisoformat(args.date_to)
    cfg = load_config(Path(args.config))
    setup_logging(cfg.logging.level, cfg.logging.json_output)

    paths = StoragePaths(cfg.storage.root)
    store = ParquetStore(cfg.storage.compression, cfg.storage.parquet_row_group_size)
    broker_a = cfg.get_broker("broker_a")
    broker_b = cfg.get_broker("broker_b")
    merge_cfg = cfg.merge.model_copy(update={"hard_fail_on_low_dual_source": False})

    for chunk_start, chunk_end in iter_chunks(start_date, end_date, args.chunk_days):
        print("")
        print(f"=== chunk {chunk_start.isoformat()} -> {chunk_end.isoformat()} ===")
        broker_summaries = run_synchronized_tick_backfill(
            broker_a,
            broker_b,
            args.symbol,
            chunk_start,
            chunk_end,
            paths=paths,
            store=store,
            checkpoint_db_path=paths.checkpoint_db_path(),
            backfill_cfg=cfg.backfill,
        )
        merge_canonical_date_range(
            broker_a.broker_id,
            broker_b.broker_id,
            args.symbol,
            chunk_start,
            chunk_end,
            paths,
            store,
            merge_cfg,
            broker_a.priority,
            broker_b.priority,
        )
        report_df = build_daily_merge_qa_report(
            paths,
            store,
            broker_a.broker_id,
            broker_b.broker_id,
            args.symbol,
            chunk_start,
            chunk_end,
            expected_bucket_ms=cfg.merge.bucket_ms,
        )
        written = write_daily_merge_qa_report(report_df, paths, store, args.symbol)
        print(f"qa_report_rows_written={written:,}")
        print_chunk_summary(chunk_start, chunk_end, broker_summaries, report_df)

    rows = load_coverage_days(paths, store, args.symbol, start_date, end_date)
    print("")
    print("=== coverage summary ===")
    print(format_history_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
