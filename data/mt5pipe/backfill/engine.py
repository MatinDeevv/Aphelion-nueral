"""Resumable chunked backfill engine."""

from __future__ import annotations

import datetime as dt

from mt5pipe.config.models import BackfillConfig, BrokerConfig, StorageConfig
from mt5pipe.ingestion.bars import fetch_bars_chunk, store_bars_by_date
from mt5pipe.ingestion.history import (
    fetch_history_deals,
    fetch_history_orders,
    store_history_deals,
    store_history_orders,
)
from mt5pipe.ingestion.snapshots import capture_symbol_metadata, capture_symbol_universe
from mt5pipe.ingestion.ticks import deduplicate_ticks, fetch_ticks_chunk, store_ticks_by_date
from mt5pipe.models.checkpoint import IngestionCheckpoint
from mt5pipe.mt5.connection import MT5Connection
from mt5pipe.storage.checkpoint_db import CheckpointDB
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths
from mt5pipe.utils.logging import get_logger
from mt5pipe.utils.time import dt_to_utc, utc_now

log = get_logger(__name__)


def _resolve_utc_day_bounds(
    date: dt.date,
    hours_start_utc: dt.time | None = None,
    hours_end_utc: dt.time | None = None,
) -> tuple[dt.datetime, dt.datetime]:
    """Resolve a single UTC day into [start, end) datetimes."""
    if (hours_start_utc is None) != (hours_end_utc is None):
        raise ValueError("hours_start_utc and hours_end_utc must both be set or both be omitted")

    start_of_day = dt.datetime.combine(date, dt.time(0, 0), tzinfo=dt.timezone.utc)
    if hours_start_utc is None:
        return start_of_day, start_of_day + dt.timedelta(days=1)

    if hours_start_utc >= hours_end_utc:
        raise ValueError("UTC intraday backfill window must satisfy hours_start_utc < hours_end_utc")

    window_start = dt.datetime.combine(date, hours_start_utc, tzinfo=dt.timezone.utc)
    window_end = dt.datetime.combine(date, hours_end_utc, tzinfo=dt.timezone.utc)
    return window_start, window_end


class BackfillEngine:
    """Coordinates resumable chunked backfill across data types."""

    def __init__(
        self,
        conn: MT5Connection,
        paths: StoragePaths,
        store: ParquetStore,
        checkpoint_db: CheckpointDB,
        backfill_cfg: BackfillConfig,
    ) -> None:
        self._conn = conn
        self._paths = paths
        self._store = store
        self._db = checkpoint_db
        self._cfg = backfill_cfg

    def backfill_ticks(
        self,
        symbol: str,
        start: dt.datetime,
        end: dt.datetime,
        *,
        respect_checkpoint: bool = True,
    ) -> int:
        """Backfill ticks in time chunks. Resumable via checkpoint."""
        broker_id = self._conn.broker_id
        start = dt_to_utc(start)
        end = dt_to_utc(end)

        # Resume from checkpoint
        cp = self._db.get_checkpoint(broker_id, symbol, "ticks")
        checkpoint_last_timestamp = cp.last_timestamp_utc if cp else start
        checkpoint_last_msc = cp.last_time_msc if cp else 0
        if respect_checkpoint and cp and cp.last_timestamp_utc > start:
            log.info(
                "backfill_ticks_resume",
                broker=broker_id,
                symbol=symbol,
                from_ts=cp.last_timestamp_utc.isoformat(),
            )
            start = cp.last_timestamp_utc

        chunk_delta = dt.timedelta(hours=self._cfg.tick_chunk_hours)
        total_rows = cp.rows_ingested if cp else 0
        job_id = self._db.start_job("backfill_ticks", broker_id, symbol)

        chunk_from = start
        empty_chunks = 0
        max_empty_chunks = 10  # Stop after N consecutive empty chunks
        total_span = max((end - start).total_seconds(), 1)

        try:
            while chunk_from < end:
                chunk_to = min(chunk_from + chunk_delta, end)

                self._conn.ensure_connected()
                df = fetch_ticks_chunk(self._conn, symbol, chunk_from, chunk_to)
                df = deduplicate_ticks(df)

                progress_pct = min(100.0, (chunk_to - start).total_seconds() / total_span * 100)

                if df.is_empty():
                    empty_chunks += 1
                    log.debug(
                        "backfill_ticks_empty_chunk",
                        broker=broker_id,
                        symbol=symbol,
                        from_ts=chunk_from.isoformat(),
                        empty_count=empty_chunks,
                    )
                    if empty_chunks >= max_empty_chunks:
                        log.info("backfill_ticks_no_more_data", broker=broker_id, symbol=symbol)
                        break
                else:
                    empty_chunks = 0
                    rows_written = store_ticks_by_date(
                        df,
                        broker_id,
                        symbol,
                        self._paths,
                        self._store,
                        self._db,
                    )
                    total_rows += rows_written

                    # Get the last timestamp from the chunk
                    last_msc = df["time_msc"].max()

                    self._db.upsert_checkpoint(IngestionCheckpoint(
                        broker_id=broker_id,
                        symbol=symbol,
                        data_type="ticks",
                        last_timestamp_utc=max(checkpoint_last_timestamp, chunk_to),
                        last_time_msc=max(checkpoint_last_msc, last_msc),
                        rows_ingested=total_rows,
                        updated_at=utc_now(),
                    ))

                    log.info(
                        "backfill_ticks_chunk",
                        broker=broker_id,
                        symbol=symbol,
                        chunk_rows=rows_written,
                        total_rows=total_rows,
                        progress=f"{progress_pct:.1f}%",
                        chunk_end=chunk_to.isoformat(),
                    )

                chunk_from = chunk_to

            self._db.finish_job(job_id, "completed", total_rows)
            log.info("backfill_ticks_done", broker=broker_id, symbol=symbol, total_rows=total_rows)

        except Exception as exc:
            self._db.finish_job(job_id, "failed", total_rows, str(exc))
            log.error("backfill_ticks_failed", broker=broker_id, symbol=symbol, error=str(exc))
            raise

        return total_rows

    def backfill_ticks_for_utc_day(
        self,
        symbol: str,
        date: dt.date,
        *,
        hours_start_utc: dt.time | None = None,
        hours_end_utc: dt.time | None = None,
    ) -> int:
        """Backfill one UTC day, optionally restricted to a UTC intraday window."""
        broker_id = self._conn.broker_id
        day_start, day_end = _resolve_utc_day_bounds(date, hours_start_utc, hours_end_utc)
        cp = self._db.get_checkpoint(broker_id, symbol, "ticks")
        if cp and cp.last_timestamp_utc >= day_end:
            log.info(
                "backfill_ticks_day_gapfill",
                broker=broker_id,
                symbol=symbol,
                date=date.isoformat(),
                checkpoint_ts=cp.last_timestamp_utc.isoformat(),
            )

        return self.backfill_ticks(symbol, day_start, day_end, respect_checkpoint=False)

    def backfill_bars(
        self,
        symbol: str,
        timeframe: str,
        start: dt.datetime,
        end: dt.datetime,
    ) -> int:
        """Backfill native bars in time chunks. Resumable."""
        broker_id = self._conn.broker_id
        start = dt_to_utc(start)
        end = dt_to_utc(end)

        cp = self._db.get_checkpoint(broker_id, symbol, "bars", timeframe)
        if cp and cp.last_timestamp_utc > start:
            log.info("backfill_bars_resume", broker=broker_id, symbol=symbol, tf=timeframe)
            start = cp.last_timestamp_utc

        chunk_delta = dt.timedelta(days=self._cfg.bar_chunk_days)
        total_rows = cp.rows_ingested if cp else 0
        job_id = self._db.start_job("backfill_bars", broker_id, symbol)

        chunk_from = start
        try:
            while chunk_from < end:
                chunk_to = min(chunk_from + chunk_delta, end)
                self._conn.ensure_connected()

                df = fetch_bars_chunk(self._conn, symbol, timeframe, chunk_from, chunk_to)
                if df.is_empty():
                    chunk_from = chunk_to
                    continue

                rows_written = store_bars_by_date(df, broker_id, symbol, timeframe, self._paths, self._store)
                total_rows += rows_written

                self._db.upsert_checkpoint(IngestionCheckpoint(
                    broker_id=broker_id,
                    symbol=symbol,
                    data_type="bars",
                    timeframe=timeframe,
                    last_timestamp_utc=chunk_to,
                    rows_ingested=total_rows,
                    updated_at=utc_now(),
                ))

                chunk_from = chunk_to

            self._db.finish_job(job_id, "completed", total_rows)
            log.info("backfill_bars_done", broker=broker_id, symbol=symbol, tf=timeframe, total=total_rows)

        except Exception as exc:
            self._db.finish_job(job_id, "failed", total_rows, str(exc))
            raise

        return total_rows

    def backfill_history_orders(
        self, start: dt.datetime, end: dt.datetime, group: str = ""
    ) -> int:
        """Backfill historical orders in chunks."""
        broker_id = self._conn.broker_id
        start, end = dt_to_utc(start), dt_to_utc(end)

        cp = self._db.get_checkpoint(broker_id, "*", "history_orders")
        if cp and cp.last_timestamp_utc > start:
            start = cp.last_timestamp_utc

        chunk_delta = dt.timedelta(days=self._cfg.history_chunk_days)
        total_rows = cp.rows_ingested if cp else 0
        chunk_from = start

        while chunk_from < end:
            chunk_to = min(chunk_from + chunk_delta, end)
            self._conn.ensure_connected()

            df = fetch_history_orders(self._conn, chunk_from, chunk_to, group)
            if not df.is_empty():
                written = store_history_orders(df, broker_id, self._paths, self._store)
                total_rows += written

            self._db.upsert_checkpoint(IngestionCheckpoint(
                broker_id=broker_id, symbol="*", data_type="history_orders",
                last_timestamp_utc=chunk_to, rows_ingested=total_rows, updated_at=utc_now(),
            ))
            chunk_from = chunk_to

        log.info("backfill_history_orders_done", broker=broker_id, total=total_rows)
        return total_rows

    def backfill_history_deals(
        self, start: dt.datetime, end: dt.datetime, group: str = ""
    ) -> int:
        """Backfill historical deals in chunks."""
        broker_id = self._conn.broker_id
        start, end = dt_to_utc(start), dt_to_utc(end)

        cp = self._db.get_checkpoint(broker_id, "*", "history_deals")
        if cp and cp.last_timestamp_utc > start:
            start = cp.last_timestamp_utc

        chunk_delta = dt.timedelta(days=self._cfg.history_chunk_days)
        total_rows = cp.rows_ingested if cp else 0
        chunk_from = start

        while chunk_from < end:
            chunk_to = min(chunk_from + chunk_delta, end)
            self._conn.ensure_connected()

            df = fetch_history_deals(self._conn, chunk_from, chunk_to, group)
            if not df.is_empty():
                written = store_history_deals(df, broker_id, self._paths, self._store)
                total_rows += written

            self._db.upsert_checkpoint(IngestionCheckpoint(
                broker_id=broker_id, symbol="*", data_type="history_deals",
                last_timestamp_utc=chunk_to, rows_ingested=total_rows, updated_at=utc_now(),
            ))
            chunk_from = chunk_to

        log.info("backfill_history_deals_done", broker=broker_id, total=total_rows)
        return total_rows

    def backfill_symbol_metadata(self, symbols: list[str]) -> None:
        """Capture symbol metadata for all configured symbols."""
        self._conn.ensure_connected()
        for symbol in symbols:
            capture_symbol_metadata(self._conn, symbol, self._paths, self._store)
        capture_symbol_universe(self._conn, self._paths, self._store)
