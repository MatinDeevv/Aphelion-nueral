"""Live continuous data collection engine."""

from __future__ import annotations

import datetime as dt
import signal
import time
from typing import Any

import polars as pl

from mt5pipe.config.models import LiveConfig
from mt5pipe.ingestion.market_book import capture_market_book, subscribe_market_book, unsubscribe_market_book
from mt5pipe.ingestion.snapshots import (
    capture_account_state,
    capture_active_orders,
    capture_active_positions,
    capture_symbol_metadata,
    capture_terminal_state,
)
from mt5pipe.ingestion.ticks import deduplicate_ticks, store_ticks_by_date
from mt5pipe.models.checkpoint import IngestionCheckpoint
from mt5pipe.mt5.connection import MT5Connection
from mt5pipe.storage.checkpoint_db import CheckpointDB
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths
from mt5pipe.utils.logging import get_logger
from mt5pipe.utils.time import utc_now

log = get_logger(__name__)


class LiveCollector:
    """Continuous live data collection for a single broker+symbol."""

    def __init__(
        self,
        conn: MT5Connection,
        symbol: str,
        paths: StoragePaths,
        store: ParquetStore,
        checkpoint_db: CheckpointDB,
        live_cfg: LiveConfig,
        enable_market_book: bool = True,
    ) -> None:
        self._conn = conn
        self._symbol = symbol
        self._paths = paths
        self._store = store
        self._db = checkpoint_db
        self._cfg = live_cfg
        self._enable_market_book = enable_market_book

        self._running = False
        self._tick_buffer: list[dict[str, Any]] = []
        self._last_time_msc: int = 0
        self._last_flush_ts = utc_now()
        self._last_snapshot_ts = utc_now()
        self._last_book_ts = utc_now()

        # Restore checkpoint
        cp = self._db.get_checkpoint(conn.broker_id, symbol, "live_ticks")
        if cp:
            self._last_time_msc = cp.last_time_msc
            log.info(
                "live_collector_resume",
                broker=conn.broker_id,
                symbol=symbol,
                last_msc=self._last_time_msc,
            )

    def start(self, duration_seconds: int = 0) -> None:
        """Start the live collection loop.

        Args:
            duration_seconds: If >0, collect for this many seconds then exit
                gracefully. If 0, run indefinitely until signal/stop().
        """
        broker_id = self._conn.broker_id
        self._running = True
        deadline = (time.monotonic() + duration_seconds) if duration_seconds > 0 else 0.0
        log.info(
            "live_collector_start",
            broker=broker_id,
            symbol=self._symbol,
            duration=duration_seconds or "infinite",
        )

        # Register signal handlers for graceful shutdown
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _stop_handler(signum: int, frame: Any) -> None:
            log.info("live_collector_stop_signal", signal=signum)
            self._running = False

        signal.signal(signal.SIGINT, _stop_handler)
        signal.signal(signal.SIGTERM, _stop_handler)

        if self._enable_market_book:
            subscribe_market_book(self._conn, self._symbol)

        cycles = 0
        try:
            while self._running:
                # Check deadline
                if deadline and time.monotonic() >= deadline:
                    log.info(
                        "live_collector_duration_reached",
                        broker=broker_id,
                        symbol=self._symbol,
                        seconds=duration_seconds,
                    )
                    break

                self._conn.ensure_connected()
                self._poll_ticks()
                self._maybe_flush()
                self._maybe_snapshot()
                self._maybe_market_book()

                cycles += 1
                if cycles % 200 == 0:
                    elapsed = duration_seconds - int(deadline - time.monotonic()) if deadline else 0
                    log.info(
                        "live_collector_heartbeat",
                        broker=broker_id,
                        symbol=self._symbol,
                        buffered=len(self._tick_buffer),
                        last_msc=self._last_time_msc,
                        elapsed_s=elapsed,
                    )

                time.sleep(self._cfg.tick_poll_seconds)
        except Exception as exc:
            log.error("live_collector_error", broker=broker_id, error=str(exc))
            raise
        finally:
            self._flush_buffer()
            if self._enable_market_book:
                unsubscribe_market_book(self._conn, self._symbol)
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)
            log.info("live_collector_stopped", broker=broker_id, symbol=self._symbol)

    def stop(self) -> None:
        """Request graceful stop."""
        self._running = False

    def _poll_ticks(self) -> None:
        """Poll for new ticks since last known timestamp."""
        # Use copy_ticks_range from 1 second before last known to now
        now = utc_now()
        if self._last_time_msc > 0:
            from_ts = dt.datetime.fromtimestamp(
                self._last_time_msc / 1000.0 - 1.0, tz=dt.timezone.utc
            )
        else:
            from_ts = now - dt.timedelta(seconds=5)

        raw = self._conn.copy_ticks_range(self._symbol, from_ts, now)
        if raw is None or len(raw) == 0:
            return

        new_count = 0
        ingest_ts = utc_now()
        for i in range(len(raw)):
            msc = int(raw["time_msc"][i])
            if msc <= self._last_time_msc:
                continue

            self._tick_buffer.append({
                "broker_id": self._conn.broker_id,
                "symbol": self._symbol,
                "time_utc": dt.datetime.fromtimestamp(int(raw["time"][i]), tz=dt.timezone.utc),
                "time_msc": msc,
                "bid": float(raw["bid"][i]),
                "ask": float(raw["ask"][i]),
                "last": float(raw["last"][i]),
                "volume": float(raw["volume"][i]),
                "volume_real": float(raw["volume_real"][i]) if "volume_real" in raw.dtype.names else 0.0,
                "flags": int(raw["flags"][i]),
                "ingest_ts": ingest_ts,
            })
            new_count += 1

        if new_count > 0:
            self._last_time_msc = int(raw["time_msc"][-1])
            log.debug("live_ticks_polled", broker=self._conn.broker_id, new=new_count, buffered=len(self._tick_buffer))

    def _maybe_flush(self) -> None:
        """Flush tick buffer to disk if interval/size threshold reached."""
        now = utc_now()
        elapsed = (now - self._last_flush_ts).total_seconds()

        if (
            elapsed >= self._cfg.flush_interval_seconds
            or len(self._tick_buffer) >= self._cfg.buffer_max_rows
        ):
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Write buffered ticks to Parquet and update checkpoint."""
        if not self._tick_buffer:
            return

        df = pl.DataFrame(self._tick_buffer)
        df = deduplicate_ticks(df)
        rows = store_ticks_by_date(df, self._conn.broker_id, self._symbol, self._paths, self._store)

        self._db.upsert_checkpoint(IngestionCheckpoint(
            broker_id=self._conn.broker_id,
            symbol=self._symbol,
            data_type="live_ticks",
            last_timestamp_utc=utc_now(),
            last_time_msc=self._last_time_msc,
            rows_ingested=rows,
            updated_at=utc_now(),
        ))

        log.info("live_ticks_flushed", broker=self._conn.broker_id, symbol=self._symbol, rows=rows)
        self._tick_buffer.clear()
        self._last_flush_ts = utc_now()

    def _maybe_snapshot(self) -> None:
        """Periodically capture snapshots."""
        now = utc_now()
        elapsed = (now - self._last_snapshot_ts).total_seconds()
        if elapsed < self._cfg.snapshot_interval_seconds:
            return

        try:
            capture_symbol_metadata(self._conn, self._symbol, self._paths, self._store)
            capture_account_state(self._conn, self._paths, self._store)
            capture_terminal_state(self._conn, self._paths, self._store)
            capture_active_orders(self._conn, self._symbol, self._paths, self._store)
            capture_active_positions(self._conn, self._symbol, self._paths, self._store)
        except Exception as exc:
            log.warning("snapshot_error", broker=self._conn.broker_id, error=str(exc))

        self._last_snapshot_ts = utc_now()

    def _maybe_market_book(self) -> None:
        """Periodically capture DOM."""
        if not self._enable_market_book:
            return

        now = utc_now()
        elapsed = (now - self._last_book_ts).total_seconds()
        if elapsed < self._cfg.market_book_interval_seconds:
            return

        try:
            capture_market_book(self._conn, self._symbol, self._paths, self._store)
        except Exception as exc:
            log.warning("market_book_error", broker=self._conn.broker_id, error=str(exc))

        self._last_book_ts = utc_now()
