"""Market book / DOM ingestion."""

from __future__ import annotations

import datetime as dt

import polars as pl

from mt5pipe.mt5.connection import MT5Connection
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths
from mt5pipe.utils.logging import get_logger
from mt5pipe.utils.time import utc_now

log = get_logger(__name__)


def capture_market_book(
    conn: MT5Connection,
    symbol: str,
    paths: StoragePaths,
    store: ParquetStore,
) -> int:
    """Capture a single DOM snapshot. Returns number of levels captured."""
    book = conn.market_book_get(symbol)
    if book is None or len(book) == 0:
        return 0

    now = utc_now()
    rows = []
    for idx, level in enumerate(book):
        rows.append({
            "broker_id": conn.broker_id,
            "symbol": symbol,
            "snapshot_ts": now,
            "level_type": int(level.type),
            "price": float(level.price),
            "volume": float(level.volume),
            "volume_real": float(getattr(level, "volume_real", 0.0)),
            "level_index": idx,
        })

    df = pl.DataFrame(rows)
    path = paths.market_book_file(conn.broker_id, symbol, now.date())
    store.write(df, path)
    log.debug("market_book_captured", broker=conn.broker_id, symbol=symbol, levels=len(rows))
    return len(rows)


def subscribe_market_book(conn: MT5Connection, symbol: str) -> bool:
    """Subscribe to market book updates for a symbol."""
    ok = conn.market_book_add(symbol)
    if ok:
        log.info("market_book_subscribed", broker=conn.broker_id, symbol=symbol)
    else:
        log.warning("market_book_subscribe_failed", broker=conn.broker_id, symbol=symbol)
    return ok


def unsubscribe_market_book(conn: MT5Connection, symbol: str) -> bool:
    """Unsubscribe from market book updates."""
    return conn.market_book_release(symbol)
