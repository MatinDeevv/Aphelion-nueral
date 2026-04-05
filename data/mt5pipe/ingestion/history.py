"""Historical orders and deals ingestion."""

from __future__ import annotations

import datetime as dt

import polars as pl

from mt5pipe.mt5.connection import MT5Connection
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths
from mt5pipe.utils.logging import get_logger
from mt5pipe.utils.time import utc_now

log = get_logger(__name__)


def _ts_field(val: int | float | None) -> dt.datetime | None:
    if val is None or val == 0:
        return None
    return dt.datetime.fromtimestamp(int(val), tz=dt.timezone.utc)


def fetch_history_orders(
    conn: MT5Connection,
    date_from: dt.datetime,
    date_to: dt.datetime,
    group: str = "",
) -> pl.DataFrame:
    """Fetch historical orders from MT5."""
    raw = conn.history_orders_get(date_from, date_to, group)
    if raw is None or len(raw) == 0:
        return pl.DataFrame()

    now = utc_now()
    rows = []
    for o in raw:
        d = o._asdict()
        rows.append({
            "broker_id": conn.broker_id,
            "ticket": int(d.get("ticket", 0)),
            "time_setup": _ts_field(d.get("time_setup")),
            "time_done": _ts_field(d.get("time_done")),
            "type": int(d.get("type", 0)),
            "state": int(d.get("state", 0)),
            "magic": int(d.get("magic", 0)),
            "position_id": int(d.get("position_id", 0)),
            "volume_initial": float(d.get("volume_initial", 0)),
            "volume_current": float(d.get("volume_current", 0)),
            "price_open": float(d.get("price_open", 0)),
            "price_current": float(d.get("price_current", 0)),
            "sl": float(d.get("sl", 0)),
            "tp": float(d.get("tp", 0)),
            "symbol": str(d.get("symbol", "")),
            "comment": str(d.get("comment", "")),
            "external_id": str(d.get("external_id", "")),
            "ingest_ts": now,
        })

    df = pl.DataFrame(rows)
    log.info("history_orders_fetched", broker=conn.broker_id, count=len(rows))
    return df


def store_history_orders(
    df: pl.DataFrame, broker_id: str, paths: StoragePaths, store: ParquetStore
) -> int:
    """Store historical orders partitioned by date."""
    if df.is_empty():
        return 0

    # Partition by time_done date
    df = df.with_columns(
        pl.col("time_done").dt.date().alias("_date")
    )

    total = 0
    for date_val in df["_date"].unique().drop_nulls().sort().to_list():
        day_df = df.filter(pl.col("_date") == date_val).drop("_date")
        path = paths.history_orders_file(broker_id, date_val)
        total += store.write(day_df, path)

    # Handle rows with null time_done
    null_df = df.filter(pl.col("_date").is_null()).drop("_date")
    if not null_df.is_empty():
        path = paths.history_orders_file(broker_id, dt.date(1970, 1, 1))
        total += store.write(null_df, path)

    return total


def fetch_history_deals(
    conn: MT5Connection,
    date_from: dt.datetime,
    date_to: dt.datetime,
    group: str = "",
) -> pl.DataFrame:
    """Fetch historical deals from MT5."""
    raw = conn.history_deals_get(date_from, date_to, group)
    if raw is None or len(raw) == 0:
        return pl.DataFrame()

    now = utc_now()
    rows = []
    for deal in raw:
        d = deal._asdict()
        rows.append({
            "broker_id": conn.broker_id,
            "ticket": int(d.get("ticket", 0)),
            "order": int(d.get("order", 0)),
            "time": _ts_field(d.get("time")),
            "type": int(d.get("type", 0)),
            "entry": int(d.get("entry", 0)),
            "magic": int(d.get("magic", 0)),
            "position_id": int(d.get("position_id", 0)),
            "volume": float(d.get("volume", 0)),
            "price": float(d.get("price", 0)),
            "commission": float(d.get("commission", 0)),
            "swap": float(d.get("swap", 0)),
            "profit": float(d.get("profit", 0)),
            "fee": float(d.get("fee", 0)),
            "symbol": str(d.get("symbol", "")),
            "comment": str(d.get("comment", "")),
            "external_id": str(d.get("external_id", "")),
            "ingest_ts": now,
        })

    df = pl.DataFrame(rows)
    log.info("history_deals_fetched", broker=conn.broker_id, count=len(rows))
    return df


def store_history_deals(
    df: pl.DataFrame, broker_id: str, paths: StoragePaths, store: ParquetStore
) -> int:
    """Store historical deals partitioned by date."""
    if df.is_empty():
        return 0

    df = df.with_columns(
        pl.col("time").dt.date().alias("_date")
    )

    total = 0
    for date_val in df["_date"].unique().drop_nulls().sort().to_list():
        day_df = df.filter(pl.col("_date") == date_val).drop("_date")
        path = paths.history_deals_file(broker_id, date_val)
        total += store.write(day_df, path)

    null_df = df.filter(pl.col("_date").is_null()).drop("_date")
    if not null_df.is_empty():
        path = paths.history_deals_file(broker_id, dt.date(1970, 1, 1))
        total += store.write(null_df, path)

    return total
