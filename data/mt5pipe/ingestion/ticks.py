"""Tick ingestion — raw tick download and storage."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from mt5pipe.models.ticks import RAW_TICK_COLUMNS
from mt5pipe.mt5.connection import MT5Connection
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths
from mt5pipe.utils.logging import get_logger
from mt5pipe.utils.time import utc_now

log = get_logger(__name__)

if TYPE_CHECKING:
    from mt5pipe.storage.checkpoint_db import CheckpointDB


def fetch_ticks_chunk(
    conn: MT5Connection,
    symbol: str,
    date_from: dt.datetime,
    date_to: dt.datetime,
) -> pl.DataFrame:
    """Fetch a chunk of ticks from MT5 and return as polars DataFrame."""
    raw = conn.copy_ticks_range(symbol, date_from, date_to)

    if raw is None or len(raw) == 0:
        return pl.DataFrame(schema={c: pl.Float64 for c in RAW_TICK_COLUMNS})

    now = utc_now()

    df = pl.DataFrame({
        "broker_id": pl.Series([conn.broker_id] * len(raw), dtype=pl.Utf8),
        "symbol": pl.Series([symbol] * len(raw), dtype=pl.Utf8),
        "time_utc": pl.Series(
            [dt.datetime.fromtimestamp(int(t), tz=dt.timezone.utc) for t in raw["time"]],
            dtype=pl.Datetime("ms", time_zone="UTC"),
        ),
        "time_msc": pl.Series(raw["time_msc"].astype(np.int64), dtype=pl.Int64),
        "bid": pl.Series(raw["bid"].astype(np.float64), dtype=pl.Float64),
        "ask": pl.Series(raw["ask"].astype(np.float64), dtype=pl.Float64),
        "last": pl.Series(raw["last"].astype(np.float64), dtype=pl.Float64),
        "volume": pl.Series(raw["volume"].astype(np.float64), dtype=pl.Float64),
        "volume_real": pl.Series(
            raw["volume_real"].astype(np.float64) if "volume_real" in raw.dtype.names else np.zeros(len(raw)),
            dtype=pl.Float64,
        ),
        "flags": pl.Series(raw["flags"].astype(np.int32), dtype=pl.Int32),
        "ingest_ts": pl.Series([now] * len(raw), dtype=pl.Datetime("ms", time_zone="UTC")),
    })

    log.info(
        "ticks_fetched",
        broker=conn.broker_id,
        symbol=symbol,
        rows=len(df),
        from_ts=date_from.isoformat(),
        to_ts=date_to.isoformat(),
    )
    return df


def store_ticks_by_date(
    df: pl.DataFrame,
    broker_id: str,
    symbol: str,
    paths: StoragePaths,
    store: ParquetStore,
    checkpoint_db: "CheckpointDB | None" = None,
) -> int:
    """Partition ticks by date and write to Parquet. Returns net-new rows after dedup."""
    if df.is_empty():
        return 0

    # Extract date from time_utc for partitioning
    df = df.with_columns(
        pl.col("time_utc").dt.date().alias("_date")
    )

    total = 0
    for date_val in df["_date"].unique().sort().to_list():
        day_df = df.filter(pl.col("_date") == date_val).drop("_date")
        path = paths.raw_ticks_file(broker_id, symbol, date_val)
        before_rows = store.count_rows(path) if path.exists() else 0
        store.write(day_df, path)
        after_rows = store.count_rows(path)
        net_new_rows = max(after_rows - before_rows, 0)
        total += net_new_rows

        if checkpoint_db is not None:
            checkpoint_db.register_file(
                str(path),
                broker_id,
                symbol,
                "ticks",
                date_val.isoformat(),
                after_rows,
                path.stat().st_size if path.exists() else 0,
            )

    return total


def deduplicate_ticks(df: pl.DataFrame) -> pl.DataFrame:
    """Remove duplicate ticks based on time_msc + bid + ask + last."""
    if df.is_empty():
        return df
    return df.unique(subset=["time_msc", "bid", "ask", "last", "volume"], keep="first")


def clean_and_deduplicate_ticks(df: pl.DataFrame) -> pl.DataFrame:
    """Full tick cleaning: dedup + validation + outlier removal.

    Wraps the quality.cleaning module for a one-call clean.
    """
    if df.is_empty():
        return df
    from mt5pipe.quality.cleaning import clean_ticks
    cleaned, _report = clean_ticks(df)
    return cleaned
