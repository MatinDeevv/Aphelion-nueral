"""Native bar ingestion from MT5."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from mt5pipe.mt5.connection import MT5Connection
from mt5pipe.mt5.constants import MT5_TIMEFRAMES
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths
from mt5pipe.utils.logging import get_logger
from mt5pipe.utils.time import utc_now

log = get_logger(__name__)


def fetch_bars_chunk(
    conn: MT5Connection,
    symbol: str,
    timeframe: str,
    date_from: dt.datetime,
    date_to: dt.datetime,
) -> pl.DataFrame:
    """Fetch native bars from MT5."""
    tf_value = MT5_TIMEFRAMES.get(timeframe)
    if tf_value is None:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    raw = conn.copy_rates_range(symbol, tf_value, date_from, date_to)

    if raw is None or len(raw) == 0:
        return pl.DataFrame()

    now = utc_now()

    df = pl.DataFrame({
        "broker_id": pl.Series([conn.broker_id] * len(raw), dtype=pl.Utf8),
        "symbol": pl.Series([symbol] * len(raw), dtype=pl.Utf8),
        "timeframe": pl.Series([timeframe] * len(raw), dtype=pl.Utf8),
        "time_utc": pl.Series(
            [dt.datetime.fromtimestamp(int(t), tz=dt.timezone.utc) for t in raw["time"]],
            dtype=pl.Datetime("ms", time_zone="UTC"),
        ),
        "open": pl.Series(raw["open"].astype(np.float64), dtype=pl.Float64),
        "high": pl.Series(raw["high"].astype(np.float64), dtype=pl.Float64),
        "low": pl.Series(raw["low"].astype(np.float64), dtype=pl.Float64),
        "close": pl.Series(raw["close"].astype(np.float64), dtype=pl.Float64),
        "tick_volume": pl.Series(raw["tick_volume"].astype(np.int64), dtype=pl.Int64),
        "spread": pl.Series(raw["spread"].astype(np.int32), dtype=pl.Int32),
        "real_volume": pl.Series(raw["real_volume"].astype(np.int64), dtype=pl.Int64),
        "ingest_ts": pl.Series([now] * len(raw), dtype=pl.Datetime("ms", time_zone="UTC")),
    })

    log.info(
        "bars_fetched",
        broker=conn.broker_id,
        symbol=symbol,
        timeframe=timeframe,
        rows=len(df),
    )
    return df


def store_bars_by_date(
    df: pl.DataFrame,
    broker_id: str,
    symbol: str,
    timeframe: str,
    paths: StoragePaths,
    store: ParquetStore,
) -> int:
    """Partition bars by date and write to Parquet."""
    if df.is_empty():
        return 0

    df = df.with_columns(pl.col("time_utc").dt.date().alias("_date"))

    total = 0
    for date_val in df["_date"].unique().sort().to_list():
        day_df = df.filter(pl.col("_date") == date_val).drop("_date")
        path = paths.native_bars_file(broker_id, symbol, timeframe, date_val)
        written = store.write(day_df, path)
        total += written

    return total
