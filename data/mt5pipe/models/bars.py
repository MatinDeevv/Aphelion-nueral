"""Bar data models — native MT5 bars and locally built bars."""

from __future__ import annotations

import datetime as dt

import polars as pl
from pydantic import BaseModel, Field

from mt5pipe.utils.time import utc_now


class NativeBar(BaseModel):
    """Native OHLCV bar from MT5."""

    broker_id: str
    symbol: str
    timeframe: str
    time_utc: dt.datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: int = 0
    spread: int = 0
    real_volume: int = 0
    ingest_ts: dt.datetime = Field(default_factory=utc_now)


NATIVE_BAR_SCHEMA: dict[str, pl.DataType] = {
    "broker_id": pl.Utf8,
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "time_utc": pl.Datetime("ms", time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "tick_volume": pl.Int64,
    "spread": pl.Int32,
    "real_volume": pl.Int64,
    "ingest_ts": pl.Datetime("ms", time_zone="UTC"),
}


class BuiltBar(BaseModel):
    """Locally built bar from canonical ticks."""

    symbol: str
    timeframe: str
    time_utc: dt.datetime
    open: float
    high: float
    low: float
    close: float
    tick_count: int
    bid_open: float = 0.0
    ask_open: float = 0.0
    bid_close: float = 0.0
    ask_close: float = 0.0
    spread_mean: float = 0.0
    spread_max: float = 0.0
    spread_min: float = 0.0
    mid_return: float = 0.0
    realized_vol: float = 0.0
    volume_sum: float = 0.0
    source_count: int = 0
    conflict_count: int = 0


BUILT_BAR_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "time_utc": pl.Datetime("ms", time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "tick_count": pl.Int64,
    "bid_open": pl.Float64,
    "ask_open": pl.Float64,
    "bid_close": pl.Float64,
    "ask_close": pl.Float64,
    "spread_mean": pl.Float64,
    "spread_max": pl.Float64,
    "spread_min": pl.Float64,
    "mid_return": pl.Float64,
    "realized_vol": pl.Float64,
    "volume_sum": pl.Float64,
    "source_count": pl.Int64,
    "conflict_count": pl.Int64,
}
