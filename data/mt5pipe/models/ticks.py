"""Tick data models — raw and canonical."""

from __future__ import annotations

import datetime as dt

import polars as pl
from pydantic import BaseModel, Field

from mt5pipe.utils.time import utc_now


class RawTick(BaseModel):
    """Single raw tick from a broker."""

    broker_id: str
    symbol: str
    time_utc: dt.datetime
    time_msc: int = Field(description="Millisecond-precision epoch timestamp from MT5")
    bid: float
    ask: float
    last: float = 0.0
    volume: float = 0.0
    volume_real: float = 0.0
    flags: int = 0
    ingest_ts: dt.datetime = Field(default_factory=utc_now)


RAW_TICK_SCHEMA: dict[str, pl.DataType] = {
    "broker_id": pl.Utf8,
    "symbol": pl.Utf8,
    "time_utc": pl.Datetime("ms", time_zone="UTC"),
    "time_msc": pl.Int64,
    "bid": pl.Float64,
    "ask": pl.Float64,
    "last": pl.Float64,
    "volume": pl.Float64,
    "volume_real": pl.Float64,
    "flags": pl.Int32,
    "ingest_ts": pl.Datetime("ms", time_zone="UTC"),
}

RAW_TICK_COLUMNS = list(RAW_TICK_SCHEMA.keys())


class CanonicalTick(BaseModel):
    """Merged canonical tick from multiple brokers."""

    ts_utc: dt.datetime
    ts_msc: int
    symbol: str
    bid: float
    ask: float
    last: float = 0.0
    volume: float = 0.0
    source_primary: str
    source_secondary: str = ""
    merge_mode: str = Field(description="'single', 'best', 'fallback', 'conflict'")
    quality_score: float = 0.0
    conflict_flag: bool = False
    broker_a_bid: float = 0.0
    broker_a_ask: float = 0.0
    broker_b_bid: float = 0.0
    broker_b_ask: float = 0.0
    mid_diff: float = 0.0
    spread_diff: float = 0.0


CANONICAL_TICK_SCHEMA: dict[str, pl.DataType] = {
    "ts_utc": pl.Datetime("ms", time_zone="UTC"),
    "ts_msc": pl.Int64,
    "symbol": pl.Utf8,
    "bid": pl.Float64,
    "ask": pl.Float64,
    "last": pl.Float64,
    "volume": pl.Float64,
    "source_primary": pl.Utf8,
    "source_secondary": pl.Utf8,
    "merge_mode": pl.Utf8,
    "quality_score": pl.Float64,
    "conflict_flag": pl.Boolean,
    "broker_a_bid": pl.Float64,
    "broker_a_ask": pl.Float64,
    "broker_b_bid": pl.Float64,
    "broker_b_ask": pl.Float64,
    "mid_diff": pl.Float64,
    "spread_diff": pl.Float64,
}

CANONICAL_TICK_COLUMNS = list(CANONICAL_TICK_SCHEMA.keys())
