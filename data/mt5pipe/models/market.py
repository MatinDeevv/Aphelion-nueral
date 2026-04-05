"""Market data models — symbol metadata, universe, DOM."""

from __future__ import annotations

import datetime as dt
from typing import Any

import polars as pl
from pydantic import BaseModel, Field

from mt5pipe.utils.time import utc_now


class SymbolMetadataSnapshot(BaseModel):
    """Full symbol_info snapshot from MT5."""

    broker_id: str
    symbol: str
    snapshot_ts: dt.datetime = Field(default_factory=utc_now)
    # Store the full MT5 SymbolInfo as a flat dict
    name: str = ""
    description: str = ""
    path: str = ""
    point: float = 0.0
    digits: int = 0
    spread: int = 0
    spread_float: bool = False
    trade_mode: int = 0
    trade_calc_mode: int = 0
    trade_contract_size: float = 0.0
    trade_tick_value: float = 0.0
    trade_tick_size: float = 0.0
    volume_min: float = 0.0
    volume_max: float = 0.0
    volume_step: float = 0.0
    swap_long: float = 0.0
    swap_short: float = 0.0
    swap_mode: int = 0
    session_open: float = 0.0
    session_close: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    currency_base: str = ""
    currency_profit: str = ""
    currency_margin: str = ""
    # Additional fields stored as JSON string for flexibility
    extra_json: str = "{}"


SYMBOL_METADATA_SCHEMA: dict[str, pl.DataType] = {
    "broker_id": pl.Utf8,
    "symbol": pl.Utf8,
    "snapshot_ts": pl.Datetime("ms", time_zone="UTC"),
    "name": pl.Utf8,
    "description": pl.Utf8,
    "path": pl.Utf8,
    "point": pl.Float64,
    "digits": pl.Int32,
    "spread": pl.Int32,
    "spread_float": pl.Boolean,
    "trade_mode": pl.Int32,
    "trade_calc_mode": pl.Int32,
    "trade_contract_size": pl.Float64,
    "trade_tick_value": pl.Float64,
    "trade_tick_size": pl.Float64,
    "volume_min": pl.Float64,
    "volume_max": pl.Float64,
    "volume_step": pl.Float64,
    "swap_long": pl.Float64,
    "swap_short": pl.Float64,
    "swap_mode": pl.Int32,
    "session_open": pl.Float64,
    "session_close": pl.Float64,
    "bid": pl.Float64,
    "ask": pl.Float64,
    "last": pl.Float64,
    "currency_base": pl.Utf8,
    "currency_profit": pl.Utf8,
    "currency_margin": pl.Utf8,
    "extra_json": pl.Utf8,
}


class SymbolUniverseSnapshot(BaseModel):
    """Snapshot of all symbols available on a broker."""

    broker_id: str
    snapshot_ts: dt.datetime = Field(default_factory=utc_now)
    symbols_total: int = 0
    symbols: list[str] = Field(default_factory=list)


class MarketBookLevel(BaseModel):
    """Single level in the market depth / DOM."""

    broker_id: str
    symbol: str
    snapshot_ts: dt.datetime
    level_type: int = Field(description="MT5 BOOK_TYPE: 1=sell, 2=buy, etc.")
    price: float
    volume: float
    volume_real: float = 0.0
    level_index: int


MARKET_BOOK_SCHEMA: dict[str, pl.DataType] = {
    "broker_id": pl.Utf8,
    "symbol": pl.Utf8,
    "snapshot_ts": pl.Datetime("ms", time_zone="UTC"),
    "level_type": pl.Int32,
    "price": pl.Float64,
    "volume": pl.Float64,
    "volume_real": pl.Float64,
    "level_index": pl.Int32,
}
