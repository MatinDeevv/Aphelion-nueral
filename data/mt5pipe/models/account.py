"""Account, terminal, order, position, and deal models."""

from __future__ import annotations

import datetime as dt
from typing import Any

import polars as pl
from pydantic import BaseModel, Field

from mt5pipe.utils.time import utc_now


# --- Account State ---

class AccountStateSnapshot(BaseModel):
    broker_id: str
    snapshot_ts: dt.datetime = Field(default_factory=utc_now)
    login: int = 0
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    margin_free: float = 0.0
    margin_level: float = 0.0
    profit: float = 0.0
    credit: float = 0.0
    currency: str = ""
    leverage: int = 0
    trade_mode: int = 0
    limit_orders: int = 0
    server: str = ""
    company: str = ""
    name: str = ""
    extra_json: str = "{}"


ACCOUNT_STATE_SCHEMA: dict[str, pl.DataType] = {
    "broker_id": pl.Utf8,
    "snapshot_ts": pl.Datetime("ms", time_zone="UTC"),
    "login": pl.Int64,
    "balance": pl.Float64,
    "equity": pl.Float64,
    "margin": pl.Float64,
    "margin_free": pl.Float64,
    "margin_level": pl.Float64,
    "profit": pl.Float64,
    "credit": pl.Float64,
    "currency": pl.Utf8,
    "leverage": pl.Int32,
    "trade_mode": pl.Int32,
    "limit_orders": pl.Int32,
    "server": pl.Utf8,
    "company": pl.Utf8,
    "name": pl.Utf8,
    "extra_json": pl.Utf8,
}


# --- Terminal State ---

class TerminalStateSnapshot(BaseModel):
    broker_id: str
    snapshot_ts: dt.datetime = Field(default_factory=utc_now)
    connected: bool = False
    community_account: bool = False
    community_connection: bool = False
    dlls_allowed: bool = False
    trade_allowed: bool = False
    tradeapi_disabled: bool = False
    email_enabled: bool = False
    ftp_enabled: bool = False
    notifications_enabled: bool = False
    mqid: bool = False
    build: int = 0
    maxbars: int = 0
    codepage: int = 0
    ping_last: int = 0
    community_balance: float = 0.0
    retransmission: float = 0.0
    company: str = ""
    name: str = ""
    language: str = ""
    path: str = ""
    data_path: str = ""
    commondata_path: str = ""


TERMINAL_STATE_SCHEMA: dict[str, pl.DataType] = {
    "broker_id": pl.Utf8,
    "snapshot_ts": pl.Datetime("ms", time_zone="UTC"),
    "connected": pl.Boolean,
    "community_account": pl.Boolean,
    "community_connection": pl.Boolean,
    "dlls_allowed": pl.Boolean,
    "trade_allowed": pl.Boolean,
    "tradeapi_disabled": pl.Boolean,
    "email_enabled": pl.Boolean,
    "ftp_enabled": pl.Boolean,
    "notifications_enabled": pl.Boolean,
    "mqid": pl.Boolean,
    "build": pl.Int32,
    "maxbars": pl.Int32,
    "codepage": pl.Int32,
    "ping_last": pl.Int64,
    "community_balance": pl.Float64,
    "retransmission": pl.Float64,
    "company": pl.Utf8,
    "name": pl.Utf8,
    "language": pl.Utf8,
    "path": pl.Utf8,
    "data_path": pl.Utf8,
    "commondata_path": pl.Utf8,
}


# --- Active Orders ---

class ActiveOrderSnapshot(BaseModel):
    broker_id: str
    snapshot_ts: dt.datetime = Field(default_factory=utc_now)
    ticket: int = 0
    time_setup: dt.datetime | None = None
    type: int = 0
    state: int = 0
    magic: int = 0
    volume_current: float = 0.0
    price_open: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    price_current: float = 0.0
    symbol: str = ""
    comment: str = ""
    external_id: str = ""


ACTIVE_ORDER_SCHEMA: dict[str, pl.DataType] = {
    "broker_id": pl.Utf8,
    "snapshot_ts": pl.Datetime("ms", time_zone="UTC"),
    "ticket": pl.Int64,
    "time_setup": pl.Datetime("ms", time_zone="UTC"),
    "type": pl.Int32,
    "state": pl.Int32,
    "magic": pl.Int64,
    "volume_current": pl.Float64,
    "price_open": pl.Float64,
    "sl": pl.Float64,
    "tp": pl.Float64,
    "price_current": pl.Float64,
    "symbol": pl.Utf8,
    "comment": pl.Utf8,
    "external_id": pl.Utf8,
}


# --- Active Positions ---

class ActivePositionSnapshot(BaseModel):
    broker_id: str
    snapshot_ts: dt.datetime = Field(default_factory=utc_now)
    ticket: int = 0
    time: dt.datetime | None = None
    type: int = 0
    magic: int = 0
    identifier: int = 0
    volume: float = 0.0
    price_open: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    price_current: float = 0.0
    profit: float = 0.0
    swap: float = 0.0
    symbol: str = ""
    comment: str = ""
    external_id: str = ""


ACTIVE_POSITION_SCHEMA: dict[str, pl.DataType] = {
    "broker_id": pl.Utf8,
    "snapshot_ts": pl.Datetime("ms", time_zone="UTC"),
    "ticket": pl.Int64,
    "time": pl.Datetime("ms", time_zone="UTC"),
    "type": pl.Int32,
    "magic": pl.Int64,
    "identifier": pl.Int64,
    "volume": pl.Float64,
    "price_open": pl.Float64,
    "sl": pl.Float64,
    "tp": pl.Float64,
    "price_current": pl.Float64,
    "profit": pl.Float64,
    "swap": pl.Float64,
    "symbol": pl.Utf8,
    "comment": pl.Utf8,
    "external_id": pl.Utf8,
}


# --- Historical Orders ---

class HistoricalOrder(BaseModel):
    broker_id: str
    ticket: int = 0
    time_setup: dt.datetime | None = None
    time_done: dt.datetime | None = None
    type: int = 0
    state: int = 0
    magic: int = 0
    position_id: int = 0
    volume_initial: float = 0.0
    volume_current: float = 0.0
    price_open: float = 0.0
    price_current: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    symbol: str = ""
    comment: str = ""
    external_id: str = ""
    ingest_ts: dt.datetime = Field(default_factory=utc_now)


HISTORICAL_ORDER_SCHEMA: dict[str, pl.DataType] = {
    "broker_id": pl.Utf8,
    "ticket": pl.Int64,
    "time_setup": pl.Datetime("ms", time_zone="UTC"),
    "time_done": pl.Datetime("ms", time_zone="UTC"),
    "type": pl.Int32,
    "state": pl.Int32,
    "magic": pl.Int64,
    "position_id": pl.Int64,
    "volume_initial": pl.Float64,
    "volume_current": pl.Float64,
    "price_open": pl.Float64,
    "price_current": pl.Float64,
    "sl": pl.Float64,
    "tp": pl.Float64,
    "symbol": pl.Utf8,
    "comment": pl.Utf8,
    "external_id": pl.Utf8,
    "ingest_ts": pl.Datetime("ms", time_zone="UTC"),
}


# --- Historical Deals ---

class HistoricalDeal(BaseModel):
    broker_id: str
    ticket: int = 0
    order: int = 0
    time: dt.datetime | None = None
    type: int = 0
    entry: int = 0
    magic: int = 0
    position_id: int = 0
    volume: float = 0.0
    price: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    profit: float = 0.0
    fee: float = 0.0
    symbol: str = ""
    comment: str = ""
    external_id: str = ""
    ingest_ts: dt.datetime = Field(default_factory=utc_now)


HISTORICAL_DEAL_SCHEMA: dict[str, pl.DataType] = {
    "broker_id": pl.Utf8,
    "ticket": pl.Int64,
    "order": pl.Int64,
    "time": pl.Datetime("ms", time_zone="UTC"),
    "type": pl.Int32,
    "entry": pl.Int32,
    "magic": pl.Int64,
    "position_id": pl.Int64,
    "volume": pl.Float64,
    "price": pl.Float64,
    "commission": pl.Float64,
    "swap": pl.Float64,
    "profit": pl.Float64,
    "fee": pl.Float64,
    "symbol": pl.Utf8,
    "comment": pl.Utf8,
    "external_id": pl.Utf8,
    "ingest_ts": pl.Datetime("ms", time_zone="UTC"),
}
