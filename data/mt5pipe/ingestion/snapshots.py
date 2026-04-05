"""Snapshot ingestion — symbol metadata, account, terminal, orders, positions."""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import polars as pl

from mt5pipe.mt5.connection import MT5Connection
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths
from mt5pipe.utils.logging import get_logger
from mt5pipe.utils.time import utc_now

log = get_logger(__name__)


def _named_tuple_to_dict(nt: Any) -> dict[str, Any]:
    """Convert MT5 named tuple to dict, handling nested structures."""
    if nt is None:
        return {}
    return nt._asdict()


# --- Symbol Metadata ---

def capture_symbol_metadata(
    conn: MT5Connection, symbol: str, paths: StoragePaths, store: ParquetStore
) -> None:
    """Capture full symbol_info and store as Parquet."""
    info = conn.symbol_info(symbol)
    if info is None:
        log.warning("symbol_info_empty", broker=conn.broker_id, symbol=symbol)
        return

    now = utc_now()
    d = _named_tuple_to_dict(info)

    # Extract known fields, put rest in extra_json
    known_fields = {
        "name", "description", "path", "point", "digits", "spread", "spread_float",
        "trade_mode", "trade_calc_mode", "trade_contract_size", "trade_tick_value",
        "trade_tick_size", "volume_min", "volume_max", "volume_step",
        "swap_long", "swap_short", "swap_mode", "session_open", "session_close",
        "bid", "ask", "last", "currency_base", "currency_profit", "currency_margin",
    }
    extra = {k: v for k, v in d.items() if k not in known_fields}

    row: dict[str, Any] = {
        "broker_id": conn.broker_id,
        "symbol": symbol,
        "snapshot_ts": now,
    }
    for field in known_fields:
        if field in d:
            row[field] = d[field]

    row["extra_json"] = json.dumps(extra, default=str)

    df = pl.DataFrame([row])
    path = paths.symbol_metadata_file(conn.broker_id, now)
    store.write(df, path)
    log.info("symbol_metadata_captured", broker=conn.broker_id, symbol=symbol)


# --- Symbol Universe ---

def capture_symbol_universe(
    conn: MT5Connection, paths: StoragePaths, store: ParquetStore
) -> None:
    """Capture all available symbols."""
    now = utc_now()
    total = conn.symbols_total()
    symbols_raw = conn.symbols_get()

    symbol_names: list[str] = []
    if symbols_raw is not None:
        symbol_names = [s.name for s in symbols_raw]

    df = pl.DataFrame({
        "broker_id": [conn.broker_id],
        "snapshot_ts": [now],
        "symbols_total": [total],
        "symbols_json": [json.dumps(symbol_names)],
    })

    path = paths.symbol_universe_file(conn.broker_id, now)
    store.write(df, path)
    log.info("symbol_universe_captured", broker=conn.broker_id, total=total)


# --- Account State ---

def capture_account_state(
    conn: MT5Connection, paths: StoragePaths, store: ParquetStore
) -> None:
    """Capture full account_info snapshot."""
    info = conn.account_info()
    if info is None:
        log.warning("account_info_empty", broker=conn.broker_id)
        return

    now = utc_now()
    d = _named_tuple_to_dict(info)

    known_fields = {
        "login", "balance", "equity", "margin", "margin_free", "margin_level",
        "profit", "credit", "currency", "leverage", "trade_mode", "limit_orders",
        "server", "company", "name",
    }
    extra = {k: v for k, v in d.items() if k not in known_fields}

    row: dict[str, Any] = {"broker_id": conn.broker_id, "snapshot_ts": now}
    for field in known_fields:
        if field in d:
            row[field] = d[field]
    row["extra_json"] = json.dumps(extra, default=str)

    df = pl.DataFrame([row])
    path = paths.account_state_file(conn.broker_id, now.date())
    store.write(df, path)
    log.info("account_state_captured", broker=conn.broker_id)


# --- Terminal State ---

def capture_terminal_state(
    conn: MT5Connection, paths: StoragePaths, store: ParquetStore
) -> None:
    """Capture full terminal_info snapshot."""
    info = conn.terminal_info()
    if info is None:
        log.warning("terminal_info_empty", broker=conn.broker_id)
        return

    now = utc_now()
    d = _named_tuple_to_dict(info)

    row: dict[str, Any] = {"broker_id": conn.broker_id, "snapshot_ts": now}
    known_fields = {
        "connected", "community_account", "community_connection", "dlls_allowed",
        "trade_allowed", "tradeapi_disabled", "email_enabled", "ftp_enabled",
        "notifications_enabled", "mqid", "build", "maxbars", "codepage", "ping_last",
        "community_balance", "retransmission", "company", "name", "language",
        "path", "data_path", "commondata_path",
    }
    for field in known_fields:
        if field in d:
            row[field] = d[field]

    df = pl.DataFrame([row])
    path = paths.terminal_state_file(conn.broker_id, now.date())
    store.write(df, path)
    log.info("terminal_state_captured", broker=conn.broker_id)


# --- Active Orders ---

def capture_active_orders(
    conn: MT5Connection, symbol: str, paths: StoragePaths, store: ParquetStore
) -> None:
    """Capture current active orders."""
    orders = conn.orders_get(symbol)
    now = utc_now()

    if orders is None or len(orders) == 0:
        log.debug("no_active_orders", broker=conn.broker_id, symbol=symbol)
        return

    rows = []
    for o in orders:
        d = _named_tuple_to_dict(o)
        d["broker_id"] = conn.broker_id
        d["snapshot_ts"] = now
        if "time_setup" in d and isinstance(d["time_setup"], (int, float)):
            d["time_setup"] = dt.datetime.fromtimestamp(int(d["time_setup"]), tz=dt.timezone.utc)
        rows.append(d)

    df = pl.DataFrame(rows)
    path = paths.orders_active_file(conn.broker_id, now.date())
    store.write(df, path)
    log.info("active_orders_captured", broker=conn.broker_id, count=len(rows))


# --- Active Positions ---

def capture_active_positions(
    conn: MT5Connection, symbol: str, paths: StoragePaths, store: ParquetStore
) -> None:
    """Capture current active positions."""
    positions = conn.positions_get(symbol)
    now = utc_now()

    if positions is None or len(positions) == 0:
        log.debug("no_active_positions", broker=conn.broker_id, symbol=symbol)
        return

    rows = []
    for p in positions:
        d = _named_tuple_to_dict(p)
        d["broker_id"] = conn.broker_id
        d["snapshot_ts"] = now
        if "time" in d and isinstance(d["time"], (int, float)):
            d["time"] = dt.datetime.fromtimestamp(int(d["time"]), tz=dt.timezone.utc)
        rows.append(d)

    df = pl.DataFrame(rows)
    path = paths.positions_active_file(conn.broker_id, now.date())
    store.write(df, path)
    log.info("active_positions_captured", broker=conn.broker_id, count=len(rows))
