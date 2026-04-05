"""MT5 connection manager — multi-terminal support with retries."""

from __future__ import annotations

import datetime as dt
import json
from contextlib import contextmanager
from typing import Any, Generator

import MetaTrader5 as mt5
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from mt5pipe.config.models import BrokerConfig
from mt5pipe.utils.logging import get_logger
from mt5pipe.utils.time import utc_now

log = get_logger(__name__)


class MT5ConnectionError(Exception):
    """Raised when MT5 connection fails."""
    pass


class MT5Connection:
    """Manages a connection to a single MT5 terminal instance.
    
    MT5 Python API is process-global (only one terminal at a time),
    so callers must coordinate access via the context manager or explicit
    initialize/shutdown calls.
    """

    def __init__(self, broker_cfg: BrokerConfig) -> None:
        self.broker_id = broker_cfg.broker_id
        self._cfg = broker_cfg
        self._connected = False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(MT5ConnectionError),
        reraise=True,
    )
    def initialize(self) -> None:
        """Initialize connection to MT5 terminal."""
        log.info("mt5_init_start", broker=self.broker_id, terminal=str(self._cfg.terminal_path))

        password = self._cfg.password.get_secret_value() if self._cfg.password else ""
        last_err: tuple[int, str] | None = None

        # Prefer explicit credential login when password is provided.
        if password:
            if mt5.initialize(
                path=str(self._cfg.terminal_path),
                login=self._cfg.login,
                password=password,
                server=self._cfg.server,
                timeout=self._cfg.timeout_ms,
            ):
                self._connected = True
                log.info("mt5_init_ok", broker=self.broker_id, mode="credentials")
                return
            last_err = mt5.last_error()
            log.warning("mt5_init_credentials_failed", broker=self.broker_id, error=last_err)

        # Fallback: attach to an already logged-in terminal session.
        if mt5.initialize(
            path=str(self._cfg.terminal_path),
            timeout=self._cfg.timeout_ms,
        ):
            self._connected = True
            log.info("mt5_init_ok", broker=self.broker_id, mode="session")
            return

        err = mt5.last_error()
        log.error("mt5_init_failed", broker=self.broker_id, error=err, previous_error=last_err)
        raise MT5ConnectionError(f"MT5 init failed for {self.broker_id}: {err}")

    def shutdown(self) -> None:
        """Shutdown MT5 connection."""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            log.info("mt5_shutdown", broker=self.broker_id)

    def health_check(self) -> bool:
        """Check if terminal is connected and responsive."""
        if not self._connected:
            return False
        info = mt5.terminal_info()
        if info is None:
            return False
        return bool(info.connected)

    def ensure_connected(self) -> None:
        """Reconnect if connection dropped."""
        if not self.health_check():
            log.warning("mt5_reconnect", broker=self.broker_id)
            self.shutdown()
            self.initialize()

    @contextmanager
    def connect(self) -> Generator[MT5Connection, None, None]:
        """Context manager for MT5 connection lifecycle."""
        self.initialize()
        try:
            yield self
        finally:
            self.shutdown()

    # --- Data retrieval methods ---

    def copy_ticks_range(
        self,
        symbol: str,
        date_from: dt.datetime,
        date_to: dt.datetime,
        flags: int = 0,  # COPY_TICKS_ALL
    ) -> Any:
        """Wrapper around mt5.copy_ticks_range."""
        broker_symbol = self._cfg.resolve_symbol(symbol)
        result = mt5.copy_ticks_range(broker_symbol, date_from, date_to, flags)
        if result is None:
            err = mt5.last_error()
            log.warning("copy_ticks_range_empty", broker=self.broker_id, symbol=broker_symbol, error=err)
        return result

    def copy_rates_range(
        self,
        symbol: str,
        timeframe: int,
        date_from: dt.datetime,
        date_to: dt.datetime,
    ) -> Any:
        """Wrapper around mt5.copy_rates_range."""
        broker_symbol = self._cfg.resolve_symbol(symbol)
        result = mt5.copy_rates_range(broker_symbol, timeframe, date_from, date_to)
        if result is None:
            err = mt5.last_error()
            log.warning("copy_rates_range_empty", broker=self.broker_id, symbol=broker_symbol, error=err)
        return result

    def symbol_info(self, symbol: str) -> Any:
        broker_symbol = self._cfg.resolve_symbol(symbol)
        # Ensure symbol is selected in MarketWatch
        mt5.symbol_select(broker_symbol, True)
        return mt5.symbol_info(broker_symbol)

    def symbol_info_tick(self, symbol: str) -> Any:
        broker_symbol = self._cfg.resolve_symbol(symbol)
        return mt5.symbol_info_tick(broker_symbol)

    def symbols_total(self) -> int:
        return mt5.symbols_total() or 0

    def symbols_get(self, group: str = "") -> Any:
        if group:
            return mt5.symbols_get(group=group)
        return mt5.symbols_get()

    def account_info(self) -> Any:
        return mt5.account_info()

    def terminal_info(self) -> Any:
        return mt5.terminal_info()

    def orders_get(self, symbol: str = "") -> Any:
        if symbol:
            broker_symbol = self._cfg.resolve_symbol(symbol)
            return mt5.orders_get(symbol=broker_symbol)
        return mt5.orders_get()

    def positions_get(self, symbol: str = "") -> Any:
        if symbol:
            broker_symbol = self._cfg.resolve_symbol(symbol)
            return mt5.positions_get(symbol=broker_symbol)
        return mt5.positions_get()

    def history_orders_get(
        self, date_from: dt.datetime, date_to: dt.datetime, group: str = ""
    ) -> Any:
        if group:
            return mt5.history_orders_get(date_from, date_to, group=group)
        return mt5.history_orders_get(date_from, date_to)

    def history_deals_get(
        self, date_from: dt.datetime, date_to: dt.datetime, group: str = ""
    ) -> Any:
        if group:
            return mt5.history_deals_get(date_from, date_to, group=group)
        return mt5.history_deals_get(date_from, date_to)

    def market_book_add(self, symbol: str) -> bool:
        broker_symbol = self._cfg.resolve_symbol(symbol)
        return bool(mt5.market_book_add(broker_symbol))

    def market_book_get(self, symbol: str) -> Any:
        broker_symbol = self._cfg.resolve_symbol(symbol)
        return mt5.market_book_get(broker_symbol)

    def market_book_release(self, symbol: str) -> bool:
        broker_symbol = self._cfg.resolve_symbol(symbol)
        return bool(mt5.market_book_release(broker_symbol))

    def version(self) -> tuple[int, int, str] | None:
        return mt5.version()

    def last_error(self) -> tuple[int, str]:
        return mt5.last_error()
