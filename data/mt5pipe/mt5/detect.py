"""Auto-detection of MT5 broker capabilities.

Connects to each configured broker terminal (session-mode or credential),
probes available symbols, date ranges, L2 support, and account info,
then disconnects.  Each call is self-contained and leaves no MT5 session open.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from mt5pipe.config.models import BrokerConfig


@dataclass
class SymbolCaps:
    """Per-symbol capability detection results."""

    canonical: str
    broker_name: str = ""
    available: bool = False
    digits: int = 0
    point: float = 0.0
    spread: int = 0
    first_tick_date: dt.date | None = None
    first_bar_date: dt.date | None = None
    market_book_ok: bool = False


@dataclass
class BrokerCaps:
    """Full capability detection results for one broker terminal."""

    broker_id: str
    connected: bool = False
    error: str | None = None

    # Account
    account_login: int = 0
    account_name: str = ""
    account_server: str = ""
    account_balance: float = 0.0
    account_currency: str = ""
    account_leverage: int = 0

    # Terminal
    terminal_name: str = ""
    terminal_company: str = ""
    terminal_version: str = ""
    terminal_build: int = 0

    # Symbols
    total_symbols: int = 0
    symbols: dict[str, SymbolCaps] = field(default_factory=dict)

    # ---------- helpers ----------

    def earliest_date(self, symbol: str) -> dt.date | None:
        """Earliest available date (bar or tick) for *symbol*."""
        sc = self.symbols.get(symbol)
        if sc is None:
            return None
        dates = [d for d in (sc.first_bar_date, sc.first_tick_date) if d]
        return min(dates) if dates else None

    def has_market_book(self, symbol: str) -> bool:
        sc = self.symbols.get(symbol)
        return sc.market_book_ok if sc else False


def detect_broker(
    broker_cfg: BrokerConfig,
    target_symbols: list[str],
) -> BrokerCaps:
    """Connect -> probe -> disconnect.  Safe for sequential per-broker calls."""
    import MetaTrader5 as mt5  # lazy: may not be installed at import-time

    caps = BrokerCaps(broker_id=broker_cfg.broker_id)

    # --- connect (credential fallback -> session attach) ---
    pw = broker_cfg.password.get_secret_value() if broker_cfg.password else ""
    ok = False
    if pw and broker_cfg.login:
        ok = mt5.initialize(
            path=str(broker_cfg.terminal_path),
            login=broker_cfg.login,
            password=pw,
            server=broker_cfg.server or "",
            timeout=broker_cfg.timeout_ms,
        )
    if not ok:
        ok = mt5.initialize(
            path=str(broker_cfg.terminal_path),
            timeout=broker_cfg.timeout_ms,
        )
    if not ok:
        e = mt5.last_error()
        caps.error = f"error {e[0]}: {e[1]}"
        return caps

    caps.connected = True

    # --- account ---
    acct = mt5.account_info()
    if acct:
        caps.account_login = acct.login
        caps.account_name = acct.name
        caps.account_server = acct.server
        caps.account_balance = acct.balance
        caps.account_currency = acct.currency
        caps.account_leverage = acct.leverage

    # --- terminal ---
    term = mt5.terminal_info()
    if term:
        caps.terminal_name = term.name
        caps.terminal_company = term.company
        caps.terminal_build = term.build
    ver = mt5.version()
    if ver:
        caps.terminal_version = f"{ver[0]}.{ver[1]}"

    caps.total_symbols = mt5.symbols_total() or 0

    # --- per symbol ---
    epoch = dt.datetime(1970, 1, 2, tzinfo=dt.timezone.utc)

    for canonical in target_symbols:
        bsym = broker_cfg.resolve_symbol(canonical)
        sc = SymbolCaps(canonical=canonical, broker_name=bsym)

        mt5.symbol_select(bsym, True)
        info = mt5.symbol_info(bsym)
        if info is None:
            caps.symbols[canonical] = sc
            continue

        sc.available = True
        sc.digits = info.digits
        sc.point = info.point
        sc.spread = info.spread

        # earliest bar
        try:
            rates = mt5.copy_rates_from(bsym, mt5.TIMEFRAME_D1, epoch, 1)
            if rates is not None and len(rates) > 0:
                sc.first_bar_date = dt.datetime.fromtimestamp(
                    int(rates[0][0]), tz=dt.timezone.utc
                ).date()
        except Exception:
            pass

        # earliest tick
        try:
            ticks = mt5.copy_ticks_from(bsym, epoch, 1, mt5.COPY_TICKS_ALL)
            if ticks is not None and len(ticks) > 0:
                sc.first_tick_date = dt.datetime.fromtimestamp(
                    int(ticks[0][0]), tz=dt.timezone.utc
                ).date()
        except Exception:
            pass

        # L2 / market book
        try:
            book = mt5.market_book_add(bsym)
            sc.market_book_ok = bool(book)
            if book:
                mt5.market_book_release(bsym)
        except Exception:
            sc.market_book_ok = False

        caps.symbols[canonical] = sc

    mt5.shutdown()
    return caps
