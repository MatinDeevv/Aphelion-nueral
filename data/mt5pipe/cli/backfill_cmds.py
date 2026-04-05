"""Backfill CLI commands."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import typer

from mt5pipe.config.loader import load_config
from mt5pipe.utils.logging import get_logger, setup_logging

log = get_logger(__name__)

backfill_app = typer.Typer(help="Backfill historical data from MT5")


def _parse_date(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)


def _parse_time(s: str | None) -> dt.time | None:
    if s is None:
        return None
    return dt.datetime.strptime(s, "%H:%M").time()


@backfill_app.command("ticks")
def backfill_ticks(
    broker: str = typer.Option(..., help="Broker ID from config"),
    symbol: str = typer.Option("XAUUSD", help="Symbol to backfill"),
    date_from: str = typer.Option(..., "--from", help="Start date YYYY-MM-DD"),
    date_to: str = typer.Option(..., "--to", help="End date YYYY-MM-DD"),
    config_path: str = typer.Option("config/pipeline.yaml", "--config", help="Config file path"),
) -> None:
    """Backfill historical ticks for a broker/symbol."""
    cfg = load_config(Path(config_path))
    setup_logging(cfg.logging.level, cfg.logging.json_output)

    from mt5pipe.backfill.engine import BackfillEngine
    from mt5pipe.mt5.connection import MT5Connection
    from mt5pipe.storage.checkpoint_db import CheckpointDB
    from mt5pipe.storage.parquet_store import ParquetStore
    from mt5pipe.storage.paths import StoragePaths

    broker_cfg = cfg.get_broker(broker)
    conn = MT5Connection(broker_cfg)
    paths = StoragePaths(cfg.storage.root)
    store = ParquetStore(cfg.storage.compression, cfg.storage.parquet_row_group_size)
    db = CheckpointDB(paths.checkpoint_db_path())

    start = _parse_date(date_from)
    end = _parse_date(date_to)

    with conn.connect():
        engine = BackfillEngine(conn, paths, store, db, cfg.backfill)
        total = engine.backfill_ticks(symbol, start, end)
        typer.echo(f"Backfilled {total:,} ticks for {broker}/{symbol}")

    db.close()


@backfill_app.command("sync-ticks")
def backfill_sync_ticks(
    broker_a: str = typer.Option(..., help="First broker ID from config"),
    broker_b: str = typer.Option(..., help="Second broker ID from config"),
    symbol: str = typer.Option("XAUUSD", help="Symbol to backfill"),
    date_from: str = typer.Option(..., "--from", help="Start date YYYY-MM-DD"),
    date_to: str = typer.Option(..., "--to", help="End date YYYY-MM-DD"),
    hours_start: str | None = typer.Option(
        None,
        "--hours-start",
        help="Optional UTC intraday window start, HH:MM",
    ),
    hours_end: str | None = typer.Option(
        None,
        "--hours-end",
        help="Optional UTC intraday window end, HH:MM",
    ),
    config_path: str = typer.Option("config/pipeline.yaml", "--config", help="Config file path"),
) -> None:
    """Backfill raw ticks for both brokers over the same UTC date range."""
    cfg = load_config(Path(config_path))
    setup_logging(cfg.logging.level, cfg.logging.json_output)

    from mt5pipe.backfill.sync import format_synchronized_backfill_summary, run_synchronized_tick_backfill
    from mt5pipe.storage.parquet_store import ParquetStore
    from mt5pipe.storage.paths import StoragePaths

    start_date = dt.date.fromisoformat(date_from)
    end_date = dt.date.fromisoformat(date_to)
    hours_start_utc = _parse_time(hours_start)
    hours_end_utc = _parse_time(hours_end)

    paths = StoragePaths(cfg.storage.root)
    store = ParquetStore(cfg.storage.compression, cfg.storage.parquet_row_group_size)

    summaries = run_synchronized_tick_backfill(
        cfg.get_broker(broker_a),
        cfg.get_broker(broker_b),
        symbol,
        start_date,
        end_date,
        paths=paths,
        store=store,
        checkpoint_db_path=paths.checkpoint_db_path(),
        backfill_cfg=cfg.backfill,
        hours_start_utc=hours_start_utc,
        hours_end_utc=hours_end_utc,
    )

    typer.echo(
        format_synchronized_backfill_summary(
            summaries,
            start_date=start_date,
            end_date=end_date,
            hours_start_utc=hours_start_utc,
            hours_end_utc=hours_end_utc,
        )
    )


@backfill_app.command("bars")
def backfill_bars(
    broker: str = typer.Option(..., help="Broker ID"),
    symbol: str = typer.Option("XAUUSD"),
    timeframe: str = typer.Option("M5", help="Timeframe, e.g. M1, M5, H1"),
    date_from: str = typer.Option(..., "--from"),
    date_to: str = typer.Option(..., "--to"),
    config_path: str = typer.Option("config/pipeline.yaml", "--config"),
) -> None:
    """Backfill native bars for a broker/symbol/timeframe."""
    cfg = load_config(Path(config_path))
    setup_logging(cfg.logging.level, cfg.logging.json_output)

    from mt5pipe.backfill.engine import BackfillEngine
    from mt5pipe.mt5.connection import MT5Connection
    from mt5pipe.storage.checkpoint_db import CheckpointDB
    from mt5pipe.storage.parquet_store import ParquetStore
    from mt5pipe.storage.paths import StoragePaths

    broker_cfg = cfg.get_broker(broker)
    conn = MT5Connection(broker_cfg)
    paths = StoragePaths(cfg.storage.root)
    store = ParquetStore(cfg.storage.compression, cfg.storage.parquet_row_group_size)
    db = CheckpointDB(paths.checkpoint_db_path())

    with conn.connect():
        engine = BackfillEngine(conn, paths, store, db, cfg.backfill)
        total = engine.backfill_bars(symbol, timeframe, _parse_date(date_from), _parse_date(date_to))
        typer.echo(f"Backfilled {total:,} bars ({timeframe}) for {broker}/{symbol}")

    db.close()


@backfill_app.command("history-orders")
def backfill_history_orders(
    broker: str = typer.Option(...),
    date_from: str = typer.Option(..., "--from"),
    date_to: str = typer.Option(..., "--to"),
    config_path: str = typer.Option("config/pipeline.yaml", "--config"),
) -> None:
    """Backfill historical orders."""
    cfg = load_config(Path(config_path))
    setup_logging(cfg.logging.level, cfg.logging.json_output)

    from mt5pipe.backfill.engine import BackfillEngine
    from mt5pipe.mt5.connection import MT5Connection
    from mt5pipe.storage.checkpoint_db import CheckpointDB
    from mt5pipe.storage.parquet_store import ParquetStore
    from mt5pipe.storage.paths import StoragePaths

    broker_cfg = cfg.get_broker(broker)
    conn = MT5Connection(broker_cfg)
    paths = StoragePaths(cfg.storage.root)
    store = ParquetStore(cfg.storage.compression, cfg.storage.parquet_row_group_size)
    db = CheckpointDB(paths.checkpoint_db_path())

    with conn.connect():
        engine = BackfillEngine(conn, paths, store, db, cfg.backfill)
        total = engine.backfill_history_orders(_parse_date(date_from), _parse_date(date_to))
        typer.echo(f"Backfilled {total:,} historical orders for {broker}")

    db.close()


@backfill_app.command("history-deals")
def backfill_history_deals(
    broker: str = typer.Option(...),
    date_from: str = typer.Option(..., "--from"),
    date_to: str = typer.Option(..., "--to"),
    config_path: str = typer.Option("config/pipeline.yaml", "--config"),
) -> None:
    """Backfill historical deals."""
    cfg = load_config(Path(config_path))
    setup_logging(cfg.logging.level, cfg.logging.json_output)

    from mt5pipe.backfill.engine import BackfillEngine
    from mt5pipe.mt5.connection import MT5Connection
    from mt5pipe.storage.checkpoint_db import CheckpointDB
    from mt5pipe.storage.parquet_store import ParquetStore
    from mt5pipe.storage.paths import StoragePaths

    broker_cfg = cfg.get_broker(broker)
    conn = MT5Connection(broker_cfg)
    paths = StoragePaths(cfg.storage.root)
    store = ParquetStore(cfg.storage.compression, cfg.storage.parquet_row_group_size)
    db = CheckpointDB(paths.checkpoint_db_path())

    with conn.connect():
        engine = BackfillEngine(conn, paths, store, db, cfg.backfill)
        total = engine.backfill_history_deals(_parse_date(date_from), _parse_date(date_to))
        typer.echo(f"Backfilled {total:,} historical deals for {broker}")

    db.close()


@backfill_app.command("symbol-metadata")
def backfill_symbol_metadata(
    broker: str = typer.Option(...),
    config_path: str = typer.Option("config/pipeline.yaml", "--config"),
) -> None:
    """Capture symbol metadata and universe snapshot."""
    cfg = load_config(Path(config_path))
    setup_logging(cfg.logging.level, cfg.logging.json_output)

    from mt5pipe.backfill.engine import BackfillEngine
    from mt5pipe.mt5.connection import MT5Connection
    from mt5pipe.storage.checkpoint_db import CheckpointDB
    from mt5pipe.storage.parquet_store import ParquetStore
    from mt5pipe.storage.paths import StoragePaths

    broker_cfg = cfg.get_broker(broker)
    conn = MT5Connection(broker_cfg)
    paths = StoragePaths(cfg.storage.root)
    store = ParquetStore(cfg.storage.compression, cfg.storage.parquet_row_group_size)
    db = CheckpointDB(paths.checkpoint_db_path())

    with conn.connect():
        engine = BackfillEngine(conn, paths, store, db, cfg.backfill)
        engine.backfill_symbol_metadata(cfg.symbols)
        typer.echo(f"Symbol metadata captured for {broker}")

    db.close()
