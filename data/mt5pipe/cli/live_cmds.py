"""Live collection CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from mt5pipe.config.loader import load_config
from mt5pipe.utils.logging import get_logger, setup_logging

log = get_logger(__name__)

live_app = typer.Typer(help="Live continuous data collection")


@live_app.command("collect")
def live_collect(
    broker: str = typer.Option(..., help="Broker ID"),
    symbol: str = typer.Option("XAUUSD", help="Symbol to collect"),
    enable_book: bool = typer.Option(True, help="Enable market book/DOM collection"),
    duration: int = typer.Option(0, help="Collect for N seconds then exit (0=infinite, Ctrl+C to stop)"),
    config_path: str = typer.Option("config/pipeline.yaml", "--config"),
) -> None:
    """Start live tick and snapshot collection for a broker/symbol."""
    cfg = load_config(Path(config_path))
    setup_logging(cfg.logging.level, cfg.logging.json_output)

    from mt5pipe.live.collector import LiveCollector
    from mt5pipe.mt5.connection import MT5Connection
    from mt5pipe.storage.checkpoint_db import CheckpointDB
    from mt5pipe.storage.parquet_store import ParquetStore
    from mt5pipe.storage.paths import StoragePaths

    broker_cfg = cfg.get_broker(broker)
    conn = MT5Connection(broker_cfg)
    paths = StoragePaths(cfg.storage.root)
    store = ParquetStore(cfg.storage.compression, cfg.storage.parquet_row_group_size)
    db = CheckpointDB(paths.checkpoint_db_path())

    mode = f"for {duration}s" if duration > 0 else "until Ctrl+C"
    typer.echo(f"Starting live collection for {broker}/{symbol} ({mode})")

    with conn.connect():
        collector = LiveCollector(
            conn, symbol, paths, store, db, cfg.live,
            enable_market_book=enable_book,
        )
        collector.start(duration_seconds=duration)

    db.close()
    typer.echo("Live collection stopped.")
