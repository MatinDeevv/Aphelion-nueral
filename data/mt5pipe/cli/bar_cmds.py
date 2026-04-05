"""Bar builder CLI commands."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import typer

from mt5pipe.config.loader import load_config
from mt5pipe.utils.logging import setup_logging

bar_app = typer.Typer(help="Build bars from canonical ticks")


@bar_app.command("build")
def build_bars(
    symbol: str = typer.Option("XAUUSD"),
    timeframe: str = typer.Option("", help="Single timeframe or empty for all configured"),
    date_from: str = typer.Option(..., "--from"),
    date_to: str = typer.Option(..., "--to"),
    config_path: str = typer.Option("config/pipeline.yaml", "--config"),
) -> None:
    """Build bars from canonical ticks for one or all timeframes."""
    cfg = load_config(Path(config_path))
    setup_logging(cfg.logging.level, cfg.logging.json_output)

    from mt5pipe.bars.builder import build_bars_date_range
    from mt5pipe.storage.parquet_store import ParquetStore
    from mt5pipe.storage.paths import StoragePaths

    paths = StoragePaths(cfg.storage.root)
    store = ParquetStore(cfg.storage.compression, cfg.storage.parquet_row_group_size)

    start = dt.date.fromisoformat(date_from)
    end = dt.date.fromisoformat(date_to)

    timeframes = [timeframe] if timeframe else cfg.bars.timeframes
    grand_total = 0

    for tf in timeframes:
        total = build_bars_date_range(symbol, tf, start, end, paths, store)
        grand_total += total
        typer.echo(f"  {tf}: {total:,} bars")

    typer.echo(f"Total: {grand_total:,} bars built for {symbol}")
