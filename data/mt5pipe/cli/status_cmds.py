"""Status and validation CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from mt5pipe.config.loader import load_config
from mt5pipe.utils.logging import setup_logging

status_app = typer.Typer(help="Status and storage validation")
console = Console()


@status_app.command("show")
def show_status(
    config_path: str = typer.Option("config/pipeline.yaml", "--config"),
) -> None:
    """Show pipeline status — checkpoints, job history, storage stats."""
    cfg = load_config(Path(config_path))
    setup_logging(cfg.logging.level, cfg.logging.json_output)

    from mt5pipe.storage.checkpoint_db import CheckpointDB
    from mt5pipe.storage.paths import StoragePaths

    paths = StoragePaths(cfg.storage.root)
    db_path = paths.checkpoint_db_path()

    if not db_path.exists():
        typer.echo("No checkpoint database found. Run a backfill first.")
        return

    db = CheckpointDB(db_path)

    # Show checkpoints
    checkpoints = db.list_checkpoints()
    if checkpoints:
        table = Table(title="Ingestion Checkpoints")
        table.add_column("Broker", style="cyan")
        table.add_column("Symbol")
        table.add_column("Type")
        table.add_column("Timeframe")
        table.add_column("Last Timestamp")
        table.add_column("Rows", justify="right")
        table.add_column("Status", style="green")

        for cp in checkpoints:
            table.add_row(
                cp.broker_id,
                cp.symbol,
                cp.data_type,
                cp.timeframe or "-",
                cp.last_timestamp_utc.strftime("%Y-%m-%d %H:%M"),
                f"{cp.rows_ingested:,}",
                cp.status,
            )
        console.print(table)
    else:
        typer.echo("No checkpoints found.")

    # Show storage totals
    for data_type in ["ticks", "bars", "history_orders", "history_deals"]:
        total = db.get_total_rows(data_type=data_type)
        if total > 0:
            typer.echo(f"  {data_type}: {total:,} rows in manifest")

    db.close()


@status_app.command("validate")
def validate_storage(
    config_path: str = typer.Option("config/pipeline.yaml", "--config"),
) -> None:
    """Validate storage integrity — check parquet files are readable."""
    cfg = load_config(Path(config_path))
    setup_logging(cfg.logging.level, cfg.logging.json_output)

    from mt5pipe.storage.paths import StoragePaths

    root = cfg.storage.root
    if not root.exists():
        typer.echo(f"Data root does not exist: {root}")
        return

    import pyarrow.parquet as pq

    total_files = 0
    total_rows = 0
    errors = 0

    for pq_file in root.rglob("*.parquet"):
        total_files += 1
        try:
            meta = pq.read_metadata(pq_file)
            total_rows += meta.num_rows
        except Exception as exc:
            errors += 1
            typer.echo(f"  ERROR: {pq_file}: {exc}")

    typer.echo(f"Validated {total_files} Parquet files, {total_rows:,} total rows, {errors} errors")

    if errors == 0:
        typer.echo("Storage validation passed.")
    else:
        typer.echo(f"Storage validation failed with {errors} errors.", err=True)
        raise typer.Exit(1)
