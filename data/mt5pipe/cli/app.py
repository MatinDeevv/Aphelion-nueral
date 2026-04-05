"""Main Typer CLI application."""

from __future__ import annotations

import typer

from mt5pipe.cli.backfill_cmds import backfill_app
from mt5pipe.cli.bar_cmds import bar_app
from mt5pipe.cli.dataset_cmds import dataset_app
from mt5pipe.cli.live_cmds import live_app
from mt5pipe.cli.merge_cmds import merge_app
from mt5pipe.cli.status_cmds import status_app
from mt5pipe.cli.train_cmds import train_app

app = typer.Typer(
    name="mt5pipe",
    help="Production-grade MetaTrader 5 data pipeline for AI/ML training.",
    no_args_is_help=True,
)

# Register sub-commands
app.add_typer(backfill_app, name="backfill", help="Backfill historical data")
app.add_typer(live_app, name="live", help="Live data collection")
app.add_typer(merge_app, name="merge", help="Canonical tick merge")
app.add_typer(bar_app, name="bars", help="Build bars from ticks")
app.add_typer(dataset_app, name="dataset", help="Build model-ready datasets")
app.add_typer(train_app, name="train", help="Run trust-gated experiments")
app.add_typer(status_app, name="status", help="Pipeline status and validation")


@app.command("validate-storage")
def validate_storage_shortcut(
    config_path: str = typer.Option("config/pipeline.yaml", "--config"),
) -> None:
    """Shortcut for status validate."""
    from mt5pipe.cli.status_cmds import validate_storage
    validate_storage(config_path)


if __name__ == "__main__":
    app()
