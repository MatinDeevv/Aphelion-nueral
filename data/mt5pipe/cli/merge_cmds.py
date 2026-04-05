"""Merge CLI commands."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import typer

from mt5pipe.config.loader import load_config
from mt5pipe.utils.logging import setup_logging

merge_app = typer.Typer(help="Canonical tick merge operations")


def _parse_bucket_values(value: str) -> list[int]:
    buckets = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not buckets:
        raise typer.BadParameter("At least one bucket_ms value is required.")
    return buckets


@merge_app.command("canonical")
def merge_canonical(
    symbol: str = typer.Option("XAUUSD"),
    broker_a: str = typer.Option(..., help="First broker ID"),
    broker_b: str = typer.Option(..., help="Second broker ID"),
    date_from: str = typer.Option(..., "--from"),
    date_to: str = typer.Option(..., "--to"),
    config_path: str = typer.Option("config/pipeline.yaml", "--config"),
) -> None:
    """Merge raw ticks from two brokers into canonical feed."""
    cfg = load_config(Path(config_path))
    setup_logging(cfg.logging.level, cfg.logging.json_output)

    from mt5pipe.merge.canonical import merge_canonical_date_range
    from mt5pipe.quality.merge_qa import (
        assert_synchronized_raw_tick_coverage,
        build_daily_merge_qa_report,
        format_daily_merge_qa_summary,
    )
    from mt5pipe.storage.parquet_store import ParquetStore
    from mt5pipe.storage.paths import StoragePaths

    paths = StoragePaths(cfg.storage.root)
    store = ParquetStore(cfg.storage.compression, cfg.storage.parquet_row_group_size)

    start = dt.date.fromisoformat(date_from)
    end = dt.date.fromisoformat(date_to)

    broker_a_cfg = cfg.get_broker(broker_a)
    broker_b_cfg = cfg.get_broker(broker_b)

    assert_synchronized_raw_tick_coverage(paths, store, broker_a, broker_b, symbol, start, end)

    total = merge_canonical_date_range(
        broker_a, broker_b, symbol, start, end,
        paths, store, cfg.merge,
        broker_a_cfg.priority, broker_b_cfg.priority,
    )
    typer.echo(f"Merged {total:,} canonical ticks for {symbol}")

    summary_df = build_daily_merge_qa_report(
        paths,
        store,
        broker_a,
        broker_b,
        symbol,
        start,
        end,
        expected_bucket_ms=cfg.merge.bucket_ms,
    )
    typer.echo("")
    typer.echo(format_daily_merge_qa_summary(summary_df))


@merge_app.command("qa-report")
def merge_qa_report(
    symbol: str = typer.Option("XAUUSD"),
    broker_a: str = typer.Option(..., help="First broker ID"),
    broker_b: str = typer.Option(..., help="Second broker ID"),
    date_from: str = typer.Option(..., "--from"),
    date_to: str = typer.Option(..., "--to"),
    config_path: str = typer.Option("config/pipeline.yaml", "--config"),
) -> None:
    """Generate and persist a daily merge QA report for a UTC date range."""
    cfg = load_config(Path(config_path))
    setup_logging(cfg.logging.level, cfg.logging.json_output)

    from mt5pipe.quality.merge_qa import (
        build_daily_merge_qa_report,
        format_daily_merge_qa_summary,
        write_daily_merge_qa_report,
    )
    from mt5pipe.storage.parquet_store import ParquetStore
    from mt5pipe.storage.paths import StoragePaths

    paths = StoragePaths(cfg.storage.root)
    store = ParquetStore(cfg.storage.compression, cfg.storage.parquet_row_group_size)

    start = dt.date.fromisoformat(date_from)
    end = dt.date.fromisoformat(date_to)

    report_df = build_daily_merge_qa_report(
        paths,
        store,
        broker_a,
        broker_b,
        symbol,
        start,
        end,
        expected_bucket_ms=cfg.merge.bucket_ms,
    )
    written = write_daily_merge_qa_report(report_df, paths, store, symbol)

    typer.echo(f"Wrote {written:,} daily merge QA rows for {symbol}")
    typer.echo("")
    typer.echo(format_daily_merge_qa_summary(report_df))


@merge_app.command("bucket-sweep")
def merge_bucket_sweep(
    symbol: str = typer.Option("XAUUSD"),
    broker_a: str = typer.Option(..., help="First broker ID"),
    broker_b: str = typer.Option(..., help="Second broker ID"),
    date_from: str = typer.Option(..., "--from"),
    date_to: str = typer.Option(..., "--to"),
    bucket_values: str = typer.Option("50,75,100,125", "--buckets", help="Comma-separated bucket_ms values"),
    config_path: str = typer.Option("config/pipeline.yaml", "--config"),
) -> None:
    """Compare alternate bucket sizes after the daily merge QA report exists."""
    cfg = load_config(Path(config_path))
    setup_logging(cfg.logging.level, cfg.logging.json_output)

    from mt5pipe.quality.merge_qa import format_bucket_sweep_summary, run_bucket_sweep
    from mt5pipe.storage.parquet_store import ParquetStore
    from mt5pipe.storage.paths import StoragePaths

    paths = StoragePaths(cfg.storage.root)
    store = ParquetStore(cfg.storage.compression, cfg.storage.parquet_row_group_size)

    start = dt.date.fromisoformat(date_from)
    end = dt.date.fromisoformat(date_to)
    buckets = _parse_bucket_values(bucket_values)

    broker_a_cfg = cfg.get_broker(broker_a)
    broker_b_cfg = cfg.get_broker(broker_b)

    report_df = run_bucket_sweep(
        paths,
        store,
        broker_a,
        broker_b,
        symbol,
        start,
        end,
        cfg.merge,
        broker_a_priority=broker_a_cfg.priority,
        broker_b_priority=broker_b_cfg.priority,
        bucket_values=buckets,
    )
    typer.echo(format_bucket_sweep_summary(report_df))
