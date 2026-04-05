"""Checkpoint compile coverage for stable machine-native selectors."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from mt5pipe.compiler.public import compile_dataset_spec, inspect_artifact
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths


UTC = dt.timezone.utc


def _make_bars(start: dt.datetime, rows: int, timeframe: str, step_minutes: int) -> pl.DataFrame:
    times = [start + dt.timedelta(minutes=step_minutes * i) for i in range(rows)]
    base_open = [3000.0 + i * 0.002 + ((i % 30) - 15) * 0.03 for i in range(rows)]
    base_close = [price + (0.05 if i % 2 == 0 else -0.05) for i, price in enumerate(base_open)]
    base_high = [max(base_open[i], base_close[i]) + 0.05 + (i % 5) * 0.005 for i in range(rows)]
    base_low = [min(base_open[i], base_close[i]) - 0.05 - (i % 5) * 0.005 for i in range(rows)]
    dual_ratio = [0.75 + (i % 4) * 0.02 for i in range(rows)]
    return pl.DataFrame(
        {
            "symbol": ["XAUUSD"] * rows,
            "timeframe": [timeframe] * rows,
            "time_utc": times,
            "open": base_open,
            "high": base_high,
            "low": base_low,
            "close": base_close,
            "tick_count": [12] * rows,
            "bid_open": [base_open[i] - 0.05 for i in range(rows)],
            "ask_open": [base_open[i] + 0.05 for i in range(rows)],
            "bid_close": [base_close[i] - 0.05 for i in range(rows)],
            "ask_close": [base_close[i] + 0.05 for i in range(rows)],
            "spread_mean": [0.1 + (i % 5) * 0.001 for i in range(rows)],
            "spread_max": [0.12 + (i % 5) * 0.001 for i in range(rows)],
            "spread_min": [0.08 + (i % 5) * 0.001 for i in range(rows)],
            "mid_return": [0.0001 + (i % 7) * 0.00001 for i in range(rows)],
            "realized_vol": [0.0005 + (i % 9) * 0.00001 for i in range(rows)],
            "volume_sum": [10.0] * rows,
            "source_count": [1 + (i % 2) for i in range(rows)],
            "conflict_count": [i % 3 for i in range(rows)],
            "dual_source_ticks": [9 + (i % 3) for i in range(rows)],
            "secondary_present_ticks": [9 + (i % 3) for i in range(rows)],
            "dual_source_ratio": dual_ratio,
        }
    )


def _write_bars_by_date(df: pl.DataFrame, paths: StoragePaths, store: ParquetStore, timeframe: str) -> None:
    dated = df.with_columns(pl.col("time_utc").dt.date().alias("_date"))
    for date_val in dated["_date"].unique().sort().to_list():
        day_df = dated.filter(pl.col("_date") == date_val).drop("_date")
        store.write(day_df, paths.built_bars_file("XAUUSD", timeframe, date_val))


def _write_merge_qa(date: dt.date, paths: StoragePaths, store: ParquetStore, dual_source_ratio: float = 0.20) -> None:
    df = pl.DataFrame(
        [
            {
                "time_utc": dt.datetime.combine(date, dt.time(0, 0), tzinfo=UTC),
                "date": date.isoformat(),
                "symbol": "XAUUSD",
                "broker_a_id": "broker_a",
                "broker_b_id": "broker_b",
                "dual_source_ratio": dual_source_ratio,
                "conflicts": 0,
            }
        ]
    )
    store.write(df, paths.merge_qa_file("XAUUSD", date))


def _write_project_config(project_root: Path, storage_root: Path) -> None:
    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "pipeline.yaml").write_text(
        "\n".join(
            [
                "brokers: {}",
                "storage:",
                f"  root: \"{storage_root.as_posix()}\"",
                "  checkpoint_db: \"checkpoints.db\"",
                "  parquet_row_group_size: 1000",
                "  compression: \"snappy\"",
                "dataset:",
                "  base_timeframe: \"M1\"",
                "  context_timeframes:",
                "    - \"M5\"",
                "    - \"M15\"",
                "    - \"H1\"",
                "    - \"H4\"",
                "    - \"D1\"",
                "logging:",
                "  level: \"INFO\"",
                "  json_output: false",
            ]
        ),
        encoding="utf-8",
    )


def _write_spec(path: Path, *, selectors: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                'schema_version: "1.0.0"',
                'dataset_name: "xau_nonhuman_ckpt"',
                'version: "1.0.0"',
                'description: "checkpoint compile for stable machine-native selectors"',
                "symbols:",
                '  - "XAUUSD"',
                'date_from: "2026-04-01"',
                'date_to: "2026-04-02"',
                'base_clock: "M1"',
                'state_version_ref: "state.default@1.0.0"',
                "feature_selectors:",
                *[f'  - "{selector}"' for selector in selectors],
                'label_pack_ref: "core_tb_volscaled@1.0.0"',
                "filters:",
                '  - "exclude:filled_rows"',
                'split_policy: "temporal_holdout"',
                "train_ratio: 0.70",
                "val_ratio: 0.15",
                "test_ratio: 0.15",
                "embargo_rows: 240",
                'truth_policy_ref: "truth.default@1.0.0"',
                "publish_on_accept: true",
            ]
        ),
        encoding="utf-8",
    )


def _seed_project_data(storage_root: Path) -> StoragePaths:
    paths = StoragePaths(storage_root)
    store = ParquetStore(compression="snappy", row_group_size=1000)

    _write_bars_by_date(_make_bars(dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC), 2000, "M1", 1), paths, store, "M1")
    _write_bars_by_date(_make_bars(dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC), 500, "M5", 5), paths, store, "M5")
    _write_bars_by_date(_make_bars(dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC), 200, "M15", 15), paths, store, "M15")
    _write_bars_by_date(_make_bars(dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC), 48, "H1", 60), paths, store, "H1")
    _write_bars_by_date(_make_bars(dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC), 12, "H4", 240), paths, store, "H4")
    _write_bars_by_date(_make_bars(dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC), 2, "D1", 24 * 60), paths, store, "D1")
    _write_merge_qa(dt.date(2026, 4, 1), paths, store)
    _write_merge_qa(dt.date(2026, 4, 2), paths, store)
    return paths


def test_checkpoint_compile_accepts_stable_machine_native_selector_set(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project_checkpoint"
    storage_root = project_root / "local_data" / "pipeline_data"
    storage_root.mkdir(parents=True, exist_ok=True)
    _seed_project_data(storage_root)
    _write_project_config(project_root, storage_root)

    spec_path = project_root / "config" / "datasets" / "xau_m1_nonhuman_checkpoint.yaml"
    selectors = [
        "time/*",
        "session/*",
        "quality/*",
        "htf_context/*",
        "disagreement/*",
        "event_shape/*",
        "entropy/*",
        "multiscale/*",
    ]
    _write_spec(spec_path, selectors=selectors)

    monkeypatch.chdir(project_root)

    result = compile_dataset_spec(spec_path)
    inspected = inspect_artifact(f"dataset://{result.spec.dataset_name}@{result.spec.version}")

    assert result.manifest.status == "published"
    assert result.trust_report.accepted_for_publication is True
    assert result.split_row_counts["train"] > 0
    assert result.split_row_counts["val"] > 0
    assert result.split_row_counts["test"] > 0
    assert inspected.requested_feature_selectors == selectors
    assert set(inspected.feature_families) == {
        "time",
        "session",
        "quality",
        "htf_context",
        "disagreement",
        "event_shape",
        "entropy",
        "multiscale",
    }
