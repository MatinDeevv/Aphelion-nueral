"""Tests for the first trust-gated training, experiment, and model registry path."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from mt5pipe.catalog.sqlite import CatalogDB
from mt5pipe.cli.app import app
from mt5pipe.compiler.service import compile_dataset_spec
from mt5pipe.compiler.training import inspect_experiment, inspect_model, run_experiment_spec
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths
from tests.test_compiler import (
    UTC,
    _write_bars_by_date,
    _write_merge_qa,
    _write_project_config,
    _write_spec,
    _write_state_fixture_partitions,
)


runner = CliRunner()


def _make_training_bars(start: dt.datetime, rows: int, timeframe: str, step_minutes: int) -> pl.DataFrame:
    times = [start + dt.timedelta(minutes=step_minutes * index) for index in range(rows)]
    base_open: list[float] = []
    base_close: list[float] = []
    base_high: list[float] = []
    base_low: list[float] = []
    dual_ratio: list[float] = []
    anchor = 3000.0

    for index in range(rows):
        regime = (index // max(1, 180 // step_minutes)) % 4
        drift = [0.09, -0.11, 0.12, -0.08][regime]
        oscillation = ((index % max(6, 60 // step_minutes)) - max(3, 30 // step_minutes)) * 0.01
        open_price = anchor + drift * 0.35 + oscillation
        close_price = anchor + drift + oscillation * 0.5
        high_price = max(open_price, close_price) + 0.12 + (index % 4) * 0.01
        low_price = min(open_price, close_price) - 0.12 - (index % 4) * 0.01
        base_open.append(open_price)
        base_close.append(close_price)
        base_high.append(high_price)
        base_low.append(low_price)
        dual_ratio.append(0.70 + (index % 5) * 0.03)
        anchor = close_price + drift * 0.2

    return pl.DataFrame(
        {
            "symbol": ["XAUUSD"] * rows,
            "timeframe": [timeframe] * rows,
            "time_utc": times,
            "open": base_open,
            "high": base_high,
            "low": base_low,
            "close": base_close,
            "tick_count": [12 + (index % 4) for index in range(rows)],
            "bid_open": [base_open[index] - 0.05 for index in range(rows)],
            "ask_open": [base_open[index] + 0.05 for index in range(rows)],
            "bid_close": [base_close[index] - 0.05 for index in range(rows)],
            "ask_close": [base_close[index] + 0.05 for index in range(rows)],
            "spread_mean": [0.10 + (index % 6) * 0.002 for index in range(rows)],
            "spread_max": [0.12 + (index % 6) * 0.002 for index in range(rows)],
            "spread_min": [0.08 + (index % 6) * 0.002 for index in range(rows)],
            "mid_return": [base_close[index] - base_open[index] for index in range(rows)],
            "realized_vol": [0.0005 + abs(base_close[index] - base_open[index]) * 0.001 for index in range(rows)],
            "volume_sum": [10.0 + (index % 3) for index in range(rows)],
            "source_count": [1 + (index % 2) for index in range(rows)],
            "conflict_count": [index % 3 for index in range(rows)],
            "dual_source_ticks": [8 + (index % 4) for index in range(rows)],
            "secondary_present_ticks": [8 + (index % 4) for index in range(rows)],
            "dual_source_ratio": dual_ratio,
        }
    )


def _seed_training_project_data(storage_root: Path, *, start_date: dt.date = dt.date(2026, 4, 1), days: int = 5) -> StoragePaths:
    paths = StoragePaths(storage_root)
    store = ParquetStore(compression="snappy", row_group_size=1000)
    start_dt = dt.datetime.combine(start_date, dt.time(0, 0), tzinfo=UTC)

    def _rows(step_minutes: int) -> int:
        return max(1, (days * 24 * 60) // step_minutes)

    m1_df = _make_training_bars(start_dt, _rows(1), "M1", 1)
    _write_bars_by_date(m1_df, paths, store, "M1")
    _write_bars_by_date(_make_training_bars(start_dt, _rows(5), "M5", 5), paths, store, "M5")
    _write_bars_by_date(_make_training_bars(start_dt, _rows(15), "M15", 15), paths, store, "M15")
    _write_bars_by_date(_make_training_bars(start_dt, _rows(60), "H1", 60), paths, store, "H1")
    _write_bars_by_date(_make_training_bars(start_dt, _rows(240), "H4", 240), paths, store, "H4")
    _write_bars_by_date(_make_training_bars(start_dt, max(days, 1), "D1", 24 * 60), paths, store, "D1")
    _write_state_fixture_partitions(m1_df, paths, store)
    for offset in range(days):
        _write_merge_qa(start_date + dt.timedelta(days=offset), paths, store, dual_source_ratio=0.35)
    return paths


def _write_experiment_spec(
    path: Path,
    *,
    dataset_ref: str,
    version: str = "1.0.0",
    min_train_rows: int = 2000,
    min_walk_forward_balanced_accuracy: float = 0.45,
    min_test_balanced_accuracy: float = 0.45,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                'schema_version: "1.0.0"',
                'experiment_name: "xau_m1_nonhuman_direction_nb"',
                'model_name: "xau_m1_nonhuman_direction_nb"',
                f'version: "{version}"',
                'description: "training flow integration test"',
                f'dataset_ref: "{dataset_ref}"',
                'target_column: "direction_60m"',
                "feature_families:",
                '  - "time"',
                '  - "session"',
                '  - "quality"',
                '  - "htf_context"',
                '  - "disagreement"',
                '  - "event_shape"',
                '  - "entropy"',
                '  - "multiscale"',
                "exclude_feature_columns: []",
                'model_family: "gaussian_nb_binary@1.0.0"',
                "positive_target_threshold: 0.0",
                "decision_threshold: 0.5",
                'evaluation_policy: "walk_forward_holdout"',
                "n_walk_forward_folds: 3",
                f"min_train_rows: {min_train_rows}",
                "embargo_rows: 240",
                f"min_walk_forward_balanced_accuracy: {min_walk_forward_balanced_accuracy}",
                f"min_test_balanced_accuracy: {min_test_balanced_accuracy}",
            ]
        ),
        encoding="utf-8",
    )


def _prepare_training_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    project_root = tmp_path / "project_training"
    storage_root = project_root / "local_data" / "pipeline_data"
    storage_root.mkdir(parents=True, exist_ok=True)
    _seed_training_project_data(storage_root, days=5)
    _write_project_config(project_root, storage_root)

    dataset_spec = project_root / "config" / "datasets" / "xau_m1_nonhuman_v1.yaml"
    _write_spec(
        dataset_spec,
        dataset_name="xau_m1_nonhuman",
        version="1.0.0",
        selectors=[
            "time/*",
            "session/*",
            "quality/*",
            "htf_context/*",
            "disagreement/*",
            "event_shape/*",
            "entropy/*",
            "multiscale/*",
        ],
        date_from="2026-04-01",
        date_to="2026-04-05",
    )
    experiment_spec = project_root / "config" / "experiments" / "xau_m1_nonhuman_direction_nb_v1.yaml"
    _write_experiment_spec(experiment_spec, dataset_ref="dataset://xau_m1_nonhuman@1.0.0")
    return project_root, dataset_spec, experiment_spec


def test_run_experiment_spec_registers_experiment_and_model_artifacts(tmp_path, monkeypatch) -> None:
    project_root, dataset_spec, experiment_spec = _prepare_training_project(tmp_path)
    monkeypatch.chdir(project_root)

    dataset_result = compile_dataset_spec(dataset_spec)
    assert dataset_result.manifest.status == "published"

    result = run_experiment_spec(experiment_spec)

    assert result.dataset_artifact_id == dataset_result.artifact_id
    assert result.experiment_manifest.status == "accepted"
    assert result.model_manifest.status in {"accepted", "trial"}
    assert result.walk_forward_summary["fold_count"] == 3
    assert result.holdout_metrics["rows_scored"] > 0
    assert result.experiment_aliases[0] == "experiment://xau_m1_nonhuman_direction_nb@1.0.0"
    assert result.model_aliases[0] == "model://xau_m1_nonhuman_direction_nb@1.0.0"
    assert Path(result.experiment_manifest.metadata["summary_path"]).exists()
    assert Path(result.experiment_manifest.metadata["predictions_path"]).exists()
    assert Path(result.model_manifest.metadata["model_payload_path"]).exists()

    experiment_view = inspect_experiment(result.experiment_aliases[0])
    model_view = inspect_model(result.model_aliases[0])
    assert experiment_view.training_run is not None
    assert experiment_view.training_run.status == result.run_status
    assert experiment_view.summary["dataset_artifact_id"] == dataset_result.artifact_id
    assert experiment_view.summary["walk_forward_summary"]["fold_count"] == 3
    assert model_view.payload["model_family"] == "gaussian_nb_binary@1.0.0"
    assert model_view.summary["holdout_metrics"]["rows_scored"] > 0

    catalog = CatalogDB(project_root / "local_data" / "pipeline_data" / "catalog" / "catalog.db")
    try:
        stored_run = catalog.get_training_run(result.run_id)
        assert stored_run is not None
        assert stored_run.status == result.run_status
        assert stored_run.experiment_artifact_id == result.experiment_artifact_id
        assert stored_run.model_artifact_id == result.model_artifact_id

        stored_spec = catalog.get_experiment_spec(result.spec.key)
        assert stored_spec is not None
        assert stored_spec.target_column == "direction_60m"

        experiment_artifact = catalog.get_artifact(result.experiment_artifact_id)
        model_artifact = catalog.get_artifact(result.model_artifact_id)
        assert experiment_artifact is not None
        assert model_artifact is not None

        experiment_inputs = catalog.list_artifact_inputs(result.experiment_artifact_id)
        model_inputs = catalog.list_artifact_inputs(result.model_artifact_id)
        assert any(record.input_kind == "experiment_spec" for record in experiment_inputs)
        assert any(record.input_kind == "artifact" and record.input_ref == dataset_result.artifact_id for record in experiment_inputs)
        assert any(record.input_kind == "experiment_spec" for record in model_inputs)
        assert any(record.input_kind == "artifact" and record.input_ref == result.experiment_artifact_id for record in model_inputs)
    finally:
        catalog.close()


def test_run_experiment_spec_rejects_untrusted_dataset_artifact(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project_training_reject"
    storage_root = project_root / "local_data" / "pipeline_data"
    storage_root.mkdir(parents=True, exist_ok=True)
    _seed_training_project_data(storage_root, days=2)
    _write_project_config(project_root, storage_root)

    rejecting_dataset_spec = project_root / "config" / "datasets" / "xau_m1_nonhuman_reject.yaml"
    _write_spec(
        rejecting_dataset_spec,
        dataset_name="xau_m1_nonhuman",
        version="2.0.0",
        selectors=[
            "time/*",
            "session/*",
            "quality/*",
            "htf_context/*",
        ],
        embargo_rows=1500,
        date_from="2026-04-01",
        date_to="2026-04-02",
    )

    experiment_spec = project_root / "config" / "experiments" / "xau_m1_nonhuman_direction_nb_v2.yaml"
    monkeypatch.chdir(project_root)

    dataset_result = compile_dataset_spec(rejecting_dataset_spec)
    assert dataset_result.manifest.status == "rejected"

    _write_experiment_spec(experiment_spec, dataset_ref=dataset_result.artifact_id, version="2.0.0", min_train_rows=500)

    with pytest.raises(ValueError, match="not trainable"):
        run_experiment_spec(experiment_spec)


def test_train_cli_runs_and_inspects_real_artifacts(tmp_path, monkeypatch) -> None:
    project_root, dataset_spec, experiment_spec = _prepare_training_project(tmp_path)
    monkeypatch.chdir(project_root)

    dataset_result = compile_dataset_spec(dataset_spec)
    assert dataset_result.manifest.status == "published"

    run_result = runner.invoke(app, ["train", "run-experiment", "--spec", str(experiment_spec)])
    assert run_result.exit_code == 0
    assert "run_id: train." in run_result.stdout
    assert "dataset_ref: dataset://xau_m1_nonhuman@1.0.0" in run_result.stdout
    assert "experiment_ref: experiment://xau_m1_nonhuman_direction_nb@1.0.0" in run_result.stdout
    assert "model_ref: model://xau_m1_nonhuman_direction_nb@1.0.0" in run_result.stdout
    assert "walk_forward_summary:" in run_result.stdout
    assert "holdout_metrics:" in run_result.stdout

    experiment_output = runner.invoke(
        app,
        ["train", "inspect-experiment", "--experiment", "experiment://xau_m1_nonhuman_direction_nb@1.0.0"],
    )
    assert experiment_output.exit_code == 0
    assert "artifact_id: experiment." in experiment_output.stdout
    assert "training_status:" in experiment_output.stdout
    assert "walk_forward_balanced_accuracy_mean:" in experiment_output.stdout

    model_output = runner.invoke(
        app,
        ["train", "inspect-model", "--model", "model://xau_m1_nonhuman_direction_nb@1.0.0"],
    )
    assert model_output.exit_code == 0
    assert "artifact_id: model." in model_output.stdout
    assert "model_family: gaussian_nb_binary@1.0.0" in model_output.stdout
    assert "class_priors:" in model_output.stdout
