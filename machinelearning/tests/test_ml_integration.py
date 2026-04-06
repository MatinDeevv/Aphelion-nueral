"""End-to-end smoke tests for the Phase 6 machinelearning stack."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT.parent))

from machinelearning.data import AphelionDataModule, DEFAULT_SCHEMA, InferenceLoader, WalkForwardSplitter
from machinelearning.data.normalizer import RobustFeatureNormalizer
from machinelearning.data.schema import (
    CLASSIFICATION_TARGET_COLUMNS,
    FILLED_FLAG_COLUMN,
    REGRESSION_TARGET_COLUMNS,
    SYMBOL_COLUMN,
    SYMBOL_INDEX_COLUMN,
    TARGET_HORIZONS_MINUTES,
    TIMEFRAME_COLUMN,
    TIME_INDEX_COLUMN,
)
from machinelearning.models import AphelionTFT, VSNInterpreter
from machinelearning.training import AphelionLightningModule


def test_full_data_to_model_forward(tmp_path: Path) -> None:
    artifact_dir = _synthetic_parquet_splits(tmp_path)
    dm = AphelionDataModule(
        artifact_dir=artifact_dir,
        context_len=16,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    model = AphelionTFT(
        n_past_features=DEFAULT_SCHEMA.n_past,
        n_future_features=DEFAULT_SCHEMA.n_future,
        n_static_features=DEFAULT_SCHEMA.n_static,
        d_model=16,
        n_heads=4,
        n_lstm_layers=1,
        dropout=0.0,
        context_len=16,
    )
    output = model(batch)

    assert set(output.direction_logits) == {"5m", "15m", "60m", "240m"}
    assert output.encoder_hidden is not None
    assert tuple(output.encoder_hidden.shape) == (4, 16)
    assert output.vsn_weights is not None
    assert tuple(output.vsn_weights["past"].shape) == (4, 16, DEFAULT_SCHEMA.n_past)
    assert output.attn_weights is not None
    assert output.attn_weights["past"] is not None


def test_full_training_step(tmp_path: Path) -> None:
    artifact_dir = _synthetic_parquet_splits(tmp_path)
    dm = AphelionDataModule(
        artifact_dir=artifact_dir,
        context_len=16,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    model = AphelionTFT(
        n_past_features=DEFAULT_SCHEMA.n_past,
        n_future_features=DEFAULT_SCHEMA.n_future,
        n_static_features=DEFAULT_SCHEMA.n_static,
        d_model=16,
        n_heads=4,
        n_lstm_layers=1,
        dropout=0.0,
        context_len=16,
    )
    module = AphelionLightningModule(model=model, total_steps=10)

    loss = module.training_step(batch, batch_idx=0)

    assert loss.ndim == 0
    assert loss.requires_grad is True
    assert torch.isfinite(loss)


def test_walkforward_on_real_dataset_split(tmp_path: Path) -> None:
    artifact_dir = _synthetic_parquet_splits(tmp_path)
    dm = AphelionDataModule(
        artifact_dir=artifact_dir,
        context_len=16,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup()

    assert dm.train_dataset is not None
    train_df = dm.train_dataset.raw_dataframe()
    splitter = WalkForwardSplitter(n_folds=2, embargo_rows=10)
    folds = list(splitter.split(train_df))

    assert len(folds) == 2
    for train_split, val_split in folds:
        assert train_split.get_column(TIME_INDEX_COLUMN).max() < val_split.get_column(TIME_INDEX_COLUMN).min()


def test_vsn_interpreter_on_real_forward(tmp_path: Path) -> None:
    artifact_dir = _synthetic_parquet_splits(tmp_path)
    dm = AphelionDataModule(
        artifact_dir=artifact_dir,
        context_len=16,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    model = AphelionTFT(
        n_past_features=DEFAULT_SCHEMA.n_past,
        n_future_features=DEFAULT_SCHEMA.n_future,
        n_static_features=DEFAULT_SCHEMA.n_static,
        d_model=16,
        n_heads=4,
        n_lstm_layers=1,
        dropout=0.0,
        context_len=16,
    )
    output = model(batch)
    interpreter = VSNInterpreter.from_output(output, DEFAULT_SCHEMA.past_observed)

    assert len(interpreter.top_features(10)) == 10
    assert sum(interpreter.past_importance.values()) == pytest.approx(1.0)


def test_inference_loader_roundtrip(tmp_path: Path) -> None:
    artifact_dir = _synthetic_parquet_splits(tmp_path)
    normalizer_path = tmp_path / "integration_normalizer.json"
    dm = AphelionDataModule(
        artifact_dir=artifact_dir,
        context_len=16,
        batch_size=4,
        num_workers=0,
        normalizer_save_path=normalizer_path,
        pin_memory=False,
    )
    dm.setup()

    assert dm.train_dataset is not None
    if not normalizer_path.exists():
        RobustFeatureNormalizer(schema=DEFAULT_SCHEMA).fit(dm.train_dataset.raw_dataframe()).save(normalizer_path)

    loader = InferenceLoader(normalizer_path, context_len=16)
    raw_df = dm.train_dataset.raw_dataframe()
    batch = loader.prepare_batch(raw_df)

    assert "targets" not in batch
    assert tuple(batch["past_features"].shape) == (1, 16, DEFAULT_SCHEMA.n_past)


def _synthetic_parquet_splits(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "artifact=synthetic"
    for split_name, offset in (("split=train", 0.0), ("split=val", 100.0), ("split=test", 200.0)):
        split_dir = artifact_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        _synthetic_frame(rows=32, feature_offset=offset).write_parquet(split_dir / "part-0.parquet")
    return artifact_dir


def _synthetic_frame(rows: int, feature_offset: float) -> pl.DataFrame:
    base_time = dt.datetime(2026, 4, 1, 0, 0, tzinfo=dt.timezone.utc)
    data: dict[str, object] = {
        SYMBOL_COLUMN: ["XAUUSD"] * rows,
        TIMEFRAME_COLUMN: ["M1"] * rows,
        TIME_INDEX_COLUMN: [base_time + dt.timedelta(minutes=index) for index in range(rows)],
        SYMBOL_INDEX_COLUMN: [0] * rows,
        FILLED_FLAG_COLUMN: [False] * rows,
    }

    for column_index, column in enumerate(DEFAULT_SCHEMA.past_observed):
        data[column] = [feature_offset + column_index + (index * 0.1) for index in range(rows)]

    for column_index, column in enumerate(DEFAULT_SCHEMA.future_known):
        data[column] = [float((index + column_index) % 4) / 3.0 for index in range(rows)]

    for horizon in TARGET_HORIZONS_MINUTES:
        for column in CLASSIFICATION_TARGET_COLUMNS:
            if column.endswith(f"_{horizon}m"):
                data[column] = [(-1, 0, 1)[index % 3] for index in range(rows)]
        for column in REGRESSION_TARGET_COLUMNS:
            if column.endswith(f"_{horizon}m"):
                data[column] = [feature_offset + (index * 0.05) + (horizon / 1000.0) for index in range(rows)]

    return pl.DataFrame(data)