"""Focused tests for the Phase 6 machinelearning data layer."""

from __future__ import annotations

import datetime as dt
import shutil
import sys
import uuid
from pathlib import Path

import polars as pl
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

TEST_TMP_ROOT = WORKSPACE_ROOT / "_test_tmp"

from data import (
    AphelionDataModule,
    AphelionDataset,
    ColumnSchema,
    DEFAULT_SCHEMA,
    InferenceLoader,
    RobustFeatureNormalizer,
    WalkForwardResult,
    WalkForwardSplitter,
)
from data.schema import (
    CLASSIFICATION_TARGET_COLUMNS,
    FILLED_FLAG_COLUMN,
    REGRESSION_TARGET_COLUMNS,
    SYMBOL_COLUMN,
    SYMBOL_INDEX_COLUMN,
    TARGET_HORIZONS_MINUTES,
    TIMEFRAME_COLUMN,
    TIME_INDEX_COLUMN,
    make_target_column,
)


@pytest.fixture
def workspace_tmp_path() -> Path:
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEST_TMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_schema_validation_catches_missing_columns() -> None:
    schema = ColumnSchema(
        past_observed=[DEFAULT_SCHEMA.past_observed[0]],
        future_known=[DEFAULT_SCHEMA.future_known[0]],
        static=[DEFAULT_SCHEMA.static[0]],
        targets=[DEFAULT_SCHEMA.targets[0]],
    )
    missing = schema.validate_dataframe_columns(
        [DEFAULT_SCHEMA.past_observed[0], DEFAULT_SCHEMA.targets[0]],
    )
    assert missing == [DEFAULT_SCHEMA.future_known[0], DEFAULT_SCHEMA.static[0]]


def test_normalizer_fit_transform_save_load_roundtrip_is_exact(workspace_tmp_path: Path) -> None:
    frame = _base_frame(rows=8, feature_offset=10.0)
    normalizer = RobustFeatureNormalizer(schema=DEFAULT_SCHEMA)
    normalizer.fit(frame)

    first = normalizer.transform(frame)
    path = workspace_tmp_path / "normalizer.json"
    normalizer.save(path)
    loaded = RobustFeatureNormalizer.load(path)
    second = loaded.transform(frame)

    assert first.to_dict(as_series=False) == second.to_dict(as_series=False)


def test_normalizer_all_null_column(workspace_tmp_path: Path) -> None:
    frame = _base_frame(rows=8, feature_offset=10.0).with_columns(
        pl.lit(None, dtype=pl.Float64).alias(DEFAULT_SCHEMA.past_observed[0]),
    )
    normalizer = RobustFeatureNormalizer(schema=DEFAULT_SCHEMA)

    normalizer.fit(frame)
    transformed = normalizer.transform(frame)

    assert DEFAULT_SCHEMA.past_observed[0] in normalizer.constant_columns
    assert transformed.get_column(DEFAULT_SCHEMA.past_observed[0]).to_list() == [0.0] * frame.height


def test_normalizer_constant_columns_survive_roundtrip(workspace_tmp_path: Path) -> None:
    frame = _base_frame(rows=8, feature_offset=10.0).with_columns(
        pl.lit(None, dtype=pl.Float64).alias(DEFAULT_SCHEMA.past_observed[1]),
    )
    normalizer = RobustFeatureNormalizer(schema=DEFAULT_SCHEMA).fit(frame)

    path = workspace_tmp_path / "normalizer_constant_columns.json"
    normalizer.save(path)
    loaded = RobustFeatureNormalizer.load(path)

    assert loaded.constant_columns == normalizer.constant_columns


def test_normalizer_refuses_transform_before_fit() -> None:
    normalizer = RobustFeatureNormalizer(schema=DEFAULT_SCHEMA)
    with pytest.raises(RuntimeError):
        normalizer.transform(_base_frame(rows=4, feature_offset=5.0))


def test_datamodule_never_uses_val_or_test_statistics(workspace_tmp_path: Path) -> None:
    artifact_dir = workspace_tmp_path / "artifact=demo"
    train_dir = artifact_dir / "split=train"
    val_dir = artifact_dir / "split=val"
    test_dir = artifact_dir / "split=test"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    _base_frame(rows=12, feature_offset=1.0).write_parquet(train_dir / "part-0.parquet")
    _base_frame(rows=12, feature_offset=1_000.0).write_parquet(val_dir / "part-0.parquet")
    _base_frame(rows=12, feature_offset=2_000.0).write_parquet(test_dir / "part-0.parquet")

    module = AphelionDataModule(
        artifact_dir=artifact_dir,
        context_len=4,
        batch_size=2,
        num_workers=0,
        stride_train=1,
        normalizer_save_path=workspace_tmp_path / "saved_normalizer.json",
    )
    module.setup()

    feature_name = DEFAULT_SCHEMA.past_observed[0]
    assert module.normalizer.stats[feature_name]["median"] == pytest.approx(6.5)
    assert module.normalizer.stats[feature_name]["median"] != pytest.approx(1005.5)
    assert (workspace_tmp_path / "saved_normalizer.json").exists()


def test_dataset_shapes_are_correct_for_all_dict_keys(workspace_tmp_path: Path) -> None:
    split_dir = workspace_tmp_path / "split=train"
    split_dir.mkdir(parents=True)
    _base_frame(rows=7, feature_offset=1.0).write_parquet(split_dir / "part-0.parquet")

    dataset = AphelionDataset(artifact_dir=split_dir, context_len=4, stride=1)
    sample = dataset[0]

    assert tuple(sample["past_features"].shape) == (4, DEFAULT_SCHEMA.n_past)
    assert tuple(sample["future_known"].shape) == (4, DEFAULT_SCHEMA.n_future)
    assert tuple(sample["static"].shape) == (DEFAULT_SCHEMA.n_static,)
    assert tuple(sample["mask"].shape) == (4,)
    assert "bool" in str(sample["mask"].dtype).lower()
    assert sample["time_idx"].item() == 3
    assert set(sample["targets"].keys()) == set(DEFAULT_SCHEMA.targets)


def test_mask_marks_filled_rows_as_false(workspace_tmp_path: Path) -> None:
    split_dir = workspace_tmp_path / "split=train"
    split_dir.mkdir(parents=True)
    frame = _base_frame(rows=5, feature_offset=2.0).with_columns(
        pl.Series(name=FILLED_FLAG_COLUMN, values=[False, True, False, False, False], dtype=pl.Boolean),
    )
    frame.write_parquet(split_dir / "part-0.parquet")

    dataset = AphelionDataset(artifact_dir=split_dir, context_len=3, stride=1)
    sample = dataset[0]
    assert sample["mask"].tolist() == [True, False, True]


def test_classification_targets_remap_minus_one_zero_and_one(workspace_tmp_path: Path) -> None:
    direction_column = make_target_column("direction", 5)
    barrier_column = make_target_column("triple_barrier", 5)
    frame = _base_frame(rows=5, feature_offset=1.0).with_columns(
        pl.Series(name=direction_column, values=[1, 1, -1, 0, 1]),
        pl.Series(name=barrier_column, values=[1, 1, -1, 0, 1]),
    )

    split_dir = workspace_tmp_path / "split=train"
    split_dir.mkdir(parents=True)
    frame.write_parquet(split_dir / "part-0.parquet")

    dataset = AphelionDataset(artifact_dir=split_dir, context_len=3, stride=1)
    assert dataset[0]["targets"][direction_column].item() == 0
    assert dataset[1]["targets"][direction_column].item() == 1
    assert dataset[2]["targets"][direction_column].item() == 2
    assert dataset[0]["targets"][barrier_column].item() == 0
    assert dataset[1]["targets"][barrier_column].item() == 1
    assert dataset[2]["targets"][barrier_column].item() == 2


def test_null_direction_target_maps_to_ignore_index(workspace_tmp_path: Path) -> None:
    direction_column = make_target_column("direction", 60)
    frame = _base_frame(rows=4, feature_offset=2.0).with_columns(
        pl.Series(name=direction_column, values=[1, 1, 1, None], dtype=pl.Int64),
    )

    split_dir = workspace_tmp_path / "split=train"
    split_dir.mkdir(parents=True)
    frame.write_parquet(split_dir / "part-0.parquet")

    dataset = AphelionDataset(artifact_dir=split_dir, context_len=4, stride=1)
    assert dataset[0]["targets"][direction_column].item() == -100


def test_targets_come_from_last_bar_only(workspace_tmp_path: Path) -> None:
    regression_column = make_target_column("future_return", 15)
    classification_column = make_target_column("direction", 15)
    frame = _base_frame(rows=5, feature_offset=3.0).with_columns(
        pl.Series(name=regression_column, values=[11.0, 12.0, 13.0, 14.0, 15.0]),
        pl.Series(name=classification_column, values=[-1, -1, 1, 0, 1], dtype=pl.Int64),
    )

    split_dir = workspace_tmp_path / "split=train"
    split_dir.mkdir(parents=True)
    frame.write_parquet(split_dir / "part-0.parquet")

    dataset = AphelionDataset(artifact_dir=split_dir, context_len=3, stride=1)
    first = dataset[0]["targets"]
    second = dataset[1]["targets"]

    assert first[regression_column].item() == pytest.approx(13.0)
    assert second[regression_column].item() == pytest.approx(14.0)
    assert first[classification_column].item() == 2
    assert second[classification_column].item() == 1


def test_walk_forward_splitter_yields_three_expanding_folds() -> None:
    frame = _walkforward_frame(rows=40)
    splitter = WalkForwardSplitter(n_folds=3, val_fraction=0.30, embargo_rows=2)

    folds = list(splitter.split(frame))

    assert len(folds) == 3

    previous_train_rows = 0
    for train_df, val_df in folds:
        assert train_df.height > previous_train_rows
        assert train_df.get_column("time_utc").max() < val_df.get_column("time_utc").min()
        train_end_row = train_df.get_column("row_id").max()
        val_start_row = val_df.get_column("row_id").min()
        assert (val_start_row - train_end_row - 1) >= 2
        previous_train_rows = train_df.height


def test_walkforward_embargo_exact_boundary() -> None:
    frame = _walkforward_frame(rows=40)
    splitter = WalkForwardSplitter(n_folds=3, val_fraction=0.30, embargo_rows=2)

    for train_df, val_df in splitter.split(frame):
        train_end_row = int(train_df.get_column("row_id").max())
        val_start_row = int(val_df.get_column("row_id").min())
        assert val_start_row == train_end_row + splitter.embargo_rows + 1


def test_walkforward_no_data_overlap() -> None:
    frame = _walkforward_frame(rows=40)
    splitter = WalkForwardSplitter(n_folds=3, val_fraction=0.30, embargo_rows=2)

    for train_df, val_df in splitter.split(frame):
        train_times = set(train_df.get_column(TIME_INDEX_COLUMN).to_list())
        val_times = set(val_df.get_column(TIME_INDEX_COLUMN).to_list())
        assert train_times.isdisjoint(val_times)


def test_walk_forward_splitter_raises_when_dataset_too_small() -> None:
    frame = _walkforward_frame(rows=20)
    splitter = WalkForwardSplitter(n_folds=3, val_fraction=0.50, embargo_rows=4)

    with pytest.raises(ValueError, match="dataset too small"):
        list(splitter.split(frame))


def test_walk_forward_result_mean_is_correct() -> None:
    result = WalkForwardResult(
        fold_metrics=[
            {"balanced_accuracy": 0.50, "ic_60m": 0.10},
            {"balanced_accuracy": 0.60, "ic_60m": 0.00},
            {"balanced_accuracy": 0.70, "ic_60m": 0.20},
        ]
    )

    assert result.mean["balanced_accuracy"] == pytest.approx(0.60)
    assert result.mean["ic_60m"] == pytest.approx(0.10)
    assert "fold_0" in result.summary_str()


def test_inference_loader_prepare_batch_shapes_and_no_targets(workspace_tmp_path: Path) -> None:
    frame = _base_frame(rows=12, feature_offset=4.0)
    normalizer = RobustFeatureNormalizer(schema=DEFAULT_SCHEMA).fit(frame)
    normalizer_path = workspace_tmp_path / "inference_normalizer.json"
    normalizer.save(normalizer_path)

    loader = InferenceLoader(normalizer_path=normalizer_path, context_len=4)
    batch = loader.prepare_batch(frame)

    assert tuple(batch["past_features"].shape) == (1, 4, DEFAULT_SCHEMA.n_past)
    assert tuple(batch["future_known"].shape) == (1, 4, DEFAULT_SCHEMA.n_future)
    assert tuple(batch["static"].shape) == (1, DEFAULT_SCHEMA.n_static)
    assert tuple(batch["mask"].shape) == (1, 4)
    assert tuple(batch["time_idx"].shape) == (1,)
    assert "targets" not in batch


def test_inference_loader_unsorted_raises(workspace_tmp_path: Path) -> None:
    frame = _base_frame(rows=12, feature_offset=4.0).sort(TIME_INDEX_COLUMN, descending=True)
    normalizer = RobustFeatureNormalizer(schema=DEFAULT_SCHEMA).fit(frame.sort(TIME_INDEX_COLUMN))
    normalizer_path = workspace_tmp_path / "unsorted_inference_normalizer.json"
    normalizer.save(normalizer_path)

    loader = InferenceLoader(normalizer_path=normalizer_path, context_len=4)

    with pytest.raises(ValueError, match="DataFrame must be sorted by time_utc in ascending order"):
        loader.prepare_batch(frame)


def test_inference_loader_sorted_succeeds(workspace_tmp_path: Path) -> None:
    frame = _base_frame(rows=12, feature_offset=4.0)
    normalizer = RobustFeatureNormalizer(schema=DEFAULT_SCHEMA).fit(frame)
    normalizer_path = workspace_tmp_path / "sorted_inference_normalizer.json"
    normalizer.save(normalizer_path)

    loader = InferenceLoader(normalizer_path=normalizer_path, context_len=4)
    batch = loader.prepare_batch(frame)

    assert tuple(batch["past_features"].shape) == (1, 4, DEFAULT_SCHEMA.n_past)
    assert tuple(batch["future_known"].shape) == (1, 4, DEFAULT_SCHEMA.n_future)


def test_inference_loader_raises_when_context_is_too_short(workspace_tmp_path: Path) -> None:
    frame = _base_frame(rows=3, feature_offset=6.0)
    normalizer = RobustFeatureNormalizer(schema=DEFAULT_SCHEMA).fit(frame)
    normalizer_path = workspace_tmp_path / "short_normalizer.json"
    normalizer.save(normalizer_path)

    loader = InferenceLoader(normalizer_path=normalizer_path, context_len=4)
    with pytest.raises(ValueError, match="at least context_len rows"):
        loader.prepare_batch(frame)


def test_inference_loader_from_checkpoint_dir_loads_saved_stats(workspace_tmp_path: Path) -> None:
    frame = _base_frame(rows=10, feature_offset=8.0)
    fitted = RobustFeatureNormalizer(schema=DEFAULT_SCHEMA).fit(frame)
    checkpoint_dir = workspace_tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    fitted.save(checkpoint_dir / "run_alpha_normalizer.json")

    loader = InferenceLoader.from_checkpoint_dir(checkpoint_dir, run_name="run_alpha", context_len=4)

    assert loader.normalizer.stats == fitted.stats


def _base_frame(rows: int, feature_offset: float) -> pl.DataFrame:
    base_time = dt.datetime(2026, 4, 1, 0, 0, tzinfo=dt.timezone.utc)
    data: dict[str, object] = {
        SYMBOL_COLUMN: ["XAUUSD"] * rows,
        TIMEFRAME_COLUMN: ["M1"] * rows,
        TIME_INDEX_COLUMN: [base_time + dt.timedelta(minutes=index) for index in range(rows)],
        SYMBOL_INDEX_COLUMN: [0] * rows,
    }

    for column_index, column in enumerate(DEFAULT_SCHEMA.past_observed):
        data[column] = [feature_offset + column_index + index for index in range(rows)]

    for column_index, column in enumerate(DEFAULT_SCHEMA.future_known):
        data[column] = [float((index + column_index) % 2) for index in range(rows)]

    for horizon in TARGET_HORIZONS_MINUTES:
        for column in CLASSIFICATION_TARGET_COLUMNS:
            if column.endswith(f"_{horizon}m"):
                data[column] = [
                    -1 if index % 3 == 0 else 0 if index % 3 == 1 else 1
                    for index in range(rows)
                ]
        for column in REGRESSION_TARGET_COLUMNS:
            if column.endswith(f"_{horizon}m"):
                data[column] = [feature_offset + (index * 0.5) for index in range(rows)]

    return pl.DataFrame(data)


def _walkforward_frame(rows: int) -> pl.DataFrame:
    base_time = dt.datetime(2026, 4, 1, 0, 0, tzinfo=dt.timezone.utc)
    return pl.DataFrame(
        {
            "row_id": list(range(rows)),
            "time_utc": [base_time + dt.timedelta(minutes=index) for index in range(rows)],
            "value": [float(index) for index in range(rows)],
        }
    )
