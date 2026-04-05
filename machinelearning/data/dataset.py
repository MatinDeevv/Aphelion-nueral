"""Parquet-backed dataset for the Phase 6 machine learning stack."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import polars as pl

from .normalizer import RobustFeatureNormalizer
from .schema import (
    CLASSIFICATION_TARGET_COLUMNS,
    DEFAULT_SCHEMA,
    FILLED_FLAG_COLUMN,
    REGRESSION_TARGET_COLUMNS,
    SYMBOL_COLUMN,
    SYMBOL_INDEX_COLUMN,
    TIMEFRAME_COLUMN,
    TIME_INDEX_COLUMN,
    ColumnSchema,
    is_classification_target,
)

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ModuleNotFoundError:
    class _CompatTensor:
        """Small tensor wrapper used only when torch is unavailable locally."""

        def __init__(self, array: Any) -> None:
            self._array = np.asarray(array)

        @property
        def shape(self) -> tuple[int, ...]:
            """Return the tensor shape."""

            return self._array.shape

        @property
        def dtype(self) -> np.dtype[Any]:
            """Return the tensor dtype."""

            return self._array.dtype

        def item(self) -> Any:
            """Return the scalar value for scalar tensors."""

            return self._array.item()

        def numpy(self) -> np.ndarray[Any, Any]:
            """Expose the underlying NumPy array."""

            return self._array

        def tolist(self) -> list[Any]:
            """Return the tensor as nested Python lists."""

            return self._array.tolist()

        def __array__(self, dtype: Any = None) -> np.ndarray[Any, Any]:
            return self._array.astype(dtype) if dtype is not None else self._array

        def __iter__(self):  # type: ignore[override]
            for item in self._array:
                yield _CompatTensor(item) if isinstance(item, np.ndarray) else item

        def __getitem__(self, key: Any) -> Any:
            value = self._array[key]
            return _CompatTensor(value) if isinstance(value, np.ndarray) else value

    class _CompatTorch:
        """Tiny subset of the torch API required by the local unit tests."""

        Tensor = _CompatTensor
        float32 = np.float32
        int64 = np.int64
        bool = np.bool_

        @staticmethod
        def tensor(data: Any, dtype: Any = None) -> _CompatTensor:
            return _CompatTensor(np.asarray(data, dtype=dtype))

        @staticmethod
        def stack(tensors: list[Any], dim: int = 0) -> _CompatTensor:
            arrays = [np.asarray(tensor) for tensor in tensors]
            return _CompatTensor(np.stack(arrays, axis=dim))

    class TorchDataset:
        """Fallback dataset base used only when torch is absent."""

        pass

    torch = _CompatTorch()


def _is_missing_number(value: Any) -> bool:
    """Return whether a scalar numeric target should be treated as missing."""

    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


def _encode_categorical_target(value: Any) -> int:
    """Map {-1, 0, 1} labels to {0, 1, 2} with -100 for missing values."""

    if value is None:
        return -100

    mapping = {-1: 0, 0: 1, 1: 2}
    try:
        return mapping[int(value)]
    except (KeyError, TypeError, ValueError):
        return -100


class AphelionDataset(TorchDataset):
    """Windowed parquet dataset that serves Phase 6 model inputs and targets."""

    def __init__(
        self,
        artifact_dir: str | Path,
        context_len: int = 240,
        schema: ColumnSchema = DEFAULT_SCHEMA,
        normalizer: RobustFeatureNormalizer | None = None,
        stride: int = 1,
    ) -> None:
        if context_len <= 0:
            raise ValueError("context_len must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")

        self.artifact_dir = Path(artifact_dir)
        self.context_len = context_len
        self.schema = schema
        self.normalizer = normalizer
        self.stride = stride

        self._dataframe = self._load_dataframe()
        self._past_matrix = self._dataframe.select(self.schema.past_observed).to_numpy()
        self._future_matrix = self._dataframe.select(self.schema.future_known).to_numpy()
        self._static_matrix = self._dataframe.select(self.schema.static).to_numpy()
        self._classification_targets = {
            column: self._dataframe.get_column(column).to_list()
            for column in CLASSIFICATION_TARGET_COLUMNS
            if column in self._dataframe.columns
        }
        self._regression_targets = {
            column: self._dataframe.get_column(column).cast(pl.Float64, strict=False).to_list()
            for column in REGRESSION_TARGET_COLUMNS
            if column in self._dataframe.columns
        }
        self._window_end_indices = list(
            range(self.context_len - 1, self._dataframe.height, self.stride),
        )

    def __len__(self) -> int:
        """Return the number of available sliding windows."""

        return len(self._window_end_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one sliding-window training sample."""

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        window_end = self._window_end_indices[index]
        window_start = window_end - self.context_len + 1

        targets = {
            column: self._build_target_tensor(column, window_end)
            for column in self.schema.targets
        }

        return {
            "past_features": torch.tensor(
                self._past_matrix[window_start : window_end + 1],
                dtype=torch.float32,
            ),
            "future_known": torch.tensor(
                self._future_matrix[window_start : window_end + 1],
                dtype=torch.float32,
            ),
            "static": torch.tensor(
                self._static_matrix[window_end],
                dtype=torch.float32,
            ),
            "mask": torch.tensor(
                self._mask_vector[window_start : window_end + 1],
                dtype=torch.bool,
            ),
            "time_idx": torch.tensor(window_end, dtype=torch.int64),
            "targets": targets,
        }

    def raw_dataframe(self) -> pl.DataFrame:
        """Return the fully prepared dataframe that backs this dataset."""

        return self._dataframe.clone()

    def _build_target_tensor(self, column: str, row_index: int) -> Any:
        """Build the last-step target tensor for a single target column."""

        if is_classification_target(column):
            value = self._classification_targets[column][row_index]
            encoded = _encode_categorical_target(value)
            return torch.tensor(encoded, dtype=torch.int64)

        value = self._regression_targets[column][row_index]
        encoded = float("nan") if _is_missing_number(value) else float(value)
        return torch.tensor(encoded, dtype=torch.float32)

    def _load_dataframe(self) -> pl.DataFrame:
        """Read, validate, and preprocess parquet files for one data split."""

        parquet_files = sorted(self.artifact_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {self.artifact_dir}.")

        frames = [pl.read_parquet(path) for path in parquet_files]
        dataframe = pl.concat(frames, how="diagonal_relaxed", rechunk=True)
        if FILLED_FLAG_COLUMN in dataframe.columns:
            self._mask_vector = (
                ~dataframe.get_column(FILLED_FLAG_COLUMN)
                .cast(pl.Boolean, strict=False)
                .fill_null(False)
                .to_numpy()
                .astype(bool)
            )
            dataframe = dataframe.drop(FILLED_FLAG_COLUMN)
        else:
            self._mask_vector = np.ones(dataframe.height, dtype=bool)

        if TIME_INDEX_COLUMN not in dataframe.columns:
            warnings.warn(
                f"Required time index column '{TIME_INDEX_COLUMN}' is missing; preserving file order instead.",
                stacklevel=2,
            )
            dataframe = dataframe.with_row_index(name=TIME_INDEX_COLUMN)

        if SYMBOL_COLUMN not in dataframe.columns:
            dataframe = dataframe.with_columns(pl.lit("").alias(SYMBOL_COLUMN))
        if TIMEFRAME_COLUMN not in dataframe.columns:
            dataframe = dataframe.with_columns(pl.lit("").alias(TIMEFRAME_COLUMN))

        if SYMBOL_INDEX_COLUMN not in dataframe.columns:
            dataframe = dataframe.with_columns(pl.lit(0).cast(pl.Int64).alias(SYMBOL_INDEX_COLUMN))

        dataframe = dataframe.sort(TIME_INDEX_COLUMN)
        dataframe = self._warn_and_add_missing_targets(dataframe)
        dataframe = self._prepare_feature_columns(dataframe)
        if self.normalizer is not None:
            dataframe = self.normalizer.transform(dataframe)
        else:
            dataframe = self._add_missing_feature_columns(dataframe)

        return dataframe

    def _prepare_feature_columns(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Cast and forward-fill feature columns without touching targets."""

        feature_columns = [
            column
            for column in self.schema.past_observed + self.schema.future_known + self.schema.static
            if column in dataframe.columns
        ]
        missing_feature_columns = [
            column
            for column in self.schema.past_observed + self.schema.future_known + self.schema.static
            if column not in dataframe.columns
        ]
        if missing_feature_columns:
            warnings.warn(
                "Missing feature columns were zero-filled: "
                + ", ".join(missing_feature_columns),
                stacklevel=2,
            )

        feature_expressions = [
            pl.col(column)
            .cast(pl.Float64, strict=False)
            .forward_fill()
            .fill_null(0.0)
            .fill_nan(0.0)
            .alias(column)
            for column in feature_columns
        ]
        target_expressions = [
            pl.col(column).cast(pl.Int64, strict=False).alias(column)
            for column in CLASSIFICATION_TARGET_COLUMNS
            if column in dataframe.columns
        ] + [
            pl.col(column).cast(pl.Float64, strict=False).alias(column)
            for column in REGRESSION_TARGET_COLUMNS
            if column in dataframe.columns
        ]

        return dataframe.with_columns(feature_expressions + target_expressions)

    def _warn_and_add_missing_targets(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Warn on missing target columns and add typed-null placeholders."""

        missing_targets = [column for column in self.schema.targets if column not in dataframe.columns]
        if not missing_targets:
            return dataframe

        warnings.warn(
            "Missing target columns were added as nulls: " + ", ".join(missing_targets),
            stacklevel=2,
        )
        expressions = []
        for column in missing_targets:
            dtype = pl.Int64 if is_classification_target(column) else pl.Float64
            expressions.append(pl.lit(None, dtype=dtype).alias(column))
        return dataframe.with_columns(expressions)

    def _add_missing_feature_columns(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Add zero-filled feature columns when no normalizer is attached."""

        missing_features = [
            column
            for column in self.schema.past_observed + self.schema.future_known + self.schema.static
            if column not in dataframe.columns
        ]
        if not missing_features:
            return dataframe
        return dataframe.with_columns([pl.lit(0.0).alias(column) for column in missing_features])


__all__ = ["AphelionDataset"]
