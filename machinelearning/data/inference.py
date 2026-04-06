"""Inference-time loading utilities for Phase 6 deployment batches."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import polars as pl

from .normalizer import RobustFeatureNormalizer
from .schema import (
    DEFAULT_SCHEMA,
    FILLED_FLAG_COLUMN,
    SYMBOL_COLUMN,
    SYMBOL_INDEX_COLUMN,
    TIMEFRAME_COLUMN,
    TIME_INDEX_COLUMN,
    ColumnSchema,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised by local no-torch fallback only
    from .dataset import torch


class InferenceLoader:
    """
    Load a saved normalizer JSON and build the exact model batch contract for inference.

    This keeps training-time feature statistics and deployment-time preprocessing
    aligned so the live model sees the same normalization and schema semantics
    that were used during fitting.
    """

    def __init__(
        self,
        normalizer_path: Path | str,
        schema: ColumnSchema = DEFAULT_SCHEMA,
        context_len: int = 240,
    ) -> None:
        if context_len <= 0:
            raise ValueError("context_len must be positive")

        self.normalizer_path = Path(normalizer_path)
        self.schema = schema
        self.context_len = context_len
        self.normalizer = RobustFeatureNormalizer.load(self.normalizer_path)
        self._validate_schema_alignment()

    @classmethod
    def from_checkpoint_dir(
        cls,
        checkpoint_dir: Path | str,
        run_name: str,
        **kwargs,
    ) -> "InferenceLoader":
        """Load an inference loader from ``checkpoint_dir/{run_name}_normalizer.json``."""

        return cls(Path(checkpoint_dir) / f"{run_name}_normalizer.json", **kwargs)

    def prepare_batch(self, df: pl.DataFrame) -> dict[str, torch.Tensor]:
        """
        Convert raw recent bars into a single-sample model batch for inference.

        The dataframe must contain at least ``context_len`` rows. The latest
        context window is used after sorting by ``time_utc`` and applying the
        same feature-preparation plus normalizer transform that training used.
        """

        if df.height < self.context_len:
            raise ValueError(
                f"Inference requires at least context_len rows ({self.context_len}), got {df.height}"
            )
        if TIME_INDEX_COLUMN not in df.columns:
            raise ValueError(f"Input dataframe is missing required time column '{TIME_INDEX_COLUMN}'")

        prepared = self._prepare_dataframe(df)
        context_df = prepared.tail(self.context_len)
        mask = self._build_mask(df.sort(TIME_INDEX_COLUMN).tail(self.context_len))

        return {
            "past_features": torch.tensor(
                np.expand_dims(context_df.select(self.schema.past_observed).to_numpy(), axis=0),
                dtype=torch.float32,
            ),
            "future_known": torch.tensor(
                np.expand_dims(context_df.select(self.schema.future_known).to_numpy(), axis=0),
                dtype=torch.float32,
            ),
            "static": torch.tensor(
                context_df.select(self.schema.static).tail(1).to_numpy(),
                dtype=torch.float32,
            ),
            "mask": torch.tensor([mask], dtype=torch.bool),
            "time_idx": torch.tensor([prepared.height - 1], dtype=torch.int64),
        }

    def _validate_schema_alignment(self) -> None:
        expected = {
            "past_observed": list(self.schema.past_observed),
            "future_known": list(self.schema.future_known),
            "static": list(self.schema.static),
        }
        loaded = {
            "past_observed": list(self.normalizer.schema.past_observed),
            "future_known": list(self.normalizer.schema.future_known),
            "static": list(self.normalizer.schema.static),
        }
        if expected != loaded:
            raise ValueError(
                "Loaded normalizer schema does not match the requested inference schema"
            )

    def _prepare_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        ordered = df.sort(TIME_INDEX_COLUMN)
        if SYMBOL_COLUMN not in ordered.columns:
            ordered = ordered.with_columns(pl.lit("").alias(SYMBOL_COLUMN))
        if TIMEFRAME_COLUMN not in ordered.columns:
            ordered = ordered.with_columns(pl.lit("").alias(TIMEFRAME_COLUMN))
        if SYMBOL_INDEX_COLUMN not in ordered.columns:
            ordered = ordered.with_columns(pl.lit(0).cast(pl.Int64).alias(SYMBOL_INDEX_COLUMN))

        all_feature_columns = self.schema.past_observed + self.schema.future_known + self.schema.static
        present_columns = [column for column in all_feature_columns if column in ordered.columns]
        missing_columns = [column for column in all_feature_columns if column not in ordered.columns]
        if missing_columns:
            warnings.warn(
                "Missing inference feature columns were zero-filled: " + ", ".join(missing_columns),
                stacklevel=2,
            )
            ordered = ordered.with_columns([pl.lit(0.0).alias(column) for column in missing_columns])

        casted = ordered.with_columns(
            [
                pl.col(column)
                .cast(pl.Float64, strict=False)
                .forward_fill()
                .fill_null(0.0)
                .fill_nan(0.0)
                .alias(column)
                for column in present_columns + missing_columns
            ]
        )
        return self.normalizer.transform(casted)

    @staticmethod
    def _build_mask(df: pl.DataFrame) -> list[bool]:
        if FILLED_FLAG_COLUMN not in df.columns:
            return [True] * df.height

        return (
            ~df.get_column(FILLED_FLAG_COLUMN)
            .cast(pl.Boolean, strict=False)
            .fill_null(False)
            .to_list()
        )


__all__ = ["InferenceLoader"]
