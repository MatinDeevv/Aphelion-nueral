"""Feature extraction helpers for the Phase 7 regime-detection stack."""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
from torch import Tensor


class RegimeFeatureExtractor:
    """Extract the fixed regime-relevant feature subset from a model feature dataframe."""

    REGIME_FEATURE_COLS: list[str] = [
        "realized_vol",
        "return_sign_shannon_entropy_30",
        "return_permutation_entropy_30",
        "burstiness_20",
        "silence_ratio_20",
        "relative_spread",
        "conflict_ratio",
        "trend_alignment_5_15_60",
        "volatility_ratio_5_60",
        "disagreement_pressure_bps",
        "tick_rate_hz",
        "direction_switch_rate_20",
        "path_efficiency_20",
    ]

    def extract(self, df: pl.DataFrame) -> np.ndarray:
        """Extract regime features into a float32 array with missing columns zero-filled."""

        row_count = df.height
        if row_count == 0:
            return np.zeros((0, len(self.REGIME_FEATURE_COLS)), dtype=np.float32)

        columns: list[np.ndarray] = []
        for column_name in self.REGIME_FEATURE_COLS:
            if column_name in df.columns:
                series = (
                    df.get_column(column_name)
                    .cast(pl.Float32, strict=False)
                    .fill_nan(None)
                    .fill_null(0.0)
                )
                values = series.to_numpy().astype(np.float32, copy=False)
            else:
                values = np.zeros(row_count, dtype=np.float32)
            columns.append(values)

        return np.stack(columns, axis=1).astype(np.float32, copy=False)

    def extract_tensor(self, df: pl.DataFrame) -> Tensor:
        """Extract regime features into a float32 tensor."""

        return torch.from_numpy(self.extract(df).copy())


__all__ = ["RegimeFeatureExtractor"]
