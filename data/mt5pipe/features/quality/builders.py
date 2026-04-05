"""Quality feature builders."""

from __future__ import annotations

import polars as pl


def add_spread_quality_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add spread and source-quality features from canonical market data."""
    features_to_add = []
    if "spread_mean" in df.columns and "close" in df.columns:
        features_to_add.append(
            (pl.col("spread_mean") / pl.col("close").clip(lower_bound=1e-10)).alias("relative_spread")
        )
    if "conflict_count" in df.columns and "tick_count" in df.columns:
        features_to_add.append(
            (pl.col("conflict_count") / pl.col("tick_count").clip(lower_bound=1)).alias("conflict_ratio")
        )
    if "source_count" in df.columns:
        features_to_add.append(pl.col("source_count").alias("broker_diversity"))
    if features_to_add:
        df = df.with_columns(features_to_add)
    return df
