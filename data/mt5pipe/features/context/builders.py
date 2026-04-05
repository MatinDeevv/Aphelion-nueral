"""Higher-timeframe context builders."""

from __future__ import annotations

import polars as pl


def add_lagged_bar_features(
    base_df: pl.DataFrame,
    higher_tf_df: pl.DataFrame,
    higher_tf_name: str,
    bar_duration_seconds: int = 0,
    base_time_col: str = "time_utc",
    higher_time_col: str = "time_utc",
) -> pl.DataFrame:
    """Join the last fully closed higher-timeframe bar onto the base frame."""
    if higher_tf_df.is_empty():
        return base_df

    prefix = f"{higher_tf_name}_"
    feature_cols = ["open", "high", "low", "close", "tick_count", "spread_mean", "mid_return", "realized_vol"]
    available = [c for c in feature_cols if c in higher_tf_df.columns]
    join_key = f"_{higher_tf_name.lower()}_time"

    if bar_duration_seconds > 0:
        shift = pl.duration(seconds=bar_duration_seconds)
        htf = higher_tf_df.select([
            (pl.col(higher_time_col) + shift).alias(join_key),
            *[pl.col(c).alias(f"{prefix}{c}") for c in available],
        ]).sort(join_key)
    else:
        htf = higher_tf_df.select([
            pl.col(higher_time_col).alias(join_key),
            *[pl.col(c).alias(f"{prefix}{c}") for c in available],
        ]).sort(join_key)

    result = base_df.sort(base_time_col).join_asof(
        htf,
        left_on=base_time_col,
        right_on=join_key,
        strategy="backward",
    )

    drop_cols = [c for c in (join_key, f"{join_key}_right") if c in result.columns]
    if drop_cols:
        result = result.drop(drop_cols)

    return result
