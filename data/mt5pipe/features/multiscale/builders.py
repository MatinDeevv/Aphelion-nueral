"""Machine-native multiscale consistency features."""

from __future__ import annotations

import polars as pl

from mt5pipe.features.internal.family_utils import (
    apply_warmup_mask,
    ensure_output_columns,
    has_all_columns,
    null_output_frame,
    resolve_time_col,
    safe_ratio_expr,
)

OUTPUT_TYPES: dict[str, pl.DataType] = {
    "trend_alignment_5_15_60": pl.Float64,
    "return_energy_ratio_5_60": pl.Float64,
    "volatility_ratio_5_60": pl.Float64,
    "range_expansion_ratio_15_60": pl.Float64,
    "tick_intensity_ratio_5_60": pl.Float64,
}


def add_multiscale_features(
    df: pl.DataFrame,
    *,
    time_col: str | None = None,
    short_window: int = 5,
    medium_window: int = 15,
    long_window: int = 60,
) -> pl.DataFrame:
    """Add PIT-safe multiscale coherence features from trailing M1 information."""
    resolved_time_col = resolve_time_col(df, time_col, family="multiscale")
    working = df.sort(resolved_time_col)

    if not has_all_columns(working, ["mid_return", "high", "low", "tick_count"]):
        return null_output_frame(working, OUTPUT_TYPES)

    mid_return = pl.col("mid_return").cast(pl.Float64)
    tick_count = pl.col("tick_count").cast(pl.Float64).clip(lower_bound=0.0)
    price_range = (pl.col("high").cast(pl.Float64) - pl.col("low").cast(pl.Float64)).clip(lower_bound=0.0)

    working = working.with_columns(
        mid_return.rolling_sum(window_size=short_window, min_samples=short_window).alias("_net_5"),
        mid_return.rolling_sum(window_size=medium_window, min_samples=medium_window).alias("_net_15"),
        mid_return.rolling_sum(window_size=long_window, min_samples=long_window).alias("_net_60"),
        mid_return.abs().rolling_sum(window_size=short_window, min_samples=short_window).alias("_abs_5"),
        mid_return.abs().rolling_sum(window_size=long_window, min_samples=long_window).alias("_abs_60"),
        mid_return.rolling_std(window_size=short_window, min_samples=short_window).alias("_vol_5"),
        mid_return.rolling_std(window_size=long_window, min_samples=long_window).alias("_vol_60"),
        price_range.rolling_sum(window_size=medium_window, min_samples=medium_window).alias("_range_15"),
        price_range.rolling_sum(window_size=long_window, min_samples=long_window).alias("_range_60"),
        tick_count.rolling_mean(window_size=short_window, min_samples=short_window).alias("_ticks_5"),
        tick_count.rolling_mean(window_size=long_window, min_samples=long_window).alias("_ticks_60"),
    )
    working = working.with_columns(
        (
            (
                _sign_expr("_net_5") * _sign_expr("_net_15")
                + _sign_expr("_net_5") * _sign_expr("_net_60")
                + _sign_expr("_net_15") * _sign_expr("_net_60")
            )
            / 3.0
        ).alias("trend_alignment_5_15_60"),
        safe_ratio_expr(pl.col("_abs_5"), pl.col("_abs_60"), lower_bound=0.0).alias("return_energy_ratio_5_60"),
        safe_ratio_expr(pl.col("_vol_5"), pl.col("_vol_60"), lower_bound=0.0).alias("volatility_ratio_5_60"),
        safe_ratio_expr(
            pl.col("_range_15"),
            pl.col("_range_60"),
            lower_bound=0.0,
            upper_bound=1.0,
        ).alias("range_expansion_ratio_15_60"),
        safe_ratio_expr(pl.col("_ticks_5"), pl.col("_ticks_60"), lower_bound=0.0).alias("tick_intensity_ratio_5_60"),
    )

    working = working.drop(
        [
            "_net_5",
            "_net_15",
            "_net_60",
            "_abs_5",
            "_abs_60",
            "_vol_5",
            "_vol_60",
            "_range_15",
            "_range_60",
            "_ticks_5",
            "_ticks_60",
        ]
    )
    working = ensure_output_columns(working, OUTPUT_TYPES)
    return apply_warmup_mask(working, OUTPUT_TYPES, warmup_rows=long_window)


def _sign_expr(column: str) -> pl.Expr:
    return (
        pl.when(pl.col(column) > 0.0)
        .then(1.0)
        .when(pl.col(column) < 0.0)
        .then(-1.0)
        .otherwise(0.0)
    )
