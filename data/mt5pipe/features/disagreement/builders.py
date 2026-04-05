"""Machine-native disagreement features built from dual-source market state."""

from __future__ import annotations

import polars as pl

from mt5pipe.features.internal.family_utils import (
    apply_column_warmups,
    ensure_output_columns,
    finite_or_null,
    has_all_columns,
    has_any_column,
    null_output_frame,
    resolve_time_col,
    safe_ratio_expr,
)
from mt5pipe.features.internal.statistics import rolling_shannon_entropy

OUTPUT_TYPES: dict[str, pl.DataType] = {
    "mid_divergence_proxy_bps": pl.Float64,
    "spread_divergence_proxy_bps": pl.Float64,
    "disagreement_pressure_bps": pl.Float64,
    "disagreement_zscore_60": pl.Float64,
    "conflict_burst_15": pl.Float64,
    "disagreement_burst_15": pl.Float64,
    "staleness_asymmetry_15": pl.Float64,
    "disagreement_entropy_30": pl.Float64,
}


def add_disagreement_features(
    df: pl.DataFrame,
    *,
    time_col: str | None = None,
    zscore_window: int = 60,
    burst_window: int = 15,
    entropy_window: int = 30,
) -> pl.DataFrame:
    """Add PIT-safe disagreement features using only the current and trailing rows."""
    resolved_time_col = resolve_time_col(df, time_col, family="disagreement")
    working = df.sort(resolved_time_col)

    if not _has_required_inputs(working):
        return null_output_frame(working, OUTPUT_TYPES)

    reference_price = _column_or_value(
        working,
        preferred=["close", "mid"],
        default=1.0,
    ).clip(lower_bound=1e-9)
    spread_value = _column_or_value(
        working,
        preferred=["spread_mean", "spread"],
        default=0.0,
    ).clip(lower_bound=0.0)
    spread_bps = (spread_value / reference_price) * 10_000.0

    tick_count = _column_or_value(working, preferred=["tick_count"], default=1.0).clip(lower_bound=1.0)
    conflict_count = _column_or_value(working, preferred=["conflict_count"], default=0.0).clip(lower_bound=0.0)
    conflict_event = (conflict_count > 0).cast(pl.Float64)
    dual_source_ratio = _dual_source_ratio_expr(working, tick_count).clip(
        lower_bound=0.0,
        upper_bound=1.0,
    )
    dual_source_gap = (1.0 - dual_source_ratio).clip(lower_bound=0.0, upper_bound=1.0)
    secondary_present = _column_or_value(working, preferred=["secondary_present_ticks"], default=0.0).clip(lower_bound=0.0)
    dual_source_ticks = _column_or_value(working, preferred=["dual_source_ticks"], default=0.0).clip(lower_bound=0.0)
    disagreement_bps = _column_or_value(working, preferred=["disagreement_bps"], default=None)

    secondary_gap_ratio = safe_ratio_expr(
        (secondary_present - dual_source_ticks).clip(lower_bound=0.0),
        tick_count,
        lower_bound=0.0,
        upper_bound=1.0,
    )
    conflict_ratio = (conflict_count / tick_count).clip(lower_bound=0.0, upper_bound=1.0)

    working = working.with_columns(
        dual_source_gap.alias("_dual_source_gap"),
        conflict_event.alias("_conflict_event"),
        secondary_gap_ratio.alias("_secondary_gap_ratio"),
        conflict_ratio.alias("_conflict_ratio"),
    )
    working = working.with_columns(
        pl.when(disagreement_bps.is_not_null())
        .then(disagreement_bps.abs())
        .otherwise((pl.col("_dual_source_gap") * spread_bps).clip(lower_bound=0.0))
        .alias("mid_divergence_proxy_bps"),
        ((pl.col("_secondary_gap_ratio") + pl.col("_conflict_ratio")) * spread_bps).clip(lower_bound=0.0).alias(
            "spread_divergence_proxy_bps"
        ),
    )
    working = working.with_columns(
        (
            pl.col("mid_divergence_proxy_bps")
            + pl.col("spread_divergence_proxy_bps")
            + (pl.col("_conflict_event") * spread_bps)
        ).alias("disagreement_pressure_bps")
    )
    working = working.with_columns(
        finite_or_null(
            pl.col("disagreement_pressure_bps")
            .rolling_mean(window_size=burst_window, min_samples=burst_window)
        ).alias("disagreement_burst_15"),
        finite_or_null(
            pl.col("_conflict_event")
            .rolling_mean(window_size=burst_window, min_samples=burst_window)
        ).alias("conflict_burst_15"),
        finite_or_null(
            pl.col("_secondary_gap_ratio")
            .rolling_mean(window_size=burst_window, min_samples=burst_window)
        ).alias("staleness_asymmetry_15"),
    )
    working = working.with_columns(
        finite_or_null(
            (
                pl.col("disagreement_pressure_bps")
                - pl.col("disagreement_pressure_bps").rolling_mean(window_size=zscore_window, min_samples=zscore_window)
            )
            / pl.col("disagreement_pressure_bps").rolling_std(window_size=zscore_window, min_samples=zscore_window)
        ).alias("disagreement_zscore_60")
    )

    conflict_series = [int(value) for value in working["_conflict_event"].to_list()]
    working = working.with_columns(
        pl.Series(
            "disagreement_entropy_30",
            rolling_shannon_entropy(conflict_series, entropy_window),
            dtype=pl.Float64,
        )
    )

    working = working.drop(["_dual_source_gap", "_conflict_event", "_secondary_gap_ratio", "_conflict_ratio"])
    working = ensure_output_columns(working, OUTPUT_TYPES)
    return apply_column_warmups(
        working,
        OUTPUT_TYPES,
        warmup_rows_by_column={
            "disagreement_burst_15": burst_window,
            "conflict_burst_15": burst_window,
            "staleness_asymmetry_15": burst_window,
            "disagreement_entropy_30": entropy_window,
            "disagreement_zscore_60": zscore_window,
        },
    )


def _has_required_inputs(df: pl.DataFrame) -> bool:
    return (
        has_any_column(df, ["close", "mid"])
        and has_any_column(df, ["spread_mean", "spread"])
        and has_all_columns(df, ["tick_count"])
        and has_any_column(
            df,
            ["dual_source_ratio", "dual_source_ticks", "secondary_present_ticks", "conflict_count", "disagreement_bps"],
        )
    )


def _column_or_value(df: pl.DataFrame, *, preferred: list[str], default: float | None) -> pl.Expr:
    for column in preferred:
        if column in df.columns:
            return pl.col(column).cast(pl.Float64)
    return pl.lit(default, dtype=pl.Float64)


def _dual_source_ratio_expr(df: pl.DataFrame, tick_count: pl.Expr) -> pl.Expr:
    if "dual_source_ratio" in df.columns:
        return pl.col("dual_source_ratio").cast(pl.Float64)
    if "dual_source_ticks" in df.columns:
        return safe_ratio_expr(pl.col("dual_source_ticks").cast(pl.Float64), tick_count, lower_bound=0.0, upper_bound=1.0)
    return pl.lit(0.0, dtype=pl.Float64)
