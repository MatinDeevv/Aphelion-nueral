"""Shared dataframe helpers for machine-native feature families."""

from __future__ import annotations

import polars as pl


def resolve_time_col(df: pl.DataFrame, time_col: str | None, *, family: str) -> str:
    """Resolve the canonical time column for a feature family."""
    if time_col is not None:
        return time_col
    if "time_utc" in df.columns:
        return "time_utc"
    if "ts_utc" in df.columns:
        return "ts_utc"
    raise KeyError(f"{family} features require a time column")


def has_all_columns(df: pl.DataFrame, columns: list[str]) -> bool:
    """Return whether all named columns exist on the frame."""
    return all(column in df.columns for column in columns)


def has_any_column(df: pl.DataFrame, columns: list[str]) -> bool:
    """Return whether any named column exists on the frame."""
    return any(column in df.columns for column in columns)


def ensure_output_columns(
    df: pl.DataFrame,
    output_types: dict[str, pl.DataType],
) -> pl.DataFrame:
    """Ensure all output columns exist with stable dtypes."""
    expressions: list[pl.Expr] = []
    for column, dtype in output_types.items():
        if column in df.columns:
            expressions.append(pl.col(column).cast(dtype).alias(column))
        else:
            expressions.append(pl.lit(None, dtype=dtype).alias(column))
    return df.with_columns(expressions)


def null_output_frame(
    df: pl.DataFrame,
    output_types: dict[str, pl.DataType],
) -> pl.DataFrame:
    """Return the input frame with all output columns materialized as nulls."""
    return df.with_columns(
        [pl.lit(None, dtype=dtype).alias(column) for column, dtype in output_types.items()]
    )


def apply_warmup_mask(
    df: pl.DataFrame,
    output_types: dict[str, pl.DataType],
    *,
    warmup_rows: int,
) -> pl.DataFrame:
    """Null out all family outputs until the declared warmup is satisfied."""
    working = ensure_output_columns(df, output_types)
    if warmup_rows <= 1 or working.is_empty():
        return working

    working = working.with_row_index("_feature_row_nr")
    expressions: list[pl.Expr] = []
    for column, dtype in output_types.items():
        expressions.append(
            pl.when(pl.col("_feature_row_nr") < warmup_rows - 1)
            .then(pl.lit(None, dtype=dtype))
            .otherwise(pl.col(column).cast(dtype))
            .alias(column)
        )
    return working.with_columns(expressions).drop("_feature_row_nr")


def apply_column_warmups(
    df: pl.DataFrame,
    output_types: dict[str, pl.DataType],
    *,
    warmup_rows_by_column: dict[str, int],
) -> pl.DataFrame:
    """Null out outputs according to per-column warmup requirements."""
    working = ensure_output_columns(df, output_types)
    if working.is_empty():
        return working

    working = working.with_row_index("_feature_row_nr")
    expressions: list[pl.Expr] = []
    for column, dtype in output_types.items():
        warmup_rows = warmup_rows_by_column.get(column, 0)
        if warmup_rows <= 1:
            expressions.append(pl.col(column).cast(dtype).alias(column))
            continue
        expressions.append(
            pl.when(pl.col("_feature_row_nr") < warmup_rows - 1)
            .then(pl.lit(None, dtype=dtype))
            .otherwise(pl.col(column).cast(dtype))
            .alias(column)
        )
    return working.with_columns(expressions).drop("_feature_row_nr")


def safe_ratio_expr(
    numerator: pl.Expr,
    denominator: pl.Expr,
    *,
    min_denominator: float = 1e-12,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> pl.Expr:
    """Return a null-safe ratio expression with optional clipping."""
    expr = (
        pl.when(
            numerator.is_not_null()
            & denominator.is_not_null()
            & (denominator.abs() > min_denominator)
        )
        .then(numerator / denominator)
        .otherwise(None)
    )
    if lower_bound is not None or upper_bound is not None:
        expr = expr.clip(lower_bound=lower_bound, upper_bound=upper_bound)
    return expr


def finite_or_null(expr: pl.Expr) -> pl.Expr:
    """Map NaN and +/-inf results to nulls."""
    return pl.when(expr.is_nan() | expr.is_infinite()).then(None).otherwise(expr)
