"""Data cleaning — tick and bar level validation, outlier removal, dedup."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import polars as pl

from mt5pipe.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class CleaningReport:
    """Tracks what was removed/fixed during cleaning."""

    input_rows: int = 0
    output_rows: int = 0
    duplicates_removed: int = 0
    invalid_price_removed: int = 0
    spread_outliers_removed: int = 0
    price_spike_removed: int = 0
    stale_ticks_removed: int = 0
    time_order_fixed: int = 0
    nulls_removed: int = 0

    @property
    def total_removed(self) -> int:
        return self.input_rows - self.output_rows

    @property
    def pct_removed(self) -> float:
        return (self.total_removed / self.input_rows * 100) if self.input_rows else 0.0


# ---------------------------------------------------------------------------
# Tick-level cleaning
# ---------------------------------------------------------------------------


def clean_ticks(
    df: pl.DataFrame,
    *,
    max_spread_ratio: float = 0.01,
    max_price_jump_pct: float = 2.0,
    stale_repeat_limit: int = 50,
    bid_col: str = "bid",
    ask_col: str = "ask",
    time_col: str = "time_msc",
) -> tuple[pl.DataFrame, CleaningReport]:
    """Full tick cleaning pipeline.

    1. Remove exact duplicates
    2. Remove rows with null/zero/negative bid or ask
    3. Remove crossed quotes (bid > ask)
    4. Remove spread outliers (spread > max_spread_ratio * mid)
    5. Remove price spikes (mid jumps > max_price_jump_pct from rolling median)
    6. Remove stale ticks (identical bid+ask repeated > N times consecutively)
    7. Enforce monotonic time ordering
    """
    report = CleaningReport(input_rows=len(df))

    if df.is_empty():
        report.output_rows = 0
        return df, report

    before = len(df)

    # 1. Exact dedup
    df = df.unique(subset=[time_col, bid_col, ask_col, "last", "volume"], keep="first")
    report.duplicates_removed = before - len(df)
    before = len(df)

    # 2. Null / zero / negative prices
    df = df.filter(
        pl.col(bid_col).is_not_null()
        & pl.col(ask_col).is_not_null()
        & (pl.col(bid_col) > 0)
        & (pl.col(ask_col) > 0)
    )
    report.invalid_price_removed = before - len(df)
    before = len(df)

    # 3. Crossed quotes (bid > ask)
    df = df.filter(pl.col(bid_col) <= pl.col(ask_col))
    report.invalid_price_removed += before - len(df)
    before = len(df)

    # 4. Spread outliers
    df = df.with_columns(
        ((pl.col(ask_col) - pl.col(bid_col)) / ((pl.col(bid_col) + pl.col(ask_col)) / 2.0)).alias("_spread_ratio")
    )
    df = df.filter(pl.col("_spread_ratio") <= max_spread_ratio)
    report.spread_outliers_removed = before - len(df)
    df = df.drop("_spread_ratio")
    before = len(df)

    # 5. Price spikes — mid jump > threshold from rolling median
    if len(df) > 20:
        df = df.sort(time_col)
        df = df.with_columns(
            ((pl.col(bid_col) + pl.col(ask_col)) / 2.0).alias("_mid"),
        )
        df = df.with_columns(
            pl.col("_mid").rolling_median(window_size=20).alias("_rolling_mid"),
        )
        df = df.with_columns(
            ((pl.col("_mid") - pl.col("_rolling_mid")).abs()
             / pl.col("_rolling_mid").clip(lower_bound=1e-10) * 100.0).alias("_jump_pct"),
        )
        # Keep first 20 rows (no rolling median yet) + rows within threshold
        first_rows = df.head(20)
        rest = df.slice(20)
        rest = rest.filter(
            pl.col("_jump_pct").is_null() | (pl.col("_jump_pct") <= max_price_jump_pct)
        )
        df = pl.concat([first_rows, rest], how="diagonal_relaxed")
        df = df.drop(["_mid", "_rolling_mid", "_jump_pct"])
        report.price_spike_removed = before - len(df)
        before = len(df)

    # 6. Stale/frozen ticks — identical bid+ask repeated many times
    if len(df) > stale_repeat_limit:
        df = df.sort(time_col)
        df = df.with_columns(
            ((pl.col(bid_col) == pl.col(bid_col).shift(1))
             & (pl.col(ask_col) == pl.col(ask_col).shift(1))).alias("_same"),
        )
        # Build run-length group
        df = df.with_columns(
            (~pl.col("_same").fill_null(False)).cum_sum().alias("_run_group"),
        )
        df = df.with_columns(
            pl.len().over("_run_group").alias("_run_len"),
        )
        # Keep first N of each frozen run, drop the excess
        df = df.with_columns(
            pl.col("_same").cum_sum().over("_run_group").alias("_run_idx"),
        )
        df = df.filter(pl.col("_run_len") <= stale_repeat_limit)
        df = df.drop(["_same", "_run_group", "_run_len", "_run_idx"])
        report.stale_ticks_removed = before - len(df)
        before = len(df)

    # 7. Enforce time ordering
    df = df.sort(time_col)
    report.time_order_fixed = 0  # just sorting, not removing

    report.output_rows = len(df)

    log.info(
        "ticks_cleaned",
        input=report.input_rows,
        output=report.output_rows,
        removed=report.total_removed,
        pct_removed=f"{report.pct_removed:.2f}%",
    )

    return df, report


# ---------------------------------------------------------------------------
# Bar-level cleaning
# ---------------------------------------------------------------------------


def validate_bars(df: pl.DataFrame) -> pl.DataFrame:
    """Validate and fix bar integrity. Returns cleaned bars.

    - Ensures high >= max(open, close) and low <= min(open, close)
    - Removes bars with zero or negative prices
    - Removes bars with tick_count == 0
    - Ensures spread_min <= spread_mean <= spread_max
    - Removes inf/nan values
    """
    if df.is_empty():
        return df

    before = len(df)

    # Remove bars where any OHLC is zero or negative
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df = df.filter(pl.col(col) > 0)

    # Fix OHLC consistency: high must be >= open,close; low must be <= open,close
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        df = df.with_columns([
            pl.max_horizontal("open", "high", "close").alias("high"),
            pl.min_horizontal("open", "low", "close").alias("low"),
        ])

    # Remove zero-tick bars
    if "tick_count" in df.columns:
        df = df.filter(pl.col("tick_count") > 0)

    # Fix spread ordering
    if all(c in df.columns for c in ["spread_min", "spread_mean", "spread_max"]):
        df = df.with_columns([
            pl.min_horizontal("spread_min", "spread_mean", "spread_max").alias("spread_min"),
            pl.max_horizontal("spread_min", "spread_mean", "spread_max").alias("spread_max"),
        ])

    # Remove rows with inf or nan in numeric columns
    numeric_cols = [c for c in df.columns if df[c].dtype in (pl.Float64, pl.Float32)]
    for col in numeric_cols:
        df = df.filter(pl.col(col).is_finite())

    removed = before - len(df)
    if removed > 0:
        log.info("bars_validated", removed=removed, remaining=len(df))

    return df


# ---------------------------------------------------------------------------
# Dataset-level cleaning  
# ---------------------------------------------------------------------------


def clean_dataset(
    df: pl.DataFrame,
    *,
    max_null_pct: float = 50.0,
    drop_inf: bool = True,
    drop_constant: bool = True,
    time_col: str = "time_utc",
) -> tuple[pl.DataFrame, dict]:
    """Final dataset-level cleaning before training.

    1. Remove columns with > max_null_pct% nulls
    2. Replace inf with null, then forward-fill
    3. Remove constant columns (zero variance)
    4. Remove fully-null rows
    5. Ensure sorted by time
    """
    stats: dict = {"input_rows": len(df), "input_cols": len(df.columns)}

    if df.is_empty():
        stats["output_rows"] = 0
        stats["output_cols"] = 0
        return df, stats

    n = len(df)

    # 1. Drop high-null columns
    dropped_cols = []
    for col in df.columns:
        null_pct = df[col].null_count() / n * 100
        if null_pct > max_null_pct:
            dropped_cols.append(col)
    if dropped_cols:
        df = df.drop(dropped_cols)
        log.info("dataset_dropped_high_null_cols", cols=dropped_cols)

    # 2. Replace inf with null, then forward-fill numeric columns
    numeric_cols = [c for c in df.columns if df[c].dtype in (pl.Float64, pl.Float32)]
    if drop_inf and numeric_cols:
        for col in numeric_cols:
            df = df.with_columns(
                pl.when(pl.col(col).is_infinite())
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )
        # Forward-fill then backward-fill to handle edges
        df = df.with_columns([
            pl.col(c).forward_fill().backward_fill() for c in numeric_cols
            if c in df.columns
        ])

    # 3. Drop constant columns (zero variance, useless for ML)
    if drop_constant:
        # Keep these diagnostics even when constant so merge/coverage audits
        # can prove whether the canonical dual-source layer is actually active.
        keep_constant_cols = {
            "source_count",
            "conflict_count",
            "dual_source_ticks",
            "secondary_present_ticks",
            "dual_source_ratio",
            "_filled",
        }
        keep_constant_prefixes = (
            "future_return_",
            "direction_",
            "triple_barrier_",
            "mae_",
            "mfe_",
        )
        constant_cols = []
        for col in df.columns:
            if col in (time_col, "symbol", "timeframe"):
                continue
            if col in keep_constant_cols:
                continue
            if col.startswith(keep_constant_prefixes):
                continue
            if df[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8):
                if df[col].n_unique() <= 1:
                    constant_cols.append(col)
        if constant_cols:
            df = df.drop(constant_cols)
            log.info("dataset_dropped_constant_cols", cols=constant_cols)

    # 4. Remove fully-null rows
    before = len(df)
    # A row is "fully null" if all numeric columns are null
    if numeric_cols:
        remaining_numeric = [c for c in numeric_cols if c in df.columns]
        if remaining_numeric:
            null_mask = pl.all_horizontal([pl.col(c).is_null() for c in remaining_numeric])
            df = df.filter(~null_mask)

    # 5. Sort by time
    if time_col in df.columns:
        df = df.sort(time_col)

    stats["output_rows"] = len(df)
    stats["output_cols"] = len(df.columns)
    stats["dropped_cols"] = dropped_cols
    stats["rows_removed"] = stats["input_rows"] - stats["output_rows"]

    log.info(
        "dataset_cleaned",
        rows_in=stats["input_rows"],
        rows_out=stats["output_rows"],
        cols_in=stats["input_cols"],
        cols_out=stats["output_cols"],
    )

    return df, stats
