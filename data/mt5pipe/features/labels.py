"""Label generation for training datasets."""

from __future__ import annotations

import polars as pl

from mt5pipe.utils.logging import get_logger

log = get_logger(__name__)


def add_future_returns(
    df: pl.DataFrame,
    horizons_minutes: list[int],
    close_col: str = "close",
) -> pl.DataFrame:
    """Add future return columns: return over N minutes forward.
    
    Assumes df is sorted by time_utc with M1 frequency.
    Each row gets the return from its close to the close N bars ahead.
    """
    for h in horizons_minutes:
        col_name = f"future_return_{h}m"
        df = df.with_columns(
            (pl.col(close_col).shift(-h) / pl.col(close_col) - 1.0).alias(col_name)
        )
    return df


def add_direction_labels(
    df: pl.DataFrame,
    horizons_minutes: list[int],
    threshold: float = 0.0,
) -> pl.DataFrame:
    """Add direction classification: 1=up, 0=flat, -1=down.
    
    threshold is in absolute return units (e.g. 0.0001 for 1 bps).
    """
    for h in horizons_minutes:
        ret_col = f"future_return_{h}m"
        dir_col = f"direction_{h}m"
        if ret_col in df.columns:
            df = df.with_columns(
                pl.when(pl.col(ret_col).is_null())
                .then(None)
                .when(pl.col(ret_col) > threshold)
                .then(1)
                .when(pl.col(ret_col) < -threshold)
                .then(-1)
                .otherwise(0)
                .alias(dir_col)
            )
    return df


def add_triple_barrier_labels(
    df: pl.DataFrame,
    horizons_minutes: list[int],
    tp_bps: float = 50.0,
    sl_bps: float = 50.0,
    vol_scale_window: int = 0,
    vol_multiplier: float = 2.0,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pl.DataFrame:
    """Add triple-barrier style labels.
    
    For each bar, look forward up to horizon bars:
    - If price hits TP first -> 1
    - If price hits SL first -> -1
    - If time expires -> 0
    
    When ``vol_scale_window`` > 0 the barriers are scaled per-row:
        barrier = vol_multiplier × rolling_std × √(horizon_minutes)
    This gives tighter barriers for short horizons and wider for long, matching
    the actual volatility regime.  The fixed ``tp_bps``/``sl_bps`` are used as
    a floor so bars with near-zero vol still get a meaningful barrier.

    Also adds MAE (max adverse excursion) and MFE (max favorable excursion).
    """
    tp_frac = tp_bps / 10_000.0
    sl_frac = sl_bps / 10_000.0

    # Pre-compute rolling volatility if requested
    rolling_vol: list[float] | None = None
    if vol_scale_window > 0 and close_col in df.columns:
        ret_series = (df[close_col] / df[close_col].shift(1) - 1.0)
        # rolling std — polars rolling_std wants a window width
        vol_series = ret_series.rolling_std(window_size=vol_scale_window, min_samples=max(vol_scale_window // 2, 2))
        rolling_vol = vol_series.to_list()

    for h in horizons_minutes:
        tb_col = f"triple_barrier_{h}m"
        mae_col = f"mae_{h}m"
        mfe_col = f"mfe_{h}m"

        tb_vals: list[int | None] = []
        mae_vals: list[float | None] = []
        mfe_vals: list[float | None] = []

        closes = df[close_col].to_list()
        highs = df[high_col].to_list()
        lows = df[low_col].to_list()
        n = len(df)

        import math
        sqrt_h = math.sqrt(h)

        for i in range(n):
            entry = closes[i]
            if entry is None or entry <= 0:
                tb_vals.append(None)
                mae_vals.append(None)
                mfe_vals.append(None)
                continue

            if i + h >= n:
                tb_vals.append(None)
                mae_vals.append(None)
                mfe_vals.append(None)
                continue

            # Compute barrier fractions
            if rolling_vol is not None and rolling_vol[i] is not None and rolling_vol[i] > 0:
                vol_barrier = vol_multiplier * rolling_vol[i] * sqrt_h
                cur_tp = max(vol_barrier, tp_frac)
                cur_sl = max(vol_barrier, sl_frac)
            else:
                cur_tp = tp_frac
                cur_sl = sl_frac

            tp_level = entry * (1.0 + cur_tp)
            sl_level = entry * (1.0 - cur_sl)

            max_high = entry
            min_low = entry
            label = 0  # time expiry

            end_idx = min(i + h, n - 1)
            for j in range(i + 1, end_idx + 1):
                max_high = max(max_high, highs[j])
                min_low = min(min_low, lows[j])

                if highs[j] >= tp_level:
                    label = 1
                    break
                if lows[j] <= sl_level:
                    label = -1
                    break

            mfe = (max_high - entry) / entry if entry > 0 else 0.0
            mae = (entry - min_low) / entry if entry > 0 else 0.0

            tb_vals.append(label)
            mae_vals.append(round(mae, 8))
            mfe_vals.append(round(mfe, 8))

        df = df.with_columns([
            pl.Series(tb_col, tb_vals, dtype=pl.Int8),
            pl.Series(mae_col, mae_vals, dtype=pl.Float64),
            pl.Series(mfe_col, mfe_vals, dtype=pl.Float64),
        ])

    return df
