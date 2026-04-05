"""Local bar builder — builds all timeframes from canonical ticks."""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

import polars as pl

from mt5pipe.mt5.constants import TIMEFRAME_SECONDS
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths
from mt5pipe.utils.logging import get_logger

log = get_logger(__name__)


def timeframe_to_seconds(tf: str) -> int:
    """Get the number of seconds for a timeframe. MN1 uses 30 days as approximation."""
    if tf == "MN1":
        return 30 * 86400  # Approximate, actual handled in _floor_to_month
    s = TIMEFRAME_SECONDS.get(tf)
    if s is None:
        raise ValueError(f"Unknown timeframe: {tf}")
    return s


def floor_timestamp_to_bar(ts: dt.datetime, tf: str) -> dt.datetime:
    """Floor a UTC timestamp to the start of its bar period."""
    if tf == "MN1":
        return ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if tf == "W1":
        # ISO week: Monday = 0
        weekday = ts.weekday()
        monday = ts - dt.timedelta(days=weekday)
        return monday.replace(hour=0, minute=0, second=0, microsecond=0)

    secs = timeframe_to_seconds(tf)
    epoch = int(ts.timestamp())
    floored = epoch - (epoch % secs)
    return dt.datetime.fromtimestamp(floored, tz=dt.timezone.utc)


def build_bars_from_ticks(
    ticks: pl.DataFrame,
    timeframe: str,
    symbol: str,
) -> pl.DataFrame:
    """Build OHLC+ bars from canonical tick data for a given timeframe.
    
    Input ticks expected to have: ts_msc, bid, ask, last, volume,
    conflict_flag, source_primary, quality_score.
    """
    if ticks.is_empty():
        return pl.DataFrame()

    tf_seconds = timeframe_to_seconds(timeframe)

    # Add bar_start column by flooring ts_msc
    if timeframe == "MN1":
        ticks = ticks.with_columns(
            pl.col("ts_utc").dt.truncate("1mo").alias("bar_start")
        )
    elif timeframe == "W1":
        ticks = ticks.with_columns(
            pl.col("ts_utc").dt.truncate("1w").alias("bar_start")
        )
    else:
        ticks = ticks.with_columns(
            (pl.col("ts_msc") // (tf_seconds * 1000) * (tf_seconds * 1000)).alias("bar_start_msc")
        )
        ticks = ticks.with_columns(
            (pl.col("bar_start_msc") * 1000).cast(pl.Datetime("us", time_zone="UTC")).alias("bar_start")
        )
        # Fix: bar_start_msc is in ms, need to convert to datetime
        ticks = ticks.drop("bar_start")
        ticks = ticks.with_columns(
            pl.from_epoch(pl.col("bar_start_msc"), time_unit="ms").dt.replace_time_zone("UTC").alias("bar_start")
        )
        ticks = ticks.drop("bar_start_msc")

    # Compute mid price for each tick
    ticks = ticks.with_columns(
        ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"),
        (pl.col("ask") - pl.col("bid")).alias("spread"),
    )

    # Group by bar_start and compute OHLC + extras
    agg_exprs = [
        # OHLC on mid
        pl.col("mid").first().alias("open"),
        pl.col("mid").max().alias("high"),
        pl.col("mid").min().alias("low"),
        pl.col("mid").last().alias("close"),

        # Tick count
        pl.len().alias("tick_count"),

        # Bid/ask open/close
        pl.col("bid").first().alias("bid_open"),
        pl.col("ask").first().alias("ask_open"),
        pl.col("bid").last().alias("bid_close"),
        pl.col("ask").last().alias("ask_close"),

        # Spread stats
        pl.col("spread").mean().alias("spread_mean"),
        pl.col("spread").max().alias("spread_max"),
        pl.col("spread").min().alias("spread_min"),

        # Volume
        pl.col("volume").sum().alias("volume_sum"),

        # Source diversity
        pl.col("source_primary").n_unique().alias("source_count"),

        # Conflict count
        pl.col("conflict_flag").sum().alias("conflict_count"),
    ]

    # Merge audit columns (only present in canonical ticks)
    if "merge_mode" in ticks.columns:
        agg_exprs.append(
            (pl.col("merge_mode") == "best").sum().alias("dual_source_ticks")
        )
    if "source_secondary" in ticks.columns:
        agg_exprs.append(
            (pl.col("source_secondary") != "").sum().alias("secondary_present_ticks")
        )

    bars = ticks.group_by("bar_start").agg(agg_exprs).sort("bar_start")

    # Compute mid return and realized vol
    bars = bars.with_columns([
        (pl.col("close") / pl.col("open") - 1.0).alias("mid_return"),
    ])

    # Realized vol: sqrt of sum of squared log returns within bar
    # We compute this from tick-level data
    rv_per_bar = _compute_realized_vol(ticks, "bar_start")
    if not rv_per_bar.is_empty():
        bars = bars.join(rv_per_bar, on="bar_start", how="left").with_columns(
            pl.col("realized_vol").fill_null(0.0)
        )
    else:
        bars = bars.with_columns(pl.lit(0.0).alias("realized_vol"))

    # Add metadata columns
    bars = bars.with_columns([
        pl.lit(symbol).alias("symbol"),
        pl.lit(timeframe).alias("timeframe"),
    ])

    # Rename bar_start to time_utc
    bars = bars.rename({"bar_start": "time_utc"})

    # Compute dual-source ratio (fraction of ticks from both brokers)
    if "dual_source_ticks" in bars.columns:
        bars = bars.with_columns(
            (pl.col("dual_source_ticks") / pl.col("tick_count").clip(lower_bound=1))
            .alias("dual_source_ratio"),
        )

    # Reorder columns
    col_order = [
        "symbol", "timeframe", "time_utc",
        "open", "high", "low", "close", "tick_count",
        "bid_open", "ask_open", "bid_close", "ask_close",
        "spread_mean", "spread_max", "spread_min",
        "mid_return", "realized_vol", "volume_sum",
        "source_count", "conflict_count",
        "dual_source_ticks", "secondary_present_ticks", "dual_source_ratio",
    ]
    available = [c for c in col_order if c in bars.columns]
    bars = bars.select(available)

    # Validate bar integrity (fix OHLC consistency, remove invalid rows)
    from mt5pipe.quality.cleaning import validate_bars
    bars = validate_bars(bars)

    return bars


def _compute_realized_vol(ticks: pl.DataFrame, group_col: str) -> pl.DataFrame:
    """Compute realized volatility per bar from tick mid-price log returns."""
    if ticks.is_empty() or "mid" not in ticks.columns:
        return pl.DataFrame()

    # Sort within each bar
    ticks_sorted = ticks.sort([group_col, "ts_msc"])

    # Log returns within each bar group
    ticks_sorted = ticks_sorted.with_columns(
        pl.col("mid").log().diff().over(group_col).alias("log_ret")
    )

    rv = ticks_sorted.group_by(group_col).agg(
        (pl.col("log_ret").pow(2).sum().sqrt()).alias("realized_vol")
    )

    return rv


def build_bars_for_date(
    symbol: str,
    timeframe: str,
    date: dt.date,
    paths: StoragePaths,
    store: ParquetStore,
) -> int:
    """Build bars for a single date from canonical ticks. Returns row count."""
    tick_dir = paths.canonical_ticks_dir(symbol, date)
    ticks = store.read_dir(tick_dir)

    if ticks.is_empty():
        return 0

    bars = build_bars_from_ticks(ticks, timeframe, symbol)
    if bars.is_empty():
        return 0

    path = paths.built_bars_file(symbol, timeframe, date)
    store.write(bars, path)
    return len(bars)


def build_bars_date_range(
    symbol: str,
    timeframe: str,
    start_date: dt.date,
    end_date: dt.date,
    paths: StoragePaths,
    store: ParquetStore,
) -> int:
    """Build bars for a date range. Returns total rows."""
    total = 0
    current = start_date

    # For timeframes >= D1, load all ticks at once to avoid partial bars
    tf_secs = timeframe_to_seconds(timeframe)
    if tf_secs >= 86400:
        # Load all ticks for the range
        all_ticks_frames: list[pl.DataFrame] = []
        d = start_date
        while d <= end_date:
            tick_dir = paths.canonical_ticks_dir(symbol, d)
            df = store.read_dir(tick_dir)
            if not df.is_empty():
                all_ticks_frames.append(df)
            d += dt.timedelta(days=1)

        if all_ticks_frames:
            all_ticks = pl.concat(all_ticks_frames, how="diagonal_relaxed")
            bars = build_bars_from_ticks(all_ticks, timeframe, symbol)
            if not bars.is_empty():
                # Partition by date and write
                bars = bars.with_columns(pl.col("time_utc").dt.date().alias("_date"))
                for date_val in bars["_date"].unique().sort().to_list():
                    day_bars = bars.filter(pl.col("_date") == date_val).drop("_date")
                    path = paths.built_bars_file(symbol, timeframe, date_val)
                    store.write(day_bars, path)
                    total += len(day_bars)
    else:
        while current <= end_date:
            total += build_bars_for_date(symbol, timeframe, current, paths, store)
            current += dt.timedelta(days=1)

    log.info(
        "bars_built",
        symbol=symbol,
        timeframe=timeframe,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        total=total,
    )
    return total
