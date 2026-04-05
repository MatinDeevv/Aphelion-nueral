"""Gap detection and handling for time-series bar data."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import polars as pl

from mt5pipe.utils.logging import get_logger

log = get_logger(__name__)

# Forex market is closed from Friday 22:00 UTC to Sunday 22:00 UTC.
# These hours are NOT considered gaps.
FOREX_CLOSE_WEEKDAY = 4  # Friday (0=Monday in Python)
FOREX_CLOSE_HOUR = 22
FOREX_OPEN_WEEKDAY = 6  # Sunday
FOREX_OPEN_HOUR = 22


@dataclass
class GapReport:
    """Report of detected gaps."""

    timeframe: str = ""
    total_expected_bars: int = 0
    actual_bars: int = 0
    missing_bars: int = 0
    gaps: list[dict] = field(default_factory=list)  # [{start, end, missing_count}]

    @property
    def completeness_pct(self) -> float:
        return (self.actual_bars / self.total_expected_bars * 100) if self.total_expected_bars else 0.0


def _is_forex_closed(ts: dt.datetime) -> bool:
    """Check if a timestamp falls in the forex weekend closure."""
    wd = ts.weekday()  # 0=Mon ... 6=Sun
    h = ts.hour
    if wd == FOREX_CLOSE_WEEKDAY and h >= FOREX_CLOSE_HOUR:
        return True
    if wd == 5:  # Saturday — always closed
        return True
    if wd == FOREX_OPEN_WEEKDAY and h < FOREX_OPEN_HOUR:
        return True
    return False


def detect_gaps(
    df: pl.DataFrame,
    timeframe: str,
    tf_seconds: int,
    *,
    time_col: str = "time_utc",
    skip_weekends: bool = True,
) -> GapReport:
    """Detect missing bars in a sorted bar DataFrame.

    Returns a GapReport listing each gap with its location and size.
    """
    report = GapReport(timeframe=timeframe, actual_bars=len(df))

    if len(df) < 2:
        report.total_expected_bars = len(df)
        return report

    df = df.sort(time_col)
    times = df[time_col].to_list()

    total_expected = 0
    for i in range(1, len(times)):
        prev = times[i - 1]
        curr = times[i]
        delta_s = (curr - prev).total_seconds()
        expected_delta = tf_seconds

        # How many bars should fit between prev and curr
        n_expected = int(delta_s / expected_delta)

        if skip_weekends and n_expected > 1:
            # Count how many of those expected slots fall in weekends
            weekend_slots = 0
            for slot_idx in range(1, n_expected):
                slot_time = prev + dt.timedelta(seconds=slot_idx * expected_delta)
                if _is_forex_closed(slot_time):
                    weekend_slots += 1
            n_expected -= weekend_slots

        total_expected += 1  # count the actual bar at position i

        if n_expected > 1:
            missing = n_expected - 1
            report.gaps.append({
                "start": prev,
                "end": curr,
                "missing_count": missing,
            })

    report.total_expected_bars = total_expected + 1  # +1 for the first bar
    report.missing_bars = sum(g["missing_count"] for g in report.gaps)

    if report.gaps:
        log.info(
            "gaps_detected",
            tf=timeframe,
            gaps=len(report.gaps),
            missing_bars=report.missing_bars,
            completeness=f"{report.completeness_pct:.1f}%",
        )

    return report


def fill_bar_gaps(
    df: pl.DataFrame,
    timeframe: str,
    tf_seconds: int,
    *,
    time_col: str = "time_utc",
    method: str = "forward",
    skip_weekends: bool = True,
) -> pl.DataFrame:
    """Fill missing bars using forward-fill (last known close becomes OHLC).

    Only fills gaps up to a configurable max gap size to avoid synthesizing
    long stretches of fake data. Gaps larger than 24h are left as-is
    (they're likely legitimate market closures).
    """
    if df.is_empty() or len(df) < 2:
        return df

    max_fill_bars = max(1, int(86400 / tf_seconds))  # max 1 day worth of fill

    df = df.sort(time_col)
    times = df[time_col].to_list()
    rows_to_insert: list[dict] = []

    for i in range(1, len(times)):
        prev = times[i - 1]
        curr = times[i]
        delta_s = (curr - prev).total_seconds()
        n_missing = int(delta_s / tf_seconds) - 1

        if n_missing <= 0 or n_missing > max_fill_bars:
            continue

        # Get previous row's close as the fill value
        prev_row = df.row(i - 1, named=True)
        close_val = prev_row.get("close", 0.0)

        for slot in range(1, n_missing + 1):
            fill_time = prev + dt.timedelta(seconds=slot * tf_seconds)

            if skip_weekends and _is_forex_closed(fill_time):
                continue

            fill_row = {
                time_col: fill_time,
                "open": close_val,
                "high": close_val,
                "low": close_val,
                "close": close_val,
                "tick_count": 0,
                "volume_sum": 0.0,
                "mid_return": 0.0,
                "realized_vol": 0.0,
                "spread_mean": prev_row.get("spread_mean", 0.0),
                "spread_max": prev_row.get("spread_max", 0.0),
                "spread_min": prev_row.get("spread_min", 0.0),
                "source_count": 0,
                "conflict_count": 0,
                "_filled": True,
            }
            # Copy symbol/timeframe if present
            for meta_col in ("symbol", "timeframe"):
                if meta_col in prev_row:
                    fill_row[meta_col] = prev_row[meta_col]

            rows_to_insert.append(fill_row)

    if not rows_to_insert:
        if "_filled" not in df.columns:
            df = df.with_columns(pl.lit(False).alias("_filled"))
        return df

    fill_df = pl.DataFrame(rows_to_insert)

    # Add _filled marker to original rows
    if "_filled" not in df.columns:
        df = df.with_columns(pl.lit(False).alias("_filled"))

    result = pl.concat([df, fill_df], how="diagonal_relaxed").sort(time_col)

    log.info("bars_gap_filled", filled=len(rows_to_insert), total=len(result))

    return result
