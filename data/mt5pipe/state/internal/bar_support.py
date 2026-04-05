"""State-local bar validation, timeframe, and gap helpers."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import polars as pl


_TIMEFRAME_SECONDS = {
    "S1": 1,
    "S5": 5,
    "S10": 10,
    "S15": 15,
    "S30": 30,
    "M1": 60,
    "M2": 120,
    "M3": 180,
    "M4": 240,
    "M5": 300,
    "M6": 360,
    "M10": 600,
    "M12": 720,
    "M15": 900,
    "M20": 1200,
    "M30": 1800,
    "H1": 3600,
    "H2": 7200,
    "H3": 10800,
    "H4": 14400,
    "H6": 21600,
    "H8": 28800,
    "H12": 43200,
    "D1": 86400,
    "W1": 7 * 86400,
    "MN1": 30 * 86400,
}

FOREX_CLOSE_WEEKDAY = 4
FOREX_CLOSE_HOUR = 22
FOREX_OPEN_WEEKDAY = 6
FOREX_OPEN_HOUR = 22


@dataclass
class GapReport:
    timeframe: str = ""
    total_expected_bars: int = 0
    actual_bars: int = 0
    missing_bars: int = 0
    gaps: list[dict[str, object]] = field(default_factory=list)


def timeframe_to_seconds(tf: str) -> int:
    normalized = tf.strip().upper()
    seconds = _TIMEFRAME_SECONDS.get(normalized)
    if seconds is None:
        raise ValueError(f"Unknown timeframe: {tf}")
    return seconds


def is_forex_closed(ts: dt.datetime) -> bool:
    wd = ts.weekday()
    hour = ts.hour
    if wd == FOREX_CLOSE_WEEKDAY and hour >= FOREX_CLOSE_HOUR:
        return True
    if wd == 5:
        return True
    if wd == FOREX_OPEN_WEEKDAY and hour < FOREX_OPEN_HOUR:
        return True
    return False


def validate_bars(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    result = df
    for column in ["open", "high", "low", "close"]:
        if column in result.columns:
            result = result.filter(pl.col(column) > 0)

    if all(column in result.columns for column in ["open", "high", "low", "close"]):
        result = result.with_columns([
            pl.max_horizontal("open", "high", "close").alias("high"),
            pl.min_horizontal("open", "low", "close").alias("low"),
        ])

    if "tick_count" in result.columns:
        result = result.filter(pl.col("tick_count") > 0)

    if all(column in result.columns for column in ["spread_min", "spread_mean", "spread_max"]):
        result = result.with_columns([
            pl.min_horizontal("spread_min", "spread_mean", "spread_max").alias("spread_min"),
            pl.max_horizontal("spread_min", "spread_mean", "spread_max").alias("spread_max"),
        ])

    numeric_columns = [column for column in result.columns if result[column].dtype in (pl.Float64, pl.Float32)]
    for column in numeric_columns:
        result = result.filter(pl.col(column).is_finite())

    return result


def detect_gaps(
    df: pl.DataFrame,
    timeframe: str,
    tf_seconds: int,
    *,
    time_col: str = "time_utc",
    skip_weekends: bool = True,
) -> GapReport:
    report = GapReport(timeframe=timeframe, actual_bars=len(df))
    if len(df) < 2:
        report.total_expected_bars = len(df)
        return report

    working = df.sort(time_col)
    times = working[time_col].to_list()
    total_expected = 0

    for index in range(1, len(times)):
        previous = times[index - 1]
        current = times[index]
        delta_seconds = (current - previous).total_seconds()
        expected_slots = int(delta_seconds / tf_seconds)

        if skip_weekends and expected_slots > 1:
            weekend_slots = 0
            for slot_index in range(1, expected_slots):
                slot_time = previous + dt.timedelta(seconds=slot_index * tf_seconds)
                if is_forex_closed(slot_time):
                    weekend_slots += 1
            expected_slots -= weekend_slots

        total_expected += 1
        if expected_slots > 1:
            report.gaps.append({
                "start": previous,
                "end": current,
                "missing_count": expected_slots - 1,
            })

    report.total_expected_bars = total_expected + 1
    report.missing_bars = sum(int(gap["missing_count"]) for gap in report.gaps)
    return report


def fill_bar_gaps(
    df: pl.DataFrame,
    timeframe: str,
    tf_seconds: int,
    *,
    time_col: str = "time_utc",
    skip_weekends: bool = True,
) -> pl.DataFrame:
    if df.is_empty() or len(df) < 2:
        return df

    max_fill_bars = max(1, int(86400 / tf_seconds))
    working = df.sort(time_col)
    times = working[time_col].to_list()
    rows_to_insert: list[dict[str, object]] = []

    for index in range(1, len(times)):
        previous = times[index - 1]
        current = times[index]
        missing = int((current - previous).total_seconds() / tf_seconds) - 1
        if missing <= 0 or missing > max_fill_bars:
            continue

        previous_row = working.row(index - 1, named=True)
        close_value = previous_row.get("close", 0.0)
        for slot in range(1, missing + 1):
            fill_time = previous + dt.timedelta(seconds=slot * tf_seconds)
            if skip_weekends and is_forex_closed(fill_time):
                continue

            fill_row = {
                time_col: fill_time,
                "open": close_value,
                "high": close_value,
                "low": close_value,
                "close": close_value,
                "tick_count": 0,
                "volume_sum": 0.0,
                "mid_return": 0.0,
                "realized_vol": 0.0,
                "spread_mean": previous_row.get("spread_mean", 0.0),
                "spread_max": previous_row.get("spread_max", 0.0),
                "spread_min": previous_row.get("spread_min", 0.0),
                "source_count": 0,
                "conflict_count": 0,
                "_filled": True,
            }
            for meta_column in ("symbol", "timeframe"):
                if meta_column in previous_row:
                    fill_row[meta_column] = previous_row[meta_column]
            rows_to_insert.append(fill_row)

    if not rows_to_insert:
        if "_filled" not in working.columns:
            working = working.with_columns(pl.lit(False).alias("_filled"))
        return working

    fill_df = pl.DataFrame(rows_to_insert)
    if "_filled" not in working.columns:
        working = working.with_columns(pl.lit(False).alias("_filled"))
    return pl.concat([working, fill_df], how="diagonal_relaxed").sort(time_col)
