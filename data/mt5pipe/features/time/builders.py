"""Time feature builders."""

from __future__ import annotations

import math

import polars as pl


def add_time_features(df: pl.DataFrame, time_col: str = "time_utc") -> pl.DataFrame:
    """Add cyclical time-of-day and day-of-week features."""
    df = df.with_columns([
        pl.col(time_col).dt.hour().alias("hour"),
        pl.col(time_col).dt.minute().alias("minute"),
        pl.col(time_col).dt.weekday().alias("weekday"),
    ])
    df = df.with_columns([
        (pl.col("hour") * 60 + pl.col("minute")).alias("minute_of_day"),
    ])
    df = df.with_columns([
        (pl.col("minute_of_day") / 1440.0 * 2.0 * math.pi).sin().alias("time_sin"),
        (pl.col("minute_of_day") / 1440.0 * 2.0 * math.pi).cos().alias("time_cos"),
        (pl.col("weekday") / 7.0 * 2.0 * math.pi).sin().alias("weekday_sin"),
        (pl.col("weekday") / 7.0 * 2.0 * math.pi).cos().alias("weekday_cos"),
    ])
    return df.drop(["minute_of_day"])
