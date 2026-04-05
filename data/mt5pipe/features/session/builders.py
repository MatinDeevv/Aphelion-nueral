"""Session feature builders."""

from __future__ import annotations

import polars as pl


def add_session_features(df: pl.DataFrame, time_col: str = "time_utc") -> pl.DataFrame:
    """Add trading session indicators for Asia, London, NY, and overlap."""
    df = df.with_columns([
        pl.col(time_col).dt.hour().alias("_hour"),
    ])
    df = df.with_columns([
        ((pl.col("_hour") >= 0) & (pl.col("_hour") < 8)).cast(pl.Int8).alias("session_asia"),
        ((pl.col("_hour") >= 7) & (pl.col("_hour") < 16)).cast(pl.Int8).alias("session_london"),
        ((pl.col("_hour") >= 13) & (pl.col("_hour") < 22)).cast(pl.Int8).alias("session_ny"),
        ((pl.col("_hour") >= 13) & (pl.col("_hour") < 16)).cast(pl.Int8).alias("session_overlap"),
    ])
    return df.drop("_hour")
