"""Dataset row model."""

from __future__ import annotations

import polars as pl
from pydantic import BaseModel


class DatasetRow(BaseModel):
    """Schema definition for a model-ready dataset row.
    
    Actual dataset construction happens in polars; this defines the contract.
    The full column list is dynamic based on config, but these are the core fields.
    """

    # Base bar fields (M1)
    symbol: str = ""
    time_utc: str = ""  # ISO string
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    tick_count: int = 0
    spread_mean: float = 0.0
    mid_return: float = 0.0
    realized_vol: float = 0.0

    # Labels (populated by label builder)
    # future_return_{horizon}m, direction_{horizon}m, triple_barrier_{horizon}m
    # mae_{horizon}m, mfe_{horizon}m


# Core dataset columns — additional columns are added dynamically
DATASET_CORE_COLUMNS: list[str] = [
    "symbol",
    "time_utc",
    "open",
    "high",
    "low",
    "close",
    "tick_count",
    "spread_mean",
    "mid_return",
    "realized_vol",
]
