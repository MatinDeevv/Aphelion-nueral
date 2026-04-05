"""
Dataset-level contracts shared across sectors.

These types describe dataset identity and split semantics.
They do NOT duplicate DatasetSpec (which lives in compiler.models)
but instead provide the minimal shared vocabulary.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DatasetSplitKind(str, Enum):
    """Standard split names used throughout the pipeline."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DatasetId(BaseModel):
    """
    Minimal identity of a compiled dataset.

    Used as a shared reference so features/state don't need to import
    the full DatasetSpec from compiler.models.
    """

    dataset_name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)

    @property
    def key(self) -> str:
        return f"{self.dataset_name}@{self.version}"

    def __str__(self) -> str:
        return self.key


# ---------------------------------------------------------------------------
# Join-key contract: canonical columns used when joining state + features + labels
# ---------------------------------------------------------------------------

DATASET_JOIN_KEYS: list[str] = ["symbol", "timeframe", "time_utc"]
"""
Primary join keys for combining state, feature, and label DataFrames.

All sectors MUST produce DataFrames that include these columns
when their output is destined for dataset compilation.
"""
