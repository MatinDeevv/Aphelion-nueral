"""Public feature-sector types."""

from __future__ import annotations

from typing import Any, Protocol

import polars as pl


class FeatureBuilder(Protocol):
    """Public callable protocol for registry-backed feature builders."""

    def __call__(self, df: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
        """Build feature columns from a PIT-safe input frame."""
