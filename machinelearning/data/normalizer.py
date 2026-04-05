"""Robust feature normalization utilities for the Phase 6 data layer."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from .schema import ColumnSchema, DEFAULT_SCHEMA, NORMALIZED_FEATURE_COLUMNS

IQR_FLOOR = 1e-8
IQR_CLIP_LIMIT = 10.0


def _coerce_float(value: Any, fallback: float) -> float:
    """Convert an aggregate result to float while keeping a sensible fallback."""

    if value is None:
        return fallback
    return float(value)


class RobustFeatureNormalizer:
    """Median/IQR feature normalizer designed for outlier-heavy gold tick data."""

    def __init__(
        self,
        schema: ColumnSchema = DEFAULT_SCHEMA,
        columns: Sequence[str] | None = None,
    ) -> None:
        self.schema = schema
        self.columns = list(columns or NORMALIZED_FEATURE_COLUMNS)
        self.stats: dict[str, dict[str, float]] = {}
        self._fitted = False

    def fit(self, df: pl.DataFrame) -> "RobustFeatureNormalizer":
        """Fit per-column median and IQR statistics on a training dataframe."""

        stats: dict[str, dict[str, float]] = {}
        for column in self.columns:
            if column not in df.columns:
                warnings.warn(
                    f"Missing feature column during normalizer fit: {column}",
                    stacklevel=2,
                )
                stats[column] = {"median": 0.0, "iqr": IQR_FLOOR}
                continue

            series = (
                df.get_column(column)
                .cast(pl.Float64, strict=False)
                .fill_nan(None)
                .drop_nulls()
            )
            if series.len() == 0:
                median = 0.0
                iqr = IQR_FLOOR
            else:
                median = _coerce_float(series.median(), 0.0)
                q1 = _coerce_float(series.quantile(0.25, interpolation="linear"), median)
                q3 = _coerce_float(series.quantile(0.75, interpolation="linear"), median)
                iqr = max(q3 - q1, IQR_FLOOR)

            stats[column] = {"median": median, "iqr": iqr}

        self.stats = stats
        self._fitted = True
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform a dataframe using fitted median/IQR statistics."""

        if not self._fitted:
            raise RuntimeError("RobustFeatureNormalizer.transform() called before fit().")

        transformed = df.clone()
        missing_columns = [column for column in self.columns if column not in transformed.columns]
        if missing_columns:
            transformed = transformed.with_columns(
                [pl.lit(0.0).alias(column) for column in missing_columns],
            )

        expressions = []
        for column in self.columns:
            if column in missing_columns:
                expressions.append(pl.lit(0.0).alias(column))
                continue

            stats = self.stats[column]
            median = stats["median"]
            iqr = max(stats["iqr"], IQR_FLOOR)
            expressions.append(
                (
                    (pl.col(column).cast(pl.Float64, strict=False).fill_nan(None) - median) / iqr
                )
                .clip(-IQR_CLIP_LIMIT, IQR_CLIP_LIMIT)
                .fill_null(0.0)
                .fill_nan(0.0)
                .alias(column),
            )

        return transformed.with_columns(expressions)

    def save(self, path: str | Path) -> Path:
        """Persist the fitted normalizer to a JSON file."""

        if not self._fitted:
            raise RuntimeError("RobustFeatureNormalizer.save() called before fit().")

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.schema.version,
            "schema": {
                "past_observed": list(self.schema.past_observed),
                "future_known": list(self.schema.future_known),
                "static": list(self.schema.static),
                "targets": list(self.schema.targets),
            },
            "columns": list(self.columns),
            "stats": self.stats,
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "RobustFeatureNormalizer":
        """Load a previously saved normalizer from JSON."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        schema_payload = payload["schema"]
        schema = ColumnSchema(
            past_observed=schema_payload["past_observed"],
            future_known=schema_payload["future_known"],
            static=schema_payload["static"],
            targets=schema_payload["targets"],
            version=payload.get("version", DEFAULT_SCHEMA.version),
        )
        normalizer = cls(schema=schema, columns=payload["columns"])
        normalizer.stats = {
            column: {
                "median": float(values["median"]),
                "iqr": float(values["iqr"]),
            }
            for column, values in payload["stats"].items()
        }
        normalizer._fitted = True
        return normalizer


__all__ = ["RobustFeatureNormalizer"]
