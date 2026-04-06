"""Offline regime labeling utilities for compiled dataset sidecar artifacts."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import torch

from machinelearning.data.schema import TIME_INDEX_COLUMN

from .detector import RegimeDetector

REGIME_DOMINANT_COLUMN = "regime_dominant"
REGIME_TRENDING_COLUMN = "regime_trending"
REGIME_MEAN_REV_COLUMN = "regime_mean_rev"
REGIME_VOLATILE_COLUMN = "regime_volatile"
REGIME_QUIET_COLUMN = "regime_quiet"
REGIME_CONFIDENCE_COLUMN = "regime_confidence"

LABEL_COLUMNS = [
    TIME_INDEX_COLUMN,
    REGIME_DOMINANT_COLUMN,
    REGIME_TRENDING_COLUMN,
    REGIME_MEAN_REV_COLUMN,
    REGIME_VOLATILE_COLUMN,
    REGIME_QUIET_COLUMN,
    REGIME_CONFIDENCE_COLUMN,
]


class RegimeLabeler:
    """Generate a sidecar parquet of regime labels keyed by ``time_utc``."""

    def __init__(
        self,
        detector: RegimeDetector,
        batch_size: int = 1000,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.detector = detector
        self.batch_size = batch_size

    def label_dataframe(
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Run rolling-window regime detection over a dataframe and return label columns."""

        if TIME_INDEX_COLUMN not in df.columns:
            raise ValueError(f"RegimeLabeler requires '{TIME_INDEX_COLUMN}' in the input dataframe.")

        ordered = df.sort(TIME_INDEX_COLUMN)
        row_count = ordered.height
        if row_count == 0:
            return self._empty_labels(ordered.schema.get(TIME_INDEX_COLUMN, pl.Datetime))

        time_values = ordered.get_column(TIME_INDEX_COLUMN).to_list()
        records: list[dict[str, object]] = []
        default_record = self._default_record()

        if not self.detector.is_fitted:
            for time_value in time_values:
                records.append({TIME_INDEX_COLUMN: time_value, **default_record})
            return pl.DataFrame(records, schema=self._output_schema())

        feature_matrix = self.detector.extractor.extract_tensor(ordered)
        warmup_rows = max(0, self.detector.window - 1)

        for batch_start in range(0, row_count, self.batch_size):
            batch_stop = min(row_count, batch_start + self.batch_size)
            for row_index in range(batch_start, batch_stop):
                if row_index < warmup_rows:
                    records.append({TIME_INDEX_COLUMN: time_values[row_index], **default_record})
                    continue

                window_start = row_index - self.detector.window + 1
                window_features = feature_matrix[window_start : row_index + 1]
                state = self.detector.forward(window_features)
                records.append({TIME_INDEX_COLUMN: time_values[row_index], **self._state_record(state)})

        return pl.DataFrame(records, schema=self._output_schema())

    def label_parquet(
        self,
        parquet_dir: Path,
        output_path: Path,
    ) -> pl.DataFrame:
        """Read parquet shards, label them, persist the label sidecar, and return it."""

        parquet_root = Path(parquet_dir)
        parquet_paths = sorted(parquet_root.rglob("*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(f"No parquet files found under '{parquet_root}'.")

        frame = pl.concat(
            [pl.read_parquet(path) for path in parquet_paths],
            how="vertical_relaxed",
            rechunk=True,
        )
        labels = self.label_dataframe(frame)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        labels.write_parquet(output)
        return labels

    @staticmethod
    def join_labels(
        dataset_df: pl.DataFrame,
        label_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Left join regime labels onto a dataset dataframe and fill missing rows safely."""

        joined = dataset_df.join(label_df, on=TIME_INDEX_COLUMN, how="left")
        return joined.with_columns(
            pl.col(REGIME_DOMINANT_COLUMN).fill_null("quiet"),
            pl.col(REGIME_TRENDING_COLUMN).fill_null(0.25),
            pl.col(REGIME_MEAN_REV_COLUMN).fill_null(0.25),
            pl.col(REGIME_VOLATILE_COLUMN).fill_null(0.25),
            pl.col(REGIME_QUIET_COLUMN).fill_null(0.25),
            pl.col(REGIME_CONFIDENCE_COLUMN).fill_null(0.25),
        )

    @staticmethod
    def _state_record(state: object) -> dict[str, object]:
        """Convert a ``RegimeState``-like object into the label-sidecar row payload."""

        probs = state.probs.tolist()
        return {
            REGIME_DOMINANT_COLUMN: str(state.dominant),
            REGIME_TRENDING_COLUMN: float(probs[0]),
            REGIME_MEAN_REV_COLUMN: float(probs[1]),
            REGIME_VOLATILE_COLUMN: float(probs[2]),
            REGIME_QUIET_COLUMN: float(probs[3]),
            REGIME_CONFIDENCE_COLUMN: float(state.confidence),
        }

    @staticmethod
    def _default_record() -> dict[str, object]:
        """Return the warmup/unfitted safe-default regime label payload."""

        return {
            REGIME_DOMINANT_COLUMN: "quiet",
            REGIME_TRENDING_COLUMN: 0.25,
            REGIME_MEAN_REV_COLUMN: 0.25,
            REGIME_VOLATILE_COLUMN: 0.25,
            REGIME_QUIET_COLUMN: 0.25,
            REGIME_CONFIDENCE_COLUMN: 0.25,
        }

    @staticmethod
    def _output_schema() -> dict[str, pl.DataType]:
        """Return the canonical output schema for label-sidecar dataframes."""

        return {
            TIME_INDEX_COLUMN: pl.Datetime(time_zone="UTC"),
            REGIME_DOMINANT_COLUMN: pl.Utf8,
            REGIME_TRENDING_COLUMN: pl.Float32,
            REGIME_MEAN_REV_COLUMN: pl.Float32,
            REGIME_VOLATILE_COLUMN: pl.Float32,
            REGIME_QUIET_COLUMN: pl.Float32,
            REGIME_CONFIDENCE_COLUMN: pl.Float32,
        }

    @classmethod
    def _empty_labels(cls, time_dtype: pl.DataType) -> pl.DataFrame:
        """Return an empty label dataframe with the canonical schema."""

        schema = cls._output_schema()
        schema[TIME_INDEX_COLUMN] = time_dtype
        return pl.DataFrame(schema=schema)


__all__ = [
    "LABEL_COLUMNS",
    "REGIME_CONFIDENCE_COLUMN",
    "REGIME_DOMINANT_COLUMN",
    "REGIME_MEAN_REV_COLUMN",
    "REGIME_QUIET_COLUMN",
    "REGIME_TRENDING_COLUMN",
    "REGIME_VOLATILE_COLUMN",
    "RegimeLabeler",
]
