"""Walk-forward splitting helpers for Phase 6 temporal evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Iterator

import polars as pl


@dataclass(slots=True)
class WalkForwardResult:
    """Container for scalar metrics collected across walk-forward folds."""

    fold_metrics: list[dict[str, float]] = field(default_factory=list)

    @property
    def mean(self) -> dict[str, float]:
        """Return the arithmetic mean of each metric across folds."""

        return self._aggregate(mean)

    @property
    def std(self) -> dict[str, float]:
        """Return the population standard deviation of each metric across folds."""

        return self._aggregate(lambda values: 0.0 if len(values) <= 1 else pstdev(values))

    def summary_str(self) -> str:
        """Return a compact human-readable summary for logs and experiment notes."""

        if not self.fold_metrics:
            return "walk_forward: no folds"

        lines: list[str] = []
        for fold_index, metrics in enumerate(self.fold_metrics):
            formatted = ", ".join(
                f"{name}={value:.6f}" for name, value in sorted(metrics.items())
            )
            lines.append(f"fold_{fold_index}: {formatted}")

        mean_metrics = ", ".join(
            f"{name}={value:.6f}" for name, value in sorted(self.mean.items())
        )
        std_metrics = ", ".join(
            f"{name}={value:.6f}" for name, value in sorted(self.std.items())
        )
        lines.append(f"mean: {mean_metrics}")
        lines.append(f"std: {std_metrics}")
        return "\n".join(lines)

    def _aggregate(self, reducer) -> dict[str, float]:
        metric_names = {
            metric_name
            for fold in self.fold_metrics
            for metric_name in fold
        }
        aggregated: dict[str, float] = {}
        for metric_name in sorted(metric_names):
            values = [float(fold[metric_name]) for fold in self.fold_metrics if metric_name in fold]
            if values:
                aggregated[metric_name] = float(reducer(values))
        return aggregated


class WalkForwardSplitter:
    """
    Generate expanding-window train/validation folds from a Polars DataFrame.

    This enforces temporal ordering rather than random k-fold shuffling.
    Each fold keeps all earlier rows for training, applies an embargo gap to
    prevent forward-horizon label leakage, and validates on the next
    contiguous block of rows.
    """

    def __init__(
        self,
        n_folds: int = 3,
        val_fraction: float = 0.15,
        embargo_rows: int = 240,
    ) -> None:
        if n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be within (0, 1)")
        if embargo_rows < 0:
            raise ValueError("embargo_rows must be >= 0")

        self.n_folds = n_folds
        self.val_fraction = val_fraction
        self.embargo_rows = embargo_rows

    def split(self, df: pl.DataFrame, time_col: str = "time_utc") -> Iterator[tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Yield temporally ordered expanding-window train/validation dataframe pairs.

        The frame is sorted by ``time_col`` first. Validation blocks occupy the
        most recent reserved portion of the dataset so the final fold reaches the
        newest rows. An embargo gap is inserted between train end and validation
        start to avoid leakage from forward-looking labels.
        """

        if time_col not in df.columns:
            raise ValueError(f"time_col '{time_col}' is missing from the dataframe")

        ordered = df.sort(time_col)
        row_count = ordered.height
        val_size = int(row_count * self.val_fraction / self.n_folds)
        reserved_rows = self.n_folds * (val_size + self.embargo_rows)
        initial_train_rows = row_count - reserved_rows

        if val_size < 1 or initial_train_rows < 1:
            raise ValueError(
                "dataset too small for walk-forward config "
                f"(rows={row_count}, n_folds={self.n_folds}, "
                f"val_fraction={self.val_fraction}, embargo_rows={self.embargo_rows})"
            )

        for fold_index in range(self.n_folds):
            train_end = initial_train_rows + fold_index * (val_size + self.embargo_rows)
            val_start = train_end + self.embargo_rows
            val_end = val_start + val_size
            if val_end > row_count:
                raise ValueError(
                    "dataset too small for requested walk-forward config after fold construction "
                    f"(fold={fold_index}, val_end={val_end}, rows={row_count})"
                )

            train_df = ordered.slice(0, train_end)
            val_df = ordered.slice(val_start, val_size)
            if train_df.is_empty() or val_df.is_empty():
                raise ValueError(
                    "walk-forward split produced an empty train or validation fold "
                    f"(fold={fold_index})"
                )
            yield train_df, val_df


__all__ = ["WalkForwardResult", "WalkForwardSplitter"]
