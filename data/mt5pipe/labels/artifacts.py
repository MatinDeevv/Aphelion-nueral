"""Label artifact references and loading helpers."""

from __future__ import annotations

import datetime as dt

import polars as pl
from pydantic import BaseModel, Field

from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths


class LabelArtifactRef(BaseModel):
    """Stable reference to a persisted label-view artifact family."""

    label_pack_key: str = Field(..., min_length=1)
    clock: str = Field(..., min_length=1)
    artifact_id: str | None = None


def load_label_artifact(
    paths: StoragePaths,
    store: ParquetStore,
    ref: LabelArtifactRef,
    *,
    date_from: dt.date | None = None,
    date_to: dt.date | None = None,
) -> pl.DataFrame:
    """Load a persisted label view by pack key and base clock."""
    if ref.artifact_id:
        artifact_root = paths.label_artifact_root(ref.label_pack_key, ref.clock, ref.artifact_id)
        artifact_frame = store.read_dir(artifact_root)
        if not artifact_frame.is_empty():
            if date_from is None or date_to is None:
                return artifact_frame.sort("time_utc")
            return artifact_frame.filter(
                pl.col("time_utc").dt.date().is_between(date_from, date_to)
            ).sort("time_utc")

    if date_from is None or date_to is None:
        return store.read_dir(paths.root / "label_views" / f"label_pack={ref.label_pack_key}" / f"clock={ref.clock}")

    frames: list[pl.DataFrame] = []
    current = date_from
    while current <= date_to:
        day = store.read_dir(paths.label_view_dir(ref.label_pack_key, ref.clock, current))
        if not day.is_empty():
            frames.append(day)
        current += dt.timedelta(days=1)

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").sort("time_utc")
