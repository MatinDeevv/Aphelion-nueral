"""Parquet storage — append-only writes with schema enforcement and atomic I/O."""

from __future__ import annotations

import datetime as dt
import os
import tempfile
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from mt5pipe.utils.logging import get_logger

log = get_logger(__name__)


def _atomic_write_table(table: pa.Table, path: Path, compression: str, row_group_size: int) -> None:
    """Write a PyArrow table to a Parquet file atomically via temp+rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".parquet.tmp", dir=str(path.parent))
    try:
        os.close(fd)
        pq.write_table(table, tmp, compression=compression, row_group_size=row_group_size)
        # Atomic rename on Windows replaces target if it exists (Python 3.12+)
        os.replace(tmp, str(path))
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class ParquetStore:
    """Handles reading and writing Parquet files with append semantics."""

    def __init__(self, compression: str = "zstd", row_group_size: int = 500_000) -> None:
        self.compression = compression
        self.row_group_size = row_group_size

    def write(self, df: pl.DataFrame, path: Path) -> int:
        """Write a polars DataFrame to Parquet. Appends if file exists.
        
        Deduplicates on append using time-based columns when available.
        Uses atomic temp+rename to prevent corruption on kill/crash.
        Returns number of rows written.
        """
        if df.is_empty():
            return 0

        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            try:
                existing = pl.read_parquet(path)
            except Exception as exc:
                log.warning("parquet_read_corrupted", path=str(path), error=str(exc))
                existing = pl.DataFrame()

            if not existing.is_empty():
                df = pl.concat([existing, df], how="diagonal_relaxed")
                # Deduplicate: prefer newer rows when time columns overlap
                dedup_cols = []
                for candidate in ["time_msc", "ts_msc", "time_utc", "ts_utc", "bar_start", "anchor_ts_utc"]:
                    if candidate in df.columns:
                        dedup_cols.append(candidate)
                        break
                if dedup_cols:
                    # Add broker/symbol context if available
                    for ctx in ["broker_id", "symbol", "timeframe"]:
                        if ctx in df.columns:
                            dedup_cols.append(ctx)
                    before = len(df)
                    df = df.unique(subset=dedup_cols, keep="last")
                    removed = before - len(df)
                    if removed > 0:
                        log.debug("parquet_dedup_on_append", path=str(path), removed=removed)

        arrow_table = df.to_arrow()
        _atomic_write_table(arrow_table, path, self.compression, self.row_group_size)

        rows = len(df)
        log.debug("parquet_write", path=str(path), rows=rows)
        return rows

    def write_new(self, df: pl.DataFrame, path: Path) -> int:
        """Write to a new Parquet file. Finds next available part number."""
        if df.is_empty():
            return 0

        path.parent.mkdir(parents=True, exist_ok=True)

        # Find next part number
        existing = sorted(path.parent.glob("part-*.parquet"))
        if existing:
            last_part = int(existing[-1].stem.split("-")[1])
            part_num = last_part + 1
        else:
            part_num = 0

        new_path = path.parent / f"part-{part_num:05d}.parquet"

        arrow_table = df.to_arrow()
        _atomic_write_table(arrow_table, new_path, self.compression, self.row_group_size)

        rows = len(df)
        log.debug("parquet_write_new", path=str(new_path), rows=rows)
        return rows

    def append_to_date_partition(self, df: pl.DataFrame, base_path: Path) -> int:
        """Append data to the single parquet file at base_path (creates or appends)."""
        return self.write(df, base_path)

    def read(self, path: Path) -> pl.DataFrame:
        """Read a single Parquet file. Returns empty DataFrame on corruption."""
        if not path.exists():
            return pl.DataFrame()
        try:
            return pl.read_parquet(path)
        except Exception as exc:
            log.warning("parquet_read_failed", path=str(path), error=str(exc))
            return pl.DataFrame()

    def read_dir(self, directory: Path, schema: dict[str, pl.DataType] | None = None) -> pl.DataFrame:
        """Read all Parquet files in a directory. Skips corrupted files."""
        if not directory.exists():
            return pl.DataFrame()

        files = sorted(directory.glob("**/*.parquet"))
        if not files:
            return pl.DataFrame()

        frames = []
        for f in files:
            try:
                frames.append(pl.read_parquet(f))
            except Exception as exc:
                log.warning("parquet_read_skip_corrupted", path=str(f), error=str(exc))

        if not frames:
            return pl.DataFrame()

        result = pl.concat(frames, how="diagonal_relaxed")

        if schema:
            for col, dtype in schema.items():
                if col in result.columns:
                    result = result.with_columns(pl.col(col).cast(dtype))

        return result

    def read_date_range(
        self,
        base_dir: Path,
        start_date: dt.date,
        end_date: dt.date,
    ) -> pl.DataFrame:
        """Read Parquet files for a date range from Hive-partitioned directory."""
        if not base_dir.exists():
            return pl.DataFrame()

        frames: list[pl.DataFrame] = []
        current = start_date
        while current <= end_date:
            date_dir = base_dir / f"date={current.isoformat()}"
            if date_dir.exists():
                for f in sorted(date_dir.glob("*.parquet")):
                    frames.append(pl.read_parquet(f))
            current += dt.timedelta(days=1)

        if not frames:
            return pl.DataFrame()

        return pl.concat(frames, how="diagonal_relaxed")

    def file_exists(self, path: Path) -> bool:
        return path.exists() and path.stat().st_size > 0

    def count_rows(self, path: Path) -> int:
        """Count rows in a Parquet file without reading all data."""
        if not path.exists():
            return 0
        meta = pq.read_metadata(path)
        return meta.num_rows
