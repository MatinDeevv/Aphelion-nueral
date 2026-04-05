"""Shared test fixtures."""

from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import polars as pl
import pytest

from mt5pipe.storage.checkpoint_db import CheckpointDB
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def paths(tmp_data_dir: Path) -> StoragePaths:
    return StoragePaths(tmp_data_dir)


@pytest.fixture
def store() -> ParquetStore:
    return ParquetStore(compression="snappy", row_group_size=1000)


@pytest.fixture
def checkpoint_db(tmp_data_dir: Path) -> CheckpointDB:
    tmp_data_dir.mkdir(parents=True, exist_ok=True)
    db = CheckpointDB(tmp_data_dir / "test_checkpoints.db")
    yield db
    db.close()


@pytest.fixture
def sample_ticks_a() -> pl.DataFrame:
    """Sample raw ticks from broker_a."""
    n = 100
    base_msc = 1700000000000  # ~2023-11-14
    return pl.DataFrame({
        "broker_id": ["broker_a"] * n,
        "symbol": ["XAUUSD"] * n,
        "time_utc": [
            dt.datetime.fromtimestamp(base_msc // 1000 + i, tz=dt.timezone.utc) for i in range(n)
        ],
        "time_msc": [base_msc + i * 1000 for i in range(n)],
        "bid": [1950.0 + i * 0.01 for i in range(n)],
        "ask": [1950.5 + i * 0.01 for i in range(n)],
        "last": [0.0] * n,
        "volume": [1.0] * n,
        "volume_real": [0.0] * n,
        "flags": [6] * n,
        "ingest_ts": [dt.datetime.now(dt.timezone.utc)] * n,
    })


@pytest.fixture
def sample_ticks_b() -> pl.DataFrame:
    """Sample raw ticks from broker_b — slightly offset."""
    n = 100
    base_msc = 1700000000050  # 50ms offset
    return pl.DataFrame({
        "broker_id": ["broker_b"] * n,
        "symbol": ["XAUUSD"] * n,
        "time_utc": [
            dt.datetime.fromtimestamp(base_msc // 1000 + i, tz=dt.timezone.utc) for i in range(n)
        ],
        "time_msc": [base_msc + i * 1000 for i in range(n)],
        "bid": [1950.1 + i * 0.01 for i in range(n)],
        "ask": [1950.4 + i * 0.01 for i in range(n)],
        "last": [0.0] * n,
        "volume": [1.0] * n,
        "volume_real": [0.0] * n,
        "flags": [6] * n,
        "ingest_ts": [dt.datetime.now(dt.timezone.utc)] * n,
    })


@pytest.fixture
def sample_canonical_ticks() -> pl.DataFrame:
    """Sample canonical ticks for bar building tests."""
    n = 500
    base_msc = 1700000000000
    return pl.DataFrame({
        "ts_utc": [
            dt.datetime.fromtimestamp(base_msc // 1000 + i * 10, tz=dt.timezone.utc) for i in range(n)
        ],
        "ts_msc": [base_msc + i * 10000 for i in range(n)],
        "symbol": ["XAUUSD"] * n,
        "bid": [1950.0 + (i % 60) * 0.1 for i in range(n)],
        "ask": [1950.5 + (i % 60) * 0.1 for i in range(n)],
        "last": [0.0] * n,
        "volume": [1.0] * n,
        "source_primary": ["broker_a" if i % 2 == 0 else "broker_b" for i in range(n)],
        "source_secondary": ["broker_b" if i % 2 == 0 else "broker_a" for i in range(n)],
        "merge_mode": ["best"] * n,
        "quality_score": [0.8] * n,
        "conflict_flag": [False] * n,
        "broker_a_bid": [1950.0 + (i % 60) * 0.1 for i in range(n)],
        "broker_a_ask": [1950.5 + (i % 60) * 0.1 for i in range(n)],
        "broker_b_bid": [1950.1 + (i % 60) * 0.1 for i in range(n)],
        "broker_b_ask": [1950.4 + (i % 60) * 0.1 for i in range(n)],
        "mid_diff": [0.1] * n,
        "spread_diff": [0.1] * n,
    })
