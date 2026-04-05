"""Tests for tick deduplication."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from mt5pipe.ingestion.ticks import deduplicate_ticks


class TestDeduplicateTicks:
    def test_removes_exact_duplicates(self) -> None:
        df = pl.DataFrame({
            "broker_id": ["a", "a", "a"],
            "symbol": ["XAUUSD"] * 3,
            "time_utc": [dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)] * 3,
            "time_msc": [1000, 1000, 2000],
            "bid": [1950.0, 1950.0, 1951.0],
            "ask": [1950.5, 1950.5, 1951.5],
            "last": [0.0, 0.0, 0.0],
            "volume": [1.0, 1.0, 1.0],
            "volume_real": [0.0, 0.0, 0.0],
            "flags": [6, 6, 6],
            "ingest_ts": [dt.datetime.now(dt.timezone.utc)] * 3,
        })
        result = deduplicate_ticks(df)
        assert len(result) == 2

    def test_keeps_different_timestamps(self) -> None:
        df = pl.DataFrame({
            "broker_id": ["a", "a"],
            "symbol": ["XAUUSD", "XAUUSD"],
            "time_utc": [dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)] * 2,
            "time_msc": [1000, 2000],
            "bid": [1950.0, 1950.0],
            "ask": [1950.5, 1950.5],
            "last": [0.0, 0.0],
            "volume": [1.0, 1.0],
            "volume_real": [0.0, 0.0],
            "flags": [6, 6],
            "ingest_ts": [dt.datetime.now(dt.timezone.utc)] * 2,
        })
        result = deduplicate_ticks(df)
        assert len(result) == 2

    def test_keeps_different_prices(self) -> None:
        df = pl.DataFrame({
            "broker_id": ["a", "a"],
            "symbol": ["XAUUSD", "XAUUSD"],
            "time_utc": [dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)] * 2,
            "time_msc": [1000, 1000],
            "bid": [1950.0, 1951.0],
            "ask": [1950.5, 1951.5],
            "last": [0.0, 0.0],
            "volume": [1.0, 1.0],
            "volume_real": [0.0, 0.0],
            "flags": [6, 6],
            "ingest_ts": [dt.datetime.now(dt.timezone.utc)] * 2,
        })
        result = deduplicate_ticks(df)
        assert len(result) == 2

    def test_empty_input(self) -> None:
        df = pl.DataFrame()
        result = deduplicate_ticks(df)
        assert result.is_empty()
