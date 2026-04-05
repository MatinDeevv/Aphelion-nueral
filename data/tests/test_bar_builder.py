"""Tests for bar builder — timeframe aggregation from ticks."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from mt5pipe.bars.builder import build_bars_from_ticks, floor_timestamp_to_bar, timeframe_to_seconds


class TestTimeframeSeconds:
    def test_m1(self) -> None:
        assert timeframe_to_seconds("M1") == 60

    def test_h1(self) -> None:
        assert timeframe_to_seconds("H1") == 3600

    def test_d1(self) -> None:
        assert timeframe_to_seconds("D1") == 86400

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown timeframe"):
            timeframe_to_seconds("X99")

    def test_all_configured(self) -> None:
        from mt5pipe.mt5.constants import TIMEFRAME_SECONDS
        for tf in TIMEFRAME_SECONDS:
            assert timeframe_to_seconds(tf) > 0


class TestFloorTimestamp:
    def test_m1_floor(self) -> None:
        ts = dt.datetime(2024, 1, 15, 10, 23, 45, tzinfo=dt.timezone.utc)
        floored = floor_timestamp_to_bar(ts, "M1")
        assert floored == dt.datetime(2024, 1, 15, 10, 23, 0, tzinfo=dt.timezone.utc)

    def test_h1_floor(self) -> None:
        ts = dt.datetime(2024, 1, 15, 10, 45, 0, tzinfo=dt.timezone.utc)
        floored = floor_timestamp_to_bar(ts, "H1")
        assert floored == dt.datetime(2024, 1, 15, 10, 0, 0, tzinfo=dt.timezone.utc)

    def test_mn1_floor(self) -> None:
        ts = dt.datetime(2024, 3, 15, 10, 0, 0, tzinfo=dt.timezone.utc)
        floored = floor_timestamp_to_bar(ts, "MN1")
        assert floored.day == 1
        assert floored.month == 3

    def test_w1_floor(self) -> None:
        # 2024-01-17 is a Wednesday
        ts = dt.datetime(2024, 1, 17, 10, 0, 0, tzinfo=dt.timezone.utc)
        floored = floor_timestamp_to_bar(ts, "W1")
        assert floored.weekday() == 0  # Monday
        assert floored.day == 15  # Mon Jan 15


class TestBuildBarsFromTicks:
    def test_basic_bar_building(self, sample_canonical_ticks: pl.DataFrame) -> None:
        bars = build_bars_from_ticks(sample_canonical_ticks, "M1", "XAUUSD")
        assert not bars.is_empty()
        assert "open" in bars.columns
        assert "high" in bars.columns
        assert "low" in bars.columns
        assert "close" in bars.columns
        assert "tick_count" in bars.columns

    def test_bar_ohlc_correctness(self, sample_canonical_ticks: pl.DataFrame) -> None:
        bars = build_bars_from_ticks(sample_canonical_ticks, "M1", "XAUUSD")
        for row in bars.iter_rows(named=True):
            assert row["high"] >= row["open"]
            assert row["high"] >= row["close"]
            assert row["low"] <= row["open"]
            assert row["low"] <= row["close"]
            assert row["high"] >= row["low"]

    def test_tick_count_positive(self, sample_canonical_ticks: pl.DataFrame) -> None:
        bars = build_bars_from_ticks(sample_canonical_ticks, "M1", "XAUUSD")
        assert (bars["tick_count"] > 0).all()

    def test_spread_stats(self, sample_canonical_ticks: pl.DataFrame) -> None:
        bars = build_bars_from_ticks(sample_canonical_ticks, "M5", "XAUUSD")
        if not bars.is_empty():
            for row in bars.iter_rows(named=True):
                assert row["spread_min"] <= row["spread_mean"]
                assert row["spread_mean"] <= row["spread_max"]

    def test_multiple_timeframes(self, sample_canonical_ticks: pl.DataFrame) -> None:
        for tf in ["M1", "M5", "M15", "H1"]:
            bars = build_bars_from_ticks(sample_canonical_ticks, tf, "XAUUSD")
            assert "timeframe" in bars.columns
            if not bars.is_empty():
                assert bars["timeframe"][0] == tf

    def test_empty_input(self) -> None:
        empty = pl.DataFrame()
        bars = build_bars_from_ticks(empty, "M1", "XAUUSD")
        assert bars.is_empty()

    def test_symbol_column(self, sample_canonical_ticks: pl.DataFrame) -> None:
        bars = build_bars_from_ticks(sample_canonical_ticks, "M1", "XAUUSD")
        assert (bars["symbol"] == "XAUUSD").all()
