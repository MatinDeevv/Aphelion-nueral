"""Tests for schema validation."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from pydantic import ValidationError

from mt5pipe.models.ticks import RawTick, CanonicalTick
from mt5pipe.models.bars import NativeBar, BuiltBar
from mt5pipe.models.checkpoint import IngestionCheckpoint
from mt5pipe.config.models import PipelineConfig, BrokerConfig


class TestRawTickModel:
    def test_valid_tick(self) -> None:
        tick = RawTick(
            broker_id="broker_a",
            symbol="XAUUSD",
            time_utc=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
            time_msc=1704067200000,
            bid=1950.0,
            ask=1950.5,
        )
        assert tick.broker_id == "broker_a"
        assert tick.bid == 1950.0

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            RawTick()  # type: ignore[call-arg]


class TestCanonicalTickModel:
    def test_valid_canonical(self) -> None:
        tick = CanonicalTick(
            ts_utc=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
            ts_msc=1704067200000,
            symbol="XAUUSD",
            bid=1950.0,
            ask=1950.5,
            source_primary="broker_a",
            merge_mode="best",
        )
        assert tick.merge_mode == "best"


class TestNativeBarModel:
    def test_valid_bar(self) -> None:
        bar = NativeBar(
            broker_id="broker_a",
            symbol="XAUUSD",
            timeframe="M5",
            time_utc=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
            open=1950.0,
            high=1951.0,
            low=1949.0,
            close=1950.5,
        )
        assert bar.timeframe == "M5"


class TestBrokerConfig:
    def test_resolve_symbol_default(self) -> None:
        cfg = BrokerConfig(
            broker_id="test",
            terminal_path="C:/MT5/terminal64.exe",
            login=12345,
            password="test",
            server="TestServer",
        )
        assert cfg.resolve_symbol("XAUUSD") == "XAUUSD"

    def test_resolve_symbol_mapped(self) -> None:
        cfg = BrokerConfig(
            broker_id="test",
            terminal_path="C:/MT5/terminal64.exe",
            login=12345,
            password="test",
            server="TestServer",
            symbol_map={"XAUUSD": "XAUUSD.raw"},
        )
        assert cfg.resolve_symbol("XAUUSD") == "XAUUSD.raw"
        assert cfg.resolve_symbol("EURUSD") == "EURUSD"
