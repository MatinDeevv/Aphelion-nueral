"""Tests for canonical dual-broker merger."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from mt5pipe.config.models import MergeConfig
from mt5pipe.merge.canonical import (
    _BrokerQuote,
    _count_wallclock_overlap_minutes,
    _estimate_near_miss_pairs,
    _median_or_zero,
    _percentile_or_zero,
    _resolve_bucket,
    _score_quote,
    merge_canonical_ticks,
)


@pytest.fixture
def merge_cfg() -> MergeConfig:
    return MergeConfig()


class TestBrokerQuote:
    def test_valid_quote(self) -> None:
        q = _BrokerQuote("broker_a", 1000, 1950.0, 1950.5, 0.0, 1.0)
        assert q.is_valid(0.005)
        assert q.mid == 1950.25
        assert q.spread == 0.5

    def test_invalid_bid_zero(self) -> None:
        q = _BrokerQuote("broker_a", 1000, 0.0, 1950.5, 0.0, 1.0)
        assert not q.is_valid(0.005)

    def test_invalid_bid_gt_ask(self) -> None:
        q = _BrokerQuote("broker_a", 1000, 1951.0, 1950.0, 0.0, 1.0)
        assert not q.is_valid(0.005)

    def test_invalid_spread_too_wide(self) -> None:
        q = _BrokerQuote("broker_a", 1000, 1940.0, 1960.0, 0.0, 1.0)
        assert not q.is_valid(0.005)  # 20/1950 ≈ 1% > 0.5%


class TestScoreQuote:
    def test_scores_between_0_and_1(self, merge_cfg: MergeConfig) -> None:
        q = _BrokerQuote("broker_a", 1000, 1950.0, 1950.5, 0.0, 1.0)
        score = _score_quote(q, None, 1950.25, merge_cfg, 1000)
        assert 0.0 <= score <= 1.0

    def test_tighter_spread_gets_better_score(self, merge_cfg: MergeConfig) -> None:
        tight = _BrokerQuote("a", 1000, 1950.0, 1950.2, 0.0, 1.0)
        wide = _BrokerQuote("b", 1000, 1949.0, 1951.0, 0.0, 1.0)
        s_tight = _score_quote(tight, None, 1950.1, merge_cfg, 1000)
        s_wide = _score_quote(wide, None, 1950.1, merge_cfg, 1000)
        assert s_tight > s_wide

    def test_fresher_quote_gets_better_score(self, merge_cfg: MergeConfig) -> None:
        fresh = _BrokerQuote("a", 1090, 1950.0, 1950.5, 0.0, 1.0)
        stale = _BrokerQuote("b", 1010, 1950.0, 1950.5, 0.0, 1.0)
        s_fresh = _score_quote(fresh, None, 1950.25, merge_cfg, 1100)
        s_stale = _score_quote(stale, None, 1950.25, merge_cfg, 1100)
        assert s_fresh > s_stale


class TestResolveBucket:
    def test_single_source_a(self, merge_cfg: MergeConfig) -> None:
        q_a = _BrokerQuote("broker_a", 1000, 1950.0, 1950.5, 0.0, 1.0)
        row = _resolve_bucket(
            q_a, None, "broker_a", "broker_b", 0, 1, merge_cfg, None, 1000, "XAUUSD"
        )
        assert row is not None
        assert row["merge_mode"] == "single"
        assert row["source_primary"] == "broker_a"
        assert not row["conflict_flag"]

    def test_single_source_b(self, merge_cfg: MergeConfig) -> None:
        q_b = _BrokerQuote("broker_b", 1000, 1950.0, 1950.5, 0.0, 1.0)
        row = _resolve_bucket(
            None, q_b, "broker_a", "broker_b", 0, 1, merge_cfg, None, 1000, "XAUUSD"
        )
        assert row is not None
        assert row["merge_mode"] == "single"
        assert row["source_primary"] == "broker_b"

    def test_both_sources_best(self, merge_cfg: MergeConfig) -> None:
        q_a = _BrokerQuote("broker_a", 1000, 1950.0, 1950.5, 0.0, 1.0)
        q_b = _BrokerQuote("broker_b", 1000, 1950.1, 1950.4, 0.0, 1.0)
        row = _resolve_bucket(
            q_a, q_b, "broker_a", "broker_b", 0, 1, merge_cfg, 1950.25, 1000, "XAUUSD"
        )
        assert row is not None
        assert row["merge_mode"] in ("best", "conflict")
        assert row["broker_a_bid"] == 1950.0
        assert row["broker_b_bid"] == 1950.1

    def test_conflict_detected(self, merge_cfg: MergeConfig) -> None:
        # Large price difference triggers conflict
        q_a = _BrokerQuote("broker_a", 1000, 1950.0, 1950.5, 0.0, 1.0)
        q_b = _BrokerQuote("broker_b", 1000, 1955.0, 1955.5, 0.0, 1.0)
        cfg = MergeConfig(conflict_log_threshold=0.001)
        row = _resolve_bucket(
            q_a, q_b, "broker_a", "broker_b", 0, 1, cfg, 1950.25, 1000, "XAUUSD"
        )
        assert row is not None
        assert row["conflict_flag"] is True
        assert row["merge_mode"] == "conflict"

    def test_both_none_returns_none(self, merge_cfg: MergeConfig) -> None:
        row = _resolve_bucket(
            None, None, "broker_a", "broker_b", 0, 1, merge_cfg, None, 1000, "XAUUSD"
        )
        assert row is None


class TestMergeDiagnosticsHelpers:
    def test_median_or_zero(self) -> None:
        assert _median_or_zero([]) == 0.0
        assert _median_or_zero([7]) == 7.0
        assert _median_or_zero([1, 5, 3]) == 3.0
        assert _median_or_zero([1, 3, 5, 7]) == 4.0

    def test_percentile_or_zero(self) -> None:
        assert _percentile_or_zero([], 0.95) == 0.0
        assert _percentile_or_zero([10], 0.95) == 10.0
        assert _percentile_or_zero([1, 2, 3, 4, 5], 0.95) >= 4.0

    def test_estimate_near_miss_pairs(self) -> None:
        # A ticks at 0,1000,2000 and B ticks slightly outside 100ms bucket.
        # Offsets are 120ms, so they are near misses when factor=5 (<=500ms).
        df_a = pl.DataFrame({"time_msc": [0, 1000, 2000]})
        df_b = pl.DataFrame({"time_msc": [120, 1120, 2120]})
        near = _estimate_near_miss_pairs(df_a, df_b, bucket_ms=100, factor=5)
        assert near == 3

    def test_estimate_near_miss_pairs_zero_when_within_bucket(self) -> None:
        df_a = pl.DataFrame({"time_msc": [0, 1000, 2000]})
        df_b = pl.DataFrame({"time_msc": [20, 1020, 2020]})
        near = _estimate_near_miss_pairs(df_a, df_b, bucket_ms=100, factor=5)
        assert near == 0

    def test_count_wallclock_overlap_minutes(self) -> None:
        # A: minute buckets 0,1,2. B: minute buckets 1,2,3 -> overlap = 2
        df_a = pl.DataFrame({"time_msc": [10_000, 70_000, 130_000]})
        df_b = pl.DataFrame({"time_msc": [65_000, 125_000, 185_000]})
        overlap = _count_wallclock_overlap_minutes(df_a, df_b)
        assert overlap == 2


def test_merge_rerun_overwrites_daily_diagnostics(paths, store, merge_cfg: MergeConfig) -> None:
    date = dt.date(2026, 4, 2)
    timestamps_a = [dt.datetime(2026, 4, 2, 0, 0, tzinfo=dt.timezone.utc), dt.datetime(2026, 4, 2, 0, 1, tzinfo=dt.timezone.utc)]
    timestamps_b = [ts + dt.timedelta(milliseconds=50) for ts in timestamps_a]

    raw_a = pl.DataFrame(
        {
            "broker_id": ["broker_a"] * len(timestamps_a),
            "symbol": ["XAUUSD"] * len(timestamps_a),
            "time_utc": timestamps_a,
            "time_msc": [int(ts.timestamp() * 1000) for ts in timestamps_a],
            "bid": [3000.0, 3000.1],
            "ask": [3000.2, 3000.3],
            "last": [0.0, 0.0],
            "volume": [1.0, 1.0],
            "volume_real": [0.0, 0.0],
            "flags": [6, 6],
            "ingest_ts": timestamps_a,
        }
    )
    raw_b = pl.DataFrame(
        {
            "broker_id": ["broker_b"] * len(timestamps_b),
            "symbol": ["XAUUSD"] * len(timestamps_b),
            "time_utc": timestamps_b,
            "time_msc": [int(ts.timestamp() * 1000) for ts in timestamps_b],
            "bid": [3000.01, 3000.11],
            "ask": [3000.19, 3000.29],
            "last": [0.0, 0.0],
            "volume": [1.0, 1.0],
            "volume_real": [0.0, 0.0],
            "flags": [6, 6],
            "ingest_ts": timestamps_b,
        }
    )

    store.write(raw_a, paths.raw_ticks_file("broker_a", "XAUUSD", date))
    store.write(raw_b, paths.raw_ticks_file("broker_b", "XAUUSD", date))

    merge_canonical_ticks("broker_a", "broker_b", "XAUUSD", date, paths, store, merge_cfg)
    merge_canonical_ticks("broker_a", "broker_b", "XAUUSD", date, paths, store, merge_cfg)

    canonical = store.read_dir(paths.canonical_ticks_dir("XAUUSD", date))
    assert canonical.height == 2
    diagnostic_files = sorted(paths.merge_diagnostics_dir("XAUUSD", date).glob("*.parquet"))
    assert len(diagnostic_files) == 1
    diagnostics = store.read_dir(paths.merge_diagnostics_dir("XAUUSD", date))
    assert diagnostics.height == 1
