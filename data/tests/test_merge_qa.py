"""Tests for daily merge QA aggregation and coverage guardrails."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from mt5pipe.quality.merge_qa import (
    CoverageParityError,
    assert_synchronized_raw_tick_coverage,
    build_daily_merge_qa_report,
    write_daily_merge_qa_report,
)


UTC = dt.timezone.utc


def _ts(date: dt.date, hour: int, minute: int = 0, second: int = 0) -> dt.datetime:
    return dt.datetime(date.year, date.month, date.day, hour, minute, second, tzinfo=UTC)


def _raw_ticks(broker_id: str, symbol: str, timestamps: list[dt.datetime]) -> pl.DataFrame:
    return pl.DataFrame({
        "broker_id": [broker_id] * len(timestamps),
        "symbol": [symbol] * len(timestamps),
        "time_utc": timestamps,
        "time_msc": [int(ts.timestamp() * 1000) for ts in timestamps],
        "bid": [1950.0 + i * 0.1 for i in range(len(timestamps))],
        "ask": [1950.3 + i * 0.1 for i in range(len(timestamps))],
        "last": [0.0] * len(timestamps),
        "volume": [1.0] * len(timestamps),
        "volume_real": [0.0] * len(timestamps),
        "flags": [6] * len(timestamps),
        "ingest_ts": timestamps,
    })


def _canonical_ticks(symbol: str, timestamps: list[dt.datetime]) -> pl.DataFrame:
    source_secondary = ["broker_b", "", "broker_b", "broker_b"][: len(timestamps)]
    conflict_flags = [False, False, True, False][: len(timestamps)]
    return pl.DataFrame({
        "ts_utc": timestamps,
        "ts_msc": [int(ts.timestamp() * 1000) for ts in timestamps],
        "symbol": [symbol] * len(timestamps),
        "bid": [1950.0 + i * 0.1 for i in range(len(timestamps))],
        "ask": [1950.2 + i * 0.1 for i in range(len(timestamps))],
        "last": [0.0] * len(timestamps),
        "volume": [1.0] * len(timestamps),
        "source_primary": ["broker_a"] * len(timestamps),
        "source_secondary": source_secondary,
        "merge_mode": ["best"] * len(timestamps),
        "quality_score": [0.9] * len(timestamps),
        "conflict_flag": conflict_flags,
        "broker_a_bid": [1950.0 + i * 0.1 for i in range(len(timestamps))],
        "broker_a_ask": [1950.2 + i * 0.1 for i in range(len(timestamps))],
        "broker_b_bid": [1950.0 + i * 0.1 for i in range(len(timestamps))],
        "broker_b_ask": [1950.2 + i * 0.1 for i in range(len(timestamps))],
        "mid_diff": [0.0] * len(timestamps),
        "spread_diff": [0.0] * len(timestamps),
    })


def _merge_diag_row(date: dt.date, symbol: str) -> pl.DataFrame:
    return pl.DataFrame([{
        "time_utc": _ts(date, 0, 0),
        "date": date.isoformat(),
        "symbol": symbol,
        "bucket_ms": 100,
        "bucket_a_only": 2,
        "bucket_b_only": 1,
        "bucket_both": 4,
        "bucket_both_valid": 4,
        "bucket_both_downgraded_to_single": 0,
        "bucket_both_rejected": 0,
        "bucket_invalid_a": 0,
        "bucket_invalid_b": 0,
        "wallclock_overlap_minutes": 4,
        "median_offset_both_ms": 29.0,
        "p95_offset_both_ms": 75.0,
        "near_miss_pairs": 9,
        "validation_reject_count": 0,
        "canonical_rows": 4,
        "canonical_dual_rows": 3,
        "dual_source_ratio": 0.75,
        "conflicts": 1,
    }])


class TestDailyMergeQaReport:
    def test_build_daily_merge_qa_report_aggregates_metrics(self, paths, store) -> None:
        symbol = "XAUUSD"
        date = dt.date(2026, 4, 2)

        raw_a = _raw_ticks(
            "broker_a",
            symbol,
            [_ts(date, 1), _ts(date, 8), _ts(date, 14), _ts(date, 18), _ts(date, 19)],
        )
        raw_b = _raw_ticks(
            "broker_b",
            symbol,
            [_ts(date, 1), _ts(date, 8), _ts(date, 14), _ts(date, 18), _ts(date, 20), _ts(date, 21)],
        )
        canonical = _canonical_ticks(
            symbol,
            [_ts(date, 1), _ts(date, 8), _ts(date, 14), _ts(date, 18)],
        )
        diag = _merge_diag_row(date, symbol)

        store.write(raw_a, paths.raw_ticks_file("broker_a", symbol, date))
        store.write(raw_b, paths.raw_ticks_file("broker_b", symbol, date))
        store.write(canonical, paths.canonical_ticks_file(symbol, date))
        store.write(diag, paths.merge_diagnostics_file(symbol, date))

        report = build_daily_merge_qa_report(
            paths,
            store,
            "broker_a",
            "broker_b",
            symbol,
            date,
            date,
            expected_bucket_ms=100,
        )

        assert report.height == 1
        row = report.row(0, named=True)
        assert row["broker_a_tick_count"] == 5
        assert row["broker_b_tick_count"] == 6
        assert row["canonical_rows"] == 4
        assert row["canonical_dual_rows"] == 3
        assert row["dual_source_ratio"] == 0.75
        assert row["bucket_both"] == 4
        assert row["near_miss_pairs"] == 9
        assert row["validation_reject_count"] == 0
        assert row["asia_rows"] == 1
        assert row["london_rows"] == 2
        assert row["ny_rows"] == 2
        assert row["overlap_rows"] == 1
        assert row["gaps_gt_1m"] == 3
        assert row["max_gap_minutes"] == 420.0
        assert row["conflicts"] == 1

    def test_write_daily_merge_qa_report_persists_per_day(self, paths, store) -> None:
        report = pl.DataFrame([
            {
                "time_utc": _ts(dt.date(2026, 4, 2), 0, 0),
                "date": "2026-04-02",
                "symbol": "XAUUSD",
                "broker_a_id": "broker_a",
                "broker_b_id": "broker_b",
                "bucket_ms": 100,
                "broker_a_tick_count": 10,
                "broker_b_tick_count": 11,
                "canonical_rows": 8,
                "canonical_dual_rows": 7,
                "dual_source_ratio": 0.875,
                "conflicts": 0,
                "bucket_a_only": 1,
                "bucket_b_only": 1,
                "bucket_both": 7,
                "near_miss_pairs": 3,
                "validation_reject_count": 0,
                "asia_rows": 2,
                "london_rows": 4,
                "ny_rows": 3,
                "overlap_rows": 1,
                "gaps_gt_1m": 0,
                "max_gap_minutes": 0.0,
            },
            {
                "time_utc": _ts(dt.date(2026, 4, 3), 0, 0),
                "date": "2026-04-03",
                "symbol": "XAUUSD",
                "broker_a_id": "broker_a",
                "broker_b_id": "broker_b",
                "bucket_ms": 100,
                "broker_a_tick_count": 12,
                "broker_b_tick_count": 13,
                "canonical_rows": 9,
                "canonical_dual_rows": 6,
                "dual_source_ratio": 0.66666667,
                "conflicts": 1,
                "bucket_a_only": 2,
                "bucket_b_only": 1,
                "bucket_both": 6,
                "near_miss_pairs": 4,
                "validation_reject_count": 0,
                "asia_rows": 2,
                "london_rows": 4,
                "ny_rows": 4,
                "overlap_rows": 2,
                "gaps_gt_1m": 1,
                "max_gap_minutes": 2.5,
            },
        ])

        written = write_daily_merge_qa_report(report, paths, store, "XAUUSD")

        assert written == 2
        day_one = store.read_dir(paths.merge_qa_dir("XAUUSD", dt.date(2026, 4, 2)))
        day_two = store.read_dir(paths.merge_qa_dir("XAUUSD", dt.date(2026, 4, 3)))
        assert day_one.height == 1
        assert day_two.height == 1
        assert day_one["canonical_rows"][0] == 8
        assert day_two["canonical_rows"][0] == 9

    def test_assert_synchronized_raw_tick_coverage_fails_on_asymmetric_dates(self, paths, store) -> None:
        symbol = "XAUUSD"
        day_one = dt.date(2026, 4, 2)
        day_two = dt.date(2026, 4, 3)

        store.write(
            _raw_ticks("broker_a", symbol, [_ts(day_one, 1)]),
            paths.raw_ticks_file("broker_a", symbol, day_one),
        )
        store.write(
            _raw_ticks("broker_a", symbol, [_ts(day_two, 2)]),
            paths.raw_ticks_file("broker_a", symbol, day_two),
        )
        store.write(
            _raw_ticks("broker_b", symbol, [_ts(day_two, 2)]),
            paths.raw_ticks_file("broker_b", symbol, day_two),
        )

        with pytest.raises(CoverageParityError, match="broker_b is missing dates present for broker_a: 2026-04-02"):
            assert_synchronized_raw_tick_coverage(
                paths,
                store,
                "broker_a",
                "broker_b",
                symbol,
                day_one,
                day_two,
            )
