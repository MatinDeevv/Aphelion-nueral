"""Tests for the data quality / cleaning pipeline."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from mt5pipe.quality.cleaning import clean_ticks, validate_bars, clean_dataset
from mt5pipe.quality.gaps import detect_gaps, fill_bar_gaps, _is_forex_closed
from mt5pipe.quality.report import dataset_quality_report, format_quality_report


# ---------------------------------------------------------------------------
# Tick cleaning tests
# ---------------------------------------------------------------------------


class TestCleanTicks:
    @staticmethod
    def _make_ticks(n: int = 100, bid_start: float = 1950.0, spread: float = 0.5) -> pl.DataFrame:
        base_msc = 1700000000000
        return pl.DataFrame({
            "time_msc": [base_msc + i * 100 for i in range(n)],
            "bid": [bid_start + i * 0.01 for i in range(n)],
            "ask": [bid_start + spread + i * 0.01 for i in range(n)],
            "last": [0.0] * n,
            "volume": [1.0] * n,
        })

    def test_removes_duplicates(self) -> None:
        df = self._make_ticks(10)
        df = pl.concat([df, df])  # double it
        cleaned, report = clean_ticks(df)
        assert report.duplicates_removed == 10
        assert len(cleaned) == 10

    def test_removes_zero_prices(self) -> None:
        df = self._make_ticks(10)
        # Inject a zero-bid row
        bad = pl.DataFrame({
            "time_msc": [9999999999999],
            "bid": [0.0],
            "ask": [1950.0],
            "last": [0.0],
            "volume": [1.0],
        })
        df = pl.concat([df, bad], how="diagonal_relaxed")
        cleaned, report = clean_ticks(df)
        assert report.invalid_price_removed >= 1

    def test_removes_crossed_quotes(self) -> None:
        df = pl.DataFrame({
            "time_msc": [1000, 2000, 3000],
            "bid": [100.0, 100.0, 100.5],  # 3rd has bid > ask
            "ask": [100.5, 100.5, 100.0],
            "last": [0.0, 0.0, 0.0],
            "volume": [1.0, 1.0, 1.0],
        })
        cleaned, report = clean_ticks(df)
        assert len(cleaned) == 2
        assert report.invalid_price_removed >= 1

    def test_removes_spread_outliers(self) -> None:
        df = pl.DataFrame({
            "time_msc": [1000, 2000, 3000],
            "bid": [100.0, 100.0, 50.0],  # 3rd has 100% spread
            "ask": [100.5, 100.5, 100.0],
            "last": [0.0, 0.0, 0.0],
            "volume": [1.0, 1.0, 1.0],
        })
        cleaned, report = clean_ticks(df, max_spread_ratio=0.01)
        assert report.spread_outliers_removed >= 1

    def test_empty_input(self) -> None:
        df = pl.DataFrame({"time_msc": [], "bid": [], "ask": [], "last": [], "volume": []})
        cleaned, report = clean_ticks(df)
        assert cleaned.is_empty()
        assert report.input_rows == 0

    def test_preserves_valid_ticks(self) -> None:
        df = self._make_ticks(50)
        cleaned, report = clean_ticks(df)
        # All ticks are valid — nothing should be removed (except maybe spike detection edge)
        assert report.output_rows >= 48  # allow tiny margin for rolling-median edge

    def test_report_pct(self) -> None:
        df = self._make_ticks(100)
        bad = pl.DataFrame({
            "time_msc": [i for i in range(10)],
            "bid": [0.0] * 10,
            "ask": [0.0] * 10,
            "last": [0.0] * 10,
            "volume": [1.0] * 10,
        })
        df = pl.concat([df, bad], how="diagonal_relaxed")
        _, report = clean_ticks(df)
        assert report.pct_removed > 0


# ---------------------------------------------------------------------------
# Bar validation tests
# ---------------------------------------------------------------------------


class TestValidateBars:
    def test_fixes_ohlc_consistency(self) -> None:
        """High should be >= open and close, low should be <= open and close."""
        df = pl.DataFrame({
            "time_utc": [dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)],
            "open": [100.0],
            "high": [99.0],   # wrong: lower than open
            "low": [101.0],   # wrong: higher than open
            "close": [100.5],
            "tick_count": [10],
            "spread_mean": [0.5],
            "spread_min": [0.3],
            "spread_max": [0.7],
        })
        result = validate_bars(df)
        assert result["high"][0] >= result["open"][0]
        assert result["high"][0] >= result["close"][0]
        assert result["low"][0] <= result["open"][0]
        assert result["low"][0] <= result["close"][0]

    def test_removes_zero_price_bars(self) -> None:
        df = pl.DataFrame({
            "time_utc": [
                dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
                dt.datetime(2024, 1, 1, 0, 1, tzinfo=dt.timezone.utc),
            ],
            "open": [100.0, 0.0],
            "high": [101.0, 0.0],
            "low": [99.0, 0.0],
            "close": [100.5, 0.0],
            "tick_count": [10, 0],
        })
        result = validate_bars(df)
        assert len(result) == 1

    def test_removes_zero_tick_bars(self) -> None:
        df = pl.DataFrame({
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "tick_count": [0],
        })
        result = validate_bars(df)
        assert len(result) == 0

    def test_empty_input(self) -> None:
        assert validate_bars(pl.DataFrame()).is_empty()


# ---------------------------------------------------------------------------
# Gap detection tests
# ---------------------------------------------------------------------------


class TestDetectGaps:
    def test_no_gaps(self) -> None:
        times = [dt.datetime(2024, 1, 2, 0, i, tzinfo=dt.timezone.utc) for i in range(60)]
        df = pl.DataFrame({"time_utc": times})
        report = detect_gaps(df, "M1", 60, skip_weekends=False)
        assert report.missing_bars == 0

    def test_detects_gap(self) -> None:
        times = [
            dt.datetime(2024, 1, 2, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2024, 1, 2, 0, 1, tzinfo=dt.timezone.utc),
            # gap: 0:02, 0:03 missing
            dt.datetime(2024, 1, 2, 0, 4, tzinfo=dt.timezone.utc),
            dt.datetime(2024, 1, 2, 0, 5, tzinfo=dt.timezone.utc),
        ]
        df = pl.DataFrame({"time_utc": times})
        report = detect_gaps(df, "M1", 60, skip_weekends=False)
        assert report.missing_bars == 2
        assert len(report.gaps) == 1

    def test_single_row(self) -> None:
        df = pl.DataFrame({"time_utc": [dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)]})
        report = detect_gaps(df, "M1", 60)
        assert report.missing_bars == 0

    def test_weekend_not_counted(self) -> None:
        """A gap that spans a forex weekend should not count weekend hours."""
        fri = dt.datetime(2024, 1, 5, 21, 59, tzinfo=dt.timezone.utc)  # Friday
        mon = dt.datetime(2024, 1, 8, 0, 0, tzinfo=dt.timezone.utc)    # Monday
        df = pl.DataFrame({"time_utc": [fri, mon]})
        report = detect_gaps(df, "M1", 60, skip_weekends=True)
        # Weekend bars are excluded, so missing should be much less than raw delta
        assert report.missing_bars < 3000  # raw delta ~3000 min, but weekend portion excluded


class TestFillBarGaps:
    def test_fills_small_gap(self) -> None:
        times = [
            dt.datetime(2024, 1, 2, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2024, 1, 2, 0, 3, tzinfo=dt.timezone.utc),
        ]
        df = pl.DataFrame({
            "time_utc": times,
            "open": [100.0, 101.0],
            "high": [100.5, 101.5],
            "low": [99.5, 100.5],
            "close": [100.2, 101.2],
            "tick_count": [10, 12],
            "volume_sum": [5.0, 6.0],
            "mid_return": [0.001, 0.002],
            "realized_vol": [0.0005, 0.0006],
            "spread_mean": [0.3, 0.4],
            "spread_max": [0.5, 0.6],
            "spread_min": [0.1, 0.2],
            "source_count": [2, 2],
            "conflict_count": [0, 0],
        })
        result = fill_bar_gaps(df, "M1", 60, skip_weekends=False)
        # Should have 4 rows: 0:00, 0:01 (filled), 0:02 (filled), 0:03
        assert len(result) == 4
        assert result.filter(pl.col("_filled"))["_filled"].sum() == 2

    def test_filled_bars_use_previous_close(self) -> None:
        times = [
            dt.datetime(2024, 1, 2, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2024, 1, 2, 0, 2, tzinfo=dt.timezone.utc),
        ]
        df = pl.DataFrame({
            "time_utc": times,
            "open": [100.0, 102.0],
            "high": [100.5, 102.5],
            "low": [99.5, 101.5],
            "close": [100.2, 102.2],
            "tick_count": [10, 12],
            "volume_sum": [5.0, 6.0],
            "mid_return": [0.001, 0.002],
            "realized_vol": [0.0005, 0.0006],
            "spread_mean": [0.3, 0.4],
            "spread_max": [0.5, 0.6],
            "spread_min": [0.1, 0.2],
            "source_count": [2, 2],
            "conflict_count": [0, 0],
        })
        result = fill_bar_gaps(df, "M1", 60, skip_weekends=False)
        filled = result.filter(pl.col("_filled"))
        assert len(filled) == 1
        # Filled bar should have OHLC = previous close (100.2)
        assert filled["open"][0] == 100.2
        assert filled["close"][0] == 100.2
        assert filled["tick_count"][0] == 0

    def test_no_fill_needed(self) -> None:
        times = [
            dt.datetime(2024, 1, 2, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2024, 1, 2, 0, 1, tzinfo=dt.timezone.utc),
        ]
        df = pl.DataFrame({
            "time_utc": times,
            "open": [100.0, 101.0],
            "high": [100.5, 101.5],
            "low": [99.5, 100.5],
            "close": [100.2, 101.2],
            "tick_count": [10, 12],
        })
        result = fill_bar_gaps(df, "M1", 60, skip_weekends=False)
        assert len(result) == 2


class TestForexClosed:
    def test_friday_late(self) -> None:
        assert _is_forex_closed(dt.datetime(2024, 1, 5, 23, 0)) is True  # Friday 23h

    def test_saturday(self) -> None:
        assert _is_forex_closed(dt.datetime(2024, 1, 6, 12, 0)) is True  # Saturday

    def test_sunday_early(self) -> None:
        assert _is_forex_closed(dt.datetime(2024, 1, 7, 10, 0)) is True  # Sunday 10h

    def test_sunday_open(self) -> None:
        assert _is_forex_closed(dt.datetime(2024, 1, 7, 22, 0)) is False  # Sunday 22h = open

    def test_monday(self) -> None:
        assert _is_forex_closed(dt.datetime(2024, 1, 8, 10, 0)) is False  # Monday


# ---------------------------------------------------------------------------
# Dataset cleaning tests
# ---------------------------------------------------------------------------


class TestCleanDataset:
    def test_drops_high_null_columns(self) -> None:
        df = pl.DataFrame({
            "time_utc": [dt.datetime(2024, 1, 1, 0, i, tzinfo=dt.timezone.utc) for i in range(10)],
            "close": [100.0 + i for i in range(10)],
            "bad_col": [None] * 10,  # 100% null
        })
        cleaned, stats = clean_dataset(df, max_null_pct=50.0)
        assert "bad_col" not in cleaned.columns

    def test_drops_constant_columns(self) -> None:
        df = pl.DataFrame({
            "time_utc": [dt.datetime(2024, 1, 1, 0, i, tzinfo=dt.timezone.utc) for i in range(10)],
            "close": [100.0 + i for i in range(10)],
            "const": [5.0] * 10,
        })
        cleaned, stats = clean_dataset(df)
        assert "const" not in cleaned.columns

    def test_replaces_inf(self) -> None:
        df = pl.DataFrame({
            "time_utc": [dt.datetime(2024, 1, 1, 0, i, tzinfo=dt.timezone.utc) for i in range(5)],
            "val": [1.0, 2.0, float("inf"), 4.0, 5.0],
        })
        cleaned, _ = clean_dataset(df)
        # inf should be replaced and forward-filled
        assert cleaned.filter(pl.col("val").is_infinite()).is_empty()

    def test_preserves_structure(self) -> None:
        df = pl.DataFrame({
            "time_utc": [dt.datetime(2024, 1, 1, 0, i, tzinfo=dt.timezone.utc) for i in range(10)],
            "close": [100.0 + i * 0.1 for i in range(10)],
            "volume": [float(i) for i in range(10)],
        })
        cleaned, stats = clean_dataset(df)
        assert stats["input_rows"] == 10
        assert stats["output_rows"] == 10


# ---------------------------------------------------------------------------
# Quality report tests
# ---------------------------------------------------------------------------


class TestQualityReport:
    def test_perfect_data(self) -> None:
        base = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        df = pl.DataFrame({
            "time_utc": [base + dt.timedelta(minutes=i) for i in range(100)],
            "close": [100.0 + i * 0.01 for i in range(100)],
            "mid_return": [0.001 * (i + 1) for i in range(100)],
        })
        report = dataset_quality_report(df)
        assert report["quality_score"] == 100.0
        assert report["null_free"] is True
        assert report["inf_free"] is True
        assert report["duplicate_timestamps"] == 0

    def test_data_with_nulls(self) -> None:
        df = pl.DataFrame({
            "time_utc": [dt.datetime(2024, 1, 1, 0, i, tzinfo=dt.timezone.utc) for i in range(10)],
            "close": [100.0, None, 101.0, None, 102.0, None, 103.0, None, 104.0, None],
        })
        report = dataset_quality_report(df)
        assert report["null_free"] is False
        assert report["quality_score"] < 100.0

    def test_format_report(self) -> None:
        df = pl.DataFrame({
            "time_utc": [dt.datetime(2024, 1, 1, 0, i, tzinfo=dt.timezone.utc) for i in range(10)],
            "close": [100.0 + i for i in range(10)],
        })
        report = dataset_quality_report(df)
        text = format_quality_report(report)
        assert "Quality Score" in text
        assert "Rows: 10" in text

    def test_empty_data(self) -> None:
        report = dataset_quality_report(pl.DataFrame())
        assert report["rows"] == 0
