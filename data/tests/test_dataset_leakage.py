"""Tests for dataset leakage prevention."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from mt5pipe.features.context import add_lagged_bar_features
from mt5pipe.features.dataset import walk_forward_splits


class TestAsofJoinNoLeakage:
    """Verify that higher-timeframe features come only from bars that closed BEFORE the base bar."""

    def test_htf_bar_precedes_base(self) -> None:
        """At 10:05, the M5 bar starting at 10:05 hasn't closed yet.
        With bar_duration_seconds=300 the join should pick the 10:00 bar."""
        base = pl.DataFrame({
            "time_utc": [
                dt.datetime(2024, 1, 15, 10, i, tzinfo=dt.timezone.utc)
                for i in range(15)
            ],
            "close": [1950.0 + i * 0.01 for i in range(15)],
        })

        # M5 bars at 10:00 and 10:05
        htf = pl.DataFrame({
            "time_utc": [
                dt.datetime(2024, 1, 15, 10, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2024, 1, 15, 10, 5, tzinfo=dt.timezone.utc),
            ],
            "close": [1900.0, 2000.0],
            "open": [1899.0, 1999.0],
            "high": [1901.0, 2001.0],
            "low": [1898.0, 1998.0],
        })

        result = add_lagged_bar_features(base, htf, "M5", bar_duration_seconds=300)

        # At 10:05 through 10:09, the 10:05 M5 bar hasn't closed yet (closes at 10:09:59).
        # The join should still return the 10:00 bar (close=1900).
        for minute in range(5, 10):
            row = result.filter(
                pl.col("time_utc") == dt.datetime(2024, 1, 15, 10, minute, tzinfo=dt.timezone.utc)
            ).row(0, named=True)
            assert row["M5_close"] == 1900.0, (
                f"At 10:{minute:02d}, expected M5_close=1900 (10:00 bar), "
                f"got {row['M5_close']} — HTF leakage!"
            )

        # At 10:10+, the 10:05 bar has closed, so it should now be visible.
        row_10 = result.filter(
            pl.col("time_utc") == dt.datetime(2024, 1, 15, 10, 10, tzinfo=dt.timezone.utc)
        ).row(0, named=True)
        assert row_10["M5_close"] == 2000.0

    def test_h1_shift(self) -> None:
        """H1 bar at 09:00 should only be visible from 10:00 onwards."""
        start = dt.datetime(2024, 1, 15, 9, 0, tzinfo=dt.timezone.utc)
        base = pl.DataFrame({
            "time_utc": [
                start + dt.timedelta(minutes=i)
                for i in range(120)  # 09:00–10:59
            ],
            "close": [1950.0] * 120,
        })

        htf = pl.DataFrame({
            "time_utc": [
                dt.datetime(2024, 1, 15, 9, 0, tzinfo=dt.timezone.utc),
                dt.datetime(2024, 1, 15, 10, 0, tzinfo=dt.timezone.utc),
            ],
            "close": [1900.0, 2000.0],
            "open": [1899.0, 1999.0],
            "high": [1901.0, 2001.0],
            "low": [1898.0, 1998.0],
        })

        result = add_lagged_bar_features(base, htf, "H1", bar_duration_seconds=3600)

        # At 09:30, the 09:00 H1 bar hasn't closed — no H1 context available
        row_930 = result.filter(
            pl.col("time_utc") == dt.datetime(2024, 1, 15, 9, 30, tzinfo=dt.timezone.utc)
        ).row(0, named=True)
        assert row_930["H1_close"] is None, "H1 bar at 09:00 should NOT be visible at 09:30"

        # At 10:00, the 09:00 H1 bar has just closed — now visible
        row_1000 = result.filter(
            pl.col("time_utc") == dt.datetime(2024, 1, 15, 10, 0, tzinfo=dt.timezone.utc)
        ).row(0, named=True)
        assert row_1000["H1_close"] == 1900.0

    def test_no_htf_data_for_very_early_rows(self) -> None:
        base = pl.DataFrame({
            "time_utc": [
                dt.datetime(2024, 1, 15, 8, i, tzinfo=dt.timezone.utc)
                for i in range(10)
            ],
            "close": [1950.0] * 10,
        })

        # H1 bar only at 10:00
        htf = pl.DataFrame({
            "time_utc": [dt.datetime(2024, 1, 15, 10, 0, tzinfo=dt.timezone.utc)],
            "close": [1950.0],
        })

        result = add_lagged_bar_features(base, htf, "H1", bar_duration_seconds=3600)
        # All base rows are at 08:xx, H1 bar at 10:00 closes at 11:00 — far in the future
        assert result["H1_close"].is_null().all()


class TestWalkForwardSplits:
    def test_train_precedes_test(self) -> None:
        n = 1000
        df = pl.DataFrame({
            "time_utc": [
                dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(minutes=i)
                for i in range(n)
            ],
            "close": list(range(n)),
        })

        splits = walk_forward_splits(df, n_splits=3)
        assert len(splits) > 0

        for train, test in splits:
            assert len(train) > 0
            assert len(test) > 0
            train_max = train["time_utc"].max()
            test_min = test["time_utc"].min()
            assert train_max < test_min, "Train data must precede test data"

    def test_no_overlap(self) -> None:
        n = 500
        df = pl.DataFrame({
            "time_utc": list(range(n)),
            "close": list(range(n)),
        })
        splits = walk_forward_splits(df, n_splits=3, gap_pct=0.05)

        for train, test in splits:
            train_indices = set(train["time_utc"].to_list())
            test_indices = set(test["time_utc"].to_list())
            overlap = train_indices & test_indices
            assert len(overlap) == 0, f"Train and test overlap on {len(overlap)} rows"
