"""Tests for label generation."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from mt5pipe.features.labels import (
    add_direction_labels,
    add_future_returns,
    add_triple_barrier_labels,
)


@pytest.fixture
def sample_bars() -> pl.DataFrame:
    """100 M1 bars with known prices for testing."""
    n = 100
    base = dt.datetime(2024, 1, 15, 10, 0, 0, tzinfo=dt.timezone.utc)
    prices = [1950.0 + i * 0.1 for i in range(n)]  # Upward trend
    return pl.DataFrame({
        "time_utc": [base + dt.timedelta(minutes=i) for i in range(n)],
        "open": prices,
        "high": [p + 0.2 for p in prices],
        "low": [p - 0.1 for p in prices],
        "close": [p + 0.05 for p in prices],
    })


class TestFutureReturns:
    def test_basic_returns(self, sample_bars: pl.DataFrame) -> None:
        result = add_future_returns(sample_bars, [5, 15])
        assert "future_return_5m" in result.columns
        assert "future_return_15m" in result.columns

    def test_returns_are_null_at_end(self, sample_bars: pl.DataFrame) -> None:
        result = add_future_returns(sample_bars, [5])
        nulls = result["future_return_5m"].is_null()
        # Last 5 rows should be null
        assert nulls[-5:].all()
        assert not nulls[:90].any()

    def test_returns_positive_for_uptrend(self, sample_bars: pl.DataFrame) -> None:
        result = add_future_returns(sample_bars, [5])
        valid = result.filter(~pl.col("future_return_5m").is_null())
        assert (valid["future_return_5m"] > 0).all()


class TestDirectionLabels:
    def test_basic_direction(self, sample_bars: pl.DataFrame) -> None:
        result = add_future_returns(sample_bars, [5])
        result = add_direction_labels(result, [5])
        assert "direction_5m" in result.columns

    def test_uptrend_direction(self, sample_bars: pl.DataFrame) -> None:
        result = add_future_returns(sample_bars, [5])
        result = add_direction_labels(result, [5])
        valid = result.filter(~pl.col("direction_5m").is_null())
        assert (valid["direction_5m"] == 1).all()  # Should be all up


class TestTripleBarrier:
    def test_basic_labels(self, sample_bars: pl.DataFrame) -> None:
        result = add_triple_barrier_labels(
            sample_bars, [10], tp_bps=50.0, sl_bps=50.0
        )
        assert "triple_barrier_10m" in result.columns
        assert "mae_10m" in result.columns
        assert "mfe_10m" in result.columns

    def test_label_values(self, sample_bars: pl.DataFrame) -> None:
        result = add_triple_barrier_labels(sample_bars, [10])
        valid_labels = result["triple_barrier_10m"].drop_nulls().to_list()
        for label in valid_labels:
            assert label in (-1, 0, 1)

    def test_mae_mfe_non_negative(self, sample_bars: pl.DataFrame) -> None:
        result = add_triple_barrier_labels(sample_bars, [10])
        assert (result["mae_10m"].drop_nulls() >= 0).all()
        assert (result["mfe_10m"].drop_nulls() >= 0).all()

    def test_tail_rows_are_null_when_horizon_is_unavailable(self, sample_bars: pl.DataFrame) -> None:
        result = add_triple_barrier_labels(sample_bars, [10])
        assert result["triple_barrier_10m"][-10:].is_null().all()
        assert result["mae_10m"][-10:].is_null().all()
        assert result["mfe_10m"][-10:].is_null().all()

    def test_horizon_is_inclusive_of_the_last_forward_bar(self) -> None:
        n = 8
        base = dt.datetime(2024, 1, 15, 10, 0, 0, tzinfo=dt.timezone.utc)
        frame = pl.DataFrame({
            "time_utc": [base + dt.timedelta(minutes=i) for i in range(n)],
            "open": [100.0] * n,
            "high": [100.0, 100.0, 100.0, 100.0, 100.6, 100.0, 100.0, 100.0],
            "low": [100.0] * n,
            "close": [100.0] * n,
        })
        result = add_triple_barrier_labels(frame, [4], tp_bps=50.0, sl_bps=50.0)
        assert result["triple_barrier_4m"][0] == 1

    def test_flat_prices_get_zero_label(self) -> None:
        n = 50
        base = dt.datetime(2024, 1, 15, 10, 0, 0, tzinfo=dt.timezone.utc)
        flat = pl.DataFrame({
            "time_utc": [base + dt.timedelta(minutes=i) for i in range(n)],
            "open": [1950.0] * n,
            "high": [1950.0] * n,
            "low": [1950.0] * n,
            "close": [1950.0] * n,
        })
        result = add_triple_barrier_labels(flat, [10], tp_bps=50.0, sl_bps=50.0)
        # Flat price should never hit barrier
        valid = result["triple_barrier_10m"].drop_nulls()
        assert (valid == 0).all()
        assert result["triple_barrier_10m"][-10:].is_null().all()

    def test_vol_scaled_barriers(self, sample_bars: pl.DataFrame) -> None:
        """Vol-scaled barriers should give more balanced labels than fixed wide barriers."""
        fixed = add_triple_barrier_labels(sample_bars, [5], tp_bps=50.0, sl_bps=50.0)
        vol = add_triple_barrier_labels(
            sample_bars, [5], tp_bps=50.0, sl_bps=50.0,
            vol_scale_window=20, vol_multiplier=2.0,
        )
        # Both should produce valid label values
        for label in vol["triple_barrier_5m"].drop_nulls().to_list():
            assert label in (-1, 0, 1)
        # Vol-scaled should use the fixed bps as floor, so results are at least
        # as tight as fixed barriers.
        vol_hits = vol.filter(pl.col("triple_barrier_5m") != 0).height
        fixed_hits = fixed.filter(pl.col("triple_barrier_5m") != 0).height
        assert vol_hits >= fixed_hits

    def test_vol_scaled_floor(self) -> None:
        """When vol is near zero, barriers should fall back to fixed bps floor."""
        n = 50
        base = dt.datetime(2024, 1, 15, 10, 0, 0, tzinfo=dt.timezone.utc)
        flat = pl.DataFrame({
            "time_utc": [base + dt.timedelta(minutes=i) for i in range(n)],
            "open": [1950.0] * n,
            "high": [1950.0] * n,
            "low": [1950.0] * n,
            "close": [1950.0] * n,
        })
        result = add_triple_barrier_labels(
            flat, [10], tp_bps=50.0, sl_bps=50.0,
            vol_scale_window=20, vol_multiplier=2.0,
        )
        # Flat prices → vol ~ 0 → falls back to 50bps fixed, still can't hit barrier
        valid = result["triple_barrier_10m"].drop_nulls()
        assert (valid == 0).all()
        assert result["triple_barrier_10m"][-10:].is_null().all()
