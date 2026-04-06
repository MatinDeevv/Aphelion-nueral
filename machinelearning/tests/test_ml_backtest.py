"""Synthetic tests for the Phase 7 backtest stack."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT.parent))

from APH.backtest import BacktestEngine
from APH.backtest.report import BacktestReport
from machinelearning.signal import SignalRecord


def test_backtest_engine_no_trades_when_all_flat() -> None:
    engine = BacktestEngine()
    signals = [_signal(index, direction=0, position_fraction=0.0) for index in range(5)]
    realized_returns = np.array([3.0, -2.0, 1.0, 0.0, 4.0], dtype=float)

    result = engine.run(signals, realized_returns)

    assert result.metrics is not None
    assert result.metrics.n_trades == 0
    assert np.all(result.equity_curve == 0.0)


def test_backtest_engine_positive_pnl_on_correct_signals() -> None:
    engine = BacktestEngine(spread_bps=3.0, notional=1_000.0)
    signals = [_signal(index, direction=1, position_fraction=1.0) for index in range(3)]
    realized_returns = np.array([8.0, 10.0, 7.0], dtype=float)

    result = engine.run(signals, realized_returns)

    assert float(result.pnl_series.sum()) > 0.0


def test_backtest_engine_spread_cost_applied() -> None:
    engine = BacktestEngine(spread_bps=3.0, notional=1_000.0)
    signals = [_signal(0, direction=1, position_fraction=1.0)]
    realized_returns = np.array([10.0], dtype=float)

    result = engine.run(signals, realized_returns)

    assert result.pnl_series[0] == 7_000.0


def test_backtest_metrics_sharpe_finite() -> None:
    engine = BacktestEngine(spread_bps=1.0, notional=1_000.0)
    signals = [
        _signal(0, direction=1, position_fraction=1.0),
        _signal(1, direction=1, position_fraction=1.0),
        _signal(2, direction=-1, position_fraction=1.0),
        _signal(3, direction=1, position_fraction=1.0),
    ]
    realized_returns = np.array([5.0, 2.0, -4.0, 1.0], dtype=float)

    result = engine.run(signals, realized_returns)

    assert result.metrics is not None
    assert np.isfinite(result.metrics.sharpe_ratio)


def test_backtest_metrics_balanced_acc_range() -> None:
    engine = BacktestEngine()
    signals = [
        _signal(0, direction=1, position_fraction=1.0),
        _signal(1, direction=-1, position_fraction=1.0, lower=-0.4, upper=-0.1),
        _signal(2, direction=0, position_fraction=0.0, probs=(0.2, 0.6, 0.2)),
    ]
    realized_returns = np.array([5.0, -3.0, 0.0], dtype=float)

    result = engine.run(signals, realized_returns)

    assert result.metrics is not None
    assert 0.0 <= result.metrics.balanced_accuracy <= 1.0


def test_backtest_report_prints_without_error(capsys) -> None:
    engine = BacktestEngine()
    signals = [_signal(index, direction=1, position_fraction=1.0) for index in range(2)]
    realized_returns = np.array([6.0, 5.0], dtype=float)
    result = engine.run(signals, realized_returns)

    report = BacktestReport.from_result(result)
    report.print_summary()

    captured = capsys.readouterr()
    assert "Aphelion Phase 7 Backtest Report" in captured.out


def test_backtest_report_to_dict_has_required_keys() -> None:
    engine = BacktestEngine()
    signals = [_signal(index, direction=1, position_fraction=1.0) for index in range(2)]
    realized_returns = np.array([6.0, 5.0], dtype=float)
    result = engine.run(signals, realized_returns)

    payload = BacktestReport.from_result(result).to_dict()

    for key in (
        "n_trades",
        "win_rate",
        "sharpe_ratio",
        "max_drawdown",
        "balanced_accuracy",
        "information_coefficient",
    ):
        assert key in payload


def _signal(
    index: int,
    *,
    direction: int,
    position_fraction: float,
    probs: tuple[float, float, float] | None = None,
    lower: float = 0.1,
    upper: float = 0.4,
) -> SignalRecord:
    if probs is None:
        probs = (0.80, 0.10, 0.10) if direction == -1 else (0.10, 0.10, 0.80)
    timestamp = datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc) + timedelta(minutes=index)
    return SignalRecord(
        timestamp_utc=timestamp,
        symbol="XAUUSD",
        model_artifact_id="model.test",
        regime="trending",
        regime_confidence=0.7,
        direction_60m=direction,
        direction_probs_60m=probs,
        return_median_60m=0.2 * direction,
        return_lower_80=lower,
        return_upper_80=upper,
        conformal_lower_60m=lower,
        conformal_upper_60m=upper,
        conformal_coverage=0.92,
        direction_5m=direction,
        direction_15m=direction,
        direction_240m=direction,
        dual_source_ratio=1.0,
        disagreement_pressure_bps=1.0,
        kelly_fraction=position_fraction,
        position_fraction=position_fraction,
        signal_strength=0.8 if direction != 0 else 0.0,
    )
