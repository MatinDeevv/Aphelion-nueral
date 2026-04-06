"""Backtest metrics that answer whether the signal layer creates tradable edge rather than pretty predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .engine import BacktestResult


@dataclass(slots=True)
class BacktestMetrics:
    """Compact summary of strategy quality so model research is judged on money-making relevance."""

    n_trades: int
    n_long: int
    n_short: int
    win_rate: float
    avg_win_bps: float
    avg_loss_bps: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    information_coefficient: float
    balanced_accuracy: float
    avg_position_fraction: float
    avg_signal_strength: float


def compute_metrics(result: "BacktestResult") -> BacktestMetrics:
    """Compute the strategy metrics that matter once model output is turned into executable signals.

    The point of Phase 7 is not simply to maximize raw prediction accuracy. These metrics ask whether the
    predicted edge survives sizing, spread costs, and directional decision rules in a way that improves on the
    published gaussian_nb baseline.
    """

    n_signals = len(result.signals)
    if n_signals == 0:
        return BacktestMetrics(
            n_trades=0,
            n_long=0,
            n_short=0,
            win_rate=0.0,
            avg_win_bps=0.0,
            avg_loss_bps=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            information_coefficient=0.0,
            balanced_accuracy=0.0,
            avg_position_fraction=0.0,
            avg_signal_strength=0.0,
        )

    directions = np.fromiter((signal.direction_60m for signal in result.signals), dtype=np.int64, count=n_signals)
    positions = np.fromiter((signal.position_fraction for signal in result.signals), dtype=np.float64, count=n_signals)
    strengths = np.fromiter((signal.signal_strength for signal in result.signals), dtype=np.float64, count=n_signals)
    predicted_returns = np.fromiter(
        (signal.return_median_60m for signal in result.signals),
        dtype=np.float64,
        count=n_signals,
    )
    trade_mask = result.trade_mask.astype(bool, copy=False)

    n_trades = int(trade_mask.sum())
    n_long = int(np.sum(trade_mask & (directions > 0)))
    n_short = int(np.sum(trade_mask & (directions < 0)))

    net_return_bps = np.zeros(n_signals, dtype=np.float64)
    valid_position = trade_mask & (positions > 0.0)
    net_return_bps[valid_position] = (
        result.pnl_series[valid_position] / (result.notional * positions[valid_position])
    )

    winning_mask = trade_mask & (result.pnl_series > 0.0)
    losing_mask = trade_mask & (result.pnl_series < 0.0)
    win_rate = float(np.mean(result.pnl_series[trade_mask] > 0.0)) if n_trades else 0.0
    avg_win_bps = float(np.mean(net_return_bps[winning_mask])) if np.any(winning_mask) else 0.0
    avg_loss_bps = float(np.mean(net_return_bps[losing_mask])) if np.any(losing_mask) else 0.0

    total_wins = float(np.sum(result.pnl_series[result.pnl_series > 0.0]))
    total_losses = float(abs(np.sum(result.pnl_series[result.pnl_series < 0.0])))
    if total_losses > 0.0:
        profit_factor = total_wins / total_losses
    elif total_wins > 0.0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    trade_returns = result.pnl_series[trade_mask] / result.notional if n_trades else np.array([], dtype=np.float64)
    if trade_returns.size >= 2 and float(np.std(trade_returns, ddof=1)) > 0.0:
        sharpe_ratio = float(np.sqrt(252.0) * np.mean(trade_returns) / np.std(trade_returns, ddof=1))
    else:
        sharpe_ratio = 0.0

    cumulative_returns = result.equity_curve / result.notional
    if cumulative_returns.size:
        peaks = np.maximum.accumulate(cumulative_returns)
        drawdowns = peaks - cumulative_returns
        max_drawdown = float(np.max(drawdowns))
    else:
        max_drawdown = 0.0

    annualized_return = float(np.mean(trade_returns) * 252.0) if trade_returns.size else 0.0
    if max_drawdown > 0.0:
        calmar_ratio = annualized_return / max_drawdown
    elif annualized_return > 0.0:
        calmar_ratio = float("inf")
    else:
        calmar_ratio = 0.0

    if n_signals >= 2 and np.std(predicted_returns) > 0.0 and np.std(result.realized_returns) > 0.0:
        information_coefficient = float(np.corrcoef(predicted_returns, result.realized_returns)[0, 1])
    else:
        information_coefficient = 0.0

    executed_directions = np.where(trade_mask, directions, 0)
    actual_directions = np.where(
        result.realized_returns > 0.0,
        1,
        np.where(result.realized_returns < 0.0, -1, 0),
    )
    per_class_accuracy = []
    for label in (-1, 0, 1):
        class_mask = actual_directions == label
        if np.any(class_mask):
            per_class_accuracy.append(float(np.mean(executed_directions[class_mask] == label)))
    balanced_accuracy = float(np.mean(per_class_accuracy)) if per_class_accuracy else 0.0

    return BacktestMetrics(
        n_trades=n_trades,
        n_long=n_long,
        n_short=n_short,
        win_rate=win_rate,
        avg_win_bps=avg_win_bps,
        avg_loss_bps=avg_loss_bps,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
        information_coefficient=information_coefficient,
        balanced_accuracy=balanced_accuracy,
        avg_position_fraction=float(np.mean(positions)) if positions.size else 0.0,
        avg_signal_strength=float(np.mean(strengths)) if strengths.size else 0.0,
    )


__all__ = ["BacktestMetrics", "compute_metrics"]
