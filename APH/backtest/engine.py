"""Vectorized backtest engine that evaluates signal quality under a simple but consistent execution rule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from machinelearning.signal import SignalRecord

from .metrics import BacktestMetrics, compute_metrics


@dataclass(slots=True)
class BacktestResult:
    """Container for the full backtest path so downstream reporting can reproduce every headline number."""

    signals: list[SignalRecord]
    realized_returns: np.ndarray
    pnl_series: np.ndarray
    equity_curve: np.ndarray
    trade_mask: np.ndarray
    metrics: BacktestMetrics | None = None
    notional: float = 1000.0
    spread_bps: float = 3.0


class BacktestEngine:
    """Run a vectorized signal-quality backtest so the research layer is judged on PnL, not only logits.

    The Phase 7 backtest is intentionally simple. Its job is not to be a full execution simulator; its job is
    to answer the first trading question: if we followed these calibrated, sized signals with a consistent cost
    model, would the resulting edge survive enough friction to be worth deeper execution research?
    """

    def __init__(
        self,
        spread_bps: float = 3.0,
        notional: float = 1000.0,
    ) -> None:
        if spread_bps < 0.0:
            raise ValueError("spread_bps must be non-negative")
        if notional <= 0.0:
            raise ValueError("notional must be positive")
        self.spread_bps = float(spread_bps)
        self.notional = float(notional)

    def run(
        self,
        signals: list[SignalRecord],
        realized_returns: np.ndarray,
    ) -> BacktestResult:
        """Run the simplified Phase 7 backtest over the full signal history in vectorized form.

        The realized return array supplies the future 60m outcome for each signal. All trade directions, masks,
        and PnL are then computed with NumPy arrays rather than per-bar Python simulation so the evaluation stays
        fast enough to iterate on signal design.
        """

        realized = np.asarray(realized_returns, dtype=float).reshape(-1)
        if len(signals) != realized.size:
            raise ValueError("signals and realized_returns must have the same length")

        n_signals = realized.size
        directions = np.fromiter((signal.direction_60m for signal in signals), dtype=np.int64, count=n_signals)
        positions = np.fromiter((signal.position_fraction for signal in signals), dtype=np.float64, count=n_signals)
        actionable = np.fromiter((signal.is_actionable() for signal in signals), dtype=bool, count=n_signals)

        signed_returns = directions.astype(np.float64) * realized
        net_returns = np.where(actionable, signed_returns - self.spread_bps, 0.0)
        pnl_series = np.where(actionable, net_returns * positions * self.notional, 0.0)
        equity_curve = np.cumsum(pnl_series)

        result = BacktestResult(
            signals=list(signals),
            realized_returns=realized,
            pnl_series=pnl_series,
            equity_curve=equity_curve,
            trade_mask=actionable,
            notional=self.notional,
            spread_bps=self.spread_bps,
        )
        result.metrics = compute_metrics(result)
        return result


__all__ = ["BacktestEngine", "BacktestResult"]
