"""Reporting helpers that turn vectorized backtest output into a trader-readable checkpoint summary."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engine import BacktestResult
from .metrics import BacktestMetrics, compute_metrics

BASELINE_BALANCED_ACCURACY = 0.5354
DEFAULT_REGIME_ORDER = ("trending", "mean_reverting", "volatile", "quiet")


@dataclass(slots=True)
class BacktestReport:
    """Summarize the backtest in a format that makes model research answer a trading question.

    Research logs tend to stop at losses and accuracy, but strategy selection happens on a different axis:
    can the signals survive costs, avoid crippling drawdowns, and beat the published baseline? This report
    packages those answers into one portable object that is easy to print, compare, and serialize.
    """

    period_start: str
    period_end: str
    n_signals: int
    n_actionable: int
    actionable_fraction: float
    metrics: BacktestMetrics
    regime_breakdown: dict[str, float]

    @classmethod
    def from_result(cls, result: BacktestResult) -> "BacktestReport":
        """Build a report from the raw backtest path so summary numbers stay tied to the underlying trades.

        Constructing the report from BacktestResult keeps the human-facing summary anchored to the exact signal
        list, trade mask, and realized outcomes used for the PnL computation, which makes later audits and
        regression comparisons much simpler.
        """

        metrics = result.metrics if result.metrics is not None else compute_metrics(result)
        timestamps = [signal.timestamp_utc for signal in result.signals]
        if timestamps:
            period_start = min(timestamps).isoformat()
            period_end = max(timestamps).isoformat()
        else:
            period_start = "N/A"
            period_end = "N/A"

        n_signals = len(result.signals)
        n_actionable = int(result.trade_mask.sum())
        actionable_fraction = (n_actionable / n_signals) if n_signals else 0.0

        counts = {name: 0 for name in DEFAULT_REGIME_ORDER}
        for signal in result.signals:
            counts[signal.regime] = counts.get(signal.regime, 0) + 1
        denominator = max(n_signals, 1)
        regime_breakdown = {
            name: counts.get(name, 0) / denominator
            for name in tuple(DEFAULT_REGIME_ORDER) + tuple(
                name for name in counts if name not in DEFAULT_REGIME_ORDER
            )
        }

        return cls(
            period_start=period_start,
            period_end=period_end,
            n_signals=n_signals,
            n_actionable=n_actionable,
            actionable_fraction=actionable_fraction,
            metrics=metrics,
            regime_breakdown=regime_breakdown,
        )

    def print_summary(self) -> None:
        """Print a compact baseline-comparable summary so strategy quality is obvious at a glance.

        The output intentionally mirrors the baseline-oriented checkpoint language from the phase brief. That
        makes it easy to compare an end-to-end signal stack against the published gaussian_nb benchmark without
        mentally translating between modeling metrics and trading metrics.
        """

        print("=== Aphelion Phase 7 Backtest Report ===")
        print(f"Period:          {self.period_start} to {self.period_end}")
        print(
            f"Signals:         {self.n_signals} total, {self.n_actionable} actionable "
            f"({self.actionable_fraction * 100.0:.1f}%)",
        )
        print(
            f"Trades:          {self.metrics.n_trades} "
            f"({self.metrics.n_long} long / {self.metrics.n_short} short)",
        )
        print(f"Win rate:        {self.metrics.win_rate * 100.0:.1f}%")
        print(
            f"Avg win/loss:    {self.metrics.avg_win_bps:+.1f} bps / "
            f"{self.metrics.avg_loss_bps:+.1f} bps",
        )
        print(f"Profit factor:   {_format_metric(self.metrics.profit_factor)}")
        print(f"Sharpe ratio:    {_format_metric(self.metrics.sharpe_ratio)} (annualized)")
        print(f"Max drawdown:    {-self.metrics.max_drawdown * 100.0:.1f}%")
        print(f"Calmar ratio:    {_format_metric(self.metrics.calmar_ratio)}")
        print(f"IC (60m):        {self.metrics.information_coefficient:.4f}")
        print(
            f"Balanced acc:    {self.metrics.balanced_accuracy:.4f}  "
            f"[baseline: {BASELINE_BALANCED_ACCURACY:.4f}]",
        )
        print(
            f"Delta baseline:  "
            f"{self.metrics.balanced_accuracy - BASELINE_BALANCED_ACCURACY:+.4f}",
        )
        print()
        print("Regime breakdown:")
        for regime_name in DEFAULT_REGIME_ORDER:
            share = self.regime_breakdown.get(regime_name, 0.0) * 100.0
            print(f"  {regime_name:<15}{share:5.1f}% of signals")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary so backtest checkpoints can be logged and compared mechanically."""

        payload: dict[str, Any] = {
            "period_start": self.period_start,
            "period_end": self.period_end,
            "n_signals": self.n_signals,
            "n_actionable": self.n_actionable,
            "actionable_fraction": _json_safe(self.actionable_fraction),
            "regime_breakdown": {
                regime_name: _json_safe(value)
                for regime_name, value in self.regime_breakdown.items()
            },
        }
        payload.update(
            {
                "n_trades": self.metrics.n_trades,
                "n_long": self.metrics.n_long,
                "n_short": self.metrics.n_short,
                "win_rate": _json_safe(self.metrics.win_rate),
                "avg_win_bps": _json_safe(self.metrics.avg_win_bps),
                "avg_loss_bps": _json_safe(self.metrics.avg_loss_bps),
                "profit_factor": _json_safe(self.metrics.profit_factor),
                "sharpe_ratio": _json_safe(self.metrics.sharpe_ratio),
                "max_drawdown": _json_safe(self.metrics.max_drawdown),
                "calmar_ratio": _json_safe(self.metrics.calmar_ratio),
                "information_coefficient": _json_safe(self.metrics.information_coefficient),
                "balanced_accuracy": _json_safe(self.metrics.balanced_accuracy),
                "avg_position_fraction": _json_safe(self.metrics.avg_position_fraction),
                "avg_signal_strength": _json_safe(self.metrics.avg_signal_strength),
                "baseline_balanced_accuracy": BASELINE_BALANCED_ACCURACY,
                "delta_baseline": _json_safe(
                    self.metrics.balanced_accuracy - BASELINE_BALANCED_ACCURACY,
                ),
            },
        )
        return payload

    def to_json(self, path: Path) -> None:
        """Persist the summary as JSON so research checkpoints can be diffed and archived automatically."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _format_metric(value: float) -> str:
    if not math.isfinite(value):
        return "inf" if value > 0.0 else "-inf"
    return f"{value:.2f}"


def _json_safe(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


__all__ = ["BacktestReport"]
