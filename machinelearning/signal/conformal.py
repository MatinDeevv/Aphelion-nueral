"""Split conformal prediction for turning neural quantile outputs into coverage-controlled intervals."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


class ConformalCalibrator:
    """Expand raw model intervals until they satisfy a distribution-free coverage guarantee.

    Neural quantile heads often look calibrated in-sample yet become too narrow in volatile regimes. Split
    conformal prediction repairs that by learning one residual expansion on a held-out calibration split, so the
    downstream sizing layer can rely on intervals that are conservative in a formally justified way.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = float(alpha)
        self._q_hat: float | None = None
        self._empirical_coverage: float | None = None
        self._n_calibration: int = 0

    def calibrate(
        self,
        predicted_lower: np.ndarray,
        predicted_upper: np.ndarray,
        actual_returns: np.ndarray,
    ) -> "ConformalCalibrator":
        """Fit a split-conformal expansion so future intervals cover at least ``1 - alpha`` in aggregate.

        The nonconformity score is the amount by which the true return falls outside the raw interval.
        Taking the finite-sample conformal quantile of those scores guarantees the expanded interval is at least
        as wide as necessary on exchangeable future samples from the same distribution.
        """

        lower = np.asarray(predicted_lower, dtype=float).reshape(-1)
        upper = np.asarray(predicted_upper, dtype=float).reshape(-1)
        actual = np.asarray(actual_returns, dtype=float).reshape(-1)
        if not (lower.size == upper.size == actual.size):
            raise ValueError("predicted_lower, predicted_upper, and actual_returns must have the same length")

        valid = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(actual)
        if not np.any(valid):
            raise ValueError("Calibration requires at least one finite sample")

        lower = lower[valid]
        upper = upper[valid]
        actual = actual[valid]
        scores = np.maximum.reduce([lower - actual, actual - upper, np.zeros_like(actual)])

        n = scores.size
        rank = int(math.ceil((n + 1) * (1.0 - self.alpha)))
        rank = min(max(rank, 1), n)
        self._q_hat = float(np.partition(scores, rank - 1)[rank - 1])
        self._n_calibration = int(n)

        expanded_lower = lower - self._q_hat
        expanded_upper = upper + self._q_hat
        covered = (actual >= expanded_lower) & (actual <= expanded_upper)
        self._empirical_coverage = float(np.mean(covered))
        return self

    def predict(
        self,
        predicted_lower: float | np.ndarray,
        predicted_upper: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return conformal intervals by symmetrically expanding the raw lower and upper quantile bounds."""

        q_hat = self.q_hat
        lower = np.asarray(predicted_lower, dtype=float)
        upper = np.asarray(predicted_upper, dtype=float)
        return lower - q_hat, upper + q_hat

    @property
    def q_hat(self) -> float:
        """Return the learned conformal expansion so downstream code cannot use an uncalibrated interval by accident."""

        if self._q_hat is None:
            raise RuntimeError("ConformalCalibrator has not been calibrated yet")
        return self._q_hat

    @property
    def empirical_coverage(self) -> float:
        """Return the observed calibration-set coverage after conformal expansion."""

        if self._empirical_coverage is None:
            raise RuntimeError("ConformalCalibrator has not been calibrated yet")
        return self._empirical_coverage

    def save(self, path: Path) -> None:
        """Persist calibration state so inference uses the exact same coverage correction as research."""

        if self._q_hat is None or self._empirical_coverage is None:
            raise RuntimeError("ConformalCalibrator must be calibrated before it can be saved")

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha": self.alpha,
            "q_hat": self._q_hat,
            "empirical_coverage": self._empirical_coverage,
            "n_calibration": self._n_calibration,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ConformalCalibrator":
        """Reload a calibrator exactly so paper-trading and backtests use the same interval correction."""

        payload = json.loads(path.read_text(encoding="utf-8"))
        calibrator = cls(alpha=float(payload["alpha"]))
        calibrator._q_hat = float(payload["q_hat"])
        calibrator._empirical_coverage = float(payload["empirical_coverage"])
        calibrator._n_calibration = int(payload.get("n_calibration", 0))
        return calibrator


__all__ = ["ConformalCalibrator"]
