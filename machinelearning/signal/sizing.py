"""Kelly-based sizing that converts interval-aware signals into controlled research positions."""

from __future__ import annotations

import math


class KellyPositionSizer:
    """Size positions conservatively so strong edges matter without letting miscalibration dominate drawdowns.

    Full Kelly is theoretically optimal under perfect probabilities, but live model probabilities are never
    perfect. Fractional Kelly keeps the same edge-aware logic while deliberately reducing exposure so a few bad
    calibration episodes do not destroy the backtest through position-size volatility.
    """

    def __init__(
        self,
        max_kelly: float = 0.25,
        kelly_fraction: float = 0.25,
    ) -> None:
        if max_kelly <= 0.0:
            raise ValueError("max_kelly must be positive")
        if not 0.0 < kelly_fraction <= 1.0:
            raise ValueError("kelly_fraction must be in (0, 1]")
        self.max_kelly = float(max_kelly)
        self.kelly_fraction = float(kelly_fraction)

    def size(
        self,
        direction: int,
        direction_prob: float,
        conformal_lower: float,
        conformal_upper: float,
        dual_source_ratio: float = 1.0,
    ) -> tuple[float, float]:
        """Return the fractional Kelly suggestion and the capped position size actually used.

        The raw Kelly term captures edge, payoff asymmetry, and confidence. The capped position term adds the
        practical risk overlay: no bet when the interval straddles zero, no negative sizes, and smaller exposure
        on lower-quality bars via the dual-source participation ratio.
        """

        if direction not in {-1, 0, 1}:
            raise ValueError("direction must be -1, 0, or 1")
        if direction == 0:
            return 0.0, 0.0

        if conformal_lower <= 0.0 <= conformal_upper:
            return 0.0, 0.0

        if direction > 0:
            if conformal_lower <= 0.0:
                return 0.0, 0.0
            expected_win = max(conformal_upper, 0.0)
            expected_loss = max(abs(conformal_lower), 1e-12)
        else:
            if conformal_upper >= 0.0:
                return 0.0, 0.0
            expected_win = max(abs(conformal_lower), 0.0)
            expected_loss = max(abs(conformal_upper), 1e-12)

        p = min(max(float(direction_prob), 0.0), 1.0)
        q = 1.0 - p
        quality = min(max(float(dual_source_ratio), 0.0), 1.0)
        odds = expected_win / expected_loss if expected_loss > 0.0 else 0.0
        if odds <= 0.0:
            return 0.0, 0.0

        full_kelly = max(((odds * p) - q) / odds, 0.0)
        raw_fraction = full_kelly * self.kelly_fraction * quality
        position_fraction = min(max(raw_fraction, 0.0), self.max_kelly)
        return raw_fraction, position_fraction

    def signal_strength(
        self,
        direction_prob: float,
        conformal_lower: float,
        conformal_upper: float,
    ) -> float:
        """Summarize how usable a signal is by blending confidence with interval separation from zero.

        Confidence alone is not enough when the return interval barely clears zero, and interval magnitude alone
        is not enough when class probabilities are weak. The composite score helps downstream filtering compare
        signals on one interpretable 0..1 scale.
        """

        if conformal_lower <= 0.0 <= conformal_upper:
            return 0.0

        confidence = min(max((float(direction_prob) - 0.45) / 0.55, 0.0), 1.0)
        distance_from_zero = min(abs(conformal_lower), abs(conformal_upper))
        interval_width = max(abs(conformal_upper - conformal_lower), 1e-12)
        clarity = math.tanh(distance_from_zero / interval_width)
        return min(max((0.6 * confidence) + (0.4 * clarity), 0.0), 1.0)


__all__ = ["KellyPositionSizer"]
