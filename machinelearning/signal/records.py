"""Signal records that mark the handoff from research predictions to risk-taking decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class SignalRecord:
    """Immutable contract between the model layer and the execution layer.

    Once a signal crosses this boundary it is no longer only a research artifact; it becomes a risk decision.
    Keeping the record frozen and traceable makes every later PnL number auditable back to the exact model,
    regime context, and sizing rationale that produced it.
    """

    timestamp_utc: datetime
    symbol: str
    model_artifact_id: str
    regime: str
    regime_confidence: float
    direction_60m: int
    direction_probs_60m: tuple[float, float, float]
    return_median_60m: float
    return_lower_80: float
    return_upper_80: float
    conformal_lower_60m: float
    conformal_upper_60m: float
    conformal_coverage: float
    direction_5m: int
    direction_15m: int
    direction_240m: int
    dual_source_ratio: float
    disagreement_pressure_bps: float
    kelly_fraction: float
    position_fraction: float
    signal_strength: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation so signals can be logged, replayed, and audited later."""

        return {
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "symbol": self.symbol,
            "model_artifact_id": self.model_artifact_id,
            "regime": self.regime,
            "regime_confidence": float(self.regime_confidence),
            "direction_60m": int(self.direction_60m),
            "direction_probs_60m": [float(value) for value in self.direction_probs_60m],
            "return_median_60m": float(self.return_median_60m),
            "return_lower_80": float(self.return_lower_80),
            "return_upper_80": float(self.return_upper_80),
            "conformal_lower_60m": float(self.conformal_lower_60m),
            "conformal_upper_60m": float(self.conformal_upper_60m),
            "conformal_coverage": float(self.conformal_coverage),
            "direction_5m": int(self.direction_5m),
            "direction_15m": int(self.direction_15m),
            "direction_240m": int(self.direction_240m),
            "dual_source_ratio": float(self.dual_source_ratio),
            "disagreement_pressure_bps": float(self.disagreement_pressure_bps),
            "kelly_fraction": float(self.kelly_fraction),
            "position_fraction": float(self.position_fraction),
            "signal_strength": float(self.signal_strength),
        }

    def is_actionable(self) -> bool:
        """Gate trades so only confident, sign-consistent, data-quality-clean signals consume risk capital.

        A research signal is not enough on its own. This method enforces the minimum bar-quality, confidence,
        and interval-consistency checks required before the execution layer is allowed to place risk.
        """

        if self.direction_60m == 0:
            return False

        probability = self.direction_probs_60m[self.direction_60m + 1]
        if probability < 0.45:
            return False

        excludes_zero = (
            self.conformal_lower_60m > 0.0 and self.conformal_upper_60m > 0.0
        ) or (
            self.conformal_lower_60m < 0.0 and self.conformal_upper_60m < 0.0
        )
        if not excludes_zero:
            return False

        if self.direction_60m > 0 and self.conformal_lower_60m <= 0.0:
            return False
        if self.direction_60m < 0 and self.conformal_upper_60m >= 0.0:
            return False

        if self.position_fraction < 0.01:
            return False

        if self.dual_source_ratio < 0.10:
            return False

        return True


__all__ = ["SignalRecord"]
