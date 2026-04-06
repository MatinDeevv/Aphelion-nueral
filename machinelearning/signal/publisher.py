"""Publisher that turns raw model output tensors into fully sized, auditable trading signals."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from machinelearning.models import ModelOutput

from .conformal import ConformalCalibrator
from .records import SignalRecord
from .sizing import KellyPositionSizer

DEFAULT_REGIME_NAMES = ("trending", "mean_reverting", "volatile", "quiet")


class SignalPublisher:
    """Assemble the single research-to-risk record so downstream execution sees one consistent signal contract.

    Without a publisher, every consumer would re-interpret model tensors differently. Centralizing the translation
    from ModelOutput into SignalRecord keeps conformal calibration, regime labeling, and Kelly sizing consistent
    across paper trading, backtests, and later live routing.
    """

    def __init__(
        self,
        model_artifact_id: str,
        calibrator: ConformalCalibrator,
        sizer: KellyPositionSizer,
    ) -> None:
        self.model_artifact_id = str(model_artifact_id)
        self.calibrator = calibrator
        self.sizer = sizer

    def publish(
        self,
        output: ModelOutput,
        regime_state: Any,
        bar_metadata: dict[str, Any],
        batch_idx: int = 0,
    ) -> SignalRecord:
        """Construct a fully calibrated and sized SignalRecord from one row of model output.

        The resulting record is the exact handoff object a backtest or execution layer can trust: every field is
        scalar, traceable, and already adjusted for conformal uncertainty and position sizing.
        """

        regime_probs = _extract_regime_probs(regime_state, batch_idx=batch_idx)
        regime_names = _extract_regime_names(regime_state, expected=len(regime_probs))
        dominant_index = int(np.argmax(regime_probs))
        regime = regime_names[dominant_index]
        regime_confidence = float(regime_probs[dominant_index])

        direction_probs_60m = _tensor_to_tuple(output.direction_probs("60m")[batch_idx])
        direction_60m = int(np.argmax(direction_probs_60m)) - 1
        return_median_60m = _tensor_to_float(output.return_median("60m")[batch_idx])
        raw_lower_80, raw_upper_80 = output.return_interval("60m", alpha=0.8)
        return_lower_80 = _tensor_to_float(raw_lower_80[batch_idx])
        return_upper_80 = _tensor_to_float(raw_upper_80[batch_idx])
        conformal_lower, conformal_upper = self.calibrator.predict(return_lower_80, return_upper_80)
        conformal_lower_60m = float(np.asarray(conformal_lower).reshape(-1)[0])
        conformal_upper_60m = float(np.asarray(conformal_upper).reshape(-1)[0])

        direction_prob = direction_probs_60m[direction_60m + 1]
        dual_source_ratio = float(bar_metadata.get("dual_source_ratio", 0.0))
        kelly_fraction, position_fraction = self.sizer.size(
            direction=direction_60m,
            direction_prob=direction_prob,
            conformal_lower=conformal_lower_60m,
            conformal_upper=conformal_upper_60m,
            dual_source_ratio=dual_source_ratio,
        )
        signal_strength = self.sizer.signal_strength(
            direction_prob=direction_prob,
            conformal_lower=conformal_lower_60m,
            conformal_upper=conformal_upper_60m,
        )

        return SignalRecord(
            timestamp_utc=_coerce_timestamp(bar_metadata),
            symbol=str(bar_metadata.get("symbol", "XAUUSD")),
            model_artifact_id=self.model_artifact_id,
            regime=regime,
            regime_confidence=regime_confidence,
            direction_60m=direction_60m,
            direction_probs_60m=direction_probs_60m,
            return_median_60m=return_median_60m,
            return_lower_80=return_lower_80,
            return_upper_80=return_upper_80,
            conformal_lower_60m=conformal_lower_60m,
            conformal_upper_60m=conformal_upper_60m,
            conformal_coverage=self.calibrator.empirical_coverage,
            direction_5m=_direction_from_output(output, "5m", batch_idx=batch_idx),
            direction_15m=_direction_from_output(output, "15m", batch_idx=batch_idx),
            direction_240m=_direction_from_output(output, "240m", batch_idx=batch_idx),
            dual_source_ratio=dual_source_ratio,
            disagreement_pressure_bps=float(
                bar_metadata.get("disagreement_pressure_bps", bar_metadata.get("disagreement_bps", 0.0))
            ),
            kelly_fraction=kelly_fraction,
            position_fraction=position_fraction,
            signal_strength=signal_strength,
        )


def _direction_from_output(output: ModelOutput, horizon: str, batch_idx: int) -> int:
    return int(torch.argmax(output.direction_logits[horizon][batch_idx]).item()) - 1


def _tensor_to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _tensor_to_tuple(value: torch.Tensor) -> tuple[float, float, float]:
    array = value.detach().cpu().numpy().astype(float).reshape(-1)
    if array.size != 3:
        raise ValueError("direction probability tensor must contain exactly 3 classes")
    return float(array[0]), float(array[1]), float(array[2])


def _coerce_timestamp(bar_metadata: dict[str, Any]) -> datetime:
    raw = bar_metadata.get("timestamp_utc", bar_metadata.get("timestamp"))
    if raw is None:
        raise KeyError("bar_metadata must include 'timestamp_utc' or 'timestamp'")
    if isinstance(raw, datetime):
        return raw if raw.tzinfo is not None else raw.replace(tzinfo=timezone.utc)
    if isinstance(raw, str):
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    raise TypeError("timestamp metadata must be a datetime or ISO-8601 string")


def _extract_regime_probs(regime_state: Any, batch_idx: int) -> np.ndarray:
    if isinstance(regime_state, dict):
        raw = regime_state.get("regime_probs", regime_state.get("probs", regime_state))
    else:
        raw = getattr(regime_state, "regime_probs", getattr(regime_state, "probs", regime_state))

    if isinstance(raw, torch.Tensor):
        probs = raw.detach().cpu().numpy()
    else:
        probs = np.asarray(raw, dtype=float)

    if probs.ndim == 2:
        selected = probs[batch_idx]
    elif probs.ndim == 1:
        selected = probs
    else:
        raise ValueError("regime probabilities must be a [4] or [B, 4] array")

    if selected.shape[-1] != 4:
        raise ValueError("regime probabilities must contain exactly 4 regime weights")
    return selected.astype(float, copy=False)


def _extract_regime_names(regime_state: Any, expected: int) -> tuple[str, ...]:
    if isinstance(regime_state, dict):
        raw = regime_state.get("regime_names")
    else:
        raw = getattr(regime_state, "regime_names", None)

    if raw is None:
        names = DEFAULT_REGIME_NAMES
    else:
        names = tuple(str(name) for name in raw)

    if len(names) != expected:
        names = tuple(f"regime_{index}" for index in range(expected))
    return names


__all__ = ["SignalPublisher"]
