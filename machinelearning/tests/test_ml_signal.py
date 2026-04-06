"""Synthetic contract tests for the Phase 7 signal-calibration and position-sizing stack."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT.parent))

from machinelearning.models import ModelOutput
from machinelearning.regime import RegimeState
from machinelearning.signal import (
    ConformalCalibrator,
    KellyPositionSizer,
    SignalPublisher,
    SignalRecord,
)


def test_conformal_calibrator_coverage() -> None:
    rng = np.random.default_rng(7)
    center = rng.normal(0.0, 1.0, size=1_000)
    width = rng.uniform(0.05, 0.25, size=1_000)
    lower = center - width
    upper = center + width
    actual = center + rng.normal(0.0, 0.30, size=1_000)

    calibrator = ConformalCalibrator(alpha=0.1).calibrate(lower, upper, actual)

    assert calibrator.empirical_coverage >= 0.9


def test_conformal_calibrator_q_hat_positive() -> None:
    calibrator = _calibrator()
    assert calibrator.q_hat >= 0.0


def test_conformal_intervals_wider_than_raw() -> None:
    calibrator = _calibrator()
    predicted_lower = np.array([-0.2, 0.1], dtype=float)
    predicted_upper = np.array([0.3, 0.8], dtype=float)

    conformal_lower, conformal_upper = calibrator.predict(predicted_lower, predicted_upper)

    assert np.all(conformal_lower <= predicted_lower)
    assert np.all(conformal_upper >= predicted_upper)


def test_conformal_save_load_roundtrip(tmp_path: Path) -> None:
    calibrator = _calibrator()
    save_path = tmp_path / "conformal.json"

    calibrator.save(save_path)
    loaded = ConformalCalibrator.load(save_path)

    assert loaded.alpha == calibrator.alpha
    assert loaded.q_hat == calibrator.q_hat
    assert loaded.empirical_coverage == calibrator.empirical_coverage


def test_kelly_sizer_zero_on_straddle() -> None:
    sizer = KellyPositionSizer()
    raw_fraction, position_fraction = sizer.size(
        direction=1,
        direction_prob=0.70,
        conformal_lower=-0.10,
        conformal_upper=0.20,
    )

    assert raw_fraction == 0.0
    assert position_fraction == 0.0


def test_kelly_sizer_capped_at_max_kelly() -> None:
    sizer = KellyPositionSizer(max_kelly=0.25, kelly_fraction=1.0)
    raw_fraction, position_fraction = sizer.size(
        direction=1,
        direction_prob=0.99,
        conformal_lower=0.10,
        conformal_upper=10.0,
    )

    assert raw_fraction > 0.25
    assert position_fraction == 0.25


def test_kelly_sizer_scales_with_quality() -> None:
    sizer = KellyPositionSizer(max_kelly=1.0, kelly_fraction=1.0)
    _, low_quality_position = sizer.size(
        direction=1,
        direction_prob=0.80,
        conformal_lower=0.10,
        conformal_upper=2.00,
        dual_source_ratio=0.10,
    )
    _, high_quality_position = sizer.size(
        direction=1,
        direction_prob=0.80,
        conformal_lower=0.10,
        conformal_upper=2.00,
        dual_source_ratio=1.00,
    )

    assert low_quality_position < high_quality_position


def test_signal_record_is_actionable_flat() -> None:
    signal = _signal_record(direction_60m=0, position_fraction=0.10)
    assert signal.is_actionable() is False


def test_signal_record_is_actionable_low_confidence() -> None:
    signal = _signal_record(
        direction_60m=0,
        direction_probs_60m=(0.33, 0.34, 0.33),
        position_fraction=0.10,
    )
    assert signal.is_actionable() is False


def test_signal_publisher_assembles_record() -> None:
    output = _model_output(batch_size=1)
    regime_state = RegimeState.from_probs(torch.tensor([0.1, 0.7, 0.1, 0.1], dtype=torch.float32))
    calibrator = _calibrator()
    sizer = KellyPositionSizer()
    publisher = SignalPublisher(
        model_artifact_id="model.aphelion.test",
        calibrator=calibrator,
        sizer=sizer,
    )

    record = publisher.publish(
        output=output,
        regime_state=regime_state,
        bar_metadata={
            "timestamp_utc": datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc),
            "symbol": "XAUUSD",
            "dual_source_ratio": 0.8,
            "disagreement_pressure_bps": 1.5,
        },
    )

    assert isinstance(record, SignalRecord)
    assert record.symbol == "XAUUSD"
    assert record.model_artifact_id == "model.aphelion.test"
    assert record.regime == "mean_reverting"
    assert record.direction_60m == 1
    assert record.conformal_lower_60m <= record.return_lower_80
    assert record.conformal_upper_60m >= record.return_upper_80
    assert record.position_fraction >= 0.0
    assert record.to_dict()["timestamp_utc"].endswith("+00:00")
    json.dumps(record.to_dict())


def _calibrator() -> ConformalCalibrator:
    predicted_lower = np.array([-0.25, -0.10, 0.05, 0.10, 0.20, -0.05], dtype=float)
    predicted_upper = np.array([0.20, 0.15, 0.35, 0.40, 0.55, 0.25], dtype=float)
    actual_returns = np.array([-0.10, 0.30, 0.10, 0.15, 0.50, -0.15], dtype=float)
    return ConformalCalibrator(alpha=0.1).calibrate(
        predicted_lower=predicted_lower,
        predicted_upper=predicted_upper,
        actual_returns=actual_returns,
    )


def _model_output(batch_size: int) -> ModelOutput:
    direction_logits = {
        "5m": torch.tensor([[3.0, 0.0, -2.0]], dtype=torch.float32).repeat(batch_size, 1),
        "15m": torch.tensor([[0.5, 2.0, -1.0]], dtype=torch.float32).repeat(batch_size, 1),
        "60m": torch.tensor([[-2.0, -0.5, 3.5]], dtype=torch.float32).repeat(batch_size, 1),
        "240m": torch.tensor([[-1.0, 2.5, 0.1]], dtype=torch.float32).repeat(batch_size, 1),
    }
    tb_logits = {
        horizon: torch.zeros(batch_size, 3, dtype=torch.float32)
        for horizon in ("5m", "15m", "60m", "240m")
    }
    return_preds = {
        "5m": torch.tensor([[0.01, 0.02, 0.03, 0.04, 0.05]], dtype=torch.float32).repeat(batch_size, 1),
        "15m": torch.tensor([[0.02, 0.03, 0.04, 0.05, 0.06]], dtype=torch.float32).repeat(batch_size, 1),
        "60m": torch.tensor([[0.08, 0.12, 0.18, 0.24, 0.30]], dtype=torch.float32).repeat(batch_size, 1),
        "240m": torch.tensor([[0.00, 0.01, 0.02, 0.03, 0.04]], dtype=torch.float32).repeat(batch_size, 1),
    }
    mae_preds = {
        horizon: torch.full((batch_size,), 0.10, dtype=torch.float32)
        for horizon in ("5m", "15m", "60m", "240m")
    }
    mfe_preds = {
        horizon: torch.full((batch_size,), 0.20, dtype=torch.float32)
        for horizon in ("5m", "15m", "60m", "240m")
    }
    return ModelOutput(
        direction_logits=direction_logits,
        tb_logits=tb_logits,
        return_preds=return_preds,
        mae_preds=mae_preds,
        mfe_preds=mfe_preds,
        encoder_hidden=torch.zeros(batch_size, 4, dtype=torch.float32),
    )


def _signal_record(
    *,
    direction_60m: int = 1,
    direction_probs_60m: tuple[float, float, float] = (0.10, 0.15, 0.75),
    position_fraction: float = 0.12,
) -> SignalRecord:
    return SignalRecord(
        timestamp_utc=datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc),
        symbol="XAUUSD",
        model_artifact_id="model.test",
        regime="trending",
        regime_confidence=0.8,
        direction_60m=direction_60m,
        direction_probs_60m=direction_probs_60m,
        return_median_60m=0.2,
        return_lower_80=0.1,
        return_upper_80=0.3,
        conformal_lower_60m=0.08,
        conformal_upper_60m=0.32,
        conformal_coverage=0.95,
        direction_5m=1,
        direction_15m=1,
        direction_240m=1,
        dual_source_ratio=0.9,
        disagreement_pressure_bps=1.0,
        kelly_fraction=position_fraction,
        position_fraction=position_fraction,
        signal_strength=0.7,
    )
