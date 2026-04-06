"""CPU-only contract tests for the Phase 7 regime detection and MoE stack."""

from __future__ import annotations

import datetime as dt
import math
import shutil
import sys
import uuid
from pathlib import Path

import polars as pl
import pytest
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT.parent))

TEST_TMP_ROOT = WORKSPACE_ROOT / "_test_tmp"

from machinelearning.data.schema import TIME_INDEX_COLUMN
from machinelearning.regime import GatingNetwork, MixtureOfExperts, RegimeDetector, RegimeLabeler, RegimeState
from machinelearning.regime.features import RegimeFeatureExtractor


@pytest.fixture
def workspace_tmp_path() -> Path:
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEST_TMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_regime_state_from_probs() -> None:
    probs = torch.tensor([0.1, 0.6, 0.2, 0.1], dtype=torch.float32)
    state = RegimeState.from_probs(probs)

    assert state.dominant == "mean_reverting"
    assert abs(state.confidence - 0.6) < 1e-5
    assert abs(sum(state.probs.tolist()) - 1.0) < 1e-5


def test_regime_state_names_correct() -> None:
    assert RegimeState.NAMES == ["trending", "mean_reverting", "volatile", "quiet"]


def test_regime_feature_extractor_shape() -> None:
    extractor = RegimeFeatureExtractor()
    frame = _regime_frame(rows=100, include_all=True)

    array = extractor.extract(frame)

    assert array.shape == (100, 13)
    assert array.dtype.name == "float32"


def test_regime_feature_extractor_missing_cols() -> None:
    extractor = RegimeFeatureExtractor()
    frame = _regime_frame(rows=12, include_all=False)

    array = extractor.extract(frame)

    assert array.shape == (12, 13)
    assert torch.isfinite(torch.from_numpy(array)).all()
    missing_columns = extractor.REGIME_FEATURE_COLS[len(extractor.REGIME_FEATURE_COLS) // 2 :]
    for column in missing_columns:
        column_index = extractor.REGIME_FEATURE_COLS.index(column)
        assert float(array[:, column_index].sum()) == 0.0


def test_regime_detector_unfitted_returns_uniform() -> None:
    detector = RegimeDetector()
    features = torch.zeros(60, 13, dtype=torch.float32)

    state = detector.forward(features)

    assert state.dominant in RegimeState.NAMES
    assert torch.allclose(state.probs, torch.full((4,), 0.25, dtype=torch.float32))


def test_regime_detector_save_load(workspace_tmp_path: Path) -> None:
    pytest.importorskip("hmmlearn")

    detector = RegimeDetector(n_iter=5)
    detector.fit(_regime_frame(rows=200, include_all=True))
    save_path = workspace_tmp_path / "regime_detector"

    detector.save(save_path)
    loaded = RegimeDetector.load(save_path)

    assert loaded.is_fitted is True
    assert loaded.window == detector.window
    assert loaded.n_regimes == detector.n_regimes


def test_regime_labeler_output_schema() -> None:
    frame = _regime_frame(rows=24, include_all=True, include_time=True)
    labeler = RegimeLabeler(RegimeDetector())

    result = labeler.label_dataframe(frame)

    assert result.columns == [
        "time_utc",
        "regime_dominant",
        "regime_trending",
        "regime_mean_rev",
        "regime_volatile",
        "regime_quiet",
        "regime_confidence",
    ]
    assert len(result) == len(frame)
    assert result.schema["regime_dominant"] == pl.Utf8


def test_regime_labeler_warmup_rows_are_quiet() -> None:
    frame = _regime_frame(rows=20, include_all=True, include_time=True)
    labeler = RegimeLabeler(_fitted_stub_detector(window=10))

    result = labeler.label_dataframe(frame)

    assert result.get_column("regime_dominant").head(9).to_list() == ["quiet"] * 9
    assert result.get_column("regime_dominant").slice(9, 1).to_list() == ["trending"]


def test_regime_labeler_join_preserves_length() -> None:
    dataset_df = _regime_frame(rows=18, include_all=True, include_time=True)
    label_df = RegimeLabeler(RegimeDetector()).label_dataframe(dataset_df).slice(4, 10)

    joined = RegimeLabeler.join_labels(dataset_df, label_df)

    assert joined.height == dataset_df.height


def test_regime_labeler_join_no_nulls_in_dominant() -> None:
    dataset_df = _regime_frame(rows=18, include_all=True, include_time=True)
    label_df = RegimeLabeler(RegimeDetector()).label_dataframe(dataset_df).slice(6, 5)

    joined = RegimeLabeler.join_labels(dataset_df, label_df)

    assert joined.get_column("regime_dominant").null_count() == 0


def test_gating_network_output_shape() -> None:
    gating = GatingNetwork(d_model=16, n_regimes=4, n_experts=4, dropout=0.0)
    hidden = torch.randn(8, 16)
    probs = torch.softmax(torch.randn(8, 4), dim=-1)

    weights = gating(hidden, probs)

    assert weights.shape == (8, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(8), atol=1e-5)


def test_gating_network_gradients() -> None:
    gating = GatingNetwork(d_model=16, n_regimes=4, n_experts=4, dropout=0.0)
    hidden = torch.randn(8, 16, requires_grad=True)
    probs = torch.softmax(torch.randn(8, 4), dim=-1)

    weights = gating(hidden, probs)
    weights.sum().backward()

    assert any(parameter.grad is not None for parameter in gating.parameters())


def test_moe_output_shape() -> None:
    torch.manual_seed(7)
    moe = MixtureOfExperts(expert_configs=[_expert_config(), _expert_config()], dropout=0.0)
    batch = _make_batch(batch_size=3, steps=8, n_past=4, n_future=2, n_static=1)
    regime_probs = torch.softmax(torch.randn(3, 4), dim=-1)

    output, weights = moe(batch, regime_probs)

    assert tuple(output.direction_logits["60m"].shape) == (3, 3)
    assert tuple(weights.shape) == (3, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_moe_expert_utilization() -> None:
    moe = MixtureOfExperts(expert_configs=[_expert_config(), _expert_config()], dropout=0.0)

    uniform = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
    degenerate = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

    uniform_utilization = moe.expert_utilization(uniform)
    degenerate_utilization = moe.expert_utilization(degenerate)

    assert uniform_utilization["expert_0"] == pytest.approx(0.5)
    assert uniform_utilization["expert_1"] == pytest.approx(0.5)
    assert degenerate_utilization["expert_0"] == pytest.approx(1.0)
    assert degenerate_utilization["expert_1"] == pytest.approx(0.0)


def test_moe_gradients_flow_to_all_experts() -> None:
    torch.manual_seed(11)
    moe = MixtureOfExperts(expert_configs=[_expert_config(), _expert_config()], dropout=0.0)
    batch = _make_batch(batch_size=2, steps=8, n_past=4, n_future=2, n_static=1)
    regime_probs = torch.softmax(torch.randn(2, 4), dim=-1)

    output, weights = moe(batch, regime_probs)
    loss = output.direction_logits["60m"].sum() + output.return_preds["60m"].sum() + weights.sum()
    loss.backward()

    for expert in moe.experts:
        expert_gradients = [
            parameter.grad
            for parameter in expert.parameters()
            if parameter.requires_grad
        ]
        assert any(gradient is not None for gradient in expert_gradients)


def _regime_frame(rows: int, include_all: bool, include_time: bool = False) -> pl.DataFrame:
    extractor = RegimeFeatureExtractor()
    columns = extractor.REGIME_FEATURE_COLS if include_all else extractor.REGIME_FEATURE_COLS[: len(extractor.REGIME_FEATURE_COLS) // 2]
    data: dict[str, list[float]] = {}
    if include_time:
        base_time = dt.datetime(2026, 4, 1, 0, 0, tzinfo=dt.timezone.utc)
        data[TIME_INDEX_COLUMN] = [base_time + dt.timedelta(minutes=index) for index in range(rows)]
    for column_index, column in enumerate(columns):
        data[column] = [
            math.sin((row_index + 1) * 0.1 + column_index) + (column_index * 0.05)
            for row_index in range(rows)
        ]
    return pl.DataFrame(data)


def _fitted_stub_detector(window: int) -> RegimeDetector:
    class _FittedStubDetector(RegimeDetector):
        @property
        def is_fitted(self) -> bool:
            return True

        def forward(self, features: torch.Tensor) -> RegimeState:
            return RegimeState.from_probs(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32))

    return _FittedStubDetector(window=window)


def _expert_config() -> dict[str, int | float]:
    return {
        "n_past_features": 4,
        "n_future_features": 2,
        "n_static_features": 1,
        "d_model": 16,
        "n_heads": 4,
        "n_lstm_layers": 1,
        "dropout": 0.0,
        "context_len": 8,
    }


def _make_batch(
    batch_size: int,
    steps: int,
    n_past: int,
    n_future: int,
    n_static: int,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    past_features = torch.randn(batch_size, steps, n_past, dtype=torch.float32)
    future_known = torch.randn(batch_size, steps, n_future, dtype=torch.float32)
    static = torch.randn(batch_size, n_static, dtype=torch.float32)
    mask = torch.ones(batch_size, steps, dtype=torch.bool)

    return {
        "past_features": past_features,
        "future_known": future_known,
        "static": static,
        "mask": mask,
        "time_idx": torch.arange(batch_size, dtype=torch.int64),
        "targets": {},
    }
