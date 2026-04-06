"""Random-tensor contract tests for the Phase 6 machinelearning models package."""

from __future__ import annotations

import torch
import pytest

from machinelearning.models import AphelionTFT, AttentionInspector, ModelOutput, VSNInterpreter
from machinelearning.models.base import HORIZONS, N_CLASSES, QUANTILES
from machinelearning.models.blocks import (
    ClassificationHead,
    GatedLinearUnit,
    GatedResidualNetwork,
    QuantileHead,
    RegressionHead,
    StaticCovariateEncoder,
    TemporalSelfAttention,
    VariableSelectionNetwork,
)


def _make_batch(
    batch_size: int = 4,
    steps: int = 8,
    n_past: int = 6,
    n_future: int = 3,
    n_static: int = 1,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    torch.manual_seed(7)
    past_features = torch.randn(batch_size, steps, n_past, dtype=torch.float32)
    future_known = torch.randn(batch_size, steps, n_future, dtype=torch.float32)
    static = torch.randn(batch_size, n_static, dtype=torch.float32)
    mask = torch.zeros(batch_size, steps, dtype=torch.bool)

    base_offsets = [0, 1, 2, 3]
    valid_lengths = torch.tensor(
        [max(1, steps - base_offsets[min(index, len(base_offsets) - 1)]) for index in range(batch_size)],
        dtype=torch.long,
    )
    for batch_index, valid_length in enumerate(valid_lengths.tolist()):
        mask[batch_index, :valid_length] = True
        past_features[batch_index, valid_length:] = 0.0
        future_known[batch_index, valid_length:] = 0.0

    return {
        "past_features": past_features,
        "future_known": future_known,
        "static": static,
        "mask": mask,
        "time_idx": torch.arange(batch_size, dtype=torch.int64),
        "targets": {},
    }


def _make_model(
    *,
    n_past_features: int = 6,
    n_future_features: int = 3,
    n_static_features: int = 1,
    d_model: int = 16,
    n_heads: int = 4,
    steps: int = 8,
) -> AphelionTFT:
    return AphelionTFT(
        n_past_features=n_past_features,
        n_future_features=n_future_features,
        n_static_features=n_static_features,
        d_model=d_model,
        n_heads=n_heads,
        n_lstm_layers=1,
        dropout=0.0,
        context_len=steps,
    )


def _head_modules(model: AphelionTFT) -> dict[tuple[str, str], torch.nn.Module]:
    modules: dict[tuple[str, str], torch.nn.Module] = {}
    for head_type, module_dict in (
        ("direction", model.direction_heads),
        ("tb", model.tb_heads),
        ("return", model.return_heads),
        ("mae", model.mae_heads),
        ("mfe", model.mfe_heads),
    ):
        for horizon, module in module_dict.items():
            modules[(head_type, horizon)] = module
    return modules


def _head_tensor(output: ModelOutput, head_type: str, horizon: str) -> torch.Tensor:
    if head_type == "direction":
        return output.direction_logits[horizon]
    if head_type == "tb":
        return output.tb_logits[horizon]
    if head_type == "return":
        return output.return_preds[horizon]
    if head_type == "mae":
        return output.mae_preds[horizon]
    if head_type == "mfe":
        return output.mfe_preds[horizon]
    raise KeyError(head_type)


def test_blocks_produce_expected_shapes() -> None:
    torch.manual_seed(0)
    batch_size, steps, n_features, n_static, d_model = 3, 6, 4, 2, 16
    sequence = torch.randn(batch_size, steps, d_model)
    feature_embeddings = torch.randn(batch_size, steps, n_features, d_model)
    static = torch.randn(batch_size, n_static)
    mask = torch.ones(batch_size, steps, dtype=torch.bool)

    glu = GatedLinearUnit(d_in=d_model, d_out=d_model, dropout=0.0)
    assert glu(sequence).shape == (batch_size, steps, d_model)

    grn = GatedResidualNetwork(
        d_input=d_model,
        d_hidden=d_model * 2,
        d_output=d_model,
        dropout=0.0,
        d_context=d_model,
    )
    assert grn(sequence, static.new_zeros(batch_size, d_model)).shape == (batch_size, steps, d_model)

    vsn = VariableSelectionNetwork(n_features=n_features, d_model=d_model, dropout=0.0, d_context=d_model)
    selected, weights = vsn(feature_embeddings, static.new_zeros(batch_size, d_model))
    assert selected.shape == (batch_size, steps, d_model)
    assert weights.shape == (batch_size, steps, n_features)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, steps), atol=1e-5)

    encoder = StaticCovariateEncoder(n_static=n_static, d_model=d_model, dropout=0.0)
    ctx_vsn, ctx_enrich, init_h, init_c = encoder(static)
    assert ctx_vsn.shape == (batch_size, d_model)
    assert ctx_enrich.shape == (batch_size, d_model)
    assert init_h.shape == (batch_size, d_model)
    assert init_c.shape == (batch_size, d_model)

    attention = TemporalSelfAttention(d_model=d_model, n_heads=4, dropout=0.0)
    attended, attn_weights = attention(sequence, mask)
    assert attended.shape == (batch_size, steps, d_model)
    assert attn_weights.shape == (batch_size, 4, steps, steps)

    classification = ClassificationHead(d_model=d_model, n_classes=N_CLASSES, dropout=0.0)
    quantiles = QuantileHead(d_model=d_model, n_quantiles=len(QUANTILES), dropout=0.0)
    regression = RegressionHead(d_model=d_model, dropout=0.0)
    pooled = sequence[:, -1, :]
    assert classification(pooled).shape == (batch_size, N_CLASSES)
    quantile_output = quantiles(pooled)
    assert quantile_output.shape == (batch_size, len(QUANTILES))
    assert torch.all(quantile_output[:, 1:] >= quantile_output[:, :-1])
    assert regression(pooled).shape == (batch_size,)


def test_tft_forward_and_output_contracts() -> None:
    batch = _make_batch(batch_size=4, steps=8, n_past=6, n_future=3, n_static=1)
    model = _make_model(n_past_features=6, n_future_features=3, n_static_features=1, d_model=16, steps=8)

    output = model(batch)

    assert output.encoder_hidden is not None
    assert output.encoder_hidden.shape == (4, 16)
    assert output.vsn_weights is not None
    assert output.vsn_weights["past"].shape == (4, 8, 6)
    assert output.vsn_weights["future"].shape == (4, 8, 3)
    assert output.attn_weights is not None
    assert output.attn_weights["past"].shape == (4, 4, 8, 8)

    for horizon in HORIZONS:
        assert horizon in output.direction_logits
        assert horizon in output.tb_logits
        assert horizon in output.return_preds
        assert horizon in output.mae_preds
        assert horizon in output.mfe_preds

        assert output.direction_logits[horizon].shape == (4, N_CLASSES)
        assert output.tb_logits[horizon].shape == (4, N_CLASSES)
        assert output.return_preds[horizon].shape == (4, len(QUANTILES))
        assert output.mae_preds[horizon].shape == (4,)
        assert output.mfe_preds[horizon].shape == (4,)

        probs = output.direction_probs(horizon)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

        lower, upper = output.return_interval(horizon)
        assert torch.all(lower <= upper)


def test_tft_backward_reaches_all_parameters() -> None:
    batch = _make_batch(batch_size=4, steps=8, n_past=6, n_future=3, n_static=1)
    model = _make_model(n_past_features=6, n_future_features=3, n_static_features=1, d_model=16, steps=8)

    output = model(batch)
    loss = torch.tensor(0.0, dtype=torch.float32)
    for predictions in (
        output.direction_logits,
        output.tb_logits,
        output.return_preds,
        output.mae_preds,
        output.mfe_preds,
    ):
        for tensor in predictions.values():
            loss = loss + tensor.float().sum()
    if output.encoder_hidden is not None:
        loss = loss + output.encoder_hidden.float().sum()
    if output.vsn_weights is not None:
        loss = loss + output.vsn_weights["past"].float().sum() + output.vsn_weights["future"].float().sum()

    loss.backward()

    missing_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not missing_gradients
    assert model.count_parameters() > 0


def test_glu_output_shape() -> None:
    module = GatedLinearUnit(d_in=16, d_out=8, dropout=0.0)
    output = module(torch.randn(4, 16))
    assert output.shape == (4, 8)


def test_grn_output_shape_no_context() -> None:
    module = GatedResidualNetwork(d_input=16, d_hidden=32, d_output=16, dropout=0.0)
    output = module(torch.randn(4, 16))
    assert output.shape == (4, 16)


def test_grn_output_shape_with_context() -> None:
    module = GatedResidualNetwork(d_input=16, d_hidden=32, d_output=16, dropout=0.0, d_context=8)
    output = module(torch.randn(4, 16), torch.randn(4, 8))
    assert output.shape == (4, 16)


def test_vsn_weights_sum_to_one() -> None:
    module = VariableSelectionNetwork(n_features=10, d_model=16, dropout=0.0)
    embeddings = torch.randn(2, 5, 10, 16)
    _, weights = module(embeddings)
    assert weights.shape == (2, 5, 10)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-5)


def test_static_encoder_output_count() -> None:
    encoder = StaticCovariateEncoder(n_static=1, d_model=16, dropout=0.0)
    outputs = encoder(torch.randn(3, 1))
    assert len(outputs) == 4
    for tensor in outputs:
        assert tensor.shape == (3, 16)


def test_temporal_attention_output_shape() -> None:
    module = TemporalSelfAttention(d_model=16, n_heads=4, dropout=0.0)
    output, attn_weights = module(torch.randn(2, 8, 16), torch.ones(2, 8, dtype=torch.bool))
    assert output.shape == (2, 8, 16)
    assert attn_weights.shape == (2, 4, 8, 8)


def test_tft_invalid_heads_raises() -> None:
    with pytest.raises(ValueError, match="divisible"):
        AphelionTFT(
            n_past_features=4,
            n_future_features=2,
            d_model=10,
            n_heads=3,
        )


def test_tft_encoder_hidden_shape() -> None:
    batch = _make_batch(batch_size=2, steps=6, n_past=4, n_future=2, n_static=1)
    model = _make_model(n_past_features=4, n_future_features=2, n_static_features=1, d_model=16, steps=6)
    output = model(batch)
    assert output.encoder_hidden is not None
    assert output.encoder_hidden.shape == (2, 16)


def test_tft_vsn_weights_past_shape() -> None:
    batch = _make_batch(batch_size=2, steps=6, n_past=4, n_future=2, n_static=1)
    model = _make_model(n_past_features=4, n_future_features=2, n_static_features=1, d_model=16, steps=6)
    output = model(batch)
    assert output.vsn_weights is not None
    assert output.vsn_weights["past"].shape == (2, 6, 4)
    assert output.vsn_weights["future"].shape == (2, 6, 2)


def test_tft_return_quantiles_are_monotonic() -> None:
    batch = _make_batch(batch_size=2, steps=6, n_past=4, n_future=2, n_static=1)
    model = _make_model(n_past_features=4, n_future_features=2, n_static_features=1, d_model=16, steps=6)
    output = model(batch)
    quantiles = output.return_preds["60m"]
    assert torch.all(quantiles[:, 1:] >= quantiles[:, :-1])


def test_tft_return_interval_monotonic() -> None:
    batch = _make_batch(batch_size=2, steps=6, n_past=4, n_future=2, n_static=1)
    model = _make_model(n_past_features=4, n_future_features=2, n_static_features=1, d_model=16, steps=6)
    output = model(batch)
    lower, upper = output.return_interval("60m", alpha=0.8)
    assert torch.all(upper >= lower)


def test_tft_direction_probs_sum_to_one() -> None:
    batch = _make_batch(batch_size=2, steps=6, n_past=4, n_future=2, n_static=1)
    model = _make_model(n_past_features=4, n_future_features=2, n_static_features=1, d_model=16, steps=6)
    output = model(batch)
    assert torch.allclose(output.direction_probs("60m").sum(dim=-1), torch.ones(2), atol=1e-5)


def test_tft_all_horizons_present() -> None:
    batch = _make_batch(batch_size=2, steps=6, n_past=4, n_future=2, n_static=1)
    model = _make_model(n_past_features=4, n_future_features=2, n_static_features=1, d_model=16, steps=6)
    output = model(batch)
    expected = {"5m", "15m", "60m", "240m"}
    assert expected == set(output.direction_logits)
    assert expected == set(output.tb_logits)
    assert expected == set(output.return_preds)
    assert expected == set(output.mae_preds)
    assert expected == set(output.mfe_preds)


def test_tft_gradient_per_head() -> None:
    batch = _make_batch(batch_size=2, steps=6, n_past=4, n_future=2, n_static=1)
    model = _make_model(n_past_features=4, n_future_features=2, n_static_features=1, d_model=16, steps=6)
    head_modules = _head_modules(model)

    for target_key, target_module in head_modules.items():
        model.zero_grad(set_to_none=True)
        output = model(batch)
        loss = _head_tensor(output, *target_key).float().sum()
        loss.backward()

        target_parameters = [parameter for parameter in target_module.parameters() if parameter.requires_grad]
        assert target_parameters
        assert all(parameter.grad is not None for parameter in target_parameters)

        for other_key, other_module in head_modules.items():
            if other_key == target_key:
                continue
            other_parameters = [parameter for parameter in other_module.parameters() if parameter.requires_grad]
            assert other_parameters
            assert all(parameter.grad is None for parameter in other_parameters)


def test_vsn_interpreter_summarizes_past_importance() -> None:
    batch = _make_batch(batch_size=2, steps=6, n_past=4, n_future=2, n_static=1)
    model = _make_model(n_past_features=4, n_future_features=2, n_static_features=1, d_model=16, steps=6)
    output = model(batch)

    feature_names = ["spread", "entropy", "microstructure", "calendar"]
    interpreter = VSNInterpreter.from_output(output, feature_names)
    importance = interpreter.past_importance

    assert set(importance) == set(feature_names)
    assert pytest.approx(sum(importance.values()), rel=1e-5, abs=1e-5) == 1.0

    top_features = interpreter.top_features(3)
    assert len(top_features) == 3
    assert top_features[0][1] >= top_features[-1][1]

    family_scores = interpreter.family_importance(
        {
            "microstructure": ["spread", "microstructure"],
            "context": ["entropy", "calendar"],
        },
    )
    assert pytest.approx(sum(family_scores.values()), rel=1e-5, abs=1e-5) == 1.0
    assert "VSN top features:" in interpreter.summary_str()


def test_vsn_interpreter_rejects_feature_count_mismatch() -> None:
    batch = _make_batch(batch_size=2, steps=6, n_past=4, n_future=2, n_static=1)
    model = _make_model(n_past_features=4, n_future_features=2, n_static_features=1, d_model=16, steps=6)
    output = model(batch)

    with pytest.raises(ValueError, match="feature_names length"):
        VSNInterpreter.from_output(output, ["only", "two"])


def test_attention_inspector_requires_encoder_hidden() -> None:
    output = ModelOutput(
        direction_logits={},
        tb_logits={},
        return_preds={},
        mae_preds={},
        mfe_preds={},
        encoder_hidden=None,
        vsn_weights=None,
        attn_weights=None,
    )
    with pytest.raises(ValueError, match="encoder_hidden"):
        AttentionInspector.from_output(output)


def test_attention_inspector_mean_shape() -> None:
    batch = _make_batch(batch_size=2, steps=6, n_past=4, n_future=2, n_static=1)
    model = _make_model(n_past_features=4, n_future_features=2, n_static_features=1, d_model=16, steps=6)
    output = model(batch)

    inspector = AttentionInspector.from_output(output)
    mean_attention = inspector.mean_attention

    assert mean_attention is not None
    assert tuple(mean_attention.shape) == (6, 6)


def test_attention_inspector_last_timestep() -> None:
    batch = _make_batch(batch_size=2, steps=6, n_past=4, n_future=2, n_static=1)
    model = _make_model(n_past_features=4, n_future_features=2, n_static_features=1, d_model=16, steps=6)
    output = model(batch)

    inspector = AttentionInspector.from_output(output)
    last_attention = inspector.last_timestep_attention()

    assert last_attention is not None
    assert tuple(last_attention.shape) == (6,)
    assert torch.allclose(last_attention, inspector.mean_attention[-1, :])


def test_model_output_attn_weights_stored() -> None:
    batch = _make_batch(batch_size=2, steps=6, n_past=4, n_future=2, n_static=1)
    model = _make_model(n_past_features=4, n_future_features=2, n_static_features=1, d_model=16, steps=6)
    output = model(batch)

    assert output.attn_weights is not None
    assert output.attn_weights["past"] is not None
    assert tuple(output.attn_weights["past"].shape) == (2, 4, 6, 6)
