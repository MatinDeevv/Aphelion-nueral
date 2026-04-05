"""Random-tensor contract tests for the Phase 6 machinelearning models package."""

from __future__ import annotations

import torch

from machinelearning.models import AphelionTFT
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
    steps: int = 16,
    n_past: int = 60,
    n_future: int = 8,
    n_static: int = 3,
) -> dict[str, object]:
    torch.manual_seed(7)
    past_features = torch.randn(batch_size, steps, n_past, dtype=torch.float32)
    future_known = torch.randn(batch_size, steps, n_future, dtype=torch.float32)
    static = torch.randn(batch_size, n_static, dtype=torch.float32)
    mask = torch.zeros(batch_size, steps, dtype=torch.bool)
    valid_lengths = torch.tensor([steps, steps - 1, steps - 3, steps - 5], dtype=torch.long)
    for batch_index, valid_length in enumerate(valid_lengths):
        mask[batch_index, : valid_length.item()] = True
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


def test_blocks_produce_expected_shapes() -> None:
    torch.manual_seed(0)
    batch_size, steps, n_features, n_static, d_model = 4, 12, 5, 3, 32
    sequence = torch.randn(batch_size, steps, d_model)
    feature_embeddings = torch.randn(batch_size, steps, n_features, d_model)
    static = torch.randn(batch_size, n_static)
    mask = torch.ones(batch_size, steps, dtype=torch.bool)

    glu = GatedLinearUnit(d_in=d_model, d_out=d_model, dropout=0.1)
    assert glu(sequence).shape == (batch_size, steps, d_model)

    grn = GatedResidualNetwork(
        d_input=d_model,
        d_hidden=d_model,
        d_output=d_model,
        dropout=0.1,
        d_context=d_model,
    )
    assert grn(sequence, static.new_zeros(batch_size, d_model)).shape == (batch_size, steps, d_model)

    vsn = VariableSelectionNetwork(n_features=n_features, d_model=d_model, dropout=0.1, d_context=d_model)
    selected, weights = vsn(feature_embeddings, static.new_zeros(batch_size, d_model))
    assert selected.shape == (batch_size, steps, d_model)
    assert weights.shape == (batch_size, steps, n_features)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, steps), atol=1e-5)

    encoder = StaticCovariateEncoder(n_static=n_static, d_model=d_model, dropout=0.1)
    ctx_vsn, ctx_enrich, init_h, init_c = encoder(static)
    assert ctx_vsn.shape == (batch_size, d_model)
    assert ctx_enrich.shape == (batch_size, d_model)
    assert init_h.shape == (batch_size, d_model)
    assert init_c.shape == (batch_size, d_model)

    attention = TemporalSelfAttention(d_model=d_model, n_heads=4, dropout=0.1)
    attended, attn_weights = attention(sequence, mask)
    assert attended.shape == (batch_size, steps, d_model)
    assert attn_weights.shape == (batch_size, 4, steps, steps)

    classification = ClassificationHead(d_model=d_model, n_classes=N_CLASSES, dropout=0.1)
    quantiles = QuantileHead(d_model=d_model, n_quantiles=len(QUANTILES), dropout=0.1)
    regression = RegressionHead(d_model=d_model, dropout=0.1)
    pooled = sequence[:, -1, :]
    assert classification(pooled).shape == (batch_size, N_CLASSES)
    assert quantiles(pooled).shape == (batch_size, len(QUANTILES))
    assert regression(pooled).shape == (batch_size,)


def test_tft_forward_and_output_contracts() -> None:
    batch = _make_batch()
    model = AphelionTFT(
        n_past_features=60,
        n_future_features=8,
        n_static_features=3,
        d_model=64,
        n_heads=4,
        n_lstm_layers=2,
        dropout=0.1,
        context_len=16,
    )

    output = model(batch)

    assert output.encoder_hidden is not None
    assert output.encoder_hidden.shape == (4, 64)
    assert output.vsn_weights is not None
    assert output.vsn_weights["past"].shape == (4, 16, 60)
    assert output.vsn_weights["future"].shape == (4, 16, 8)

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
    batch = _make_batch()
    model = AphelionTFT(
        n_past_features=60,
        n_future_features=8,
        n_static_features=3,
        d_model=64,
        n_heads=4,
        n_lstm_layers=2,
        dropout=0.1,
        context_len=16,
    )

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
