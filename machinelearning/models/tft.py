"""Temporal Fusion Transformer model used for the Phase 6 deep-learning stack."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .base import AphelionModel, HORIZONS, ModelOutput, N_CLASSES, QUANTILES
from .blocks import (
    ClassificationHead,
    GatedLinearUnit,
    GatedResidualNetwork,
    QuantileHead,
    RegressionHead,
    StaticCovariateEncoder,
    TemporalSelfAttention,
    VariableSelectionNetwork,
)


class AphelionTFT(AphelionModel):
    """Temporal Fusion Transformer that turns multi-source bar windows into multi-horizon forecasts."""

    model_name = "aphelion-tft"
    model_version = "1.0.0"

    def __init__(
        self,
        n_past_features: int,
        n_future_features: int,
        n_static_features: int = 1,
        d_model: int = 128,
        n_heads: int = 4,
        n_lstm_layers: int = 2,
        dropout: float = 0.1,
        context_len: int = 240,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.n_past_features = n_past_features
        self.n_future_features = n_future_features
        self.n_static_features = n_static_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_lstm_layers = n_lstm_layers
        self.dropout = dropout
        self.context_len = context_len

        # At B=512, T=240, d_model=128: ~6-8GB per GPU. Fits in 24GB L4.
        self.past_feature_projections = nn.ModuleList(nn.Linear(1, d_model) for _ in range(n_past_features))
        self.future_feature_projections = nn.ModuleList(nn.Linear(1, d_model) for _ in range(n_future_features))

        self.static_encoder = StaticCovariateEncoder(n_static_features, d_model, dropout)
        self.past_vsn = VariableSelectionNetwork(n_past_features, d_model, dropout, d_context=d_model)
        self.future_vsn = VariableSelectionNetwork(n_future_features, d_model, dropout, d_context=d_model)

        lstm_dropout = dropout if n_lstm_layers > 1 else 0.0
        self.temporal_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.post_lstm_gate = GatedLinearUnit(d_model, d_model, dropout)
        self.post_lstm_norm = nn.LayerNorm(d_model)

        self.static_enrichment = GatedResidualNetwork(d_model, d_model, d_model, dropout, d_context=d_model)
        self.temporal_attention = TemporalSelfAttention(d_model, n_heads, dropout)
        self.positionwise_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.positionwise_glu = GatedLinearUnit(d_model, d_model, dropout)
        self.positionwise_norm = nn.LayerNorm(d_model)

        self.direction_heads = nn.ModuleDict(
            {horizon: ClassificationHead(d_model, n_classes=N_CLASSES, dropout=dropout) for horizon in HORIZONS}
        )
        self.tb_heads = nn.ModuleDict(
            {horizon: ClassificationHead(d_model, n_classes=N_CLASSES, dropout=dropout) for horizon in HORIZONS}
        )
        self.return_heads = nn.ModuleDict(
            {horizon: QuantileHead(d_model, n_quantiles=len(QUANTILES), dropout=dropout) for horizon in HORIZONS}
        )
        self.mae_heads = nn.ModuleDict({horizon: RegressionHead(d_model, dropout=dropout) for horizon in HORIZONS})
        self.mfe_heads = nn.ModuleDict({horizon: RegressionHead(d_model, dropout=dropout) for horizon in HORIZONS})

        self._reset_parameters()

    def forward(self, batch: dict[str, Any]) -> ModelOutput:
        past_features = batch["past_features"]
        future_known = batch["future_known"]
        static = batch["static"]
        mask = batch["mask"]

        if past_features.dim() != 3:
            raise ValueError(f"past_features must be [batch, time, features], got {tuple(past_features.shape)}")
        if future_known.dim() != 3:
            raise ValueError(f"future_known must be [batch, time, features], got {tuple(future_known.shape)}")
        if static.dim() != 2:
            raise ValueError(f"static must be [batch, features], got {tuple(static.shape)}")
        if mask.dim() != 2:
            raise ValueError(f"mask must be [batch, time], got {tuple(mask.shape)}")

        batch_size, steps, n_past = past_features.shape
        _, future_steps, n_future = future_known.shape
        if steps != future_steps:
            raise ValueError(f"past and future time dimensions must match, got {steps} and {future_steps}")
        if batch_size != static.size(0) or batch_size != mask.size(0):
            raise ValueError("Batch dimension mismatch across inputs")
        if steps != mask.size(1):
            raise ValueError("Mask time dimension must match temporal inputs")
        if n_past != self.n_past_features:
            raise ValueError(f"Expected {self.n_past_features} past features, received {n_past}")
        if n_future != self.n_future_features:
            raise ValueError(f"Expected {self.n_future_features} future features, received {n_future}")
        if self.context_len and steps > self.context_len:
            raise ValueError(f"Received sequence length {steps}, but context_len is {self.context_len}")

        past_embedded = self._project_features(past_features, self.past_feature_projections)
        future_embedded = self._project_features(future_known, self.future_feature_projections)

        ctx_vsn, ctx_enrich, init_h, init_c = self.static_encoder(static)
        past_selected, past_weights = self.past_vsn(past_embedded, ctx_vsn)
        future_selected, future_weights = self.future_vsn(future_embedded, ctx_vsn)

        temporal_mask = mask.to(dtype=torch.bool)
        past_selected = self._apply_temporal_mask(past_selected, temporal_mask)
        future_selected = self._apply_temporal_mask(future_selected, temporal_mask)

        lengths = temporal_mask.long().sum(dim=1).clamp(min=1).cpu()
        packed_inputs = pack_padded_sequence(
            past_selected,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        h0 = init_h.unsqueeze(0).expand(self.n_lstm_layers, -1, -1).contiguous()
        c0 = init_c.unsqueeze(0).expand(self.n_lstm_layers, -1, -1).contiguous()
        packed_outputs, _ = self.temporal_lstm(packed_inputs, (h0, c0))
        lstm_output, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=steps,
        )

        temporal_residual = past_selected + future_selected
        lstm_output = self.post_lstm_norm(self.post_lstm_gate(lstm_output) + temporal_residual)
        lstm_output = self._apply_temporal_mask(lstm_output, temporal_mask)

        enriched = self.static_enrichment(lstm_output, ctx_enrich)
        enriched = self._apply_temporal_mask(enriched, temporal_mask)

        attended, _ = self.temporal_attention(enriched, temporal_mask)
        attended = self._apply_temporal_mask(attended, temporal_mask)

        refined = self.positionwise_grn(attended)
        refined = self.positionwise_norm(self.positionwise_glu(refined) + attended)
        refined = self._apply_temporal_mask(refined, temporal_mask)

        last_index = temporal_mask.long().sum(dim=1).sub(1).clamp(min=0)
        pred_repr = refined[torch.arange(batch_size, device=refined.device), last_index]

        direction_logits = {
            horizon: head(pred_repr)
            for horizon, head in self.direction_heads.items()
        }
        tb_logits = {
            horizon: head(pred_repr)
            for horizon, head in self.tb_heads.items()
        }
        return_preds = {
            horizon: head(pred_repr)
            for horizon, head in self.return_heads.items()
        }
        mae_preds = {
            horizon: head(pred_repr)
            for horizon, head in self.mae_heads.items()
        }
        mfe_preds = {
            horizon: head(pred_repr)
            for horizon, head in self.mfe_heads.items()
        }

        return ModelOutput(
            direction_logits=direction_logits,
            tb_logits=tb_logits,
            return_preds=return_preds,
            mae_preds=mae_preds,
            mfe_preds=mfe_preds,
            encoder_hidden=pred_repr,
            vsn_weights={"past": past_weights, "future": future_weights},
        )

    def _project_features(self, values: Tensor, projections: nn.ModuleList) -> Tensor:
        projected = [layer(values[:, :, index : index + 1]) for index, layer in enumerate(projections)]
        if not projected:
            return values.new_zeros(values.size(0), values.size(1), 0, self.d_model)
        return torch.stack(projected, dim=2)

    @staticmethod
    def _apply_temporal_mask(values: Tensor, mask: Tensor) -> Tensor:
        return values * mask.unsqueeze(-1).to(values.dtype)

    def _reset_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            if "bias" in name:
                nn.init.zeros_(parameter)
            elif parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
