"""Reusable Temporal Fusion Transformer building blocks for Phase 6."""

from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _broadcast_context(context: Tensor, target: Tensor) -> Tensor:
    """Broadcast a context tensor across the non-feature dimensions of a target tensor."""

    expanded = context
    while expanded.dim() < target.dim():
        expanded = expanded.unsqueeze(1)
    expand_sizes = list(expanded.shape)
    for axis in range(target.dim() - 1):
        if expand_sizes[axis] == 1:
            expand_sizes[axis] = target.shape[axis]
    return expanded.expand(*expand_sizes)


class GatedLinearUnit(nn.Module):
    """Project inputs into value and gate streams, then modulate values by the learned gate."""

    def __init__(self, d_in: int, d_out: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_in, d_out * 2)

    def forward(self, inputs: Tensor) -> Tensor:
        projected = self.proj(self.dropout(inputs))
        value, gate = projected.chunk(2, dim=-1)
        return value * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """Transform inputs with optional context conditioning, then mix them back through a gated residual path."""

    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        d_output: int,
        dropout: float,
        d_context: int | None = None,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_hidden)
        self.context_proj = nn.Linear(d_context, d_hidden) if d_context is not None else None
        self.hidden_proj = nn.Linear(d_hidden, d_hidden)
        self.glu = GatedLinearUnit(d_hidden, d_output, dropout)
        self.skip = nn.Linear(d_input, d_output) if d_input != d_output else nn.Identity()
        self.norm = nn.LayerNorm(d_output)

    def forward(self, inputs: Tensor, context: Tensor | None = None) -> Tensor:
        residual = self.skip(inputs)
        hidden = F.elu(self.input_proj(inputs))
        if context is not None and self.context_proj is not None:
            hidden = hidden + _broadcast_context(self.context_proj(context), hidden)
        hidden = F.elu(self.hidden_proj(hidden))
        hidden = self.glu(hidden)
        return self.norm(hidden + residual)


class VariableSelectionNetwork(nn.Module):
    """Select and combine per-feature embeddings into one temporal representation with interpretable softmax weights."""

    def __init__(self, n_features: int, d_model: int, dropout: float, d_context: int | None = None) -> None:
        super().__init__()
        self.n_features: Final[int] = n_features
        self.d_model: Final[int] = d_model
        self.feature_grns = nn.ModuleList(
            GatedResidualNetwork(d_model, d_model, d_model, dropout, d_context=d_context)
            for _ in range(n_features)
        )
        self.weight_grn = (
            GatedResidualNetwork(n_features * d_model, d_model, n_features, dropout, d_context=d_context)
            if n_features > 0
            else None
        )

    def forward(self, inputs: Tensor, context: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if inputs.dim() != 4:
            raise ValueError(
                "VariableSelectionNetwork expects inputs shaped [batch, time, features, d_model], "
                f"got {tuple(inputs.shape)}"
            )

        batch_size, steps, n_features, _ = inputs.shape
        if n_features != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, received {n_features}")

        if self.n_features == 0:
            selected = inputs.new_zeros(batch_size, steps, self.d_model)
            weights = inputs.new_zeros(batch_size, steps, 0)
            return selected, weights

        feature_outputs = [
            feature_grn(inputs[:, :, feature_index, :], context)
            for feature_index, feature_grn in enumerate(self.feature_grns)
        ]
        transformed = torch.stack(feature_outputs, dim=2)
        flattened = inputs.reshape(batch_size, steps, n_features * self.d_model)
        weights = torch.softmax(self.weight_grn(flattened, context), dim=-1)
        selected = torch.sum(transformed * weights.unsqueeze(-1), dim=2)
        return selected, weights


class StaticCovariateEncoder(nn.Module):
    """Encode static features into the TFT contexts used for variable selection, enrichment, and LSTM state seeding."""

    def __init__(self, n_static: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.n_static = max(1, n_static)
        self.ctx_vsn_grn = GatedResidualNetwork(self.n_static, d_model, d_model, dropout)
        self.ctx_enrich_grn = GatedResidualNetwork(self.n_static, d_model, d_model, dropout)
        self.init_h_grn = GatedResidualNetwork(self.n_static, d_model, d_model, dropout)
        self.init_c_grn = GatedResidualNetwork(self.n_static, d_model, d_model, dropout)

    def forward(self, static: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if static.dim() != 2:
            raise ValueError(f"StaticCovariateEncoder expects [batch, static_features], got {tuple(static.shape)}")
        static_inputs = static if static.size(-1) > 0 else static.new_zeros(static.size(0), self.n_static)
        return (
            self.ctx_vsn_grn(static_inputs),
            self.ctx_enrich_grn(static_inputs),
            self.init_h_grn(static_inputs),
            self.init_c_grn(static_inputs),
        )


class TemporalSelfAttention(nn.Module):
    """Attend across the temporal dimension with padding-aware multi-head attention and a gated residual output path."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.glu = GatedLinearUnit(d_model, d_model, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        key_padding_mask = None if mask is None else ~mask
        attended, weights = self.attention(
            inputs,
            inputs,
            inputs,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        output = self.norm(self.glu(attended) + inputs)
        return output, weights


class ClassificationHead(nn.Module):
    """Map the shared prediction representation into raw logits for a horizon-specific classification task."""

    def __init__(self, d_model: int, n_classes: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        self.grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.output = nn.Linear(d_model, n_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.output(self.grn(inputs))


class QuantileHead(nn.Module):
    """Map the shared prediction representation into a fixed set of return quantile forecasts."""

    def __init__(self, d_model: int, n_quantiles: int, dropout: float) -> None:
        super().__init__()
        self.grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.output = nn.Linear(d_model, n_quantiles)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.output(self.grn(inputs))


class RegressionHead(nn.Module):
    """Map the shared prediction representation into a single scalar regression output."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = max(1, d_model // 2)
        self.grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.hidden = nn.Linear(d_model, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.hidden(self.grn(inputs))
        return self.output(hidden).squeeze(-1)
