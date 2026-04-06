"""Mixture-of-experts routing for regime-specialized Phase 7 models."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from machinelearning.models import AphelionTFT, ModelOutput


class GatingNetwork(nn.Module):
    """Soft router that blends regime probabilities with encoder-hidden context."""

    def __init__(
        self,
        d_model: int,
        n_regimes: int = 4,
        n_experts: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_experts <= 0:
            raise ValueError("n_experts must be positive.")
        self.d_model = d_model
        self.n_regimes = n_regimes
        self.n_experts = n_experts
        self.dropout = nn.Dropout(dropout)
        self.hidden_proj = nn.Linear(d_model, n_experts)
        self.regime_proj = nn.Linear(n_regimes, n_experts)

    def forward(
        self,
        encoder_hidden: Tensor,
        regime_probs: Tensor,
    ) -> Tensor:
        """Return expert weights [B, n_experts] that sum to 1."""

        if encoder_hidden.dim() != 2:
            raise ValueError(
                f"encoder_hidden must be [batch, d_model], got {tuple(encoder_hidden.shape)}",
            )
        if regime_probs.dim() != 2:
            raise ValueError(
                f"regime_probs must be [batch, n_regimes], got {tuple(regime_probs.shape)}",
            )
        if encoder_hidden.size(0) != regime_probs.size(0):
            raise ValueError("encoder_hidden and regime_probs batch sizes must match.")
        if encoder_hidden.size(1) != self.d_model:
            raise ValueError(f"Expected encoder_hidden width {self.d_model}, got {encoder_hidden.size(1)}.")
        if regime_probs.size(1) != self.n_regimes:
            raise ValueError(f"Expected regime_probs width {self.n_regimes}, got {regime_probs.size(1)}.")

        normalized_regime = self._normalize_regime_probs(regime_probs)
        logits = self.hidden_proj(self.dropout(encoder_hidden)) + self.regime_proj(self.dropout(normalized_regime))
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def _normalize_regime_probs(regime_probs: Tensor) -> Tensor:
        """Normalize regime probabilities when the caller supplies arbitrary float tensors."""

        sums = regime_probs.sum(dim=-1, keepdim=True)
        if torch.isfinite(regime_probs).all() and (regime_probs >= 0).all() and torch.allclose(
            sums,
            torch.ones_like(sums),
            atol=1e-4,
            rtol=1e-4,
        ):
            return regime_probs
        return torch.softmax(regime_probs, dim=-1)


class MixtureOfExperts(nn.Module):
    """Blend multiple AphelionTFT experts through a learned regime-aware router."""

    def __init__(
        self,
        expert_configs: list[dict[str, Any]],
        n_regimes: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not expert_configs:
            raise ValueError("expert_configs must contain at least one expert configuration.")

        self.experts = nn.ModuleList([AphelionTFT(**config) for config in expert_configs])
        self._validate_expert_shapes()

        self.gating = GatingNetwork(
            d_model=self.experts[0].d_model,
            n_regimes=n_regimes,
            n_experts=len(self.experts),
            dropout=dropout,
        )

    @property
    def n_experts(self) -> int:
        """Return the number of experts in the mixture."""

        return len(self.experts)

    def forward(
        self,
        batch: dict[str, Tensor],
        regime_probs: Tensor,
    ) -> tuple[ModelOutput, Tensor]:
        """Return a blended model output together with expert weights."""

        expert_outputs = [expert(batch) for expert in self.experts]
        encoder_hiddens = [output.encoder_hidden for output in expert_outputs]
        if any(hidden is None for hidden in encoder_hiddens):
            raise RuntimeError("All experts must return encoder_hidden for MoE routing.")

        shared_hidden = torch.stack([hidden for hidden in encoder_hiddens if hidden is not None], dim=0).mean(dim=0)
        expert_weights = self.gating(shared_hidden, regime_probs)
        blended = self._blend_outputs(expert_outputs, expert_weights)
        return blended, expert_weights

    def expert_utilization(self, expert_weights: Tensor) -> dict[str, float]:
        """Return mean routing weight per expert for logging and collapse detection."""

        if expert_weights.dim() != 2 or expert_weights.size(1) != self.n_experts:
            raise ValueError(
                f"expert_weights must be [batch, {self.n_experts}], got {tuple(expert_weights.shape)}",
            )
        mean_weights = expert_weights.detach().float().mean(dim=0)
        return {
            f"expert_{index}": float(mean_weights[index].item())
            for index in range(self.n_experts)
        }

    def _blend_outputs(self, outputs: list[ModelOutput], weights: Tensor) -> ModelOutput:
        """Blend expert outputs into a single model output using routing weights."""

        return ModelOutput(
            direction_logits=self._blend_mapping([output.direction_logits for output in outputs], weights),
            tb_logits=self._blend_mapping([output.tb_logits for output in outputs], weights),
            return_preds=self._blend_mapping([output.return_preds for output in outputs], weights),
            mae_preds=self._blend_mapping([output.mae_preds for output in outputs], weights),
            mfe_preds=self._blend_mapping([output.mfe_preds for output in outputs], weights),
            encoder_hidden=self._blend_optional_tensor([output.encoder_hidden for output in outputs], weights),
            vsn_weights=self._blend_optional_mapping([output.vsn_weights for output in outputs], weights),
            attn_weights=self._blend_optional_mapping([output.attn_weights for output in outputs], weights),
        )

    def _blend_mapping(self, mappings: list[dict[str, Tensor]], weights: Tensor) -> dict[str, Tensor]:
        """Blend a list of horizon->tensor mappings using expert weights."""

        keys = mappings[0].keys()
        return {
            key: self._blend_tensors([mapping[key] for mapping in mappings], weights)
            for key in keys
        }

    def _blend_optional_mapping(
        self,
        mappings: list[dict[str, Tensor] | None],
        weights: Tensor,
    ) -> dict[str, Tensor] | None:
        """Blend optional diagnostic mappings when all experts expose them."""

        if any(mapping is None for mapping in mappings):
            return None
        return self._blend_mapping([mapping for mapping in mappings if mapping is not None], weights)

    def _blend_optional_tensor(
        self,
        tensors: list[Tensor | None],
        weights: Tensor,
    ) -> Tensor | None:
        """Blend optional tensors when all experts expose them."""

        if any(tensor is None for tensor in tensors):
            return None
        return self._blend_tensors([tensor for tensor in tensors if tensor is not None], weights)

    @staticmethod
    def _blend_tensors(tensors: list[Tensor], weights: Tensor) -> Tensor:
        """Blend expert tensors with broadcasted routing weights."""

        stacked = torch.stack(tensors, dim=1)
        view_shape = (weights.size(0), weights.size(1)) + (1,) * (stacked.dim() - 2)
        return (stacked * weights.view(view_shape)).sum(dim=1)

    def _validate_expert_shapes(self) -> None:
        """Ensure all experts share the same architecture shape contract."""

        reference = self.experts[0]
        expected = (
            reference.d_model,
            reference.n_past_features,
            reference.n_future_features,
            reference.n_static_features,
        )
        for expert in self.experts[1:]:
            observed = (
                expert.d_model,
                expert.n_past_features,
                expert.n_future_features,
                expert.n_static_features,
            )
            if observed != expected:
                raise ValueError(
                    "All experts in Phase 7 MoE must share the same shape contract. "
                    f"Expected {expected}, got {observed}.",
                )


__all__ = ["GatingNetwork", "MixtureOfExperts"]
