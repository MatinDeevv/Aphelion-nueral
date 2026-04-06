"""Interpretability helpers for the Phase 6 Temporal Fusion Transformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from machinelearning.models import ModelOutput


@dataclass(slots=True)
class VSNInterpreter:
    """Summarize Variable Selection Network weights into feature-level importance."""

    feature_names: list[str]
    weights: Tensor

    @staticmethod
    def from_output(
        output: "ModelOutput",
        feature_names: list[str],
    ) -> "VSNInterpreter":
        """Build an interpreter from a model forward-pass output."""

        if output.vsn_weights is None or "past" not in output.vsn_weights:
            raise ValueError("ModelOutput does not include past VSN weights.")

        weights = output.vsn_weights["past"].detach().float().cpu()
        if weights.dim() != 3:
            raise ValueError(f"Expected past VSN weights shaped [B, T, F], got {tuple(weights.shape)}")
        if len(feature_names) != weights.shape[-1]:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) does not match VSN feature axis ({weights.shape[-1]}).",
            )

        return VSNInterpreter(feature_names=list(feature_names), weights=weights)

    @property
    def past_importance(self) -> dict[str, float]:
        """Return mean normalized VSN weight per past feature."""

        importance = self.weights.mean(dim=(0, 1))
        total = float(importance.sum().item())
        if total <= 0.0:
            return {name: 0.0 for name in self.feature_names}
        normalized = importance / total
        return {
            name: float(normalized[index].item())
            for index, name in enumerate(self.feature_names)
        }

    def top_features(self, n: int = 20) -> list[tuple[str, float]]:
        """Return the top-n past features by mean VSN weight."""

        return sorted(
            self.past_importance.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:n]

    def family_importance(
        self,
        family_map: dict[str, list[str]],
    ) -> dict[str, float]:
        """Aggregate feature importance by family and renormalize across families."""

        per_feature = self.past_importance
        family_scores = {
            family: sum(per_feature.get(feature_name, 0.0) for feature_name in feature_names)
            for family, feature_names in family_map.items()
        }
        total = sum(family_scores.values())
        if total <= 0.0:
            return {family: 0.0 for family in family_map}
        return {family: score / total for family, score in family_scores.items()}

    def summary_str(self, n: int = 15) -> str:
        """Format the top features as a compact multi-line summary string."""

        lines = ["VSN top features:"]
        lines.extend(
            f"{feature_name}: {weight:.4f}"
            for feature_name, weight in self.top_features(n)
        )
        return "\n".join(lines)


@dataclass(slots=True)
class AttentionInspector:
    """Inspect temporal self-attention summaries when attention weights are available."""

    encoder_hidden: Tensor
    attention_weights: Tensor | None = None

    @staticmethod
    def from_output(output: "ModelOutput") -> "AttentionInspector":
        """Build an attention inspector from a model output."""

        if output.encoder_hidden is None:
            raise ValueError("ModelOutput does not include encoder_hidden.")

        raw_attention = output.attn_weights
        attention_weights = None
        if raw_attention is not None and "past" in raw_attention:
            attention_weights = raw_attention["past"].detach().float().cpu()

        return AttentionInspector(
            encoder_hidden=output.encoder_hidden.detach().float().cpu(),
            attention_weights=attention_weights,
        )

    @property
    def mean_attention(self) -> Tensor | None:
        """Return mean attention across batch and heads as a [T, T] tensor."""

        if self.attention_weights is None:
            return None
        if self.attention_weights.dim() == 4:
            return self.attention_weights.mean(dim=(0, 1))
        if self.attention_weights.dim() == 3:
            return self.attention_weights.mean(dim=0)
        raise ValueError(
            "Attention weights must be shaped [B, H, T, T] or [B, T, T], "
            f"got {tuple(self.attention_weights.shape)}.",
        )

    def last_timestep_attention(self) -> Tensor | None:
        """Return last-timestep attention over the lookback window as a [T] tensor."""

        mean_attention = self.mean_attention
        if mean_attention is None:
            return None
        return mean_attention[-1, :]


__all__ = ["AttentionInspector", "VSNInterpreter"]
