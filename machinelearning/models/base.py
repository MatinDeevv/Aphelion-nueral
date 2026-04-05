"""Base contracts shared by Phase 6 Aphelion models."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

HORIZONS = ["5m", "15m", "60m", "240m"]
N_CLASSES = 3
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


@dataclass(slots=True)
class ModelOutput:
    """Container for multi-head Aphelion model predictions and diagnostics."""

    direction_logits: dict[str, Tensor]
    tb_logits: dict[str, Tensor]
    return_preds: dict[str, Tensor]
    mae_preds: dict[str, Tensor]
    mfe_preds: dict[str, Tensor]
    encoder_hidden: Tensor | None = None
    vsn_weights: dict[str, Tensor] | None = None

    def direction_probs(self, horizon: str) -> Tensor:
        """Return normalized direction probabilities for a given horizon."""

        return self._get_tensor(self.direction_logits, horizon).softmax(dim=-1)

    def return_median(self, horizon: str) -> Tensor:
        """Return the median quantile prediction for a given horizon."""

        median_index = QUANTILES.index(0.5)
        return self._get_tensor(self.return_preds, horizon)[..., median_index]

    def return_interval(self, horizon: str, alpha: float = 0.8) -> tuple[Tensor, Tensor]:
        """Return an approximately central predictive interval for a given horizon."""

        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

        tail_mass = (1.0 - alpha) / 2.0
        lower_index = min(range(len(QUANTILES)), key=lambda index: abs(QUANTILES[index] - tail_mass))
        upper_target = 1.0 - tail_mass
        upper_index = min(range(len(QUANTILES)), key=lambda index: abs(QUANTILES[index] - upper_target))

        preds = self._get_tensor(self.return_preds, horizon)
        lower = preds[..., lower_index]
        upper = preds[..., upper_index]
        return torch.minimum(lower, upper), torch.maximum(lower, upper)

    @staticmethod
    def _get_tensor(mapping: dict[str, Tensor], horizon: str) -> Tensor:
        if horizon not in mapping:
            available = ", ".join(sorted(mapping))
            raise KeyError(f"Unknown horizon '{horizon}'. Available horizons: {available}")
        return mapping[horizon]


class AphelionModel(ABC, nn.Module):
    """Abstract base class for all trainable Aphelion neural models."""

    model_name: str
    model_version: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        for attribute in ("model_name", "model_version"):
            value = getattr(cls, attribute, None)
            if not isinstance(value, str) or not value:
                raise TypeError(f"{cls.__name__} must define a non-empty string '{attribute}'")

    @abstractmethod
    def forward(self, batch: dict[str, Any]) -> ModelOutput:
        """Run a batch through the model and return all prediction heads."""

    def count_parameters(self) -> int:
        """Return the number of trainable parameters in the model."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
