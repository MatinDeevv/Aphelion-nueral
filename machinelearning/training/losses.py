"""Loss functions for Phase 6 training, designed around market structure rather than generic ML defaults."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from machinelearning.data.schema import TARGET_HORIZONS_MINUTES, make_target_column

IGNORE_INDEX = -100
N_CLASSES = 3
HORIZON_LABELS = [f"{horizon}m" for horizon in TARGET_HORIZONS_MINUTES]


def quantile_loss(
    preds: Tensor,
    target: Tensor,
    quantiles: list[float],
) -> Tensor:
    """Penalize quantile errors so the model learns calibrated return ranges, not just a single point estimate.

    Gold returns are heteroskedastic: quiet sessions and macro-event bars live on very different scales.
    Pinball loss keeps each quantile honest across that distribution and ignores NaN targets so the model
    does not learn from rows whose forward horizon is structurally unavailable.
    """

    if preds.ndim != 2:
        raise ValueError(f"preds must be [batch, quantiles], got {tuple(preds.shape)}")
    if preds.size(1) != len(quantiles):
        raise ValueError(f"Expected {len(quantiles)} quantiles, received {preds.size(1)}")

    valid = torch.isfinite(target)
    if not torch.any(valid):
        return preds.new_zeros(())

    valid_preds = preds[valid]
    valid_target = target[valid].unsqueeze(-1)
    errors = valid_target - valid_preds
    q = torch.as_tensor(quantiles, device=preds.device, dtype=preds.dtype)
    losses = torch.maximum(q * errors, (q - 1.0) * errors)
    return losses.mean()


def classification_loss(
    logits: Tensor,
    target: Tensor,
) -> Tensor:
    """Use focal cross-entropy so the model keeps caring about rare directional moves instead of predicting flat.

    On M1 XAU data the flat class dominates. Plain cross-entropy rewards a model that predicts flat almost all
    the time, which looks statistically fine but has little trading value. Focal weighting suppresses easy
    already-correct examples and keeps optimization pressure on the minority move classes where alpha matters.
    """

    if logits.ndim != 2 or logits.size(-1) != N_CLASSES:
        raise ValueError(f"logits must be [batch, {N_CLASSES}], got {tuple(logits.shape)}")

    valid = target != IGNORE_INDEX
    if not torch.any(valid):
        return logits.new_zeros(())

    safe_target = target.clamp(min=0)
    ce = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX, reduction="none")
    probs = torch.softmax(logits, dim=-1)
    p_t = probs.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    focal_weight = (1.0 - p_t).pow(2).detach()
    return (ce[valid] * focal_weight[valid]).mean()


def regression_loss(
    preds: Tensor,
    target: Tensor,
) -> Tensor:
    """Use Huber loss so MAE/MFE heads respect tails without letting a few event spikes dominate training.

    Drawdown and excursion targets can jump hard around news. Huber keeps those heads sensitive to scale while
    remaining more robust than pure MSE, and the NaN mask avoids treating unavailable horizons as valid labels.
    """

    valid = torch.isfinite(target)
    if not torch.any(valid):
        return preds.new_zeros(())
    return F.huber_loss(preds[valid], target[valid], delta=1.0)


def ic_loss(
    return_preds: Tensor,
    return_targets: Tensor,
) -> Tensor:
    """Optimize negative Information Coefficient so the return head learns ranking power, not just magnitude fit.

    Trading systems care most about ordering opportunities correctly. A model that gets the rank ordering right
    can be useful even when absolute return magnitudes are noisy. Negative Pearson correlation makes higher IC
    lower loss, and the 10-sample floor avoids unstable tiny-batch correlations.
    """

    valid = torch.isfinite(return_preds) & torch.isfinite(return_targets)
    if int(valid.sum().item()) < 10:
        return return_preds.new_zeros(())

    preds = return_preds[valid]
    targets = return_targets[valid]
    preds_centered = preds - preds.mean()
    targets_centered = targets - targets.mean()
    preds_scale = torch.linalg.vector_norm(preds_centered)
    targets_scale = torch.linalg.vector_norm(targets_centered)
    denom = preds_scale * targets_scale
    if not torch.isfinite(denom) or float(denom.item()) <= 1e-12:
        return return_preds.new_zeros(())

    corr = torch.sum(preds_centered * targets_centered) / denom
    return -corr.clamp(min=-1.0, max=1.0)


class LearnedTaskWeights(nn.Module):
    """Learn uncertainty-based task weights so the 20-head objective can rebalance itself during training.

    Manual per-head tuning becomes brittle once direction, triple-barrier, return quantiles, MAE, and MFE are
    all trained together across four horizons. Homoscedastic uncertainty weighting lets the optimizer relax
    noisier heads and trust cleaner ones without hard-coding fragile scalar weights.
    """

    def __init__(self, n_tasks: int) -> None:
        super().__init__()
        if n_tasks <= 0:
            raise ValueError("n_tasks must be positive")
        self.n_tasks = int(n_tasks)
        self.log_sigma = nn.Parameter(torch.zeros(self.n_tasks))

    def forward(self, losses: list[Tensor]) -> Tensor:
        """Combine scalar task losses while keeping the weighting parameters trainable."""

        if len(losses) != self.n_tasks:
            raise ValueError(f"Expected {self.n_tasks} task losses, received {len(losses)}")

        weighted_terms = []
        for index, loss in enumerate(losses):
            sigma_term = torch.exp(2.0 * self.log_sigma[index])
            weighted_terms.append(loss / (2.0 * sigma_term) + self.log_sigma[index])
        return torch.stack(weighted_terms).sum()


__all__ = [
    "HORIZON_LABELS",
    "IGNORE_INDEX",
    "LearnedTaskWeights",
    "N_CLASSES",
    "TARGET_HORIZONS_MINUTES",
    "classification_loss",
    "ic_loss",
    "make_target_column",
    "quantile_loss",
    "regression_loss",
]
