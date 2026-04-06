"""Lightning-style training module that turns the multi-head TFT into one research objective."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from machinelearning.data import DEFAULT_SCHEMA
from machinelearning.models import AphelionModel, ModelOutput
from machinelearning.training.losses import (
    HORIZON_LABELS,
    IGNORE_INDEX,
    LearnedTaskWeights,
    N_CLASSES,
    TARGET_HORIZONS_MINUTES,
    classification_loss,
    ic_loss,
    make_target_column,
    quantile_loss,
    regression_loss,
)

try:
    import lightning as pl
except ModuleNotFoundError:
    try:
        import pytorch_lightning as pl
    except ModuleNotFoundError:
        class _LightningModule(nn.Module):
            """Fallback LightningModule so the training logic stays unit-testable without Lightning installed."""

            def __init__(self) -> None:
                super().__init__()
                self.logged_values: dict[str, Tensor | float] = {}

            def log(self, name: str, value: Tensor | float, **_: Any) -> None:
                self.logged_values[name] = value.detach() if isinstance(value, Tensor) else value

            def save_hyperparameters(self, *args: Any, **kwargs: Any) -> None:
                self.hparams = {"args": args, "kwargs": kwargs}

        class _LightningNamespace:
            LightningModule = _LightningModule

        pl = _LightningNamespace()


class AphelionLightningModule(pl.LightningModule):
    """Coordinate the multi-head TFT losses so research compares directly to the published 60m baseline.

    The model predicts many related targets. This module is where we make those heads economically coherent:
    focal loss keeps direction predictions from collapsing to the dominant flat class, quantile loss preserves
    calibrated return ranges, and IC regularization steers the 60m return head toward usable ranking signal.
    """

    def __init__(
        self,
        model: AphelionModel,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        quantiles: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        warmup_pct: float = 0.05,
        total_steps: int = 10_000,
        ic_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.quantiles = list(quantiles)
        self.warmup_pct = float(warmup_pct)
        self.total_steps = max(int(total_steps), 1)
        self.ic_loss_weight = float(ic_loss_weight)
        self.task_weights = LearnedTaskWeights(n_tasks=20)
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters(ignore=["model"])

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        """Run the full multi-task objective each step so rare directional edges shape the shared encoder."""

        output = self.model(batch)
        loss = self._compute_loss(output, batch["targets"], "train")
        self._compute_metrics(output, batch["targets"], "train")
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Track validation IC and balanced accuracy because they are the closest proxies for deployable alpha."""

        output = self.model(batch)
        self._compute_loss(output, batch["targets"], "val")
        self._compute_metrics(output, batch["targets"], "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Mirror validation metrics on the held-out split so the best checkpoint is evaluated consistently."""

        output = self.model(batch)
        self._compute_loss(output, batch["targets"], "test")
        self._compute_metrics(output, batch["targets"], "test")

    def configure_optimizers(self) -> dict[str, Any]:
        """Use AdamW plus OneCycle because TFT training is sensitive to optimizer scale and warmup shape."""

        optimizer = AdamW(
            list(self.model.parameters()) + list(self.task_weights.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.total_steps,
            pct_start=self.warmup_pct,
            anneal_strategy="cos",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _compute_loss(self, output: ModelOutput, targets: dict[str, Tensor], stage: str) -> Tensor:
        task_losses: list[Tensor] = []
        for horizon_minutes, horizon_label in zip(TARGET_HORIZONS_MINUTES, HORIZON_LABELS, strict=True):
            direction_column = make_target_column("direction", horizon_minutes)
            tb_column = make_target_column("triple_barrier", horizon_minutes)
            return_column = make_target_column("future_return", horizon_minutes)
            mae_column = make_target_column("mae", horizon_minutes)
            mfe_column = make_target_column("mfe", horizon_minutes)

            direction_loss = classification_loss(
                output.direction_logits[horizon_label],
                targets[direction_column].long(),
            )
            tb_loss = classification_loss(
                output.tb_logits[horizon_label],
                targets[tb_column].long(),
            )
            return_loss = quantile_loss(
                output.return_preds[horizon_label],
                targets[return_column].float(),
                self.quantiles,
            )
            mae_loss = regression_loss(
                output.mae_preds[horizon_label].view(-1),
                targets[mae_column].float().view(-1),
            )
            mfe_loss = regression_loss(
                output.mfe_preds[horizon_label].view(-1),
                targets[mfe_column].float().view(-1),
            )

            task_losses.extend([direction_loss, tb_loss, return_loss, mae_loss, mfe_loss])
            self._log_metric(f"{stage}/loss_dir_{horizon_label}", direction_loss, stage)
            self._log_metric(f"{stage}/loss_tb_{horizon_label}", tb_loss, stage)
            self._log_metric(f"{stage}/loss_ret_{horizon_label}", return_loss, stage)
            self._log_metric(f"{stage}/loss_mae_{horizon_label}", mae_loss, stage)
            self._log_metric(f"{stage}/loss_mfe_{horizon_label}", mfe_loss, stage)

        total = self.task_weights(task_losses)
        ic_component = ic_loss(
            output.return_median("60m"),
            targets["future_return_60m"].float(),
        )
        total = total + (self.ic_loss_weight * ic_component)

        self._log_metric(f"{stage}/total_loss", total, stage)
        self._log_metric(f"{stage}/ic_loss_60m", ic_component, stage)
        return total

    def _compute_metrics(self, output: ModelOutput, targets: dict[str, Tensor], stage: str) -> None:
        for horizon_minutes, horizon_label in zip(TARGET_HORIZONS_MINUTES, HORIZON_LABELS, strict=True):
            target_column = make_target_column("direction", horizon_minutes)
            direction_target = targets[target_column].long()
            valid = direction_target != IGNORE_INDEX
            if torch.any(valid):
                predictions = output.direction_logits[horizon_label][valid].argmax(dim=-1)
                accuracy = (predictions == direction_target[valid]).float().mean()
                self._log_metric(f"{stage}/dir_acc_{horizon_label}", accuracy, stage)

                per_class = []
                for class_index in range(N_CLASSES):
                    class_mask = valid & (direction_target == class_index)
                    if torch.any(class_mask):
                        class_acc = (
                            output.direction_logits[horizon_label][class_mask].argmax(dim=-1)
                            == direction_target[class_mask]
                        ).float().mean()
                        per_class.append(class_acc)
                if per_class:
                    balanced_acc = torch.stack(per_class).mean()
                    self._log_metric(f"{stage}/balanced_acc_{horizon_label}", balanced_acc, stage)

            return_column = make_target_column("future_return", horizon_minutes)
            correlation = -ic_loss(
                output.return_median(horizon_label),
                targets[return_column].float(),
            )
            self._log_metric(f"{stage}/ic_{horizon_label}", correlation, stage)

    def _log_metric(self, name: str, value: Tensor, stage: str) -> None:
        self.log(
            name,
            value,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=(stage == "val"),
            sync_dist=True,
        )


__all__ = ["AphelionLightningModule"]
