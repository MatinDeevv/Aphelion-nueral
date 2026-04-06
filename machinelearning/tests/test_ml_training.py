"""Synthetic tests for the Phase 6 training stack."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT.parent))

from machinelearning.data import DEFAULT_SCHEMA
from machinelearning.models import AphelionTFT
from machinelearning.training import AphelionLightningModule, build_trainer, validate_artifact_dir
from machinelearning.training.losses import (
    HORIZON_LABELS,
    LearnedTaskWeights,
    classification_loss,
    ic_loss,
    make_target_column,
    quantile_loss,
    regression_loss,
)
from machinelearning.training.train import train


def test_quantile_loss_returns_zero_when_all_targets_are_nan() -> None:
    preds = torch.zeros(4, 5)
    target = torch.full((4,), float("nan"))
    loss = quantile_loss(preds, target, [0.1, 0.25, 0.5, 0.75, 0.9])
    assert loss.item() == 0.0


def test_quantile_loss_matches_known_pinball_value() -> None:
    preds = torch.tensor([[0.0]])
    target = torch.tensor([1.0])
    loss = quantile_loss(preds, target, [0.9])
    assert torch.isclose(loss, torch.tensor(0.9))


def test_classification_loss_ignores_missing_targets() -> None:
    logits = torch.randn(3, 3)
    target = torch.full((3,), -100, dtype=torch.int64)
    loss = classification_loss(logits, target)
    assert loss.item() == 0.0


def test_classification_loss_is_lower_for_confident_correct_predictions() -> None:
    easy_logits = torch.tensor([[-4.0, -4.0, 10.0]], dtype=torch.float32)
    hard_logits = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    target = torch.tensor([2], dtype=torch.int64)

    easy_loss = classification_loss(easy_logits, target)
    hard_loss = classification_loss(hard_logits, target)
    easy_ce = F.cross_entropy(easy_logits, target)

    assert easy_loss < hard_loss
    assert easy_loss < easy_ce


def test_ic_loss_is_near_negative_one_for_perfect_correlation() -> None:
    preds = torch.arange(1, 11, dtype=torch.float32)
    target = torch.arange(1, 11, dtype=torch.float32)
    loss = ic_loss(preds, target)
    assert loss.item() < -0.999


def test_ic_loss_returns_zero_for_too_few_valid_samples() -> None:
    preds = torch.arange(1, 6, dtype=torch.float32)
    target = torch.arange(1, 6, dtype=torch.float32)
    loss = ic_loss(preds, target)
    assert loss.item() == 0.0


def test_learned_task_weights_receive_gradients() -> None:
    weights = LearnedTaskWeights(n_tasks=3)
    losses = [
        torch.tensor(1.0, requires_grad=True),
        torch.tensor(2.0, requires_grad=True),
        torch.tensor(3.0, requires_grad=True),
    ]
    total = weights(losses)
    total.backward()
    assert weights.log_sigma.grad is not None
    assert torch.any(weights.log_sigma.grad != 0)


def test_learned_task_weights_with_zero_losses_returns_sum_of_log_sigma() -> None:
    weights = LearnedTaskWeights(n_tasks=2)
    with torch.no_grad():
        weights.log_sigma.copy_(torch.tensor([0.2, -0.1]))
    total = weights([torch.tensor(0.0), torch.tensor(0.0)])
    assert torch.isclose(total, torch.tensor(0.1))


def test_lightning_module_training_step_returns_scalar_tensor() -> None:
    module = AphelionLightningModule(
        model=AphelionTFT(
            n_past_features=DEFAULT_SCHEMA.n_past,
            n_future_features=DEFAULT_SCHEMA.n_future,
            n_static_features=DEFAULT_SCHEMA.n_static,
            d_model=16,
            n_heads=4,
            n_lstm_layers=1,
            dropout=0.1,
            context_len=8,
        ),
        total_steps=20,
    )
    batch = _synthetic_batch(batch_size=4, context_len=8)
    loss = module.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_configure_optimizers_returns_expected_structure() -> None:
    module = AphelionLightningModule(
        model=AphelionTFT(
            n_past_features=DEFAULT_SCHEMA.n_past,
            n_future_features=DEFAULT_SCHEMA.n_future,
            n_static_features=DEFAULT_SCHEMA.n_static,
            d_model=16,
            n_heads=4,
            n_lstm_layers=1,
            dropout=0.1,
            context_len=8,
        ),
        total_steps=25,
    )
    config = module.configure_optimizers()
    assert "optimizer" in config
    assert "lr_scheduler" in config
    assert config["lr_scheduler"]["interval"] == "step"
    assert config["lr_scheduler"]["frequency"] == 1

    optimizer_params = {
        id(parameter)
        for group in config["optimizer"].param_groups
        for parameter in group["params"]
    }
    assert id(module.task_weights.log_sigma) in optimizer_params


def test_build_trainer_cpu_safe() -> None:
    trainer = build_trainer(
        max_epochs=2,
        n_gpus=0,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
    )

    assert trainer.aphelion_config["accelerator"] == "cpu"
    assert trainer.aphelion_config["devices"] == 1


def test_build_trainer_bf16_requires_gpu() -> None:
    trainer = build_trainer(
        max_epochs=2,
        n_gpus=0,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
    )

    assert trainer.aphelion_config["precision"] == "32"


def test_validate_artifact_dir_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        validate_artifact_dir(Path("/nonexistent/path"))


def test_train_function_signature() -> None:
    signature = inspect.signature(train)
    parameters = signature.parameters

    assert list(parameters) == [
        "artifact_dir",
        "run_name",
        "project_name",
        "d_model",
        "n_heads",
        "n_lstm_layers",
        "dropout",
        "context_len",
        "batch_size",
        "lr",
        "weight_decay",
        "ic_loss_weight",
        "max_epochs",
        "n_gpus",
        "num_workers",
        "precision",
        "gradient_clip_val",
        "checkpoint_dir",
    ]
    assert parameters["artifact_dir"].default is inspect._empty
    assert parameters["run_name"].default is inspect._empty
    assert parameters["project_name"].default == "aphelion-research"
    assert parameters["precision"].default == "bf16-mixed"
    assert parameters["checkpoint_dir"].default == "checkpoints"


def _synthetic_batch(batch_size: int, context_len: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    batch: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
        "past_features": torch.randn(batch_size, context_len, DEFAULT_SCHEMA.n_past),
        "future_known": torch.randn(batch_size, context_len, DEFAULT_SCHEMA.n_future),
        "static": torch.zeros(batch_size, DEFAULT_SCHEMA.n_static),
        "mask": torch.ones(batch_size, context_len, dtype=torch.bool),
        "time_idx": torch.arange(batch_size, dtype=torch.int64),
    }
    targets: dict[str, torch.Tensor] = {}
    for horizon_label in HORIZON_LABELS:
        horizon_minutes = int(horizon_label[:-1])
        direction_column = make_target_column("direction", horizon_minutes)
        tb_column = make_target_column("triple_barrier", horizon_minutes)
        return_column = make_target_column("future_return", horizon_minutes)
        mae_column = make_target_column("mae", horizon_minutes)
        mfe_column = make_target_column("mfe", horizon_minutes)

        targets[direction_column] = torch.tensor([0, 1, 2, -100], dtype=torch.int64)
        targets[tb_column] = torch.tensor([2, 1, 0, -100], dtype=torch.int64)
        targets[return_column] = torch.tensor([0.1, -0.2, 0.3, float("nan")], dtype=torch.float32)
        targets[mae_column] = torch.tensor([0.5, 0.2, 0.1, float("nan")], dtype=torch.float32)
        targets[mfe_column] = torch.tensor([0.4, 0.6, 0.3, float("nan")], dtype=torch.float32)

    batch["targets"] = targets
    return batch
