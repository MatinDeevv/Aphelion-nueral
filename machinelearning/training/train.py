"""Training entry point for Phase 6 so runs are reproducible, logged, and comparable to the baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from machinelearning.data import AphelionDataModule, DEFAULT_SCHEMA
from machinelearning.models import AphelionTFT
from machinelearning.training.module import AphelionLightningModule

try:
    import lightning as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.strategies import DDPStrategy
except ModuleNotFoundError:
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
        from pytorch_lightning.loggers import WandbLogger
        from pytorch_lightning.strategies import DDPStrategy
    except ModuleNotFoundError:
        pl = None
        EarlyStopping = LearningRateMonitor = ModelCheckpoint = WandbLogger = DDPStrategy = None

BASELINE_BALANCED_ACC = 0.5354
BASELINE_HOLDOUT_ACC = 0.5096
DATASET_ARTIFACT_ID = "219cc1cdb344"
DATASET_TRUST_SCORE = 98.11


def train(
    artifact_dir: str | Path,
    run_name: str,
    project_name: str = "aphelion-research",
    d_model: int = 128,
    n_heads: int = 4,
    n_lstm_layers: int = 2,
    dropout: float = 0.1,
    context_len: int = 240,
    batch_size: int = 512,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    ic_loss_weight: float = 0.1,
    max_epochs: int = 50,
    n_gpus: int = 2,
    num_workers: int = 8,
    precision: str = "bf16-mixed",
    gradient_clip_val: float = 1.0,
    checkpoint_dir: str | Path = "checkpoints",
) -> None:
    """Run one end-to-end research training job so model quality, checkpoints, and metadata stay linked."""

    if pl is None or WandbLogger is None or ModelCheckpoint is None or EarlyStopping is None or DDPStrategy is None:
        raise RuntimeError("lightning and wandb must be installed to run machinelearning.training.train().")

    artifact_dir = Path(artifact_dir)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    normalizer_path = checkpoint_dir / f"{run_name}_normalizer.json"

    dm = AphelionDataModule(
        artifact_dir=artifact_dir,
        context_len=context_len,
        batch_size=batch_size,
        num_workers=num_workers,
        normalizer_save_path=normalizer_path,
    )
    dm.setup()

    model = AphelionTFT(
        n_past_features=DEFAULT_SCHEMA.n_past,
        n_future_features=DEFAULT_SCHEMA.n_future,
        n_static_features=DEFAULT_SCHEMA.n_static,
        d_model=d_model,
        n_heads=n_heads,
        n_lstm_layers=n_lstm_layers,
        dropout=dropout,
        context_len=context_len,
    )

    n_train_samples = len(dm.train_dataset) if dm.train_dataset is not None else 0
    setattr(dm, "n_train_samples", n_train_samples)
    steps_per_epoch = max(n_train_samples // batch_size, 1)
    total_steps = max(steps_per_epoch * max_epochs, 1)

    lightning_module = AphelionLightningModule(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        ic_loss_weight=ic_loss_weight,
        total_steps=total_steps,
    )

    wandb_logger = WandbLogger(
        project=project_name,
        name=run_name,
        log_model=True,
    )
    wandb_logger.experiment.config.update(
        {
            "artifact_dir": str(artifact_dir),
            "schema_version": DEFAULT_SCHEMA.version,
            "n_past": DEFAULT_SCHEMA.n_past,
            "n_future": DEFAULT_SCHEMA.n_future,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_lstm_layers": n_lstm_layers,
            "dropout": dropout,
            "context_len": context_len,
            "batch_size": batch_size,
            "lr": lr,
            "ic_loss_weight": ic_loss_weight,
            "max_epochs": max_epochs,
            "n_gpus": n_gpus,
            "precision": precision,
            "model_params": model.count_parameters(),
            "baseline_balanced_acc": BASELINE_BALANCED_ACC,
            "baseline_holdout_acc": BASELINE_HOLDOUT_ACC,
            "dataset_artifact_id": DATASET_ARTIFACT_ID,
            "dataset_trust_score": DATASET_TRUST_SCORE,
        }
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{run_name}-ep{{epoch:02d}}-ic{{val/ic_60m:.4f}}",
        monitor="val/ic_60m",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    callbacks = [
        checkpoint_callback,
        EarlyStopping(
            monitor="val/ic_60m",
            mode="max",
            patience=10,
            min_delta=1e-4,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer_kwargs = _build_trainer_kwargs(
        max_epochs=max_epochs,
        n_gpus=n_gpus,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        logger=wandb_logger,
        callbacks=callbacks,
    )
    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(lightning_module, dm)
    fit_metrics = {name: _metric_to_float(value) for name, value in trainer.callback_metrics.items()}
    trainer.test(lightning_module, dm, ckpt_path="best")

    best_ic = _metric_to_float(checkpoint_callback.best_model_score)
    if best_ic != best_ic:
        best_ic = fit_metrics.get("val/ic_60m", float("nan"))
    best_balanced_acc = fit_metrics.get("val/balanced_acc_60m", float("nan"))
    print(f"val/ic_60m:          {best_ic:.4f}")
    print(f"val/balanced_acc_60m:{best_balanced_acc:.4f}")
    print(f"baseline:             {BASELINE_BALANCED_ACC:.4f}")
    print(f"delta:               {best_balanced_acc - BASELINE_BALANCED_ACC:+.4f}")


def _build_trainer_kwargs(
    max_epochs: int,
    n_gpus: int,
    precision: str,
    gradient_clip_val: float,
    logger: Any,
    callbacks: list[Any],
) -> dict[str, Any]:
    """Prefer the requested GPU/DDP path, but fall back cleanly so local validation is still possible."""

    has_gpu = torch.cuda.is_available() and n_gpus > 0
    if has_gpu:
        return {
            "max_epochs": max_epochs,
            "accelerator": "gpu",
            "devices": n_gpus,
            "strategy": DDPStrategy(
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            ),
            "precision": precision,
            "gradient_clip_val": gradient_clip_val,
            "gradient_clip_algorithm": "norm",
            "logger": logger,
            "callbacks": callbacks,
            "log_every_n_steps": 50,
            "benchmark": True,
            "deterministic": False,
            "sync_batchnorm": True,
        }

    resolved_precision = precision if precision not in {"bf16-mixed", "16-mixed"} else "32-true"
    return {
        "max_epochs": max_epochs,
        "accelerator": "cpu",
        "devices": 1,
        "strategy": "auto",
        "precision": resolved_precision,
        "gradient_clip_val": gradient_clip_val,
        "gradient_clip_algorithm": "norm",
        "logger": logger,
        "callbacks": callbacks,
        "log_every_n_steps": 50,
        "benchmark": False,
        "deterministic": False,
        "sync_batchnorm": False,
    }


def _metric_to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase 6 Aphelion TFT.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project-name", default="aphelion-research")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-lstm-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--context-len", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ic-loss-weight", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--n-gpus", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    train(
        artifact_dir=arguments.artifact_dir,
        run_name=arguments.run_name,
        project_name=arguments.project_name,
        d_model=arguments.d_model,
        n_heads=arguments.n_heads,
        n_lstm_layers=arguments.n_lstm_layers,
        dropout=arguments.dropout,
        context_len=arguments.context_len,
        batch_size=arguments.batch_size,
        lr=arguments.lr,
        weight_decay=arguments.weight_decay,
        ic_loss_weight=arguments.ic_loss_weight,
        max_epochs=arguments.max_epochs,
        n_gpus=arguments.n_gpus,
        num_workers=arguments.num_workers,
        precision=arguments.precision,
        gradient_clip_val=arguments.gradient_clip_val,
        checkpoint_dir=arguments.checkpoint_dir,
    )
