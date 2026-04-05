"""Lightning data module for the Phase 6 machine learning parquet artifact."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

from .dataset import AphelionDataset
from .normalizer import RobustFeatureNormalizer
from .schema import DEFAULT_SCHEMA, ColumnSchema

try:
    import lightning as pl
except ModuleNotFoundError:
    try:
        import pytorch_lightning as pl
    except ModuleNotFoundError:
        class _LightningDataModule:
            """Fallback LightningDataModule used only in the local test environment."""

            pass

        class _LightningNamespace:
            LightningDataModule = _LightningDataModule

        pl = _LightningNamespace()

try:
    from torch.utils.data import DataLoader
except ModuleNotFoundError:
    from . import dataset as dataset_module

    class DataLoader:
        """Minimal DataLoader fallback used when torch is unavailable locally."""

        def __init__(
            self,
            dataset: AphelionDataset,
            batch_size: int = 1,
            shuffle: bool = False,
            drop_last: bool = False,
            collate_fn: Any = None,
            **_: Any,
        ) -> None:
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.drop_last = drop_last
            self.collate_fn = collate_fn

        def __iter__(self):
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                chunk = indices[start : start + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                batch = [self.dataset[index] for index in chunk]
                if self.collate_fn is None:
                    yield batch
                else:
                    yield self.collate_fn(batch)

        def __len__(self) -> int:
            if self.drop_last:
                return len(self.dataset) // self.batch_size
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    torch = dataset_module.torch
else:
    from . import dataset as dataset_module

    torch = dataset_module.torch


class AphelionDataModule(pl.LightningDataModule):
    """Lightning data module for the published XAUUSD parquet artifact."""

    def __init__(
        self,
        artifact_dir: str | Path,
        context_len: int = 240,
        batch_size: int = 512,
        num_workers: int = 8,
        stride_train: int = 1,
        schema: ColumnSchema = DEFAULT_SCHEMA,
        normalizer: RobustFeatureNormalizer | None = None,
        normalizer_save_path: str | Path | None = None,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.artifact_dir = Path(artifact_dir)
        self.context_len = context_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stride_train = stride_train
        self.schema = schema
        self.normalizer = normalizer or RobustFeatureNormalizer(schema=self.schema)
        self.normalizer_save_path = (
            None if normalizer_save_path is None else Path(normalizer_save_path)
        )
        self.pin_memory = pin_memory

        self.train_dataset: AphelionDataset | None = None
        self.val_dataset: AphelionDataset | None = None
        self.test_dataset: AphelionDataset | None = None
        self._is_setup = False

    def setup(self, stage: str | None = None) -> None:
        """Load split directories, fit the train-only normalizer, and build datasets."""

        if self._is_setup:
            return

        train_dir = self.artifact_dir / "split=train"
        val_dir = self.artifact_dir / "split=val"
        test_dir = self.artifact_dir / "split=test"
        for split_dir in (train_dir, val_dir, test_dir):
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Missing split directory: {split_dir}")

        train_unscaled = AphelionDataset(
            artifact_dir=train_dir,
            context_len=self.context_len,
            schema=self.schema,
            normalizer=None,
            stride=self.stride_train,
        )
        self.normalizer.fit(train_unscaled.raw_dataframe())
        if self.normalizer_save_path is not None:
            self.normalizer.save(self.normalizer_save_path)

        self.train_dataset = AphelionDataset(
            artifact_dir=train_dir,
            context_len=self.context_len,
            schema=self.schema,
            normalizer=self.normalizer,
            stride=self.stride_train,
        )
        self.val_dataset = AphelionDataset(
            artifact_dir=val_dir,
            context_len=self.context_len,
            schema=self.schema,
            normalizer=self.normalizer,
            stride=1,
        )
        self.test_dataset = AphelionDataset(
            artifact_dir=test_dir,
            context_len=self.context_len,
            schema=self.schema,
            normalizer=self.normalizer,
            stride=1,
        )
        self._is_setup = True

    def train_dataloader(self) -> DataLoader:
        """Return the shuffled training dataloader."""

        if self.train_dataset is None:
            raise RuntimeError("setup() must be called before requesting train_dataloader().")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_batch,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""

        if self.val_dataset is None:
            raise RuntimeError("setup() must be called before requesting val_dataloader().")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_batch,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""

        if self.test_dataset is None:
            raise RuntimeError("setup() must be called before requesting test_dataloader().")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_batch,
        )

    def _collate_batch(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Stack model inputs and nested target dictionaries into batch tensors."""

        target_names = list(batch[0]["targets"].keys())
        return {
            "past_features": torch.stack([sample["past_features"] for sample in batch], dim=0),
            "future_known": torch.stack([sample["future_known"] for sample in batch], dim=0),
            "static": torch.stack([sample["static"] for sample in batch], dim=0),
            "mask": torch.stack([sample["mask"] for sample in batch], dim=0),
            "time_idx": torch.stack([sample["time_idx"] for sample in batch], dim=0),
            "targets": {
                name: torch.stack([sample["targets"][name] for sample in batch], dim=0)
                for name in target_names
            },
        }


__all__ = ["AphelionDataModule"]
