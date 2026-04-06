"""Zero-retrain feature-family ablation utilities for TFT interpretability."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import torch
from torch import Tensor

from machinelearning.data.schema import (
    BASE_BAR_COLUMNS,
    DEFAULT_SCHEMA,
    DISAGREEMENT_FAMILY_COLUMNS,
    ENTROPY_FAMILY_COLUMNS,
    EVENT_SHAPE_FAMILY_COLUMNS,
    HTF_CONTEXT_COLUMNS,
    PAST_OBSERVED_COLUMNS,
    QUALITY_FAMILY_COLUMNS,
    SESSION_FAMILY_COLUMNS,
    TIME_FAMILY_COLUMNS,
    MULTISCALE_FAMILY_COLUMNS,
)

from .base import AphelionModel
from .interpret import VSNInterpreter


def _filter_past_columns(columns: list[str]) -> list[str]:
    """Return only the columns from a family that are part of ``past_observed``."""

    past_set = set(DEFAULT_SCHEMA.past_observed)
    return [column for column in columns if column in past_set]


FEATURE_FAMILY_MAP: dict[str, list[str]] = {
    "time": _filter_past_columns(TIME_FAMILY_COLUMNS),
    "session": _filter_past_columns(SESSION_FAMILY_COLUMNS),
    "quality": _filter_past_columns(QUALITY_FAMILY_COLUMNS),
    "htf_context": _filter_past_columns(HTF_CONTEXT_COLUMNS),
    "disagreement": _filter_past_columns(DISAGREEMENT_FAMILY_COLUMNS),
    "event_shape": _filter_past_columns(EVENT_SHAPE_FAMILY_COLUMNS),
    "entropy": _filter_past_columns(ENTROPY_FAMILY_COLUMNS),
    "multiscale": _filter_past_columns(MULTISCALE_FAMILY_COLUMNS),
    "base_bar": _filter_past_columns(BASE_BAR_COLUMNS),
}
PAST_FEATURE_INDEX = {column: index for index, column in enumerate(PAST_OBSERVED_COLUMNS)}


@dataclass(frozen=True, slots=True)
class AblationConfig:
    """Define one zero-retrain feature-family ablation experiment."""

    name: str
    excluded_families: list[str]
    description: str

    @property
    def active_columns(self) -> list[str]:
        """Return ``past_observed`` minus all columns owned by excluded families."""

        self._validate_families()
        excluded_columns = {
            column
            for family_name in self.excluded_families
            for column in FEATURE_FAMILY_MAP[family_name]
        }
        return [column for column in PAST_OBSERVED_COLUMNS if column not in excluded_columns]

    def _validate_families(self) -> None:
        """Raise when a config references an unknown feature family."""

        unknown = sorted(set(self.excluded_families) - set(FEATURE_FAMILY_MAP))
        if unknown:
            raise ValueError(
                f"Unknown ablation families: {', '.join(unknown)}. "
                f"Available families: {', '.join(FEATURE_FAMILY_MAP)}.",
            )


STANDARD_ABLATIONS: list[AblationConfig] = [
    AblationConfig(
        name="full",
        excluded_families=[],
        description="All features — baseline",
    ),
    AblationConfig(
        name="no_disagreement",
        excluded_families=["disagreement"],
        description="Remove dual-broker microstructure edge",
    ),
    AblationConfig(
        name="no_entropy",
        excluded_families=["entropy"],
        description="Remove complexity/entropy features",
    ),
    AblationConfig(
        name="no_multiscale",
        excluded_families=["multiscale"],
        description="Remove cross-timeframe coherence",
    ),
    AblationConfig(
        name="calendar_only",
        excluded_families=["disagreement", "entropy", "multiscale", "event_shape", "quality"],
        description="Only time and session features — sanity floor",
    ),
    AblationConfig(
        name="no_htf_context",
        excluded_families=["htf_context"],
        description="Remove higher-timeframe lagged bars",
    ),
]


@dataclass(frozen=True, slots=True)
class AblationResult:
    """Capture the output of one ablation run."""

    config: AblationConfig
    n_features: int
    vsn_importance: dict[str, float]
    family_importance: dict[str, float]
    top_features: list[tuple[str, float]]

    def vs_baseline(
        self,
        baseline: "AblationResult",
    ) -> dict[str, float]:
        """Return per-family importance deltas versus a baseline result."""

        family_names = list(dict.fromkeys([*baseline.family_importance.keys(), *self.family_importance.keys()]))
        return {
            family_name: self.family_importance.get(family_name, 0.0) - baseline.family_importance.get(family_name, 0.0)
            for family_name in family_names
        }


class AblationRunner:
    """Run zero-retrain feature-family ablations against a trained Aphelion model."""

    def __init__(self, model: AphelionModel) -> None:
        self.model = model

    def run_single(
        self,
        batch: dict[str, Tensor],
        config: AblationConfig,
    ) -> AblationResult:
        """Clone a batch, zero the ablated past-feature columns, and summarize VSN importance."""

        config._validate_families()
        ablated_batch = self._clone_batch(batch)
        past_features = ablated_batch.get("past_features")
        if not isinstance(past_features, Tensor):
            raise ValueError("AblationRunner requires batch['past_features'] as a tensor.")
        if past_features.dim() != 3:
            raise ValueError(
                f"batch['past_features'] must be [batch, time, features], got {tuple(past_features.shape)}.",
            )
        if past_features.size(-1) != len(PAST_OBSERVED_COLUMNS):
            raise ValueError(
                "AblationRunner expects full-schema past_features shaped "
                f"[B, T, {len(PAST_OBSERVED_COLUMNS)}], got {tuple(past_features.shape)}.",
            )

        excluded_indices = [
            PAST_FEATURE_INDEX[column]
            for family_name in config.excluded_families
            for column in FEATURE_FAMILY_MAP[family_name]
        ]
        if excluded_indices:
            past_features[..., excluded_indices] = 0.0

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model(ablated_batch)
        finally:
            if was_training:
                self.model.train()

        interpreter = VSNInterpreter.from_output(output, DEFAULT_SCHEMA.past_observed)
        return AblationResult(
            config=config,
            n_features=len(config.active_columns),
            vsn_importance=interpreter.past_importance,
            family_importance=interpreter.family_importance(FEATURE_FAMILY_MAP),
            top_features=interpreter.top_features(),
        )

    def run_all(
        self,
        batch: dict[str, Tensor],
        ablation_configs: list[AblationConfig] | None = None,
    ) -> list[AblationResult]:
        """Run all ablation configs in order and return the result list."""

        configs = STANDARD_ABLATIONS if ablation_configs is None else ablation_configs
        return [self.run_single(batch, config) for config in configs]

    def comparison_table(
        self,
        results: list[AblationResult],
    ) -> pl.DataFrame:
        """Return a family-by-ablation comparison table of mean VSN importance."""

        if not results:
            return pl.DataFrame({"family_name": []})

        rows: list[dict[str, float | str]] = []
        for family_name in FEATURE_FAMILY_MAP:
            row: dict[str, float | str] = {"family_name": family_name}
            for result in results:
                row[result.config.name] = float(result.family_importance.get(family_name, 0.0))
            rows.append(row)
        return pl.DataFrame(rows)

    def print_summary(self, results: list[AblationResult]) -> None:
        """Print a compact ablation summary table for operator logs."""

        table = self.comparison_table(results)
        if table.is_empty():
            print("=== Feature Family Ablation Summary ===")
            print("No ablation results available.")
            return

        config_names = [result.config.name for result in results]
        family_names = table.get_column("family_name").to_list()
        family_width = max(len("Family"), max(len(name) for name in family_names))
        value_width = max(8, max(len(name) for name in config_names))

        header = f"{'Family':<{family_width}} " + " ".join(
            f"{name:>{value_width}}" for name in config_names
        )
        separator = f"{'-' * family_width} " + " ".join("-" * value_width for _ in config_names)

        print("=== Feature Family Ablation Summary ===")
        print(header)
        print(separator)
        for row in table.iter_rows(named=True):
            print(
                f"{row['family_name']:<{family_width}} "
                + " ".join(f"{float(row[name]):>{value_width}.3f}" for name in config_names)
            )
        print("")
        print("Key question: does disagreement/* rank above time/* in the full model?")

    @staticmethod
    def _clone_batch(value: object) -> object:
        """Clone tensor batches recursively without mutating caller-owned inputs."""

        if isinstance(value, Tensor):
            return value.clone()
        if isinstance(value, dict):
            return {key: AblationRunner._clone_batch(child) for key, child in value.items()}
        if isinstance(value, list):
            return [AblationRunner._clone_batch(child) for child in value]
        if isinstance(value, tuple):
            return tuple(AblationRunner._clone_batch(child) for child in value)
        return value


__all__ = [
    "AblationConfig",
    "AblationResult",
    "AblationRunner",
    "STANDARD_ABLATIONS",
]
