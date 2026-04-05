"""Typed contracts for the dataset compiler."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


ArtifactStatus = Literal["building", "truth_pending", "accepted", "rejected", "published", "superseded", "trial"]
ArtifactKind = Literal["state", "feature_view", "label_view", "dataset", "experiment", "model"]


class DatasetSpec(BaseModel):
    """Compiler input spec for a dataset artifact."""

    schema_version: str = Field(default="1.0.0")
    dataset_name: str
    version: str
    description: str | None = None
    symbols: list[str]
    date_from: dt.date
    date_to: dt.date
    base_clock: str
    state_version_ref: str | None = None
    state_artifact_ref: str | None = None
    feature_selectors: list[str]
    feature_artifact_refs: list[str] = Field(default_factory=list)
    label_pack_ref: str | None = None
    label_artifact_ref: str | None = None
    required_raw_brokers: list[str] = Field(default_factory=list)
    require_synchronized_raw_coverage: bool = False
    require_dual_source_overlap: bool = False
    min_dual_source_ratio: float = 0.0
    filters: list[str] = Field(default_factory=list)
    split_policy: Literal["temporal_holdout", "walk_forward"]
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    embargo_rows: int
    n_walk_forward_splits: int | None = None
    truth_policy_ref: str
    publish_on_accept: bool = True
    tags: list[str] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.dataset_name}@{self.version}"

    @model_validator(mode="after")
    def validate_spec(self) -> "DatasetSpec":
        if not self.symbols:
            raise ValueError("symbols must not be empty")
        if self.date_from > self.date_to:
            raise ValueError("date_from must be <= date_to")
        if not self.feature_selectors:
            raise ValueError("feature_selectors must not be empty")
        if not self.state_version_ref and not self.state_artifact_ref:
            raise ValueError("either state_version_ref or state_artifact_ref is required")
        if not self.label_pack_ref and not self.label_artifact_ref:
            raise ValueError("either label_pack_ref or label_artifact_ref is required")
        if self.embargo_rows <= 0:
            raise ValueError("embargo_rows must be > 0")
        self.required_raw_brokers = sorted({broker.strip() for broker in self.required_raw_brokers if broker.strip()})
        if self.min_dual_source_ratio < 0.0 or self.min_dual_source_ratio > 1.0:
            raise ValueError("min_dual_source_ratio must be within [0, 1]")
        if (self.require_synchronized_raw_coverage or self.require_dual_source_overlap) and not self.required_raw_brokers:
            raise ValueError(
                "required_raw_brokers must be provided when synchronized coverage or dual-source overlap is required"
            )
        if self.require_dual_source_overlap and len(self.required_raw_brokers) < 2:
            raise ValueError("require_dual_source_overlap requires at least two required_raw_brokers")
        if self.split_policy == "temporal_holdout":
            total = self.train_ratio + self.val_ratio + self.test_ratio
            if abs(total - 1.0) > 1e-9:
                raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        if self.split_policy == "walk_forward" and not self.n_walk_forward_splits:
            raise ValueError("n_walk_forward_splits is required for walk_forward specs")
        return self


class ExperimentSpec(BaseModel):
    """Compiler-owned training/evaluation spec for trusted dataset artifacts."""

    schema_version: str = Field(default="1.0.0")
    experiment_name: str
    model_name: str
    version: str
    description: str | None = None
    dataset_ref: str
    target_column: str
    feature_families: list[str] = Field(default_factory=list)
    exclude_feature_columns: list[str] = Field(default_factory=list)
    model_family: Literal["gaussian_nb_binary@1.0.0"] = "gaussian_nb_binary@1.0.0"
    positive_target_threshold: float = 0.0
    decision_threshold: float = 0.5
    evaluation_policy: Literal["walk_forward_holdout"] = "walk_forward_holdout"
    n_walk_forward_folds: int = 3
    min_train_rows: int = 2000
    embargo_rows: int | None = None
    min_walk_forward_balanced_accuracy: float = 0.50
    min_test_balanced_accuracy: float = 0.50
    tags: list[str] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.experiment_name}@{self.version}"

    @model_validator(mode="after")
    def validate_spec(self) -> "ExperimentSpec":
        if not self.dataset_ref.strip():
            raise ValueError("dataset_ref must not be empty")
        if not self.target_column.strip():
            raise ValueError("target_column must not be empty")
        if self.n_walk_forward_folds <= 0:
            raise ValueError("n_walk_forward_folds must be > 0")
        if self.min_train_rows <= 0:
            raise ValueError("min_train_rows must be > 0")
        if self.embargo_rows is not None and self.embargo_rows < 0:
            raise ValueError("embargo_rows must be >= 0 when provided")
        for field_name in [
            "decision_threshold",
            "min_walk_forward_balanced_accuracy",
            "min_test_balanced_accuracy",
        ]:
            value = float(getattr(self, field_name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{field_name} must be within [0, 1]")
        return self


class LineageManifest(BaseModel):
    """Immutable lineage record for a compiled artifact."""

    schema_version: str = Field(default="1.0.0")
    manifest_id: str
    artifact_id: str
    artifact_kind: ArtifactKind
    logical_name: str
    logical_version: str
    artifact_uri: str
    content_hash: str
    build_id: str
    created_at: dt.datetime
    status: ArtifactStatus
    dataset_spec_ref: str | None = None
    experiment_spec_ref: str | None = None
    state_artifact_refs: list[str] = Field(default_factory=list)
    feature_spec_refs: list[str] = Field(default_factory=list)
    label_pack_ref: str | None = None
    truth_report_ref: str | None = None
    code_version: str
    merge_config_ref: str | None = None
    input_partition_refs: list[str] = Field(default_factory=list)
    parent_artifact_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_manifest(self) -> "LineageManifest":
        if not self.content_hash:
            raise ValueError("content_hash must not be empty")
        if not self.input_partition_refs:
            raise ValueError("input_partition_refs must not be empty")
        if self.status == "published" and not self.truth_report_ref:
            raise ValueError("published manifests must reference a truth report")
        return self
