"""Typed contracts for feature registry entries."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FeatureSpec(BaseModel):
    """Declarative description of a point-in-time-safe feature family entry."""

    schema_version: str = Field(default="1.0.0")
    feature_name: str
    family: str
    version: str
    description: str | None = None
    input_contract: Literal["StateSnapshot", "BuiltBar"]
    input_clock: str
    output_clock: str
    builder_ref: str
    output_columns: list[str]
    dependencies: list[str]
    lookback_rows: int = 0
    warmup_rows: int = 0
    latency_class: Literal["online", "offline"] = "offline"
    point_in_time_safe: bool = True
    missingness_policy: Literal["fail", "allow", "drop_row", "impute_forward", "impute_zero"]
    qa_policy_ref: str
    status: Literal["draft", "stable", "deprecated"] = "draft"
    ablation_group: str | None = None
    trainability_tags: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.family}.{self.feature_name}@{self.version}"

    @model_validator(mode="after")
    def validate_spec(self) -> "FeatureSpec":
        if not self.output_columns:
            raise ValueError("output_columns must not be empty")
        if len(set(self.output_columns)) != len(self.output_columns):
            raise ValueError("output_columns must be unique")
        if not self.dependencies:
            raise ValueError("dependencies must not be empty")
        if self.lookback_rows < 0 or self.warmup_rows < 0:
            raise ValueError("lookback_rows and warmup_rows must be >= 0")
        if self.warmup_rows < self.lookback_rows:
            raise ValueError("warmup_rows must be >= lookback_rows")
        if len(set(self.trainability_tags)) != len(self.trainability_tags):
            raise ValueError("trainability_tags must be unique")
        return self
