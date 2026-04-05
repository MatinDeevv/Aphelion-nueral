"""Typed contracts for truth-layer outputs."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class QaCheckResult(BaseModel):
    """Result of a single truth-layer check."""

    check_name: str
    status: Literal["passed", "failed", "warning"]
    score: float
    metrics: dict[str, Any] = Field(default_factory=dict)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str = ""

    @model_validator(mode="after")
    def validate_check(self) -> "QaCheckResult":
        if not (0.0 <= self.score <= 100.0):
            raise ValueError("score must be within [0, 100]")
        if self.status == "failed" and not self.failure_reason:
            raise ValueError("failed checks must provide a failure_reason")
        return self


class TrustReport(BaseModel):
    """Publication-gating trust assessment for a candidate artifact."""

    schema_version: str = Field(default="1.0.0")
    report_id: str
    artifact_id: str
    artifact_kind: Literal["state", "feature_view", "label_view", "dataset"]
    truth_policy_version: str
    status: Literal["accepted", "rejected", "warning"]
    accepted_for_publication: bool
    trust_score_total: float
    coverage_score: float
    leakage_score: float
    feature_quality_score: float
    label_quality_score: float
    source_quality_score: float
    lineage_score: float
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    hard_failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    warning_reasons: list[str] = Field(default_factory=list)
    check_status_counts: dict[str, int] = Field(default_factory=dict)
    decision_summary: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    generated_at: dt.datetime
    checks: list[QaCheckResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_report(self) -> "TrustReport":
        for score in [
            self.trust_score_total,
            self.coverage_score,
            self.leakage_score,
            self.feature_quality_score,
            self.label_quality_score,
            self.source_quality_score,
            self.lineage_score,
        ]:
            if not (0.0 <= score <= 100.0):
                raise ValueError("all scores must be within [0, 100]")
        if self.hard_failures and self.accepted_for_publication:
            raise ValueError("accepted_for_publication must be false when hard_failures are present")
        if self.status == "accepted" and not self.accepted_for_publication:
            raise ValueError("accepted reports must be publishable")
        return self
