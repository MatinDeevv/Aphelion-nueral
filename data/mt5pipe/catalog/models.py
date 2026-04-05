"""Typed models for compiler catalog records."""

from __future__ import annotations

import datetime as dt
from typing import Any

from mt5pipe.compiler.models import ArtifactKind, ArtifactStatus

from pydantic import BaseModel, Field


class BuildRunRecord(BaseModel):
    build_id: str
    dataset_spec_key: str
    status: str
    code_version: str
    started_at: dt.datetime
    finished_at: dt.datetime | None = None
    error_message: str = ""
    artifact_id: str | None = None


class ArtifactRecord(BaseModel):
    artifact_id: str
    artifact_kind: ArtifactKind
    logical_name: str
    logical_version: str
    artifact_uri: str
    manifest_uri: str
    content_hash: str
    status: ArtifactStatus
    build_id: str
    created_at: dt.datetime


class ArtifactInputRecord(BaseModel):
    artifact_id: str
    input_kind: str
    input_ref: str
    role: str
    ordinal: int = 0


class AliasRecord(BaseModel):
    alias_key: str
    artifact_id: str
    alias_type: str
    created_at: dt.datetime


class ArtifactStatusEventRecord(BaseModel):
    event_id: int
    artifact_id: str
    status: ArtifactStatus
    created_at: dt.datetime
    detail: str = ""


class TrainingRunRecord(BaseModel):
    run_id: str
    experiment_spec_key: str
    dataset_artifact_id: str
    status: str
    code_version: str
    started_at: dt.datetime
    finished_at: dt.datetime | None = None
    error_message: str = ""
    experiment_artifact_id: str | None = None
    model_artifact_id: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
