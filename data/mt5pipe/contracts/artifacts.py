"""
Artifact reference contracts.

These types are the shared vocabulary for referencing artifacts across sectors.
State, features, and compiler all use ArtifactRef to point at artifacts
without depending on each other's internals.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ArtifactKind(str, Enum):
    """Artifact taxonomy shared across sectors."""

    CANONICAL_TICK = "canonical_tick"
    STATE = "state"
    STATE_WINDOW = "state_window"
    FEATURE_VIEW = "feature_view"
    LABEL_VIEW = "label_view"
    DATASET = "dataset"


class ArtifactRef(BaseModel):
    """
    Lightweight immutable reference to a pipeline artifact.

    The ``artifact_id`` format follows the existing convention:
        ``{kind}.{logical_name}.{content_hash[:12]}``
    """

    artifact_id: str = Field(..., min_length=1, description="Unique artifact identifier")
    kind: ArtifactKind
    logical_name: str = Field(..., min_length=1)
    version: str | None = None
    content_hash: str | None = Field(
        None,
        description="SHA-256 hash prefix or full hash for integrity verification",
    )

    def __str__(self) -> str:
        return self.artifact_id
