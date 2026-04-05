"""
Lineage contracts shared across sectors.

Provides a lightweight lineage node that sectors attach to their
artifacts without importing the full LineageManifest.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class LineageNode(BaseModel):
    """
    Minimal lineage reference attached to an artifact.

    Sectors use this when recording upstream dependencies.
    The full LineageManifest (compiler.models) extends this with
    richer metadata.
    """

    artifact_id: str = Field(..., min_length=1)
    parent_ids: list[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    code_version: Optional[str] = None
