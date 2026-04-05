"""
Labels sector — public boundary module.

This is the ONLY module other sectors may import from the labels package.
All cross-sector consumers should import from here, never from
``mt5pipe.labels.service`` or ``mt5pipe.labels.registry.models`` directly.

Re-exports
----------
LabelPack, LabelArtifactRef : public label contracts
LabelService : label materialization service
load_label_artifact : persisted label-view loader
get_default_label_packs, resolve_label_pack : registry helpers
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mt5pipe.labels.artifacts import LabelArtifactRef, load_label_artifact
from mt5pipe.labels.registry.defaults import get_default_label_packs, resolve_label_pack
from mt5pipe.labels.registry.models import LabelPack

if TYPE_CHECKING:
    from mt5pipe.labels.service import LabelService

__all__ = [
    "LabelPack",
    "LabelArtifactRef",
    "LabelService",
    "load_label_artifact",
    "get_default_label_packs",
    "resolve_label_pack",
]


def __getattr__(name: str):
    if name == "LabelService":
        from mt5pipe.labels.service import LabelService as _LabelService

        return _LabelService
    raise AttributeError(name)
