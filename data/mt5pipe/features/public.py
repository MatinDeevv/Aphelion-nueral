"""
Features sector — public boundary module.

This is the ONLY module other sectors may import from the features/labels
packages. All cross-sector consumers should import from here, never from
``mt5pipe.features.builder``, ``mt5pipe.features.service``,
``mt5pipe.labels.service``, etc. directly.

Labels are part of the features sector (Agent 2 owns both).
``mt5pipe.labels.public`` is the intra-sector boundary; this module is the
cross-sector boundary for the combined features+labels surface.

Re-exports
----------
FeatureSpec, FeatureBuilder, FeatureArtifactRef : public feature contracts
FeatureService : feature materialization service
load_feature_artifact : persisted feature-view loader
get_default_feature_specs, resolve_feature_selectors : registry helpers
LabelPack, LabelArtifactRef : public label contracts
LabelService : label materialization service
load_label_artifact : persisted label-view loader
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mt5pipe.features.builder import (
    add_disagreement_features,
    add_entropy_features,
    add_event_shape_features,
    add_lagged_bar_features,
    add_multiscale_features,
    add_session_features,
    add_spread_quality_features,
    add_time_features,
)
from mt5pipe.features.artifacts import FeatureArtifactRef, load_feature_artifact
from mt5pipe.features.registry.defaults import get_default_feature_specs, resolve_feature_selectors
from mt5pipe.features.registry.models import FeatureSpec
from mt5pipe.features.types import FeatureBuilder
from mt5pipe.labels.artifacts import LabelArtifactRef, load_label_artifact
from mt5pipe.labels.registry.defaults import get_default_label_packs, resolve_label_pack
from mt5pipe.labels.registry.models import LabelPack

if TYPE_CHECKING:
    from mt5pipe.features.service import FeatureService
    from mt5pipe.labels.service import LabelService

__all__ = [
    # Feature models
    "FeatureSpec",
    "FeatureBuilder",
    "FeatureArtifactRef",
    # Feature services
    "FeatureService",
    "load_feature_artifact",
    # Feature registry
    "get_default_feature_specs",
    "resolve_feature_selectors",
    # Feature builders
    "add_time_features",
    "add_session_features",
    "add_spread_quality_features",
    "add_lagged_bar_features",
    "add_disagreement_features",
    "add_event_shape_features",
    "add_entropy_features",
    "add_multiscale_features",
    # Label models
    "LabelPack",
    "LabelArtifactRef",
    # Label services
    "LabelService",
    "load_label_artifact",
    # Label registry
    "get_default_label_packs",
    "resolve_label_pack",
]


def __getattr__(name: str):
    if name == "FeatureService":
        from mt5pipe.features.service import FeatureService as _FeatureService

        return _FeatureService
    if name == "LabelService":
        from mt5pipe.labels.service import LabelService as _LabelService

        return _LabelService
    raise AttributeError(name)
