"""Compatibility facade for feature family builders.

Prefer importing from:
- `mt5pipe.features.time`
- `mt5pipe.features.session`
- `mt5pipe.features.quality`
- `mt5pipe.features.context`
- `mt5pipe.features.disagreement`
- `mt5pipe.features.event_shape`
- `mt5pipe.features.entropy`
- `mt5pipe.features.multiscale`
"""

from __future__ import annotations

from mt5pipe.features.context import add_lagged_bar_features
from mt5pipe.features.disagreement import add_disagreement_features
from mt5pipe.features.entropy import add_entropy_features
from mt5pipe.features.event_shape import add_event_shape_features
from mt5pipe.features.multiscale import add_multiscale_features
from mt5pipe.features.quality import add_spread_quality_features
from mt5pipe.features.session import add_session_features
from mt5pipe.features.time import add_time_features

__all__ = [
    "add_time_features",
    "add_session_features",
    "add_spread_quality_features",
    "add_lagged_bar_features",
    "add_disagreement_features",
    "add_event_shape_features",
    "add_entropy_features",
    "add_multiscale_features",
]
