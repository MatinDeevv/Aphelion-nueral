"""
State sector public boundary.

Other sectors may import from here, but not from ``mt5pipe.state`` internals.
"""

from mt5pipe.contracts.state import (
    StateArtifactRef,
    StateWindowArtifactRef,
    StateWindowRequest,
    TickArtifactRef,
    parse_window_size,
)
from mt5pipe.state.models import (
    StateArtifactManifest,
    StateCoverageSummary,
    StateIntervalReadinessSummary,
    StateReadinessSummary,
    StateSnapshot,
    StateSourceQualitySummary,
    StateWindowRecord,
)
from mt5pipe.state.service import (
    StateMaterializationResult,
    StateService,
    StateWindowMaterializationResult,
    load_state_artifact,
    load_state_window_artifact,
    materialize_state_windows,
)

StateBuilder = StateService

__all__ = [
    "StateArtifactManifest",
    "StateArtifactRef",
    "StateBuilder",
    "StateCoverageSummary",
    "StateIntervalReadinessSummary",
    "StateMaterializationResult",
    "StateReadinessSummary",
    "StateService",
    "StateSnapshot",
    "StateSourceQualitySummary",
    "StateWindowArtifactRef",
    "StateWindowMaterializationResult",
    "StateWindowRecord",
    "StateWindowRequest",
    "TickArtifactRef",
    "load_state_artifact",
    "load_state_window_artifact",
    "materialize_state_windows",
    "parse_window_size",
]
