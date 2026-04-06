"""Bridge the repository-level model contract tests into the machinelearning test suite."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT.parent))

from tests.test_ml_models import *  # noqa: F401,F403