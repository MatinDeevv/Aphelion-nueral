"""State-sector boundary safety tests."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_ROOT = REPO_ROOT / "mt5pipe" / "state"

FORBIDDEN_PREFIXES = [
    "mt5pipe.bars",
    "mt5pipe.quality",
    "mt5pipe.compiler",
    "mt5pipe.features",
    "mt5pipe.labels",
    "mt5pipe.catalog",
    "mt5pipe.truth",
]

ALLOWED_CROSS_SECTOR = {
    "mt5pipe.contracts",
    "mt5pipe.compiler.public",
    "mt5pipe.features.public",
    "mt5pipe.labels.public",
}


def _imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("mt5pipe."):
                    found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mt5pipe."):
            found.append((node.lineno, node.module))
    return found


def test_state_sector_has_no_forbidden_cross_imports() -> None:
    violations: list[str] = []
    for py_file in sorted(STATE_ROOT.rglob("*.py")):
        for lineno, module in _imports(py_file):
            if any(module == allowed or module.startswith(allowed + ".") for allowed in ALLOWED_CROSS_SECTOR):
                continue
            if any(module == prefix or module.startswith(prefix + ".") for prefix in FORBIDDEN_PREFIXES):
                violations.append(f"{py_file.relative_to(REPO_ROOT)}:{lineno} imports {module}")
    assert violations == []
