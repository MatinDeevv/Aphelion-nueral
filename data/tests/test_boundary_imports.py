"""
Architecture boundary enforcement tests.

Target rule (from docs/parallel_dev_architecture.md):
    Cross-sector imports may ONLY target:
        - mt5pipe.contracts.*
        - mt5pipe.state.public
        - mt5pipe.features.public
        - mt5pipe.labels.public
        - mt5pipe.compiler.public

    Cross-sector imports must NOT use sector package roots:
        - mt5pipe.state
        - mt5pipe.features
        - mt5pipe.labels
        - mt5pipe.compiler

Sector ownership:
    state    → mt5pipe/state/
    features → mt5pipe/features/, mt5pipe/labels/
    compiler → mt5pipe/compiler/, mt5pipe/truth/, mt5pipe/catalog/

Shared / neutral (not sector-gated):
    mt5pipe/contracts/, mt5pipe/models/, mt5pipe/config/,
    mt5pipe/storage/, mt5pipe/quality/, mt5pipe/merge/,
    mt5pipe/mt5/, mt5pipe/ingestion/, mt5pipe/backfill/,
    mt5pipe/bars/, mt5pipe/live/, mt5pipe/cli/, mt5pipe/utils/,
    mt5pipe/tools/
"""

from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
from typing import NamedTuple

import pytest

# ---------------------------------------------------------------------------
# Sector definitions
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
MT5PIPE = REPO_ROOT / "mt5pipe"

# Map: sector name → list of package prefixes owned by that sector
SECTOR_PACKAGES: dict[str, list[str]] = {
    "state": ["mt5pipe.state"],
    "features": ["mt5pipe.features", "mt5pipe.labels"],
    "compiler": ["mt5pipe.compiler", "mt5pipe.truth", "mt5pipe.catalog"],
}

# Allowed cross-sector import targets (always OK to import)
ALLOWED_CROSS_SECTOR: set[str] = {
    "mt5pipe.contracts",
    "mt5pipe.state.public",
    "mt5pipe.features.public",
    "mt5pipe.labels.public",
    "mt5pipe.compiler.public",
}

# Neutral packages — not owned by any sector, anyone can import
NEUTRAL_PREFIXES: list[str] = [
    "mt5pipe.contracts",
    "mt5pipe.models",
    "mt5pipe.config",
    "mt5pipe.storage",
    "mt5pipe.quality",
    "mt5pipe.merge",
    "mt5pipe.mt5",
    "mt5pipe.ingestion",
    "mt5pipe.backfill",
    "mt5pipe.bars",
    "mt5pipe.live",
    "mt5pipe.cli",
    "mt5pipe.utils",
    "mt5pipe.tools",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ImportViolation(NamedTuple):
    file: str
    line: int
    source_sector: str
    target_module: str
    target_sector: str


class RootImportViolation(NamedTuple):
    file: str
    line: int
    source_sector: str
    target_module: str
    target_sector: str


def _sector_of(module: str) -> str | None:
    """Return the sector name that owns *module*, or None if neutral."""
    for sector, prefixes in SECTOR_PACKAGES.items():
        for prefix in prefixes:
            if module == prefix or module.startswith(prefix + "."):
                return sector
    return None


def _is_allowed_target(module: str) -> bool:
    """Return True if *module* is an allowed cross-sector target."""
    for allowed in ALLOWED_CROSS_SECTOR:
        if module == allowed or module.startswith(allowed + "."):
            return True
    for neutral in NEUTRAL_PREFIXES:
        if module == neutral or module.startswith(neutral + "."):
            return True
    return False


def _collect_imports(filepath: Path) -> list[tuple[int, str]]:
    """Extract (line_number, module_name) from all imports in a Python file."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    results: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("mt5pipe."):
                    results.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("mt5pipe."):
                results.append((node.lineno, node.module))
    return results


def _module_path_of(filepath: Path) -> str:
    """Convert a file path to a dotted module name."""
    rel = filepath.relative_to(REPO_ROOT)
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _find_python_files(root: Path) -> list[Path]:
    """Recursively find all .py files under *root*."""
    return sorted(root.rglob("*.py"))


# ---------------------------------------------------------------------------
# Violation scanner
# ---------------------------------------------------------------------------


def scan_boundary_violations() -> list[ImportViolation]:
    """
    Walk all Python files in mt5pipe/ and flag cross-sector imports
    that don't go through an allowed boundary module.
    """
    violations: list[ImportViolation] = []

    for py_file in _find_python_files(MT5PIPE):
        source_module = _module_path_of(py_file)
        source_sector = _sector_of(source_module)

        if source_sector is None:
            # File is in a neutral package — no restrictions
            continue

        for lineno, target in _collect_imports(py_file):
            target_sector = _sector_of(target)

            if target_sector is None:
                # Target is neutral — always OK
                continue

            if target_sector == source_sector:
                # Intra-sector — always OK
                continue

            # Cross-sector: must go through allowed boundary
            if not _is_allowed_target(target):
                violations.append(
                    ImportViolation(
                        file=str(py_file.relative_to(REPO_ROOT)),
                        line=lineno,
                        source_sector=source_sector,
                        target_module=target,
                        target_sector=target_sector,
                    )
                )

    return violations


def scan_cross_sector_root_import_violations() -> list[RootImportViolation]:
    """
    Walk all Python files in mt5pipe/ and flag cross-sector imports that
    target sector package roots (e.g., ``mt5pipe.state`` instead of
    ``mt5pipe.state.public``).
    """
    sector_roots = {"mt5pipe.state", "mt5pipe.features", "mt5pipe.labels", "mt5pipe.compiler"}
    violations: list[RootImportViolation] = []

    for py_file in _find_python_files(MT5PIPE):
        source_module = _module_path_of(py_file)
        source_sector = _sector_of(source_module)

        if source_sector is None:
            continue

        for lineno, target in _collect_imports(py_file):
            if target not in sector_roots:
                continue

            target_sector = _sector_of(target)
            if target_sector is None or target_sector == source_sector:
                continue

            violations.append(
                RootImportViolation(
                    file=str(py_file.relative_to(REPO_ROOT)),
                    line=lineno,
                    source_sector=source_sector,
                    target_module=target,
                    target_sector=target_sector,
                )
            )

    return violations


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBoundaryRules:
    """Verify that the import boundary rules are well-defined."""

    def test_sectors_are_disjoint(self):
        """No package prefix appears in two sectors."""
        all_prefixes: list[str] = []
        for prefixes in SECTOR_PACKAGES.values():
            all_prefixes.extend(prefixes)
        assert len(all_prefixes) == len(set(all_prefixes))

    def test_public_modules_importable(self):
        """Each public boundary module can be imported."""
        for mod_name in [
            "mt5pipe.contracts",
            "mt5pipe.state.public",
            "mt5pipe.features.public",
            "mt5pipe.labels.public",
            "mt5pipe.compiler.public",
        ]:
            mod = importlib.import_module(mod_name)
            assert hasattr(mod, "__all__") or mod_name == "mt5pipe.contracts"

    def test_contracts_package_importable(self):
        """The shared contracts package exposes its core types."""
        from mt5pipe.contracts import ArtifactRef, DatasetId, TrustVerdict, LineageNode

        assert ArtifactRef is not None
        assert DatasetId is not None
        assert TrustVerdict is not None
        assert LineageNode is not None


class TestBoundaryEnforcement:
    """
    Scan for cross-sector import violations.

    TODO: Currently the existing codebase has many direct cross-sector
    imports (e.g., compiler.service imports state.service directly).
    These will be migrated to use public boundary modules over time.

    Target state: zero violations.
    Current state: violations are collected and reported but the test
    is marked xfail until migration is complete.
    """

    @pytest.mark.xfail(
        reason="Existing code has cross-sector imports that predate boundary setup. "
        "Agents will migrate these incrementally.",
        strict=False,
    )
    def test_no_cross_sector_violations(self):
        violations = scan_boundary_violations()
        if violations:
            msg_lines = ["Cross-sector import violations found:\n"]
            for v in violations:
                msg_lines.append(
                    f"  {v.file}:{v.line}  "
                    f"[{v.source_sector}] imports {v.target_module} "
                    f"(owned by [{v.target_sector}])"
                )
            pytest.fail("\n".join(msg_lines))

    def test_scan_runs_without_error(self):
        """The scanner itself doesn't crash, regardless of violations."""
        violations = scan_boundary_violations()
        assert isinstance(violations, list)
        # Report count for visibility
        print(f"\n[boundary] Found {len(violations)} cross-sector import violation(s)")

    def test_no_cross_sector_root_imports(self):
        """Cross-sector imports must not target sector package roots."""
        violations = scan_cross_sector_root_import_violations()
        if violations:
            msg_lines = ["Cross-sector root import violations found:\n"]
            for v in violations:
                msg_lines.append(
                    f"  {v.file}:{v.line}  "
                    f"[{v.source_sector}] imports {v.target_module} "
                    f"(owned by [{v.target_sector}]); use <sector>.public instead"
                )
            pytest.fail("\n".join(msg_lines))
