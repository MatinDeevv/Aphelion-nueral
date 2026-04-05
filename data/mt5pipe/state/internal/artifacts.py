"""State-sector artifact helpers with no compiler-sector dependency."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from mt5pipe.state.models import StateArtifactManifest
from mt5pipe.storage.paths import StoragePaths


def state_code_version() -> str:
    """Best-effort code version for state-side artifacts."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        sha = result.stdout.strip()
        if sha:
            return sha
    except Exception:
        pass
    return "workspace-local-no-git"


def compute_content_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact_id(artifact_kind: str, logical_name: str, content_hash: str) -> str:
    return f"{artifact_kind}.{logical_name}.{content_hash[:12]}"


def build_manifest_id(artifact_kind: str, logical_name: str, content_hash: str) -> str:
    return f"manifest.{artifact_kind}.{logical_name}.{content_hash[:12]}"


def write_state_manifest(manifest: StateArtifactManifest, paths: StoragePaths) -> Path:
    path = paths.manifest_file(manifest.artifact_kind, manifest.logical_name, manifest.manifest_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return path


def read_state_manifest(path: Path) -> StateArtifactManifest:
    return StateArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))


def build_id_now(prefix: str = "state") -> str:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{prefix}.{ts}"
