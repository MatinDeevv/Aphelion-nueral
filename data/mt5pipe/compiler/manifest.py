"""Manifest and spec helpers for the dataset compiler."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from mt5pipe.compiler.models import DatasetSpec, ExperimentSpec, LineageManifest
from mt5pipe.config.models import MergeConfig
from mt5pipe.storage.paths import StoragePaths


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(raw)
    return json.loads(raw)


def load_dataset_spec(path: Path) -> DatasetSpec:
    """Load a DatasetSpec from YAML or JSON."""
    return DatasetSpec.model_validate(_load_yaml_or_json(path))


def load_experiment_spec(path: Path) -> ExperimentSpec:
    """Load an ExperimentSpec from YAML or JSON."""
    return ExperimentSpec.model_validate(_load_yaml_or_json(path))


def build_id_now() -> str:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"build.{ts}"


def code_version() -> str:
    """Best-effort code version identifier."""
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


def merge_config_ref(cfg: MergeConfig) -> str:
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"merge.default@{digest}"


def compute_content_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_stage_artifact_id(artifact_kind: str, logical_name: str, created_at: dt.datetime, content_hash: str) -> str:
    _ = created_at
    return f"{artifact_kind}.{logical_name}.{content_hash[:12]}"


def build_artifact_id(dataset_name: str, created_at: dt.datetime, content_hash: str) -> str:
    return build_stage_artifact_id("dataset", dataset_name, created_at, content_hash)


def build_stage_manifest_id(artifact_kind: str, logical_name: str, created_at: dt.datetime, content_hash: str) -> str:
    _ = created_at
    return f"manifest.{artifact_kind}.{logical_name}.{content_hash[:12]}"


def build_manifest_id(dataset_name: str, created_at: dt.datetime, content_hash: str) -> str:
    return build_stage_manifest_id("dataset", dataset_name, created_at, content_hash)


def write_manifest_sidecar(manifest: LineageManifest, paths: StoragePaths) -> Path:
    path = paths.manifest_file(manifest.artifact_kind, manifest.logical_name, manifest.manifest_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def read_manifest_sidecar(path: Path) -> LineageManifest:
    return LineageManifest.model_validate_json(path.read_text(encoding="utf-8"))
