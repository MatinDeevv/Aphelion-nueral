"""Configuration loader — YAML file + environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import SecretStr

from mt5pipe.config.models import BrokerConfig, PipelineConfig

_CONFIG_ENV_VAR = "MT5PIPE_CONFIG"
_DEFAULT_PATH = Path("config/pipeline.yaml")


def _substitute_env(data: Any) -> Any:
    """Recursively substitute ${VAR} patterns with environment variables."""
    if isinstance(data, str) and data.startswith("${") and data.endswith("}"):
        var_name = data[2:-1]
        val = os.environ.get(var_name)
        # Keep config usable without hard-failing when broker password env vars
        # are not provided. MT5 can still connect through an already logged-in
        # terminal session.
        return "" if val is None else val
    if isinstance(data, dict):
        return {k: _substitute_env(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_substitute_env(v) for v in data]
    return data


def load_config(path: Path | str | None = None) -> PipelineConfig:
    """Load pipeline config from YAML, with env substitution for secrets."""
    if path is None:
        env_path = os.environ.get(_CONFIG_ENV_VAR)
        path = Path(env_path) if env_path else _DEFAULT_PATH

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    raw = _substitute_env(raw)
    return PipelineConfig.model_validate(raw)
