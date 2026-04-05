"""Training and experiment registry CLI commands."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

import typer

from mt5pipe.config.loader import load_config
from mt5pipe.utils.logging import setup_logging


train_app = typer.Typer(help="Run trust-gated experiments and inspect model artifacts")
_MISSING = object()


def _build_runner(config_path: str):
    cfg = load_config(Path(config_path))
    setup_logging(cfg.logging.level, cfg.logging.json_output)

    from mt5pipe.compiler.training import ExperimentRunner

    return ExperimentRunner.from_config_path(config_path)


def _resolve_attr(value: Any, path: str, default: Any = _MISSING) -> Any:
    current = value
    for part in path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
            continue
        if not hasattr(current, part):
            return default
        current = getattr(current, part)
    return current


def _first_present(value: Any, *paths: str, default: Any = None) -> Any:
    for path in paths:
        resolved = _resolve_attr(value, path, _MISSING)
        if resolved is not _MISSING:
            return resolved
    return default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int | float):
        return f"{value:.4f}"
    return str(value)


def _run_experiment_spec(spec_path: str | Path, *, config_path: str) -> Any:
    try:
        training = import_module("mt5pipe.compiler.training")
    except ModuleNotFoundError:
        training = None
    run_fn = getattr(training, "run_experiment_spec", None) if training is not None else None
    if callable(run_fn):
        return run_fn(spec_path)

    runner, catalog = _build_runner(config_path)
    try:
        return runner.run_experiment(Path(spec_path))
    finally:
        catalog.close()


def _inspect_experiment(ref: str, *, config_path: str) -> Any:
    try:
        training = import_module("mt5pipe.compiler.training")
    except ModuleNotFoundError:
        training = None
    inspect_fn = getattr(training, "inspect_experiment", None) if training is not None else None
    if callable(inspect_fn):
        return inspect_fn(ref)

    runner, catalog = _build_runner(config_path)
    try:
        return runner.inspect_experiment(ref)
    finally:
        catalog.close()


def _inspect_model(ref: str, *, config_path: str) -> Any:
    try:
        training = import_module("mt5pipe.compiler.training")
    except ModuleNotFoundError:
        training = None
    inspect_fn = getattr(training, "inspect_model", None) if training is not None else None
    if callable(inspect_fn):
        return inspect_fn(ref)

    runner, catalog = _build_runner(config_path)
    try:
        return runner.inspect_model(ref)
    finally:
        catalog.close()


def _run_result_lines(result: Any) -> list[str]:
    walk_forward = _first_present(result, "walk_forward_summary", default={}) or {}
    holdout = _first_present(result, "holdout_metrics", default={}) or {}
    experiment_aliases = [str(value) for value in _as_list(_first_present(result, "experiment_aliases", default=[]))]
    model_aliases = [str(value) for value in _as_list(_first_present(result, "model_aliases", default=[]))]
    return [
        f"run_id: {_first_present(result, 'run_id', default='-')}",
        f"dataset_ref: {_first_present(result, 'dataset_ref', default='-')}",
        f"dataset_artifact_id: {_first_present(result, 'dataset_artifact_id', default='-')}",
        f"experiment_artifact_id: {_first_present(result, 'experiment_artifact_id', default='-')}",
        f"model_artifact_id: {_first_present(result, 'model_artifact_id', default='-')}",
        f"experiment_manifest_path: {_first_present(result, 'experiment_manifest_path', default='-')}",
        f"model_manifest_path: {_first_present(result, 'model_manifest_path', default='-')}",
        f"run_status: {_first_present(result, 'run_status', default='-')}",
        f"model_status: {_first_present(result, 'model_status', default='-')}",
        f"selected_feature_families: {_json(_first_present(result, 'selected_feature_families', default=[]))}",
        f"selected_feature_count: {len(_as_list(_first_present(result, 'selected_feature_columns', default=[])))}",
        f"active_feature_count: {len(_as_list(_first_present(result, 'active_feature_columns', default=[])))}",
        f"walk_forward_summary: {_json(walk_forward)}",
        f"holdout_metrics: {_json(holdout)}",
        f"experiment_ref: {experiment_aliases[0] if experiment_aliases else '-'}",
        f"model_ref: {model_aliases[0] if model_aliases else '-'}",
    ]


def _inspect_experiment_lines(result: Any) -> list[str]:
    summary = _first_present(result, "summary", default={}) or {}
    aliases = [str(alias.alias_key) for alias in _as_list(_first_present(result, "aliases", default=[]))]
    return [
        f"artifact_id: {_first_present(result, 'artifact.artifact_id', default='-')}",
        f"logical: {_first_present(result, 'artifact.logical_name', default='-')}@{_first_present(result, 'artifact.logical_version', default='-')}",
        f"status: {_first_present(result, 'artifact.status', default='-')}",
        f"manifest_path: {_first_present(result, 'manifest_path', default='-')}",
        f"experiment_spec_ref: {_first_present(result, 'manifest.experiment_spec_ref', default='-')}",
        f"dataset_artifact_id: {_first_present(result, 'dataset_artifact.artifact_id', 'summary.dataset_artifact_id', default='-')}",
        f"model_artifact_id: {_first_present(result, 'model_artifact_id', 'summary.model_artifact_id', default='-')}",
        f"run_id: {_first_present(result, 'training_run.run_id', 'summary.run_id', default='-')}",
        f"training_status: {_first_present(result, 'training_run.status', 'summary.model_status', default='-')}",
        f"target_column: {_first_present(result, 'summary.target_column', default='-')}",
        f"model_family: {_first_present(result, 'summary.model_family', default='-')}",
        f"feature_families: {_json(_first_present(result, 'summary.feature_families', default=[]))}",
        f"selected_feature_count: {len(_as_list(_first_present(result, 'summary.selected_feature_columns', default=[])))}",
        f"active_feature_count: {len(_as_list(_first_present(result, 'summary.active_feature_columns', default=[])))}",
        f"walk_forward_balanced_accuracy_mean: {_format_float(_first_present(summary, 'walk_forward_summary.balanced_accuracy_mean', default=None))}",
        f"holdout_balanced_accuracy: {_format_float(_first_present(summary, 'holdout_metrics.balanced_accuracy', default=None))}",
        f"predictions_path: {_first_present(result, 'predictions_path', default='-')}",
        f"aliases: {_json(sorted(aliases))}",
    ]


def _inspect_model_lines(result: Any) -> list[str]:
    summary = _first_present(result, "summary", default={}) or {}
    payload = _first_present(result, "payload", default={}) or {}
    aliases = [str(alias.alias_key) for alias in _as_list(_first_present(result, "aliases", default=[]))]
    return [
        f"artifact_id: {_first_present(result, 'artifact.artifact_id', default='-')}",
        f"logical: {_first_present(result, 'artifact.logical_name', default='-')}@{_first_present(result, 'artifact.logical_version', default='-')}",
        f"status: {_first_present(result, 'artifact.status', default='-')}",
        f"manifest_path: {_first_present(result, 'manifest_path', default='-')}",
        f"experiment_spec_ref: {_first_present(result, 'manifest.experiment_spec_ref', default='-')}",
        f"experiment_artifact_id: {_first_present(result, 'experiment_artifact.artifact_id', 'summary.experiment_artifact_id', default='-')}",
        f"dataset_artifact_id: {_first_present(result, 'dataset_artifact.artifact_id', 'summary.dataset_artifact_id', default='-')}",
        f"run_id: {_first_present(result, 'training_run.run_id', 'summary.run_id', default='-')}",
        f"training_status: {_first_present(result, 'training_run.status', 'artifact.status', default='-')}",
        f"model_family: {_first_present(result, 'payload.model_family', 'summary.model_family', default='-')}",
        f"target_column: {_first_present(result, 'summary.target_column', default='-')}",
        f"active_feature_count: {len(_as_list(_first_present(result, 'payload.feature_columns', default=[])))}",
        f"active_feature_columns_sample: {_json(_as_list(_first_present(result, 'payload.feature_columns', default=[]))[:8])}",
        f"walk_forward_balanced_accuracy_mean: {_format_float(_first_present(summary, 'walk_forward_summary.balanced_accuracy_mean', default=None))}",
        f"holdout_balanced_accuracy: {_format_float(_first_present(summary, 'holdout_metrics.balanced_accuracy', default=None))}",
        f"class_priors: {_json({key: value.get('prior') for key, value in dict(_first_present(result, 'payload.class_stats', default={})).items()})}",
        f"aliases: {_json(sorted(aliases))}",
    ]


@train_app.command("run-experiment")
def run_experiment_cmd(
    spec_path: str = typer.Option(..., "--spec", help="Path to ExperimentSpec YAML/JSON"),
    config_path: str = typer.Option("config/pipeline.yaml", "--config", help="Legacy adapter config path"),
) -> None:
    """Run a trust-gated experiment and register experiment/model artifacts."""
    try:
        result = _run_experiment_spec(Path(spec_path), config_path=config_path)
        typer.echo("\n".join(_run_result_lines(result)))
    except Exception as exc:
        typer.echo(f"reason: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@train_app.command("inspect-experiment")
def inspect_experiment_cmd(
    experiment_ref: str = typer.Option(..., "--experiment", help="Experiment artifact id, alias, or manifest path"),
    config_path: str = typer.Option("config/pipeline.yaml", "--config", help="Legacy adapter config path"),
) -> None:
    """Inspect a registered experiment artifact and its linked training summary."""
    try:
        result = _inspect_experiment(experiment_ref, config_path=config_path)
        typer.echo("\n".join(_inspect_experiment_lines(result)))
    except Exception as exc:
        typer.echo(f"reason: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@train_app.command("inspect-model")
def inspect_model_cmd(
    model_ref: str = typer.Option(..., "--model", help="Model artifact id, alias, or manifest path"),
    config_path: str = typer.Option("config/pipeline.yaml", "--config", help="Legacy adapter config path"),
) -> None:
    """Inspect a registered model artifact and its evaluation summary."""
    try:
        result = _inspect_model(model_ref, config_path=config_path)
        typer.echo("\n".join(_inspect_model_lines(result)))
    except Exception as exc:
        typer.echo(f"reason: {exc}", err=True)
        raise typer.Exit(code=1) from exc
