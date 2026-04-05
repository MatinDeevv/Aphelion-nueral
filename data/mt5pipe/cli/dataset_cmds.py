"""Dataset builder CLI commands."""

from __future__ import annotations

import datetime as dt
import json
from importlib import import_module
from pathlib import Path
from typing import Any

import typer

from mt5pipe.config.loader import load_config
from mt5pipe.utils.logging import setup_logging

dataset_app = typer.Typer(help="Build model-ready datasets")


_MISSING = object()


def _build_compiler(config_path: str):
    cfg = load_config(Path(config_path))
    setup_logging(cfg.logging.level, cfg.logging.json_output)

    from mt5pipe.catalog.sqlite import CatalogDB
    from mt5pipe.compiler.service import DatasetCompiler
    from mt5pipe.storage.parquet_store import ParquetStore
    from mt5pipe.storage.paths import StoragePaths

    paths = StoragePaths(cfg.storage.root)
    store = ParquetStore(cfg.storage.compression, cfg.storage.parquet_row_group_size)
    catalog = CatalogDB(paths.catalog_db_path())
    compiler = DatasetCompiler(cfg, paths, store, catalog)
    return compiler, catalog


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


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _format_float(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int | float):
        return f"{value:.2f}"
    return str(value)


def _trust_report(result: Any) -> Any:
    return _first_present(result, "trust_report", default=None)


def _manifest(result: Any) -> Any:
    return _first_present(result, "manifest", default=result)


def _logical_name(result: Any) -> str:
    return str(_first_present(result, "manifest.logical_name", "logical_name", "spec.dataset_name", default="-"))


def _logical_version(result: Any) -> str:
    return str(_first_present(result, "manifest.logical_version", "logical_version", "spec.version", default="-"))


def _artifact_id(result: Any) -> str:
    return str(_first_present(result, "artifact_id", "manifest.artifact_id", default="-"))


def _artifact_status(result: Any) -> str:
    return str(_first_present(result, "manifest.status", "status", default="-"))


def _manifest_path(result: Any) -> str:
    return str(_first_present(result, "manifest_path", default="-"))


def _dataset_ref(result: Any) -> str:
    logical_name = _logical_name(result)
    logical_version = _logical_version(result)
    if logical_name == "-" or logical_version == "-":
        return "-"
    return f"dataset://{logical_name}@{logical_version}"


def _time_range(result: Any) -> tuple[str, str]:
    start = _first_present(result, "manifest.metadata.time_range_start", "metadata.time_range_start", default="-")
    end = _first_present(result, "manifest.metadata.time_range_end", "metadata.time_range_end", default="-")
    return str(start), str(end)


def _schema_columns(result: Any) -> list[str]:
    columns = _first_present(result, "manifest.metadata.schema_columns", "schema_columns", default=[])
    return [str(column) for column in _as_list(columns)]


def _split_rows(result: Any) -> dict[str, Any]:
    rows = _first_present(result, "manifest.metadata.split_row_counts", "split_row_counts", default={})
    return {str(key): value for key, value in _as_dict(rows).items()}


def _feature_refs(result: Any) -> list[str]:
    refs = _first_present(result, "manifest.feature_spec_refs", "feature_spec_refs", default=[])
    return [str(ref) for ref in _as_list(refs)]


def _feature_families(result: Any) -> list[str]:
    families: set[str] = set()
    for ref in _feature_refs(result):
        base = ref.split("@", 1)[0]
        if "." in base:
            family = base.split(".", 1)[0]
        elif "/" in base:
            family = base.split("/", 1)[0]
        else:
            family = base
        if family:
            families.add(family)
    return sorted(families)


def _label_pack_ref(result: Any) -> str:
    return str(_first_present(result, "manifest.label_pack_ref", "label_pack_ref", default="-"))


def _dataset_spec_ref(result: Any) -> str:
    return str(_first_present(result, "manifest.dataset_spec_ref", "dataset_spec_ref", default="-"))


def _feature_selectors(result: Any) -> list[str]:
    selectors = _first_present(result, "manifest.metadata.requested_feature_selectors", "requested_feature_selectors", default=[])
    return [str(selector) for selector in _as_list(selectors)]


def _feature_artifact_refs(result: Any) -> list[str]:
    refs = _first_present(result, "manifest.metadata.feature_artifact_refs", "feature_artifact_refs", default=[])
    return [str(ref) for ref in _as_list(refs)]


def _source_modes(result: Any) -> dict[str, Any]:
    return {str(key): value for key, value in _as_dict(_first_present(result, "manifest.metadata.source_modes", "source_modes", default={})).items()}


def _build_row_stats(result: Any) -> dict[str, Any]:
    return {str(key): value for key, value in _as_dict(_first_present(result, "manifest.metadata.build_row_stats", "build_row_stats", default={})).items()}


def _lineage_refs(result: Any) -> dict[str, list[str]]:
    return {
        "state_artifacts": [str(ref) for ref in _as_list(_first_present(result, "manifest.state_artifact_refs", "state_artifact_refs", default=[]))],
        "parent_artifacts": [str(ref) for ref in _as_list(_first_present(result, "manifest.parent_artifact_refs", "parent_artifact_refs", default=[]))],
    }


def _trust_status(result: Any) -> str:
    return str(_first_present(result, "trust_report.status", "trust_status", default="-"))


def _trust_score(result: Any) -> Any:
    return _first_present(result, "trust_report.trust_score_total", "trust_score_total", default=None)


def _trust_breakdown(result: Any) -> dict[str, Any]:
    trust = _trust_report(result)
    if trust is None:
        return {}
    return {
        "coverage": _first_present(trust, "coverage_score", default=None),
        "leakage": _first_present(trust, "leakage_score", default=None),
        "feature_quality": _first_present(trust, "feature_quality_score", default=None),
        "label_quality": _first_present(trust, "label_quality_score", default=None),
        "source_quality": _first_present(trust, "source_quality_score", default=None),
        "lineage": _first_present(trust, "lineage_score", default=None),
    }


def _trust_decision_summary(result: Any) -> str:
    return str(_first_present(result, "trust_report.decision_summary", "trust_decision_summary", default="-"))


def _trust_warning_reasons(result: Any) -> list[str]:
    reasons = _first_present(result, "trust_report.warning_reasons", "trust_warning_reasons", default=[])
    return [str(reason) for reason in _as_list(reasons)]


def _trust_rejection_reasons(result: Any) -> list[str]:
    reasons = _first_present(result, "trust_report.rejection_reasons", "trust_rejection_reasons", default=[])
    return [str(reason) for reason in _as_list(reasons)]


def _trust_check_counts(result: Any) -> dict[str, Any]:
    counts = _first_present(result, "trust_report.check_status_counts", "trust_check_status_counts", default={})
    return {str(key): value for key, value in _as_dict(counts).items()}


def _quality_caveats(result: Any) -> dict[str, Any]:
    summary = _as_dict(_first_present(result, "trust_report.metrics.quality_caveat_summary", default={}))
    return {
        "accepted_caveats": [str(value) for value in _as_list(summary.get("accepted_caveats"))],
        "green_blockers": [str(value) for value in _as_list(summary.get("green_blockers"))],
        "publication_blockers": [str(value) for value in _as_list(summary.get("publication_blockers"))],
    }


def _quality_family_summary(result: Any) -> dict[str, Any]:
    family_notes = _as_dict(
        _first_present(
            result,
            "trust_report.metrics.quality_caveat_summary.family_warning_summary",
            "trust_report.metrics.dataset_quality.family_warning_summary",
            default={},
        )
    )
    return {
        str(family): str(_as_list(notes)[0])
        for family, notes in sorted(family_notes.items())
        if _as_list(notes)
    }


def _source_quality_metrics(result: Any) -> dict[str, Any]:
    metrics = _as_dict(_first_present(result, "trust_report.metrics.source_quality", default={}))
    keys = [
        "required_raw_brokers",
        "required_raw_missing_dates",
        "required_raw_asymmetric_dates",
        "merge_observability_source",
        "merge_qa_days",
        "merge_diagnostics_days",
        "effective_observability_days",
        "state_quality_mean",
        "state_filled_ratio",
        "dual_source_ratio_mean",
        "effective_dual_source_ratio_mean",
        "diagnostic_dual_source_ratio_mean",
        "merge_conflict_mean",
        "diagnostic_conflict_mean",
        "effective_conflict_mean",
        "synchronized_raw_days",
        "synchronized_raw_coverage_ratio",
        "dual_source_days",
        "bucket_both_total",
        "canonical_dual_rows_total",
    ]
    return {
        key: metrics[key]
        for key in keys
        if key in metrics
    }


def _dataset_quality_alerts(result: Any) -> dict[str, Any]:
    metrics = _as_dict(_first_present(result, "trust_report.metrics.dataset_quality", default={}))
    null_columns = {
        str(key): value
        for key, value in sorted(_as_dict(metrics.get("null_columns", {})).items())
    }
    expected_sparse_null_columns = {
        str(key): _as_dict(value).get("null_count")
        for key, value in sorted(_as_dict(metrics.get("expected_sparse_null_columns", {})).items())
    }
    unexpected_null_columns = {
        str(key): _as_dict(value).get("null_count")
        for key, value in sorted(_as_dict(metrics.get("unexpected_null_columns", {})).items())
    }
    constant_columns = sorted(str(value) for value in _as_list(metrics.get("constant_columns", [])))
    slice_trivial_constant_columns = sorted(
        str(key) for key in _as_dict(metrics.get("slice_trivial_constant_columns", {}))
    )
    blocking_constant_columns = sorted(
        str(key) for key in _as_dict(metrics.get("blocking_constant_columns", {}))
    )
    if (
        not metrics
        and not null_columns
        and not constant_columns
        and not expected_sparse_null_columns
        and not unexpected_null_columns
        and not slice_trivial_constant_columns
        and not blocking_constant_columns
    ):
        return {}
    return {
        "quality_score": metrics.get("quality_score"),
        "total_nulls": metrics.get("total_nulls"),
        "null_columns_sample": dict(list(null_columns.items())[:5]),
        "expected_sparse_null_columns_sample": dict(list(expected_sparse_null_columns.items())[:5]),
        "unexpected_null_columns_sample": dict(list(unexpected_null_columns.items())[:5]),
        "constant_columns_sample": constant_columns[:5],
        "slice_trivial_constant_columns_sample": slice_trivial_constant_columns[:5],
        "blocking_constant_columns_sample": blocking_constant_columns[:5],
    }


def _compile_failure_reason(result: Any) -> str | None:
    rejection_reasons = _trust_rejection_reasons(result)
    if rejection_reasons:
        return "; ".join(rejection_reasons)
    decision_summary = _trust_decision_summary(result)
    if _artifact_status(result) in {"rejected", "failed"} and decision_summary not in {"", "-"}:
        return decision_summary
    trust = _trust_report(result)
    hard_failures = [str(item) for item in _as_list(_first_present(trust, "hard_failures", default=[]))]
    if hard_failures:
        return "; ".join(hard_failures)
    warning_reasons = _trust_warning_reasons(result)
    if _artifact_status(result) in {"rejected", "failed"}:
        return "; ".join(warning_reasons) if warning_reasons else f"artifact status is {_artifact_status(result)}"
    accepted = _first_present(trust, "accepted_for_publication", default=True)
    if accepted is False:
        return "; ".join(warning_reasons) if warning_reasons else "artifact was not accepted for publication"
    return None


def _compile_result_lines(result: Any) -> list[str]:
    published_ref = _first_present(result, "published_logical_address", "published_ref", default=None)
    if not published_ref and _artifact_status(result) == "published":
        published_ref = _dataset_ref(result)
    return [
        f"artifact_id: {_artifact_id(result)}",
        f"logical: {_logical_name(result)}@{_logical_version(result)}",
        f"status: {_artifact_status(result)}",
        f"manifest_path: {_manifest_path(result)}",
        f"split_rows: {_json(_split_rows(result))}",
        f"trust_status: {_trust_status(result)}",
        f"trust_score_total: {_format_float(_trust_score(result))}",
        f"trust_decision: {_trust_decision_summary(result)}",
        f"trust_check_counts: {_json(_trust_check_counts(result))}",
        f"trust_warning_reasons: {_json(_trust_warning_reasons(result))}",
        f"trust_rejection_reasons: {_json(_trust_rejection_reasons(result))}",
        f"quality_caveats: {_json(_quality_caveats(result))}",
        f"source_quality_metrics: {_json(_source_quality_metrics(result))}",
        f"published_ref: {published_ref or '-'}",
    ]


def _inspect_result_lines(result: Any) -> list[str]:
    schema_columns = _schema_columns(result)
    time_start, time_end = _time_range(result)
    return [
        f"artifact_id: {_artifact_id(result)}",
        f"logical: {_logical_name(result)}@{_logical_version(result)}",
        f"status: {_artifact_status(result)}",
        f"manifest_path: {_manifest_path(result)}",
        f"dataset_spec_ref: {_dataset_spec_ref(result)}",
        f"time_range: {time_start} -> {time_end}",
        f"split_rows: {_json(_split_rows(result))}",
        f"schema_columns_count: {len(schema_columns)}",
        f"schema_columns_sample: {_json(schema_columns[:8])}",
        f"feature_selectors: {_json(_feature_selectors(result))}",
        f"feature_families: {_json(_feature_families(result))}",
        f"feature_artifact_refs: {_json(_feature_artifact_refs(result))}",
        f"source_modes: {_json(_source_modes(result))}",
        f"build_row_stats: {_json(_build_row_stats(result))}",
        f"label_pack: {_label_pack_ref(result)}",
        f"trust_status: {_trust_status(result)}",
        f"trust_score_total: {_format_float(_trust_score(result))}",
        f"trust_decision: {_trust_decision_summary(result)}",
        f"trust_check_counts: {_json(_trust_check_counts(result))}",
        f"trust_warning_reasons: {_json(_trust_warning_reasons(result))}",
        f"trust_rejection_reasons: {_json(_trust_rejection_reasons(result))}",
        f"trust_breakdown: {_json(_trust_breakdown(result))}",
        f"quality_caveats: {_json(_quality_caveats(result))}",
        f"quality_family_summary: {_json(_quality_family_summary(result))}",
        f"source_quality_metrics: {_json(_source_quality_metrics(result))}",
        f"dataset_quality_alerts: {_json(_dataset_quality_alerts(result))}",
        f"lineage_refs: {_json(_lineage_refs(result))}",
    ]


def _split_row_deltas(left_rows: dict[str, Any], right_rows: dict[str, Any]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for split_name in sorted(set(left_rows) | set(right_rows)):
        left_value = left_rows.get(split_name, 0)
        right_value = right_rows.get(split_name, 0)
        if isinstance(left_value, int | float) and isinstance(right_value, int | float):
            deltas[split_name] = right_value - left_value
        else:
            deltas[split_name] = {"left": left_value, "right": right_value}
    return deltas


def _diff_result_lines(result: Any) -> list[str]:
    left = _first_present(result, "left", default=None)
    right = _first_present(result, "right", default=None)
    if left is None:
        left = _first_present(result, "left_artifact", "left_inspection", default={})
    if right is None:
        right = _first_present(result, "right_artifact", "right_inspection", default={})

    left_rows = _split_rows(left)
    right_rows = _split_rows(right)
    left_schema = set(_schema_columns(left))
    right_schema = set(_schema_columns(right))
    left_features = set(_feature_refs(left))
    right_features = set(_feature_refs(right))

    return [
        f"left_artifact_id: {_artifact_id(left)}",
        f"right_artifact_id: {_artifact_id(right)}",
        f"left_spec_ref: {_dataset_spec_ref(left)}",
        f"right_spec_ref: {_dataset_spec_ref(right)}",
        f"feature_selectors_left: {_json(_feature_selectors(left))}",
        f"feature_selectors_right: {_json(_feature_selectors(right))}",
        f"feature_families_left: {_json(_feature_families(left))}",
        f"feature_families_right: {_json(_feature_families(right))}",
        f"feature_refs_added: {_json(sorted(right_features - left_features))}",
        f"feature_refs_removed: {_json(sorted(left_features - right_features))}",
        f"feature_artifact_refs_left: {_json(_feature_artifact_refs(left))}",
        f"feature_artifact_refs_right: {_json(_feature_artifact_refs(right))}",
        f"label_pack_left: {_label_pack_ref(left)}",
        f"label_pack_right: {_label_pack_ref(right)}",
        f"split_rows_left: {_json(left_rows)}",
        f"split_rows_right: {_json(right_rows)}",
        f"split_row_deltas: {_json(_split_row_deltas(left_rows, right_rows))}",
        f"source_modes_left: {_json(_source_modes(left))}",
        f"source_modes_right: {_json(_source_modes(right))}",
        f"build_row_stats_left: {_json(_build_row_stats(left))}",
        f"build_row_stats_right: {_json(_build_row_stats(right))}",
        f"schema_columns_added: {_json(sorted(right_schema - left_schema))}",
        f"schema_columns_removed: {_json(sorted(left_schema - right_schema))}",
        f"trust_status_left: {_trust_status(left)}",
        f"trust_status_right: {_trust_status(right)}",
        f"trust_score_left: {_format_float(_trust_score(left))}",
        f"trust_score_right: {_format_float(_trust_score(right))}",
        f"trust_decision_left: {_trust_decision_summary(left)}",
        f"trust_decision_right: {_trust_decision_summary(right)}",
        f"trust_check_counts_left: {_json(_trust_check_counts(left))}",
        f"trust_check_counts_right: {_json(_trust_check_counts(right))}",
        f"trust_breakdown_left: {_json(_trust_breakdown(left))}",
        f"trust_breakdown_right: {_json(_trust_breakdown(right))}",
        f"quality_caveats_left: {_json(_quality_caveats(left))}",
        f"quality_caveats_right: {_json(_quality_caveats(right))}",
        f"quality_family_summary_left: {_json(_quality_family_summary(left))}",
        f"quality_family_summary_right: {_json(_quality_family_summary(right))}",
        f"source_quality_metrics_left: {_json(_source_quality_metrics(left))}",
        f"source_quality_metrics_right: {_json(_source_quality_metrics(right))}",
        f"trust_warning_reasons_added: {_json(sorted(set(_trust_warning_reasons(right)) - set(_trust_warning_reasons(left))))}",
        f"trust_warning_reasons_removed: {_json(sorted(set(_trust_warning_reasons(left)) - set(_trust_warning_reasons(right))))}",
        f"trust_rejection_reasons_added: {_json(sorted(set(_trust_rejection_reasons(right)) - set(_trust_rejection_reasons(left))))}",
        f"trust_rejection_reasons_removed: {_json(sorted(set(_trust_rejection_reasons(left)) - set(_trust_rejection_reasons(right))))}",
        f"lineage_refs_left: {_json(_lineage_refs(left))}",
        f"lineage_refs_right: {_json(_lineage_refs(right))}",
    ]


def _compile_dataset_spec(spec_path: str | Path, *, publish: bool, config_path: str) -> Any:
    try:
        service = import_module("mt5pipe.compiler.service")
    except ModuleNotFoundError:
        service = None
    compile_fn = getattr(service, "compile_dataset_spec", None) if service is not None else None
    if callable(compile_fn):
        return compile_fn(spec_path, publish=publish)

    compiler, catalog = _build_compiler(config_path)
    try:
        return compiler.compile_dataset(Path(spec_path))
    finally:
        catalog.close()


def _inspect_artifact(ref: str, *, config_path: str) -> Any:
    try:
        service = import_module("mt5pipe.compiler.service")
    except ModuleNotFoundError:
        service = None
    inspect_fn = getattr(service, "inspect_artifact", None) if service is not None else None
    if callable(inspect_fn):
        return inspect_fn(ref)

    compiler, catalog = _build_compiler(config_path)
    try:
        return compiler.inspect_dataset(ref)
    finally:
        catalog.close()


def _diff_artifacts(left_ref: str, right_ref: str, *, config_path: str) -> Any:
    try:
        service = import_module("mt5pipe.compiler.service")
    except ModuleNotFoundError:
        service = None
    diff_fn = getattr(service, "diff_artifacts", None) if service is not None else None
    if callable(diff_fn):
        return diff_fn(left_ref, right_ref)

    compiler, catalog = _build_compiler(config_path)
    try:
        return compiler.diff_datasets(left_ref, right_ref)
    finally:
        catalog.close()


@dataset_app.command("build")
def build_dataset_cmd(
    symbol: str = typer.Option("XAUUSD"),
    date_from: str = typer.Option(..., "--from"),
    date_to: str = typer.Option(..., "--to"),
    name: str = typer.Option("default", help="Dataset name"),
    base_timeframe: str = typer.Option("", help="Override base timeframe from config"),
    config_path: str = typer.Option("config/pipeline.yaml", "--config"),
) -> None:
    """Build a model-ready dataset with features and labels."""
    cfg = load_config(Path(config_path))
    setup_logging(cfg.logging.level, cfg.logging.json_output)

    from mt5pipe.features.dataset import build_dataset
    from mt5pipe.storage.parquet_store import ParquetStore
    from mt5pipe.storage.paths import StoragePaths

    paths = StoragePaths(cfg.storage.root)
    store = ParquetStore(cfg.storage.compression, cfg.storage.parquet_row_group_size)

    start = dt.date.fromisoformat(date_from)
    end = dt.date.fromisoformat(date_to)

    dataset_cfg = cfg.dataset
    if base_timeframe:
        dataset_cfg = dataset_cfg.model_copy(update={"base_timeframe": base_timeframe})

    df = build_dataset(symbol, start, end, paths, store, dataset_cfg, name)
    if df.is_empty():
        typer.echo("No data available for dataset.")
    else:
        from mt5pipe.quality.report import dataset_quality_report, format_quality_report
        qr = dataset_quality_report(df)
        typer.echo(format_quality_report(qr))
        typer.echo(f"\nDataset '{name}' built: {len(df):,} rows, {len(df.columns)} columns")
        typer.echo(f"Columns: {', '.join(df.columns[:20])}{'...' if len(df.columns) > 20 else ''}")


@dataset_app.command("compile-dataset")
def compile_dataset_cmd(
    spec_path: str = typer.Option(..., "--spec", help="Path to DatasetSpec YAML/JSON"),
    publish: bool = typer.Option(True, "--publish/--no-publish", help="Publish aliases when the trust gate accepts the artifact"),
    config_path: str = typer.Option("config/pipeline.yaml", "--config", help="Legacy adapter config path"),
) -> None:
    """Compile a versioned dataset artifact from a spec-driven, artifact-aware DatasetSpec."""
    try:
        result = _compile_dataset_spec(Path(spec_path), publish=publish, config_path=config_path)
        typer.echo("\n".join(_compile_result_lines(result)))
        failure_reason = _compile_failure_reason(result)
        if failure_reason:
            typer.echo(f"reason: {failure_reason}", err=True)
            raise typer.Exit(code=5)
    except typer.Exit:
        raise
    except Exception as exc:
        typer.echo(f"reason: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@dataset_app.command("inspect-dataset")
def inspect_dataset_cmd(
    artifact_ref: str = typer.Option(..., "--artifact", help="Artifact id, logical ref, or manifest path"),
    config_path: str = typer.Option("config/pipeline.yaml", "--config", help="Legacy adapter config path"),
) -> None:
    """Inspect a compiled dataset artifact, its lineage sources, and its trust report."""
    try:
        result = _inspect_artifact(artifact_ref, config_path=config_path)
        typer.echo("\n".join(_inspect_result_lines(result)))
    except Exception as exc:
        typer.echo(f"reason: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@dataset_app.command("diff-dataset")
def diff_dataset_cmd(
    left_ref: str = typer.Option(..., "--left", help="Left artifact id, logical ref, or manifest path"),
    right_ref: str = typer.Option(..., "--right", help="Right artifact id, logical ref, or manifest path"),
    config_path: str = typer.Option("config/pipeline.yaml", "--config", help="Legacy adapter config path"),
) -> None:
    """Compare two compiled dataset artifacts with deterministic lineage-aware summaries."""
    try:
        result = _diff_artifacts(left_ref, right_ref, config_path=config_path)
        typer.echo("\n".join(_diff_result_lines(result)))
    except Exception as exc:
        typer.echo(f"reason: {exc}", err=True)
        raise typer.Exit(code=1) from exc
