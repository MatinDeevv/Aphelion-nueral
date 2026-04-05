"""Trust-gated experiment training, evaluation, and model registry services."""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from mt5pipe.catalog.models import (
    AliasRecord,
    ArtifactInputRecord,
    ArtifactRecord,
    ArtifactStatusEventRecord,
    TrainingRunRecord,
)
from mt5pipe.catalog.sqlite import CatalogDB
from mt5pipe.compiler.manifest import (
    build_stage_artifact_id,
    build_stage_manifest_id,
    code_version,
    compute_content_hash,
    load_experiment_spec,
    read_manifest_sidecar,
    write_manifest_sidecar,
)
from mt5pipe.compiler.models import ExperimentSpec, LineageManifest
from mt5pipe.compiler.service import (
    DatasetCompiler,
    _resolve_pipeline_config_path_from_ref,
    _resolve_pipeline_config_path_from_spec,
)
from mt5pipe.config.loader import load_config
from mt5pipe.features.public import FeatureSpec
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths


TRAINING_ALLOWED_DATASET_STATUSES = {"accepted", "published"}


@dataclass
class ExperimentRunResult:
    spec: ExperimentSpec
    run_id: str
    dataset_ref: str
    dataset_artifact_id: str
    experiment_artifact_id: str
    model_artifact_id: str
    experiment_manifest: LineageManifest
    experiment_manifest_path: Path
    model_manifest: LineageManifest
    model_manifest_path: Path
    run_status: str
    model_status: str
    walk_forward_summary: dict[str, Any]
    holdout_metrics: dict[str, Any]
    selected_feature_families: list[str] = field(default_factory=list)
    selected_feature_columns: list[str] = field(default_factory=list)
    active_feature_columns: list[str] = field(default_factory=list)
    experiment_aliases: list[str] = field(default_factory=list)
    model_aliases: list[str] = field(default_factory=list)


@dataclass
class ExperimentInspection:
    ref: str
    artifact: ArtifactRecord
    manifest: LineageManifest
    manifest_path: Path
    experiment_spec: ExperimentSpec | None
    training_run: TrainingRunRecord | None
    dataset_artifact: ArtifactRecord | None
    dataset_manifest: LineageManifest | None
    aliases: list[AliasRecord]
    artifact_inputs: list[ArtifactInputRecord]
    status_history: list[ArtifactStatusEventRecord]
    summary: dict[str, Any]
    predictions_path: Path | None
    model_artifact_id: str | None


@dataclass
class ModelInspection:
    ref: str
    artifact: ArtifactRecord
    manifest: LineageManifest
    manifest_path: Path
    experiment_spec: ExperimentSpec | None
    training_run: TrainingRunRecord | None
    dataset_artifact: ArtifactRecord | None
    experiment_artifact: ArtifactRecord | None
    aliases: list[AliasRecord]
    artifact_inputs: list[ArtifactInputRecord]
    status_history: list[ArtifactStatusEventRecord]
    payload: dict[str, Any]
    summary: dict[str, Any]


@dataclass
class _PreparedMatrix:
    feature_columns: list[str]
    rows: list[list[float]]
    targets: list[int]
    target_values: list[float]
    timestamps: list[dt.datetime]
    symbols: list[str]
    clocks: list[str]


class ExperimentRunner:
    """Run disciplined training/evaluation jobs on trusted dataset artifacts."""

    def __init__(
        self,
        cfg: Any,
        paths: StoragePaths,
        store: ParquetStore,
        catalog: CatalogDB,
    ) -> None:
        self._cfg = cfg
        self._paths = paths
        self._store = store
        self._catalog = catalog

    @classmethod
    def from_config_path(cls, config_path: str | Path | None = None) -> tuple["ExperimentRunner", CatalogDB]:
        if config_path is not None:
            resolved_config_path = Path(config_path).resolve()
        else:
            resolved_config_path = _resolve_pipeline_config_path_from_ref("")
        cfg = load_config(resolved_config_path)
        paths = StoragePaths(cfg.storage.root)
        store = ParquetStore(cfg.storage.compression, cfg.storage.parquet_row_group_size)
        catalog = CatalogDB(paths.catalog_db_path())
        return cls(cfg, paths, store, catalog), catalog

    def run_experiment(self, spec_path: Path) -> ExperimentRunResult:
        spec = load_experiment_spec(spec_path)
        self._catalog.register_experiment_spec(spec)

        dataset_artifact = self._catalog.resolve_artifact(spec.dataset_ref)
        if dataset_artifact is None:
            raise KeyError(f"Dataset artifact '{spec.dataset_ref}' was not found in the compiler catalog")

        run_id = _training_run_id_now()
        self._catalog.start_training_run(
            spec.key,
            dataset_artifact.artifact_id,
            code_version(),
            run_id,
        )

        try:
            compiler = self._build_compiler()
            dataset = compiler.inspect_dataset(spec.dataset_ref)
            self._assert_trainable_dataset(dataset.artifact, dataset.trust_report)

            split_frames = self._load_dataset_split_frames(dataset.manifest)
            self._assert_required_splits(split_frames)

            selected_feature_specs = _select_feature_specs(dataset.feature_specs, spec.feature_families)
            selected_feature_columns = _select_feature_columns(
                schema_columns=dataset.schema_columns,
                feature_specs=selected_feature_specs,
                exclude_columns=spec.exclude_feature_columns,
            )
            if not selected_feature_columns:
                raise ValueError("ExperimentSpec resolved zero usable feature columns from the trusted dataset artifact")
            if spec.target_column not in dataset.schema_columns:
                raise ValueError(
                    f"Experiment target_column '{spec.target_column}' is not present in dataset schema"
                )

            source_dataset_spec = dataset.dataset_spec
            embargo_rows = spec.embargo_rows if spec.embargo_rows is not None else (
                source_dataset_spec.embargo_rows if source_dataset_spec is not None else 0
            )

            train_eval_df = _combine_split_frames(split_frames.get("train"), split_frames.get("val"))
            holdout_df = _sorted_split_frame(split_frames["test"])

            walk_forward_summary, walk_forward_predictions = self._evaluate_walk_forward(
                dataset_df=train_eval_df,
                spec=spec,
                feature_columns=selected_feature_columns,
                embargo_rows=embargo_rows,
            )
            final_model_payload, final_training_summary = self._fit_final_model(
                dataset_df=train_eval_df,
                spec=spec,
                feature_columns=selected_feature_columns,
            )
            holdout_metrics, holdout_predictions = _score_model(
                final_model_payload,
                holdout_df,
                target_column=spec.target_column,
                positive_target_threshold=spec.positive_target_threshold,
                decision_threshold=spec.decision_threshold,
                split_name="holdout",
                fold_index=0,
            )

            model_status = (
                "accepted"
                if walk_forward_summary["balanced_accuracy_mean"] >= spec.min_walk_forward_balanced_accuracy
                and float(holdout_metrics["balanced_accuracy"]) >= spec.min_test_balanced_accuracy
                else "trial"
            )
            run_status = model_status

            predictions_df = _combine_prediction_frames(
                walk_forward_predictions,
                holdout_predictions,
                run_id=run_id,
                dataset_artifact_id=dataset.artifact.artifact_id,
                experiment_key=spec.key,
                model_status=model_status,
            )
            created_at = dt.datetime.now(dt.timezone.utc)
            experiment_summary = {
                "schema_version": "1.0.0",
                "run_id": run_id,
                "experiment_spec_key": spec.key,
                "dataset_ref": spec.dataset_ref,
                "dataset_artifact_id": dataset.artifact.artifact_id,
                "dataset_logical": f"{dataset.artifact.logical_name}@{dataset.artifact.logical_version}",
                "dataset_trust_status": dataset.trust_report.status if dataset.trust_report else "missing",
                "dataset_trust_score_total": dataset.trust_report.trust_score_total if dataset.trust_report else None,
                "target_column": spec.target_column,
                "feature_families": sorted({feature.family for feature in selected_feature_specs}),
                "feature_spec_refs": [feature.key for feature in selected_feature_specs],
                "selected_feature_columns": selected_feature_columns,
                "active_feature_columns": list(final_model_payload["feature_columns"]),
                "excluded_feature_columns": list(spec.exclude_feature_columns),
                "evaluation_policy": spec.evaluation_policy,
                "n_walk_forward_folds": spec.n_walk_forward_folds,
                "embargo_rows": embargo_rows,
                "walk_forward_summary": walk_forward_summary,
                "holdout_metrics": holdout_metrics,
                "final_training_summary": final_training_summary,
                "model_family": spec.model_family,
                "model_status": model_status,
                "started_at": created_at.isoformat(),
            }

            experiment_artifact_id, experiment_manifest, experiment_manifest_path = self._persist_experiment_artifact(
                spec=spec,
                run_id=run_id,
                created_at=created_at,
                dataset_manifest=dataset.manifest,
                dataset_artifact=dataset.artifact,
                experiment_summary=experiment_summary,
                predictions_df=predictions_df,
                selected_feature_specs=selected_feature_specs,
            )
            experiment_summary["summary_path"] = experiment_manifest.metadata["summary_path"]
            experiment_summary["predictions_path"] = experiment_manifest.metadata["predictions_path"]
            model_artifact_id, model_manifest, model_manifest_path = self._persist_model_artifact(
                spec=spec,
                run_id=run_id,
                created_at=created_at,
                dataset_manifest=dataset.manifest,
                dataset_artifact=dataset.artifact,
                experiment_artifact_id=experiment_artifact_id,
                experiment_summary=experiment_summary,
                model_payload=final_model_payload,
                selected_feature_specs=selected_feature_specs,
                model_status=model_status,
            )

            experiment_summary["model_artifact_id"] = model_artifact_id
            experiment_manifest = experiment_manifest.model_copy(
                update={"metadata": {**experiment_manifest.metadata, "model_artifact_id": model_artifact_id}}
            )
            experiment_manifest_path = self._persist_manifest(
                experiment_manifest,
                detail="experiment artifact linked to model artifact",
            )

            experiment_aliases = [
                f"experiment://{spec.experiment_name}@{spec.version}",
                f"experiment://{spec.experiment_name}:latest",
            ]
            model_aliases = [
                f"model://{spec.model_name}@{spec.version}",
                f"model://{spec.model_name}:latest",
            ]
            for alias in experiment_aliases:
                self._catalog.upsert_alias(alias, experiment_artifact_id, alias_type="experiment")
            for alias in model_aliases:
                self._catalog.upsert_alias(alias, model_artifact_id, alias_type="model")

            training_summary = {
                "walk_forward_balanced_accuracy_mean": walk_forward_summary["balanced_accuracy_mean"],
                "walk_forward_balanced_accuracy_min": walk_forward_summary["balanced_accuracy_min"],
                "holdout_balanced_accuracy": holdout_metrics["balanced_accuracy"],
                "model_status": model_status,
                "dataset_artifact_id": dataset.artifact.artifact_id,
                "experiment_artifact_id": experiment_artifact_id,
                "model_artifact_id": model_artifact_id,
            }
            self._catalog.finish_training_run(
                run_id,
                run_status,
                experiment_artifact_id=experiment_artifact_id,
                model_artifact_id=model_artifact_id,
                summary=training_summary,
            )

            return ExperimentRunResult(
                spec=spec,
                run_id=run_id,
                dataset_ref=spec.dataset_ref,
                dataset_artifact_id=dataset.artifact.artifact_id,
                experiment_artifact_id=experiment_artifact_id,
                model_artifact_id=model_artifact_id,
                experiment_manifest=experiment_manifest,
                experiment_manifest_path=experiment_manifest_path,
                model_manifest=model_manifest,
                model_manifest_path=model_manifest_path,
                run_status=run_status,
                model_status=model_status,
                walk_forward_summary=walk_forward_summary,
                holdout_metrics=holdout_metrics,
                selected_feature_families=sorted({feature.family for feature in selected_feature_specs}),
                selected_feature_columns=selected_feature_columns,
                active_feature_columns=list(final_model_payload["feature_columns"]),
                experiment_aliases=experiment_aliases,
                model_aliases=model_aliases,
            )
        except Exception as exc:
            self._catalog.finish_training_run(run_id, "failed", error_message=str(exc))
            raise

    def inspect_experiment(self, ref: str) -> ExperimentInspection:
        manifest, manifest_path = self._resolve_manifest_with_path(ref)
        if manifest.artifact_kind != "experiment":
            raise ValueError(f"Artifact '{ref}' is not an experiment artifact")

        artifact = self._catalog.get_artifact(manifest.artifact_id)
        if artifact is None:
            raise KeyError(f"Artifact '{manifest.artifact_id}' is missing from the compiler catalog")

        experiment_spec = (
            self._catalog.get_experiment_spec(manifest.experiment_spec_ref)
            if manifest.experiment_spec_ref
            else None
        )
        training_run = self._catalog.get_training_run(manifest.build_id)
        dataset_artifact = self._catalog.get_artifact(str(manifest.metadata.get("dataset_artifact_id", "")))
        dataset_manifest = (
            read_manifest_sidecar(Path(dataset_artifact.manifest_uri))
            if dataset_artifact is not None
            else None
        )
        summary = _load_json_payload(manifest.metadata.get("summary_path"))
        predictions_path = _maybe_existing_path(manifest.metadata.get("predictions_path"))
        return ExperimentInspection(
            ref=ref,
            artifact=artifact,
            manifest=manifest,
            manifest_path=manifest_path,
            experiment_spec=experiment_spec,
            training_run=training_run,
            dataset_artifact=dataset_artifact,
            dataset_manifest=dataset_manifest,
            aliases=self._catalog.list_aliases(manifest.artifact_id),
            artifact_inputs=self._catalog.list_artifact_inputs(manifest.artifact_id),
            status_history=self._catalog.get_artifact_status_history(manifest.artifact_id),
            summary=summary,
            predictions_path=predictions_path,
            model_artifact_id=str(manifest.metadata.get("model_artifact_id", "")) or None,
        )

    def inspect_model(self, ref: str) -> ModelInspection:
        manifest, manifest_path = self._resolve_manifest_with_path(ref)
        if manifest.artifact_kind != "model":
            raise ValueError(f"Artifact '{ref}' is not a model artifact")

        artifact = self._catalog.get_artifact(manifest.artifact_id)
        if artifact is None:
            raise KeyError(f"Artifact '{manifest.artifact_id}' is missing from the compiler catalog")

        experiment_spec = (
            self._catalog.get_experiment_spec(manifest.experiment_spec_ref)
            if manifest.experiment_spec_ref
            else None
        )
        training_run = self._catalog.get_training_run(manifest.build_id)
        dataset_artifact = self._catalog.get_artifact(str(manifest.metadata.get("dataset_artifact_id", "")))
        experiment_artifact = self._catalog.get_artifact(str(manifest.metadata.get("experiment_artifact_id", "")))
        payload = _load_json_payload(manifest.metadata.get("model_payload_path"))
        summary = _load_json_payload(manifest.metadata.get("summary_path"))
        return ModelInspection(
            ref=ref,
            artifact=artifact,
            manifest=manifest,
            manifest_path=manifest_path,
            experiment_spec=experiment_spec,
            training_run=training_run,
            dataset_artifact=dataset_artifact,
            experiment_artifact=experiment_artifact,
            aliases=self._catalog.list_aliases(manifest.artifact_id),
            artifact_inputs=self._catalog.list_artifact_inputs(manifest.artifact_id),
            status_history=self._catalog.get_artifact_status_history(manifest.artifact_id),
            payload=payload,
            summary=summary,
        )

    def _build_compiler(self) -> DatasetCompiler:
        return DatasetCompiler(self._cfg, self._paths, self._store, self._catalog)

    def _load_dataset_split_frames(self, manifest: LineageManifest) -> dict[str, pl.DataFrame]:
        artifact_root = Path(manifest.artifact_uri)
        split_names = sorted(str(name) for name in dict(manifest.metadata.get("split_row_counts", {})))
        frames: dict[str, pl.DataFrame] = {}
        for split_name in split_names:
            frame = self._store.read_dir(artifact_root / f"split={split_name}")
            frames[split_name] = _sorted_split_frame(frame)
        return frames

    @staticmethod
    def _assert_trainable_dataset(artifact: ArtifactRecord, trust_report: Any) -> None:
        if artifact.artifact_kind != "dataset":
            raise ValueError(f"Experiment dataset_ref must resolve to a dataset artifact, got {artifact.artifact_kind}")
        if artifact.status not in TRAINING_ALLOWED_DATASET_STATUSES:
            raise ValueError(
                f"Dataset artifact '{artifact.artifact_id}' is not trainable because status={artifact.status}"
            )
        if trust_report is None or not trust_report.accepted_for_publication:
            raise ValueError(
                f"Dataset artifact '{artifact.artifact_id}' is not trainable because it is not accepted by the truth gate"
            )

    @staticmethod
    def _assert_required_splits(split_frames: dict[str, pl.DataFrame]) -> None:
        missing = [name for name in ["train", "val", "test"] if name not in split_frames]
        if missing:
            raise ValueError(f"Dataset artifact is missing required splits for training: {', '.join(missing)}")
        empty = [name for name, frame in split_frames.items() if name in {"train", "val", "test"} and frame.is_empty()]
        if empty:
            raise ValueError(f"Dataset artifact contains empty required splits for training: {', '.join(empty)}")

    def _evaluate_walk_forward(
        self,
        *,
        dataset_df: pl.DataFrame,
        spec: ExperimentSpec,
        feature_columns: list[str],
        embargo_rows: int,
    ) -> tuple[dict[str, Any], pl.DataFrame]:
        folds = _build_walk_forward_folds(
            row_count=len(dataset_df),
            n_folds=spec.n_walk_forward_folds,
            min_train_rows=spec.min_train_rows,
            embargo_rows=embargo_rows,
        )
        fold_summaries: list[dict[str, Any]] = []
        prediction_frames: list[pl.DataFrame] = []
        balanced_accuracies: list[float] = []

        for fold_index, fold in enumerate(folds, start=1):
            train_df = dataset_df.slice(0, fold["train_end"])
            eval_df = dataset_df.slice(fold["eval_start"], fold["eval_rows"])
            model_payload, fit_summary = _fit_gaussian_nb_model(
                train_df,
                feature_columns=feature_columns,
                target_column=spec.target_column,
                positive_target_threshold=spec.positive_target_threshold,
            )
            metrics, predictions = _score_model(
                model_payload,
                eval_df,
                target_column=spec.target_column,
                positive_target_threshold=spec.positive_target_threshold,
                decision_threshold=spec.decision_threshold,
                split_name="walk_forward",
                fold_index=fold_index,
            )
            balanced_accuracies.append(float(metrics["balanced_accuracy"]))
            prediction_frames.append(predictions)
            fold_summaries.append(
                {
                    "fold_index": fold_index,
                    "train_rows": fit_summary["train_rows"],
                    "eval_rows": metrics["rows_scored"],
                    "embargo_rows": embargo_rows,
                    "train_positive_rate": fit_summary["positive_rate"],
                    "eval_positive_rate": metrics["positive_rate_true"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "accuracy": metrics["accuracy"],
                    "active_feature_count": len(model_payload["feature_columns"]),
                    "active_feature_columns": list(model_payload["feature_columns"]),
                    "dropped_all_null_columns": fit_summary["dropped_all_null_columns"],
                    "dropped_constant_columns": fit_summary["dropped_constant_columns"],
                }
            )

        if not balanced_accuracies:
            raise ValueError("Walk-forward evaluation produced zero valid folds")

        return (
            {
                "folds": fold_summaries,
                "balanced_accuracy_mean": round(sum(balanced_accuracies) / len(balanced_accuracies), 6),
                "balanced_accuracy_min": round(min(balanced_accuracies), 6),
                "balanced_accuracy_max": round(max(balanced_accuracies), 6),
                "fold_count": len(fold_summaries),
            },
            _concat_frames(prediction_frames),
        )

    @staticmethod
    def _fit_final_model(
        *,
        dataset_df: pl.DataFrame,
        spec: ExperimentSpec,
        feature_columns: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return _fit_gaussian_nb_model(
            dataset_df,
            feature_columns=feature_columns,
            target_column=spec.target_column,
            positive_target_threshold=spec.positive_target_threshold,
        )

    def _persist_experiment_artifact(
        self,
        *,
        spec: ExperimentSpec,
        run_id: str,
        created_at: dt.datetime,
        dataset_manifest: LineageManifest,
        dataset_artifact: ArtifactRecord,
        experiment_summary: dict[str, Any],
        predictions_df: pl.DataFrame,
        selected_feature_specs: list[FeatureSpec],
    ) -> tuple[str, LineageManifest, Path]:
        content_hash = compute_content_hash(
            {
                "experiment_spec": spec.model_dump(mode="json"),
                "dataset_artifact_id": dataset_artifact.artifact_id,
                "summary": experiment_summary,
                "prediction_rows": len(predictions_df),
            }
        )
        artifact_id = build_stage_artifact_id("experiment", spec.experiment_name, created_at, content_hash)
        experiment_root = self._paths.experiment_root(spec.experiment_name, artifact_id)
        predictions_path = self._paths.experiment_predictions_file(spec.experiment_name, artifact_id)
        summary_path = self._paths.experiment_summary_file(spec.experiment_name, artifact_id)
        experiment_root.mkdir(parents=True, exist_ok=True)
        if not predictions_df.is_empty():
            self._store.write(predictions_df, predictions_path)
        summary_path.write_text(json.dumps(experiment_summary, indent=2, sort_keys=True, default=str), encoding="utf-8")

        metadata = {
            **experiment_summary,
            "predictions_path": str(predictions_path),
            "summary_path": str(summary_path),
        }
        manifest = LineageManifest(
            manifest_id=build_stage_manifest_id("experiment", spec.experiment_name, created_at, content_hash),
            artifact_id=artifact_id,
            artifact_kind="experiment",
            logical_name=spec.experiment_name,
            logical_version=spec.version,
            artifact_uri=str(experiment_root),
            content_hash=content_hash,
            build_id=run_id,
            created_at=created_at,
            status="building",
            dataset_spec_ref=dataset_manifest.dataset_spec_ref,
            experiment_spec_ref=spec.key,
            state_artifact_refs=list(dataset_manifest.state_artifact_refs),
            feature_spec_refs=[feature.key for feature in selected_feature_specs],
            label_pack_ref=dataset_manifest.label_pack_ref,
            code_version=code_version(),
            merge_config_ref=dataset_manifest.merge_config_ref,
            input_partition_refs=list(dataset_manifest.input_partition_refs),
            parent_artifact_refs=[dataset_artifact.artifact_id],
            metadata=metadata,
        )
        self._persist_manifest(manifest, detail="experiment artifact building")
        final_manifest = manifest.model_copy(update={"status": "accepted"})
        final_manifest_path = self._persist_manifest(final_manifest, detail="experiment artifact registered")
        return artifact_id, final_manifest, final_manifest_path

    def _persist_model_artifact(
        self,
        *,
        spec: ExperimentSpec,
        run_id: str,
        created_at: dt.datetime,
        dataset_manifest: LineageManifest,
        dataset_artifact: ArtifactRecord,
        experiment_artifact_id: str,
        experiment_summary: dict[str, Any],
        model_payload: dict[str, Any],
        selected_feature_specs: list[FeatureSpec],
        model_status: str,
    ) -> tuple[str, LineageManifest, Path]:
        content_hash = compute_content_hash(
            {
                "experiment_spec": spec.model_dump(mode="json"),
                "dataset_artifact_id": dataset_artifact.artifact_id,
                "experiment_artifact_id": experiment_artifact_id,
                "model_payload": model_payload,
                "model_status": model_status,
            }
        )
        artifact_id = build_stage_artifact_id("model", spec.model_name, created_at, content_hash)
        model_root = self._paths.model_root(spec.model_name, artifact_id)
        payload_path = self._paths.model_payload_file(spec.model_name, artifact_id)
        model_root.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(json.dumps(model_payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

        metadata = {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "experiment_spec_key": spec.key,
            "dataset_ref": spec.dataset_ref,
            "dataset_artifact_id": dataset_artifact.artifact_id,
            "experiment_artifact_id": experiment_artifact_id,
            "target_column": spec.target_column,
            "model_family": spec.model_family,
            "model_status": model_status,
            "feature_families": experiment_summary["feature_families"],
            "selected_feature_columns": experiment_summary["selected_feature_columns"],
            "active_feature_columns": list(model_payload["feature_columns"]),
            "walk_forward_summary": experiment_summary["walk_forward_summary"],
            "holdout_metrics": experiment_summary["holdout_metrics"],
            "summary_path": experiment_summary["summary_path"],
            "model_payload_path": str(payload_path),
        }
        manifest = LineageManifest(
            manifest_id=build_stage_manifest_id("model", spec.model_name, created_at, content_hash),
            artifact_id=artifact_id,
            artifact_kind="model",
            logical_name=spec.model_name,
            logical_version=spec.version,
            artifact_uri=str(model_root),
            content_hash=content_hash,
            build_id=run_id,
            created_at=created_at,
            status="building",
            dataset_spec_ref=dataset_manifest.dataset_spec_ref,
            experiment_spec_ref=spec.key,
            state_artifact_refs=list(dataset_manifest.state_artifact_refs),
            feature_spec_refs=[feature.key for feature in selected_feature_specs],
            label_pack_ref=dataset_manifest.label_pack_ref,
            code_version=code_version(),
            merge_config_ref=dataset_manifest.merge_config_ref,
            input_partition_refs=list(dataset_manifest.input_partition_refs),
            parent_artifact_refs=[dataset_artifact.artifact_id, experiment_artifact_id],
            metadata=metadata,
        )
        self._persist_manifest(manifest, detail="model artifact building")
        final_manifest = manifest.model_copy(update={"status": model_status})
        final_manifest_path = self._persist_manifest(final_manifest, detail=f"model artifact registered with status={model_status}")
        return artifact_id, final_manifest, final_manifest_path

    def _persist_manifest(self, manifest: LineageManifest, *, detail: str) -> Path:
        manifest_path = write_manifest_sidecar(manifest, self._paths)
        self._catalog.register_artifact(manifest, str(manifest_path), detail=detail)
        return manifest_path

    def _resolve_manifest_with_path(self, ref: str) -> tuple[LineageManifest, Path]:
        maybe_path = Path(ref)
        if maybe_path.exists():
            return read_manifest_sidecar(maybe_path), maybe_path
        artifact = self._catalog.resolve_artifact(ref)
        if artifact is None:
            raise KeyError(f"Artifact '{ref}' was not found in the compiler catalog")
        manifest_path = Path(artifact.manifest_uri)
        return read_manifest_sidecar(manifest_path), manifest_path


def run_experiment_spec(spec_path: str | Path) -> ExperimentRunResult:
    runner, catalog = ExperimentRunner.from_config_path(_resolve_pipeline_config_path_from_spec(spec_path))
    try:
        return runner.run_experiment(Path(spec_path))
    finally:
        catalog.close()


def inspect_experiment(ref: str) -> ExperimentInspection:
    runner, catalog = ExperimentRunner.from_config_path(_resolve_pipeline_config_path_from_ref(ref))
    try:
        return runner.inspect_experiment(ref)
    finally:
        catalog.close()


def inspect_model(ref: str) -> ModelInspection:
    runner, catalog = ExperimentRunner.from_config_path(_resolve_pipeline_config_path_from_ref(ref))
    try:
        return runner.inspect_model(ref)
    finally:
        catalog.close()


def _training_run_id_now() -> str:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"train.{ts}"


def _select_feature_specs(feature_specs: list[FeatureSpec], families: list[str]) -> list[FeatureSpec]:
    if not families:
        return list(feature_specs)
    family_set = {family.strip() for family in families if family.strip()}
    return [feature for feature in feature_specs if feature.family in family_set]


def _select_feature_columns(
    *,
    schema_columns: list[str],
    feature_specs: list[FeatureSpec],
    exclude_columns: list[str],
) -> list[str]:
    excluded = set(exclude_columns)
    allowed = {
        column
        for feature in feature_specs
        for column in feature.output_columns
        if column not in excluded
    }
    return [column for column in schema_columns if column in allowed]


def _sorted_split_frame(frame: pl.DataFrame | None) -> pl.DataFrame:
    if frame is None or frame.is_empty():
        return pl.DataFrame()
    result = frame
    if {"symbol", "timeframe", "time_utc"}.issubset(set(result.columns)):
        result = result.unique(subset=["symbol", "timeframe", "time_utc"], keep="last")
    if "time_utc" in result.columns:
        result = result.sort("time_utc")
    return result


def _combine_split_frames(*frames: pl.DataFrame | None) -> pl.DataFrame:
    present = [frame for frame in frames if frame is not None and not frame.is_empty()]
    if not present:
        return pl.DataFrame()
    return _sorted_split_frame(pl.concat(present, how="diagonal_relaxed"))


def _build_walk_forward_folds(
    *,
    row_count: int,
    n_folds: int,
    min_train_rows: int,
    embargo_rows: int,
) -> list[dict[str, int]]:
    available_eval_rows = row_count - min_train_rows - embargo_rows
    if available_eval_rows < n_folds:
        raise ValueError(
            "Dataset artifact does not have enough sequential rows for the requested walk-forward evaluation "
            f"(rows={row_count}, min_train_rows={min_train_rows}, embargo_rows={embargo_rows}, n_folds={n_folds})"
        )
    step = max(1, available_eval_rows // n_folds)
    folds: list[dict[str, int]] = []
    for fold_index in range(n_folds):
        train_end = min_train_rows + fold_index * step
        eval_start = train_end + embargo_rows
        if eval_start >= row_count:
            break
        eval_end = row_count if fold_index == n_folds - 1 else min(row_count, eval_start + step)
        eval_rows = max(0, eval_end - eval_start)
        if train_end <= 0 or eval_rows <= 0:
            continue
        folds.append(
            {
                "train_end": train_end,
                "eval_start": eval_start,
                "eval_rows": eval_rows,
            }
        )
    if not folds:
        raise ValueError("Walk-forward evaluation did not yield any valid temporal folds")
    return folds


def _fit_gaussian_nb_model(
    dataset_df: pl.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    positive_target_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if dataset_df.is_empty():
        raise ValueError("Cannot train a model from an empty dataset frame")
    if "time_utc" not in dataset_df.columns:
        raise ValueError("Dataset frame is missing required time_utc column for temporal training")
    if target_column not in dataset_df.columns:
        raise ValueError(f"Dataset frame is missing required target column '{target_column}'")

    feature_frame = dataset_df.select(
        [
            *([pl.col("symbol")] if "symbol" in dataset_df.columns else [pl.lit("").alias("symbol")]),
            *([pl.col("timeframe")] if "timeframe" in dataset_df.columns else [pl.lit("").alias("timeframe")]),
            pl.col("time_utc"),
            pl.col(target_column).cast(pl.Float64).alias(target_column),
            *[pl.col(column).cast(pl.Float64, strict=False).alias(column) for column in feature_columns if column in dataset_df.columns],
        ]
    ).filter(pl.col(target_column).is_not_null())

    if feature_frame.is_empty():
        raise ValueError(f"No trainable rows remain after filtering null target values for '{target_column}'")

    active_feature_columns: list[str] = []
    dropped_all_null_columns: list[str] = []
    dropped_constant_columns: list[str] = []
    impute_values: dict[str, float] = {}

    for column in feature_columns:
        if column not in feature_frame.columns:
            continue
        series = feature_frame[column]
        non_null = series.drop_nulls()
        if non_null.is_empty():
            dropped_all_null_columns.append(column)
            continue
        median_value = float(non_null.median())
        filled = series.fill_null(median_value)
        if filled.n_unique() <= 1:
            dropped_constant_columns.append(column)
            continue
        active_feature_columns.append(column)
        impute_values[column] = median_value

    if not active_feature_columns:
        raise ValueError("No trainable feature columns remain after removing all-null and constant columns")

    prepared = _prepare_matrix(
        feature_frame,
        feature_columns=active_feature_columns,
        target_column=target_column,
        positive_target_threshold=positive_target_threshold,
        impute_values=impute_values,
    )
    class_counts = {
        0: sum(1 for target in prepared.targets if target == 0),
        1: sum(1 for target in prepared.targets if target == 1),
    }
    if class_counts[0] == 0 or class_counts[1] == 0:
        raise ValueError(
            "Training rows must contain both classes after target binarization; "
            f"observed counts={class_counts}"
        )

    class_stats: dict[str, Any] = {}
    total_rows = len(prepared.targets)
    for cls in [0, 1]:
        class_rows = [row for row, target in zip(prepared.rows, prepared.targets) if target == cls]
        prior = len(class_rows) / total_rows
        means: list[float] = []
        variances: list[float] = []
        for index in range(len(active_feature_columns)):
            values = [row[index] for row in class_rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            means.append(float(mean))
            variances.append(float(max(variance, 1e-9)))
        class_stats[str(cls)] = {
            "prior": float(prior),
            "means": means,
            "variances": variances,
        }

    model_payload = {
        "schema_version": "1.0.0",
        "model_family": "gaussian_nb_binary@1.0.0",
        "feature_columns": active_feature_columns,
        "impute_values": impute_values,
        "positive_target_threshold": float(positive_target_threshold),
        "class_stats": class_stats,
    }
    summary = {
        "train_rows": total_rows,
        "positive_rate": round(class_counts[1] / total_rows, 6),
        "class_counts": class_counts,
        "active_feature_columns": active_feature_columns,
        "dropped_all_null_columns": dropped_all_null_columns,
        "dropped_constant_columns": dropped_constant_columns,
    }
    return model_payload, summary


def _prepare_matrix(
    dataset_df: pl.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    positive_target_threshold: float,
    impute_values: dict[str, float],
) -> _PreparedMatrix:
    rows: list[list[float]] = []
    targets: list[int] = []
    target_values: list[float] = []
    timestamps: list[dt.datetime] = []
    symbols: list[str] = []
    clocks: list[str] = []

    ordered = dataset_df.sort("time_utc") if "time_utc" in dataset_df.columns else dataset_df
    for row in ordered.iter_rows(named=True):
        target_value = row[target_column]
        if target_value is None:
            continue
        target_float = float(target_value)
        feature_row: list[float] = []
        for column in feature_columns:
            value = row.get(column)
            if value is None:
                feature_row.append(float(impute_values[column]))
            else:
                feature_row.append(float(value))
        rows.append(feature_row)
        targets.append(1 if target_float > positive_target_threshold else 0)
        target_values.append(target_float)
        timestamps.append(row["time_utc"])
        symbols.append(str(row.get("symbol", "")))
        clocks.append(str(row.get("timeframe", "")))

    if not rows:
        raise ValueError("No usable rows remained for model scoring after filtering null targets")

    return _PreparedMatrix(
        feature_columns=feature_columns,
        rows=rows,
        targets=targets,
        target_values=target_values,
        timestamps=timestamps,
        symbols=symbols,
        clocks=clocks,
    )


def _score_model(
    model_payload: dict[str, Any],
    dataset_df: pl.DataFrame,
    *,
    target_column: str,
    positive_target_threshold: float,
    decision_threshold: float,
    split_name: str,
    fold_index: int,
) -> tuple[dict[str, Any], pl.DataFrame]:
    prepared = _prepare_matrix(
        dataset_df,
        feature_columns=list(model_payload["feature_columns"]),
        target_column=target_column,
        positive_target_threshold=positive_target_threshold,
        impute_values={str(key): float(value) for key, value in dict(model_payload["impute_values"]).items()},
    )
    class_stats = dict(model_payload["class_stats"])
    probabilities: list[float] = []
    predictions: list[int] = []

    for row in prepared.rows:
        log_probabilities: dict[int, float] = {}
        for cls in [0, 1]:
            stats = class_stats[str(cls)]
            log_prob = math.log(max(float(stats["prior"]), 1e-12))
            for value, mean, variance in zip(row, stats["means"], stats["variances"]):
                variance_value = max(float(variance), 1e-9)
                log_prob += -0.5 * math.log(2.0 * math.pi * variance_value)
                log_prob += -((value - float(mean)) ** 2) / (2.0 * variance_value)
            log_probabilities[cls] = log_prob

        max_log_prob = max(log_probabilities.values())
        normalized = {
            cls: math.exp(log_prob - max_log_prob)
            for cls, log_prob in log_probabilities.items()
        }
        prob_positive = normalized[1] / (normalized[0] + normalized[1])
        probabilities.append(float(prob_positive))
        predictions.append(1 if prob_positive >= decision_threshold else 0)

    metrics = _binary_classification_metrics(prepared.targets, predictions)
    metrics.update(
        {
            "rows_scored": len(prepared.targets),
            "positive_rate_true": round(sum(prepared.targets) / len(prepared.targets), 6),
            "positive_rate_pred": round(sum(predictions) / len(predictions), 6),
            "decision_threshold": float(decision_threshold),
        }
    )

    prediction_frame = pl.DataFrame(
        {
            "time_utc": prepared.timestamps,
            "symbol": prepared.symbols,
            "timeframe": prepared.clocks,
            "split": [split_name] * len(prepared.targets),
            "fold_index": [fold_index] * len(prepared.targets),
            "target_column": [target_column] * len(prepared.targets),
            "target_value": prepared.target_values,
            "target_binary": prepared.targets,
            "pred_prob_positive": probabilities,
            "pred_binary": predictions,
            "correct": [pred == actual for pred, actual in zip(predictions, prepared.targets)],
        }
    )
    return metrics, prediction_frame


def _binary_classification_metrics(targets: list[int], predictions: list[int]) -> dict[str, Any]:
    if len(targets) != len(predictions):
        raise ValueError("Targets and predictions must have identical lengths")
    if not targets:
        raise ValueError("Cannot compute classification metrics on zero rows")

    tp = sum(1 for actual, pred in zip(targets, predictions) if actual == 1 and pred == 1)
    tn = sum(1 for actual, pred in zip(targets, predictions) if actual == 0 and pred == 0)
    fp = sum(1 for actual, pred in zip(targets, predictions) if actual == 0 and pred == 1)
    fn = sum(1 for actual, pred in zip(targets, predictions) if actual == 1 and pred == 0)
    positives = tp + fn
    negatives = tn + fp
    if positives == 0 or negatives == 0:
        raise ValueError(
            "Evaluation rows must contain both classes to compute balanced accuracy; "
            f"observed positives={positives}, negatives={negatives}"
        )

    tpr = tp / positives
    tnr = tn / negatives
    accuracy = (tp + tn) / len(targets)
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "balanced_accuracy": round((tpr + tnr) / 2.0, 6),
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(tpr, 6),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _concat_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
    present = [frame for frame in frames if not frame.is_empty()]
    if not present:
        return pl.DataFrame()
    return pl.concat(present, how="diagonal_relaxed")


def _combine_prediction_frames(
    *frames: pl.DataFrame,
    run_id: str,
    dataset_artifact_id: str,
    experiment_key: str,
    model_status: str,
) -> pl.DataFrame:
    combined = _concat_frames(list(frames))
    if combined.is_empty():
        return combined
    return combined.with_columns(
        [
            pl.lit(run_id).alias("run_id"),
            pl.lit(dataset_artifact_id).alias("dataset_artifact_id"),
            pl.lit(experiment_key).alias("experiment_spec_key"),
            pl.lit(model_status).alias("model_status"),
        ]
    )


def _load_json_payload(raw_path: Any) -> dict[str, Any]:
    path = _maybe_existing_path(raw_path)
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_existing_path(raw_path: Any) -> Path | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.exists():
        return None
    return path
