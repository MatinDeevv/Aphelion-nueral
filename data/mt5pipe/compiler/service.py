"""Core compiler services for build, inspection, and artifact diffing."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path

import polars as pl

from mt5pipe.catalog.models import AliasRecord, ArtifactInputRecord, ArtifactRecord, ArtifactStatusEventRecord
from mt5pipe.catalog.sqlite import CatalogDB
from mt5pipe.compiler.manifest import (
    build_artifact_id,
    build_id_now,
    build_manifest_id,
    build_stage_artifact_id,
    build_stage_manifest_id,
    code_version,
    compute_content_hash,
    load_dataset_spec,
    merge_config_ref,
    read_manifest_sidecar,
    write_manifest_sidecar,
)
from mt5pipe.compiler.models import DatasetSpec, LineageManifest
from mt5pipe.compiler.registry import (
    feature_spec_index,
    register_builtin_contracts,
    resolve_feature_selectors,
    resolve_label_pack,
)
from mt5pipe.config.loader import load_config
from mt5pipe.config.models import PipelineConfig
from mt5pipe.features.public import FeatureService, FeatureSpec, LabelPack, LabelService
from mt5pipe.quality.cleaning import clean_dataset
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths
from mt5pipe.truth.models import TrustReport
from mt5pipe.truth.service import TruthService


MANDATORY_DATASET_COLUMNS = [
    "symbol",
    "timeframe",
    "time_utc",
    "open",
    "high",
    "low",
    "close",
    "tick_count",
    "spread_mean",
    "mid_return",
    "realized_vol",
    "source_count",
    "conflict_count",
    "dual_source_ticks",
    "secondary_present_ticks",
    "dual_source_ratio",
]

DATASET_JOIN_KEYS = ["symbol", "timeframe", "time_utc"]
DEFAULT_CONFIG_RELATIVE_PATH = Path("config/pipeline.yaml")
STATE_CONTROL_COLUMNS = [
    "_filled",
    "quality_score",
    "conflict_flag",
    "trust_flags",
    "source_quality_hint",
    "disagreement_bps",
    "spread_disagreement_bps",
]


@dataclass
class BuildResult:
    spec: DatasetSpec
    build_id: str
    artifact_id: str
    manifest: LineageManifest
    manifest_path: Path
    trust_report: TrustReport
    truth_report_path: Path
    split_row_counts: dict[str, int]
    published_aliases: list[str] = field(default_factory=list)
    status_history: list[str] = field(default_factory=list)


@dataclass
class ArtifactInspection:
    ref: str
    artifact: ArtifactRecord
    manifest: LineageManifest
    manifest_path: Path
    trust_report: TrustReport | None
    dataset_spec: DatasetSpec | None
    feature_specs: list[FeatureSpec]
    label_pack: LabelPack | None
    aliases: list[AliasRecord]
    artifact_inputs: list[ArtifactInputRecord]
    status_history: list[ArtifactStatusEventRecord]
    feature_families: list[str]
    time_range: dict[str, str]
    split_row_counts: dict[str, int]
    schema_columns: list[str]
    trust_score_breakdown: dict[str, float | None]
    trust_decision_summary: str
    trust_rejection_reasons: list[str]
    trust_warning_reasons: list[str]
    trust_check_status_counts: dict[str, int]
    lineage_refs: dict[str, list[str]]
    requested_feature_selectors: list[str]
    build_row_stats: dict[str, int]
    source_modes: dict[str, str]
    feature_artifact_refs: list[str]


@dataclass
class ArtifactDiff:
    left: ArtifactInspection
    right: ArtifactInspection
    diff: dict[str, object]


@dataclass
class ResolvedStateInput:
    artifact_id: str
    manifest: LineageManifest
    manifest_path: Path
    state_df: pl.DataFrame
    base_df: pl.DataFrame
    source_mode: str


@dataclass
class ResolvedFeatureSource:
    spec: FeatureSpec
    artifact_id: str
    manifest: LineageManifest
    manifest_path: Path
    frame: pl.DataFrame
    source_mode: str


@dataclass
class ResolvedLabelInput:
    label_pack: LabelPack
    artifact_id: str
    manifest: LineageManifest
    manifest_path: Path
    label_df: pl.DataFrame
    source_mode: str


CompileDatasetResult = BuildResult
InspectDatasetResult = ArtifactInspection
DiffDatasetResult = ArtifactDiff


class DatasetCompiler:
    """Compiler-first implementation for explicit dataset artifacts."""

    def __init__(
        self,
        cfg: PipelineConfig,
        paths: StoragePaths,
        store: ParquetStore,
        catalog: CatalogDB,
    ) -> None:
        self._cfg = cfg
        self._paths = paths
        self._store = store
        self._catalog = catalog
        self._truth = TruthService()
        self._state, self._state_boot_error = self._build_state_service(paths, store, catalog)
        self._features = FeatureService(paths, store, catalog, cfg.dataset)
        self._labels = LabelService(paths, store, catalog)
        register_builtin_contracts(self._catalog)

    @staticmethod
    def _build_state_service(paths: StoragePaths, store: ParquetStore, catalog: CatalogDB) -> tuple[object | None, Exception | None]:
        try:
            state_public = import_module("mt5pipe.state.public")
            state_service_cls = getattr(state_public, "StateService")
            return state_service_cls(paths, store, catalog), None
        except Exception as exc:  # pragma: no cover - exercised through compiler tests when state sector is broken
            return None, exc

    @classmethod
    def from_config_path(cls, config_path: str | Path | None = None) -> tuple["DatasetCompiler", CatalogDB]:
        resolved_config_path = _resolve_pipeline_config_path(config_path)
        cfg = load_config(resolved_config_path)
        paths = StoragePaths(cfg.storage.root)
        store = ParquetStore(cfg.storage.compression, cfg.storage.parquet_row_group_size)
        catalog = CatalogDB(paths.catalog_db_path())
        return cls(cfg, paths, store, catalog), catalog

    def compile_dataset(self, spec_path: Path, *, publish: bool | None = None) -> BuildResult:
        spec = load_dataset_spec(spec_path)
        self._catalog.register_dataset_spec(spec)

        if len(spec.symbols) != 1:
            raise NotImplementedError("Current compiler core supports exactly one symbol per DatasetSpec")
        if spec.base_clock != "M1":
            raise NotImplementedError("Current compiler core supports base_clock='M1' only")

        symbol = spec.symbols[0]
        resolved_code_version = code_version()
        resolved_merge_config_ref = merge_config_ref(self._cfg.merge)
        publish_requested = spec.publish_on_accept if publish is None else bool(publish)

        build_id = build_id_now()
        self._catalog.start_build(spec.key, resolved_code_version, build_id)

        try:
            state_input = self._resolve_state_input(
                spec=spec,
                symbol=symbol,
                build_id=build_id,
                code_version_value=resolved_code_version,
                merge_config_ref_value=resolved_merge_config_ref,
            )

            provided_feature_sources = self._resolve_feature_artifact_sources(
                spec=spec,
                symbol=symbol,
            )
            extra_feature_specs = [source.spec for source in provided_feature_sources.values()]
            feature_specs = resolve_feature_selectors(
                spec.feature_selectors,
                catalog=self._catalog,
                extra_specs=extra_feature_specs,
            )
            self._catalog.register_feature_specs(feature_specs)

            feature_sources = self._build_feature_sources(
                spec=spec,
                symbol=symbol,
                feature_specs=feature_specs,
                provided_feature_sources=provided_feature_sources,
                state_input=state_input,
                build_id=build_id,
                code_version_value=resolved_code_version,
            )

            label_input = self._resolve_label_input(
                spec=spec,
                symbol=symbol,
                state_input=state_input,
                build_id=build_id,
                code_version_value=resolved_code_version,
            )

            feature_df = self._combine_feature_frames(state_input.base_df, feature_sources)
            compiled_df, build_row_stats = self._compile_dataset_frame(
                base_df=state_input.base_df,
                feature_df=feature_df,
                label_df=label_input.label_df,
                spec=spec,
                feature_specs=feature_specs,
                label_pack=label_input.label_pack,
            )
            selected_df = self._select_dataset_columns(compiled_df, feature_specs, label_input.label_pack)
            split_frames = self._split_dataset(selected_df, spec)

            created_at = dt.datetime.now(dt.timezone.utc)
            content_hash = self._dataset_content_hash(
                spec=spec,
                dataset_df=selected_df,
                split_frames=split_frames,
                state_artifact_id=state_input.artifact_id,
                feature_artifact_ids=[source.artifact_id for source in feature_sources],
                label_artifact_id=label_input.artifact_id,
            )
            artifact_id = build_artifact_id(spec.dataset_name, created_at, content_hash)
            artifact_root = self._paths.root / "datasets" / f"name={spec.dataset_name}" / f"artifact={artifact_id}"

            self._write_compiled_artifact(spec, artifact_id, split_frames)

            parent_artifact_refs = [
                state_input.artifact_id,
                *[source.artifact_id for source in feature_sources],
                label_input.artifact_id,
            ]
            input_partition_refs = sorted(
                set(
                    state_input.manifest.input_partition_refs
                    + [ref for source in feature_sources for ref in source.manifest.input_partition_refs]
                    + label_input.manifest.input_partition_refs
                )
            )
            feature_families = sorted({feature.family for feature in feature_specs})
            metadata = {
                "schema_columns": selected_df.columns,
                "split_row_counts": {name: len(frame) for name, frame in split_frames.items()},
                "time_range_start": str(selected_df["time_utc"].min()) if not selected_df.is_empty() else "",
                "time_range_end": str(selected_df["time_utc"].max()) if not selected_df.is_empty() else "",
                "feature_families": feature_families,
                "requested_feature_selectors": list(spec.feature_selectors),
                "feature_artifact_refs": list(spec.feature_artifact_refs),
                "state_artifact_ref": spec.state_artifact_ref or "",
                "label_artifact_ref": spec.label_artifact_ref or "",
                "source_modes": {
                    "state": state_input.source_mode,
                    "features": "mixed" if _has_multiple_source_modes(feature_sources) else _single_feature_source_mode(feature_sources),
                    "label": label_input.source_mode,
                },
                "build_row_stats": build_row_stats,
                "upstream_artifacts": parent_artifact_refs,
            }
            manifest = LineageManifest(
                manifest_id=build_manifest_id(spec.dataset_name, created_at, content_hash),
                artifact_id=artifact_id,
                artifact_kind="dataset",
                logical_name=spec.dataset_name,
                logical_version=spec.version,
                artifact_uri=str(artifact_root),
                content_hash=content_hash,
                build_id=build_id,
                created_at=created_at,
                status="building",
                dataset_spec_ref=spec.key,
                state_artifact_refs=[state_input.artifact_id],
                feature_spec_refs=[feature.key for feature in feature_specs],
                label_pack_ref=label_input.label_pack.key,
                code_version=resolved_code_version,
                merge_config_ref=resolved_merge_config_ref,
                input_partition_refs=input_partition_refs,
                parent_artifact_refs=parent_artifact_refs,
                metadata=metadata,
            )

            manifest_path = self._persist_manifest(manifest, detail="compiler candidate created")
            self._catalog.update_build_status(build_id, "building", artifact_id=artifact_id)

            manifest = manifest.model_copy(update={"status": "truth_pending"})
            manifest_path = self._persist_manifest(manifest, detail="awaiting truth gate")
            persisted_truth_pending = read_manifest_sidecar(manifest_path)
            self._catalog.update_build_status(build_id, "truth_pending", artifact_id=artifact_id)

            trust_report = self._truth.evaluate_dataset(
                artifact_id=artifact_id,
                dataset_df=selected_df,
                split_frames=split_frames,
                spec=spec,
                feature_specs=feature_specs,
                label_pack=label_input.label_pack,
                manifest=persisted_truth_pending,
                paths=self._paths,
                store=self._store,
                state_df=state_input.state_df,
                build_row_stats=build_row_stats,
                expected_content_hash=content_hash,
                manifest_path=manifest_path,
            )
            truth_path = self._write_truth_report(trust_report)
            self._catalog.register_trust_report(trust_report)

            final_manifest = persisted_truth_pending.model_copy(
                update={"status": "rejected", "truth_report_ref": trust_report.report_id}
            )
            published_aliases: list[str] = []

            if trust_report.accepted_for_publication:
                accepted_manifest = final_manifest.model_copy(update={"status": "accepted"})
                manifest_path = self._persist_manifest(accepted_manifest, detail="truth gate accepted artifact")
                self._catalog.update_build_status(build_id, "accepted", artifact_id=artifact_id)
                final_manifest = accepted_manifest

                if publish_requested:
                    try:
                        self._assert_publishable(final_manifest, manifest_path, trust_report, content_hash)
                    except Exception as exc:
                        final_manifest = final_manifest.model_copy(update={"status": "rejected"})
                        manifest_path = self._persist_manifest(
                            final_manifest,
                            detail=f"publication rejected: {exc}",
                        )
                        self._catalog.finish_build(
                            build_id,
                            "rejected",
                            artifact_id=artifact_id,
                            error_message=str(exc),
                        )
                    else:
                        final_manifest = final_manifest.model_copy(update={"status": "published"})
                        manifest_path = self._persist_manifest(final_manifest, detail="artifact published")
                        published_aliases = [
                            f"dataset://{spec.dataset_name}@{spec.version}",
                            f"dataset://{spec.dataset_name}:latest",
                        ]
                        for alias in published_aliases:
                            self._catalog.upsert_alias(alias, artifact_id)
                        self._catalog.finish_build(build_id, "published", artifact_id=artifact_id)
                else:
                    self._catalog.finish_build(build_id, "accepted", artifact_id=artifact_id)
            else:
                manifest_path = self._persist_manifest(final_manifest, detail="truth gate rejected artifact")
                self._catalog.finish_build(
                    build_id,
                    "rejected",
                    artifact_id=artifact_id,
                    error_message="truth gate rejected candidate dataset",
                )

            status_history = [event.status for event in self._catalog.get_artifact_status_history(artifact_id)]
            return BuildResult(
                spec=spec,
                build_id=build_id,
                artifact_id=artifact_id,
                manifest=final_manifest,
                manifest_path=manifest_path,
                trust_report=trust_report,
                truth_report_path=truth_path,
                split_row_counts={name: len(frame) for name, frame in split_frames.items()},
                published_aliases=published_aliases,
                status_history=status_history,
            )
        except Exception as exc:
            self._catalog.finish_build(build_id, "failed", error_message=str(exc))
            raise

    def inspect_dataset(self, ref: str) -> ArtifactInspection:
        manifest, manifest_path = self._resolve_manifest_with_path(ref)
        artifact = self._catalog.get_artifact(manifest.artifact_id)
        if artifact is None:
            raise KeyError(f"Artifact '{manifest.artifact_id}' is missing from the compiler catalog")

        trust_report = self._catalog.get_trust_report(manifest.artifact_id)
        dataset_spec = self._catalog.get_dataset_spec(manifest.dataset_spec_ref) if manifest.dataset_spec_ref else None
        feature_specs = [
            spec
            for spec in (self._catalog.get_feature_spec(key) for key in manifest.feature_spec_refs)
            if spec is not None
        ]
        label_pack = self._catalog.get_label_pack(manifest.label_pack_ref) if manifest.label_pack_ref else None
        aliases = self._catalog.list_aliases(manifest.artifact_id)
        artifact_inputs = self._catalog.list_artifact_inputs(manifest.artifact_id)
        status_history = self._catalog.get_artifact_status_history(manifest.artifact_id)
        feature_families = sorted({spec.family for spec in feature_specs})
        time_range = {
            "start": str(manifest.metadata.get("time_range_start", "")),
            "end": str(manifest.metadata.get("time_range_end", "")),
        }
        split_row_counts = dict(manifest.metadata.get("split_row_counts", {}))
        schema_columns = list(manifest.metadata.get("schema_columns", []))
        trust_score_breakdown = {
            "total": trust_report.trust_score_total if trust_report else None,
            "coverage": trust_report.coverage_score if trust_report else None,
            "leakage": trust_report.leakage_score if trust_report else None,
            "feature_quality": trust_report.feature_quality_score if trust_report else None,
            "label_quality": trust_report.label_quality_score if trust_report else None,
            "source_quality": trust_report.source_quality_score if trust_report else None,
            "lineage": trust_report.lineage_score if trust_report else None,
        }
        trust_decision_summary = trust_report.decision_summary if trust_report else ""
        trust_rejection_reasons = list(trust_report.rejection_reasons) if trust_report else []
        trust_warning_reasons = list(trust_report.warning_reasons) if trust_report else []
        trust_check_status_counts = dict(trust_report.check_status_counts) if trust_report else {}
        lineage_refs = {
            "input_partition_refs": list(manifest.input_partition_refs),
            "state_artifact_refs": list(manifest.state_artifact_refs),
            "parent_artifact_refs": list(manifest.parent_artifact_refs),
            "artifact_inputs": [record.input_ref for record in artifact_inputs],
        }

        return ArtifactInspection(
            ref=ref,
            artifact=artifact,
            manifest=manifest,
            manifest_path=manifest_path,
            trust_report=trust_report,
            dataset_spec=dataset_spec,
            feature_specs=feature_specs,
            label_pack=label_pack,
            aliases=aliases,
            artifact_inputs=artifact_inputs,
            status_history=status_history,
            feature_families=feature_families,
            time_range=time_range,
            split_row_counts=split_row_counts,
            schema_columns=schema_columns,
            trust_score_breakdown=trust_score_breakdown,
            trust_decision_summary=trust_decision_summary,
            trust_rejection_reasons=trust_rejection_reasons,
            trust_warning_reasons=trust_warning_reasons,
            trust_check_status_counts=trust_check_status_counts,
            lineage_refs=lineage_refs,
            requested_feature_selectors=[str(item) for item in manifest.metadata.get("requested_feature_selectors", [])],
            build_row_stats={str(key): int(value) for key, value in dict(manifest.metadata.get("build_row_stats", {})).items()},
            source_modes={str(key): str(value) for key, value in dict(manifest.metadata.get("source_modes", {})).items()},
            feature_artifact_refs=[str(item) for item in manifest.metadata.get("feature_artifact_refs", [])],
        )

    def diff_datasets(self, left_ref: str, right_ref: str) -> ArtifactDiff:
        left = self.inspect_dataset(left_ref)
        right = self.inspect_dataset(right_ref)

        left_spec_dump = left.dataset_spec.model_dump(mode="json") if left.dataset_spec is not None else {}
        right_spec_dump = right.dataset_spec.model_dump(mode="json") if right.dataset_spec is not None else {}
        spec_diff = {
            key: {"left": left_spec_dump.get(key), "right": right_spec_dump.get(key)}
            for key in sorted(set(left_spec_dump) | set(right_spec_dump))
            if left_spec_dump.get(key) != right_spec_dump.get(key)
        }

        left_feature_keys = [spec.key for spec in left.feature_specs]
        right_feature_keys = [spec.key for spec in right.feature_specs]
        schema_left = set(left.schema_columns)
        schema_right = set(right.schema_columns)
        lineage_left = set(left.lineage_refs["artifact_inputs"])
        lineage_right = set(right.lineage_refs["artifact_inputs"])

        diff = {
            "artifact_id_changed": left.artifact.artifact_id != right.artifact.artifact_id,
            "logical_name_changed": left.artifact.logical_name != right.artifact.logical_name,
            "logical_version_changed": left.artifact.logical_version != right.artifact.logical_version,
            "dataset_spec_ref_changed": left.manifest.dataset_spec_ref != right.manifest.dataset_spec_ref,
            "dataset_spec_differences": spec_diff,
            "feature_spec_refs_added": sorted(set(right_feature_keys) - set(left_feature_keys)),
            "feature_spec_refs_removed": sorted(set(left_feature_keys) - set(right_feature_keys)),
            "feature_families_added": sorted(set(right.feature_families) - set(left.feature_families)),
            "feature_families_removed": sorted(set(left.feature_families) - set(right.feature_families)),
            "feature_selectors_left": left.requested_feature_selectors,
            "feature_selectors_right": right.requested_feature_selectors,
            "feature_artifact_refs_added": sorted(set(right.feature_artifact_refs) - set(left.feature_artifact_refs)),
            "feature_artifact_refs_removed": sorted(set(left.feature_artifact_refs) - set(right.feature_artifact_refs)),
            "label_pack_changed": (left.label_pack.key if left.label_pack else None) != (right.label_pack.key if right.label_pack else None),
            "label_pack_left": left.label_pack.key if left.label_pack else None,
            "label_pack_right": right.label_pack.key if right.label_pack else None,
            "schema_columns_added": sorted(schema_right - schema_left),
            "schema_columns_removed": sorted(schema_left - schema_right),
            "split_row_counts_left": left.split_row_counts,
            "split_row_counts_right": right.split_row_counts,
            "build_row_stats_left": left.build_row_stats,
            "build_row_stats_right": right.build_row_stats,
            "source_modes_left": left.source_modes,
            "source_modes_right": right.source_modes,
            "trust_score_left": left.trust_score_breakdown,
            "trust_score_right": right.trust_score_breakdown,
            "trust_decision_left": left.trust_decision_summary,
            "trust_decision_right": right.trust_decision_summary,
            "trust_rejection_reasons_added": sorted(set(right.trust_rejection_reasons) - set(left.trust_rejection_reasons)),
            "trust_rejection_reasons_removed": sorted(set(left.trust_rejection_reasons) - set(right.trust_rejection_reasons)),
            "trust_warning_reasons_added": sorted(set(right.trust_warning_reasons) - set(left.trust_warning_reasons)),
            "trust_warning_reasons_removed": sorted(set(left.trust_warning_reasons) - set(right.trust_warning_reasons)),
            "trust_check_status_counts_left": left.trust_check_status_counts,
            "trust_check_status_counts_right": right.trust_check_status_counts,
            "lineage_inputs_added": sorted(lineage_right - lineage_left),
            "lineage_inputs_removed": sorted(lineage_left - lineage_right),
            "state_artifact_refs_left": left.manifest.state_artifact_refs,
            "state_artifact_refs_right": right.manifest.state_artifact_refs,
        }
        return ArtifactDiff(left=left, right=right, diff=diff)

    def _resolve_state_input(
        self,
        *,
        spec: DatasetSpec,
        symbol: str,
        build_id: str,
        code_version_value: str,
        merge_config_ref_value: str,
    ) -> ResolvedStateInput:
        if spec.state_artifact_ref:
            manifest, manifest_path = self._resolve_manifest_with_path(spec.state_artifact_ref)
            if manifest.artifact_kind != "state":
                raise ValueError(f"state_artifact_ref must resolve to a state artifact, got {manifest.artifact_kind}")
            if spec.state_version_ref and manifest.logical_version != spec.state_version_ref:
                raise ValueError(
                    f"state_artifact_ref resolved to version '{manifest.logical_version}', "
                    f"expected '{spec.state_version_ref}'"
                )
            self._ensure_artifact_registered(manifest, manifest_path, detail="resolved upstream state artifact")
            state_df = self._load_artifact_frame(
                manifest,
                time_col="ts_utc",
                symbol=symbol,
                clock=spec.base_clock,
                date_from=spec.date_from,
                date_to=spec.date_to,
            )
            base_df = self._load_base_df_from_state_manifest(
                manifest=manifest,
                symbol=symbol,
                base_clock=spec.base_clock,
                date_from=spec.date_from,
                date_to=spec.date_to,
            )
            base_df = self._augment_base_df_with_state_controls(
                base_df=base_df,
                state_df=state_df,
                base_clock=spec.base_clock,
            )
            return ResolvedStateInput(
                artifact_id=manifest.artifact_id,
                manifest=manifest,
                manifest_path=manifest_path,
                state_df=state_df,
                base_df=base_df,
                source_mode="artifact_ref",
            )

        state_result = None
        materialization_error: Exception | None = None
        if self._state is not None:
            try:
                state_result = self._state.materialize_state(
                    symbol=symbol,
                    clock=spec.base_clock,
                    state_version_ref=spec.state_version_ref or "",
                    date_from=spec.date_from,
                    date_to=spec.date_to,
                    build_id=build_id,
                    dataset_spec_ref=spec.key,
                    code_version=code_version_value,
                    merge_config_ref=merge_config_ref_value,
                )
            except Exception as exc:
                materialization_error = exc
        else:
            materialization_error = self._state_boot_error

        if state_result is None:
            recovered = self._recover_state_input_from_storage(
                spec=spec,
                symbol=symbol,
                build_id=build_id,
                code_version_value=code_version_value,
                merge_config_ref_value=merge_config_ref_value,
                materialization_error=materialization_error,
            )
            if recovered is not None:
                return recovered

            if materialization_error is None:
                raise FileNotFoundError(
                    f"No compiler-visible state artifact was available for symbol={symbol} "
                    f"clock={spec.base_clock} range={spec.date_from.isoformat()}..{spec.date_to.isoformat()} "
                    f"and state version '{spec.state_version_ref}'."
                )
            raise RuntimeError(
                "Unable to materialize state through the public state boundary and no reusable "
                f"state parquet payload was found for compiler recovery: {materialization_error}"
            ) from materialization_error

        return ResolvedStateInput(
            artifact_id=getattr(state_result, "artifact_id"),
            manifest=getattr(state_result, "manifest"),
            manifest_path=getattr(state_result, "manifest_path"),
            state_df=getattr(state_result, "state_df"),
            base_df=self._augment_base_df_with_state_controls(
                base_df=getattr(state_result, "base_df"),
                state_df=getattr(state_result, "state_df"),
                base_clock=spec.base_clock,
            ),
            source_mode="materialized",
        )

    def _recover_state_input_from_storage(
        self,
        *,
        spec: DatasetSpec,
        symbol: str,
        build_id: str,
        code_version_value: str,
        merge_config_ref_value: str,
        materialization_error: Exception | None,
    ) -> ResolvedStateInput | None:
        state_version_ref = spec.state_version_ref or ""
        if not state_version_ref:
            return None

        state_root = self._paths.state_root(symbol, spec.base_clock, state_version_ref)
        state_df = self._store.read_dir(state_root)
        if state_df.is_empty():
            return None

        state_dedup_keys = [column for column in ["symbol", "clock", "ts_utc", "ts_msc"] if column in state_df.columns]
        if state_dedup_keys:
            state_df = state_df.unique(subset=state_dedup_keys, keep="last")
        state_df = _filter_frame_to_range(
            state_df,
            time_col="ts_utc",
            symbol=symbol,
            clock=spec.base_clock,
            date_from=spec.date_from,
            date_to=spec.date_to,
        )
        if state_df.is_empty():
            return None

        input_partition_refs = self._collect_built_bar_partition_refs(
            symbol=symbol,
            clock=spec.base_clock,
            date_from=spec.date_from,
            date_to=spec.date_to,
        )
        if not input_partition_refs:
            return None

        base_df = self._load_base_df_from_partition_refs(
            partition_refs=input_partition_refs,
            symbol=symbol,
            base_clock=spec.base_clock,
            date_from=spec.date_from,
            date_to=spec.date_to,
        )
        if base_df.is_empty():
            return None
        base_df = self._augment_base_df_with_state_controls(
            base_df=base_df,
            state_df=state_df,
            base_clock=spec.base_clock,
        )

        created_at = dt.datetime.now(dt.timezone.utc)
        logical_name = f"{symbol}.{spec.base_clock}"
        content_hash = compute_content_hash(
            {
                "artifact_kind": "state",
                "logical_name": logical_name,
                "logical_version": state_version_ref,
                "symbol": symbol,
                "clock": spec.base_clock,
                "rows": len(state_df),
                "columns": state_df.columns,
                "time_range_start": str(state_df["ts_utc"].min()) if not state_df.is_empty() else "",
                "time_range_end": str(state_df["ts_utc"].max()) if not state_df.is_empty() else "",
                "input_partition_refs": input_partition_refs,
            }
        )
        artifact_id = build_stage_artifact_id("state", logical_name, created_at, content_hash)
        manifest = LineageManifest(
            manifest_id=build_stage_manifest_id("state", logical_name, created_at, content_hash),
            artifact_id=artifact_id,
            artifact_kind="state",
            logical_name=logical_name,
            logical_version=state_version_ref,
            artifact_uri=str(state_root),
            content_hash=content_hash,
            build_id=build_id,
            created_at=created_at,
            status="accepted",
            dataset_spec_ref=spec.key,
            state_artifact_refs=[],
            feature_spec_refs=[],
            label_pack_ref=None,
            truth_report_ref=None,
            code_version=code_version_value,
            merge_config_ref=merge_config_ref_value,
            input_partition_refs=input_partition_refs,
            parent_artifact_refs=[],
            metadata={
                "row_count": len(state_df),
                "column_count": len(state_df.columns),
                "time_range_start": str(state_df["ts_utc"].min()) if not state_df.is_empty() else "",
                "time_range_end": str(state_df["ts_utc"].max()) if not state_df.is_empty() else "",
                "symbol": symbol,
                "clock": spec.base_clock,
                "recovered_via_compiler_state_storage_fallback": True,
                "state_materialization_error": str(materialization_error) if materialization_error is not None else "",
            },
        )
        manifest_path = write_manifest_sidecar(manifest, self._paths)
        self._ensure_artifact_registered(
            manifest,
            manifest_path,
            detail="recovered compiler-visible state artifact after public state materialization validation failure",
        )
        return ResolvedStateInput(
            artifact_id=artifact_id,
            manifest=manifest,
            manifest_path=manifest_path,
            state_df=state_df,
            base_df=base_df,
            source_mode="materialized",
        )

    def _collect_built_bar_partition_refs(
        self,
        *,
        symbol: str,
        clock: str,
        date_from: dt.date,
        date_to: dt.date,
    ) -> list[str]:
        refs: list[str] = []
        current = date_from
        while current <= date_to:
            partition_dir = self._paths.built_bars_dir(symbol, clock, current)
            if partition_dir.exists():
                refs.append(str(partition_dir))
            current += dt.timedelta(days=1)
        return refs

    def _load_base_df_from_partition_refs(
        self,
        *,
        partition_refs: list[str],
        symbol: str,
        base_clock: str,
        date_from: dt.date,
        date_to: dt.date,
    ) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for ref in partition_refs:
            frame = self._store.read_dir(Path(ref))
            if frame.is_empty():
                continue
            frames.append(
                _filter_frame_to_range(
                    frame,
                    time_col="time_utc",
                    symbol=symbol,
                    clock=base_clock,
                    date_from=date_from,
                    date_to=date_to,
                )
            )
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed").sort("time_utc")

    def _augment_base_df_with_state_controls(
        self,
        *,
        base_df: pl.DataFrame,
        state_df: pl.DataFrame,
        base_clock: str,
    ) -> pl.DataFrame:
        if base_df.is_empty() or state_df.is_empty():
            return base_df

        state_controls_df = state_df
        if "_filled" not in state_controls_df.columns and "trust_flags" in state_controls_df.columns:
            state_controls_df = state_controls_df.with_columns(
                pl.col("trust_flags").map_elements(
                    lambda flags: isinstance(flags, list) and "filled_gap" in flags,
                    return_dtype=pl.Boolean,
                ).alias("_filled")
            )

        control_columns = [
            column for column in STATE_CONTROL_COLUMNS
            if column in state_controls_df.columns and column not in base_df.columns
        ]
        if not control_columns:
            return base_df

        state_controls = state_controls_df.select(
            [
                pl.col("symbol"),
                pl.lit(base_clock).alias("timeframe"),
                pl.col("ts_utc").alias("time_utc"),
                *[pl.col(column) for column in control_columns],
            ]
        )
        state_controls = state_controls.unique(subset=DATASET_JOIN_KEYS, keep="last")
        return base_df.join(state_controls, on=DATASET_JOIN_KEYS, how="left")

    def _resolve_feature_artifact_sources(
        self,
        *,
        spec: DatasetSpec,
        symbol: str,
    ) -> dict[str, ResolvedFeatureSource]:
        if not spec.feature_artifact_refs:
            return {}

        spec_index = feature_spec_index(catalog=self._catalog)
        resolved: dict[str, ResolvedFeatureSource] = {}
        for ref in spec.feature_artifact_refs:
            try:
                manifest, manifest_path = self._resolve_manifest_with_path(ref)
            except KeyError as exc:
                raise KeyError(
                    "DatasetSpec feature_artifact_refs entry "
                    f"'{ref}' could not be resolved in the compiler catalog. "
                    "If this dataset should materialize stable features from selectors, "
                    "remove the stale feature_artifact_refs entry and rely on feature_selectors."
                ) from exc
            if manifest.artifact_kind != "feature_view":
                raise ValueError(f"feature_artifact_refs must resolve to feature_view artifacts, got {manifest.artifact_kind}")
            self._ensure_artifact_registered(manifest, manifest_path, detail="resolved upstream feature artifact")

            if len(manifest.feature_spec_refs) != 1:
                raise ValueError(
                    f"Feature artifact '{manifest.artifact_id}' must declare exactly one feature_spec_ref for compiler resolution"
                )
            spec_key = manifest.feature_spec_refs[0]
            feature_spec = spec_index.get(spec_key) or self._catalog.get_feature_spec(spec_key)
            if feature_spec is None:
                raise KeyError(
                    f"Feature artifact '{manifest.artifact_id}' references spec '{spec_key}', "
                    "but that feature spec is not registered in the compiler catalog"
                )
            self._catalog.register_feature_specs([feature_spec])

            frame = self._load_artifact_frame(
                manifest,
                time_col="time_utc",
                symbol=symbol,
                clock=spec.base_clock,
                date_from=spec.date_from,
                date_to=spec.date_to,
            )
            resolved[feature_spec.key] = ResolvedFeatureSource(
                spec=feature_spec,
                artifact_id=manifest.artifact_id,
                manifest=manifest,
                manifest_path=manifest_path,
                frame=frame.select([column for column in [*DATASET_JOIN_KEYS, *feature_spec.output_columns] if column in frame.columns]),
                source_mode="artifact_ref",
            )
        return resolved

    def _build_feature_sources(
        self,
        *,
        spec: DatasetSpec,
        symbol: str,
        feature_specs: list[FeatureSpec],
        provided_feature_sources: dict[str, ResolvedFeatureSource],
        state_input: ResolvedStateInput,
        build_id: str,
        code_version_value: str,
    ) -> list[ResolvedFeatureSource]:
        materialized_specs = [feature for feature in feature_specs if feature.key not in provided_feature_sources]
        materialized_sources: list[ResolvedFeatureSource] = []

        if materialized_specs:
            feature_result = self._features.materialize_features(
                symbol=symbol,
                base_clock=spec.base_clock,
                date_from=spec.date_from,
                date_to=spec.date_to,
                feature_specs=materialized_specs,
                base_df=state_input.base_df,
                state_artifact_id=state_input.artifact_id,
                build_id=build_id,
                dataset_spec_ref=spec.key,
                code_version=code_version_value,
            )
            for artifact in getattr(feature_result, "artifacts", []):
                materialized_sources.append(
                    ResolvedFeatureSource(
                        spec=getattr(artifact, "spec"),
                        artifact_id=getattr(artifact, "artifact_id"),
                        manifest=getattr(artifact, "manifest"),
                        manifest_path=getattr(artifact, "manifest_path"),
                        frame=getattr(artifact, "frame"),
                        source_mode="materialized",
                    )
                )

        ordered_sources: list[ResolvedFeatureSource] = []
        materialized_by_key = {source.spec.key: source for source in materialized_sources}
        for feature_spec in feature_specs:
            source = provided_feature_sources.get(feature_spec.key) or materialized_by_key.get(feature_spec.key)
            if source is None:
                raise KeyError(
                    f"Feature '{feature_spec.key}' was requested but neither an artifact ref nor a materialized feature artifact was available"
                )
            ordered_sources.append(source)
        return ordered_sources

    def _resolve_label_input(
        self,
        *,
        spec: DatasetSpec,
        symbol: str,
        state_input: ResolvedStateInput,
        build_id: str,
        code_version_value: str,
    ) -> ResolvedLabelInput:
        if spec.label_artifact_ref:
            manifest, manifest_path = self._resolve_manifest_with_path(spec.label_artifact_ref)
            if manifest.artifact_kind != "label_view":
                raise ValueError(f"label_artifact_ref must resolve to a label_view artifact, got {manifest.artifact_kind}")
            self._ensure_artifact_registered(manifest, manifest_path, detail="resolved upstream label artifact")

            label_pack_key = manifest.label_pack_ref or spec.label_pack_ref or ""
            label_pack = resolve_label_pack(label_pack_key, catalog=self._catalog)
            if spec.label_pack_ref and label_pack.key != spec.label_pack_ref:
                raise ValueError(
                    f"label_artifact_ref resolved to label pack '{label_pack.key}', expected '{spec.label_pack_ref}'"
                )
            self._catalog.register_label_packs([label_pack])
            label_df = self._load_artifact_frame(
                manifest,
                time_col="time_utc",
                symbol=symbol,
                clock=spec.base_clock,
                date_from=spec.date_from,
                date_to=spec.date_to,
            )
            return ResolvedLabelInput(
                label_pack=label_pack,
                artifact_id=manifest.artifact_id,
                manifest=manifest,
                manifest_path=manifest_path,
                label_df=label_df.select([column for column in [*DATASET_JOIN_KEYS, *label_pack.output_columns] if column in label_df.columns]),
                source_mode="artifact_ref",
            )

        label_pack = resolve_label_pack(spec.label_pack_ref or "", catalog=self._catalog)
        self._catalog.register_label_packs([label_pack])
        label_result = self._labels.materialize_labels(
            symbol=symbol,
            base_clock=spec.base_clock,
            date_from=spec.date_from,
            date_to=spec.date_to,
            label_pack=label_pack,
            base_df=state_input.base_df,
            state_artifact_id=state_input.artifact_id,
            build_id=build_id,
            dataset_spec_ref=spec.key,
            code_version=code_version_value,
        )
        return ResolvedLabelInput(
            label_pack=label_pack,
            artifact_id=getattr(label_result, "artifact_id"),
            manifest=getattr(label_result, "manifest"),
            manifest_path=getattr(label_result, "manifest_path"),
            label_df=getattr(label_result, "label_df"),
            source_mode="materialized",
        )

    def _combine_feature_frames(
        self,
        base_df: pl.DataFrame,
        feature_sources: list[ResolvedFeatureSource],
    ) -> pl.DataFrame:
        combined = base_df.select(DATASET_JOIN_KEYS).sort("time_utc")
        for source in feature_sources:
            source_columns = [column for column in source.frame.columns if column not in DATASET_JOIN_KEYS]
            if source_columns:
                combined = combined.join(
                    source.frame.select([*DATASET_JOIN_KEYS, *source_columns]),
                    on=DATASET_JOIN_KEYS,
                    how="left",
                )
        return combined

    def _compile_dataset_frame(
        self,
        *,
        base_df: pl.DataFrame,
        feature_df: pl.DataFrame,
        label_df: pl.DataFrame,
        spec: DatasetSpec,
        feature_specs: list[FeatureSpec],
        label_pack: LabelPack,
    ) -> tuple[pl.DataFrame, dict[str, int]]:
        df = base_df.sort("time_utc")

        feature_cols = [col for col in feature_df.columns if col not in DATASET_JOIN_KEYS]
        if feature_cols:
            df = df.join(feature_df.select([*DATASET_JOIN_KEYS, *feature_cols]), on=DATASET_JOIN_KEYS, how="left")

        label_cols = [col for col in label_df.columns if col not in DATASET_JOIN_KEYS]
        if label_cols:
            df = df.join(label_df.select([*DATASET_JOIN_KEYS, *label_cols]), on=DATASET_JOIN_KEYS, how="left")

        build_row_stats: dict[str, int] = {
            "rows_after_join": len(df),
            "feature_drop_row_rows_removed": 0,
            "label_purge_rows_applied": 0,
            "filled_rows_removed": 0,
            "zero_source_rows_removed": 0,
            "rows_after_filters": len(df),
            "cleaning_rows_removed": 0,
            "rows_after_cleaning": len(df),
        }

        drop_row_columns = sorted(
            {
                column
                for feature in feature_specs
                if feature.missingness_policy == "drop_row"
                for column in feature.output_columns
                if column in df.columns
            }
        )
        if drop_row_columns:
            before = len(df)
            df = df.drop_nulls(drop_row_columns)
            build_row_stats["feature_drop_row_rows_removed"] = before - len(df)

        if len(df) <= label_pack.purge_rows:
            build_row_stats["label_purge_rows_applied"] = len(df)
            build_row_stats["rows_after_filters"] = 0
            build_row_stats["cleaning_rows_removed"] = 0
            build_row_stats["rows_after_cleaning"] = 0
            return pl.DataFrame(), build_row_stats

        build_row_stats["label_purge_rows_applied"] = label_pack.purge_rows
        df = df.head(len(df) - label_pack.purge_rows)
        df, filter_stats = self._apply_dataset_filters(df, spec.filters)
        build_row_stats["filled_rows_removed"] = filter_stats["filled_rows_removed"]
        build_row_stats["zero_source_rows_removed"] = filter_stats["zero_source_rows_removed"]

        if "source_count" in df.columns:
            before = len(df)
            df = df.filter(pl.col("source_count") > 0)
            build_row_stats["zero_source_rows_removed"] += before - len(df)

        build_row_stats["rows_after_filters"] = len(df)
        cleaned, cleaning_stats = clean_dataset(
            df,
            max_null_pct=100.0,
            drop_constant=False,
        )
        build_row_stats["cleaning_rows_removed"] = int(cleaning_stats.get("rows_removed", 0))
        build_row_stats["rows_after_cleaning"] = len(cleaned)
        return cleaned.sort("time_utc"), build_row_stats

    def _apply_dataset_filters(self, dataset_df: pl.DataFrame, filters: list[str]) -> tuple[pl.DataFrame, dict[str, int]]:
        df = dataset_df
        filter_stats = {"filled_rows_removed": 0, "zero_source_rows_removed": 0}
        for raw_filter in filters:
            current = raw_filter.strip().lower()
            if not current:
                continue
            if current == "exclude:filled_rows":
                if "_filled" in df.columns:
                    before = len(df)
                    df = df.filter(~pl.col("_filled"))
                    filter_stats["filled_rows_removed"] += before - len(df)
                elif "trust_flags" in df.columns:
                    before = len(df)
                    df = df.filter(
                        ~pl.col("trust_flags").map_elements(
                            lambda flags: isinstance(flags, list) and "filled_gap" in flags,
                            return_dtype=pl.Boolean,
                        )
                    )
                    filter_stats["filled_rows_removed"] += before - len(df)
                else:
                    raise ValueError("DatasetSpec filter 'exclude:filled_rows' requires '_filled' or 'trust_flags'")
            elif current == "exclude:zero_source_rows" and "source_count" in df.columns:
                before = len(df)
                df = df.filter(pl.col("source_count") > 0)
                filter_stats["zero_source_rows_removed"] += before - len(df)
            else:
                raise ValueError(f"Unsupported DatasetSpec filter '{raw_filter}'")
        return df, filter_stats

    def _select_dataset_columns(self, dataset_df: pl.DataFrame, feature_specs: list[FeatureSpec], label_pack: LabelPack) -> pl.DataFrame:
        df = dataset_df
        for feature in feature_specs:
            for column in feature.output_columns:
                if column not in df.columns:
                    if feature.missingness_policy == "allow":
                        df = df.with_columns(pl.lit(None).alias(column))
                    elif feature.missingness_policy == "impute_zero":
                        df = df.with_columns(pl.lit(0.0).alias(column))
                    elif feature.missingness_policy == "impute_forward":
                        df = df.with_columns(pl.lit(None).alias(column))
                        df = df.with_columns(pl.col(column).forward_fill().backward_fill())

        selected_columns: list[str] = []
        for column in MANDATORY_DATASET_COLUMNS:
            if column in df.columns and column not in selected_columns:
                selected_columns.append(column)
        for feature in feature_specs:
            for column in feature.output_columns:
                if column in df.columns and column not in selected_columns:
                    selected_columns.append(column)
        for column in label_pack.output_columns:
            if column in df.columns and column not in selected_columns:
                selected_columns.append(column)

        return df.select(selected_columns)

    def _split_dataset(self, dataset_df: pl.DataFrame, spec: DatasetSpec) -> dict[str, pl.DataFrame]:
        if spec.split_policy == "temporal_holdout":
            n = len(dataset_df)
            train_end = int(n * spec.train_ratio)
            val_start = train_end + spec.embargo_rows
            val_end = int(n * (spec.train_ratio + spec.val_ratio))
            test_start = val_end + spec.embargo_rows

            return {
                "train": dataset_df.slice(0, train_end),
                "val": dataset_df.slice(val_start, max(0, val_end - val_start)) if val_start < n else pl.DataFrame(),
                "test": dataset_df.slice(test_start, max(0, n - test_start)) if test_start < n else pl.DataFrame(),
            }

        split_pairs = _walk_forward_splits(
            dataset_df,
            n_splits=spec.n_walk_forward_splits or 5,
            train_pct=spec.train_ratio,
            embargo_rows=spec.embargo_rows,
        )
        split_frames: dict[str, pl.DataFrame] = {}
        for idx, (train_df, test_df) in enumerate(split_pairs, start=1):
            split_frames[f"train_fold_{idx}"] = train_df
            split_frames[f"test_fold_{idx}"] = test_df
        return split_frames

    def _write_compiled_artifact(self, spec: DatasetSpec, artifact_id: str, split_frames: dict[str, pl.DataFrame]) -> None:
        for split_name, split_df in split_frames.items():
            if split_df.is_empty():
                continue
            path = self._paths.compiler_dataset_file(spec.dataset_name, artifact_id, split_name)
            self._store.write(split_df, path)

    def _write_truth_report(self, report: TrustReport) -> Path:
        path = self._paths.truth_report_file(report.artifact_id, report.report_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True, default=str), encoding="utf-8")
        return path

    def _persist_manifest(self, manifest: LineageManifest, *, detail: str = "") -> Path:
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

    def _ensure_artifact_registered(self, manifest: LineageManifest, manifest_path: Path, *, detail: str) -> None:
        if self._catalog.get_artifact(manifest.artifact_id) is None:
            self._catalog.register_artifact(manifest, str(manifest_path), detail=detail)

    def _load_base_df_from_state_manifest(
        self,
        *,
        manifest: LineageManifest,
        symbol: str,
        base_clock: str,
        date_from: dt.date,
        date_to: dt.date,
    ) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for ref in manifest.input_partition_refs:
            if not _looks_like_bars_ref(ref):
                continue
            frame = self._store.read_dir(Path(ref))
            if frame.is_empty():
                continue
            frames.append(
                _filter_frame_to_range(
                    frame,
                    time_col="time_utc",
                    symbol=symbol,
                    clock=base_clock,
                    date_from=date_from,
                    date_to=date_to,
                )
            )
        if not frames:
            raise FileNotFoundError(
                f"State artifact '{manifest.artifact_id}' did not expose any built-bar input partitions for compiler reuse"
            )
        return pl.concat(frames, how="diagonal_relaxed").sort("time_utc")

    def _load_artifact_frame(
        self,
        manifest: LineageManifest,
        *,
        time_col: str,
        symbol: str,
        clock: str,
        date_from: dt.date,
        date_to: dt.date,
    ) -> pl.DataFrame:
        frame = self._store.read_dir(Path(manifest.artifact_uri))
        if frame.is_empty():
            raise FileNotFoundError(f"Artifact '{manifest.artifact_id}' has no readable parquet payload at {manifest.artifact_uri}")
        dedup_keys = [column for column in ["symbol", "timeframe", "clock", time_col] if column in frame.columns]
        if dedup_keys:
            frame = frame.unique(subset=dedup_keys, keep="last")
        filtered = _filter_frame_to_range(
            frame,
            time_col=time_col,
            symbol=symbol,
            clock=clock,
            date_from=date_from,
            date_to=date_to,
        )
        if filtered.is_empty():
            raise FileNotFoundError(
                f"Artifact '{manifest.artifact_id}' did not contain rows for "
                f"symbol={symbol} clock={clock} range={date_from.isoformat()}..{date_to.isoformat()}"
            )
        return filtered

    @staticmethod
    def _dataset_content_hash(
        *,
        spec: DatasetSpec,
        dataset_df: pl.DataFrame,
        split_frames: dict[str, pl.DataFrame],
        state_artifact_id: str,
        feature_artifact_ids: list[str],
        label_artifact_id: str,
    ) -> str:
        return compute_content_hash(
            {
                "dataset_spec": spec.model_dump(mode="json"),
                "columns": dataset_df.columns,
                "rows": len(dataset_df),
                "split_rows": {name: len(frame) for name, frame in split_frames.items()},
                "time_range_start": str(dataset_df["time_utc"].min()) if not dataset_df.is_empty() else "",
                "time_range_end": str(dataset_df["time_utc"].max()) if not dataset_df.is_empty() else "",
                "state_artifact_id": state_artifact_id,
                "feature_artifact_ids": feature_artifact_ids,
                "label_artifact_id": label_artifact_id,
            }
        )

    def _assert_publishable(
        self,
        manifest: LineageManifest,
        manifest_path: Path,
        trust_report: TrustReport,
        expected_content_hash: str,
    ) -> None:
        if not trust_report.accepted_for_publication:
            raise RuntimeError("Artifact cannot be published because the trust report is not accepted")
        if not manifest.truth_report_ref:
            raise RuntimeError("Artifact cannot be published without a linked TrustReport reference")
        if manifest.content_hash != expected_content_hash:
            raise RuntimeError("Artifact cannot be published because the manifest content hash does not match the compiler payload hash")

        manifest_from_disk = read_manifest_sidecar(manifest_path)
        if manifest_from_disk.content_hash != expected_content_hash:
            raise RuntimeError("Artifact cannot be published because the persisted manifest content hash is inconsistent")

        artifact = self._catalog.get_artifact(manifest.artifact_id)
        if artifact is None:
            raise RuntimeError("Artifact cannot be published because the compiler catalog entry is missing")
        if artifact.content_hash != expected_content_hash:
            raise RuntimeError("Artifact cannot be published because the catalog content hash is inconsistent")

    @staticmethod
    def format_inspect_result(result: ArtifactInspection) -> str:
        trust = result.trust_report
        lines = [
            f"artifact_id: {result.artifact.artifact_id}",
            f"logical: {result.artifact.logical_name}@{result.artifact.logical_version}",
            f"status: {result.artifact.status}",
            f"manifest_path: {result.manifest_path}",
            f"artifact_uri: {result.artifact.artifact_uri}",
            f"time_range: {result.time_range['start']} -> {result.time_range['end']}",
            f"split_rows: {json.dumps(result.split_row_counts, sort_keys=True)}",
            f"feature_families: {', '.join(result.feature_families) if result.feature_families else '-'}",
            f"feature_selectors: {json.dumps(result.requested_feature_selectors)}",
            f"feature_artifact_refs: {json.dumps(sorted(result.feature_artifact_refs))}",
            f"source_modes: {json.dumps(result.source_modes, sort_keys=True)}",
            f"build_row_stats: {json.dumps(result.build_row_stats, sort_keys=True)}",
            f"label_pack: {result.label_pack.key if result.label_pack else '-'}",
            f"dataset_spec_ref: {result.manifest.dataset_spec_ref or '-'}",
            f"lineage_inputs: {len(result.lineage_refs['artifact_inputs'])}",
        ]
        if trust is not None:
            lines.append(
                "trust_scores: "
                + json.dumps(
                    {
                        "total": trust.trust_score_total,
                        "coverage": trust.coverage_score,
                        "leakage": trust.leakage_score,
                        "feature_quality": trust.feature_quality_score,
                        "label_quality": trust.label_quality_score,
                        "source_quality": trust.source_quality_score,
                        "lineage": trust.lineage_score,
                    },
                    sort_keys=True,
                )
            )
        return "\n".join(lines)

    @staticmethod
    def format_diff_result(result: ArtifactDiff) -> str:
        lines = ["Artifact diff"]
        for key, value in result.diff.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)


def compile_dataset_spec(spec_path: str | Path, *, publish: bool = True) -> BuildResult:
    compiler, catalog = DatasetCompiler.from_config_path(_resolve_pipeline_config_path_from_spec(spec_path))
    try:
        return compiler.compile_dataset(Path(spec_path), publish=publish)
    finally:
        catalog.close()


def inspect_artifact(ref: str) -> ArtifactInspection:
    compiler, catalog = _build_compiler_for_ref(ref)
    try:
        return compiler.inspect_dataset(ref)
    finally:
        catalog.close()


def diff_artifacts(left_ref: str, right_ref: str) -> ArtifactDiff:
    compiler, catalog = _build_compiler_for_ref(left_ref)
    try:
        return compiler.diff_datasets(left_ref, right_ref)
    finally:
        catalog.close()


def _build_compiler_for_ref(ref: str) -> tuple[DatasetCompiler, CatalogDB]:
    config_path = _resolve_pipeline_config_path_from_ref(ref)
    return DatasetCompiler.from_config_path(config_path)


def _resolve_pipeline_config_path_from_spec(spec_path: str | Path) -> Path:
    spec = Path(spec_path)
    if spec.exists():
        candidate = _find_repo_config(spec)
        if candidate is not None:
            return candidate
    return _resolve_pipeline_config_path()


def _resolve_pipeline_config_path_from_ref(ref: str) -> Path:
    maybe_path = Path(ref)
    if maybe_path.exists():
        inferred_root = _infer_storage_root_from_manifest_path(maybe_path)
        if inferred_root is not None:
            candidate = _resolve_pipeline_config_path()
            if candidate.exists():
                return candidate
    return _resolve_pipeline_config_path()


def _resolve_pipeline_config_path(config_path: str | Path | None = None) -> Path:
    if config_path is not None:
        resolved = Path(config_path)
        if resolved.exists():
            return resolved.resolve()
        raise FileNotFoundError(f"Config file not found: {resolved}")

    candidate = _find_repo_config(Path.cwd())
    if candidate is not None:
        return candidate
    raise FileNotFoundError(f"Config file not found: {DEFAULT_CONFIG_RELATIVE_PATH}")


def _find_repo_config(anchor: Path) -> Path | None:
    current = anchor.resolve()
    if current.is_file():
        current = current.parent
    for parent in [current, *current.parents]:
        candidate = parent / DEFAULT_CONFIG_RELATIVE_PATH
        if candidate.exists():
            return candidate
    return None


def _infer_storage_root_from_manifest_path(path: Path) -> Path | None:
    resolved = path.resolve()
    for parent in [resolved.parent, *resolved.parents]:
        if parent.name == "manifests":
            return parent.parent
    return None


def _walk_forward_splits(
    df: pl.DataFrame,
    *,
    n_splits: int,
    train_pct: float,
    gap_pct: float = 0.02,
    embargo_rows: int,
) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    n = len(df)
    splits: list[tuple[pl.DataFrame, pl.DataFrame]] = []
    step = int(n * (1.0 - train_pct) / n_splits)
    gap = max(int(n * gap_pct), embargo_rows)

    for idx in range(n_splits):
        test_start = int(n * train_pct) + idx * step
        test_end = min(test_start + step, n)
        train_end = test_start - gap

        if train_end < 1 or test_start >= n:
            continue

        train_df = df.slice(0, train_end)
        test_df = df.slice(test_start, test_end - test_start)
        if train_df.is_empty() or test_df.is_empty():
            continue
        splits.append((train_df, test_df))

    return splits


def _filter_frame_to_range(
    frame: pl.DataFrame,
    *,
    time_col: str,
    symbol: str,
    clock: str,
    date_from: dt.date,
    date_to: dt.date,
) -> pl.DataFrame:
    result = frame
    if "symbol" in result.columns:
        result = result.filter(pl.col("symbol") == symbol)
    if "timeframe" in result.columns:
        result = result.filter(pl.col("timeframe") == clock)
    elif "clock" in result.columns:
        result = result.filter(pl.col("clock") == clock)
    if time_col in result.columns:
        result = result.filter(pl.col(time_col).dt.date().is_between(date_from, date_to))
        result = result.sort(time_col)
    return result


def _looks_like_bars_ref(ref: str) -> bool:
    normalized = ref.replace("\\", "/")
    return "/bars/" in normalized or normalized.endswith("/bars")


def _has_multiple_source_modes(feature_sources: list[ResolvedFeatureSource]) -> bool:
    return len({source.source_mode for source in feature_sources}) > 1


def _single_feature_source_mode(feature_sources: list[ResolvedFeatureSource]) -> str:
    if not feature_sources:
        return "none"
    return sorted({source.source_mode for source in feature_sources})[0]
