"""Feature view materialization services."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from mt5pipe.bars.builder import timeframe_to_seconds
from mt5pipe.contracts.dataset import DATASET_JOIN_KEYS
from mt5pipe.features.artifacts import FeatureArtifactRef
from mt5pipe.catalog.sqlite import CatalogDB
from mt5pipe.compiler.manifest import (
    build_stage_artifact_id,
    build_stage_manifest_id,
    compute_content_hash,
    write_manifest_sidecar,
)
from mt5pipe.compiler.models import LineageManifest
from mt5pipe.config.models import DatasetConfig
from mt5pipe.features.context import add_lagged_bar_features
from mt5pipe.features.disagreement import add_disagreement_features
from mt5pipe.features.entropy import add_entropy_features
from mt5pipe.features.event_shape import add_event_shape_features
from mt5pipe.features.multiscale import add_multiscale_features
from mt5pipe.features.quality import add_spread_quality_features
from mt5pipe.features.registry.models import FeatureSpec
from mt5pipe.features.session import add_session_features
from mt5pipe.features.time import add_time_features
from mt5pipe.quality.cleaning import validate_bars
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths

FEATURE_JOIN_KEYS = DATASET_JOIN_KEYS


@dataclass
class MaterializedFeatureArtifact:
    spec: FeatureSpec
    artifact_id: str
    artifact_ref: FeatureArtifactRef
    manifest: LineageManifest
    manifest_path: Path
    frame: pl.DataFrame


@dataclass
class FeatureMaterializationResult:
    artifacts: list[MaterializedFeatureArtifact]
    combined_df: pl.DataFrame


class FeatureService:
    """Materialize registered feature views as first-class compiler artifacts."""

    def __init__(
        self,
        paths: StoragePaths,
        store: ParquetStore,
        catalog: CatalogDB,
        dataset_cfg: DatasetConfig,
    ) -> None:
        self._paths = paths
        self._store = store
        self._catalog = catalog
        self._dataset_cfg = dataset_cfg

    def materialize_features(
        self,
        *,
        symbol: str,
        base_clock: str,
        date_from: dt.date,
        date_to: dt.date,
        feature_specs: list[FeatureSpec],
        base_df: pl.DataFrame,
        state_artifact_id: str,
        build_id: str,
        dataset_spec_ref: str,
        code_version: str,
    ) -> FeatureMaterializationResult:
        artifacts: list[MaterializedFeatureArtifact] = []
        combined_df = base_df.select(FEATURE_JOIN_KEYS)

        for spec in feature_specs:
            feature_df = self._build_feature_view(symbol, spec, base_df, date_from, date_to)
            input_partition_refs = self._collect_input_partition_refs(symbol, base_clock, date_from, date_to, spec)
            created_at = dt.datetime.now(dt.timezone.utc)
            content_hash = compute_content_hash({
                "artifact_kind": "feature_view",
                "feature_spec": spec.model_dump(mode="json"),
                "rows": len(feature_df),
                "columns": feature_df.columns,
                "time_range_start": str(feature_df["time_utc"].min()) if not feature_df.is_empty() else "",
                "time_range_end": str(feature_df["time_utc"].max()) if not feature_df.is_empty() else "",
                "state_artifact_id": state_artifact_id,
                "input_partition_refs": input_partition_refs,
            })
            artifact_id = build_stage_artifact_id("feature_view", spec.key, created_at, content_hash)
            manifest = LineageManifest(
                manifest_id=build_stage_manifest_id("feature_view", spec.key, created_at, content_hash),
                artifact_id=artifact_id,
                artifact_kind="feature_view",
                logical_name=spec.key,
                logical_version=spec.version,
                artifact_uri=str(
                    self._paths.feature_artifact_root(spec.key, spec.output_clock, artifact_id)
                ),
                content_hash=content_hash,
                build_id=build_id,
                created_at=created_at,
                status="accepted",
                dataset_spec_ref=dataset_spec_ref,
                state_artifact_refs=[state_artifact_id],
                feature_spec_refs=[spec.key],
                code_version=code_version,
                input_partition_refs=input_partition_refs,
                parent_artifact_refs=[state_artifact_id],
                metadata={
                    "family": spec.family,
                    "status": spec.status,
                    "ablation_group": spec.ablation_group or spec.family,
                    "family_tags": spec.tags,
                    "trainability_tags": spec.trainability_tags,
                    "row_count": len(feature_df),
                    "column_count": len(feature_df.columns),
                    "output_columns": spec.output_columns,
                    "trainability_diagnostics": _feature_trainability_diagnostics(feature_df, spec),
                    "time_range_start": str(feature_df["time_utc"].min()) if not feature_df.is_empty() else "",
                    "time_range_end": str(feature_df["time_utc"].max()) if not feature_df.is_empty() else "",
                },
            )

            self._write_feature_partitions(spec, artifact_id, feature_df)
            manifest_path = write_manifest_sidecar(manifest, self._paths)
            self._catalog.register_artifact(manifest, str(manifest_path))
            artifacts.append(MaterializedFeatureArtifact(
                spec=spec,
                artifact_id=artifact_id,
                artifact_ref=FeatureArtifactRef(
                    artifact_id=artifact_id,
                    feature_key=spec.key,
                    clock=spec.output_clock,
                ),
                manifest=manifest,
                manifest_path=manifest_path,
                frame=feature_df,
            ))
            combined_df = combined_df.join(feature_df, on=FEATURE_JOIN_KEYS, how="left")

        return FeatureMaterializationResult(artifacts=artifacts, combined_df=combined_df)

    def _build_feature_view(
        self,
        symbol: str,
        spec: FeatureSpec,
        base_df: pl.DataFrame,
        date_from: dt.date,
        date_to: dt.date,
    ) -> pl.DataFrame:
        working = base_df.clone()
        if spec.family == "htf_context":
            for htf in self._dataset_cfg.context_timeframes:
                htf_df = self._load_bars_range(symbol, htf, date_from, date_to)
                if htf_df.is_empty():
                    continue
                working = add_lagged_bar_features(
                    working,
                    validate_bars(htf_df),
                    htf,
                    bar_duration_seconds=timeframe_to_seconds(htf),
                )
        else:
            builder = _family_builder(spec.family)
            builder_kwargs: dict[str, object] = {}
            if spec.family == "event_shape":
                builder_kwargs["bar_duration_seconds"] = timeframe_to_seconds(base_df["timeframe"][0]) if "timeframe" in base_df.columns and len(base_df) > 0 else timeframe_to_seconds(spec.input_clock)
            working = builder(working, **builder_kwargs)

        working = self._apply_missingness_policy(working, spec)
        output_columns = [col for col in spec.output_columns if col in working.columns]
        return working.select([*FEATURE_JOIN_KEYS, *output_columns]).sort("time_utc")

    def _load_bars_range(
        self,
        symbol: str,
        timeframe: str,
        date_from: dt.date,
        date_to: dt.date,
    ) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        current = date_from
        while current <= date_to:
            day = self._store.read_dir(self._paths.built_bars_dir(symbol, timeframe, current))
            if not day.is_empty():
                frames.append(day)
            current += dt.timedelta(days=1)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed").sort("time_utc")

    def _apply_missingness_policy(self, df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
        working = df
        for column in spec.output_columns:
            if column in working.columns:
                continue
            if spec.missingness_policy == "allow":
                working = working.with_columns(pl.lit(None).alias(column))
            elif spec.missingness_policy == "impute_zero":
                working = working.with_columns(pl.lit(0.0).alias(column))
            elif spec.missingness_policy == "impute_forward":
                working = working.with_columns(pl.lit(None).alias(column))
                working = working.with_columns(pl.col(column).forward_fill())
            elif spec.missingness_policy == "drop_row":
                working = working.with_columns(pl.lit(None).alias(column))
            else:
                raise KeyError(f"Feature '{spec.key}' did not materialize required column '{column}'")

        if spec.missingness_policy == "drop_row":
            existing = [col for col in spec.output_columns if col in working.columns]
            if existing:
                working = working.drop_nulls(existing)
        return working

    def _collect_input_partition_refs(
        self,
        symbol: str,
        base_clock: str,
        date_from: dt.date,
        date_to: dt.date,
        spec: FeatureSpec,
    ) -> list[str]:
        refs: list[str] = []
        current = date_from
        while current <= date_to:
            base_dir = self._paths.built_bars_dir(symbol, base_clock, current)
            if base_dir.exists():
                refs.append(str(base_dir))
            if spec.family == "htf_context":
                for htf in self._dataset_cfg.context_timeframes:
                    htf_dir = self._paths.built_bars_dir(symbol, htf, current)
                    if htf_dir.exists():
                        refs.append(str(htf_dir))
            current += dt.timedelta(days=1)
        return sorted(set(refs))

    def _write_feature_partitions(self, spec: FeatureSpec, artifact_id: str, feature_df: pl.DataFrame) -> None:
        dated = feature_df.with_columns(pl.col("time_utc").dt.date().alias("_date"))
        for date_val in dated["_date"].unique().sort().to_list():
            day_df = dated.filter(pl.col("_date") == date_val).drop("_date")
            self._store.write(day_df, self._paths.feature_view_file(spec.key, spec.output_clock, date_val))
            self._store.write(
                day_df,
                self._paths.feature_artifact_file(spec.key, spec.output_clock, artifact_id, date_val),
            )


def _family_builder(family: str):
    builders = {
        "time": add_time_features,
        "session": add_session_features,
        "quality": add_spread_quality_features,
        "disagreement": add_disagreement_features,
        "event_shape": add_event_shape_features,
        "entropy": add_entropy_features,
        "multiscale": add_multiscale_features,
    }
    try:
        return builders[family]
    except KeyError as exc:
        raise NotImplementedError(f"Feature family '{family}' is not materialized in the feature service") from exc


def _feature_trainability_diagnostics(feature_df: pl.DataFrame, spec: FeatureSpec) -> dict[str, Any]:
    output_columns = [column for column in spec.output_columns if column in feature_df.columns]
    total_rows = int(feature_df.height)
    warmup_excluded_rows = min(max(spec.warmup_rows - 1, 0), total_rows)
    post_warmup = feature_df.slice(warmup_excluded_rows) if warmup_excluded_rows else feature_df
    post_warmup_rows = int(post_warmup.height)

    column_summaries: dict[str, dict[str, Any]] = {}
    constant_columns: list[str] = []
    low_variation_columns: list[str] = []
    null_heavy_columns: list[str] = []
    total_non_null = 0

    for column in output_columns:
        series = post_warmup[column]
        null_count = int(series.null_count())
        non_null_count = post_warmup_rows - null_count
        non_null_ratio = round(non_null_count / post_warmup_rows, 6) if post_warmup_rows else None
        distinct_non_null = int(series.drop_nulls().n_unique()) if non_null_count > 0 else 0

        is_constant = non_null_count > 0 and distinct_non_null == 1
        is_low_variation = non_null_count > 0 and distinct_non_null <= 3
        is_null_heavy = non_null_ratio is not None and non_null_ratio < 0.75

        if is_constant:
            constant_columns.append(column)
        if is_low_variation and not is_constant:
            low_variation_columns.append(column)
        if is_null_heavy:
            null_heavy_columns.append(column)

        total_non_null += non_null_count
        column_summaries[column] = {
            "null_count_post_warmup": null_count,
            "non_null_count_post_warmup": non_null_count,
            "non_null_ratio_post_warmup": non_null_ratio,
            "distinct_non_null_post_warmup": distinct_non_null,
            "constant_post_warmup": is_constant,
            "low_variation_post_warmup": is_low_variation,
        }

    complete_row_count = 0
    if output_columns and post_warmup_rows:
        complete_row_count = int(
            post_warmup.select(
                pl.all_horizontal([pl.col(column).is_not_null() for column in output_columns]).sum().alias("count")
            )["count"][0]
        )

    complete_row_ratio = round(complete_row_count / post_warmup_rows, 6) if post_warmup_rows else None
    family_non_null_ratio = (
        round(total_non_null / float(post_warmup_rows * len(output_columns)), 6)
        if post_warmup_rows and output_columns
        else None
    )

    warnings: list[str] = []
    if total_rows <= spec.warmup_rows:
        warnings.append("insufficient_rows_vs_warmup")
    if null_heavy_columns:
        warnings.append("post_warmup_null_heavy_columns")
    if constant_columns:
        warnings.append("constant_columns_present")
    if len(constant_columns) == len(output_columns) and output_columns:
        warnings.append("family_degenerate_all_constant")
    if low_variation_columns:
        warnings.append("low_variation_columns_present")
    if complete_row_ratio is not None and complete_row_ratio < 0.75:
        warnings.append("incomplete_rows_post_warmup")

    return {
        "feature_key": spec.key,
        "family": spec.family,
        "status": spec.status,
        "ablation_group": spec.ablation_group or spec.family,
        "family_tags": spec.tags,
        "trainability_tags": spec.trainability_tags,
        "missingness_policy": spec.missingness_policy,
        "warmup_rows": spec.warmup_rows,
        "lookback_rows": spec.lookback_rows,
        "row_count_total": total_rows,
        "warmup_excluded_rows": warmup_excluded_rows,
        "row_count_post_warmup": post_warmup_rows,
        "complete_row_ratio_post_warmup": complete_row_ratio,
        "family_non_null_ratio_post_warmup": family_non_null_ratio,
        "constant_columns": sorted(constant_columns),
        "low_variation_columns": sorted(low_variation_columns),
        "null_heavy_columns": sorted(null_heavy_columns),
        "column_summaries": column_summaries,
        "warning_reasons": warnings,
    }
