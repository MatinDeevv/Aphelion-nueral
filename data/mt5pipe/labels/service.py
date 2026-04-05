"""Label view materialization services."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from mt5pipe.contracts.dataset import DATASET_JOIN_KEYS
from mt5pipe.labels.artifacts import LabelArtifactRef
from mt5pipe.catalog.sqlite import CatalogDB
from mt5pipe.compiler.manifest import (
    build_stage_artifact_id,
    build_stage_manifest_id,
    compute_content_hash,
    write_manifest_sidecar,
)
from mt5pipe.compiler.models import LineageManifest
from mt5pipe.features.labels import (
    add_direction_labels,
    add_future_returns,
    add_triple_barrier_labels,
)
from mt5pipe.labels.registry.models import LabelPack
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths

LABEL_JOIN_KEYS = DATASET_JOIN_KEYS


@dataclass
class LabelMaterializationResult:
    artifact_id: str
    artifact_ref: LabelArtifactRef
    manifest: LineageManifest
    manifest_path: Path
    label_df: pl.DataFrame


class LabelService:
    """Materialize registered label packs as first-class compiler artifacts."""

    def __init__(
        self,
        paths: StoragePaths,
        store: ParquetStore,
        catalog: CatalogDB,
    ) -> None:
        self._paths = paths
        self._store = store
        self._catalog = catalog

    def materialize_labels(
        self,
        *,
        symbol: str,
        base_clock: str,
        date_from: dt.date,
        date_to: dt.date,
        label_pack: LabelPack,
        base_df: pl.DataFrame,
        state_artifact_id: str,
        build_id: str,
        dataset_spec_ref: str,
        code_version: str,
    ) -> LabelMaterializationResult:
        label_df = self._build_label_view(base_df, label_pack)
        input_partition_refs = self._collect_input_partition_refs(symbol, base_clock, date_from, date_to)
        created_at = dt.datetime.now(dt.timezone.utc)
        content_hash = compute_content_hash({
            "artifact_kind": "label_view",
            "label_pack": label_pack.model_dump(mode="json"),
            "rows": len(label_df),
            "columns": label_df.columns,
            "time_range_start": str(label_df["time_utc"].min()) if not label_df.is_empty() else "",
            "time_range_end": str(label_df["time_utc"].max()) if not label_df.is_empty() else "",
            "state_artifact_id": state_artifact_id,
            "input_partition_refs": input_partition_refs,
        })
        artifact_id = build_stage_artifact_id("label_view", label_pack.key, created_at, content_hash)
        manifest = LineageManifest(
            manifest_id=build_stage_manifest_id("label_view", label_pack.key, created_at, content_hash),
            artifact_id=artifact_id,
            artifact_kind="label_view",
            logical_name=label_pack.key,
            logical_version=label_pack.version,
            artifact_uri=str(
                self._paths.label_artifact_root(label_pack.key, label_pack.base_clock, artifact_id)
            ),
            content_hash=content_hash,
            build_id=build_id,
            created_at=created_at,
            status="accepted",
            dataset_spec_ref=dataset_spec_ref,
            state_artifact_refs=[state_artifact_id],
            label_pack_ref=label_pack.key,
            code_version=code_version,
            input_partition_refs=input_partition_refs,
            parent_artifact_refs=[state_artifact_id],
            metadata={
                "row_count": len(label_df),
                "column_count": len(label_df.columns),
                "output_columns": label_pack.output_columns,
                "purge_rows": label_pack.purge_rows,
                "exclusions": label_pack.exclusions,
                "status": label_pack.status,
                "qa_policy_ref": label_pack.qa_policy_ref,
                "ablation_group": label_pack.ablation_group or label_pack.label_pack_name,
                "trainability_tags": label_pack.trainability_tags,
                "target_groups": label_pack.target_groups,
                "tail_policy": label_pack.tail_policy,
                "label_diagnostics": _label_manifest_diagnostics(label_df, label_pack),
                "time_range_start": str(label_df["time_utc"].min()) if not label_df.is_empty() else "",
                "time_range_end": str(label_df["time_utc"].max()) if not label_df.is_empty() else "",
            },
        )

        self._write_label_partitions(label_pack, artifact_id, label_df)
        manifest_path = write_manifest_sidecar(manifest, self._paths)
        self._catalog.register_artifact(manifest, str(manifest_path))
        return LabelMaterializationResult(
            artifact_id=artifact_id,
            artifact_ref=LabelArtifactRef(
                artifact_id=artifact_id,
                label_pack_key=label_pack.key,
                clock=label_pack.base_clock,
            ),
            manifest=manifest,
            manifest_path=manifest_path,
            label_df=label_df,
        )

    def _build_label_view(self, base_df: pl.DataFrame, label_pack: LabelPack) -> pl.DataFrame:
        working = base_df.clone().sort("time_utc")
        working = add_future_returns(working, label_pack.horizons_minutes)
        working = add_direction_labels(
            working,
            label_pack.horizons_minutes,
            threshold=float(label_pack.parameters.get("direction_threshold_bps", 0.0)) / 10_000.0,
        )
        working = add_triple_barrier_labels(
            working,
            label_pack.horizons_minutes,
            tp_bps=float(label_pack.parameters.get("tp_bps", 50.0)),
            sl_bps=float(label_pack.parameters.get("sl_bps", 50.0)),
            vol_scale_window=int(label_pack.parameters.get("vol_scale_window", 0)),
            vol_multiplier=float(label_pack.parameters.get("vol_multiplier", 2.0)),
        )
        return working.select([*LABEL_JOIN_KEYS, *label_pack.output_columns]).sort("time_utc")

    def _collect_input_partition_refs(
        self,
        symbol: str,
        base_clock: str,
        date_from: dt.date,
        date_to: dt.date,
    ) -> list[str]:
        refs: list[str] = []
        current = date_from
        while current <= date_to:
            base_dir = self._paths.built_bars_dir(symbol, base_clock, current)
            if base_dir.exists():
                refs.append(str(base_dir))
            current += dt.timedelta(days=1)
        return sorted(set(refs))

    def _write_label_partitions(self, label_pack: LabelPack, artifact_id: str, label_df: pl.DataFrame) -> None:
        dated = label_df.with_columns(pl.col("time_utc").dt.date().alias("_date"))
        for date_val in dated["_date"].unique().sort().to_list():
            day_df = dated.filter(pl.col("_date") == date_val).drop("_date")
            self._store.write(day_df, self._paths.label_view_file(label_pack.key, label_pack.base_clock, date_val))
            self._store.write(
                day_df,
                self._paths.label_artifact_file(label_pack.key, label_pack.base_clock, artifact_id, date_val),
            )


def _label_manifest_diagnostics(label_df: pl.DataFrame, label_pack: LabelPack) -> dict[str, Any]:
    """Build compact horizon-level diagnostics for manifest metadata."""
    horizon_summaries: dict[str, dict[str, Any]] = {}
    max_horizon = max(label_pack.horizons_minutes)
    direction_degenerate_horizons: list[str] = []
    triple_barrier_degenerate_horizons: list[str] = []
    low_dispersion_horizons: list[str] = []
    tail_mismatch_horizons: list[str] = []
    future_return_std_by_horizon: dict[str, float | None] = {}
    direction_dominant_share_by_horizon: dict[str, float | None] = {}
    triple_barrier_hit_rate_by_horizon: dict[str, float | None] = {}
    usable_rows_by_horizon: dict[str, int] = {}

    for horizon in label_pack.horizons_minutes:
        suffix = f"{horizon}m"
        future_col = f"future_return_{suffix}"
        direction_col = f"direction_{suffix}"
        tb_col = f"triple_barrier_{suffix}"
        mae_col = f"mae_{suffix}"
        mfe_col = f"mfe_{suffix}"

        direction_balance = _class_balance(label_df, direction_col)
        tb_balance = _class_balance(label_df, tb_col)
        future_return_null_rows = _null_count(label_df, future_col)
        direction_null_rows = _null_count(label_df, direction_col)
        triple_barrier_null_rows = _null_count(label_df, tb_col)
        expected_tail_null_rows = min(horizon, label_df.height)
        direction_entropy = _classification_entropy(direction_balance)
        direction_dominant_share = _dominant_class_share(direction_balance)
        triple_barrier_entropy = _classification_entropy(tb_balance)
        triple_barrier_dominant_share = _dominant_class_share(tb_balance)
        future_return_distribution = _numeric_distribution_summary(label_df, future_col)
        mae_distribution = _numeric_distribution_summary(label_df, mae_col)
        mfe_distribution = _numeric_distribution_summary(label_df, mfe_col)
        mfe_mae_ratio_mean = _mfe_mae_ratio_mean(label_df, mae_col, mfe_col)

        warnings: list[str] = []
        if future_return_null_rows != expected_tail_null_rows or direction_null_rows != expected_tail_null_rows or triple_barrier_null_rows != expected_tail_null_rows:
            warnings.append("tail_null_mismatch")
            tail_mismatch_horizons.append(suffix)
        if direction_dominant_share is not None and direction_dominant_share >= 0.95:
            warnings.append("direction_degenerate")
            direction_degenerate_horizons.append(suffix)
        if triple_barrier_dominant_share is not None and triple_barrier_dominant_share >= 0.95:
            warnings.append("triple_barrier_degenerate")
            triple_barrier_degenerate_horizons.append(suffix)
        if future_return_distribution["std"] is None or future_return_distribution["std"] <= 1e-6:
            warnings.append("future_return_low_dispersion")
            low_dispersion_horizons.append(suffix)

        horizon_summaries[suffix] = {
            "expected_tail_null_rows": expected_tail_null_rows,
            "usable_rows": max(label_df.height - future_return_null_rows, 0),
            "future_return_null_rows": future_return_null_rows,
            "direction_null_rows": direction_null_rows,
            "triple_barrier_null_rows": triple_barrier_null_rows,
            "future_return_distribution": future_return_distribution,
            "direction_class_balance": direction_balance,
            "direction_entropy": direction_entropy,
            "direction_dominant_class_share": direction_dominant_share,
            "triple_barrier_class_balance": tb_balance,
            "triple_barrier_entropy": triple_barrier_entropy,
            "triple_barrier_dominant_class_share": triple_barrier_dominant_share,
            "triple_barrier_hit_rate": _hit_rate(tb_balance),
            "mae_distribution": mae_distribution,
            "mfe_distribution": mfe_distribution,
            "mfe_mae_ratio_mean": mfe_mae_ratio_mean,
            "warning_reasons": warnings,
        }
        future_return_std_by_horizon[suffix] = future_return_distribution["std"]
        direction_dominant_share_by_horizon[suffix] = direction_dominant_share
        triple_barrier_hit_rate_by_horizon[suffix] = horizon_summaries[suffix]["triple_barrier_hit_rate"]
        usable_rows_by_horizon[suffix] = horizon_summaries[suffix]["usable_rows"]

    return {
        "base_clock": label_pack.base_clock,
        "status": label_pack.status,
        "qa_policy_ref": label_pack.qa_policy_ref,
        "ablation_group": label_pack.ablation_group or label_pack.label_pack_name,
        "trainability_tags": label_pack.trainability_tags,
        "target_groups": label_pack.target_groups,
        "tail_policy": label_pack.tail_policy,
        "horizons_minutes": label_pack.horizons_minutes,
        "max_horizon_minutes": max_horizon,
        "purge_rows": label_pack.purge_rows,
        "recommended_min_embargo_rows": label_pack.purge_rows,
        "exclusions": label_pack.exclusions,
        "direction_threshold_bps": float(label_pack.parameters.get("direction_threshold_bps", 0.0)),
        "tp_bps": float(label_pack.parameters.get("tp_bps", 50.0)),
        "sl_bps": float(label_pack.parameters.get("sl_bps", 50.0)),
        "vol_scale_window": int(label_pack.parameters.get("vol_scale_window", 0)),
        "vol_multiplier": float(label_pack.parameters.get("vol_multiplier", 2.0)),
        "constant_output_columns": _constant_output_columns(label_df, label_pack.output_columns),
        "degenerate_direction_horizons": sorted(set(direction_degenerate_horizons)),
        "degenerate_triple_barrier_horizons": sorted(set(triple_barrier_degenerate_horizons)),
        "low_dispersion_horizons": sorted(set(low_dispersion_horizons)),
        "tail_mismatch_horizons": sorted(set(tail_mismatch_horizons)),
        "target_distribution_summary": {
            "usable_rows_by_horizon": usable_rows_by_horizon,
            "future_return_std_by_horizon": future_return_std_by_horizon,
            "direction_dominant_class_share_by_horizon": direction_dominant_share_by_horizon,
            "triple_barrier_hit_rate_by_horizon": triple_barrier_hit_rate_by_horizon,
        },
        "warning_reasons": _label_warning_reasons(
            direction_degenerate_horizons,
            triple_barrier_degenerate_horizons,
            low_dispersion_horizons,
            tail_mismatch_horizons,
        ),
        "horizon_summaries": horizon_summaries,
    }


def _null_count(label_df: pl.DataFrame, column: str) -> int:
    if column not in label_df.columns:
        return label_df.height
    return int(label_df[column].null_count())


def _class_balance(label_df: pl.DataFrame, column: str) -> dict[str, int]:
    if column not in label_df.columns:
        return {"-1": 0, "0": 0, "1": 0}

    non_null = label_df[column].drop_nulls().to_list()
    return {
        "-1": int(sum(1 for value in non_null if value == -1)),
        "0": int(sum(1 for value in non_null if value == 0)),
        "1": int(sum(1 for value in non_null if value == 1)),
    }


def _constant_output_columns(label_df: pl.DataFrame, output_columns: list[str]) -> list[str]:
    constant_columns: list[str] = []
    for column in output_columns:
        if column not in label_df.columns:
            continue
        non_null = label_df[column].drop_nulls()
        if non_null.is_empty():
            continue
        if non_null.n_unique() == 1:
            constant_columns.append(column)
    return sorted(constant_columns)


def _numeric_distribution_summary(label_df: pl.DataFrame, column: str) -> dict[str, float | None]:
    if column not in label_df.columns:
        return {"count": 0, "mean": None, "std": None, "abs_mean": None, "p05": None, "p50": None, "p95": None}

    values = label_df[column].drop_nulls().cast(pl.Float64)
    if values.is_empty():
        return {"count": 0, "mean": None, "std": None, "abs_mean": None, "p05": None, "p50": None, "p95": None}

    return {
        "count": int(values.len()),
        "mean": _rounded_float(values.mean()),
        "std": _rounded_float(values.std(ddof=0)),
        "abs_mean": _rounded_float(values.abs().mean()),
        "p05": _rounded_float(values.quantile(0.05)),
        "p50": _rounded_float(values.quantile(0.50)),
        "p95": _rounded_float(values.quantile(0.95)),
    }


def _mfe_mae_ratio_mean(label_df: pl.DataFrame, mae_col: str, mfe_col: str) -> float | None:
    if mae_col not in label_df.columns or mfe_col not in label_df.columns:
        return None

    ratios: list[float] = []
    for mae_value, mfe_value in zip(label_df[mae_col].to_list(), label_df[mfe_col].to_list(), strict=True):
        if mae_value is None or mfe_value is None or mae_value <= 0:
            continue
        ratios.append(float(mfe_value) / float(mae_value))
    if not ratios:
        return None
    return round(sum(ratios) / len(ratios), 6)


def _classification_entropy(class_balance: dict[str, int]) -> float | None:
    total = sum(class_balance.values())
    if total <= 0:
        return None

    entropy = 0.0
    categories = 0
    for count in class_balance.values():
        if count <= 0:
            continue
        categories += 1
        probability = count / total
        entropy -= probability * math.log(probability)
    if categories <= 1:
        return 0.0
    return round(entropy / math.log(categories), 6)


def _dominant_class_share(class_balance: dict[str, int]) -> float | None:
    total = sum(class_balance.values())
    if total <= 0:
        return None
    return round(max(class_balance.values()) / total, 6)


def _rounded_float(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 8)


def _label_warning_reasons(
    direction_degenerate_horizons: list[str],
    triple_barrier_degenerate_horizons: list[str],
    low_dispersion_horizons: list[str],
    tail_mismatch_horizons: list[str],
) -> list[str]:
    warning_reasons: list[str] = []
    if direction_degenerate_horizons:
        warning_reasons.append("degenerate_direction_horizons_present")
    if triple_barrier_degenerate_horizons:
        warning_reasons.append("degenerate_triple_barrier_horizons_present")
    if low_dispersion_horizons:
        warning_reasons.append("low_dispersion_horizons_present")
    if tail_mismatch_horizons:
        warning_reasons.append("tail_null_mismatch_present")
    return warning_reasons


def _hit_rate(class_balance: dict[str, int]) -> float | None:
    total = sum(class_balance.values())
    if total <= 0:
        return None
    return round((class_balance["-1"] + class_balance["1"]) / total, 6)
