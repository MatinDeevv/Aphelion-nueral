"""Truth gate for compiler-era dataset artifacts."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from mt5pipe.compiler.models import DatasetSpec, LineageManifest
from mt5pipe.features.public import FeatureSpec
from mt5pipe.labels.public import LabelPack
from mt5pipe.quality.report import dataset_quality_report
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths
from mt5pipe.truth.models import QaCheckResult, TrustReport


class TruthService:
    """Evaluate a candidate dataset artifact and gate publication."""

    MIN_TOTAL_SCORE = 80.0
    MIN_COVERAGE_SCORE = 85.0
    MIN_FEATURE_QUALITY_SCORE = 80.0
    MIN_LABEL_QUALITY_SCORE = 80.0
    MIN_SOURCE_QUALITY_SCORE = 60.0
    PREFERRED_SOURCE_QUALITY_SCORE = 75.0
    CRITICAL_BASE_CONSTANT_COLUMNS = {
        "open",
        "high",
        "low",
        "close",
        "spread_mean",
        "mid_return",
        "realized_vol",
        "tick_count",
    }

    FAMILY_MISSINGNESS_THRESHOLDS = {
        "time": 0.0,
        "session": 0.0,
        "quality": 0.0,
        "htf_context": 100.0,
        "disagreement": 15.0,
        "event_shape": 15.0,
        "entropy": 20.0,
        "multiscale": 10.0,
    }

    def evaluate_dataset(
        self,
        *,
        artifact_id: str,
        dataset_df: pl.DataFrame,
        split_frames: dict[str, pl.DataFrame],
        spec: DatasetSpec,
        feature_specs: list[FeatureSpec],
        label_pack: LabelPack,
        manifest: LineageManifest,
        paths: StoragePaths,
        store: ParquetStore,
        state_df: pl.DataFrame | None = None,
        build_row_stats: dict[str, int] | None = None,
        expected_content_hash: str | None = None,
        manifest_path: Path | None = None,
    ) -> TrustReport:
        quality = dataset_quality_report(dataset_df)
        dataset_quality_metrics = self._dataset_quality_metrics(dataset_df, quality, feature_specs, label_pack)
        checks: list[QaCheckResult] = []
        hard_failures: list[str] = []
        warnings: list[str] = []
        build_row_stats = build_row_stats or {}

        coverage_check = self._coverage_check(dataset_df, split_frames)
        checks.append(coverage_check)
        if coverage_check.status == "failed":
            hard_failures.append("dataset_coverage_failure")
        coverage_score = coverage_check.score

        split_integrity_check = self._split_integrity_check(dataset_df, split_frames, spec)
        checks.append(split_integrity_check)
        if split_integrity_check.status == "failed":
            hard_failures.append("split_integrity_failure")

        leakage_check = self._leakage_check(dataset_df, feature_specs, split_integrity_check)
        checks.append(leakage_check)
        if leakage_check.status == "failed":
            hard_failures.append("leakage_or_duplicate_timestamp_failure")
        leakage_score = leakage_check.score

        feature_contracts_check = self._feature_contracts_check(dataset_df, feature_specs)
        checks.append(feature_contracts_check)
        if feature_contracts_check.status == "failed":
            hard_failures.append("missing_required_feature_columns")

        family_missingness_check = self._feature_family_missingness_check(dataset_df, feature_specs)
        checks.append(family_missingness_check)
        if family_missingness_check.status == "failed":
            hard_failures.append("feature_family_missingness_threshold_exceeded")
        elif family_missingness_check.status == "warning":
            warnings.append("feature_family_missingness_warning")

        warmup_check = self._warmup_and_row_loss_check(build_row_stats, feature_specs, label_pack)
        checks.append(warmup_check)
        if warmup_check.status == "failed":
            hard_failures.append("warmup_or_drop_row_sanity_failure")

        feature_quality_score = round(
            (
                feature_contracts_check.score
                + family_missingness_check.score
                + warmup_check.score
            ) / 3.0,
            4,
        )

        if dataset_quality_metrics["unexpected_null_columns"]:
            warnings.append("dataset_contains_nulls")
        if dataset_quality_metrics["blocking_constant_columns"]:
            warnings.append("dataset_contains_constant_columns")

        label_check = self._label_quality_check(dataset_df, label_pack)
        checks.append(label_check)
        if label_check.status == "failed":
            hard_failures.append("label_columns_missing")
        elif label_check.status == "warning":
            warnings.append("label_quality_warning")
        label_quality_score = label_check.score

        resolved_state_df = state_df if state_df is not None else pl.DataFrame()
        source_check = self._source_quality_check(
            spec=spec,
            symbol=spec.symbols[0],
            date_from=spec.date_from,
            date_to=spec.date_to,
            paths=paths,
            store=store,
            state_df=resolved_state_df,
        )
        checks.append(source_check)
        if source_check.status == "failed":
            hard_failures.append("source_quality_below_threshold")
        elif source_check.status == "warning":
            warnings.append("source_quality_warning")
        source_quality_score = source_check.score
        quality_caveat_summary = self._quality_caveat_summary(
            dataset_quality_metrics=dataset_quality_metrics,
            source_check=source_check,
        )

        lineage_check = self._lineage_check(manifest)
        checks.append(lineage_check)
        if lineage_check.status == "failed":
            hard_failures.append("lineage_incomplete")
        lineage_score = lineage_check.score

        manifest_integrity_check = self._manifest_integrity_check(
            manifest=manifest,
            expected_content_hash=expected_content_hash,
            manifest_path=manifest_path,
        )
        checks.append(manifest_integrity_check)
        if manifest_integrity_check.status == "failed":
            hard_failures.append("manifest_hash_mismatch")

        trust_score_total = round(
            coverage_score * 0.20
            + leakage_score * 0.20
            + feature_quality_score * 0.20
            + label_quality_score * 0.15
            + source_quality_score * 0.10
            + lineage_score * 0.15,
            4,
        )

        accepted = (
            not hard_failures
            and trust_score_total >= self.MIN_TOTAL_SCORE
            and coverage_score >= self.MIN_COVERAGE_SCORE
            and leakage_score == 100.0
            and feature_quality_score >= self.MIN_FEATURE_QUALITY_SCORE
            and label_quality_score >= self.MIN_LABEL_QUALITY_SCORE
            and source_quality_score >= self.MIN_SOURCE_QUALITY_SCORE
            and lineage_score == 100.0
        )
        status = "accepted" if accepted else "rejected"
        score_breakdown = {
            "total": round(trust_score_total, 4),
            "coverage": round(coverage_score, 4),
            "leakage": round(leakage_score, 4),
            "feature_quality": round(feature_quality_score, 4),
            "label_quality": round(label_quality_score, 4),
            "source_quality": round(source_quality_score, 4),
            "lineage": round(lineage_score, 4),
        }
        failed_checks = [check for check in checks if check.status == "failed"]
        warning_checks = [check for check in checks if check.status == "warning"]
        threshold_shortfalls = self._threshold_shortfalls(
            trust_score_total=trust_score_total,
            coverage_score=coverage_score,
            leakage_score=leakage_score,
            feature_quality_score=feature_quality_score,
            label_quality_score=label_quality_score,
            source_quality_score=source_quality_score,
            lineage_score=lineage_score,
        )
        rejection_reasons = self._rejection_reasons(failed_checks, threshold_shortfalls)
        warning_reasons = self._warning_reasons(
            warning_checks,
            warnings,
            dataset_quality_metrics=dataset_quality_metrics,
        )
        check_status_counts = {
            "passed": sum(1 for check in checks if check.status == "passed"),
            "warning": len(warning_checks),
            "failed": len(failed_checks),
        }
        decision_summary = self._decision_summary(
            accepted=accepted,
            score_breakdown=score_breakdown,
            rejection_reasons=rejection_reasons,
            warning_reasons=warning_reasons,
        )

        return TrustReport(
            report_id=f"trust.{artifact_id}",
            artifact_id=artifact_id,
            artifact_kind="dataset",
            truth_policy_version=spec.truth_policy_ref,
            status=status,
            accepted_for_publication=accepted,
            trust_score_total=trust_score_total,
            coverage_score=round(coverage_score, 4),
            leakage_score=round(leakage_score, 4),
            feature_quality_score=round(feature_quality_score, 4),
            label_quality_score=round(label_quality_score, 4),
            source_quality_score=round(source_quality_score, 4),
            lineage_score=round(lineage_score, 4),
            score_breakdown=score_breakdown,
            hard_failures=sorted(set(hard_failures)),
            warnings=sorted(set(warnings)),
            rejection_reasons=rejection_reasons,
            warning_reasons=warning_reasons,
            check_status_counts=check_status_counts,
            decision_summary=decision_summary,
            metrics={
                "rows": len(dataset_df),
                "columns": len(dataset_df.columns),
                "duplicate_primary_clock_rows": int(leakage_check.metrics.get("duplicate_primary_clock_rows", 0)),
                "quality_score": float(quality.get("quality_score", 0.0)),
                "dataset_quality": dataset_quality_metrics,
                "quality_caveat_summary": quality_caveat_summary,
                "build_row_stats": build_row_stats,
                "split_rows": coverage_check.metrics.get("split_rows", {}),
                "feature_family_missingness": family_missingness_check.metrics,
                "label_quality": label_check.metrics,
                "source_quality": source_check.metrics,
            },
            thresholds={
                "min_total_score": self.MIN_TOTAL_SCORE,
                "min_coverage_score": self.MIN_COVERAGE_SCORE,
                "min_feature_quality_score": self.MIN_FEATURE_QUALITY_SCORE,
                "min_label_quality_score": self.MIN_LABEL_QUALITY_SCORE,
                "min_source_quality_score": self.MIN_SOURCE_QUALITY_SCORE,
                "family_missingness_thresholds": self.FAMILY_MISSINGNESS_THRESHOLDS,
            },
            generated_at=dt.datetime.now(dt.timezone.utc),
            checks=checks,
        )

    def _coverage_check(self, dataset_df: pl.DataFrame, split_frames: dict[str, pl.DataFrame]) -> QaCheckResult:
        dataset_empty = dataset_df.is_empty()
        empty_splits = [name for name, frame in split_frames.items() if frame.is_empty()]
        unique_dates = (
            sorted(dataset_df["time_utc"].dt.date().unique().cast(pl.Utf8).to_list())
            if not dataset_df.is_empty() and "time_utc" in dataset_df.columns
            else []
        )
        failed = dataset_empty or bool(empty_splits)
        return QaCheckResult(
            check_name="coverage",
            status="failed" if failed else "passed",
            score=0.0 if failed else 100.0,
            metrics={
                "rows": len(dataset_df),
                "split_rows": {k: len(v) for k, v in split_frames.items()},
                "observed_dates": unique_dates,
                "observed_date_count": len(unique_dates),
            },
            thresholds={"min_rows": 1, "required_non_empty_splits": sorted(split_frames)},
            failure_reason=(
                "dataset artifact has no rows"
                if dataset_empty
                else f"one or more required splits is empty: {', '.join(empty_splits)}"
                if empty_splits
                else ""
            ),
        )

    def _split_integrity_check(
        self,
        dataset_df: pl.DataFrame,
        split_frames: dict[str, pl.DataFrame],
        spec: DatasetSpec,
    ) -> QaCheckResult:
        split_ranges: dict[str, tuple[str, str]] = {}
        overlap_detected = False
        ordering_failure = False

        previous_end: dt.datetime | None = None
        ordered_split_names = self._ordered_split_names(split_frames)
        for split_name in ordered_split_names:
            frame = split_frames[split_name]
            if frame.is_empty() or "time_utc" not in frame.columns:
                continue
            start = frame["time_utc"].min()
            end = frame["time_utc"].max()
            split_ranges[split_name] = (str(start), str(end))
            if previous_end is not None and start is not None and start <= previous_end:
                overlap_detected = True
            previous_end = end

        if not dataset_df.is_empty() and {"symbol", "timeframe", "time_utc"}.issubset(set(dataset_df.columns)):
            total_unique = dataset_df.select(["symbol", "timeframe", "time_utc"]).unique().height
            split_unique = 0
            for frame in split_frames.values():
                if frame.is_empty():
                    continue
                current = frame.select([col for col in ["symbol", "timeframe", "time_utc"] if col in frame.columns]).unique().height
                split_unique += current
            ordering_failure = split_unique > total_unique

        failed = overlap_detected or ordering_failure
        return QaCheckResult(
            check_name="split_integrity",
            status="failed" if failed else "passed",
            score=0.0 if failed else 100.0,
            metrics={
                "split_ranges": split_ranges,
                "split_policy": spec.split_policy,
                "overlap_detected": overlap_detected,
                "ordering_failure": ordering_failure,
            },
            thresholds={"strict_temporal_ordering": True, "no_overlap": True},
            failure_reason="splits overlap or violate strict temporal ordering" if failed else "",
        )

    def _leakage_check(
        self,
        dataset_df: pl.DataFrame,
        feature_specs: list[FeatureSpec],
        split_integrity_check: QaCheckResult,
    ) -> QaCheckResult:
        duplicate_rows = self._duplicate_primary_clock_rows(dataset_df)
        all_pit_safe = all(feature.point_in_time_safe for feature in feature_specs)
        failed = duplicate_rows > 0 or not all_pit_safe or split_integrity_check.status == "failed"
        return QaCheckResult(
            check_name="leakage",
            status="failed" if failed else "passed",
            score=0.0 if failed else 100.0,
            metrics={
                "duplicate_primary_clock_rows": duplicate_rows,
                "all_features_pit_safe": all_pit_safe,
                "split_integrity_status": split_integrity_check.status,
            },
            thresholds={"duplicate_primary_clock_rows": 0, "all_features_pit_safe": True, "split_integrity": "passed"},
            failure_reason="duplicate primary clock rows, non-PIT feature, or split leakage detected" if failed else "",
        )

    def _feature_contracts_check(self, dataset_df: pl.DataFrame, feature_specs: list[FeatureSpec]) -> QaCheckResult:
        missing_required_feature_columns = [
            col
            for feature in feature_specs
            for col in feature.output_columns
            if col not in dataset_df.columns
        ]
        failed = bool(missing_required_feature_columns)
        return QaCheckResult(
            check_name="feature_contracts",
            status="failed" if failed else "passed",
            score=0.0 if failed else 100.0,
            metrics={"missing_required_feature_columns": missing_required_feature_columns},
            thresholds={"missing_required_feature_columns": []},
            failure_reason="required feature outputs are missing from the dataset artifact" if failed else "",
        )

    def _feature_family_missingness_check(self, dataset_df: pl.DataFrame, feature_specs: list[FeatureSpec]) -> QaCheckResult:
        family_metrics: dict[str, dict[str, float | int]] = {}
        failed_families: list[str] = []
        warning_families: list[str] = []

        for family, family_specs in self._group_feature_specs_by_family(feature_specs).items():
            columns = sorted({column for feature in family_specs for column in feature.output_columns})
            present_columns = [column for column in columns if column in dataset_df.columns]
            threshold = self._family_missingness_threshold(family, family_specs)

            if not present_columns or dataset_df.is_empty():
                null_pct = 100.0 if columns else 0.0
            else:
                total_cells = len(dataset_df) * len(present_columns)
                null_cells = sum(int(dataset_df[column].null_count()) for column in present_columns)
                null_pct = round((null_cells / total_cells * 100.0) if total_cells else 0.0, 4)

            family_metrics[family] = {
                "expected_columns": len(columns),
                "present_columns": len(present_columns),
                "null_pct": null_pct,
                "threshold_pct": threshold,
            }

            if len(present_columns) != len(columns) or null_pct > threshold:
                failed_families.append(family)
            elif threshold > 0.0 and null_pct > threshold * 0.75:
                warning_families.append(family)

        score = 100.0
        for metrics in family_metrics.values():
            null_pct = float(metrics["null_pct"])
            threshold = float(metrics["threshold_pct"])
            if threshold <= 0.0:
                if null_pct > 0.0:
                    score = 0.0
                    break
            else:
                score -= min((null_pct / threshold) * 10.0, 20.0)
        score = max(0.0, round(score, 4))

        status = "failed" if failed_families else "warning" if warning_families else "passed"
        return QaCheckResult(
            check_name="feature_family_missingness",
            status=status,
            score=0.0 if failed_families else score,
            metrics={"families": family_metrics, "failed_families": failed_families, "warning_families": warning_families},
            thresholds={"family_missingness_thresholds": self.FAMILY_MISSINGNESS_THRESHOLDS},
            failure_reason=f"feature family missingness exceeded threshold for: {', '.join(sorted(failed_families))}" if failed_families else "",
        )

    def _warmup_and_row_loss_check(
        self,
        build_row_stats: dict[str, int],
        feature_specs: list[FeatureSpec],
        label_pack: LabelPack,
    ) -> QaCheckResult:
        drop_row_removed = int(build_row_stats.get("feature_drop_row_rows_removed", 0))
        label_purge_applied = int(build_row_stats.get("label_purge_rows_applied", 0))
        rows_after_join = int(build_row_stats.get("rows_after_join", 0))
        drop_row_budget = max(
            [feature.warmup_rows for feature in feature_specs if feature.missingness_policy == "drop_row"] or [0]
        )
        failed = False
        reasons: list[str] = []

        if drop_row_removed > max(drop_row_budget, 0):
            failed = True
            reasons.append("drop-row removal exceeded declared warmup budget")
        if rows_after_join > label_pack.purge_rows and label_purge_applied != label_pack.purge_rows:
            failed = True
            reasons.append("label purge rows applied does not match label pack contract")

        return QaCheckResult(
            check_name="warmup_and_row_loss",
            status="failed" if failed else "passed",
            score=0.0 if failed else 100.0,
            metrics={
                "feature_drop_row_rows_removed": drop_row_removed,
                "drop_row_budget": drop_row_budget,
                "label_purge_rows_applied": label_purge_applied,
                "expected_label_purge_rows": label_pack.purge_rows,
                "rows_after_join": rows_after_join,
            },
            thresholds={"max_drop_row_rows_removed": drop_row_budget, "expected_label_purge_rows": label_pack.purge_rows},
            failure_reason="; ".join(reasons) if reasons else "",
        )

    def _label_quality_check(self, dataset_df: pl.DataFrame, label_pack: LabelPack) -> QaCheckResult:
        label_cols = [col for col in label_pack.output_columns if col in dataset_df.columns]
        missing_label_columns = [col for col in label_pack.output_columns if col not in dataset_df.columns]
        all_null_label_columns = [
            col for col in label_cols
            if int(dataset_df[col].null_count()) == len(dataset_df) and not dataset_df.is_empty()
        ]
        label_nulls = sum(int(dataset_df[col].null_count()) for col in label_cols)
        horizon_metrics: dict[str, dict[str, int]] = {}
        for horizon in label_pack.horizons_minutes:
            horizon_token = f"_{horizon}m"
            horizon_columns = [col for col in label_pack.output_columns if horizon_token in col]
            present_columns = [col for col in horizon_columns if col in dataset_df.columns]
            null_columns = [
                col for col in present_columns
                if int(dataset_df[col].null_count()) == len(dataset_df) and not dataset_df.is_empty()
            ]
            horizon_metrics[str(horizon)] = {
                "expected_columns": len(horizon_columns),
                "present_columns": len(present_columns),
                "all_null_columns": len(null_columns),
            }
        horizons_with_missing_columns = sorted(
            horizon for horizon, metrics in horizon_metrics.items()
            if metrics["present_columns"] != metrics["expected_columns"]
        )
        horizons_with_all_null_columns = sorted(
            horizon for horizon, metrics in horizon_metrics.items()
            if metrics["all_null_columns"] > 0
        )

        if missing_label_columns or all_null_label_columns:
            score = 0.0
            status = "failed"
        elif label_nulls > 0:
            score = 80.0
            status = "warning"
        else:
            score = 100.0
            status = "passed"

        return QaCheckResult(
            check_name="label_quality",
            status=status,
            score=score,
            metrics={
                "label_columns_present": len(label_cols),
                "label_nulls": label_nulls,
                "missing_label_columns": missing_label_columns,
                "all_null_label_columns": all_null_label_columns,
                "horizon_metrics": horizon_metrics,
                "horizons_with_missing_columns": horizons_with_missing_columns,
                "horizons_with_all_null_columns": horizons_with_all_null_columns,
            },
            thresholds={"expected_label_columns": len(label_pack.output_columns), "label_nulls": 0},
            failure_reason=(
                "not all label columns were materialized"
                + (
                    f"; affected horizons: {', '.join(horizons_with_missing_columns)}"
                    if horizons_with_missing_columns
                    else ""
                )
                if missing_label_columns
                else "one or more label columns is entirely null"
                + (
                    f"; affected horizons: {', '.join(horizons_with_all_null_columns)}"
                    if horizons_with_all_null_columns
                    else ""
                )
                if all_null_label_columns
                else ""
            ),
        )

    def _source_quality_check(
        self,
        *,
        spec: DatasetSpec,
        symbol: str,
        date_from: dt.date,
        date_to: dt.date,
        paths: StoragePaths,
        store: ParquetStore,
        state_df: pl.DataFrame,
    ) -> QaCheckResult:
        metrics = self._source_quality_metrics(
            spec=spec,
            symbol=symbol,
            date_from=date_from,
            date_to=date_to,
            paths=paths,
            store=store,
            state_df=state_df,
        )
        score = self._source_quality_score(metrics)
        requirement_failures = [str(reason) for reason in metrics.get("requirement_failures", [])]
        failed = bool(requirement_failures) or score < self.MIN_SOURCE_QUALITY_SCORE
        warning = not failed and score < self.PREFERRED_SOURCE_QUALITY_SCORE
        return QaCheckResult(
            check_name="source_quality",
            status="failed" if failed else "warning" if warning else "passed",
            score=score,
            metrics=metrics,
            thresholds={
                "min_source_quality_score": self.MIN_SOURCE_QUALITY_SCORE,
                "preferred_source_quality_score": self.PREFERRED_SOURCE_QUALITY_SCORE,
                "required_raw_brokers": list(spec.required_raw_brokers),
                "require_synchronized_raw_coverage": spec.require_synchronized_raw_coverage,
                "require_dual_source_overlap": spec.require_dual_source_overlap,
                "min_dual_source_ratio": spec.min_dual_source_ratio,
            },
            failure_reason=(
                "; ".join(requirement_failures)
                if requirement_failures
                else "source/merge quality metrics are below publication threshold"
                if failed
                else ""
            ),
        )

    def _lineage_check(self, manifest: LineageManifest) -> QaCheckResult:
        lineage_complete = bool(
            manifest.input_partition_refs
            and manifest.feature_spec_refs
            and manifest.label_pack_ref
            and manifest.state_artifact_refs
            and manifest.parent_artifact_refs
            and manifest.dataset_spec_ref
        )
        return QaCheckResult(
            check_name="lineage",
            status="passed" if lineage_complete else "failed",
            score=100.0 if lineage_complete else 0.0,
            metrics={
                "input_partition_refs": len(manifest.input_partition_refs),
                "state_artifact_refs": len(manifest.state_artifact_refs),
                "feature_spec_refs": len(manifest.feature_spec_refs),
                "label_pack_ref_present": bool(manifest.label_pack_ref),
                "dataset_spec_ref_present": bool(manifest.dataset_spec_ref),
                "parent_artifact_refs": len(manifest.parent_artifact_refs),
            },
            thresholds={
                "min_input_partition_refs": 1,
                "min_state_artifact_refs": 1,
                "min_feature_spec_refs": 1,
                "label_pack_ref_present": True,
                "dataset_spec_ref_present": True,
                "min_parent_artifact_refs": 1,
            },
            failure_reason="lineage manifest is incomplete" if not lineage_complete else "",
        )

    def _manifest_integrity_check(
        self,
        *,
        manifest: LineageManifest,
        expected_content_hash: str | None,
        manifest_path: Path | None,
    ) -> QaCheckResult:
        manifest_hash_matches = expected_content_hash is None or manifest.content_hash == expected_content_hash
        manifest_path_exists = manifest_path is None or manifest_path.exists()
        failed = not manifest_hash_matches or not manifest_path_exists
        return QaCheckResult(
            check_name="manifest_integrity",
            status="failed" if failed else "passed",
            score=0.0 if failed else 100.0,
            metrics={
                "manifest_content_hash": manifest.content_hash,
                "expected_content_hash": expected_content_hash or manifest.content_hash,
                "manifest_path": str(manifest_path) if manifest_path is not None else "",
                "manifest_path_exists": manifest_path_exists,
            },
            thresholds={"manifest_content_hash_matches": True, "manifest_path_exists": True},
            failure_reason="manifest content hash or persisted manifest path is inconsistent" if failed else "",
        )

    @staticmethod
    def _duplicate_primary_clock_rows(dataset_df: pl.DataFrame) -> int:
        required = {"symbol", "timeframe", "time_utc"}
        if not required.issubset(set(dataset_df.columns)):
            return 0
        unique_rows = dataset_df.select(["symbol", "timeframe", "time_utc"]).unique().height
        return max(0, len(dataset_df) - unique_rows)

    def _source_quality_metrics(
        self,
        *,
        spec: DatasetSpec,
        symbol: str,
        date_from: dt.date,
        date_to: dt.date,
        paths: StoragePaths,
        store: ParquetStore,
        state_df: pl.DataFrame,
    ) -> dict[str, object]:
        qa_rows: list[dict[str, object]] = []
        diagnostic_rows: list[dict[str, object]] = []
        requested_dates: list[str] = []
        current = date_from
        while current <= date_to:
            requested_dates.append(current.isoformat())
            qa_row = self._latest_daily_observability_row(store.read_dir(paths.merge_qa_dir(symbol, current)))
            if qa_row is not None:
                qa_rows.append(qa_row)
            diagnostic_row = self._latest_daily_observability_row(store.read_dir(paths.merge_diagnostics_dir(symbol, current)))
            if diagnostic_row is not None:
                diagnostic_rows.append(diagnostic_row)
            current += dt.timedelta(days=1)

        merge_df = pl.DataFrame(qa_rows) if qa_rows else pl.DataFrame()
        diagnostic_df = pl.DataFrame(diagnostic_rows) if diagnostic_rows else pl.DataFrame()
        observability_df = merge_df if not merge_df.is_empty() else diagnostic_df
        observability_source = "merge_qa" if not merge_df.is_empty() else "merge_diagnostics" if not diagnostic_df.is_empty() else "none"

        required_raw_broker_stats = self._required_raw_broker_stats(
            spec=spec,
            symbol=symbol,
            date_from=date_from,
            date_to=date_to,
            paths=paths,
            store=store,
        )
        requested_date_set = set(requested_dates)
        covered_by_broker = {
            broker_id: set(stats.get("covered_dates", []))
            for broker_id, stats in required_raw_broker_stats.items()
        }
        synchronized_dates = set.intersection(*covered_by_broker.values()) if covered_by_broker else set()
        asymmetric_dates = set.union(*covered_by_broker.values()) - synchronized_dates if covered_by_broker else set()
        missing_dates = {
            broker_id: sorted(requested_date_set - covered_dates)
            for broker_id, covered_dates in covered_by_broker.items()
        }

        metrics = {
            "required_raw_brokers": list(spec.required_raw_brokers),
            "merge_qa_days": float(merge_df.height),
            "merge_diagnostics_days": float(diagnostic_df.height),
            "merge_observability_source": observability_source,
            "dual_source_ratio_mean": float(merge_df["dual_source_ratio"].mean()) if "dual_source_ratio" in merge_df.columns else 0.0,
            "merge_conflict_mean": float(merge_df["conflicts"].mean()) if "conflicts" in merge_df.columns else 0.0,
            "diagnostic_dual_source_ratio_mean": (
                float(diagnostic_df["dual_source_ratio"].mean()) if "dual_source_ratio" in diagnostic_df.columns else 0.0
            ),
            "diagnostic_conflict_mean": (
                float(diagnostic_df["conflicts"].mean()) if "conflicts" in diagnostic_df.columns else 0.0
            ),
            "effective_observability_days": float(observability_df.height),
            "effective_dual_source_ratio_mean": (
                float(observability_df["dual_source_ratio"].mean()) if "dual_source_ratio" in observability_df.columns else 0.0
            ),
            "effective_conflict_mean": (
                float(observability_df["conflicts"].mean()) if "conflicts" in observability_df.columns else 0.0
            ),
            "dual_source_days": int(
                observability_df.filter(
                    (pl.col("dual_source_ratio") > 0.0)
                    | (pl.col("canonical_dual_rows") > 0)
                    | (pl.col("bucket_both") > 0)
                ).height
            ) if not observability_df.is_empty() and {"dual_source_ratio", "canonical_dual_rows", "bucket_both"}.issubset(set(observability_df.columns)) else 0,
            "bucket_both_total": int(observability_df["bucket_both"].sum()) if "bucket_both" in observability_df.columns else 0,
            "canonical_dual_rows_total": int(observability_df["canonical_dual_rows"].sum()) if "canonical_dual_rows" in observability_df.columns else 0,
            "required_raw_broker_stats": required_raw_broker_stats,
            "required_raw_missing_dates": missing_dates,
            "required_raw_asymmetric_dates": sorted(asymmetric_dates),
            "synchronized_raw_days": len(synchronized_dates),
            "synchronized_raw_coverage_ratio": (
                round(len(synchronized_dates) / len(requested_dates), 6) if requested_dates else 0.0
            ),
            "state_rows": float(len(state_df)),
            "state_quality_mean": float(state_df["quality_score"].mean()) if "quality_score" in state_df.columns and not state_df.is_empty() else 0.0,
            "state_conflict_rate": float(state_df["conflict_flag"].cast(pl.Float64).mean()) if "conflict_flag" in state_df.columns and not state_df.is_empty() else 0.0,
            "state_filled_ratio": float(
                state_df["trust_flags"].map_elements(
                    lambda flags: 1.0 if isinstance(flags, list) and "filled_gap" in flags else 0.0,
                    return_dtype=pl.Float64,
                ).mean()
            ) if "trust_flags" in state_df.columns and not state_df.is_empty() else 0.0,
        }
        metrics["requirement_failures"] = self._source_requirement_failures(spec, metrics)
        return metrics

    def _source_quality_score(self, metrics: dict[str, object]) -> float:
        state_quality_mean = float(metrics.get("state_quality_mean", 0.0) or 0.0)
        effective_observability_days = float(metrics.get("effective_observability_days", 0.0) or 0.0)
        effective_dual_source_ratio_mean = float(metrics.get("effective_dual_source_ratio_mean", 0.0) or 0.0)
        effective_conflict_mean = float(metrics.get("effective_conflict_mean", 0.0) or 0.0)
        synchronized_raw_coverage_ratio = float(metrics.get("synchronized_raw_coverage_ratio", 0.0) or 0.0)
        state_conflict_rate = float(metrics.get("state_conflict_rate", 0.0) or 0.0)
        state_filled_ratio = float(metrics.get("state_filled_ratio", 0.0) or 0.0)

        base = state_quality_mean if state_quality_mean > 0 else 70.0
        dual_source_component = min(max(effective_dual_source_ratio_mean, 0.0) * 100.0, 100.0)
        synchronized_coverage_component = min(max(synchronized_raw_coverage_ratio, 0.0) * 100.0, 100.0)
        score = base
        if effective_observability_days > 0:
            score += max(dual_source_component - 50.0, 0.0) * 0.10
            score += max(synchronized_coverage_component - 50.0, 0.0) * 0.05
            score -= min(max(effective_conflict_mean, 0.0) * 10.0, 10.0)

        if state_quality_mean <= 0:
            score -= min(max(state_conflict_rate, 0.0) * 20.0, 10.0)

        score -= min(max(state_filled_ratio, 0.0) * 35.0, 15.0)
        return round(max(0.0, min(100.0, score)), 4)

    @staticmethod
    def _latest_daily_observability_row(df: pl.DataFrame) -> dict[str, object] | None:
        if df.is_empty():
            return None
        sort_columns = [column for column in ["time_utc", "date"] if column in df.columns]
        if sort_columns:
            df = df.sort(sort_columns)
        return df.tail(1).row(0, named=True)

    def _required_raw_broker_stats(
        self,
        *,
        spec: DatasetSpec,
        symbol: str,
        date_from: dt.date,
        date_to: dt.date,
        paths: StoragePaths,
        store: ParquetStore,
    ) -> dict[str, dict[str, object]]:
        stats_by_broker: dict[str, dict[str, object]] = {}
        for broker_id in spec.required_raw_brokers:
            covered_dates: list[str] = []
            total_ticks_written = 0
            first_timestamp: dt.datetime | None = None
            last_timestamp: dt.datetime | None = None
            current = date_from
            while current <= date_to:
                day_df = store.read_dir(paths.raw_ticks_dir(broker_id, symbol, current))
                if not day_df.is_empty():
                    covered_dates.append(current.isoformat())
                    total_ticks_written += day_df.height
                    if "time_utc" in day_df.columns:
                        day_first = day_df["time_utc"].min()
                        day_last = day_df["time_utc"].max()
                        if first_timestamp is None or (day_first is not None and day_first < first_timestamp):
                            first_timestamp = day_first
                        if last_timestamp is None or (day_last is not None and day_last > last_timestamp):
                            last_timestamp = day_last
                current += dt.timedelta(days=1)

            stats_by_broker[broker_id] = {
                "days_requested": (date_to - date_from).days + 1,
                "days_written": len(covered_dates),
                "total_ticks_written": total_ticks_written,
                "covered_dates": covered_dates,
                "first_timestamp": first_timestamp.isoformat() if first_timestamp else "",
                "last_timestamp": last_timestamp.isoformat() if last_timestamp else "",
            }
        return stats_by_broker

    @staticmethod
    def _source_requirement_failures(spec: DatasetSpec, metrics: dict[str, object]) -> list[str]:
        failures: list[str] = []
        required_raw_brokers = [str(broker) for broker in metrics.get("required_raw_brokers", [])]

        if spec.require_synchronized_raw_coverage:
            missing_dates = {
                broker_id: [str(date) for date in dates]
                for broker_id, dates in dict(metrics.get("required_raw_missing_dates", {})).items()
                if dates
            }
            asymmetric_dates = [str(date) for date in metrics.get("required_raw_asymmetric_dates", [])]
            synchronized_raw_coverage_ratio = float(metrics.get("synchronized_raw_coverage_ratio", 0.0) or 0.0)

            if missing_dates:
                details = "; ".join(
                    f"{broker_id} missing {TruthService._format_name_sample(sorted(values))}"
                    for broker_id, values in sorted(missing_dates.items())
                )
                failures.append(f"synchronized raw coverage is incomplete for required brokers: {details}")
            if asymmetric_dates:
                failures.append(
                    "required raw coverage is asymmetric across brokers: "
                    f"{TruthService._format_name_sample(sorted(asymmetric_dates))}"
                )
            if required_raw_brokers and synchronized_raw_coverage_ratio < 1.0 and not missing_dates and not asymmetric_dates:
                failures.append(
                    f"synchronized raw coverage ratio is {synchronized_raw_coverage_ratio:.4f}, expected 1.0000"
                )

        if spec.require_dual_source_overlap:
            observability_source = str(metrics.get("merge_observability_source", "none"))
            bucket_both_total = int(metrics.get("bucket_both_total", 0) or 0)
            canonical_dual_rows_total = int(metrics.get("canonical_dual_rows_total", 0) or 0)
            dual_source_days = int(metrics.get("dual_source_days", 0) or 0)
            effective_dual_source_ratio_mean = float(metrics.get("effective_dual_source_ratio_mean", 0.0) or 0.0)
            if observability_source == "none":
                failures.append("dual-source overlap is required but no merge observability artifacts were found")
            if bucket_both_total <= 0 or canonical_dual_rows_total <= 0 or dual_source_days <= 0:
                failures.append(
                    "dual-source overlap is required but observed "
                    f"bucket_both_total={bucket_both_total}, "
                    f"canonical_dual_rows_total={canonical_dual_rows_total}, "
                    f"dual_source_days={dual_source_days}"
                )
            if effective_dual_source_ratio_mean < spec.min_dual_source_ratio:
                failures.append(
                    f"effective_dual_source_ratio_mean={effective_dual_source_ratio_mean:.6f} "
                    f"is below required min_dual_source_ratio={spec.min_dual_source_ratio:.6f}"
                )

        return failures

    def _dataset_quality_metrics(
        self,
        dataset_df: pl.DataFrame,
        quality: dict[str, object],
        feature_specs: list[FeatureSpec],
        label_pack: LabelPack,
    ) -> dict[str, object]:
        null_columns = {
            column: int(dataset_df[column].null_count())
            for column in dataset_df.columns
            if int(dataset_df[column].null_count()) > 0
        }
        constant_columns = sorted(str(column) for column in quality.get("constant_columns", []))
        feature_specs_by_column = self._feature_specs_by_output_column(feature_specs)
        label_columns = set(label_pack.output_columns)

        expected_sparse_null_columns: dict[str, dict[str, object]] = {}
        unexpected_null_columns: dict[str, dict[str, object]] = {}
        family_warning_summary: dict[str, list[str]] = {}

        for column, null_count in sorted(null_columns.items()):
            feature_spec = feature_specs_by_column.get(column)
            if feature_spec is None:
                if column in label_columns:
                    unexpected_null_columns[column] = {
                        "family": "label",
                        "null_count": null_count,
                        "reason": "label_nulls",
                    }
                    family_warning_summary.setdefault("label", []).append(
                        f"unexpected label nulls in {column}={null_count}"
                    )
                else:
                    unexpected_null_columns[column] = {
                        "family": "base",
                        "null_count": null_count,
                        "reason": "base_nulls",
                    }
                    family_warning_summary.setdefault("base", []).append(
                        f"unexpected base nulls in {column}={null_count}"
                    )
                continue

            if feature_spec.family == "htf_context":
                expected_sparse_null_columns[column] = {
                    "family": feature_spec.family,
                    "null_count": null_count,
                    "reason": "expected_alignment_sparsity",
                }
                family_warning_summary.setdefault(feature_spec.family, []).append(
                    f"expected alignment sparsity in {column}={null_count}"
                )
                continue

            if feature_spec.missingness_policy == "allow" and null_count <= int(feature_spec.warmup_rows):
                expected_sparse_null_columns[column] = {
                    "family": feature_spec.family,
                    "null_count": null_count,
                    "reason": "expected_warmup_sparsity",
                }
                family_warning_summary.setdefault(feature_spec.family, []).append(
                    f"expected warmup sparsity in {column}={null_count}"
                )
                continue

            unexpected_null_columns[column] = {
                "family": feature_spec.family,
                "null_count": null_count,
                "reason": "unexpected_feature_nulls",
            }
            family_warning_summary.setdefault(feature_spec.family, []).append(
                f"unexpected nulls in {column}={null_count}"
            )

        slice_trivial_constant_columns: dict[str, dict[str, object]] = {}
        blocking_constant_columns: dict[str, dict[str, object]] = {}
        feature_output_columns = {column for spec in feature_specs for column in spec.output_columns}

        for feature_spec in feature_specs:
            constant_outputs = sorted(set(feature_spec.output_columns).intersection(constant_columns))
            if not constant_outputs:
                continue
            target = slice_trivial_constant_columns
            reason = "slice_trivial_feature_constant"
            if len(constant_outputs) == len(feature_spec.output_columns):
                target = blocking_constant_columns
                reason = "all_feature_outputs_constant"
                family_warning_summary.setdefault(feature_spec.family, []).append(
                    f"all {len(constant_outputs)} feature outputs are constant on this slice"
                )
            else:
                family_warning_summary.setdefault(feature_spec.family, []).append(
                    f"{len(constant_outputs)} of {len(feature_spec.output_columns)} feature columns are slice-trivial constants"
                )
            for column in constant_outputs:
                target[column] = {
                    "family": feature_spec.family,
                    "reason": reason,
                }

        for horizon in label_pack.horizons_minutes:
            token = f"_{horizon}m"
            horizon_columns = [column for column in label_pack.output_columns if token in column]
            constant_horizon_columns = sorted(set(horizon_columns).intersection(constant_columns))
            if not constant_horizon_columns:
                continue
            target = slice_trivial_constant_columns
            reason = "slice_trivial_label_constant"
            if len(constant_horizon_columns) == len(horizon_columns):
                target = blocking_constant_columns
                reason = "all_label_horizon_columns_constant"
                family_warning_summary.setdefault("label", []).append(
                    f"all label columns for horizon {horizon}m are constant"
                )
            else:
                family_warning_summary.setdefault("label", []).append(
                    f"{len(constant_horizon_columns)} label columns are slice-trivial constants at {horizon}m"
                )
            for column in constant_horizon_columns:
                target[column] = {
                    "family": "label",
                    "reason": reason,
                }

        for column in constant_columns:
            if column in feature_output_columns or column in label_columns:
                continue
            target = slice_trivial_constant_columns
            reason = "slice_trivial_base_constant"
            if column in self.CRITICAL_BASE_CONSTANT_COLUMNS:
                target = blocking_constant_columns
                reason = "critical_base_column_constant"
                family_warning_summary.setdefault("base", []).append(
                    f"critical base column {column} is constant"
                )
            else:
                family_warning_summary.setdefault("base", []).append(
                    f"slice-trivial base/source column {column} is constant"
                )
            target[column] = {
                "family": "base",
                "reason": reason,
            }

        family_warning_summary = {
            family: sorted(set(notes))
            for family, notes in family_warning_summary.items()
            if notes
        }

        return {
            "quality_score": float(quality.get("quality_score", 0.0)),
            "total_nulls": int(quality.get("total_nulls", 0)),
            "null_columns": null_columns,
            "constant_columns": constant_columns,
            "duplicate_timestamps": int(quality.get("duplicate_timestamps", 0)),
            "expected_sparse_null_columns": expected_sparse_null_columns,
            "unexpected_null_columns": unexpected_null_columns,
            "slice_trivial_constant_columns": slice_trivial_constant_columns,
            "blocking_constant_columns": blocking_constant_columns,
            "family_warning_summary": family_warning_summary,
        }

    @staticmethod
    def _feature_specs_by_output_column(feature_specs: list[FeatureSpec]) -> dict[str, FeatureSpec]:
        mapping: dict[str, FeatureSpec] = {}
        for spec in feature_specs:
            for column in spec.output_columns:
                mapping[column] = spec
        return mapping

    @staticmethod
    def _group_feature_specs_by_family(feature_specs: list[FeatureSpec]) -> dict[str, list[FeatureSpec]]:
        grouped: dict[str, list[FeatureSpec]] = {}
        for spec in feature_specs:
            grouped.setdefault(spec.family, []).append(spec)
        return grouped

    def _family_missingness_threshold(self, family: str, specs: list[FeatureSpec]) -> float:
        if family in self.FAMILY_MISSINGNESS_THRESHOLDS:
            return self.FAMILY_MISSINGNESS_THRESHOLDS[family]
        if any(spec.missingness_policy in {"allow"} for spec in specs):
            return 100.0
        return 0.0

    @staticmethod
    def _ordered_split_names(split_frames: dict[str, pl.DataFrame]) -> list[str]:
        if {"train", "val", "test"}.issubset(set(split_frames)):
            return ["train", "val", "test"]
        return sorted(split_frames)

    def _threshold_shortfalls(
        self,
        *,
        trust_score_total: float,
        coverage_score: float,
        leakage_score: float,
        feature_quality_score: float,
        label_quality_score: float,
        source_quality_score: float,
        lineage_score: float,
    ) -> list[str]:
        shortfalls: list[str] = []
        if trust_score_total < self.MIN_TOTAL_SCORE:
            shortfalls.append(f"trust total {trust_score_total:.2f} is below minimum {self.MIN_TOTAL_SCORE:.2f}")
        if coverage_score < self.MIN_COVERAGE_SCORE:
            shortfalls.append(f"coverage {coverage_score:.2f} is below minimum {self.MIN_COVERAGE_SCORE:.2f}")
        if leakage_score < 100.0:
            shortfalls.append(f"leakage score {leakage_score:.2f} must equal 100.00")
        if feature_quality_score < self.MIN_FEATURE_QUALITY_SCORE:
            shortfalls.append(
                f"feature quality {feature_quality_score:.2f} is below minimum {self.MIN_FEATURE_QUALITY_SCORE:.2f}"
            )
        if label_quality_score < self.MIN_LABEL_QUALITY_SCORE:
            shortfalls.append(f"label quality {label_quality_score:.2f} is below minimum {self.MIN_LABEL_QUALITY_SCORE:.2f}")
        if source_quality_score < self.MIN_SOURCE_QUALITY_SCORE:
            shortfalls.append(
                f"source quality {source_quality_score:.2f} is below minimum {self.MIN_SOURCE_QUALITY_SCORE:.2f}"
            )
        if lineage_score < 100.0:
            shortfalls.append(f"lineage score {lineage_score:.2f} must equal 100.00")
        return shortfalls

    @staticmethod
    def _rejection_reasons(failed_checks: list[QaCheckResult], threshold_shortfalls: list[str]) -> list[str]:
        reasons = [
            f"{check.check_name}: {check.failure_reason or 'check failed'}"
            for check in failed_checks
        ]
        reasons.extend(threshold_shortfalls)
        return reasons

    def _warning_reasons(
        self,
        warning_checks: list[QaCheckResult],
        warning_codes: list[str],
        *,
        dataset_quality_metrics: dict[str, object],
    ) -> list[str]:
        reasons = []
        warning_check_names = {check.check_name for check in warning_checks}
        reasons.extend(
            f"{check.check_name}: {self._warning_detail_from_check(check)}"
            for check in warning_checks
        )
        reasons.extend(
            self._warning_detail_from_code(
                code,
                dataset_quality_metrics=dataset_quality_metrics,
                warning_check_names=warning_check_names,
            )
            for code in warning_codes
        )
        return sorted({reason for reason in reasons if reason})

    def _warning_detail_from_check(self, check: QaCheckResult) -> str:
        if check.failure_reason:
            return check.failure_reason
        if check.check_name == "feature_family_missingness":
            warning_families = check.metrics.get("warning_families", [])
            if warning_families:
                return f"families approaching threshold: {', '.join(sorted(warning_families))}"
        if check.check_name == "label_quality":
            label_nulls = int(check.metrics.get("label_nulls", 0))
            return f"label nulls remain after purge: {label_nulls}"
        if check.check_name == "source_quality":
            merge_observability_source = str(check.metrics.get("merge_observability_source", "none"))
            observability_detail = (
                (
                    f"merge_qa_days={float(check.metrics.get('merge_qa_days', 0.0)):.0f}, "
                    f"dual_source_ratio_mean={float(check.metrics.get('dual_source_ratio_mean', 0.0)):.4f}"
                )
                if merge_observability_source == "merge_qa"
                else (
                    f"merge_diagnostics_days={float(check.metrics.get('merge_diagnostics_days', 0.0)):.0f}, "
                    f"diagnostic_dual_source_ratio_mean={float(check.metrics.get('diagnostic_dual_source_ratio_mean', 0.0)):.4f}"
                    if merge_observability_source == "merge_diagnostics"
                    else "no merge observability artifacts found"
                )
            )
            return (
                f"score {check.score:.2f} is below preferred {self.PREFERRED_SOURCE_QUALITY_SCORE:.2f}; "
                f"{observability_detail}; "
                f"synchronized_raw_coverage_ratio={float(check.metrics.get('synchronized_raw_coverage_ratio', 0.0)):.4f}; "
                f"state_quality_mean={float(check.metrics.get('state_quality_mean', 0.0)):.2f}, "
                f"state_filled_ratio={float(check.metrics.get('state_filled_ratio', 0.0)):.4f}"
            )
        return "warning threshold reached"

    @staticmethod
    def _warning_detail_from_code(
        code: str,
        *,
        dataset_quality_metrics: dict[str, object],
        warning_check_names: set[str],
    ) -> str:
        if code == "dataset_contains_nulls":
            unexpected_null_columns = {
                str(name): int(details.get("null_count", 0))
                for name, details in dict(dataset_quality_metrics.get("unexpected_null_columns", {})).items()
            }
            expected_sparse_null_columns = dict(dataset_quality_metrics.get("expected_sparse_null_columns", {}))
            if unexpected_null_columns:
                return f"unexpected nulls remain in: {TruthService._format_column_metric_sample(unexpected_null_columns)}"
            if expected_sparse_null_columns:
                family_summary = TruthService._format_family_note_sample(
                    dict(dataset_quality_metrics.get("family_warning_summary", {})),
                    families={"htf_context", "event_shape", "entropy", "multiscale"},
                )
                return f"expected sparse nulls remain: {family_summary}"
            return "dataset contains null values"
        if code == "dataset_contains_constant_columns":
            blocking_constant_columns = {
                str(name): str(details.get("reason", "blocking_constant"))
                for name, details in dict(dataset_quality_metrics.get("blocking_constant_columns", {})).items()
            }
            slice_trivial_constant_columns = dict(dataset_quality_metrics.get("slice_trivial_constant_columns", {}))
            if blocking_constant_columns:
                return (
                    "blocking constant columns remain: "
                    f"{TruthService._format_name_sample(sorted(blocking_constant_columns))}"
                )
            if slice_trivial_constant_columns:
                family_summary = TruthService._format_family_note_sample(
                    dict(dataset_quality_metrics.get("family_warning_summary", {})),
                    families={"base", "quality", "disagreement", "label"},
                )
                return f"slice-trivial constants remain: {family_summary}"
            return "dataset contains constant columns"
        if code == "source_quality_warning" and "source_quality" in warning_check_names:
            return ""
        if code == "label_quality_warning" and "label_quality" in warning_check_names:
            return ""
        if code == "feature_family_missingness_warning" and "feature_family_missingness" in warning_check_names:
            return ""
        messages = {
            "source_quality_warning": "source quality is acceptable but below preferred research comfort",
            "label_quality_warning": "label columns contain null values",
            "feature_family_missingness_warning": "one or more feature families is approaching its missingness threshold",
        }
        return messages.get(code, code)

    @staticmethod
    def _format_column_metric_sample(metrics: dict[str, int], *, limit: int = 5) -> str:
        if not metrics:
            return "-"
        ordered_items = sorted(metrics.items())
        visible = [f"{name}={value}" for name, value in ordered_items[:limit]]
        remainder = len(ordered_items) - len(visible)
        if remainder > 0:
            visible.append(f"+{remainder} more")
        return ", ".join(visible)

    @staticmethod
    def _format_name_sample(values: list[str], *, limit: int = 5) -> str:
        if not values:
            return "-"
        ordered = sorted(values)
        visible = ordered[:limit]
        remainder = len(ordered) - len(visible)
        if remainder > 0:
            visible.append(f"+{remainder} more")
        return ", ".join(visible)

    @staticmethod
    def _format_family_note_sample(
        family_notes: dict[str, list[str]],
        *,
        families: set[str] | None = None,
        limit: int = 4,
    ) -> str:
        selected_items = [
            (family, notes)
            for family, notes in sorted(family_notes.items())
            if not families or family in families
        ]
        if not selected_items:
            return "-"
        visible = [f"{family}: {notes[0]}" for family, notes in selected_items[:limit] if notes]
        remainder = len(selected_items) - len(visible)
        if remainder > 0:
            visible.append(f"+{remainder} more families")
        return "; ".join(visible)

    def _quality_caveat_summary(
        self,
        *,
        dataset_quality_metrics: dict[str, object],
        source_check: QaCheckResult,
    ) -> dict[str, object]:
        accepted_caveats: list[str] = []
        green_blockers: list[str] = []
        publication_blockers: list[str] = []

        expected_sparse_null_columns = dict(dataset_quality_metrics.get("expected_sparse_null_columns", {}))
        unexpected_null_columns = dict(dataset_quality_metrics.get("unexpected_null_columns", {}))
        slice_trivial_constant_columns = dict(dataset_quality_metrics.get("slice_trivial_constant_columns", {}))
        blocking_constant_columns = dict(dataset_quality_metrics.get("blocking_constant_columns", {}))
        family_warning_summary = dict(dataset_quality_metrics.get("family_warning_summary", {}))

        if expected_sparse_null_columns:
            accepted_caveats.append(
                "expected sparse nulls: "
                + self._format_family_note_sample(
                    family_warning_summary,
                    families={"htf_context", "event_shape", "entropy", "multiscale"},
                )
            )
        if slice_trivial_constant_columns:
            accepted_caveats.append(
                "slice-trivial constants: "
                + self._format_family_note_sample(
                    family_warning_summary,
                    families={"base", "quality", "disagreement", "label"},
                )
            )
        if unexpected_null_columns:
            green_blockers.append(
                "unexpected nulls remain: "
                + self._format_column_metric_sample(
                    {name: int(details.get("null_count", 0)) for name, details in unexpected_null_columns.items()}
                )
            )
        if blocking_constant_columns:
            green_blockers.append(
                "blocking constant columns remain: "
                + self._format_name_sample(sorted(str(name) for name in blocking_constant_columns))
            )
        if source_check.status == "warning":
            green_blockers.append(f"source_quality below preferred threshold ({source_check.score:.2f} < {self.PREFERRED_SOURCE_QUALITY_SCORE:.2f})")
        elif source_check.status == "failed":
            publication_blockers.append(
                f"source_quality below publication threshold ({source_check.score:.2f} < {self.MIN_SOURCE_QUALITY_SCORE:.2f})"
            )

        return {
            "accepted_caveats": accepted_caveats,
            "green_blockers": green_blockers,
            "publication_blockers": publication_blockers,
            "family_warning_summary": family_warning_summary,
        }

    @staticmethod
    def _decision_summary(
        *,
        accepted: bool,
        score_breakdown: dict[str, float],
        rejection_reasons: list[str],
        warning_reasons: list[str],
    ) -> str:
        if accepted:
            if warning_reasons:
                return (
                    f"accepted for publication with warnings; total={score_breakdown['total']:.2f}, "
                    f"warnings={len(warning_reasons)}"
                )
            return f"accepted for publication; total={score_breakdown['total']:.2f}"
        if rejection_reasons:
            return f"rejected for publication; {rejection_reasons[0]}"
        return f"rejected for publication; total={score_breakdown['total']:.2f}"
