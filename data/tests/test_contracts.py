"""Tests for Phase 1 Dataset OS contracts."""

from __future__ import annotations

import datetime as dt

import pytest

from mt5pipe.contracts import StateArtifactRef, StateWindowArtifactRef, StateWindowRequest, TickArtifactRef, parse_window_size
from mt5pipe.compiler.models import DatasetSpec, LineageManifest
from mt5pipe.features.registry.models import FeatureSpec
from mt5pipe.labels.registry.models import LabelPack
from mt5pipe.state.models import (
    StateCoverageSummary,
    StateIntervalReadinessSummary,
    StateReadinessSummary,
    StateSnapshot,
    StateSourceQualitySummary,
    StateWindowRecord,
)
from mt5pipe.truth.models import QaCheckResult, TrustReport


UTC = dt.timezone.utc


def test_state_snapshot_validates_core_invariants() -> None:
    snapshot = StateSnapshot(
        state_version="1.0.0",
        snapshot_id="state:XAUUSD:M1:1",
        symbol="XAUUSD",
        ts_utc=dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC),
        ts_msc=1775001600000,
        clock="M1",
        window_start_utc=dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC),
        window_end_utc=dt.datetime(2026, 4, 1, 0, 0, 59, tzinfo=UTC),
        bid=3000.0,
        ask=3000.2,
        mid=3000.1,
        spread=0.2,
        source_primary="broker_a",
        source_count=2,
        merge_mode="best",
        conflict_flag=False,
        expected_interval_ms=60_000,
        observed_interval_ms=0,
        quality_score=92.0,
        source_quality_hint=92.0,
        source_participation_score=0.95,
        overlap_confidence_hint=0.93,
        session_code="asia",
        provenance_refs=["canonical://XAUUSD/2026-04-01"],
    )
    assert snapshot.mid == 3000.1


def test_state_refs_and_window_request_validate_shape() -> None:
    tick_ref = TickArtifactRef(
        artifact_id="canonical_tick.XAUUSD.abc123",
        logical_name="XAUUSD",
        version="1.0.0",
        content_hash="abc123",
        symbol=" xauusd ",
        date_from=dt.date(2026, 4, 1),
        date_to=dt.date(2026, 4, 2),
    )
    state_ref = StateArtifactRef(
        artifact_id="state.XAUUSD.M1.abc123",
        logical_name="XAUUSD.M1",
        version="state.default@1.0.0",
        content_hash="abc123",
        symbol="xauusd",
        clock="m1",
        state_version="state.default@1.0.0",
        date_from=dt.date(2026, 4, 1),
        date_to=dt.date(2026, 4, 2),
    )
    window_ref = StateWindowArtifactRef(
        artifact_id="state_window.XAUUSD.M1.5m.abc123",
        logical_name="XAUUSD.M1.5m",
        version="state.default@1.0.0",
        content_hash="abc123",
        symbol=" xauusd ",
        clock="m1",
        state_version="state.default@1.0.0",
        window_size=" 5M ",
        date_from=dt.date(2026, 4, 1),
        date_to=dt.date(2026, 4, 2),
        source_artifact_id=state_ref.artifact_id,
    )
    request = StateWindowRequest(
        symbol="xauusd",
        clock="m1",
        state_version=" state.default@1.0.0 ",
        date_from=dt.date(2026, 4, 1),
        date_to=dt.date(2026, 4, 2),
        window_sizes=[" 30S ", "60S", "5M"],
    )

    assert tick_ref.kind.value == "canonical_tick"
    assert tick_ref.symbol == "XAUUSD"
    assert state_ref.kind.value == "state"
    assert state_ref.symbol == "XAUUSD"
    assert state_ref.clock == "M1"
    assert window_ref.kind.value == "state_window"
    assert window_ref.window_size == "5m"
    assert parse_window_size("5m") == dt.timedelta(minutes=5)
    assert request.symbol == "XAUUSD"
    assert request.clock == "M1"
    assert request.window_sizes == ["30s", "60s", "5m"]


def test_state_window_record_validates_machine_native_series_alignment() -> None:
    anchor = dt.datetime(2026, 4, 1, 0, 5, tzinfo=UTC)
    record = StateWindowRecord(
        state_version="state.default@1.0.0",
        window_id="state-window:XAUUSD:M1:5m:1",
        symbol="XAUUSD",
        clock="M1",
        anchor_ts_utc=anchor,
        anchor_ts_msc=int(anchor.timestamp() * 1000),
        window_size="5m",
        window_start_utc=dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC),
        window_end_utc=anchor,
        row_count=5,
        expected_row_count=5,
        missing_row_count=0,
        warmup_missing_rows=0,
        warmup_satisfied=True,
        completeness=1.0,
        coverage_mode="regular_clock",
        observed_span_ms=240_000,
        source_count_mean=1.8,
        dual_source_ratio_window=0.8,
        quality_score_mean=90.0,
        source_quality_hint_mean=90.0,
        source_participation_score_mean=0.82,
        overlap_confidence_mean=0.79,
        conflict_count_window=1,
        conflict_ratio=0.2,
        disagreement_bps_mean=0.5,
        staleness_ms_max=60_000,
        filled_row_count=0,
        filled_ratio=0.0,
        gap_count=0,
        max_gap_ms=0,
        mid_values=[1.0, 1.1, 1.2, 1.3, 1.4],
        spread_values=[0.1, 0.1, 0.1, 0.1, 0.1],
        mid_return_bps_values=[0.0, 10.0, 10.0, 10.0, 10.0],
        source_count_values=[2, 2, 2, 1, 2],
        quality_score_values=[90.0, 91.0, 89.0, 90.0, 90.0],
        disagreement_bps_values=[0.4, 0.5, 0.6, None, 0.5],
        staleness_ms_values=[0, 60_000, 60_000, 60_000, 60_000],
        conflict_flags=[False, False, True, False, False],
        source_offset_ms_values=[None, None, None, None, None],
        provenance_refs=["state://XAUUSD/M1"],
    )
    assert record.completeness == 1.0

    with pytest.raises(ValueError):
        StateWindowRecord(
            state_version="state.default@1.0.0",
            window_id="state-window:XAUUSD:M1:5m:bad",
            symbol="XAUUSD",
            clock="M1",
            anchor_ts_utc=anchor,
            anchor_ts_msc=int(anchor.timestamp() * 1000),
            window_size="5m",
            window_start_utc=dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC),
            window_end_utc=anchor,
            row_count=2,
            expected_row_count=5,
            missing_row_count=3,
            warmup_missing_rows=3,
            warmup_satisfied=False,
            completeness=0.4,
            coverage_mode="regular_clock",
            observed_span_ms=60_000,
            source_count_mean=1.0,
            dual_source_ratio_window=0.0,
            quality_score_mean=90.0,
            conflict_count_window=0,
            conflict_ratio=0.0,
            mid_values=[1.0],
            spread_values=[0.1, 0.1],
            mid_return_bps_values=[0.0, 10.0],
            source_count_values=[1, 1],
            quality_score_values=[90.0, 90.0],
            disagreement_bps_values=[None, None],
            staleness_ms_values=[0, 60_000],
            conflict_flags=[False, False],
            source_offset_ms_values=[None, None],
            provenance_refs=["state://XAUUSD/M1"],
        )


def test_state_coverage_and_source_quality_summaries_validate() -> None:
    coverage = StateCoverageSummary(
        coverage_mode="regular_clock",
        resolution_ms=60_000,
        row_count=10,
        expected_rows=12,
        missing_rows=2,
        completeness_ratio=10 / 12,
        filled_row_count=2,
        filled_ratio=0.2,
        gap_count=1,
        max_gap_ms=120_000,
        observed_span_ms=540_000,
        time_range_start_utc=dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC),
        time_range_end_utc=dt.datetime(2026, 4, 1, 0, 9, tzinfo=UTC),
    )
    source = StateSourceQualitySummary(
        mean_source_count=1.7,
        dual_source_ratio=0.7,
        conflict_ratio=0.1,
        mean_quality_score=88.0,
        min_quality_score=50.0,
        mean_source_quality_hint=80.0,
        mean_source_participation_score=0.72,
        mean_overlap_confidence=0.66,
        median_primary_staleness_ms=60_000,
        p95_primary_staleness_ms=120_000,
        max_primary_staleness_ms=180_000,
    )

    assert coverage.gap_count == 1
    assert source.dual_source_ratio == 0.7


def test_state_readiness_and_interval_rollups_validate() -> None:
    readiness = StateReadinessSummary(
        interval_count=10,
        effective_observation_count=9,
        effective_coverage_ratio=0.9,
        ready_interval_count=8,
        ready_interval_ratio=0.8,
        gap_heavy_interval_count=1,
        gap_heavy_interval_ratio=0.1,
        low_overlap_interval_ratio=0.2,
        low_quality_interval_ratio=0.1,
        source_reliability_band="medium",
        overlap_quality_band="medium",
        gap_burden_band="medium",
        readiness_band="ready",
        eligible_anchor_count=10,
        available_window_count=8,
        missing_window_count=2,
        available_window_ratio=0.8,
        full_window_ratio=0.75,
        partial_window_ratio=0.25,
    )
    rollup = StateIntervalReadinessSummary(
        interval_kind="day",
        interval_key="2026-04-01",
        date=dt.date(2026, 4, 1),
        interval_count=5,
        effective_coverage_ratio=0.92,
        filled_ratio=0.08,
        gap_burden_ratio=0.2,
        mean_quality_score=81.0,
        mean_source_quality_hint=79.0,
        mean_source_participation_score=0.45,
        mean_overlap_confidence=0.22,
        ready_interval_count=4,
        ready_interval_ratio=0.8,
        gap_heavy_interval_count=1,
        source_reliability_band="medium",
        overlap_quality_band="medium",
        gap_burden_band="high",
        readiness_band="ready",
    )

    assert readiness.available_window_ratio == 0.8
    assert rollup.interval_kind == "day"


def test_feature_spec_requires_unique_output_columns() -> None:
    with pytest.raises(ValueError):
        FeatureSpec(
            feature_name="bad",
            family="quality",
            version="1.0.0",
            input_contract="BuiltBar",
            input_clock="M1",
            output_clock="M1",
            builder_ref="x:y",
            output_columns=["dup", "dup"],
            dependencies=["close"],
            missingness_policy="fail",
            qa_policy_ref="qa.feature.default@1.0.0",
        )


def test_label_pack_requires_purge_ge_max_horizon() -> None:
    with pytest.raises(ValueError):
        LabelPack(
            label_pack_name="bad_pack",
            version="1.0.0",
            base_clock="M1",
            horizons_minutes=[5, 60],
            generator_refs=["labels.future_return"],
            purge_rows=10,
            output_columns=["future_return_5m"],
        )


def test_dataset_spec_validates_ratios_and_walk_forward_requirements() -> None:
    spec = DatasetSpec(
        dataset_name="xau_core",
        version="1.0.0",
        symbols=["XAUUSD"],
        date_from=dt.date(2026, 4, 1),
        date_to=dt.date(2026, 4, 2),
        base_clock="M1",
        state_version_ref="state.default@1.0.0",
        feature_selectors=["time/*"],
        label_pack_ref="core_tb_volscaled@1.0.0",
        split_policy="temporal_holdout",
        embargo_rows=240,
        truth_policy_ref="truth.default@1.0.0",
    )
    assert spec.key == "xau_core@1.0.0"

    with pytest.raises(ValueError):
        DatasetSpec(
            dataset_name="xau_core",
            version="1.0.0",
            symbols=["XAUUSD"],
            date_from=dt.date(2026, 4, 1),
            date_to=dt.date(2026, 4, 2),
            base_clock="M1",
            state_version_ref="state.default@1.0.0",
            feature_selectors=["time/*"],
            label_pack_ref="core_tb_volscaled@1.0.0",
            split_policy="walk_forward",
            embargo_rows=240,
            truth_policy_ref="truth.default@1.0.0",
        )


def test_lineage_manifest_requires_inputs() -> None:
    with pytest.raises(ValueError):
        LineageManifest(
            manifest_id="manifest.1",
            artifact_id="artifact.1",
            artifact_kind="dataset",
            logical_name="xau_core",
            logical_version="1.0.0",
            artifact_uri="data/datasets/x",
            content_hash="abc",
            build_id="build.1",
            created_at=dt.datetime.now(UTC),
            status="building",
            code_version="workspace-local-no-git",
        )


def test_lineage_manifest_accepts_truth_pending_status() -> None:
    manifest = LineageManifest(
        manifest_id="manifest.truth_pending.1",
        artifact_id="artifact.truth_pending.1",
        artifact_kind="dataset",
        logical_name="xau_core",
        logical_version="1.0.0",
        artifact_uri="data/datasets/x",
        content_hash="abc123",
        build_id="build.1",
        created_at=dt.datetime.now(UTC),
        status="truth_pending",
        code_version="workspace-local-no-git",
        input_partition_refs=["data/bars/symbol=XAUUSD/timeframe=M1/date=2026-04-01"],
    )
    assert manifest.status == "truth_pending"


def test_trust_report_rejects_publishable_hard_failures() -> None:
    with pytest.raises(ValueError):
        TrustReport(
            report_id="trust.1",
            artifact_id="artifact.1",
            artifact_kind="dataset",
            truth_policy_version="truth.default@1.0.0",
            status="rejected",
            accepted_for_publication=True,
            trust_score_total=50.0,
            coverage_score=50.0,
            leakage_score=100.0,
            feature_quality_score=60.0,
            label_quality_score=60.0,
            source_quality_score=60.0,
            lineage_score=100.0,
            hard_failures=["dataset_empty"],
            generated_at=dt.datetime.now(UTC),
            checks=[
                QaCheckResult(
                    check_name="coverage",
                    status="failed",
                    score=0.0,
                    failure_reason="empty",
                )
            ],
        )
