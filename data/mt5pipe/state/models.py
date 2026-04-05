"""Typed models for state artifacts and rolling state windows."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from mt5pipe.contracts.state import parse_window_size


def _ensure_utc(ts: dt.datetime, field_name: str) -> None:
    if ts.tzinfo is None or ts.utcoffset() != dt.timedelta(0):
        raise ValueError(f"{field_name} must be UTC-aware")


class StateSnapshot(BaseModel):
    """Canonical machine-readable market state at a point in time."""

    schema_version: str = Field(default="2.0.0")
    state_version: str
    snapshot_id: str
    symbol: str
    ts_utc: dt.datetime
    ts_msc: int
    clock: str
    window_start_utc: dt.datetime
    window_end_utc: dt.datetime
    bid: float
    ask: float
    mid: float
    spread: float
    source_primary: str
    source_secondary: str | None = None
    source_count: int
    merge_mode: Literal["single", "best", "conflict"]
    conflict_flag: bool
    disagreement_bps: float | None = None
    spread_disagreement_bps: float | None = None
    broker_a_mid: float | None = None
    broker_b_mid: float | None = None
    broker_a_spread: float | None = None
    broker_b_spread: float | None = None
    primary_staleness_ms: int = 0
    secondary_staleness_ms: int | None = None
    source_offset_ms: int | None = None
    expected_interval_ms: int = 0
    observed_interval_ms: int = 0
    quality_score: float
    source_quality_hint: float | None = None
    source_participation_score: float | None = None
    overlap_confidence_hint: float | None = None
    expected_observations: int = 1
    observed_observations: int = 1
    missing_observations: int = 0
    window_completeness: float = 1.0
    gap_fill_flag: bool = False
    session_code: Literal["asia", "london", "ny", "overlap", "weekend_closed", "other"]
    event_flags: list[str] = Field(default_factory=list)
    trust_flags: list[str] = Field(default_factory=list)
    provenance_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_snapshot(self) -> "StateSnapshot":
        _ensure_utc(self.ts_utc, "ts_utc")
        _ensure_utc(self.window_start_utc, "window_start_utc")
        _ensure_utc(self.window_end_utc, "window_end_utc")
        if self.bid <= 0 or self.ask <= 0:
            raise ValueError("bid and ask must be positive")
        if self.bid > self.ask:
            raise ValueError("bid must be <= ask")
        if self.source_count < 0:
            raise ValueError("source_count must be >= 0")
        if self.window_start_utc > self.window_end_utc:
            raise ValueError("window_start_utc must be <= window_end_utc")
        if not (self.window_start_utc <= self.ts_utc <= self.window_end_utc):
            raise ValueError("ts_utc must lie within the snapshot window")
        expected_mid = (self.bid + self.ask) / 2.0
        expected_spread = self.ask - self.bid
        if abs(self.mid - expected_mid) > 1e-9:
            raise ValueError("mid must equal (bid + ask) / 2")
        if abs(self.spread - expected_spread) > 1e-9:
            raise ValueError("spread must equal ask - bid")
        if not (0.0 <= self.quality_score <= 100.0):
            raise ValueError("quality_score must be within [0, 100]")
        if self.source_quality_hint is not None and not (0.0 <= self.source_quality_hint <= 100.0):
            raise ValueError("source_quality_hint must be within [0, 100]")
        if self.primary_staleness_ms < 0:
            raise ValueError("primary_staleness_ms must be >= 0")
        if self.secondary_staleness_ms is not None and self.secondary_staleness_ms < 0:
            raise ValueError("secondary_staleness_ms must be >= 0 when provided")
        if self.expected_interval_ms < 0:
            raise ValueError("expected_interval_ms must be >= 0")
        if self.observed_interval_ms < 0:
            raise ValueError("observed_interval_ms must be >= 0")
        if self.expected_observations < 1:
            raise ValueError("expected_observations must be >= 1")
        if self.observed_observations < 0:
            raise ValueError("observed_observations must be >= 0")
        if self.missing_observations < 0:
            raise ValueError("missing_observations must be >= 0")
        if self.observed_observations > self.expected_observations:
            raise ValueError("observed_observations must be <= expected_observations")
        if self.missing_observations != (self.expected_observations - self.observed_observations):
            raise ValueError("missing_observations must equal expected_observations - observed_observations")
        if not (0.0 <= self.window_completeness <= 1.0):
            raise ValueError("window_completeness must be within [0, 1]")
        if self.source_participation_score is not None and not (0.0 <= self.source_participation_score <= 1.0):
            raise ValueError("source_participation_score must be within [0, 1]")
        if self.overlap_confidence_hint is not None and not (0.0 <= self.overlap_confidence_hint <= 1.0):
            raise ValueError("overlap_confidence_hint must be within [0, 1]")
        return self


class StateCoverageSummary(BaseModel):
    """Coverage and gap summary for a state artifact or window artifact."""

    coverage_mode: Literal["regular_clock", "activity_clock"]
    resolution_ms: int
    row_count: int
    expected_rows: int
    missing_rows: int
    completeness_ratio: float
    filled_row_count: int = 0
    filled_ratio: float = 0.0
    gap_count: int = 0
    max_gap_ms: int = 0
    observed_span_ms: int = 0
    time_range_start_utc: dt.datetime | None = None
    time_range_end_utc: dt.datetime | None = None

    @model_validator(mode="after")
    def validate_summary(self) -> "StateCoverageSummary":
        if self.resolution_ms <= 0:
            raise ValueError("resolution_ms must be > 0")
        if self.row_count < 0:
            raise ValueError("row_count must be >= 0")
        if self.expected_rows < 0:
            raise ValueError("expected_rows must be >= 0")
        if self.missing_rows < 0:
            raise ValueError("missing_rows must be >= 0")
        if self.filled_row_count < 0:
            raise ValueError("filled_row_count must be >= 0")
        if self.gap_count < 0:
            raise ValueError("gap_count must be >= 0")
        if self.max_gap_ms < 0:
            raise ValueError("max_gap_ms must be >= 0")
        if self.observed_span_ms < 0:
            raise ValueError("observed_span_ms must be >= 0")
        if not (0.0 <= self.completeness_ratio <= 1.0):
            raise ValueError("completeness_ratio must be within [0, 1]")
        if not (0.0 <= self.filled_ratio <= 1.0):
            raise ValueError("filled_ratio must be within [0, 1]")
        if self.time_range_start_utc is not None:
            _ensure_utc(self.time_range_start_utc, "time_range_start_utc")
        if self.time_range_end_utc is not None:
            _ensure_utc(self.time_range_end_utc, "time_range_end_utc")
        if self.time_range_start_utc is not None and self.time_range_end_utc is not None:
            if self.time_range_start_utc > self.time_range_end_utc:
                raise ValueError("time_range_start_utc must be <= time_range_end_utc")
        return self


class StateSourceQualitySummary(BaseModel):
    """Source participation and quality summary for a state artifact."""

    mean_source_count: float
    dual_source_ratio: float
    conflict_ratio: float
    mean_quality_score: float
    min_quality_score: float
    mean_source_quality_hint: float | None = None
    mean_source_participation_score: float | None = None
    mean_overlap_confidence: float | None = None
    median_primary_staleness_ms: float = 0.0
    p95_primary_staleness_ms: float = 0.0
    max_primary_staleness_ms: int = 0

    @model_validator(mode="after")
    def validate_summary(self) -> "StateSourceQualitySummary":
        if not (0.0 <= self.dual_source_ratio <= 1.0):
            raise ValueError("dual_source_ratio must be within [0, 1]")
        if not (0.0 <= self.conflict_ratio <= 1.0):
            raise ValueError("conflict_ratio must be within [0, 1]")
        if not (0.0 <= self.mean_quality_score <= 100.0):
            raise ValueError("mean_quality_score must be within [0, 100]")
        if not (0.0 <= self.min_quality_score <= 100.0):
            raise ValueError("min_quality_score must be within [0, 100]")
        if self.mean_source_quality_hint is not None and not (0.0 <= self.mean_source_quality_hint <= 100.0):
            raise ValueError("mean_source_quality_hint must be within [0, 100]")
        if self.mean_source_participation_score is not None and not (0.0 <= self.mean_source_participation_score <= 1.0):
            raise ValueError("mean_source_participation_score must be within [0, 1]")
        if self.mean_overlap_confidence is not None and not (0.0 <= self.mean_overlap_confidence <= 1.0):
            raise ValueError("mean_overlap_confidence must be within [0, 1]")
        if self.median_primary_staleness_ms < 0 or self.p95_primary_staleness_ms < 0 or self.max_primary_staleness_ms < 0:
            raise ValueError("staleness summary values must be >= 0")
        return self


class StateReadinessSummary(BaseModel):
    """Range-level training-readiness summary for state or state-window artifacts."""

    interval_count: int
    effective_observation_count: int
    effective_coverage_ratio: float
    ready_interval_count: int
    ready_interval_ratio: float
    gap_heavy_interval_count: int
    gap_heavy_interval_ratio: float
    low_overlap_interval_ratio: float
    low_quality_interval_ratio: float
    source_reliability_band: Literal["low", "medium", "high"]
    overlap_quality_band: Literal["low", "medium", "high"]
    gap_burden_band: Literal["low", "medium", "high"]
    readiness_band: Literal["not_ready", "limited", "ready"]
    eligible_anchor_count: int | None = None
    available_window_count: int | None = None
    missing_window_count: int | None = None
    available_window_ratio: float | None = None
    full_window_ratio: float | None = None
    partial_window_ratio: float | None = None

    @model_validator(mode="after")
    def validate_summary(self) -> "StateReadinessSummary":
        if self.interval_count < 0:
            raise ValueError("interval_count must be >= 0")
        if self.effective_observation_count < 0:
            raise ValueError("effective_observation_count must be >= 0")
        if self.ready_interval_count < 0:
            raise ValueError("ready_interval_count must be >= 0")
        if self.gap_heavy_interval_count < 0:
            raise ValueError("gap_heavy_interval_count must be >= 0")
        if self.ready_interval_count > self.interval_count:
            raise ValueError("ready_interval_count must be <= interval_count")
        if self.gap_heavy_interval_count > self.interval_count:
            raise ValueError("gap_heavy_interval_count must be <= interval_count")
        for field_name in (
            "effective_coverage_ratio",
            "ready_interval_ratio",
            "gap_heavy_interval_ratio",
            "low_overlap_interval_ratio",
            "low_quality_interval_ratio",
            "available_window_ratio",
            "full_window_ratio",
            "partial_window_ratio",
        ):
            value = getattr(self, field_name)
            if value is not None and not (0.0 <= value <= 1.0):
                raise ValueError(f"{field_name} must be within [0, 1]")
        if self.eligible_anchor_count is not None and self.eligible_anchor_count < 0:
            raise ValueError("eligible_anchor_count must be >= 0")
        if self.available_window_count is not None and self.available_window_count < 0:
            raise ValueError("available_window_count must be >= 0")
        if self.missing_window_count is not None and self.missing_window_count < 0:
            raise ValueError("missing_window_count must be >= 0")
        if self.eligible_anchor_count is not None and self.available_window_count is not None:
            if self.available_window_count > self.eligible_anchor_count:
                raise ValueError("available_window_count must be <= eligible_anchor_count")
        if self.eligible_anchor_count is not None and self.missing_window_count is not None:
            if self.missing_window_count > self.eligible_anchor_count:
                raise ValueError("missing_window_count must be <= eligible_anchor_count")
        return self


class StateIntervalReadinessSummary(BaseModel):
    """Daily or session-scoped readiness rollup for a state artifact."""

    interval_kind: Literal["day", "session"]
    interval_key: str
    date: dt.date | None = None
    session_code: Literal["asia", "london", "ny", "overlap", "weekend_closed", "other"] | None = None
    interval_count: int
    effective_coverage_ratio: float
    filled_ratio: float
    gap_burden_ratio: float
    mean_quality_score: float
    mean_source_quality_hint: float | None = None
    mean_source_participation_score: float | None = None
    mean_overlap_confidence: float | None = None
    ready_interval_count: int
    ready_interval_ratio: float
    gap_heavy_interval_count: int
    source_reliability_band: Literal["low", "medium", "high"]
    overlap_quality_band: Literal["low", "medium", "high"]
    gap_burden_band: Literal["low", "medium", "high"]
    readiness_band: Literal["not_ready", "limited", "ready"]

    @model_validator(mode="after")
    def validate_rollup(self) -> "StateIntervalReadinessSummary":
        if not self.interval_key:
            raise ValueError("interval_key must not be empty")
        if self.interval_count < 0:
            raise ValueError("interval_count must be >= 0")
        if self.ready_interval_count < 0:
            raise ValueError("ready_interval_count must be >= 0")
        if self.gap_heavy_interval_count < 0:
            raise ValueError("gap_heavy_interval_count must be >= 0")
        if self.ready_interval_count > self.interval_count:
            raise ValueError("ready_interval_count must be <= interval_count")
        if self.gap_heavy_interval_count > self.interval_count:
            raise ValueError("gap_heavy_interval_count must be <= interval_count")
        for field_name in (
            "effective_coverage_ratio",
            "filled_ratio",
            "gap_burden_ratio",
            "ready_interval_ratio",
        ):
            value = getattr(self, field_name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{field_name} must be within [0, 1]")
        if not (0.0 <= self.mean_quality_score <= 100.0):
            raise ValueError("mean_quality_score must be within [0, 100]")
        if self.mean_source_quality_hint is not None and not (0.0 <= self.mean_source_quality_hint <= 100.0):
            raise ValueError("mean_source_quality_hint must be within [0, 100]")
        if self.mean_source_participation_score is not None and not (0.0 <= self.mean_source_participation_score <= 1.0):
            raise ValueError("mean_source_participation_score must be within [0, 1]")
        if self.mean_overlap_confidence is not None and not (0.0 <= self.mean_overlap_confidence <= 1.0):
            raise ValueError("mean_overlap_confidence must be within [0, 1]")
        return self


class StateArtifactManifest(BaseModel):
    """State-sector artifact manifest with compiler-compatible core fields."""

    schema_version: str = Field(default="1.1.0")
    manifest_id: str
    artifact_id: str
    artifact_kind: Literal["state", "state_window"]
    logical_name: str
    logical_version: str
    artifact_uri: str
    content_hash: str
    build_id: str
    created_at: dt.datetime
    status: Literal["building", "accepted", "rejected", "published"] = "accepted"
    dataset_spec_ref: str | None = None
    state_artifact_refs: list[str] = Field(default_factory=list)
    feature_spec_refs: list[str] = Field(default_factory=list)
    label_pack_ref: str | None = None
    truth_report_ref: str | None = None
    code_version: str
    merge_config_ref: str | None = None
    input_partition_refs: list[str] = Field(default_factory=list)
    parent_artifact_refs: list[str] = Field(default_factory=list)
    symbol: str
    clock: str
    window_size: str | None = None
    time_range_start_utc: dt.datetime | None = None
    time_range_end_utc: dt.datetime | None = None
    coverage_summary: StateCoverageSummary | None = None
    source_quality_summary: StateSourceQualitySummary | None = None
    readiness_summary: StateReadinessSummary | None = None
    daily_readiness_rollups: list[StateIntervalReadinessSummary] = Field(default_factory=list)
    session_readiness_rollups: list[StateIntervalReadinessSummary] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_manifest(self) -> "StateArtifactManifest":
        _ensure_utc(self.created_at, "created_at")
        if not self.content_hash:
            raise ValueError("content_hash must not be empty")
        if not self.input_partition_refs:
            raise ValueError("input_partition_refs must not be empty")
        if self.time_range_start_utc is not None:
            _ensure_utc(self.time_range_start_utc, "time_range_start_utc")
        if self.time_range_end_utc is not None:
            _ensure_utc(self.time_range_end_utc, "time_range_end_utc")
        if self.time_range_start_utc is not None and self.time_range_end_utc is not None:
            if self.time_range_start_utc > self.time_range_end_utc:
                raise ValueError("time_range_start_utc must be <= time_range_end_utc")
        if self.window_size is not None:
            parse_window_size(self.window_size)
        return self


class StateWindowRecord(BaseModel):
    """Machine-native rolling state window anchored at a PIT-safe timestamp."""

    schema_version: str = Field(default="1.0.0")
    state_version: str
    window_id: str
    symbol: str
    clock: str
    anchor_ts_utc: dt.datetime
    anchor_ts_msc: int
    window_size: str
    window_start_utc: dt.datetime
    window_end_utc: dt.datetime
    row_count: int
    expected_row_count: int
    missing_row_count: int
    warmup_missing_rows: int
    warmup_satisfied: bool
    completeness: float
    coverage_mode: Literal["regular_clock", "activity_clock"]
    observed_span_ms: int
    source_count_mean: float
    dual_source_ratio_window: float
    quality_score_mean: float
    source_quality_hint_mean: float | None = None
    source_participation_score_mean: float | None = None
    overlap_confidence_mean: float | None = None
    conflict_count_window: int
    conflict_ratio: float
    disagreement_bps_mean: float | None = None
    staleness_ms_max: int | None = None
    filled_row_count: int = 0
    filled_ratio: float = 0.0
    gap_count: int = 0
    max_gap_ms: int = 0
    mid_values: list[float] = Field(default_factory=list)
    spread_values: list[float] = Field(default_factory=list)
    mid_return_bps_values: list[float] = Field(default_factory=list)
    source_count_values: list[int] = Field(default_factory=list)
    quality_score_values: list[float] = Field(default_factory=list)
    disagreement_bps_values: list[float | None] = Field(default_factory=list)
    staleness_ms_values: list[int | None] = Field(default_factory=list)
    conflict_flags: list[bool] = Field(default_factory=list)
    source_offset_ms_values: list[int | None] = Field(default_factory=list)
    provenance_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_window(self) -> "StateWindowRecord":
        _ensure_utc(self.anchor_ts_utc, "anchor_ts_utc")
        _ensure_utc(self.window_start_utc, "window_start_utc")
        _ensure_utc(self.window_end_utc, "window_end_utc")
        parse_window_size(self.window_size)
        if self.row_count < 0:
            raise ValueError("row_count must be >= 0")
        if self.expected_row_count < 1:
            raise ValueError("expected_row_count must be >= 1")
        if self.missing_row_count < 0:
            raise ValueError("missing_row_count must be >= 0")
        if self.warmup_missing_rows < 0:
            raise ValueError("warmup_missing_rows must be >= 0")
        if self.missing_row_count > self.expected_row_count:
            raise ValueError("missing_row_count must be <= expected_row_count")
        if self.warmup_missing_rows > self.expected_row_count:
            raise ValueError("warmup_missing_rows must be <= expected_row_count")
        if self.observed_span_ms < 0:
            raise ValueError("observed_span_ms must be >= 0")
        if not (0.0 <= self.completeness <= 1.0):
            raise ValueError("completeness must be within [0, 1]")
        if not (0.0 <= self.conflict_ratio <= 1.0):
            raise ValueError("conflict_ratio must be within [0, 1]")
        if not (0.0 <= self.dual_source_ratio_window <= 1.0):
            raise ValueError("dual_source_ratio_window must be within [0, 1]")
        if self.source_quality_hint_mean is not None and not (0.0 <= self.source_quality_hint_mean <= 100.0):
            raise ValueError("source_quality_hint_mean must be within [0, 100]")
        if self.source_participation_score_mean is not None and not (0.0 <= self.source_participation_score_mean <= 1.0):
            raise ValueError("source_participation_score_mean must be within [0, 1]")
        if self.overlap_confidence_mean is not None and not (0.0 <= self.overlap_confidence_mean <= 1.0):
            raise ValueError("overlap_confidence_mean must be within [0, 1]")
        if not (0.0 <= self.filled_ratio <= 1.0):
            raise ValueError("filled_ratio must be within [0, 1]")
        if self.filled_row_count < 0 or self.gap_count < 0 or self.max_gap_ms < 0:
            raise ValueError("filled/gap summary values must be >= 0")
        if self.expected_row_count > 0:
            expected_completeness = (self.expected_row_count - self.missing_row_count) / self.expected_row_count
            if abs(self.completeness - expected_completeness) > 1e-9:
                raise ValueError("completeness must match expected_row_count and missing_row_count")
        if self.warmup_satisfied != (self.warmup_missing_rows == 0):
            raise ValueError("warmup_satisfied must match warmup_missing_rows")

        series_lengths = {
            len(self.mid_values),
            len(self.spread_values),
            len(self.mid_return_bps_values),
            len(self.source_count_values),
            len(self.quality_score_values),
            len(self.disagreement_bps_values),
            len(self.staleness_ms_values),
            len(self.conflict_flags),
            len(self.source_offset_ms_values),
        }
        if series_lengths != {self.row_count}:
            raise ValueError("All machine-native state window series must align to row_count")
        return self
