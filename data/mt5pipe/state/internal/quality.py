"""Coverage, readiness, and source-quality summaries for state artifacts."""

from __future__ import annotations

import datetime as dt
import math

import polars as pl

from mt5pipe.state.internal.bar_support import timeframe_to_seconds
from mt5pipe.state.models import (
    StateCoverageSummary,
    StateIntervalReadinessSummary,
    StateReadinessSummary,
    StateSourceQualitySummary,
)


def state_resolution_ms(clock: str, *, tick_resolution_ms: int = 1_000) -> int:
    """Return the expected observation resolution for a state clock."""
    if clock.lower() == "tick":
        return tick_resolution_ms
    return timeframe_to_seconds(clock) * 1_000


def coverage_mode_for_clock(clock: str) -> str:
    return "activity_clock" if clock.lower() == "tick" else "regular_clock"


def snapshot_source_participation_score(
    *,
    source_count: int,
    conflict_flag: bool,
    disagreement_bps: float | None,
    gap_fill_flag: bool,
    quality_score: float,
) -> float:
    """Normalized source participation quality for a single state snapshot."""
    base = 0.15 if source_count <= 0 else 0.55 if source_count == 1 else 1.0
    conflict_penalty = 0.35 if conflict_flag else 0.0
    disagreement_penalty = min((disagreement_bps or 0.0) / 25.0, 0.25)
    fill_penalty = 0.45 if gap_fill_flag else 0.0
    quality_modifier = max(0.0, min(quality_score / 100.0, 1.0))
    return max(0.0, min(1.0, (base - conflict_penalty - disagreement_penalty - fill_penalty) * quality_modifier))


def snapshot_overlap_confidence_hint(
    *,
    source_count: int,
    source_participation_score: float,
    window_completeness: float,
    source_quality_hint: float | None,
) -> float:
    """Normalized overlap-confidence hint for a single state snapshot."""
    dual_bonus = 1.0 if source_count >= 2 else 0.55 if source_count == 1 else 0.20
    quality_modifier = max(0.0, min((source_quality_hint if source_quality_hint is not None else 50.0) / 100.0, 1.0))
    confidence = dual_bonus * source_participation_score * max(window_completeness, 0.0) * quality_modifier
    return max(0.0, min(1.0, confidence))


def build_state_coverage_summary(state_df: pl.DataFrame, *, clock: str) -> StateCoverageSummary:
    """Summarize state coverage and gap behavior for a state frame."""
    resolution_ms = state_resolution_ms(clock)
    coverage_mode = coverage_mode_for_clock(clock)
    if state_df.is_empty():
        return StateCoverageSummary(
            coverage_mode=coverage_mode,
            resolution_ms=resolution_ms,
            row_count=0,
            expected_rows=0,
            missing_rows=0,
            completeness_ratio=0.0,
            filled_row_count=0,
            filled_ratio=0.0,
            gap_count=0,
            max_gap_ms=0,
            observed_span_ms=0,
        )

    ts_values = _timestamp_values_ms(state_df)
    time_values = _time_values(state_df)
    observed_span_ms = max(ts_values[-1] - ts_values[0], 0) if len(ts_values) > 1 else 0
    filled_flags = _filled_flags(state_df)
    filled_row_count = _filled_row_count(state_df, filled_flags)
    filled_ratio = _filled_ratio(state_df, filled_row_count)

    if coverage_mode == "regular_clock":
        expected_rows = max(1, math.floor(observed_span_ms / resolution_ms) + 1)
        missing_rows = max(expected_rows - state_df.height, 0)
        gap_count = _gap_count(state_df, filled_flags)
        max_gap_ms = _max_gap_ms(state_df, filled_flags, resolution_ms)
    else:
        expected_rows = state_df.height
        missing_rows = 0
        staleness_values = _staleness_values(state_df)
        gap_events = [value for value in staleness_values if value > resolution_ms]
        gap_count = len(gap_events)
        max_gap_ms = max(gap_events, default=0)

    completeness_ratio = ((expected_rows - missing_rows) / expected_rows) if expected_rows > 0 else 0.0
    return StateCoverageSummary(
        coverage_mode=coverage_mode,
        resolution_ms=resolution_ms,
        row_count=state_df.height,
        expected_rows=expected_rows,
        missing_rows=missing_rows,
        completeness_ratio=completeness_ratio,
        filled_row_count=filled_row_count,
        filled_ratio=filled_ratio,
        gap_count=gap_count,
        max_gap_ms=max_gap_ms,
        observed_span_ms=observed_span_ms,
        time_range_start_utc=time_values[0],
        time_range_end_utc=time_values[-1],
    )


def build_state_source_quality_summary(state_df: pl.DataFrame) -> StateSourceQualitySummary:
    """Summarize source participation and quality hints for a state frame."""
    if state_df.is_empty():
        return StateSourceQualitySummary(
            mean_source_count=0.0,
            dual_source_ratio=0.0,
            conflict_ratio=0.0,
            mean_quality_score=0.0,
            min_quality_score=0.0,
            mean_source_quality_hint=0.0,
            mean_source_participation_score=0.0,
            mean_overlap_confidence=0.0,
            median_primary_staleness_ms=0.0,
            p95_primary_staleness_ms=0.0,
            max_primary_staleness_ms=0,
        )

    source_counts = _source_count_values(state_df)
    quality_scores = _quality_score_values(state_df)
    quality_hints = _source_quality_hint_values(state_df)
    participation_scores = _source_participation_values(state_df)
    overlap_confidences = _overlap_confidence_values(state_df)
    conflicts = _conflict_flags(state_df)
    staleness_values = _staleness_values(state_df)

    return StateSourceQualitySummary(
        mean_source_count=_mean(source_counts),
        dual_source_ratio=_dual_source_ratio(state_df, source_counts),
        conflict_ratio=_ratio(sum(1 for value in conflicts if value), len(conflicts)),
        mean_quality_score=_mean(quality_scores),
        min_quality_score=min(quality_scores) if quality_scores else 0.0,
        mean_source_quality_hint=_mean_optional(quality_hints),
        mean_source_participation_score=_mean_optional(participation_scores),
        mean_overlap_confidence=_mean_optional(overlap_confidences),
        median_primary_staleness_ms=_percentile(staleness_values, 0.5),
        p95_primary_staleness_ms=_percentile(staleness_values, 0.95),
        max_primary_staleness_ms=max(staleness_values, default=0),
    )


def build_state_readiness_summary(
    state_df: pl.DataFrame,
    *,
    clock: str,
    eligible_anchor_count: int | None = None,
) -> StateReadinessSummary:
    """Summarize whether a state frame looks training-ready over its materialized range."""
    metrics = _row_readiness_metrics(state_df, clock=clock)
    if not metrics:
        return StateReadinessSummary(
            interval_count=0,
            effective_observation_count=0,
            effective_coverage_ratio=0.0,
            ready_interval_count=0,
            ready_interval_ratio=0.0,
            gap_heavy_interval_count=0,
            gap_heavy_interval_ratio=0.0,
            low_overlap_interval_ratio=0.0,
            low_quality_interval_ratio=0.0,
            source_reliability_band="low",
            overlap_quality_band="low",
            gap_burden_band="high",
            readiness_band="not_ready",
            eligible_anchor_count=eligible_anchor_count,
            available_window_count=0 if eligible_anchor_count is not None else None,
            missing_window_count=eligible_anchor_count if eligible_anchor_count is not None else None,
            available_window_ratio=0.0 if eligible_anchor_count is not None else None,
            full_window_ratio=0.0 if eligible_anchor_count is not None else None,
            partial_window_ratio=0.0 if eligible_anchor_count is not None else None,
        )

    interval_count = len(metrics)
    ready_interval_count = sum(1 for metric in metrics if metric["ready_flag"])
    gap_heavy_interval_count = sum(1 for metric in metrics if metric["gap_heavy_flag"])
    low_overlap_interval_count = sum(1 for metric in metrics if metric["low_overlap_flag"])
    low_quality_interval_count = sum(1 for metric in metrics if metric["low_quality_flag"])
    mean_quality_score = _mean([metric["quality_score"] for metric in metrics])
    mean_participation = _mean([metric["source_participation_score"] for metric in metrics])
    mean_overlap = _mean([metric["overlap_confidence"] for metric in metrics])
    effective_observation_count = sum(metric["effective_observation_count"] for metric in metrics)
    effective_coverage_ratio = _mean([metric["effective_coverage_ratio"] for metric in metrics])
    source_reliability_band = _source_reliability_band(mean_quality_score, mean_participation)
    overlap_quality_band = _overlap_quality_band(mean_overlap)
    gap_burden_ratio = _ratio(gap_heavy_interval_count, interval_count)
    gap_burden_band = _gap_burden_band(gap_burden_ratio)
    ready_interval_ratio = _ratio(ready_interval_count, interval_count)
    readiness_band = _readiness_band(
        effective_coverage_ratio=effective_coverage_ratio,
        ready_interval_ratio=ready_interval_ratio,
        source_reliability_band=source_reliability_band,
        gap_burden_ratio=gap_burden_ratio,
    )

    available_window_count = interval_count if eligible_anchor_count is not None else None
    missing_window_count = (
        max(eligible_anchor_count - interval_count, 0) if eligible_anchor_count is not None else None
    )
    available_window_ratio = (
        _ratio(available_window_count or 0, eligible_anchor_count)
        if eligible_anchor_count is not None
        else None
    )
    full_window_ratio = (
        _ratio(sum(1 for metric in metrics if metric["window_full_flag"]), interval_count)
        if eligible_anchor_count is not None
        else None
    )
    partial_window_ratio = (
        _ratio(sum(1 for metric in metrics if metric["window_partial_flag"]), interval_count)
        if eligible_anchor_count is not None
        else None
    )

    return StateReadinessSummary(
        interval_count=interval_count,
        effective_observation_count=effective_observation_count,
        effective_coverage_ratio=effective_coverage_ratio,
        ready_interval_count=ready_interval_count,
        ready_interval_ratio=ready_interval_ratio,
        gap_heavy_interval_count=gap_heavy_interval_count,
        gap_heavy_interval_ratio=gap_burden_ratio,
        low_overlap_interval_ratio=_ratio(low_overlap_interval_count, interval_count),
        low_quality_interval_ratio=_ratio(low_quality_interval_count, interval_count),
        source_reliability_band=source_reliability_band,
        overlap_quality_band=overlap_quality_band,
        gap_burden_band=gap_burden_band,
        readiness_band=readiness_band,
        eligible_anchor_count=eligible_anchor_count,
        available_window_count=available_window_count,
        missing_window_count=missing_window_count,
        available_window_ratio=available_window_ratio,
        full_window_ratio=full_window_ratio,
        partial_window_ratio=partial_window_ratio,
    )


def build_state_interval_readiness_rollups(
    state_df: pl.DataFrame,
    *,
    clock: str,
    interval_kind: str,
) -> list[StateIntervalReadinessSummary]:
    """Build daily or session-scoped readiness rollups from a state frame."""
    if interval_kind not in {"day", "session"}:
        raise ValueError("interval_kind must be 'day' or 'session'")

    metrics = _row_readiness_metrics(state_df, clock=clock)
    if not metrics:
        return []

    grouped: dict[tuple[str, dt.date | None, str | None], list[dict[str, float | int | bool | str | dt.datetime | dt.date | None]]] = {}
    for metric in metrics:
        metric_date = metric["date"]
        metric_session = metric["session_code"]
        if interval_kind == "day":
            key = (str(metric_date), metric_date, None)
        else:
            key = (f"{metric_date.isoformat()}:{metric_session}", metric_date, metric_session)
        grouped.setdefault(key, []).append(metric)

    rollups: list[StateIntervalReadinessSummary] = []
    for interval_key, values in sorted(grouped.items(), key=lambda item: item[0][0]):
        key_value, date_value, session_value = interval_key
        interval_count = len(values)
        gap_heavy_count = sum(1 for value in values if value["gap_heavy_flag"])
        ready_count = sum(1 for value in values if value["ready_flag"])
        effective_coverage_ratio = _mean([float(value["effective_coverage_ratio"]) for value in values])
        mean_quality_score = _mean([float(value["quality_score"]) for value in values])
        mean_source_quality_hint = _mean_optional([float(value["source_quality_hint"]) for value in values])
        mean_source_participation = _mean_optional([float(value["source_participation_score"]) for value in values])
        mean_overlap = _mean_optional([float(value["overlap_confidence"]) for value in values])
        source_band = _source_reliability_band(mean_quality_score, mean_source_participation or 0.0)
        overlap_band = _overlap_quality_band(mean_overlap or 0.0)
        gap_burden_ratio = _ratio(gap_heavy_count, interval_count)
        gap_burden_band = _gap_burden_band(gap_burden_ratio)
        ready_ratio = _ratio(ready_count, interval_count)

        rollups.append(
            StateIntervalReadinessSummary(
                interval_kind=interval_kind,
                interval_key=key_value,
                date=date_value,
                session_code=session_value,
                interval_count=interval_count,
                effective_coverage_ratio=effective_coverage_ratio,
                filled_ratio=_mean([float(value["filled_ratio"]) for value in values]),
                gap_burden_ratio=gap_burden_ratio,
                mean_quality_score=mean_quality_score,
                mean_source_quality_hint=mean_source_quality_hint,
                mean_source_participation_score=mean_source_participation,
                mean_overlap_confidence=mean_overlap,
                ready_interval_count=ready_count,
                ready_interval_ratio=ready_ratio,
                gap_heavy_interval_count=gap_heavy_count,
                source_reliability_band=source_band,
                overlap_quality_band=overlap_band,
                gap_burden_band=gap_burden_band,
                readiness_band=_readiness_band(
                    effective_coverage_ratio=effective_coverage_ratio,
                    ready_interval_ratio=ready_ratio,
                    source_reliability_band=source_band,
                    gap_burden_ratio=gap_burden_ratio,
                ),
            )
        )
    return rollups


def _filled_flags(state_df: pl.DataFrame) -> list[bool]:
    if "filled_row_count" in state_df.columns:
        return [int(value or 0) > 0 for value in state_df["filled_row_count"].fill_null(0).to_list()]
    if "gap_fill_flag" in state_df.columns:
        return [bool(value) for value in state_df["gap_fill_flag"].to_list()]
    if "trust_flags" not in state_df.columns:
        return [False] * state_df.height
    flags = state_df["trust_flags"].to_list()
    return ["filled_gap" in (value or []) for value in flags]


def _staleness_values(state_df: pl.DataFrame) -> list[int]:
    if "observed_interval_ms" in state_df.columns:
        return [int(value) for value in state_df["observed_interval_ms"].fill_null(0).to_list()]
    if "primary_staleness_ms" in state_df.columns:
        return [int(value) for value in state_df["primary_staleness_ms"].fill_null(0).to_list()]
    if "staleness_ms_max" in state_df.columns:
        return [int(value) for value in state_df["staleness_ms_max"].fill_null(0).to_list()]
    return [0] * state_df.height


def _timestamp_values_ms(state_df: pl.DataFrame) -> list[int]:
    if "ts_msc" in state_df.columns:
        return [int(value) for value in state_df["ts_msc"].to_list()]
    if "anchor_ts_msc" in state_df.columns:
        return [int(value) for value in state_df["anchor_ts_msc"].to_list()]
    raise KeyError("State frame must contain ts_msc or anchor_ts_msc")


def _time_values(state_df: pl.DataFrame) -> list[object]:
    if "ts_utc" in state_df.columns:
        return state_df["ts_utc"].to_list()
    if "anchor_ts_utc" in state_df.columns:
        return state_df["anchor_ts_utc"].to_list()
    raise KeyError("State frame must contain ts_utc or anchor_ts_utc")


def _filled_row_count(state_df: pl.DataFrame, filled_flags: list[bool]) -> int:
    if "filled_row_count" in state_df.columns:
        return sum(int(value or 0) for value in state_df["filled_row_count"].fill_null(0).to_list())
    return sum(1 for flag in filled_flags if flag)


def _filled_ratio(state_df: pl.DataFrame, filled_row_count: int) -> float:
    if "row_count" in state_df.columns:
        total_rows = sum(int(value or 0) for value in state_df["row_count"].fill_null(0).to_list())
        return float(filled_row_count / total_rows) if total_rows > 0 else 0.0
    return float(filled_row_count / state_df.height) if state_df.height > 0 else 0.0


def _gap_count(state_df: pl.DataFrame, filled_flags: list[bool]) -> int:
    if "gap_count" in state_df.columns:
        return sum(int(value or 0) for value in state_df["gap_count"].fill_null(0).to_list())
    return _run_count(filled_flags)


def _max_gap_ms(state_df: pl.DataFrame, filled_flags: list[bool], resolution_ms: int) -> int:
    if "max_gap_ms" in state_df.columns:
        return max((int(value or 0) for value in state_df["max_gap_ms"].fill_null(0).to_list()), default=0)
    return _max_gap_run_ms(filled_flags, resolution_ms)


def _source_count_values(df: pl.DataFrame) -> list[float]:
    if "source_count" in df.columns:
        return _numeric_list(df, "source_count")
    if "source_count_mean" in df.columns:
        return _numeric_list(df, "source_count_mean")
    return []


def _quality_score_values(df: pl.DataFrame) -> list[float]:
    if "quality_score" in df.columns:
        return _numeric_list(df, "quality_score")
    if "quality_score_mean" in df.columns:
        return _numeric_list(df, "quality_score_mean")
    return []


def _source_quality_hint_values(df: pl.DataFrame) -> list[float | None]:
    if "source_quality_hint" in df.columns:
        return _optional_numeric_list(df, "source_quality_hint")
    if "source_quality_hint_mean" in df.columns:
        return _optional_numeric_list(df, "source_quality_hint_mean")
    return []


def _source_participation_values(df: pl.DataFrame) -> list[float | None]:
    if "source_participation_score" in df.columns:
        return _optional_numeric_list(df, "source_participation_score")
    if "source_participation_score_mean" in df.columns:
        return _optional_numeric_list(df, "source_participation_score_mean")
    return []


def _overlap_confidence_values(df: pl.DataFrame) -> list[float | None]:
    if "overlap_confidence_hint" in df.columns:
        return _optional_numeric_list(df, "overlap_confidence_hint")
    if "overlap_confidence_mean" in df.columns:
        return _optional_numeric_list(df, "overlap_confidence_mean")
    return []


def _conflict_flags(df: pl.DataFrame) -> list[bool]:
    if "conflict_flag" in df.columns:
        return _bool_list(df, "conflict_flag")
    if "conflict_count_window" in df.columns:
        return [int(value or 0) > 0 for value in df["conflict_count_window"].fill_null(0).to_list()]
    return [False] * df.height


def _dual_source_ratio(df: pl.DataFrame, source_counts: list[float]) -> float:
    if "dual_source_ratio_window" in df.columns:
        ratios = _numeric_list(df, "dual_source_ratio_window")
        return _mean(ratios)
    return _ratio(sum(1 for value in source_counts if value >= 2), len(source_counts))


def _run_count(flags: list[bool]) -> int:
    count = 0
    in_run = False
    for flag in flags:
        if flag and not in_run:
            count += 1
            in_run = True
        elif not flag:
            in_run = False
    return count


def _max_gap_run_ms(flags: list[bool], resolution_ms: int) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best * resolution_ms


def _numeric_list(df: pl.DataFrame, column: str) -> list[float]:
    if column not in df.columns:
        return []
    return [float(value) for value in df[column].fill_null(0).to_list()]


def _optional_numeric_list(df: pl.DataFrame, column: str) -> list[float | None]:
    if column not in df.columns:
        return []
    result: list[float | None] = []
    for value in df[column].to_list():
        result.append(None if value is None else float(value))
    return result


def _bool_list(df: pl.DataFrame, column: str) -> list[bool]:
    if column not in df.columns:
        return [False] * df.height
    return [bool(value) for value in df[column].fill_null(False).to_list()]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _mean_optional(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def _ratio(true_count: int | object, total: int) -> float:
    count = int(true_count) if not isinstance(true_count, int) else true_count
    if total <= 0:
        return 0.0
    return float(count / total)


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    series = pl.Series("values", values)
    quantile = series.quantile(q)
    return float(quantile) if quantile is not None else 0.0


def _row_readiness_metrics(
    state_df: pl.DataFrame,
    *,
    clock: str,
) -> list[dict[str, float | int | bool | str | dt.date | None]]:
    if state_df.is_empty():
        return []

    resolution_ms = state_resolution_ms(clock)
    metrics: list[dict[str, float | int | bool | str | dt.date | None]] = []
    for row in state_df.to_dicts():
        timestamp = _timestamp_from_row(row)
        metric_date = timestamp.date()
        session = _session_code_from_row(row, timestamp)
        is_window = "window_id" in row or "anchor_ts_utc" in row

        if is_window:
            row_count = max(int(row.get("row_count", 0) or 0), 0)
            expected_row_count = max(int(row.get("expected_row_count", row_count or 1) or 1), 1)
            filled_row_count = max(int(row.get("filled_row_count", 0) or 0), 0)
            completeness = max(0.0, min(float(row.get("completeness", 0.0) or 0.0), 1.0))
            effective_observation_count = max(row_count - filled_row_count, 0)
            effective_coverage_ratio = max(0.0, min(effective_observation_count / expected_row_count, 1.0))
            quality_score = _coalesce_numeric(row, "source_quality_hint_mean", "quality_score_mean")
            participation = _coalesce_numeric(
                row,
                "source_participation_score_mean",
                default=_fallback_participation_from_source_count(row.get("source_count_mean"), quality_score),
            )
            overlap = _coalesce_numeric(row, "overlap_confidence_mean", default=0.0)
            filled_ratio = max(0.0, min(float(row.get("filled_ratio", 0.0) or 0.0), 1.0))
            gap_count = max(int(row.get("gap_count", 0) or 0), 0)
            max_gap_ms = max(int(row.get("max_gap_ms", 0) or 0), 0)
            warmup_satisfied = bool(row.get("warmup_satisfied", False))
            gap_heavy_flag = filled_ratio > 0.15 or gap_count > 1 or max_gap_ms > (resolution_ms * 5)
            ready_flag = (
                warmup_satisfied
                and completeness >= 0.90
                and effective_coverage_ratio >= 0.85
                and quality_score >= 60.0
                and participation >= 0.20
                and not gap_heavy_flag
            )
            window_full_flag = warmup_satisfied and completeness >= 0.999
            window_partial_flag = not window_full_flag
        else:
            filled_flag = bool(row.get("gap_fill_flag", False))
            completeness = max(0.0, min(float(row.get("window_completeness", 1.0) or 0.0), 1.0))
            effective_observation_count = 0 if filled_flag else 1
            effective_coverage_ratio = 0.0 if filled_flag else completeness
            quality_score = _coalesce_numeric(row, "source_quality_hint", "quality_score")
            participation = _coalesce_numeric(
                row,
                "source_participation_score",
                default=_fallback_participation_from_source_count(row.get("source_count"), quality_score),
            )
            overlap = _coalesce_numeric(row, "overlap_confidence_hint", default=0.0)
            observed_interval_ms = max(
                int(row.get("observed_interval_ms", row.get("primary_staleness_ms", 0)) or 0),
                0,
            )
            expected_interval_ms = max(int(row.get("expected_interval_ms", resolution_ms) or resolution_ms), 1)
            filled_ratio = 1.0 if filled_flag else 0.0
            gap_heavy_flag = filled_flag or observed_interval_ms > (expected_interval_ms * 5)
            gap_count = 1 if gap_heavy_flag else 0
            max_gap_ms = observed_interval_ms if gap_heavy_flag else 0
            ready_flag = (
                effective_coverage_ratio >= 0.85
                and quality_score >= 60.0
                and participation >= 0.20
                and not gap_heavy_flag
            )
            window_full_flag = False
            window_partial_flag = False

        metrics.append(
            {
                "date": metric_date,
                "session_code": session,
                "effective_observation_count": effective_observation_count,
                "effective_coverage_ratio": effective_coverage_ratio,
                "quality_score": quality_score,
                "source_quality_hint": quality_score,
                "source_participation_score": participation,
                "overlap_confidence": overlap,
                "filled_ratio": filled_ratio,
                "gap_heavy_flag": gap_heavy_flag,
                "gap_count": gap_count,
                "max_gap_ms": max_gap_ms,
                "ready_flag": ready_flag,
                "low_overlap_flag": overlap < 0.15,
                "low_quality_flag": quality_score < 60.0,
                "window_full_flag": window_full_flag,
                "window_partial_flag": window_partial_flag,
            }
        )
    return metrics


def _timestamp_from_row(row: dict[str, object]) -> dt.datetime:
    timestamp = row.get("ts_utc", row.get("anchor_ts_utc"))
    if not isinstance(timestamp, dt.datetime):
        raise TypeError("State readiness metrics require ts_utc or anchor_ts_utc datetimes")
    return timestamp


def _session_code_from_row(row: dict[str, object], timestamp: dt.datetime) -> str:
    session = row.get("session_code")
    if isinstance(session, str) and session:
        return session
    return _session_code_from_timestamp(timestamp)


def _session_code_from_timestamp(timestamp: dt.datetime) -> str:
    if timestamp.weekday() == 5 or (timestamp.weekday() == 4 and timestamp.hour >= 22) or (
        timestamp.weekday() == 6 and timestamp.hour < 22
    ):
        return "weekend_closed"
    hour = timestamp.hour
    if 13 <= hour < 16:
        return "overlap"
    if 13 <= hour < 22:
        return "ny"
    if 7 <= hour < 16:
        return "london"
    if 0 <= hour < 8:
        return "asia"
    return "other"


def _coalesce_numeric(
    row: dict[str, object],
    *columns: str,
    default: float = 0.0,
) -> float:
    for column in columns:
        value = row.get(column)
        if value is not None:
            return float(value)
    return float(default)


def _fallback_participation_from_source_count(source_count: object, quality_score: float) -> float:
    if source_count is None:
        return max(0.0, min(quality_score / 100.0, 1.0)) * 0.25
    normalized = max(0.0, min(float(source_count) / 2.0, 1.0))
    return max(0.0, min(normalized * max(0.0, min(quality_score / 100.0, 1.0)), 1.0))


def _source_reliability_band(mean_quality_score: float, mean_participation_score: float) -> str:
    if mean_quality_score >= 80.0 and mean_participation_score >= 0.50:
        return "high"
    if mean_quality_score >= 65.0 and mean_participation_score >= 0.25:
        return "medium"
    return "low"


def _overlap_quality_band(mean_overlap_confidence: float) -> str:
    if mean_overlap_confidence >= 0.50:
        return "high"
    if mean_overlap_confidence >= 0.15:
        return "medium"
    return "low"


def _gap_burden_band(gap_burden_ratio: float) -> str:
    if gap_burden_ratio <= 0.05:
        return "low"
    if gap_burden_ratio <= 0.15:
        return "medium"
    return "high"


def _readiness_band(
    *,
    effective_coverage_ratio: float,
    ready_interval_ratio: float,
    source_reliability_band: str,
    gap_burden_ratio: float,
) -> str:
    if (
        effective_coverage_ratio >= 0.85
        and ready_interval_ratio >= 0.80
        and source_reliability_band != "low"
        and gap_burden_ratio <= 0.15
    ):
        return "ready"
    if effective_coverage_ratio >= 0.70 and ready_interval_ratio >= 0.55:
        return "limited"
    return "not_ready"
