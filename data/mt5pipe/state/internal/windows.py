"""Pure state-frame and rolling-window builders."""

from __future__ import annotations

import datetime as dt
from typing import Callable

import polars as pl

from mt5pipe.contracts.state import parse_window_size
from mt5pipe.state.internal.bar_support import is_forex_closed
from mt5pipe.state.internal.quality import (
    coverage_mode_for_clock,
    snapshot_overlap_confidence_hint,
    snapshot_source_participation_score,
    state_resolution_ms,
)
from mt5pipe.state.models import StateSnapshot, StateWindowRecord


def session_code(ts_utc: dt.datetime) -> str:
    """Map UTC timestamps to the repo's canonical session labels."""
    if is_forex_closed(ts_utc):
        return "weekend_closed"
    hour = ts_utc.hour
    if 13 <= hour < 16:
        return "overlap"
    if 13 <= hour < 22:
        return "ny"
    if 7 <= hour < 16:
        return "london"
    if 0 <= hour < 8:
        return "asia"
    return "other"
def canonical_ticks_to_state_rows(
    canonical_df: pl.DataFrame,
    *,
    symbol: str,
    state_version_ref: str,
    provenance_resolver: Callable[[dt.date], list[str]],
) -> pl.DataFrame:
    """Convert canonical ticks into state snapshots with disagreement/staleness substrate."""
    if canonical_df.is_empty():
        return pl.DataFrame()

    working = canonical_df.sort("ts_msc")
    required = {"ts_utc", "ts_msc", "bid", "ask", "symbol"}
    missing = sorted(required - set(working.columns))
    if missing:
        raise KeyError(f"Canonical ticks are missing required columns: {missing}")

    prev_ts_msc: int | None = None
    rows: list[dict[str, object]] = []
    for row in working.iter_rows(named=True):
        ts_utc = row["ts_utc"]
        if not isinstance(ts_utc, dt.datetime):
            raise TypeError("Canonical tick ts_utc must be a timezone-aware datetime")
        bid = float(row["bid"])
        ask = float(row["ask"])
        broker_a_mid = _mid_from_pair(row.get("broker_a_bid"), row.get("broker_a_ask"))
        broker_b_mid = _mid_from_pair(row.get("broker_b_bid"), row.get("broker_b_ask"))
        broker_a_spread = _spread_from_pair(row.get("broker_a_bid"), row.get("broker_a_ask"))
        broker_b_spread = _spread_from_pair(row.get("broker_b_bid"), row.get("broker_b_ask"))
        disagreement_bps = _disagreement_bps(broker_a_mid, broker_b_mid)
        spread_disagreement_bps = _disagreement_bps(broker_a_spread, broker_b_spread)
        source_secondary = _normalize_source_secondary(row.get("source_secondary"))
        source_count = 1 + int(bool(source_secondary))
        ts_msc = int(row["ts_msc"])
        primary_staleness_ms = 0 if prev_ts_msc is None else max(ts_msc - prev_ts_msc, 0)
        prev_ts_msc = ts_msc
        quality_score = float(row.get("quality_score", 0.0) or 0.0)
        source_participation_score = snapshot_source_participation_score(
            source_count=source_count,
            conflict_flag=bool(row.get("conflict_flag", False)),
            disagreement_bps=disagreement_bps,
            gap_fill_flag=False,
            quality_score=quality_score,
        )
        overlap_confidence = snapshot_overlap_confidence_hint(
            source_count=source_count,
            source_participation_score=source_participation_score,
            window_completeness=1.0,
            source_quality_hint=quality_score,
        )

        snapshot = StateSnapshot(
            state_version=state_version_ref,
            snapshot_id=f"state:{symbol}:tick:{ts_utc.isoformat()}",
            symbol=symbol,
            ts_utc=ts_utc,
            ts_msc=ts_msc,
            clock="TICK",
            window_start_utc=ts_utc,
            window_end_utc=ts_utc,
            bid=bid,
            ask=ask,
            mid=(bid + ask) / 2.0,
            spread=ask - bid,
            source_primary=str(row.get("source_primary") or "canonical"),
            source_secondary=source_secondary or None,
            source_count=source_count,
            merge_mode=str(row.get("merge_mode") or ("best" if source_count >= 2 else "single")),
            conflict_flag=bool(row.get("conflict_flag", False)),
            disagreement_bps=disagreement_bps,
            spread_disagreement_bps=spread_disagreement_bps,
            broker_a_mid=broker_a_mid,
            broker_b_mid=broker_b_mid,
            broker_a_spread=broker_a_spread,
            broker_b_spread=broker_b_spread,
            primary_staleness_ms=primary_staleness_ms,
            secondary_staleness_ms=None,
            source_offset_ms=None,
            expected_interval_ms=state_resolution_ms("tick"),
            observed_interval_ms=primary_staleness_ms,
            quality_score=quality_score,
            source_quality_hint=quality_score,
            source_participation_score=source_participation_score,
            overlap_confidence_hint=overlap_confidence,
            expected_observations=1,
            observed_observations=1,
            missing_observations=0,
            window_completeness=1.0,
            gap_fill_flag=False,
            session_code=session_code(ts_utc),
            trust_flags=["conflict"] if bool(row.get("conflict_flag", False)) else [],
            provenance_refs=provenance_resolver(ts_utc.date()),
        )
        rows.append(snapshot.model_dump())

    return pl.DataFrame(rows).sort("ts_utc")


def build_state_windows(
    state_df: pl.DataFrame,
    *,
    symbol: str,
    clock: str,
    state_version: str,
    window_size: str,
    base_provenance_refs: list[str],
    include_partial_windows: bool = False,
) -> pl.DataFrame:
    """Build PIT-safe rolling windows with machine-native list payloads."""
    if state_df.is_empty():
        return pl.DataFrame()

    if "ts_utc" not in state_df.columns:
        raise KeyError("State DataFrame must contain ts_utc")

    window_delta = parse_window_size(window_size)
    resolution_ms = state_resolution_ms(clock)
    expected_bins = max(1, int(window_delta.total_seconds() * 1000 / resolution_ms))
    working = state_df.sort("ts_utc")
    rows: list[dict[str, object]] = []

    ts_values = working["ts_utc"].to_list()
    if not ts_values:
        return pl.DataFrame()
    rows_as_dicts = working.to_dicts()
    left = 0

    for idx, anchor in enumerate(rows_as_dicts):
        anchor_ts = anchor["ts_utc"]
        if not isinstance(anchor_ts, dt.datetime):
            raise TypeError("State row ts_utc must be a timezone-aware datetime")
        lower_bound = anchor_ts - window_delta
        while left <= idx and rows_as_dicts[left]["ts_utc"] <= lower_bound:
            left += 1
        if left > idx:
            continue

        window_rows = rows_as_dicts[left : idx + 1]
        observed_bins = _observed_bins_from_rows(window_rows, resolution_ms)
        missing_bins = max(expected_bins - observed_bins, 0)
        completeness = (expected_bins - missing_bins) / expected_bins
        warmup_satisfied = missing_bins == 0
        if not include_partial_windows and not warmup_satisfied:
            continue

        mids = [float(row.get("mid", 0.0) or 0.0) for row in window_rows]
        spreads = [float(row.get("spread", 0.0) or 0.0) for row in window_rows]
        source_counts = [int(row.get("source_count", 0) or 0) for row in window_rows]
        quality_scores = [float(row.get("quality_score", 0.0) or 0.0) for row in window_rows]
        source_quality_hints = _optional_list_from_rows(window_rows, "source_quality_hint", float)
        participation_scores = _optional_list_from_rows(window_rows, "source_participation_score", float)
        overlap_confidences = _optional_list_from_rows(window_rows, "overlap_confidence_hint", float)
        disagreements = _optional_list_from_rows(window_rows, "disagreement_bps", float)
        staleness = _optional_list_from_rows(window_rows, "primary_staleness_ms", int)
        conflicts = [bool(row.get("conflict_flag", False)) for row in window_rows]
        source_offsets = _optional_list_from_rows(window_rows, "source_offset_ms", int)
        gap_fill_flags = [bool(row.get("gap_fill_flag", False)) for row in window_rows]
        observed_span_ms = max(int(window_rows[-1]["ts_msc"]) - int(window_rows[0]["ts_msc"]), 0) if len(window_rows) > 1 else 0
        gap_count, max_gap_ms = _gap_stats(clock, resolution_ms, staleness, gap_fill_flags)
        mid_returns = _mid_return_bps(mids)
        row_count = len(window_rows)
        filled_row_count = sum(1 for flag in gap_fill_flags if flag)

        record = StateWindowRecord(
            state_version=state_version,
            window_id=f"state-window:{symbol}:{clock}:{window_size}:{anchor_ts.isoformat()}",
            symbol=symbol,
            clock=clock,
            anchor_ts_utc=anchor_ts,
            anchor_ts_msc=int(anchor.get("ts_msc")) if anchor.get("ts_msc") is not None else int(anchor_ts.timestamp() * 1000),
            window_size=window_size,
            window_start_utc=lower_bound,
            window_end_utc=anchor_ts,
            row_count=row_count,
            expected_row_count=expected_bins,
            missing_row_count=missing_bins,
            warmup_missing_rows=missing_bins,
            warmup_satisfied=warmup_satisfied,
            completeness=completeness,
            coverage_mode=coverage_mode_for_clock(clock),
            observed_span_ms=observed_span_ms,
            source_count_mean=float(sum(source_counts) / row_count) if source_counts else 0.0,
            dual_source_ratio_window=float(sum(1 for value in source_counts if value >= 2) / row_count) if row_count else 0.0,
            quality_score_mean=float(sum(quality_scores) / row_count) if quality_scores else 0.0,
            source_quality_hint_mean=_mean_optional(source_quality_hints),
            source_participation_score_mean=_mean_optional(participation_scores),
            overlap_confidence_mean=_mean_optional(overlap_confidences),
            conflict_count_window=sum(1 for value in conflicts if value),
            conflict_ratio=float(sum(1 for value in conflicts if value) / row_count) if row_count else 0.0,
            disagreement_bps_mean=_mean_optional(disagreements),
            staleness_ms_max=_max_optional(staleness),
            filled_row_count=filled_row_count,
            filled_ratio=float(filled_row_count / row_count) if row_count else 0.0,
            gap_count=gap_count,
            max_gap_ms=max_gap_ms,
            mid_values=mids,
            spread_values=spreads,
            mid_return_bps_values=mid_returns,
            source_count_values=source_counts,
            quality_score_values=quality_scores,
            disagreement_bps_values=disagreements,
            staleness_ms_values=staleness,
            conflict_flags=conflicts,
            source_offset_ms_values=source_offsets,
            provenance_refs=list(base_provenance_refs),
        )
        rows.append(record.model_dump())

    return pl.DataFrame(rows).sort("anchor_ts_utc") if rows else pl.DataFrame()


def _normalize_source_secondary(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _mid_from_pair(bid: object, ask: object) -> float | None:
    if bid is None or ask is None:
        return None
    bid_f = float(bid)
    ask_f = float(ask)
    if bid_f <= 0 or ask_f <= 0:
        return None
    return (bid_f + ask_f) / 2.0


def _spread_from_pair(bid: object, ask: object) -> float | None:
    if bid is None or ask is None:
        return None
    bid_f = float(bid)
    ask_f = float(ask)
    if bid_f <= 0 or ask_f <= 0:
        return None
    return max(ask_f - bid_f, 0.0)


def _disagreement_bps(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    avg = (left + right) / 2.0
    if avg <= 0:
        return None
    return abs(left - right) / avg * 10_000.0


def _observed_bins_from_rows(window_rows: list[dict[str, object]], resolution_ms: int) -> int:
    ts_values = [
        int(row["ts_msc"]) if row.get("ts_msc") is not None else int(row["ts_utc"].timestamp() * 1000)
        for row in window_rows
    ]
    return len({ts // resolution_ms for ts in ts_values})


def _optional_list_from_rows(window_rows: list[dict[str, object]], column: str, caster: type[int] | type[float]) -> list[int | float | None]:
    values: list[int | float | None] = []
    for row in window_rows:
        value = row.get(column)
        values.append(None if value is None else caster(value))
    return values


def _gap_stats(
    clock: str,
    resolution_ms: int,
    staleness: list[int | None],
    gap_fill_flags: list[bool],
) -> tuple[int, int]:
    if clock.lower() == "tick":
        gap_events = [int(value) for value in staleness if value is not None and int(value) > resolution_ms]
        return len(gap_events), max(gap_events, default=0)
    gap_count = 0
    max_run = 0
    current_run = 0
    for flag in gap_fill_flags:
        if flag:
            current_run += 1
            max_run = max(max_run, current_run)
        elif current_run:
            gap_count += 1
            current_run = 0
    if current_run:
        gap_count += 1
    return gap_count, max_run * resolution_ms


def _mean_optional(values: list[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def _max_optional(values: list[int | None]) -> int | None:
    filtered = [int(value) for value in values if value is not None]
    if not filtered:
        return None
    return max(filtered)


def _mid_return_bps(mids: list[float]) -> list[float]:
    if not mids:
        return []
    returns = [0.0]
    for prev, cur in zip(mids[:-1], mids[1:]):
        if prev <= 0:
            returns.append(0.0)
        else:
            returns.append(((cur - prev) / prev) * 10_000.0)
    return returns
