"""State-sector tests for machine-native state substrate behavior."""

from __future__ import annotations

import datetime as dt

import polars as pl

from mt5pipe.contracts.artifacts import ArtifactKind
from mt5pipe.contracts.state import StateWindowRequest, TickArtifactRef
from mt5pipe.state.public import (
    StateArtifactRef,
    StateCoverageSummary,
    StateIntervalReadinessSummary,
    StateReadinessSummary,
    StateService,
    StateSnapshot,
    StateSourceQualitySummary,
    StateWindowArtifactRef,
    StateWindowRecord,
    load_state_artifact,
    load_state_window_artifact,
    materialize_state_windows,
)


UTC = dt.timezone.utc


def _bars(start: dt.datetime, rows: int) -> pl.DataFrame:
    times = [start + dt.timedelta(minutes=i) for i in range(rows)]
    source_pattern = [2, 2, 2, 1, 2, 2]
    conflict_pattern = [0, 0, 1, 0, 0, 0]
    dual_ratio_pattern = [0.8, 0.8, 0.8, 0.0, 0.8, 0.8]
    return pl.DataFrame(
        {
            "symbol": ["XAUUSD"] * rows,
            "timeframe": ["M1"] * rows,
            "time_utc": times,
            "open": [3000.0 + i for i in range(rows)],
            "high": [3000.4 + i for i in range(rows)],
            "low": [2999.8 + i for i in range(rows)],
            "close": [3000.2 + i for i in range(rows)],
            "tick_count": [10] * rows,
            "spread_mean": [0.2] * rows,
            "bid_close": [3000.1 + i for i in range(rows)],
            "ask_close": [3000.3 + i for i in range(rows)],
            "source_count": [source_pattern[i % len(source_pattern)] for i in range(rows)],
            "conflict_count": [conflict_pattern[i % len(conflict_pattern)] for i in range(rows)],
            "dual_source_ratio": [dual_ratio_pattern[i % len(dual_ratio_pattern)] for i in range(rows)],
        }
    )


def _canonical_ticks(start: dt.datetime) -> pl.DataFrame:
    times = [start + dt.timedelta(seconds=i * 10) for i in range(4)]
    return pl.DataFrame(
        {
            "ts_utc": times,
            "ts_msc": [int(ts.timestamp() * 1000) for ts in times],
            "symbol": ["XAUUSD"] * 4,
            "bid": [3000.0, 3000.1, 3000.2, 3000.3],
            "ask": [3000.2, 3000.3, 3000.4, 3000.5],
            "last": [0.0] * 4,
            "volume": [1.0] * 4,
            "source_primary": ["broker_a"] * 4,
            "source_secondary": ["broker_b"] * 4,
            "merge_mode": ["best", "best", "conflict", "best"],
            "quality_score": [95.0, 94.0, 70.0, 96.0],
            "conflict_flag": [False, False, True, False],
            "broker_a_bid": [3000.0, 3000.1, 3000.2, 3000.3],
            "broker_a_ask": [3000.2, 3000.3, 3000.4, 3000.5],
            "broker_b_bid": [3000.05, 3000.12, 3000.18, 3000.31],
            "broker_b_ask": [3000.24, 3000.34, 3000.44, 3000.55],
        }
    )


def _canonical_ticks_for_bars(
    start: dt.datetime,
    rows: int,
    *,
    quality_score: float = 0.82,
    dual_source: bool = False,
) -> pl.DataFrame:
    times = [start + dt.timedelta(minutes=i, seconds=10) for i in range(rows)]
    return pl.DataFrame(
        {
            "ts_utc": times,
            "ts_msc": [int(ts.timestamp() * 1000) for ts in times],
            "symbol": ["XAUUSD"] * rows,
            "bid": [3000.0 + i for i in range(rows)],
            "ask": [3000.2 + i for i in range(rows)],
            "last": [0.0] * rows,
            "volume": [1.0] * rows,
            "source_primary": ["broker_a"] * rows,
            "source_secondary": ["broker_b" if dual_source else ""] * rows,
            "merge_mode": ["best" if dual_source else "single"] * rows,
            "quality_score": [quality_score] * rows,
            "conflict_flag": [False] * rows,
        }
    )


def test_tick_state_materialization_builds_disagreement_and_staleness(paths, store) -> None:
    service = StateService(paths, store)
    start = dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    ticks = _canonical_ticks(start)
    store.write(ticks, paths.canonical_ticks_file("XAUUSD", start.date()))

    result = service.materialize_tick_state(symbol="XAUUSD", date_from=start.date(), date_to=start.date())

    assert result.ref.clock == "TICK"
    assert result.state_df.height == 4
    assert result.state_df["primary_staleness_ms"].to_list() == [0, 10_000, 10_000, 10_000]
    assert result.state_df["disagreement_bps"].drop_nulls().len() == 4
    assert result.state_df["conflict_flag"].to_list() == [False, False, True, False]
    assert result.state_df["source_participation_score"].drop_nulls().len() == 4
    assert result.coverage_summary.coverage_mode == "activity_clock"
    assert result.source_quality_summary.conflict_ratio == 0.25

    loaded = load_state_artifact(paths, store, result.ref)
    assert loaded.height == 4
    assert loaded["clock"].unique().to_list() == ["TICK"]


def test_state_window_materialization_is_pit_safe_and_persisted(paths, store) -> None:
    service = StateService(paths, store)
    start = dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    bars = _bars(start, 6)
    bars = bars.filter(pl.col("time_utc") != (start + dt.timedelta(minutes=3)))
    store.write(bars, paths.built_bars_file("XAUUSD", "M1", start.date()))

    state_result = service.materialize_state(
        symbol="XAUUSD",
        clock="M1",
        state_version_ref="state.default@1.0.0",
        date_from=start.date(),
        date_to=start.date(),
        build_id="state.test.build",
        dataset_spec_ref="dataset.test@1.0.0",
        code_version="workspace-local-no-git",
        merge_config_ref="merge.default@test",
    )

    windows = materialize_state_windows(
        paths,
        store,
        state_result.ref,
        request=StateWindowRequest(
            symbol="XAUUSD",
            clock="M1",
            state_version="state.default@1.0.0",
            date_from=start.date(),
            date_to=start.date(),
            window_sizes=["5m"],
            include_partial_windows=False,
        ),
    )

    window_result = windows["5m"]
    assert isinstance(window_result.ref, StateWindowArtifactRef)
    assert window_result.coverage_summary.coverage_mode == "regular_clock"
    assert window_result.window_df.height == 2
    assert window_result.window_df["anchor_ts_utc"].to_list()[0] == start + dt.timedelta(minutes=4)

    second_window = window_result.window_df.row(1, named=True)
    assert second_window["row_count"] == 5
    assert second_window["warmup_satisfied"] is True
    assert second_window["gap_count"] == 1
    assert second_window["filled_row_count"] == 1
    assert second_window["mid_values"] == [3001.2, 3002.2, 3002.2, 3004.2, 3005.2]
    assert second_window["conflict_count_window"] == 1

    loaded = load_state_window_artifact(paths, store, window_result.ref)
    assert loaded.height == 2
    assert loaded["window_size"].unique().to_list() == ["5m"]
    assert loaded["filled_ratio"].max() > 0.0


def test_state_manifest_contains_coverage_and_source_quality_metadata(paths, store) -> None:
    service = StateService(paths, store)
    start = dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    bars = _bars(start, 6).filter(pl.col("time_utc") != (start + dt.timedelta(minutes=2)))
    store.write(bars, paths.built_bars_file("XAUUSD", "M1", start.date()))

    result = service.materialize_state(
        symbol="XAUUSD",
        clock="M1",
        state_version_ref="state.default@1.0.0",
        date_from=start.date(),
        date_to=start.date(),
        build_id="state.test.build",
        dataset_spec_ref="dataset.test@1.0.0",
        code_version="workspace-local-no-git",
        merge_config_ref="merge.default@test",
    )

    manifest = result.manifest
    assert isinstance(manifest.coverage_summary, StateCoverageSummary)
    assert isinstance(manifest.source_quality_summary, StateSourceQualitySummary)
    assert isinstance(manifest.readiness_summary, StateReadinessSummary)
    assert all(isinstance(rollup, StateIntervalReadinessSummary) for rollup in manifest.daily_readiness_rollups)
    assert all(isinstance(rollup, StateIntervalReadinessSummary) for rollup in manifest.session_readiness_rollups)
    assert manifest.coverage_summary.filled_row_count == 1
    assert manifest.coverage_summary.gap_count == 1
    assert manifest.source_quality_summary.mean_source_count < 2.0
    assert manifest.readiness_summary.readiness_band in {"limited", "ready"}
    assert manifest.readiness_summary.source_reliability_band in {"medium", "high"}
    assert manifest.clock == "M1"
    assert manifest.symbol == "XAUUSD"
    assert manifest.time_range_start_utc == result.coverage_summary.time_range_start_utc


def test_state_materialization_uses_canonical_tick_quality_for_source_quality(paths, store) -> None:
    service = StateService(paths, store)
    start = dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    bars = _bars(start, 6).with_columns(
        [
            pl.lit(1).alias("source_count"),
            pl.lit(0).alias("conflict_count"),
            pl.lit(0.0).alias("dual_source_ratio"),
        ]
    )
    ticks = _canonical_ticks_for_bars(start, 6, quality_score=0.82, dual_source=False)
    store.write(bars, paths.built_bars_file("XAUUSD", "M1", start.date()))
    store.write(ticks, paths.canonical_ticks_file("XAUUSD", start.date()))

    result = service.materialize_state(
        symbol="XAUUSD",
        clock="M1",
        state_version_ref="state.default@1.0.0",
        date_from=start.date(),
        date_to=start.date(),
        build_id="state.test.quality",
        dataset_spec_ref="dataset.test@1.0.0",
        code_version="workspace-local-no-git",
        merge_config_ref="merge.default@test",
    )

    assert result.source_quality_summary.mean_quality_score >= 80.0
    assert result.state_df["quality_score"].min() >= 80.0
    assert result.state_df["source_quality_hint"].min() >= 80.0


def test_state_window_materialization_respects_requested_anchor_range_with_prior_warmup(paths, store) -> None:
    service = StateService(paths, store)
    start = dt.datetime(2026, 4, 1, 23, 56, tzinfo=UTC)
    bars = _bars(start, 10)
    for day in sorted({ts.date() for ts in bars["time_utc"].to_list()}):
        day_df = bars.filter(pl.col("time_utc").dt.date() == day)
        store.write(day_df, paths.built_bars_file("XAUUSD", "M1", day))

    state_result = service.materialize_state(
        symbol="XAUUSD",
        clock="M1",
        state_version_ref="state.default@1.0.0",
        date_from=dt.date(2026, 4, 1),
        date_to=dt.date(2026, 4, 2),
        build_id="state.test.cross_day",
        dataset_spec_ref="dataset.test@1.0.0",
        code_version="workspace-local-no-git",
        merge_config_ref="merge.default@test",
    )

    windows = materialize_state_windows(
        paths,
        store,
        state_result.ref,
        request=StateWindowRequest(
            symbol="XAUUSD",
            clock="M1",
            state_version="state.default@1.0.0",
            date_from=dt.date(2026, 4, 2),
            date_to=dt.date(2026, 4, 2),
            window_sizes=["5m"],
            include_partial_windows=False,
        ),
    )

    window_result = windows["5m"]
    anchors = window_result.window_df["anchor_ts_utc"].to_list()
    assert anchors[0] == dt.datetime(2026, 4, 2, 0, 0, tzinfo=UTC)
    assert all(anchor.date() == dt.date(2026, 4, 2) for anchor in anchors)
    assert window_result.window_df.height == 6
    assert window_result.window_df.row(0, named=True)["warmup_satisfied"] is True
    assert len(window_result.manifest.input_partition_refs) == 2


def test_state_materialization_is_idempotent_for_persisted_state_loads(paths, store) -> None:
    service = StateService(paths, store)
    start = dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    bars = _bars(start, 6)
    store.write(bars, paths.built_bars_file("XAUUSD", "M1", start.date()))

    first = service.materialize_state(
        symbol="XAUUSD",
        clock="M1",
        state_version_ref="state.default@1.0.0",
        date_from=start.date(),
        date_to=start.date(),
        build_id="state.test.idempotent",
        dataset_spec_ref="dataset.test@1.0.0",
        code_version="workspace-local-no-git",
        merge_config_ref="merge.default@test",
    )
    second = service.materialize_state(
        symbol="XAUUSD",
        clock="M1",
        state_version_ref="state.default@1.0.0",
        date_from=start.date(),
        date_to=start.date(),
        build_id="state.test.idempotent",
        dataset_spec_ref="dataset.test@1.0.0",
        code_version="workspace-local-no-git",
        merge_config_ref="merge.default@test",
    )

    loaded = load_state_artifact(paths, store, second.ref)
    assert first.state_df.height == second.state_df.height
    assert loaded.height == second.state_df.height


def test_state_window_materialization_is_deterministic_for_repeated_requests(paths, store) -> None:
    service = StateService(paths, store)
    start = dt.datetime(2026, 4, 1, 23, 56, tzinfo=UTC)
    bars = _bars(start, 10)
    for day in sorted({ts.date() for ts in bars["time_utc"].to_list()}):
        day_df = bars.filter(pl.col("time_utc").dt.date() == day)
        store.write(day_df, paths.built_bars_file("XAUUSD", "M1", day))

    state_result = service.materialize_state(
        symbol="XAUUSD",
        clock="M1",
        state_version_ref="state.default@1.0.0",
        date_from=dt.date(2026, 4, 1),
        date_to=dt.date(2026, 4, 2),
        build_id="state.test.deterministic",
        dataset_spec_ref="dataset.test@1.0.0",
        code_version="workspace-local-no-git",
        merge_config_ref="merge.default@test",
    )
    request = StateWindowRequest(
        symbol="XAUUSD",
        clock="M1",
        state_version="state.default@1.0.0",
        date_from=dt.date(2026, 4, 2),
        date_to=dt.date(2026, 4, 2),
        window_sizes=["5m"],
        include_partial_windows=False,
    )

    first = materialize_state_windows(paths, store, state_result.ref, request=request)["5m"]
    second = materialize_state_windows(paths, store, state_result.ref, request=request)["5m"]

    assert first.ref.artifact_id == second.ref.artifact_id
    assert first.manifest.content_hash == second.manifest.content_hash
    assert first.window_df.equals(second.window_df)
    loaded = load_state_window_artifact(paths, store, second.ref)
    assert loaded.height == second.window_df.height
    assert second.readiness_summary.available_window_count == second.window_df.height
    assert second.readiness_summary.available_window_ratio == 1.0


def test_state_window_readiness_rollups_capture_wider_range_completeness(paths, store) -> None:
    service = StateService(paths, store)
    start = dt.datetime(2026, 4, 1, 23, 56, tzinfo=UTC)
    bars = _bars(start, 12)
    bars = bars.filter(
        ~pl.col("time_utc").is_in(
            [
                start + dt.timedelta(minutes=2),
                start + dt.timedelta(minutes=7),
            ]
        )
    )
    for day in sorted({ts.date() for ts in bars["time_utc"].to_list()}):
        day_df = bars.filter(pl.col("time_utc").dt.date() == day)
        store.write(day_df, paths.built_bars_file("XAUUSD", "M1", day))

    state_result = service.materialize_state(
        symbol="xauusd",
        clock="m1",
        state_version_ref="state.default@1.0.0",
        date_from=dt.date(2026, 4, 1),
        date_to=dt.date(2026, 4, 2),
        build_id="state.test.rollups",
        dataset_spec_ref="dataset.test@1.0.0",
        code_version="workspace-local-no-git",
        merge_config_ref="merge.default@test",
    )
    window_result = materialize_state_windows(
        paths,
        store,
        state_result.ref,
        request=StateWindowRequest(
            symbol="xauusd",
            clock="m1",
            state_version="state.default@1.0.0",
            date_from=dt.date(2026, 4, 2),
            date_to=dt.date(2026, 4, 2),
            window_sizes=["5m"],
            include_partial_windows=False,
        ),
    )["5m"]

    assert window_result.readiness_summary.available_window_ratio is not None
    assert window_result.readiness_summary.available_window_ratio == 1.0
    assert window_result.readiness_summary.gap_heavy_interval_count > 0
    assert window_result.readiness_summary.gap_burden_band in {"medium", "high"}
    assert len(window_result.daily_readiness_rollups) == 1
    assert all(rollup.interval_kind == "session" for rollup in window_result.session_readiness_rollups)
    assert any(rollup.gap_heavy_interval_count > 0 for rollup in window_result.session_readiness_rollups)


def test_state_window_materialization_rejects_request_outside_source_range(paths, store) -> None:
    service = StateService(paths, store)
    start = dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC)
    bars = _bars(start, 6)
    store.write(bars, paths.built_bars_file("XAUUSD", "M1", start.date()))

    state_result = service.materialize_state(
        symbol="XAUUSD",
        clock="M1",
        state_version_ref="state.default@1.0.0",
        date_from=start.date(),
        date_to=start.date(),
        build_id="state.test.range_guard",
        dataset_spec_ref="dataset.test@1.0.0",
        code_version="workspace-local-no-git",
        merge_config_ref="merge.default@test",
    )

    try:
        materialize_state_windows(
            paths,
            store,
            state_result.ref,
            request=StateWindowRequest(
                symbol="XAUUSD",
                clock="M1",
                state_version="state.default@1.0.0",
                date_from=dt.date(2026, 4, 1),
                date_to=dt.date(2026, 4, 2),
                window_sizes=["5m"],
                include_partial_windows=False,
            ),
        )
    except ValueError as exc:
        assert "must lie within the source StateArtifactRef range" in str(exc)
    else:
        raise AssertionError("Expected a ValueError when the request range exceeds the source state range")


def test_state_public_boundary_exports_are_available() -> None:
    assert StateSnapshot is not None
    assert StateArtifactRef is not None
    assert StateCoverageSummary is not None
    assert StateIntervalReadinessSummary is not None
    assert StateReadinessSummary is not None
    assert StateWindowArtifactRef is not None
    assert StateWindowRecord is not None
    assert StateService is not None
    assert StateSourceQualitySummary is not None

    tick_ref = TickArtifactRef(
        artifact_id="canonical_tick.XAUUSD.abc",
        kind=ArtifactKind.CANONICAL_TICK,
        logical_name="XAUUSD",
        version="1.0.0",
        symbol="XAUUSD",
        date_from=dt.date(2026, 4, 1),
        date_to=dt.date(2026, 4, 1),
    )
    assert tick_ref.kind == ArtifactKind.CANONICAL_TICK
