"""State services and public-facing loaders for state artifacts."""

from __future__ import annotations

import datetime as dt
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from mt5pipe.contracts.artifacts import ArtifactKind
from mt5pipe.contracts.state import (
    StateArtifactRef,
    StateWindowArtifactRef,
    StateWindowRequest,
    TickArtifactRef,
)
from mt5pipe.state.internal.artifacts import (
    build_artifact_id,
    build_id_now,
    build_manifest_id,
    compute_content_hash,
    state_code_version,
    write_state_manifest,
)
from mt5pipe.state.internal.bar_support import detect_gaps, fill_bar_gaps, timeframe_to_seconds, validate_bars
from mt5pipe.state.internal.quality import (
    build_state_coverage_summary,
    build_state_interval_readiness_rollups,
    build_state_readiness_summary,
    build_state_source_quality_summary,
    snapshot_overlap_confidence_hint,
    snapshot_source_participation_score,
    state_resolution_ms,
)
from mt5pipe.state.internal.windows import build_state_windows, canonical_ticks_to_state_rows, session_code
from mt5pipe.state.models import (
    StateArtifactManifest,
    StateCoverageSummary,
    StateIntervalReadinessSummary,
    StateReadinessSummary,
    StateSnapshot,
    StateSourceQualitySummary,
)
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths


@dataclass
class StateMaterializationResult:
    ref: StateArtifactRef
    artifact_id: str
    manifest: StateArtifactManifest
    manifest_path: Path
    coverage_summary: StateCoverageSummary
    source_quality_summary: StateSourceQualitySummary
    readiness_summary: StateReadinessSummary
    daily_readiness_rollups: list[StateIntervalReadinessSummary]
    session_readiness_rollups: list[StateIntervalReadinessSummary]
    state_df: pl.DataFrame
    base_df: pl.DataFrame


@dataclass
class StateWindowMaterializationResult:
    ref: StateWindowArtifactRef
    manifest: StateArtifactManifest
    manifest_path: Path
    coverage_summary: StateCoverageSummary
    source_quality_summary: StateSourceQualitySummary
    readiness_summary: StateReadinessSummary
    daily_readiness_rollups: list[StateIntervalReadinessSummary]
    session_readiness_rollups: list[StateIntervalReadinessSummary]
    window_df: pl.DataFrame


class StateService:
    """Materialize and load state-side artifacts without compiler-sector imports."""

    def __init__(
        self,
        paths: StoragePaths,
        store: ParquetStore,
        catalog: Any | None = None,
    ) -> None:
        self._paths = paths
        self._store = store
        self._catalog = catalog

    def materialize_state(
        self,
        *,
        symbol: str,
        clock: str,
        state_version_ref: str,
        date_from: dt.date,
        date_to: dt.date,
        build_id: str,
        dataset_spec_ref: str,
        code_version: str,
        merge_config_ref: str,
    ) -> StateMaterializationResult:
        """Materialize bar-backed state snapshots. Kept compatible for compiler usage."""
        symbol = self._normalize_symbol(symbol)
        clock = self._normalize_clock(clock)
        base_df = self._load_bars_range(symbol, clock, date_from, date_to)
        if base_df.is_empty():
            raise FileNotFoundError(
                f"No built bars found for state materialization: symbol={symbol} clock={clock} "
                f"range={date_from.isoformat()}..{date_to.isoformat()}"
            )

        normalized_base = self._normalize_base_bars(base_df, clock)
        normalized_base = self._enrich_base_bars_with_canonical_quality(
            symbol=symbol,
            clock=clock,
            date_from=date_from,
            date_to=date_to,
            base_df=normalized_base,
        )
        state_df = self._build_bar_state_rows(symbol, clock, state_version_ref, normalized_base)
        input_partition_refs = self._collect_bar_input_partition_refs(symbol, clock, date_from, date_to)
        return self._persist_state_artifact(
            symbol=symbol,
            clock=clock,
            state_version_ref=state_version_ref,
            date_from=date_from,
            date_to=date_to,
            build_id=build_id,
            dataset_spec_ref=dataset_spec_ref,
            code_version=code_version,
            merge_config_ref=merge_config_ref,
            state_df=state_df,
            base_df=normalized_base,
            input_partition_refs=input_partition_refs,
        )

    def materialize_tick_state(
        self,
        *,
        symbol: str,
        date_from: dt.date,
        date_to: dt.date,
        state_version_ref: str = "state.tick@1.0.0",
        build_id: str | None = None,
        dataset_spec_ref: str | None = None,
        code_version: str | None = None,
        merge_config_ref: str | None = None,
    ) -> StateMaterializationResult:
        """Materialize tick-level state snapshots from canonical ticks."""
        symbol = self._normalize_symbol(symbol)
        canonical_df = self.load_tick_artifact(
            TickArtifactRef(
                artifact_id=f"canonical_tick.{symbol}.{date_from.isoformat()}.{date_to.isoformat()}",
                kind=ArtifactKind.CANONICAL_TICK,
                logical_name=symbol,
                version="1.0.0",
                symbol=symbol,
                date_from=date_from,
                date_to=date_to,
            )
        )
        if canonical_df.is_empty():
            raise FileNotFoundError(
                f"No canonical ticks found for tick-state materialization: symbol={symbol} "
                f"range={date_from.isoformat()}..{date_to.isoformat()}"
            )

        state_df = canonical_ticks_to_state_rows(
            canonical_df,
            symbol=symbol,
            state_version_ref=state_version_ref,
            provenance_resolver=lambda day: [str(self._paths.canonical_ticks_dir(symbol, day))],
        )
        input_partition_refs = self._collect_tick_input_partition_refs(symbol, date_from, date_to)
        return self._persist_state_artifact(
            symbol=symbol,
            clock=self._normalize_clock("tick"),
            state_version_ref=state_version_ref,
            date_from=date_from,
            date_to=date_to,
            build_id=build_id or build_id_now("state"),
            dataset_spec_ref=dataset_spec_ref,
            code_version=code_version or state_code_version(),
            merge_config_ref=merge_config_ref,
            state_df=state_df,
            base_df=canonical_df.sort("ts_utc"),
            input_partition_refs=input_partition_refs,
        )

    def load_tick_artifact(self, ref: TickArtifactRef) -> pl.DataFrame:
        """Load canonical ticks for a typed artifact reference."""
        frames: list[pl.DataFrame] = []
        current = ref.date_from
        while current <= ref.date_to:
            day = self._store.read_dir(self._paths.canonical_ticks_dir(ref.symbol, current))
            if not day.is_empty():
                frames.append(day)
            current += dt.timedelta(days=1)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed").sort("ts_msc")

    def load_state_artifact(self, ref: StateArtifactRef) -> pl.DataFrame:
        """Load persisted state snapshots for a typed artifact reference."""
        frames: list[pl.DataFrame] = []
        current = ref.date_from
        while current <= ref.date_to:
            day = self._load_first_available_dir(
                [
                    self._paths.state_artifact_dir(ref.symbol, ref.clock, current, ref.state_version, ref.artifact_id),
                    self._paths.state_dir(ref.symbol, ref.clock, current, ref.state_version),
                ]
            )
            if not day.is_empty():
                frames.append(day)
            current += dt.timedelta(days=1)
        if not frames:
            return pl.DataFrame()
        state_df = pl.concat(frames, how="diagonal_relaxed")
        dedup_cols = [column for column in ["symbol", "clock", "ts_utc", "ts_msc"] if column in state_df.columns]
        if dedup_cols:
            state_df = state_df.unique(subset=dedup_cols, keep="last")
        return state_df.sort("ts_utc")

    def materialize_state_windows(
        self,
        source_ref: StateArtifactRef | TickArtifactRef,
        *,
        request: StateWindowRequest,
        build_id: str | None = None,
        code_version: str | None = None,
    ) -> dict[str, StateWindowMaterializationResult]:
        """Materialize PIT-safe rolling state windows from a state or canonical tick source."""
        if isinstance(source_ref, TickArtifactRef):
            if request.anchor_on != "canonical_tick":
                raise ValueError("TickArtifactRef requires request.anchor_on='canonical_tick'")
            if request.symbol != source_ref.symbol:
                raise ValueError("StateWindowRequest.symbol must match the TickArtifactRef symbol")
            self._validate_request_date_range(
                request.date_from,
                request.date_to,
                source_ref.date_from,
                source_ref.date_to,
                source_name="TickArtifactRef",
            )
            canonical_df = self.load_tick_artifact(source_ref)
            if canonical_df.is_empty():
                raise FileNotFoundError(f"No canonical ticks found for {source_ref}")
            state_df = canonical_ticks_to_state_rows(
                canonical_df,
                symbol=request.symbol,
                state_version_ref=request.state_version,
                provenance_resolver=lambda day: [str(self._paths.canonical_ticks_dir(request.symbol, day))],
            )
            source_artifact_id = source_ref.artifact_id
            clock = "tick"
            input_partition_refs = self._collect_tick_input_partition_refs(
                request.symbol,
                source_ref.date_from,
                source_ref.date_to,
            )
        else:
            if request.anchor_on != "state":
                raise ValueError("StateArtifactRef requires request.anchor_on='state'")
            if request.symbol != source_ref.symbol or request.clock != source_ref.clock or request.state_version != source_ref.state_version:
                raise ValueError("StateWindowRequest must match the source StateArtifactRef symbol/clock/state_version")
            self._validate_request_date_range(
                request.date_from,
                request.date_to,
                source_ref.date_from,
                source_ref.date_to,
                source_name="StateArtifactRef",
            )
            state_df = self.load_state_artifact(source_ref)
            if state_df.is_empty():
                raise FileNotFoundError(f"No state snapshots found for {source_ref}")
            source_artifact_id = source_ref.artifact_id
            clock = source_ref.clock
            input_partition_refs = self._collect_state_input_partition_refs(
                request.symbol,
                source_ref.clock,
                request.state_version,
                source_ref.date_from,
                source_ref.date_to,
            )

        anchor_state_df = state_df.filter(
            pl.col("ts_utc").dt.date().is_between(request.date_from, request.date_to, closed="both")
        )
        results: dict[str, StateWindowMaterializationResult] = {}
        resolved_build_id = build_id or build_id_now("state-window")
        resolved_code_version = code_version or state_code_version()
        base_provenance = list(input_partition_refs)

        for window_size in request.window_sizes:
            window_df = build_state_windows(
                state_df,
                symbol=request.symbol,
                clock=clock,
                state_version=request.state_version,
                window_size=window_size,
                base_provenance_refs=base_provenance,
                include_partial_windows=request.include_partial_windows,
            )
            window_df = self._filter_window_anchors_by_date(window_df, request.date_from, request.date_to)
            logical_name = f"{request.symbol}.{clock}.{window_size}"
            coverage_summary = build_state_coverage_summary(window_df, clock=clock)
            source_quality_summary = build_state_source_quality_summary(window_df)
            readiness_summary = build_state_readiness_summary(
                window_df,
                clock=clock,
                eligible_anchor_count=anchor_state_df.height,
            )
            daily_readiness_rollups = build_state_interval_readiness_rollups(window_df, clock=clock, interval_kind="day")
            session_readiness_rollups = build_state_interval_readiness_rollups(
                window_df,
                clock=clock,
                interval_kind="session",
            )
            content_hash = compute_content_hash(
                {
                    "artifact_kind": "state_window",
                    "logical_name": logical_name,
                    "state_version": request.state_version,
                    "window_size": window_size,
                    "rows": len(window_df),
                    "columns": window_df.columns,
                    "anchor_start": str(window_df["anchor_ts_utc"].min()) if not window_df.is_empty() else "",
                    "anchor_end": str(window_df["anchor_ts_utc"].max()) if not window_df.is_empty() else "",
                    "source_artifact_id": source_artifact_id,
                    "input_partition_refs": input_partition_refs,
                    "coverage_summary": coverage_summary.model_dump(mode="json"),
                    "source_quality_summary": source_quality_summary.model_dump(mode="json"),
                    "readiness_summary": readiness_summary.model_dump(mode="json"),
                }
            )
            artifact_id = build_artifact_id("state_window", logical_name, content_hash)
            ref = StateWindowArtifactRef(
                artifact_id=artifact_id,
                logical_name=logical_name,
                version=request.state_version,
                content_hash=content_hash,
                symbol=request.symbol,
                clock=clock,
                state_version=request.state_version,
                window_size=window_size,
                date_from=request.date_from,
                date_to=request.date_to,
                source_artifact_id=source_artifact_id,
            )
            manifest = StateArtifactManifest(
                manifest_id=build_manifest_id("state_window", logical_name, content_hash),
                artifact_id=artifact_id,
                artifact_kind="state_window",
                logical_name=logical_name,
                logical_version=request.state_version,
                artifact_uri=str(
                    self._paths.state_window_artifact_root(
                        request.symbol,
                        clock,
                        request.state_version,
                        window_size,
                        artifact_id,
                    )
                ),
                content_hash=content_hash,
                build_id=resolved_build_id,
                created_at=dt.datetime.now(dt.timezone.utc),
                status="accepted",
                dataset_spec_ref=None,
                state_artifact_refs=[source_artifact_id] if isinstance(source_ref, StateArtifactRef) else [],
                code_version=resolved_code_version,
                input_partition_refs=input_partition_refs,
                parent_artifact_refs=[source_artifact_id],
                symbol=request.symbol,
                clock=clock,
                window_size=window_size,
                time_range_start_utc=coverage_summary.time_range_start_utc,
                time_range_end_utc=coverage_summary.time_range_end_utc,
                coverage_summary=coverage_summary,
                source_quality_summary=source_quality_summary,
                readiness_summary=readiness_summary,
                daily_readiness_rollups=daily_readiness_rollups,
                session_readiness_rollups=session_readiness_rollups,
                metadata={
                    "row_count": len(window_df),
                    "column_count": len(window_df.columns),
                    "window_size": window_size,
                    "anchor_start": str(window_df["anchor_ts_utc"].min()) if not window_df.is_empty() else "",
                    "anchor_end": str(window_df["anchor_ts_utc"].max()) if not window_df.is_empty() else "",
                    "source_artifact_id": source_artifact_id,
                    "coverage_summary": coverage_summary.model_dump(mode="json"),
                    "source_quality_summary": source_quality_summary.model_dump(mode="json"),
                    "readiness_summary": readiness_summary.model_dump(mode="json"),
                    "daily_readiness_rollups": [rollup.model_dump(mode="json") for rollup in daily_readiness_rollups],
                    "session_readiness_rollups": [rollup.model_dump(mode="json") for rollup in session_readiness_rollups],
                },
            )
            self._write_state_window_partitions(
                request.symbol,
                clock,
                request.state_version,
                window_size,
                artifact_id,
                window_df,
            )
            manifest_path = write_state_manifest(manifest, self._paths)
            self._register_manifest(manifest, manifest_path)
            results[window_size] = StateWindowMaterializationResult(
                ref=ref,
                manifest=manifest,
                manifest_path=manifest_path,
                coverage_summary=coverage_summary,
                source_quality_summary=source_quality_summary,
                readiness_summary=readiness_summary,
                daily_readiness_rollups=daily_readiness_rollups,
                session_readiness_rollups=session_readiness_rollups,
                window_df=window_df,
            )

        return results

    def load_state_window_artifact(self, ref: StateWindowArtifactRef) -> pl.DataFrame:
        """Load persisted rolling state windows for a typed artifact reference."""
        frames: list[pl.DataFrame] = []
        current = ref.date_from
        while current <= ref.date_to:
            day = self._load_first_available_dir(
                [
                    self._paths.state_window_artifact_dir(
                        ref.symbol,
                        ref.clock,
                        current,
                        ref.state_version,
                        ref.window_size,
                        ref.artifact_id,
                    ),
                    self._paths.state_window_dir(ref.symbol, ref.clock, current, ref.state_version, ref.window_size),
                ]
            )
            if not day.is_empty():
                frames.append(day)
            current += dt.timedelta(days=1)
        if not frames:
            return pl.DataFrame()
        window_df = pl.concat(frames, how="diagonal_relaxed")
        dedup_cols = [column for column in ["window_id", "symbol", "clock", "window_size", "anchor_ts_utc"] if column in window_df.columns]
        if dedup_cols:
            window_df = window_df.unique(subset=dedup_cols, keep="last")
        return window_df.sort("anchor_ts_utc")

    def _persist_state_artifact(
        self,
        *,
        symbol: str,
        clock: str,
        state_version_ref: str,
        date_from: dt.date,
        date_to: dt.date,
        build_id: str,
        dataset_spec_ref: str | None,
        code_version: str,
        merge_config_ref: str | None,
        state_df: pl.DataFrame,
        base_df: pl.DataFrame,
        input_partition_refs: list[str],
    ) -> StateMaterializationResult:
        created_at = dt.datetime.now(dt.timezone.utc)
        logical_name = f"{symbol}.{clock}"
        coverage_summary = build_state_coverage_summary(state_df, clock=clock)
        source_quality_summary = build_state_source_quality_summary(state_df)
        readiness_summary = build_state_readiness_summary(state_df, clock=clock)
        daily_readiness_rollups = build_state_interval_readiness_rollups(state_df, clock=clock, interval_kind="day")
        session_readiness_rollups = build_state_interval_readiness_rollups(state_df, clock=clock, interval_kind="session")
        content_hash = compute_content_hash(
            {
                "artifact_kind": "state",
                "logical_name": logical_name,
                "logical_version": state_version_ref,
                "rows": len(state_df),
                "columns": state_df.columns,
                "time_range_start": str(state_df["ts_utc"].min()) if not state_df.is_empty() else "",
                "time_range_end": str(state_df["ts_utc"].max()) if not state_df.is_empty() else "",
                "input_partition_refs": input_partition_refs,
                "coverage_summary": coverage_summary.model_dump(mode="json"),
                "source_quality_summary": source_quality_summary.model_dump(mode="json"),
                "readiness_summary": readiness_summary.model_dump(mode="json"),
            }
        )
        artifact_id = build_artifact_id("state", logical_name, content_hash)
        ref = StateArtifactRef(
            artifact_id=artifact_id,
            logical_name=logical_name,
            version=state_version_ref,
            content_hash=content_hash,
            symbol=symbol,
            clock=clock,
            state_version=state_version_ref,
            date_from=date_from,
            date_to=date_to,
        )
        manifest = StateArtifactManifest(
            manifest_id=build_manifest_id("state", logical_name, content_hash),
            artifact_id=artifact_id,
            artifact_kind="state",
            logical_name=logical_name,
            logical_version=state_version_ref,
            artifact_uri=str(
                self._paths.state_artifact_root(symbol, clock, state_version_ref, artifact_id)
            ),
            content_hash=content_hash,
            build_id=build_id,
            created_at=created_at,
            status="accepted",
            dataset_spec_ref=dataset_spec_ref,
            code_version=code_version,
            merge_config_ref=merge_config_ref,
            input_partition_refs=input_partition_refs,
            symbol=symbol,
            clock=clock,
            time_range_start_utc=coverage_summary.time_range_start_utc,
            time_range_end_utc=coverage_summary.time_range_end_utc,
            coverage_summary=coverage_summary,
            source_quality_summary=source_quality_summary,
            readiness_summary=readiness_summary,
            daily_readiness_rollups=daily_readiness_rollups,
            session_readiness_rollups=session_readiness_rollups,
            metadata={
                "row_count": len(state_df),
                "column_count": len(state_df.columns),
                "time_range_start": str(state_df["ts_utc"].min()) if not state_df.is_empty() else "",
                "time_range_end": str(state_df["ts_utc"].max()) if not state_df.is_empty() else "",
                "clock": clock,
                "coverage_summary": coverage_summary.model_dump(mode="json"),
                "source_quality_summary": source_quality_summary.model_dump(mode="json"),
                "readiness_summary": readiness_summary.model_dump(mode="json"),
                "daily_readiness_rollups": [rollup.model_dump(mode="json") for rollup in daily_readiness_rollups],
                "session_readiness_rollups": [rollup.model_dump(mode="json") for rollup in session_readiness_rollups],
            },
        )

        self._write_state_partitions(symbol, clock, state_version_ref, artifact_id, state_df)
        manifest_path = write_state_manifest(manifest, self._paths)
        self._register_manifest(manifest, manifest_path)
        return StateMaterializationResult(
            ref=ref,
            artifact_id=artifact_id,
            manifest=manifest,
            manifest_path=manifest_path,
            coverage_summary=coverage_summary,
            source_quality_summary=source_quality_summary,
            readiness_summary=readiness_summary,
            daily_readiness_rollups=daily_readiness_rollups,
            session_readiness_rollups=session_readiness_rollups,
            state_df=state_df,
            base_df=base_df,
        )

    def _register_manifest(self, manifest: StateArtifactManifest, manifest_path: Path) -> None:
        if self._catalog is not None and hasattr(self._catalog, "register_artifact"):
            self._catalog.register_artifact(manifest, str(manifest_path))

    def _load_bars_range(
        self,
        symbol: str,
        clock: str,
        date_from: dt.date,
        date_to: dt.date,
    ) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        current = date_from
        while current <= date_to:
            day = self._store.read_dir(self._paths.built_bars_dir(symbol, clock, current))
            if not day.is_empty():
                frames.append(day)
            current += dt.timedelta(days=1)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed").sort("time_utc")

    def _normalize_base_bars(self, base_df: pl.DataFrame, clock: str) -> pl.DataFrame:
        normalized = validate_bars(base_df)
        tf_seconds = timeframe_to_seconds(clock)
        gap_report = detect_gaps(normalized, clock, tf_seconds)
        if gap_report.missing_bars > 0:
            normalized = fill_bar_gaps(normalized, clock, tf_seconds)
        elif "_filled" not in normalized.columns:
            normalized = normalized.with_columns(pl.lit(False).alias("_filled"))
        return normalized.sort("time_utc")

    def _enrich_base_bars_with_canonical_quality(
        self,
        *,
        symbol: str,
        clock: str,
        date_from: dt.date,
        date_to: dt.date,
        base_df: pl.DataFrame,
    ) -> pl.DataFrame:
        if base_df.is_empty():
            return base_df

        tick_metrics = self._canonical_bar_quality_metrics(symbol=symbol, clock=clock, date_from=date_from, date_to=date_to)
        if tick_metrics.is_empty():
            return base_df
        return base_df.join(tick_metrics, on="time_utc", how="left")

    def _canonical_bar_quality_metrics(
        self,
        *,
        symbol: str,
        clock: str,
        date_from: dt.date,
        date_to: dt.date,
    ) -> pl.DataFrame:
        if clock in {"W1", "MN1"}:
            return pl.DataFrame()

        tf_seconds = timeframe_to_seconds(clock)
        frames: list[pl.DataFrame] = []
        current = date_from
        while current <= date_to:
            day = self._store.read_dir(self._paths.canonical_ticks_dir(symbol, current))
            if not day.is_empty():
                frames.append(day)
            current += dt.timedelta(days=1)
        if not frames:
            return pl.DataFrame()

        ticks = pl.concat(frames, how="diagonal_relaxed")
        required = {"ts_msc", "quality_score"}
        if not required.issubset(set(ticks.columns)):
            return pl.DataFrame()

        dual_expr = (
            (pl.col("source_secondary").fill_null("") != "")
            | (pl.col("merge_mode").fill_null("") == "best")
        ) if {"source_secondary", "merge_mode"}.issubset(set(ticks.columns)) else pl.lit(False)
        conflict_expr = pl.col("conflict_flag").cast(pl.Float64) if "conflict_flag" in ticks.columns else pl.lit(0.0)
        quality_expr = pl.when(pl.col("quality_score") <= 1.5).then(pl.col("quality_score") * 100.0).otherwise(pl.col("quality_score"))

        return (
            ticks.with_columns([
                pl.from_epoch((pl.col("ts_msc") // (tf_seconds * 1000)) * (tf_seconds * 1000), time_unit="ms")
                .dt.replace_time_zone("UTC")
                .alias("time_utc"),
                dual_expr.cast(pl.Float64).alias("_tick_dual_source_flag"),
                quality_expr.cast(pl.Float64).alias("_tick_quality_score_100"),
                conflict_expr.alias("_tick_conflict_flag"),
            ])
            .group_by("time_utc")
            .agg([
                pl.col("_tick_quality_score_100").mean().alias("_canonical_quality_mean"),
                pl.col("_tick_dual_source_flag").mean().alias("_canonical_dual_source_ratio"),
                pl.col("_tick_dual_source_flag").max().cast(pl.Int64).alias("_canonical_dual_source_present"),
                pl.col("_tick_conflict_flag").mean().alias("_canonical_conflict_ratio"),
            ])
            .sort("time_utc")
        )

    def _build_bar_state_rows(
        self,
        symbol: str,
        clock: str,
        state_version_ref: str,
        base_df: pl.DataFrame,
    ) -> pl.DataFrame:
        tf_seconds = timeframe_to_seconds(clock)
        expected_interval_ms = state_resolution_ms(clock)
        prev_ts_msc: int | None = None
        rows: list[dict[str, object]] = []

        for row in base_df.iter_rows(named=True):
            ts_utc = row["time_utc"]
            if not isinstance(ts_utc, dt.datetime):
                raise TypeError("Base bar time_utc must be a timezone-aware datetime")

            spread = self._resolve_spread(row)
            bid = self._resolve_bid(row, spread)
            ask = self._resolve_ask(row, spread, bid)
            gap_fill_flag = bool(row.get("_filled", False))
            raw_source_count = row.get("source_count", 1)
            source_count = int(raw_source_count) if raw_source_count is not None else (0 if gap_fill_flag else 1)
            canonical_dual_present = int(row.get("_canonical_dual_source_present", 0) or 0)
            if source_count < 2 and canonical_dual_present > 0:
                source_count = 2
            conflict_count = int(row.get("conflict_count", 0) or 0)
            canonical_conflict_ratio = row.get("_canonical_conflict_ratio")
            if conflict_count <= 0 and canonical_conflict_ratio is not None and float(canonical_conflict_ratio) > 0.0:
                conflict_count = 1
            dual_ratio = row.get("dual_source_ratio")
            ts_msc = int(ts_utc.timestamp() * 1000)
            observed_interval_ms = 0 if prev_ts_msc is None else max(ts_msc - prev_ts_msc, 0)
            primary_staleness_ms = observed_interval_ms
            prev_ts_msc = ts_msc
            canonical_quality_mean = row.get("_canonical_quality_mean")
            canonical_quality_f = float(canonical_quality_mean) if canonical_quality_mean is not None else None
            canonical_dual_ratio = row.get("_canonical_dual_source_ratio")
            dual_ratio_f = float(canonical_dual_ratio) if canonical_dual_ratio is not None else (float(dual_ratio) if dual_ratio is not None else None)
            merge_mode = "conflict" if conflict_count > 0 else "best" if source_count >= 2 else "single"
            quality_score = self._quality_score(
                source_count=source_count,
                conflict_count=conflict_count,
                dual_source_ratio=dual_ratio_f,
                filled=gap_fill_flag,
                canonical_quality_score=canonical_quality_f,
                observed_interval_ms=observed_interval_ms,
                expected_interval_ms=expected_interval_ms,
            )
            observed_observations = 0 if gap_fill_flag else 1
            missing_observations = 1 - observed_observations
            window_completeness = float(observed_observations)
            source_quality_hint = canonical_quality_f if canonical_quality_f is not None else quality_score
            if gap_fill_flag:
                source_quality_hint = max(0.0, source_quality_hint * 0.7)
            source_participation_score = snapshot_source_participation_score(
                source_count=source_count,
                conflict_flag=conflict_count > 0,
                disagreement_bps=None,
                gap_fill_flag=gap_fill_flag,
                quality_score=quality_score,
            )
            overlap_confidence = snapshot_overlap_confidence_hint(
                source_count=source_count,
                source_participation_score=source_participation_score,
                window_completeness=window_completeness,
                source_quality_hint=source_quality_hint,
            )
            source_primary = "gap_fill" if gap_fill_flag else "canonical"
            trust_flags: list[str] = []
            if gap_fill_flag:
                trust_flags.append("filled_gap")
            if source_count <= 0:
                trust_flags.append("no_direct_source")
            if conflict_count > 0:
                trust_flags.append("conflict")

            snapshot = StateSnapshot(
                state_version=state_version_ref,
                snapshot_id=f"state:{symbol}:{clock}:{ts_utc.isoformat()}",
                symbol=symbol,
                ts_utc=ts_utc,
                ts_msc=ts_msc,
                clock=clock,
                window_start_utc=ts_utc,
                window_end_utc=ts_utc + dt.timedelta(seconds=tf_seconds) - dt.timedelta(milliseconds=1),
                bid=bid,
                ask=ask,
                mid=(bid + ask) / 2.0,
                spread=ask - bid,
                source_primary=source_primary,
                source_secondary="multi" if source_count >= 2 else None,
                source_count=source_count,
                merge_mode=merge_mode,
                conflict_flag=conflict_count > 0,
                disagreement_bps=None,
                spread_disagreement_bps=None,
                broker_a_mid=None,
                broker_b_mid=None,
                broker_a_spread=None,
                broker_b_spread=None,
                primary_staleness_ms=primary_staleness_ms,
                secondary_staleness_ms=None,
                source_offset_ms=None,
                expected_interval_ms=expected_interval_ms,
                observed_interval_ms=observed_interval_ms,
                quality_score=quality_score,
                source_quality_hint=source_quality_hint,
                source_participation_score=source_participation_score,
                overlap_confidence_hint=overlap_confidence,
                expected_observations=1,
                observed_observations=observed_observations,
                missing_observations=missing_observations,
                window_completeness=window_completeness,
                gap_fill_flag=gap_fill_flag,
                session_code=session_code(ts_utc),
                trust_flags=trust_flags,
                provenance_refs=self._bar_provenance_refs(symbol, clock, ts_utc.date()),
            )
            rows.append(snapshot.model_dump())

        return pl.DataFrame(rows).sort("ts_utc")

    def _collect_bar_input_partition_refs(
        self,
        symbol: str,
        clock: str,
        date_from: dt.date,
        date_to: dt.date,
    ) -> list[str]:
        refs: list[str] = []
        current = date_from
        while current <= date_to:
            bar_dir = self._paths.built_bars_dir(symbol, clock, current)
            if bar_dir.exists():
                refs.append(str(bar_dir))
            merge_qa_dir = self._paths.merge_qa_dir(symbol, current)
            if merge_qa_dir.exists():
                refs.append(str(merge_qa_dir))
            current += dt.timedelta(days=1)
        return sorted(set(refs))

    def _collect_tick_input_partition_refs(
        self,
        symbol: str,
        date_from: dt.date,
        date_to: dt.date,
    ) -> list[str]:
        refs: list[str] = []
        current = date_from
        while current <= date_to:
            tick_dir = self._paths.canonical_ticks_dir(symbol, current)
            if tick_dir.exists():
                refs.append(str(tick_dir))
            current += dt.timedelta(days=1)
        return sorted(set(refs))

    def _collect_state_input_partition_refs(
        self,
        symbol: str,
        clock: str,
        state_version: str,
        date_from: dt.date,
        date_to: dt.date,
    ) -> list[str]:
        refs: list[str] = []
        current = date_from
        while current <= date_to:
            state_dir = self._paths.state_dir(symbol, clock, current, state_version)
            if state_dir.exists():
                refs.append(str(state_dir))
            current += dt.timedelta(days=1)
        return sorted(set(refs))

    def _bar_provenance_refs(self, symbol: str, clock: str, day: dt.date) -> list[str]:
        refs = [str(self._paths.built_bars_dir(symbol, clock, day))]
        merge_qa_dir = self._paths.merge_qa_dir(symbol, day)
        if merge_qa_dir.exists():
            refs.append(str(merge_qa_dir))
        return refs

    def _write_state_partitions(
        self,
        symbol: str,
        clock: str,
        state_version_ref: str,
        artifact_id: str,
        state_df: pl.DataFrame,
    ) -> None:
        dated = state_df.with_columns(pl.col("ts_utc").dt.date().alias("_date"))
        for date_val in dated["_date"].unique().sort().to_list():
            day_df = dated.filter(pl.col("_date") == date_val).drop("_date")
            self._reset_partition_dir(self._paths.state_dir(symbol, clock, date_val, state_version_ref))
            self._reset_partition_dir(
                self._paths.state_artifact_dir(symbol, clock, date_val, state_version_ref, artifact_id)
            )
            self._store.write(day_df, self._paths.state_file(symbol, clock, date_val, state_version_ref))
            self._store.write(
                day_df,
                self._paths.state_artifact_file(symbol, clock, date_val, state_version_ref, artifact_id),
            )

    def _write_state_window_partitions(
        self,
        symbol: str,
        clock: str,
        state_version: str,
        window_size: str,
        artifact_id: str,
        window_df: pl.DataFrame,
    ) -> None:
        if window_df.is_empty():
            return
        dated = window_df.with_columns(pl.col("anchor_ts_utc").dt.date().alias("_date"))
        for date_val in dated["_date"].unique().sort().to_list():
            day_df = dated.filter(pl.col("_date") == date_val).drop("_date")
            self._reset_partition_dir(self._paths.state_window_dir(symbol, clock, date_val, state_version, window_size))
            self._reset_partition_dir(
                self._paths.state_window_artifact_dir(
                    symbol,
                    clock,
                    date_val,
                    state_version,
                    window_size,
                    artifact_id,
                )
            )
            self._store.write(
                day_df,
                self._paths.state_window_file(symbol, clock, date_val, state_version, window_size),
            )
            self._store.write(
                day_df,
                self._paths.state_window_artifact_file(
                    symbol,
                    clock,
                    date_val,
                    state_version,
                    window_size,
                    artifact_id,
                ),
            )

    @staticmethod
    def _reset_partition_dir(directory: Path) -> None:
        if directory.exists():
            shutil.rmtree(directory)

    def _load_first_available_dir(self, directories: list[Path]) -> pl.DataFrame:
        for directory in directories:
            day = self._store.read_dir(directory)
            if not day.is_empty():
                return day
        return pl.DataFrame()

    @staticmethod
    def _validate_request_date_range(
        request_date_from: dt.date,
        request_date_to: dt.date,
        source_date_from: dt.date,
        source_date_to: dt.date,
        *,
        source_name: str,
    ) -> None:
        if request_date_from < source_date_from or request_date_to > source_date_to:
            raise ValueError(
                f"StateWindowRequest date range {request_date_from.isoformat()}..{request_date_to.isoformat()} "
                f"must lie within the source {source_name} range "
                f"{source_date_from.isoformat()}..{source_date_to.isoformat()}"
            )

    @staticmethod
    def _filter_window_anchors_by_date(window_df: pl.DataFrame, date_from: dt.date, date_to: dt.date) -> pl.DataFrame:
        if window_df.is_empty():
            return window_df
        return window_df.filter(
            pl.col("anchor_ts_utc").dt.date().is_between(date_from, date_to, closed="both")
        ).sort("anchor_ts_utc")

    @staticmethod
    def _resolve_spread(row: dict[str, object]) -> float:
        spread = float(row.get("spread_mean", 0.0) or 0.0)
        if spread > 0:
            return spread
        bid = row.get("bid_close")
        ask = row.get("ask_close")
        if bid is not None and ask is not None:
            return max(float(ask) - float(bid), 1e-9)
        return 1e-6

    @staticmethod
    def _resolve_bid(row: dict[str, object], spread: float) -> float:
        bid = row.get("bid_close")
        if bid is not None:
            return float(bid)
        close = float(row.get("close", 0.0) or 0.0)
        return max(close - (spread / 2.0), 1e-9)

    @staticmethod
    def _resolve_ask(row: dict[str, object], spread: float, bid: float) -> float:
        ask = row.get("ask_close")
        if ask is not None:
            return float(ask)
        close = float(row.get("close", 0.0) or 0.0)
        return max(close + (spread / 2.0), bid)

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return symbol.strip().upper()

    @staticmethod
    def _normalize_clock(clock: str) -> str:
        return clock.strip().upper()

    @staticmethod
    def _quality_score(
        *,
        source_count: int,
        conflict_count: int,
        dual_source_ratio: float | None,
        filled: bool,
        canonical_quality_score: float | None,
        observed_interval_ms: int,
        expected_interval_ms: int,
    ) -> float:
        if canonical_quality_score is not None:
            base = max(0.0, min(float(canonical_quality_score), 100.0))
        else:
            base = 78.0 if source_count <= 1 else 84.0

        dual_component = min(max(dual_source_ratio or 0.0, 0.0) * 12.0, 12.0)
        source_component = 4.0 if source_count >= 2 else 0.0
        conflict_penalty = 18.0 if conflict_count > 0 else 0.0
        filled_penalty = 22.0 if filled else 0.0
        staleness_penalty = 0.0
        if expected_interval_ms > 0 and observed_interval_ms > expected_interval_ms:
            interval_excess = observed_interval_ms / expected_interval_ms
            staleness_penalty = min((interval_excess - 1.0) * 4.0, 10.0)

        return max(
            0.0,
            min(100.0, base + dual_component + source_component - conflict_penalty - filled_penalty - staleness_penalty),
        )


def load_state_artifact(
    paths: StoragePaths,
    store: ParquetStore,
    ref: StateArtifactRef,
) -> pl.DataFrame:
    """Module-level public loader for persisted state snapshots."""
    return StateService(paths, store).load_state_artifact(ref)


def load_state_window_artifact(
    paths: StoragePaths,
    store: ParquetStore,
    ref: StateWindowArtifactRef,
) -> pl.DataFrame:
    """Module-level public loader for persisted rolling state windows."""
    return StateService(paths, store).load_state_window_artifact(ref)


def materialize_state_windows(
    paths: StoragePaths,
    store: ParquetStore,
    source_ref: StateArtifactRef | TickArtifactRef,
    *,
    request: StateWindowRequest,
    catalog: Any | None = None,
) -> dict[str, StateWindowMaterializationResult]:
    """Module-level public helper for rolling state window materialization."""
    return StateService(paths, store, catalog=catalog).materialize_state_windows(source_ref, request=request)
