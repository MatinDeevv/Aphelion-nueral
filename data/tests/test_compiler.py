"""Tests for compiler success, rejection, inspection, and diff behavior."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from mt5pipe.catalog.sqlite import CatalogDB
from mt5pipe.compiler.manifest import build_stage_artifact_id, build_stage_manifest_id, compute_content_hash, write_manifest_sidecar
from mt5pipe.compiler.models import LineageManifest
from mt5pipe.compiler.service import compile_dataset_spec, diff_artifacts, inspect_artifact
from mt5pipe.features.public import FeatureSpec
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths


UTC = dt.timezone.utc
DATASET_JOIN_KEYS = ["symbol", "timeframe", "time_utc"]


def _make_time_index(start: dt.datetime, rows: int, step_minutes: int) -> list[dt.datetime]:
    return [start + dt.timedelta(minutes=step_minutes * i) for i in range(rows)]


def _make_bars(start: dt.datetime, rows: int, timeframe: str, step_minutes: int) -> pl.DataFrame:
    times = _make_time_index(start, rows, step_minutes)
    base_open = [3000.0 + i * 0.002 + ((i % 30) - 15) * 0.03 for i in range(rows)]
    base_close = [price + (0.05 if i % 2 == 0 else -0.05) for i, price in enumerate(base_open)]
    base_high = [max(base_open[i], base_close[i]) + 0.05 + (i % 5) * 0.005 for i in range(rows)]
    base_low = [min(base_open[i], base_close[i]) - 0.05 - (i % 5) * 0.005 for i in range(rows)]
    dual_ratio = [0.75 + (i % 4) * 0.02 for i in range(rows)]
    return pl.DataFrame(
        {
            "symbol": ["XAUUSD"] * rows,
            "timeframe": [timeframe] * rows,
            "time_utc": times,
            "open": base_open,
            "high": base_high,
            "low": base_low,
            "close": base_close,
            "tick_count": [12] * rows,
            "bid_open": [base_open[i] - 0.05 for i in range(rows)],
            "ask_open": [base_open[i] + 0.05 for i in range(rows)],
            "bid_close": [base_close[i] - 0.05 for i in range(rows)],
            "ask_close": [base_close[i] + 0.05 for i in range(rows)],
            "spread_mean": [0.1 + (i % 5) * 0.001 for i in range(rows)],
            "spread_max": [0.12 + (i % 5) * 0.001 for i in range(rows)],
            "spread_min": [0.08 + (i % 5) * 0.001 for i in range(rows)],
            "mid_return": [0.0001 + (i % 7) * 0.00001 for i in range(rows)],
            "realized_vol": [0.0005 + (i % 9) * 0.00001 for i in range(rows)],
            "volume_sum": [10.0] * rows,
            "source_count": [1 + (i % 2) for i in range(rows)],
            "conflict_count": [i % 3 for i in range(rows)],
            "dual_source_ticks": [9 + (i % 3) for i in range(rows)],
            "secondary_present_ticks": [9 + (i % 3) for i in range(rows)],
            "dual_source_ratio": dual_ratio,
        }
    )


def _write_bars_by_date(df: pl.DataFrame, paths: StoragePaths, store: ParquetStore, timeframe: str) -> None:
    dated = df.with_columns(pl.col("time_utc").dt.date().alias("_date"))
    for date_val in dated["_date"].unique().sort().to_list():
        day_df = dated.filter(pl.col("_date") == date_val).drop("_date")
        store.write(day_df, paths.built_bars_file("XAUUSD", timeframe, date_val))


def _write_merge_qa(date: dt.date, paths: StoragePaths, store: ParquetStore, dual_source_ratio: float = 0.20) -> None:
    df = pl.DataFrame(
        [
            {
                "time_utc": dt.datetime.combine(date, dt.time(0, 0), tzinfo=UTC),
                "date": date.isoformat(),
                "symbol": "XAUUSD",
                "broker_a_id": "broker_a",
                "broker_b_id": "broker_b",
                "dual_source_ratio": dual_source_ratio,
                "conflicts": 0,
            }
        ]
    )
    store.write(df, paths.merge_qa_file("XAUUSD", date))


def _build_state_fixture(base_df: pl.DataFrame, state_version_ref: str = "state.default@1.0.0") -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index, row in enumerate(base_df.sort("time_utc").iter_rows(named=True)):
        ts_utc = row["time_utc"]
        close = float(row["close"])
        spread = float(row["spread_mean"])
        source_count = int(row["source_count"])
        conflict_count = int(row["conflict_count"])
        rows.append(
            {
                "schema_version": "1.0.0",
                "state_version": state_version_ref,
                "snapshot_id": f"snapshot-{index}",
                "symbol": row["symbol"],
                "ts_utc": ts_utc,
                "ts_msc": int(ts_utc.timestamp() * 1000),
                "clock": row["timeframe"],
                "window_start_utc": ts_utc,
                "window_end_utc": ts_utc,
                "bid": close - spread / 2.0,
                "ask": close + spread / 2.0,
                "mid": close,
                "spread": spread,
                "source_primary": "broker_a",
                "source_secondary": "broker_b" if source_count > 1 else "",
                "source_count": source_count,
                "merge_mode": "fixture",
                "conflict_flag": conflict_count > 0,
                "disagreement_bps": conflict_count * 0.5,
                "spread_disagreement_bps": spread * 10.0,
                "broker_a_mid": close,
                "broker_b_mid": close + 0.01 if source_count > 1 else close,
                "broker_a_spread": spread,
                "broker_b_spread": spread + 0.01 if source_count > 1 else spread,
                "primary_staleness_ms": 0,
                "secondary_staleness_ms": 0,
                "source_offset_ms": 0,
                "quality_score": max(60.0, 82.0 - conflict_count * 4.0),
                "source_quality_hint": "fixture",
                "expected_observations": 1,
                "observed_observations": 1,
                "missing_observations": 0,
                "window_completeness": 1.0,
                "session_code": "fixture",
                "event_flags": [],
                "trust_flags": [],
                "provenance_refs": [f"bar://{row['symbol']}/{row['timeframe']}/{ts_utc.isoformat()}"],
            }
        )
    return pl.DataFrame(rows)


def _write_state_fixture_partitions(
    base_df: pl.DataFrame,
    paths: StoragePaths,
    store: ParquetStore,
    *,
    state_version_ref: str = "state.default@1.0.0",
) -> pl.DataFrame:
    state_df = _build_state_fixture(base_df, state_version_ref=state_version_ref)
    dated = state_df.with_columns(pl.col("ts_utc").dt.date().alias("_date"))
    for date_val in dated["_date"].unique().sort().to_list():
        day_df = dated.filter(pl.col("_date") == date_val).drop("_date")
        store.write(day_df, paths.state_file("XAUUSD", "M1", date_val, state_version_ref))
    return state_df


def _collect_partition_refs(paths: StoragePaths, date_from: dt.date, date_to: dt.date) -> list[str]:
    refs: list[str] = []
    current = date_from
    while current <= date_to:
        refs.append(str(paths.built_bars_dir("XAUUSD", "M1", current)))
        current += dt.timedelta(days=1)
    return refs


def _register_state_artifact(
    *,
    paths: StoragePaths,
    store: ParquetStore,
    catalog: CatalogDB,
    base_df: pl.DataFrame,
    state_version_ref: str,
    date_from: dt.date,
    date_to: dt.date,
    build_id: str,
    dataset_spec_ref: str,
) -> tuple[str, LineageManifest, pl.DataFrame]:
    state_df = _write_state_fixture_partitions(base_df, paths, store, state_version_ref=state_version_ref)
    input_partition_refs = _collect_partition_refs(paths, date_from, date_to)
    created_at = dt.datetime.now(UTC)
    logical_name = "XAUUSD.M1"
    content_hash = compute_content_hash(
        {
            "artifact_kind": "state",
            "state_version": state_version_ref,
            "rows": len(state_df),
            "columns": state_df.columns,
            "input_partition_refs": input_partition_refs,
        }
    )
    artifact_id = build_stage_artifact_id("state", logical_name, created_at, content_hash)
    artifact_uri = paths.state_artifact_root("XAUUSD", "M1", state_version_ref, artifact_id)
    dated = state_df.with_columns(pl.col("ts_utc").dt.date().alias("_date"))
    for date_val in dated["_date"].unique().sort().to_list():
        day_df = dated.filter(pl.col("_date") == date_val).drop("_date")
        store.write(day_df, paths.state_artifact_file("XAUUSD", "M1", date_val, state_version_ref, artifact_id))
    manifest = LineageManifest(
        manifest_id=build_stage_manifest_id("state", logical_name, created_at, content_hash),
        artifact_id=artifact_id,
        artifact_kind="state",
        logical_name=logical_name,
        logical_version=state_version_ref,
        artifact_uri=str(artifact_uri),
        content_hash=content_hash,
        build_id=build_id,
        created_at=created_at,
        status="accepted",
        dataset_spec_ref=dataset_spec_ref,
        state_artifact_refs=[],
        feature_spec_refs=[],
        label_pack_ref=None,
        truth_report_ref=None,
        code_version="workspace-local-no-git",
        merge_config_ref="merge.default@test",
        input_partition_refs=input_partition_refs,
        parent_artifact_refs=[],
        metadata={
            "row_count": len(state_df),
            "column_count": len(state_df.columns),
            "time_range_start": str(state_df["ts_utc"].min()) if not state_df.is_empty() else "",
            "time_range_end": str(state_df["ts_utc"].max()) if not state_df.is_empty() else "",
            "symbol": "XAUUSD",
            "clock": "M1",
            "fixture": True,
        },
    )
    manifest_path = write_manifest_sidecar(manifest, paths)
    catalog.register_artifact(manifest, str(manifest_path), detail="registered fixture state artifact")
    return artifact_id, manifest, state_df


def _write_project_config(project_root: Path, storage_root: Path) -> Path:
    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "pipeline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "brokers: {}",
                "storage:",
                f"  root: \"{storage_root.as_posix()}\"",
                "  checkpoint_db: \"checkpoints.db\"",
                "  parquet_row_group_size: 1000",
                "  compression: \"snappy\"",
                "dataset:",
                "  base_timeframe: \"M1\"",
                "  context_timeframes:",
                "    - \"M5\"",
                "    - \"M15\"",
                "    - \"H1\"",
                "    - \"H4\"",
                "    - \"D1\"",
                "logging:",
                "  level: \"INFO\"",
                "  json_output: false",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_spec(
    path: Path,
    *,
    dataset_name: str = "xau_m1_core",
    version: str,
    selectors: list[str],
    embargo_rows: int = 240,
    date_from: str = "2026-04-01",
    date_to: str = "2026-04-02",
    state_version_ref: str | None = "state.default@1.0.0",
    state_artifact_ref: str | None = None,
    feature_artifact_refs: list[str] | None = None,
    label_pack_ref: str = "core_tb_volscaled@1.0.0",
    required_raw_brokers: list[str] | None = None,
    require_synchronized_raw_coverage: bool = False,
    require_dual_source_overlap: bool = False,
    min_dual_source_ratio: float = 0.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        'schema_version: "1.0.0"',
        f'dataset_name: "{dataset_name}"',
        f'version: "{version}"',
        'description: "compiler test dataset"',
        'symbols:',
        '  - "XAUUSD"',
        f'date_from: "{date_from}"',
        f'date_to: "{date_to}"',
        'base_clock: "M1"',
    ]
    if state_version_ref:
        lines.append(f'state_version_ref: "{state_version_ref}"')
    if state_artifact_ref:
        lines.append(f'state_artifact_ref: "{state_artifact_ref}"')
    lines.extend([
        'feature_selectors:',
        *[f'  - "{selector}"' for selector in selectors],
    ])
    if feature_artifact_refs:
        lines.extend([
            'feature_artifact_refs:',
            *[f'  - "{ref}"' for ref in feature_artifact_refs],
        ])
    if required_raw_brokers:
        lines.extend([
            "required_raw_brokers:",
            *[f'  - "{broker}"' for broker in required_raw_brokers],
        ])
    if require_synchronized_raw_coverage:
        lines.append("require_synchronized_raw_coverage: true")
    if require_dual_source_overlap:
        lines.append("require_dual_source_overlap: true")
    if min_dual_source_ratio > 0.0:
        lines.append(f"min_dual_source_ratio: {min_dual_source_ratio}")
    lines.extend([
        f'label_pack_ref: "{label_pack_ref}"',
        'filters:',
        '  - "exclude:filled_rows"',
        'split_policy: "temporal_holdout"',
        'train_ratio: 0.70',
        'val_ratio: 0.15',
        'test_ratio: 0.15',
        f"embargo_rows: {embargo_rows}",
        'truth_policy_ref: "truth.default@1.0.0"',
        "publish_on_accept: true",
    ])
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _seed_project_data(
    storage_root: Path,
    *,
    start_date: dt.date = dt.date(2026, 4, 1),
    days: int = 2,
) -> StoragePaths:
    paths = StoragePaths(storage_root)
    store = ParquetStore(compression="snappy", row_group_size=1000)

    start_dt = dt.datetime.combine(start_date, dt.time(0, 0), tzinfo=UTC)

    def _rows(step_minutes: int) -> int:
        return max(1, (days * 24 * 60) // step_minutes)

    m1_df = _make_bars(start_dt, _rows(1), "M1", 1)
    _write_bars_by_date(m1_df, paths, store, "M1")
    _write_bars_by_date(_make_bars(start_dt, _rows(5), "M5", 5), paths, store, "M5")
    _write_bars_by_date(_make_bars(start_dt, _rows(15), "M15", 15), paths, store, "M15")
    _write_bars_by_date(_make_bars(start_dt, _rows(60), "H1", 60), paths, store, "H1")
    _write_bars_by_date(_make_bars(start_dt, _rows(240), "H4", 240), paths, store, "H4")
    _write_bars_by_date(_make_bars(start_dt, max(days, 1), "D1", 24 * 60), paths, store, "D1")
    _write_state_fixture_partitions(m1_df, paths, store)
    for offset in range(days):
        _write_merge_qa(start_date + dt.timedelta(days=offset), paths, store)
    return paths


def _phase3_feature_specs() -> list[FeatureSpec]:
    return [
        FeatureSpec(
            feature_name="state_disagreement",
            family="disagreement",
            version="1.0.0",
            description="Fixture disagreement family artifact for compiler integration tests",
            input_contract="StateSnapshot",
            input_clock="M1",
            output_clock="M1",
            builder_ref="fixture://disagreement",
            output_columns=["disagreement_bps_feature", "disagreement_conflict_flag"],
            dependencies=["time_utc"],
            missingness_policy="fail",
            qa_policy_ref="qa.feature.default@1.0.0",
            status="stable",
        ),
        FeatureSpec(
            feature_name="event_shape",
            family="event_shape",
            version="1.0.0",
            description="Fixture event-shape family artifact for compiler integration tests",
            input_contract="BuiltBar",
            input_clock="M1",
            output_clock="M1",
            builder_ref="fixture://event_shape",
            output_columns=["event_body_ratio", "event_true_range"],
            dependencies=["open", "high", "low", "close"],
            missingness_policy="fail",
            qa_policy_ref="qa.feature.default@1.0.0",
            status="stable",
        ),
        FeatureSpec(
            feature_name="micro_entropy",
            family="entropy",
            version="1.0.0",
            description="Fixture entropy family artifact for compiler integration tests",
            input_contract="BuiltBar",
            input_clock="M1",
            output_clock="M1",
            builder_ref="fixture://entropy",
            output_columns=["entropy_return_30", "entropy_sign_30"],
            dependencies=["mid_return"],
            missingness_policy="fail",
            qa_policy_ref="qa.feature.default@1.0.0",
            status="stable",
        ),
    ]


def _build_phase3_feature_frame(base_df: pl.DataFrame, spec: FeatureSpec) -> pl.DataFrame:
    if spec.family == "disagreement":
        return base_df.with_columns(
            [
                (pl.col("conflict_count").cast(pl.Float64) * 0.5).alias("disagreement_bps_feature"),
                (pl.col("conflict_count") > 0).cast(pl.Int8).alias("disagreement_conflict_flag"),
            ]
        ).select([*DATASET_JOIN_KEYS, *spec.output_columns])
    if spec.family == "event_shape":
        return base_df.with_columns(
            [
                (
                    (pl.col("close") - pl.col("open")).abs()
                    / (pl.col("high") - pl.col("low")).clip(lower_bound=1e-9)
                ).alias("event_body_ratio"),
                (pl.col("high") - pl.col("low")).alias("event_true_range"),
            ]
        ).select([*DATASET_JOIN_KEYS, *spec.output_columns])
    if spec.family == "entropy":
        sign_series = pl.when(pl.col("mid_return") >= 0).then(1.0).otherwise(-1.0)
        return base_df.with_columns(
            [
                pl.col("mid_return").abs().rolling_mean(window_size=30).fill_null(0.0).alias("entropy_return_30"),
                sign_series.rolling_mean(window_size=30).abs().fill_null(0.0).alias("entropy_sign_30"),
            ]
        ).select([*DATASET_JOIN_KEYS, *spec.output_columns])
    raise KeyError(spec.family)


def _register_feature_artifact(
    *,
    paths: StoragePaths,
    store: ParquetStore,
    catalog: CatalogDB,
    spec: FeatureSpec,
    frame: pl.DataFrame,
    state_artifact_id: str,
    input_partition_refs: list[str],
) -> str:
    created_at = dt.datetime.now(UTC)
    content_hash = compute_content_hash(
        {
            "artifact_kind": "feature_view",
            "feature_spec": spec.model_dump(mode="json"),
            "rows": len(frame),
            "columns": frame.columns,
            "state_artifact_id": state_artifact_id,
            "input_partition_refs": input_partition_refs,
        }
    )
    artifact_id = build_stage_artifact_id("feature_view", spec.key, created_at, content_hash)
    artifact_uri = paths.feature_artifact_root(spec.key, spec.output_clock, artifact_id)
    dated = frame.with_columns(pl.col("time_utc").dt.date().alias("_date"))
    for date_val in dated["_date"].unique().sort().to_list():
        day_df = dated.filter(pl.col("_date") == date_val).drop("_date")
        store.write(day_df, paths.feature_view_file(spec.key, spec.output_clock, date_val))
        store.write(day_df, paths.feature_artifact_file(spec.key, spec.output_clock, artifact_id, date_val))

    manifest = LineageManifest(
        manifest_id=build_stage_manifest_id("feature_view", spec.key, created_at, content_hash),
        artifact_id=artifact_id,
        artifact_kind="feature_view",
        logical_name=spec.key,
        logical_version=spec.version,
        artifact_uri=str(artifact_uri),
        content_hash=content_hash,
        build_id="build.phase3.fixtures",
        created_at=created_at,
        status="accepted",
        feature_spec_refs=[spec.key],
        state_artifact_refs=[state_artifact_id],
        code_version="workspace-local-no-git",
        input_partition_refs=input_partition_refs,
        parent_artifact_refs=[state_artifact_id],
        metadata={
            "row_count": len(frame),
            "column_count": len(frame.columns),
            "output_columns": spec.output_columns,
        },
    )
    manifest_path = write_manifest_sidecar(manifest, paths)
    catalog.register_feature_specs([spec])
    catalog.register_artifact(manifest, str(manifest_path), detail="registered phase3 fixture feature artifact")
    return artifact_id


def _prepare_phase3_artifacts(project_root: Path, paths: StoragePaths, store: ParquetStore) -> tuple[str, list[str]]:
    catalog = CatalogDB(paths.catalog_db_path())
    try:
        base_df = pl.concat(
            [
                store.read_dir(paths.built_bars_dir("XAUUSD", "M1", dt.date(2026, 4, 1))),
                store.read_dir(paths.built_bars_dir("XAUUSD", "M1", dt.date(2026, 4, 2))),
            ],
            how="diagonal_relaxed",
        ).sort("time_utc")
        state_artifact_id, state_manifest, _ = _register_state_artifact(
            paths=paths,
            store=store,
            catalog=catalog,
            base_df=base_df,
            state_version_ref="state.default@1.0.0",
            date_from=dt.date(2026, 4, 1),
            date_to=dt.date(2026, 4, 2),
            build_id="build.phase3.fixtures",
            dataset_spec_ref="fixture.nonhuman@1.0.0",
        )

        artifact_ids: list[str] = []
        for spec in _phase3_feature_specs():
            frame = _build_phase3_feature_frame(base_df, spec)
            artifact_ids.append(
                _register_feature_artifact(
                    paths=paths,
                    store=store,
                    catalog=catalog,
                    spec=spec,
                    frame=frame,
                    state_artifact_id=state_artifact_id,
                    input_partition_refs=state_manifest.input_partition_refs,
                )
            )

        return state_artifact_id, artifact_ids
    finally:
        catalog.close()


def test_compile_dataset_spec_builds_real_artifact_and_supports_inspect_diff(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    storage_root = project_root / "local_data" / "pipeline_data"
    storage_root.mkdir(parents=True, exist_ok=True)
    paths = _seed_project_data(storage_root)
    _write_project_config(project_root, storage_root)

    spec_one = project_root / "config" / "datasets" / "xau_m1_core_v1.yaml"
    spec_two = project_root / "config" / "datasets" / "xau_m1_core_v2.yaml"
    _write_spec(spec_one, version="1.0.0", selectors=["time/*", "session/*", "quality/*", "htf_context/*"])
    _write_spec(spec_two, version="1.0.1", selectors=["time/*", "session/*", "quality/*"])

    monkeypatch.chdir(project_root)

    result_one = compile_dataset_spec(spec_one)
    assert result_one.manifest.status == "published"
    assert result_one.trust_report.accepted_for_publication is True
    assert result_one.published_aliases == [
        "dataset://xau_m1_core@1.0.0",
        "dataset://xau_m1_core:latest",
    ]
    assert result_one.status_history == ["building", "truth_pending", "accepted", "published"]
    assert result_one.split_row_counts["train"] > 0
    assert result_one.split_row_counts["val"] > 0
    assert result_one.split_row_counts["test"] > 0
    assert paths.compiler_dataset_dir("xau_m1_core", result_one.artifact_id, "train").exists()

    second_result_same_spec = compile_dataset_spec(spec_one)
    assert second_result_same_spec.artifact_id == result_one.artifact_id

    catalog = CatalogDB(paths.catalog_db_path())
    try:
        artifact = catalog.get_artifact(result_one.artifact_id)
        assert artifact is not None
        assert artifact.status == "published"

        build = catalog.get_build_run(result_one.build_id)
        assert build is not None
        assert build.status == "published"

        stored_spec = catalog.get_dataset_spec(result_one.spec.key)
        assert stored_spec is not None
        assert stored_spec.dataset_name == "xau_m1_core"

        trust_report = catalog.get_trust_report(result_one.artifact_id)
        assert trust_report is not None
        assert trust_report.status == "accepted"

        inputs = catalog.list_artifact_inputs(result_one.artifact_id)
        assert any(record.input_kind == "dataset_spec" for record in inputs)
        assert any(record.input_kind == "feature_spec" for record in inputs)
        assert any(record.input_kind == "label_pack" for record in inputs)
        assert any(record.input_kind == "merge_config" for record in inputs)
    finally:
        catalog.close()

    inspected_alias = inspect_artifact("dataset://xau_m1_core@1.0.0")
    assert inspected_alias.artifact.artifact_id == result_one.artifact_id
    assert inspected_alias.feature_families == ["htf_context", "quality", "session", "time"]
    assert inspected_alias.label_pack is not None
    assert inspected_alias.label_pack.key == "core_tb_volscaled@1.0.0"
    assert inspected_alias.split_row_counts == result_one.split_row_counts
    assert inspected_alias.time_range["start"]
    assert inspected_alias.time_range["end"]

    inspected_by_manifest = inspect_artifact(str(result_one.manifest_path))
    assert inspected_by_manifest.manifest_path == result_one.manifest_path
    assert inspected_by_manifest.trust_score_breakdown["total"] == result_one.trust_report.trust_score_total

    result_two = compile_dataset_spec(spec_two)
    diff = diff_artifacts(result_one.artifact_id, result_two.artifact_id)
    assert diff.diff["logical_version_changed"] is True
    assert "htf_context.standard_context@1.0.0" in diff.diff["feature_spec_refs_removed"]
    assert "H1_open" in diff.diff["schema_columns_removed"]
    assert diff.diff["label_pack_changed"] is False


def test_compile_phase4_nonhuman_dataset_from_stable_selectors_over_wider_fixture(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project_phase4"
    storage_root = project_root / "local_data" / "pipeline_data"
    storage_root.mkdir(parents=True, exist_ok=True)
    paths = _seed_project_data(
        storage_root,
        start_date=dt.date(2026, 4, 1),
        days=5,
    )
    _write_project_config(project_root, storage_root)

    nonhuman_spec = project_root / "config" / "datasets" / "xau_m1_nonhuman_v1.yaml"
    core_spec = project_root / "config" / "datasets" / "xau_m1_core_v1.yaml"
    nonhuman_selectors = [
        "time/*",
        "session/*",
        "quality/*",
        "htf_context/*",
        "disagreement/*",
        "event_shape/*",
        "entropy/*",
        "multiscale/*",
    ]
    _write_spec(
        nonhuman_spec,
        dataset_name="xau_m1_nonhuman",
        version="1.0.0",
        selectors=nonhuman_selectors,
        date_from="2026-04-01",
        date_to="2026-04-05",
    )
    _write_spec(
        core_spec,
        dataset_name="xau_m1_core",
        version="1.0.0",
        selectors=["time/*", "session/*", "quality/*", "htf_context/*"],
        date_from="2026-04-01",
        date_to="2026-04-05",
    )

    monkeypatch.chdir(project_root)

    core_result = compile_dataset_spec(core_spec)
    result = compile_dataset_spec(nonhuman_spec)

    assert result.manifest.status == "published"
    assert result.trust_report.accepted_for_publication is True
    assert result.split_row_counts["train"] > 0
    assert result.split_row_counts["val"] > 0
    assert result.split_row_counts["test"] > 0
    assert result.trust_report.decision_summary.startswith("accepted for publication")
    assert result.trust_report.check_status_counts["failed"] == 0

    inspected = inspect_artifact("dataset://xau_m1_nonhuman@1.0.0")
    assert inspected.artifact.artifact_id == result.artifact_id
    assert inspected.requested_feature_selectors == nonhuman_selectors
    assert inspected.feature_artifact_refs == []
    assert inspected.source_modes["state"] == "materialized"
    assert inspected.source_modes["features"] == "materialized"
    assert inspected.source_modes["label"] == "materialized"
    assert set(inspected.feature_families) == {
        "time",
        "session",
        "quality",
        "htf_context",
        "disagreement",
        "event_shape",
        "entropy",
        "multiscale",
    }
    assert inspected.trust_check_status_counts["failed"] == 0

    diff = diff_artifacts(core_result.artifact_id, result.artifact_id)
    assert "multiscale.consistency@1.0.0" in diff.diff["feature_spec_refs_added"]
    assert diff.diff["trust_decision_right"].startswith("accepted for publication")
    assert diff.diff["trust_check_status_counts_right"]["failed"] == 0

    catalog = CatalogDB(paths.catalog_db_path())
    try:
        artifact = catalog.get_artifact(result.artifact_id)
        assert artifact is not None
        assert artifact.status == "published"

        trust_report = catalog.get_trust_report(result.artifact_id)
        assert trust_report is not None
        assert trust_report.status == "accepted"
        assert trust_report.check_status_counts["failed"] == 0
    finally:
        catalog.close()


def test_compile_phase3_nonhuman_dataset_from_state_and_feature_artifacts(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project_phase3"
    storage_root = project_root / "local_data" / "pipeline_data"
    storage_root.mkdir(parents=True, exist_ok=True)
    paths = _seed_project_data(storage_root)
    _write_project_config(project_root, storage_root)
    store = ParquetStore(compression="snappy", row_group_size=1000)

    state_artifact_ref, feature_artifact_refs = _prepare_phase3_artifacts(project_root, paths, store)

    nonhuman_spec = project_root / "config" / "datasets" / "xau_m1_nonhuman_v1.yaml"
    core_spec = project_root / "config" / "datasets" / "xau_m1_core_v1.yaml"
    phase3_selectors = [
        "time/*",
        "session/*",
        "quality/*",
        "htf_context/*",
        "disagreement/*",
        "event_shape/*",
        "entropy/*",
    ]
    _write_spec(
        nonhuman_spec,
        dataset_name="xau_m1_nonhuman",
        version="1.0.0",
        selectors=phase3_selectors,
        date_from="2026-04-01",
        date_to="2026-04-02",
        state_version_ref=None,
        state_artifact_ref=state_artifact_ref,
        feature_artifact_refs=feature_artifact_refs,
    )
    _write_spec(
        core_spec,
        dataset_name="xau_m1_core",
        version="1.0.0",
        selectors=["time/*", "session/*", "quality/*", "htf_context/*"],
        date_from="2026-04-01",
        date_to="2026-04-02",
    )

    monkeypatch.chdir(project_root)

    core_result = compile_dataset_spec(core_spec)
    result = compile_dataset_spec(nonhuman_spec)

    assert result.manifest.status == "published"
    assert result.trust_report.accepted_for_publication is True
    assert result.published_aliases == [
        "dataset://xau_m1_nonhuman@1.0.0",
        "dataset://xau_m1_nonhuman:latest",
    ]
    assert result.split_row_counts["train"] > 0
    assert result.split_row_counts["val"] > 0
    assert result.split_row_counts["test"] > 0

    inspected = inspect_artifact("dataset://xau_m1_nonhuman@1.0.0")
    assert inspected.artifact.artifact_id == result.artifact_id
    assert inspected.source_modes["state"] == "artifact_ref"
    assert inspected.source_modes["features"] == "mixed"
    assert inspected.source_modes["label"] == "materialized"
    assert inspected.requested_feature_selectors == phase3_selectors
    assert len(inspected.feature_artifact_refs) == 3
    assert set(inspected.feature_families) == {
        "time",
        "session",
        "quality",
        "htf_context",
        "disagreement",
        "event_shape",
        "entropy",
    }

    diff = diff_artifacts(core_result.artifact_id, result.artifact_id)
    assert "disagreement.state_disagreement@1.0.0" in diff.diff["feature_spec_refs_added"]
    assert "event_shape.event_shape@1.0.0" in diff.diff["feature_spec_refs_added"]
    assert "entropy.micro_entropy@1.0.0" in diff.diff["feature_spec_refs_added"]
    assert "disagreement_bps_feature" in diff.diff["schema_columns_added"]
    assert "event_body_ratio" in diff.diff["schema_columns_added"]
    assert "entropy_return_30" in diff.diff["schema_columns_added"]

    catalog = CatalogDB(paths.catalog_db_path())
    try:
        artifact = catalog.get_artifact(result.artifact_id)
        assert artifact is not None
        assert artifact.status == "published"

        build = catalog.get_build_run(result.build_id)
        assert build is not None
        assert build.status == "published"

        trust_report = catalog.get_trust_report(result.artifact_id)
        assert trust_report is not None
        assert trust_report.status == "accepted"

        inputs = catalog.list_artifact_inputs(result.artifact_id)
        assert any(record.input_kind == "dataset_spec" for record in inputs)
        assert any(record.input_kind == "state_artifact" and record.input_ref == state_artifact_ref for record in inputs)
        assert any(record.input_kind == "feature_spec" and record.input_ref == "disagreement.state_disagreement@1.0.0" for record in inputs)
        assert any(record.input_kind == "feature_spec" and record.input_ref == "event_shape.event_shape@1.0.0" for record in inputs)
        assert any(record.input_kind == "feature_spec" and record.input_ref == "entropy.micro_entropy@1.0.0" for record in inputs)
        assert any(record.input_kind == "artifact" and record.input_ref in feature_artifact_refs for record in inputs)
    finally:
        catalog.close()


def test_compile_dataset_spec_rejects_publication_on_truth_failure(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project_reject"
    storage_root = project_root / "local_data" / "pipeline_data"
    storage_root.mkdir(parents=True, exist_ok=True)
    paths = _seed_project_data(storage_root)
    _write_project_config(project_root, storage_root)

    rejecting_spec = project_root / "config" / "datasets" / "xau_m1_core_reject.yaml"
    _write_spec(
        rejecting_spec,
        version="2.0.0",
        selectors=["time/*", "session/*", "quality/*", "htf_context/*"],
        embargo_rows=1500,
    )

    monkeypatch.chdir(project_root)

    result = compile_dataset_spec(rejecting_spec)
    assert result.manifest.status == "rejected"
    assert result.trust_report.accepted_for_publication is False
    assert any(status == "rejected" for status in result.status_history)
    assert "dataset_coverage_failure" in result.trust_report.hard_failures

    catalog = CatalogDB(paths.catalog_db_path())
    try:
        artifact = catalog.get_artifact(result.artifact_id)
        assert artifact is not None
        assert artifact.status == "rejected"

        aliases = catalog.list_aliases(result.artifact_id)
        assert aliases == []

        build = catalog.get_build_run(result.build_id)
        assert build is not None
        assert build.status == "rejected"
    finally:
        catalog.close()


def test_compile_phase3_rejects_when_artifact_backed_family_is_incomplete(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project_phase3_reject"
    storage_root = project_root / "local_data" / "pipeline_data"
    storage_root.mkdir(parents=True, exist_ok=True)
    paths = _seed_project_data(storage_root)
    _write_project_config(project_root, storage_root)
    store = ParquetStore(compression="snappy", row_group_size=1000)

    catalog = CatalogDB(paths.catalog_db_path())
    try:
        base_df = pl.concat(
            [
                store.read_dir(paths.built_bars_dir("XAUUSD", "M1", dt.date(2026, 4, 1))),
                store.read_dir(paths.built_bars_dir("XAUUSD", "M1", dt.date(2026, 4, 2))),
            ],
            how="diagonal_relaxed",
        ).sort("time_utc")
        state_artifact_id, state_manifest, _ = _register_state_artifact(
            paths=paths,
            store=store,
            catalog=catalog,
            base_df=base_df,
            state_version_ref="state.default@1.0.0",
            date_from=dt.date(2026, 4, 1),
            date_to=dt.date(2026, 4, 2),
            build_id="build.phase3.reject",
            dataset_spec_ref="fixture.nonhuman.reject@1.0.0",
        )

        feature_artifact_refs: list[str] = []
        for spec in _phase3_feature_specs():
            frame = _build_phase3_feature_frame(base_df, spec)
            if spec.family == "entropy":
                frame = frame.drop("entropy_sign_30")
            feature_artifact_refs.append(
                _register_feature_artifact(
                    paths=paths,
                    store=store,
                    catalog=catalog,
                    spec=spec,
                    frame=frame,
                    state_artifact_id=state_artifact_id,
                    input_partition_refs=state_manifest.input_partition_refs,
                )
            )
    finally:
        catalog.close()

    rejecting_spec = project_root / "config" / "datasets" / "xau_m1_nonhuman_reject.yaml"
    _write_spec(
        rejecting_spec,
        dataset_name="xau_m1_nonhuman",
        version="2.0.0",
        selectors=[
            "time/*",
            "session/*",
            "quality/*",
            "htf_context/*",
            "disagreement/*",
            "event_shape/*",
            "entropy/*",
        ],
        date_from="2026-04-01",
        date_to="2026-04-02",
        state_version_ref=None,
        state_artifact_ref=state_artifact_id,
        feature_artifact_refs=feature_artifact_refs,
    )

    monkeypatch.chdir(project_root)

    result = compile_dataset_spec(rejecting_spec)
    assert result.manifest.status == "rejected"
    assert result.trust_report.accepted_for_publication is False
    assert "missing_required_feature_columns" in result.trust_report.hard_failures
    assert any(status == "rejected" for status in result.status_history)


def test_compile_dataset_spec_reports_unresolved_feature_artifact_refs_clearly(tmp_path, monkeypatch) -> None:
    project_root = tmp_path / "project_missing_feature_ref"
    storage_root = project_root / "local_data" / "pipeline_data"
    storage_root.mkdir(parents=True, exist_ok=True)
    _seed_project_data(storage_root)
    _write_project_config(project_root, storage_root)

    spec_path = project_root / "config" / "datasets" / "xau_m1_nonhuman_missing_ref.yaml"
    _write_spec(
        spec_path,
        dataset_name="xau_m1_nonhuman",
        version="1.0.0",
        selectors=[
            "time/*",
            "session/*",
            "quality/*",
            "htf_context/*",
            "disagreement/*",
        ],
        date_from="2026-04-01",
        date_to="2026-04-02",
        feature_artifact_refs=["feature.disagreement.state_disagreement.latest"],
    )

    monkeypatch.chdir(project_root)

    with pytest.raises(KeyError, match="remove the stale feature_artifact_refs entry"):
        compile_dataset_spec(spec_path)
