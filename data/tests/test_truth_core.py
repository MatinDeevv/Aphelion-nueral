"""Focused tests for truth-layer publish gate behavior."""

from __future__ import annotations

import datetime as dt

import polars as pl

from mt5pipe.compiler.models import DatasetSpec, LineageManifest
from mt5pipe.features.public import FeatureSpec
from mt5pipe.features.registry.defaults import get_default_feature_specs
from mt5pipe.labels.public import LabelPack
from mt5pipe.labels.registry.defaults import get_default_label_packs
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths
from mt5pipe.truth.service import TruthService


UTC = dt.timezone.utc


def _phase4_truth_feature_specs() -> list[FeatureSpec]:
    return [
        FeatureSpec(
            feature_name="minute_index",
            family="time",
            version="1.0.0",
            description="Minimal time feature for truth tests",
            input_contract="BuiltBar",
            input_clock="M1",
            output_clock="M1",
            builder_ref="fixture://time",
            output_columns=["minute"],
            dependencies=["time_utc"],
            missingness_policy="fail",
            qa_policy_ref="qa.feature.default@1.0.0",
            status="stable",
        ),
        FeatureSpec(
            feature_name="spread_quality",
            family="quality",
            version="1.0.0",
            description="Minimal quality feature for truth tests",
            input_contract="BuiltBar",
            input_clock="M1",
            output_clock="M1",
            builder_ref="fixture://quality",
            output_columns=["relative_spread", "conflict_ratio", "broker_diversity"],
            dependencies=["spread_mean", "close", "conflict_count", "tick_count", "source_count"],
            missingness_policy="fail",
            qa_policy_ref="qa.feature.default@1.0.0",
            status="stable",
        ),
        FeatureSpec(
            feature_name="standard_context",
            family="htf_context",
            version="1.0.0",
            description="Minimal HTF feature for truth tests",
            input_contract="BuiltBar",
            input_clock="M1",
            output_clock="M1",
            builder_ref="fixture://htf_context",
            output_columns=["H1_tick_count"],
            dependencies=["time_utc"],
            missingness_policy="allow",
            qa_policy_ref="qa.feature.default@1.0.0",
            status="stable",
        ),
        FeatureSpec(
            feature_name="flow_shape",
            family="event_shape",
            version="1.0.0",
            description="Minimal event-shape feature for truth tests",
            input_contract="BuiltBar",
            input_clock="M1",
            output_clock="M1",
            builder_ref="fixture://event_shape",
            output_columns=[
                "tick_rate_hz",
                "interarrival_mean_ms",
                "burstiness_20",
                "silence_ratio_20",
                "direction_switch_rate_20",
                "signed_run_length",
                "path_efficiency_20",
                "tortuosity_20",
            ],
            dependencies=["tick_count", "mid_return"],
            lookback_rows=20,
            warmup_rows=20,
            missingness_policy="allow",
            qa_policy_ref="qa.feature.default@1.0.0",
            status="stable",
        ),
    ]


def _phase4_truth_label_pack() -> LabelPack:
    return LabelPack(
        label_pack_name="phase4_truth",
        version="1.0.0",
        description="Minimal label pack for truth tests",
        base_clock="M1",
        horizons_minutes=[5],
        generator_refs=["fixture://label"],
        purge_rows=5,
        output_columns=["label_5m"],
        status="stable",
    )


def _phase4_truth_dataset(
    *,
    rows: int = 40,
    htf_nulls: int = 24,
    signed_run_length_nulls: int = 4,
    minute_constant: bool = False,
) -> pl.DataFrame:
    times = [dt.datetime(2026, 4, 1, 0, i, tzinfo=UTC) for i in range(rows)]
    minute_values = [0] * rows if minute_constant else list(range(rows))
    signed_run_values = [
        None if index < signed_run_length_nulls else int((index % 5) + 1)
        for index in range(rows)
    ]
    return pl.DataFrame(
        {
            "symbol": ["XAUUSD"] * rows,
            "timeframe": ["M1"] * rows,
            "time_utc": times,
            "open": [3000.0 + index * 0.1 for index in range(rows)],
            "high": [3000.2 + index * 0.1 for index in range(rows)],
            "low": [2999.8 + index * 0.1 for index in range(rows)],
            "close": [3000.1 + index * 0.1 for index in range(rows)],
            "tick_count": [10 + (index % 4) for index in range(rows)],
            "spread_mean": [0.10 + (index % 3) * 0.01 for index in range(rows)],
            "mid_return": [0.0010 + ((index % 5) - 2) * 0.0001 for index in range(rows)],
            "realized_vol": [0.0010 + (index % 4) * 0.0001 for index in range(rows)],
            "source_count": [2] * rows,
            "conflict_count": [0] * rows,
            "dual_source_ticks": [8] * rows,
            "secondary_present_ticks": [8] * rows,
            "dual_source_ratio": [0.80] * rows,
            "minute": minute_values,
            "relative_spread": [0.00003 + (index % 3) * 0.00001 for index in range(rows)],
            "conflict_ratio": [0.0] * rows,
            "broker_diversity": [2] * rows,
            "H1_tick_count": [None if index < htf_nulls else 10 for index in range(rows)],
            "tick_rate_hz": [1.0 + (index % 5) * 0.1 for index in range(rows)],
            "interarrival_mean_ms": [1000.0 - (index % 4) * 25.0 for index in range(rows)],
            "burstiness_20": [0.2 + (index % 4) * 0.05 for index in range(rows)],
            "silence_ratio_20": [0.1 + (index % 3) * 0.02 for index in range(rows)],
            "direction_switch_rate_20": [0.3 + (index % 4) * 0.03 for index in range(rows)],
            "signed_run_length": signed_run_values,
            "path_efficiency_20": [0.7 - (index % 3) * 0.05 for index in range(rows)],
            "tortuosity_20": [1.1 + (index % 3) * 0.05 for index in range(rows)],
            "label_5m": [1 if index % 2 == 0 else 0 for index in range(rows)],
        }
    )


def _phase4_truth_split_frames(dataset_df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    rows = len(dataset_df)
    train_rows = max(1, int(rows * 0.70))
    val_rows = max(1, int(rows * 0.15))
    return {
        "train": dataset_df.slice(0, train_rows),
        "val": dataset_df.slice(train_rows, val_rows),
        "test": dataset_df.slice(train_rows + val_rows, rows - train_rows - val_rows),
    }


def _phase4_truth_spec() -> DatasetSpec:
    return DatasetSpec(
        dataset_name="xau_nonhuman",
        version="1.0.0",
        symbols=["XAUUSD"],
        date_from=dt.date(2026, 4, 1),
        date_to=dt.date(2026, 4, 1),
        base_clock="M1",
        state_version_ref="state.default@1.0.0",
        feature_selectors=["time/*", "quality/*", "htf_context/*", "event_shape/*"],
        label_pack_ref="phase4_truth@1.0.0",
        split_policy="temporal_holdout",
        embargo_rows=5,
        truth_policy_ref="truth.default@1.0.0",
    )


def _phase4_truth_manifest(
    spec: DatasetSpec,
    feature_specs: list[FeatureSpec],
    label_pack: LabelPack,
    *,
    artifact_id: str,
) -> LineageManifest:
    return LineageManifest(
        manifest_id=f"manifest.{artifact_id}",
        artifact_id=artifact_id,
        artifact_kind="dataset",
        logical_name=spec.dataset_name,
        logical_version=spec.version,
        artifact_uri=f"artifact://{artifact_id}",
        content_hash="good-hash",
        build_id="build.test",
        created_at=dt.datetime.now(UTC),
        status="truth_pending",
        dataset_spec_ref=spec.key,
        state_artifact_refs=["state.XAUUSD.M1.123"],
        feature_spec_refs=[feature.key for feature in feature_specs],
        label_pack_ref=label_pack.key,
        truth_report_ref=None,
        code_version="workspace-local-no-git",
        input_partition_refs=["data/bars/symbol=XAUUSD/timeframe=M1/date=2026-04-01"],
        parent_artifact_refs=["state.XAUUSD.M1.123"],
    )


def test_truth_rejects_manifest_hash_mismatch(tmp_path) -> None:
    dataset_df = pl.DataFrame(
        {
            "symbol": ["XAUUSD"] * 4,
            "timeframe": ["M1"] * 4,
            "time_utc": [dt.datetime(2026, 4, 1, 0, i, tzinfo=UTC) for i in range(4)],
            "open": [3000.0, 3000.1, 3000.2, 3000.3],
            "high": [3000.2, 3000.3, 3000.4, 3000.5],
            "low": [2999.9, 3000.0, 3000.1, 3000.2],
            "close": [3000.1, 3000.2, 3000.3, 3000.4],
            "tick_count": [10, 10, 10, 10],
            "spread_mean": [0.1, 0.1, 0.1, 0.1],
            "mid_return": [0.001, 0.001, 0.001, 0.001],
            "realized_vol": [0.001, 0.001, 0.001, 0.001],
            "source_count": [2, 2, 2, 2],
            "conflict_count": [0, 0, 0, 0],
            "hour": [0, 0, 0, 0],
            "minute": [0, 1, 2, 3],
            "weekday": [3, 3, 3, 3],
            "time_sin": [0.0, 0.1, 0.2, 0.3],
            "time_cos": [1.0, 0.9, 0.8, 0.7],
            "weekday_sin": [0.1, 0.1, 0.1, 0.1],
            "weekday_cos": [0.9, 0.9, 0.9, 0.9],
            "session_asia": [1, 1, 1, 1],
            "session_london": [0, 0, 0, 0],
            "session_ny": [0, 0, 0, 0],
            "session_overlap": [0, 0, 0, 0],
            "relative_spread": [0.00003, 0.00003, 0.00003, 0.00003],
            "conflict_ratio": [0.0, 0.0, 0.0, 0.0],
            "broker_diversity": [2, 2, 2, 2],
            "future_return_5m": [0.1, 0.1, 0.1, 0.1],
            "direction_5m": [1, 1, 1, 1],
            "triple_barrier_5m": [1, 1, 1, 1],
            "mae_5m": [0.01, 0.01, 0.01, 0.01],
            "mfe_5m": [0.02, 0.02, 0.02, 0.02],
            "future_return_15m": [0.1, 0.1, 0.1, 0.1],
            "direction_15m": [1, 1, 1, 1],
            "triple_barrier_15m": [1, 1, 1, 1],
            "mae_15m": [0.01, 0.01, 0.01, 0.01],
            "mfe_15m": [0.02, 0.02, 0.02, 0.02],
            "future_return_60m": [0.1, 0.1, 0.1, 0.1],
            "direction_60m": [1, 1, 1, 1],
            "triple_barrier_60m": [1, 1, 1, 1],
            "mae_60m": [0.01, 0.01, 0.01, 0.01],
            "mfe_60m": [0.02, 0.02, 0.02, 0.02],
            "future_return_240m": [0.1, 0.1, 0.1, 0.1],
            "direction_240m": [1, 1, 1, 1],
            "triple_barrier_240m": [1, 1, 1, 1],
            "mae_240m": [0.01, 0.01, 0.01, 0.01],
            "mfe_240m": [0.02, 0.02, 0.02, 0.02],
        }
    )
    split_frames = {"train": dataset_df.slice(0, 2), "val": dataset_df.slice(2, 1), "test": dataset_df.slice(3, 1)}
    spec = DatasetSpec(
        dataset_name="xau_core",
        version="1.0.0",
        symbols=["XAUUSD"],
        date_from=dt.date(2026, 4, 1),
        date_to=dt.date(2026, 4, 1),
        base_clock="M1",
        state_version_ref="state.default@1.0.0",
        feature_selectors=["time/*", "session/*", "quality/*"],
        label_pack_ref="core_tb_volscaled@1.0.0",
        split_policy="temporal_holdout",
        embargo_rows=1,
        truth_policy_ref="truth.default@1.0.0",
    )
    manifest = LineageManifest(
        manifest_id="manifest.dataset.xau_core.bad",
        artifact_id="dataset.xau_core.bad",
        artifact_kind="dataset",
        logical_name="xau_core",
        logical_version="1.0.0",
        artifact_uri="artifact://dataset.xau_core.bad",
        content_hash="bad-hash",
        build_id="build.test",
        created_at=dt.datetime.now(UTC),
        status="truth_pending",
        dataset_spec_ref=spec.key,
        state_artifact_refs=["state.XAUUSD.M1.123"],
        feature_spec_refs=[feature.key for feature in get_default_feature_specs()[:3]],
        label_pack_ref=get_default_label_packs()[0].key,
        truth_report_ref=None,
        code_version="workspace-local-no-git",
        input_partition_refs=["data/bars/symbol=XAUUSD/timeframe=M1/date=2026-04-01"],
        parent_artifact_refs=["state.XAUUSD.M1.123"],
    )

    paths = StoragePaths(tmp_path / "pipeline_data")
    store = ParquetStore(compression="snappy", row_group_size=1000)
    report = TruthService().evaluate_dataset(
        artifact_id=manifest.artifact_id,
        dataset_df=dataset_df,
        split_frames=split_frames,
        spec=spec,
        feature_specs=get_default_feature_specs()[:3],
        label_pack=get_default_label_packs()[0],
        manifest=manifest,
        paths=paths,
        store=store,
        expected_content_hash="good-hash",
        manifest_path=tmp_path / "manifest.json",
    )

    assert report.status == "rejected"
    assert report.accepted_for_publication is False
    assert "manifest_hash_mismatch" in report.hard_failures


def test_truth_rejects_required_dual_broker_dataset_without_synchronized_coverage(tmp_path) -> None:
    dataset_df = _phase4_truth_dataset()
    split_frames = _phase4_truth_split_frames(dataset_df)
    feature_specs = _phase4_truth_feature_specs()
    label_pack = _phase4_truth_label_pack()
    spec = _phase4_truth_spec().model_copy(
        update={
            "required_raw_brokers": ["broker_a", "broker_b"],
            "require_synchronized_raw_coverage": True,
            "require_dual_source_overlap": True,
            "min_dual_source_ratio": 0.05,
        }
    )
    manifest = _phase4_truth_manifest(spec, feature_specs, label_pack, artifact_id="dataset.xau_nonhuman.bad_source")
    paths = StoragePaths(tmp_path / "pipeline_data")
    store = ParquetStore(compression="snappy", row_group_size=1000)
    date = dt.date(2026, 4, 1)

    raw_b = pl.DataFrame(
        {
            "broker_id": ["broker_b"],
            "symbol": ["XAUUSD"],
            "time_utc": [dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC)],
            "time_msc": [int(dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC).timestamp() * 1000)],
            "bid": [3000.0],
            "ask": [3000.2],
            "last": [0.0],
            "volume": [1.0],
            "volume_real": [0.0],
            "flags": [6],
            "ingest_ts": [dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC)],
        }
    )
    store.write(raw_b, paths.raw_ticks_file("broker_b", "XAUUSD", date))
    merge_qa = pl.DataFrame(
        [
            {
                "time_utc": dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC),
                "date": "2026-04-01",
                "symbol": "XAUUSD",
                "dual_source_ratio": 0.0,
                "conflicts": 0,
                "bucket_both": 0,
                "canonical_dual_rows": 0,
            }
        ]
    )
    store.write(merge_qa, paths.merge_qa_file("XAUUSD", date))

    state_df = pl.DataFrame(
        {
            "quality_score": [95.0, 95.0],
            "conflict_flag": [False, False],
            "trust_flags": [[], []],
        }
    )

    report = TruthService().evaluate_dataset(
        artifact_id=manifest.artifact_id,
        dataset_df=dataset_df,
        split_frames=split_frames,
        spec=spec,
        feature_specs=feature_specs,
        label_pack=label_pack,
        manifest=manifest,
        paths=paths,
        store=store,
        state_df=state_df,
    )

    assert report.status == "rejected"
    assert report.accepted_for_publication is False
    assert "source_quality_below_threshold" in report.hard_failures
    assert any("synchronized raw coverage is incomplete" in reason for reason in report.rejection_reasons)
    assert any("dual-source overlap is required" in reason for reason in report.rejection_reasons)


def test_synchronized_coverage_excludes_weekends(tmp_path) -> None:
    dataset_df = _phase4_truth_dataset()
    split_frames = _phase4_truth_split_frames(dataset_df)
    feature_specs = _phase4_truth_feature_specs()
    label_pack = _phase4_truth_label_pack()
    spec = _phase4_truth_spec().model_copy(
        update={
            "date_from": dt.date(2026, 4, 6),
            "date_to": dt.date(2026, 4, 12),
            "required_raw_brokers": ["broker_a", "broker_b"],
            "require_synchronized_raw_coverage": True,
            "require_dual_source_overlap": False,
        }
    )
    state_df = pl.DataFrame(
        {
            "quality_score": [95.0, 95.0],
            "conflict_flag": [False, False],
            "trust_flags": [[], []],
        }
    )

    def build_report(root_name: str, *, skip_date: dt.date | None = None):
        paths = StoragePaths(tmp_path / root_name / "pipeline_data")
        store = ParquetStore(compression="snappy", row_group_size=1000)
        manifest = _phase4_truth_manifest(
            spec,
            feature_specs,
            label_pack,
            artifact_id=f"dataset.xau_nonhuman.{root_name}",
        )

        current = spec.date_from
        while current <= spec.date_to:
            if current.weekday() < 5 and current != skip_date:
                for broker_id in spec.required_raw_brokers:
                    raw_ticks = pl.DataFrame(
                        {
                            "broker_id": [broker_id],
                            "symbol": ["XAUUSD"],
                            "time_utc": [dt.datetime.combine(current, dt.time(0, 0), tzinfo=UTC)],
                            "time_msc": [int(dt.datetime.combine(current, dt.time(0, 0), tzinfo=UTC).timestamp() * 1000)],
                            "bid": [3000.0],
                            "ask": [3000.2],
                            "last": [0.0],
                            "volume": [1.0],
                            "volume_real": [0.0],
                            "flags": [6],
                            "ingest_ts": [dt.datetime.combine(current, dt.time(0, 0), tzinfo=UTC)],
                        }
                    )
                    store.write(raw_ticks, paths.raw_ticks_file(broker_id, "XAUUSD", current))
            current += dt.timedelta(days=1)

        return TruthService().evaluate_dataset(
            artifact_id=manifest.artifact_id,
            dataset_df=dataset_df,
            split_frames=split_frames,
            spec=spec,
            feature_specs=feature_specs,
            label_pack=label_pack,
            manifest=manifest,
            paths=paths,
            store=store,
            state_df=state_df,
        )

    weekend_ok = build_report("weekend_ok")
    assert weekend_ok.status == "accepted"
    assert weekend_ok.accepted_for_publication is True
    assert not any("synchronized raw coverage is incomplete" in reason for reason in weekend_ok.rejection_reasons)

    missing_wednesday = build_report("missing_wednesday", skip_date=dt.date(2026, 4, 8))
    assert missing_wednesday.status == "rejected"
    assert missing_wednesday.accepted_for_publication is False
    assert any(
        "synchronized raw coverage is incomplete" in reason and "2026-04-08" in reason
        for reason in missing_wednesday.rejection_reasons
    )


def test_truth_rejects_feature_family_missingness_threshold_breach(tmp_path) -> None:
    dataset_df = pl.DataFrame(
        {
            "symbol": ["XAUUSD"] * 4,
            "timeframe": ["M1"] * 4,
            "time_utc": [dt.datetime(2026, 4, 1, 0, i, tzinfo=UTC) for i in range(4)],
            "open": [3000.0, 3000.1, 3000.2, 3000.3],
            "high": [3000.2, 3000.3, 3000.4, 3000.5],
            "low": [2999.9, 3000.0, 3000.1, 3000.2],
            "close": [3000.1, 3000.2, 3000.3, 3000.4],
            "tick_count": [10, 10, 10, 10],
            "spread_mean": [0.1, 0.1, 0.1, 0.1],
            "mid_return": [0.001, 0.001, 0.001, 0.001],
            "realized_vol": [0.001, 0.001, 0.001, 0.001],
            "source_count": [2, 2, 2, 2],
            "conflict_count": [0, 0, 0, 0],
            "hour": [0, 0, 0, 0],
            "minute": [0, 1, 2, 3],
            "weekday": [3, 3, 3, 3],
            "time_sin": [0.0, 0.1, 0.2, 0.3],
            "time_cos": [1.0, 0.9, 0.8, 0.7],
            "weekday_sin": [0.1, 0.1, 0.1, 0.1],
            "weekday_cos": [0.9, 0.9, 0.9, 0.9],
            "session_asia": [1, 1, 1, 1],
            "session_london": [0, 0, 0, 0],
            "session_ny": [0, 0, 0, 0],
            "session_overlap": [0, 0, 0, 0],
            "relative_spread": [0.00003, 0.00003, 0.00003, 0.00003],
            "conflict_ratio": [0.0, 0.0, 0.0, 0.0],
            "broker_diversity": [2, 2, 2, 2],
            "entropy_signal": [None, None, None, None],
            "future_return_5m": [0.1, 0.1, 0.1, 0.1],
            "direction_5m": [1, 1, 1, 1],
            "triple_barrier_5m": [1, 1, 1, 1],
            "mae_5m": [0.01, 0.01, 0.01, 0.01],
            "mfe_5m": [0.02, 0.02, 0.02, 0.02],
            "future_return_15m": [0.1, 0.1, 0.1, 0.1],
            "direction_15m": [1, 1, 1, 1],
            "triple_barrier_15m": [1, 1, 1, 1],
            "mae_15m": [0.01, 0.01, 0.01, 0.01],
            "mfe_15m": [0.02, 0.02, 0.02, 0.02],
            "future_return_60m": [0.1, 0.1, 0.1, 0.1],
            "direction_60m": [1, 1, 1, 1],
            "triple_barrier_60m": [1, 1, 1, 1],
            "mae_60m": [0.01, 0.01, 0.01, 0.01],
            "mfe_60m": [0.02, 0.02, 0.02, 0.02],
            "future_return_240m": [0.1, 0.1, 0.1, 0.1],
            "direction_240m": [1, 1, 1, 1],
            "triple_barrier_240m": [1, 1, 1, 1],
            "mae_240m": [0.01, 0.01, 0.01, 0.01],
            "mfe_240m": [0.02, 0.02, 0.02, 0.02],
        }
    )
    split_frames = {"train": dataset_df.slice(0, 2), "val": dataset_df.slice(2, 1), "test": dataset_df.slice(3, 1)}
    spec = DatasetSpec(
        dataset_name="xau_nonhuman",
        version="1.0.0",
        symbols=["XAUUSD"],
        date_from=dt.date(2026, 4, 1),
        date_to=dt.date(2026, 4, 1),
        base_clock="M1",
        state_version_ref="state.default@1.0.0",
        feature_selectors=["time/*", "session/*", "quality/*", "entropy/*"],
        label_pack_ref="core_tb_volscaled@1.0.0",
        split_policy="temporal_holdout",
        embargo_rows=1,
        truth_policy_ref="truth.default@1.0.0",
    )
    entropy_spec = get_default_feature_specs()[0].model_copy(
        update={
            "feature_name": "entropy_signal",
            "family": "entropy",
            "output_columns": ["entropy_signal"],
            "missingness_policy": "fail",
        }
    )
    manifest = LineageManifest(
        manifest_id="manifest.dataset.xau_nonhuman.bad",
        artifact_id="dataset.xau_nonhuman.bad",
        artifact_kind="dataset",
        logical_name="xau_nonhuman",
        logical_version="1.0.0",
        artifact_uri="artifact://dataset.xau_nonhuman.bad",
        content_hash="good-hash",
        build_id="build.test",
        created_at=dt.datetime.now(UTC),
        status="truth_pending",
        dataset_spec_ref=spec.key,
        state_artifact_refs=["state.XAUUSD.M1.123"],
        feature_spec_refs=["time.cyclical_time@1.0.0", "session.session_flags@1.0.0", "quality.spread_quality@1.0.0", entropy_spec.key],
        label_pack_ref=get_default_label_packs()[0].key,
        truth_report_ref=None,
        code_version="workspace-local-no-git",
        input_partition_refs=["data/bars/symbol=XAUUSD/timeframe=M1/date=2026-04-01"],
        parent_artifact_refs=["state.XAUUSD.M1.123"],
    )

    paths = StoragePaths(tmp_path / "pipeline_data")
    store = ParquetStore(compression="snappy", row_group_size=1000)
    report = TruthService().evaluate_dataset(
        artifact_id=manifest.artifact_id,
        dataset_df=dataset_df,
        split_frames=split_frames,
        spec=spec,
        feature_specs=[*get_default_feature_specs()[:3], entropy_spec],
        label_pack=get_default_label_packs()[0],
        manifest=manifest,
        paths=paths,
        store=store,
        expected_content_hash="good-hash",
        manifest_path=tmp_path / "manifest.json",
    )

    assert report.status == "rejected"
    assert "feature_family_missingness_threshold_exceeded" in report.hard_failures


def test_truth_reports_informative_warning_reasons_for_source_quality(tmp_path) -> None:
    dataset_df = pl.DataFrame(
        {
            "symbol": ["XAUUSD"] * 4,
            "timeframe": ["M1"] * 4,
            "time_utc": [dt.datetime(2026, 4, 1, 0, i, tzinfo=UTC) for i in range(4)],
            "open": [3000.0, 3000.1, 3000.2, 3000.3],
            "high": [3000.2, 3000.4, 3000.5, 3000.6],
            "low": [2999.9, 3000.0, 3000.1, 3000.2],
            "close": [3000.1, 3000.2, 3000.35, 3000.45],
            "tick_count": [10, 11, 12, 13],
            "spread_mean": [0.10, 0.11, 0.10, 0.12],
            "mid_return": [0.0010, 0.0011, -0.0005, 0.0008],
            "realized_vol": [0.0010, 0.0011, 0.0012, 0.0011],
            "source_count": [2, 1, 2, 1],
            "conflict_count": [0, 1, 0, 1],
            "dual_source_ticks": [8, 7, 9, 8],
            "secondary_present_ticks": [8, 7, 9, 8],
            "dual_source_ratio": [0.80, 0.70, 0.90, 0.75],
            "hour": [0, 0, 0, 0],
            "minute": [0, 1, 2, 3],
            "weekday": [3, 3, 3, 3],
            "time_sin": [0.0, 0.1, 0.2, 0.3],
            "time_cos": [1.0, 0.9, 0.8, 0.7],
            "weekday_sin": [0.1, 0.1, 0.1, 0.1],
            "weekday_cos": [0.9, 0.9, 0.9, 0.9],
            "session_asia": [1, 1, 1, 1],
            "session_london": [0, 0, 0, 0],
            "session_ny": [0, 0, 0, 0],
            "session_overlap": [0, 0, 0, 0],
            "relative_spread": [0.00003, 0.00004, 0.00003, 0.00004],
            "conflict_ratio": [0.00, 0.05, 0.00, 0.05],
            "broker_diversity": [2, 1, 2, 1],
            "future_return_5m": [0.1, -0.1, 0.2, -0.2],
            "direction_5m": [1, 0, 1, 0],
            "triple_barrier_5m": [1, -1, 1, -1],
            "mae_5m": [0.01, 0.02, 0.01, 0.03],
            "mfe_5m": [0.02, 0.01, 0.03, 0.02],
            "future_return_15m": [0.1, -0.1, 0.2, -0.2],
            "direction_15m": [1, 0, 1, 0],
            "triple_barrier_15m": [1, -1, 1, -1],
            "mae_15m": [0.01, 0.02, 0.01, 0.03],
            "mfe_15m": [0.02, 0.01, 0.03, 0.02],
            "future_return_60m": [0.1, -0.1, 0.2, -0.2],
            "direction_60m": [1, 0, 1, 0],
            "triple_barrier_60m": [1, -1, 1, -1],
            "mae_60m": [0.01, 0.02, 0.01, 0.03],
            "mfe_60m": [0.02, 0.01, 0.03, 0.02],
            "future_return_240m": [0.1, -0.1, 0.2, -0.2],
            "direction_240m": [1, 0, 1, 0],
            "triple_barrier_240m": [1, -1, 1, -1],
            "mae_240m": [0.01, 0.02, 0.01, 0.03],
            "mfe_240m": [0.02, 0.01, 0.03, 0.02],
        }
    )
    split_frames = {"train": dataset_df.slice(0, 2), "val": dataset_df.slice(2, 1), "test": dataset_df.slice(3, 1)}
    spec = DatasetSpec(
        dataset_name="xau_core",
        version="1.0.0",
        symbols=["XAUUSD"],
        date_from=dt.date(2026, 4, 1),
        date_to=dt.date(2026, 4, 1),
        base_clock="M1",
        state_version_ref="state.default@1.0.0",
        feature_selectors=["time/*", "session/*", "quality/*"],
        label_pack_ref="core_tb_volscaled@1.0.0",
        split_policy="temporal_holdout",
        embargo_rows=1,
        truth_policy_ref="truth.default@1.0.0",
    )
    manifest = LineageManifest(
        manifest_id="manifest.dataset.xau_core.warning",
        artifact_id="dataset.xau_core.warning",
        artifact_kind="dataset",
        logical_name="xau_core",
        logical_version="1.0.0",
        artifact_uri="artifact://dataset.xau_core.warning",
        content_hash="good-hash",
        build_id="build.test",
        created_at=dt.datetime.now(UTC),
        status="truth_pending",
        dataset_spec_ref=spec.key,
        state_artifact_refs=["state.XAUUSD.M1.123"],
        feature_spec_refs=[feature.key for feature in get_default_feature_specs()[:3]],
        label_pack_ref=get_default_label_packs()[0].key,
        truth_report_ref=None,
        code_version="workspace-local-no-git",
        input_partition_refs=["data/bars/symbol=XAUUSD/timeframe=M1/date=2026-04-01"],
        parent_artifact_refs=["state.XAUUSD.M1.123"],
    )

    state_df = pl.DataFrame(
        {
            "quality_score": [70.0, 70.0, 70.0, 70.0],
            "conflict_flag": [False, False, False, False],
            "trust_flags": [[], [], [], []],
        }
    )

    paths = StoragePaths(tmp_path / "pipeline_data")
    store = ParquetStore(compression="snappy", row_group_size=1000)
    diagnostics = pl.DataFrame(
        [
            {
                "time_utc": dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC),
                "symbol": "XAUUSD",
                "dual_source_ratio": 0.0,
                "conflicts": 0,
            }
        ]
    )
    store.write(diagnostics, paths.merge_diagnostics_file("XAUUSD", dt.date(2026, 4, 1)))

    report = TruthService().evaluate_dataset(
        artifact_id=manifest.artifact_id,
        dataset_df=dataset_df,
        split_frames=split_frames,
        spec=spec,
        feature_specs=get_default_feature_specs()[:3],
        label_pack=get_default_label_packs()[0],
        manifest=manifest,
        paths=paths,
        store=store,
        state_df=state_df,
        expected_content_hash="good-hash",
    )

    assert report.status == "accepted"
    assert report.accepted_for_publication is True
    assert report.check_status_counts["warning"] >= 1
    assert any(reason.startswith("source_quality: score 70.00 is below preferred 75.00") for reason in report.warning_reasons)
    assert any("merge_diagnostics_days=1" in reason for reason in report.warning_reasons)
    assert any("diagnostic_dual_source_ratio_mean=0.0000" in reason for reason in report.warning_reasons)
    assert "source quality is acceptable but below preferred research comfort" not in report.warning_reasons


def test_truth_classifies_expected_sparse_nulls_and_slice_trivial_constants_as_accepted_caveats(tmp_path) -> None:
    dataset_df = _phase4_truth_dataset(rows=40, htf_nulls=24, signed_run_length_nulls=4, minute_constant=False)
    split_frames = _phase4_truth_split_frames(dataset_df)
    feature_specs = _phase4_truth_feature_specs()
    label_pack = _phase4_truth_label_pack()
    spec = _phase4_truth_spec()
    manifest = _phase4_truth_manifest(
        spec,
        feature_specs,
        label_pack,
        artifact_id="dataset.xau_nonhuman.caveats",
    )
    state_df = pl.DataFrame(
        {
            "quality_score": [90.0] * len(dataset_df),
            "conflict_flag": [False] * len(dataset_df),
            "trust_flags": [[] for _ in range(len(dataset_df))],
        }
    )

    report = TruthService().evaluate_dataset(
        artifact_id=manifest.artifact_id,
        dataset_df=dataset_df,
        split_frames=split_frames,
        spec=spec,
        feature_specs=feature_specs,
        label_pack=label_pack,
        manifest=manifest,
        paths=StoragePaths(tmp_path / "pipeline_data"),
        store=ParquetStore(compression="snappy", row_group_size=1000),
        state_df=state_df,
        expected_content_hash="good-hash",
    )

    caveats = report.metrics["quality_caveat_summary"]

    assert report.status == "accepted"
    assert report.accepted_for_publication is True
    assert report.warning_reasons == []
    assert report.decision_summary == f"accepted for publication; total={report.trust_score_total:.2f}"
    assert caveats["green_blockers"] == []
    assert caveats["publication_blockers"] == []
    assert any("expected sparse nulls:" in caveat for caveat in caveats["accepted_caveats"])
    assert any("slice-trivial constants:" in caveat for caveat in caveats["accepted_caveats"])


def test_truth_keeps_unexpected_nulls_and_blocking_constants_visible_as_green_blockers(tmp_path) -> None:
    dataset_df = _phase4_truth_dataset(rows=40, htf_nulls=24, signed_run_length_nulls=21, minute_constant=True)
    split_frames = _phase4_truth_split_frames(dataset_df)
    feature_specs = _phase4_truth_feature_specs()
    label_pack = _phase4_truth_label_pack()
    spec = _phase4_truth_spec()
    manifest = _phase4_truth_manifest(
        spec,
        feature_specs,
        label_pack,
        artifact_id="dataset.xau_nonhuman.blockers",
    )
    state_df = pl.DataFrame(
        {
            "quality_score": [90.0] * len(dataset_df),
            "conflict_flag": [False] * len(dataset_df),
            "trust_flags": [[] for _ in range(len(dataset_df))],
        }
    )

    report = TruthService().evaluate_dataset(
        artifact_id=manifest.artifact_id,
        dataset_df=dataset_df,
        split_frames=split_frames,
        spec=spec,
        feature_specs=feature_specs,
        label_pack=label_pack,
        manifest=manifest,
        paths=StoragePaths(tmp_path / "pipeline_data"),
        store=ParquetStore(compression="snappy", row_group_size=1000),
        state_df=state_df,
        expected_content_hash="good-hash",
    )

    caveats = report.metrics["quality_caveat_summary"]

    assert report.status == "accepted"
    assert report.accepted_for_publication is True
    assert any(reason.startswith("unexpected nulls remain in: signed_run_length=21") for reason in report.warning_reasons)
    assert any("blocking constant columns remain: minute" in reason for reason in report.warning_reasons)
    assert any("unexpected nulls remain:" in blocker for blocker in caveats["green_blockers"])
    assert any("blocking constant columns remain: minute" in blocker for blocker in caveats["green_blockers"])
