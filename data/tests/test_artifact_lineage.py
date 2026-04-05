"""Focused tests for immutable upstream artifact addressing."""

from __future__ import annotations

import datetime as dt

import polars as pl

from mt5pipe.contracts.artifacts import ArtifactKind
from mt5pipe.features.public import FeatureArtifactRef, load_feature_artifact
from mt5pipe.labels.public import LabelArtifactRef, load_label_artifact
from mt5pipe.state.public import StateArtifactRef, load_state_artifact


UTC = dt.timezone.utc


def test_public_loaders_read_artifact_scoped_state_feature_and_label_views(paths, store) -> None:
    date = dt.date(2026, 4, 2)
    time_utc = dt.datetime(2026, 4, 2, 0, 0, tzinfo=UTC)

    state_artifact_id = "state.XAUUSD.M1.abc123"
    state_df = pl.DataFrame(
        {
            "schema_version": ["1.0.0"],
            "state_version": ["state.default@1.0.0"],
            "snapshot_id": ["snapshot-1"],
            "symbol": ["XAUUSD"],
            "ts_utc": [time_utc],
            "ts_msc": [int(time_utc.timestamp() * 1000)],
            "clock": ["M1"],
            "window_start_utc": [time_utc],
            "window_end_utc": [time_utc],
            "bid": [3000.0],
            "ask": [3000.2],
            "mid": [3000.1],
            "spread": [0.2],
            "source_primary": ["broker_a"],
            "source_secondary": ["broker_b"],
            "source_count": [2],
            "merge_mode": ["best"],
            "conflict_flag": [False],
            "disagreement_bps": [0.0],
            "spread_disagreement_bps": [0.0],
            "broker_a_mid": [3000.1],
            "broker_b_mid": [3000.1],
            "broker_a_spread": [0.2],
            "broker_b_spread": [0.2],
            "primary_staleness_ms": [0],
            "secondary_staleness_ms": [0],
            "source_offset_ms": [0],
            "quality_score": [99.0],
            "source_quality_hint": ["fixture"],
            "expected_observations": [1],
            "observed_observations": [1],
            "missing_observations": [0],
            "window_completeness": [1.0],
            "session_code": ["london"],
            "event_flags": [[]],
            "trust_flags": [[]],
            "provenance_refs": [["bar://XAUUSD/M1/2026-04-02T00:00:00+00:00"]],
        }
    )
    store.write(
        state_df,
        paths.state_artifact_file("XAUUSD", "M1", date, "state.default@1.0.0", state_artifact_id),
    )

    state_ref = StateArtifactRef(
        artifact_id=state_artifact_id,
        kind=ArtifactKind.STATE,
        logical_name="XAUUSD.M1",
        symbol="XAUUSD",
        clock="M1",
        state_version="state.default@1.0.0",
        date_from=date,
        date_to=date,
    )
    loaded_state = load_state_artifact(paths, store, state_ref)
    assert loaded_state.height == 1
    assert loaded_state["snapshot_id"][0] == "snapshot-1"

    feature_artifact_id = "feature_view.time.cyclical_time.abc123"
    feature_df = pl.DataFrame(
        {
            "symbol": ["XAUUSD"],
            "timeframe": ["M1"],
            "time_utc": [time_utc],
            "minute": [0],
        }
    )
    store.write(
        feature_df,
        paths.feature_artifact_file("time.cyclical_time@1.0.0", "M1", feature_artifact_id, date),
    )
    loaded_feature = load_feature_artifact(
        paths,
        store,
        FeatureArtifactRef(
            artifact_id=feature_artifact_id,
            feature_key="time.cyclical_time@1.0.0",
            clock="M1",
        ),
        date_from=date,
        date_to=date,
    )
    assert loaded_feature.height == 1
    assert loaded_feature["minute"][0] == 0

    label_artifact_id = "label_view.core_tb_volscaled.abc123"
    label_df = pl.DataFrame(
        {
            "symbol": ["XAUUSD"],
            "timeframe": ["M1"],
            "time_utc": [time_utc],
            "direction_60m": [1],
        }
    )
    store.write(
        label_df,
        paths.label_artifact_file("core_tb_volscaled@1.0.0", "M1", label_artifact_id, date),
    )
    loaded_label = load_label_artifact(
        paths,
        store,
        LabelArtifactRef(
            artifact_id=label_artifact_id,
            label_pack_key="core_tb_volscaled@1.0.0",
            clock="M1",
        ),
        date_from=date,
        date_to=date,
    )
    assert loaded_label.height == 1
    assert loaded_label["direction_60m"][0] == 1
