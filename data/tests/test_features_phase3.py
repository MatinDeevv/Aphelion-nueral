"""Phase 3 feature-family and public-boundary tests."""

from __future__ import annotations

import datetime as dt

import polars as pl

from mt5pipe.features.disagreement import add_disagreement_features
from mt5pipe.features.entropy import add_entropy_features
from mt5pipe.features.event_shape import add_event_shape_features
from mt5pipe.features.public import (
    FeatureArtifactRef,
    FeatureBuilder,
    FeatureService,
    FeatureSpec,
    LabelArtifactRef,
    LabelPack,
    LabelService,
    get_default_feature_specs,
    load_feature_artifact,
    load_label_artifact,
    resolve_feature_selectors,
)
from mt5pipe.features.registry.defaults import get_default_feature_specs as get_default_feature_specs_direct
from mt5pipe.features.labels import add_direction_labels, add_future_returns, add_triple_barrier_labels
from mt5pipe.labels.public import get_default_label_packs, resolve_label_pack


UTC = dt.timezone.utc


def _phase3_input_frame(rows: int = 140) -> pl.DataFrame:
    times = [dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC) + dt.timedelta(minutes=i) for i in range(rows)]
    tick_count = [max(1, 8 + (i % 9) - (3 if i % 17 == 0 else 0)) for i in range(rows)]
    dual_source_ticks = [max(0, count - (1 if i % 11 == 0 else 0) - (1 if i % 23 == 0 else 0)) for i, count in enumerate(tick_count)]
    secondary_present = [min(count, dual_source_ticks[i] + (1 if i % 7 == 0 else 0)) for i, count in enumerate(tick_count)]
    dual_source_ratio = [round(dual_source_ticks[i] / count, 8) if count else 0.0 for i, count in enumerate(tick_count)]
    conflict_count = [2 if i % 29 == 0 else (1 if i % 13 == 0 else 0) for i in range(rows)]
    mid_return = [((i % 8) - 3) * 0.00018 for i in range(rows)]
    realized_vol = [0.0005 + (i % 6) * 0.00007 for i in range(rows)]
    close = [3000.0 + i * 0.15 + ((-1) ** i) * 0.2 for i in range(rows)]

    return pl.DataFrame(
        {
            "symbol": ["XAUUSD"] * rows,
            "timeframe": ["M1"] * rows,
            "time_utc": times,
            "open": close,
            "high": [value + 0.25 for value in close],
            "low": [value - 0.25 for value in close],
            "close": close,
            "tick_count": tick_count,
            "spread_mean": [0.08 + (i % 5) * 0.01 for i in range(rows)],
            "mid_return": mid_return,
            "realized_vol": realized_vol,
            "source_count": [2 if i % 5 else 1 for i in range(rows)],
            "conflict_count": conflict_count,
            "dual_source_ticks": dual_source_ticks,
            "secondary_present_ticks": secondary_present,
            "dual_source_ratio": dual_source_ratio,
            "_filled": [i % 19 == 0 for i in range(rows)],
        }
    )


def test_phase3_registry_selectors_resolve_stable_families() -> None:
    specs = get_default_feature_specs()
    keys = {spec.key for spec in specs}

    assert "disagreement.microstructure_pressure@1.0.0" in keys
    assert "event_shape.flow_shape@1.0.0" in keys
    assert "entropy.market_complexity@1.0.0" in keys

    resolved = resolve_feature_selectors(["disagreement/*", "event_shape/*", "entropy/*"])
    assert {spec.family for spec in resolved} == {"disagreement", "event_shape", "entropy"}
    assert all(spec.status == "stable" for spec in resolved)
    assert all(spec.point_in_time_safe for spec in resolved)


def test_disagreement_builder_outputs_registered_columns_and_warmup() -> None:
    frame = _phase3_input_frame()
    spec = next(spec for spec in get_default_feature_specs_direct() if spec.family == "disagreement")

    result = add_disagreement_features(frame)

    for column in spec.output_columns:
        assert column in result.columns
    assert result["conflict_burst_15"][:14].is_null().all()
    assert result["disagreement_entropy_30"][:29].is_null().all()
    assert result["disagreement_zscore_60"][:59].is_null().all()
    assert result["disagreement_zscore_60"][70] is not None


def test_event_shape_builder_outputs_registered_columns_and_warmup() -> None:
    frame = _phase3_input_frame()
    spec = next(spec for spec in get_default_feature_specs_direct() if spec.family == "event_shape")

    result = add_event_shape_features(frame, bar_duration_seconds=60)
    rolling_columns = {
        "burstiness_20",
        "silence_ratio_20",
        "direction_switch_rate_20",
        "path_efficiency_20",
        "tortuosity_20",
    }

    for column in spec.output_columns:
        assert column in result.columns
        if column in rolling_columns:
            assert result[column][:19].is_null().all()
        else:
            assert result[column][0] is not None
    assert result["signed_run_length"][30] is not None


def test_entropy_builder_outputs_registered_columns_and_warmup() -> None:
    frame = _phase3_input_frame()
    spec = next(spec for spec in get_default_feature_specs_direct() if spec.family == "entropy")

    result = add_entropy_features(frame)

    for column in spec.output_columns:
        assert column in result.columns
    for column in spec.output_columns:
        assert result[column][:29].is_null().all()
        assert result[column][45] is not None


def test_phase3_feature_builders_are_point_in_time_safe() -> None:
    base = _phase3_input_frame()
    prefix_rows = 100

    future_spike = base.with_columns(
        pl.when(pl.arange(0, pl.len()) >= prefix_rows)
        .then(pl.lit(25))
        .otherwise(pl.col("conflict_count"))
        .alias("conflict_count"),
        pl.when(pl.arange(0, pl.len()) >= prefix_rows)
        .then(pl.lit(0.02))
        .otherwise(pl.col("mid_return"))
        .alias("mid_return"),
        pl.when(pl.arange(0, pl.len()) >= prefix_rows)
        .then(pl.lit(0.01))
        .otherwise(pl.col("realized_vol"))
        .alias("realized_vol"),
    )

    disagreement_full = add_disagreement_features(future_spike)
    disagreement_prefix = add_disagreement_features(future_spike.head(prefix_rows))
    event_full = add_event_shape_features(future_spike, bar_duration_seconds=60)
    event_prefix = add_event_shape_features(future_spike.head(prefix_rows), bar_duration_seconds=60)
    entropy_full = add_entropy_features(future_spike)
    entropy_prefix = add_entropy_features(future_spike.head(prefix_rows))

    for column in [
        "disagreement_pressure_bps",
        "disagreement_zscore_60",
        "disagreement_burst_15",
        "direction_switch_rate_20",
        "path_efficiency_20",
        "return_permutation_entropy_30",
        "volatility_approx_entropy_30",
    ]:
        if column in disagreement_full.columns:
            assert disagreement_full[column].head(prefix_rows).to_list() == disagreement_prefix[column].to_list()
        elif column in event_full.columns:
            assert event_full[column].head(prefix_rows).to_list() == event_prefix[column].to_list()
        else:
            assert entropy_full[column].head(prefix_rows).to_list() == entropy_prefix[column].to_list()


def test_public_feature_and_label_symbols_are_available() -> None:
    assert FeatureSpec is not None
    assert FeatureService is not None
    assert FeatureBuilder is not None
    assert FeatureArtifactRef is not None
    assert LabelPack is not None
    assert LabelService is not None
    assert LabelArtifactRef is not None
    assert callable(load_feature_artifact)
    assert callable(load_label_artifact)


def test_feature_and_label_artifact_loaders_round_trip(paths, store) -> None:
    frame = _phase3_input_frame(rows=40)
    feature_spec = next(spec for spec in get_default_feature_specs_direct() if spec.family == "disagreement")
    feature_frame = add_disagreement_features(frame).select(["symbol", "timeframe", "time_utc", *feature_spec.output_columns])
    date_val = dt.date(2026, 4, 1)
    store.write(feature_frame, paths.feature_view_file(feature_spec.key, feature_spec.output_clock, date_val))

    feature_ref = FeatureArtifactRef(feature_key=feature_spec.key, clock=feature_spec.output_clock)
    loaded_feature = load_feature_artifact(paths, store, feature_ref, date_from=date_val, date_to=date_val)
    assert loaded_feature.height == feature_frame.height
    assert set(feature_spec.output_columns).issubset(set(loaded_feature.columns))

    label_pack = resolve_label_pack("core_tb_volscaled@1.0.0")
    label_frame = add_future_returns(frame, label_pack.horizons_minutes)
    label_frame = add_direction_labels(label_frame, label_pack.horizons_minutes)
    label_frame = add_triple_barrier_labels(
        label_frame,
        label_pack.horizons_minutes,
        tp_bps=float(label_pack.parameters["tp_bps"]),
        sl_bps=float(label_pack.parameters["sl_bps"]),
        vol_scale_window=int(label_pack.parameters["vol_scale_window"]),
        vol_multiplier=float(label_pack.parameters["vol_multiplier"]),
    ).select(["symbol", "timeframe", "time_utc", *label_pack.output_columns])
    store.write(label_frame, paths.label_view_file(label_pack.key, label_pack.base_clock, date_val))

    label_ref = LabelArtifactRef(label_pack_key=label_pack.key, clock=label_pack.base_clock)
    loaded_label = load_label_artifact(paths, store, label_ref, date_from=date_val, date_to=date_val)
    assert loaded_label.height == label_frame.height
    assert set(label_pack.output_columns).issubset(set(loaded_label.columns))


def test_label_registry_remains_compatible() -> None:
    pack = resolve_label_pack("core_tb_volscaled@1.0.0")
    public_packs = get_default_label_packs()

    assert pack.status == "stable"
    assert pack.key == "core_tb_volscaled@1.0.0"
    assert any(candidate.key == pack.key for candidate in public_packs)
