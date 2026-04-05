"""Phase 4 hardening tests for machine-native features and labels."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from mt5pipe.features.disagreement import add_disagreement_features
from mt5pipe.features.entropy import add_entropy_features
from mt5pipe.features.event_shape import add_event_shape_features
from mt5pipe.features.multiscale import add_multiscale_features
from mt5pipe.features.public import add_multiscale_features as add_multiscale_features_public
from mt5pipe.features.public import get_default_feature_specs, resolve_feature_selectors
from mt5pipe.features.labels import add_direction_labels, add_future_returns, add_triple_barrier_labels
from mt5pipe.labels.public import resolve_label_pack
from mt5pipe.labels.service import _label_manifest_diagnostics


UTC = dt.timezone.utc


def _phase4_input_frame(rows: int = 180) -> pl.DataFrame:
    times = [dt.datetime(2026, 4, 2, 0, 0, tzinfo=UTC) + dt.timedelta(minutes=i) for i in range(rows)]
    close = [3000.0 + i * 0.11 + ((-1) ** i) * 0.18 + ((i % 9) - 4) * 0.03 for i in range(rows)]
    tick_count = [max(0, 7 + (i % 8) - (5 if i % 23 == 0 else 0)) for i in range(rows)]
    dual_source_ticks = [max(0, count - (1 if i % 11 == 0 else 0)) for i, count in enumerate(tick_count)]
    secondary_present = [max(dual_source_ticks[i], min(count, dual_source_ticks[i] + (1 if i % 5 == 0 else 0))) for i, count in enumerate(tick_count)]
    dual_source_ratio = [round(dual_source_ticks[i] / count, 8) if count else 0.0 for i, count in enumerate(tick_count)]

    return pl.DataFrame(
        {
            "symbol": ["XAUUSD"] * rows,
            "timeframe": ["M1"] * rows,
            "time_utc": times,
            "open": close,
            "high": [value + 0.32 + (i % 4) * 0.01 for i, value in enumerate(close)],
            "low": [value - 0.28 - (i % 3) * 0.01 for i, value in enumerate(close)],
            "close": close,
            "tick_count": tick_count,
            "spread_mean": [0.07 + (i % 6) * 0.01 for i in range(rows)],
            "mid_return": [((i % 10) - 4) * 0.00015 for i in range(rows)],
            "realized_vol": [0.0004 + (i % 7) * 0.00009 for i in range(rows)],
            "source_count": [2 if i % 4 else 1 for i in range(rows)],
            "conflict_count": [2 if i % 31 == 0 else (1 if i % 13 == 0 else 0) for i in range(rows)],
            "dual_source_ticks": dual_source_ticks,
            "secondary_present_ticks": secondary_present,
            "dual_source_ratio": dual_source_ratio,
            "_filled": [i % 29 == 0 for i in range(rows)],
        }
    )


def _machine_native_spec(family: str):
    return next(spec for spec in get_default_feature_specs() if spec.family == family)


def test_phase4_registry_resolves_multiscale_and_hardened_families() -> None:
    resolved = resolve_feature_selectors(["disagreement/*", "event_shape/*", "entropy/*", "multiscale/*"])
    keys = {spec.key for spec in resolved}

    assert "multiscale.consistency@1.0.0" in keys
    for spec in resolved:
        assert spec.status == "stable"
        assert spec.point_in_time_safe is True
        assert spec.warmup_rows >= spec.lookback_rows
        assert spec.missingness_policy == "allow"
        assert spec.qa_policy_ref
        assert spec.output_columns
        assert spec.dependencies

    disagreement_spec = _machine_native_spec("disagreement")
    assert disagreement_spec.output_columns == [
        "mid_divergence_proxy_bps",
        "disagreement_pressure_bps",
        "disagreement_zscore_60",
        "disagreement_burst_15",
    ]

    htf_spec = next(spec for spec in get_default_feature_specs() if spec.family == "htf_context")
    assert all(not column.endswith("_tick_count") for column in htf_spec.output_columns)


@pytest.mark.parametrize(
    ("family", "builder", "builder_kwargs", "missing_columns"),
    [
        ("disagreement", add_disagreement_features, {}, ["tick_count"]),
        ("event_shape", add_event_shape_features, {"bar_duration_seconds": 60}, ["mid_return"]),
        ("entropy", add_entropy_features, {}, ["realized_vol", "spread_mean"]),
        ("multiscale", add_multiscale_features, {}, ["high"]),
    ],
)
def test_machine_native_families_return_null_columns_when_core_inputs_are_missing(
    family: str,
    builder,
    builder_kwargs: dict[str, object],
    missing_columns: list[str],
) -> None:
    frame = _phase4_input_frame().drop(missing_columns)
    spec = _machine_native_spec(family)

    result = builder(frame, **builder_kwargs)

    for column in spec.output_columns:
        assert column in result.columns
        assert result[column].null_count() == result.height


@pytest.mark.parametrize(
    ("family", "builder", "builder_kwargs", "expected_warmups"),
    [
        (
            "disagreement",
            add_disagreement_features,
            {},
            {
                "mid_divergence_proxy_bps": 0,
                "disagreement_pressure_bps": 0,
                "disagreement_burst_15": 14,
                "disagreement_zscore_60": 59,
            },
        ),
        (
            "event_shape",
            add_event_shape_features,
            {"bar_duration_seconds": 60},
            {
                "tick_rate_hz": 0,
                "interarrival_mean_ms": 0,
                "signed_run_length": 0,
                "burstiness_20": 19,
                "silence_ratio_20": 19,
                "direction_switch_rate_20": 19,
                "path_efficiency_20": 19,
                "tortuosity_20": 19,
            },
        ),
        (
            "entropy",
            add_entropy_features,
            {},
            {
                "return_sign_shannon_entropy_30": 29,
                "return_permutation_entropy_30": 29,
                "return_sample_entropy_30": 29,
                "volatility_approx_entropy_30": 29,
            },
        ),
        (
            "multiscale",
            add_multiscale_features,
            {},
            {
                "trend_alignment_5_15_60": 59,
                "return_energy_ratio_5_60": 59,
                "volatility_ratio_5_60": 59,
                "range_expansion_ratio_15_60": 59,
                "tick_intensity_ratio_5_60": 59,
            },
        ),
    ],
)
def test_machine_native_families_enforce_expected_warmups(
    family: str,
    builder,
    builder_kwargs: dict[str, object],
    expected_warmups: dict[str, int],
) -> None:
    frame = _phase4_input_frame()
    spec = _machine_native_spec(family)

    result = builder(frame, **builder_kwargs)

    assert set(spec.output_columns) == set(expected_warmups)
    for column, null_prefix_rows in expected_warmups.items():
        if null_prefix_rows > 0:
            assert result[column][:null_prefix_rows].is_null().all()
        assert result[column][null_prefix_rows] is not None


def test_multiscale_builder_outputs_registered_columns_and_is_point_in_time_safe() -> None:
    base = _phase4_input_frame()
    spec = _machine_native_spec("multiscale")
    prefix_rows = 120

    future_spike = base.with_columns(
        pl.when(pl.arange(0, pl.len()) >= prefix_rows)
        .then(pl.lit(0.004))
        .otherwise(pl.col("mid_return"))
        .alias("mid_return"),
        pl.when(pl.arange(0, pl.len()) >= prefix_rows)
        .then(pl.col("high") + 1.5)
        .otherwise(pl.col("high"))
        .alias("high"),
        pl.when(pl.arange(0, pl.len()) >= prefix_rows)
        .then(pl.col("low") - 1.5)
        .otherwise(pl.col("low"))
        .alias("low"),
        pl.when(pl.arange(0, pl.len()) >= prefix_rows)
        .then(pl.lit(25))
        .otherwise(pl.col("tick_count"))
        .alias("tick_count"),
    )

    result = add_multiscale_features(future_spike)
    prefix_result = add_multiscale_features(future_spike.head(prefix_rows))

    for column in spec.output_columns:
        assert column in result.columns
        assert result[column][:59].is_null().all()
        assert result[column].head(prefix_rows).to_list() == prefix_result[column].to_list()


def test_public_boundary_exports_multiscale_builder() -> None:
    assert add_multiscale_features_public is add_multiscale_features


def test_label_diagnostics_report_tail_nulls_and_class_balance() -> None:
    frame = _phase4_input_frame(rows=80)
    pack = resolve_label_pack("core_tb_volscaled@1.0.0")
    label_df = add_future_returns(frame, pack.horizons_minutes)
    label_df = add_direction_labels(label_df, pack.horizons_minutes)
    label_df = add_triple_barrier_labels(
        label_df,
        pack.horizons_minutes,
        tp_bps=float(pack.parameters["tp_bps"]),
        sl_bps=float(pack.parameters["sl_bps"]),
        vol_scale_window=int(pack.parameters["vol_scale_window"]),
        vol_multiplier=float(pack.parameters["vol_multiplier"]),
    ).select(["symbol", "timeframe", "time_utc", *pack.output_columns])

    diagnostics = _label_manifest_diagnostics(label_df, pack)

    assert diagnostics["horizons_minutes"] == pack.horizons_minutes
    assert diagnostics["max_horizon_minutes"] == max(pack.horizons_minutes)
    assert diagnostics["purge_rows"] == pack.purge_rows
    assert diagnostics["recommended_min_embargo_rows"] == pack.purge_rows
    assert diagnostics["exclusions"] == pack.exclusions
    assert {
        "direction_5m",
        "direction_15m",
        "direction_60m",
        "triple_barrier_5m",
        "triple_barrier_15m",
        "triple_barrier_60m",
    }.issubset(set(diagnostics["constant_output_columns"]))
    assert diagnostics["horizon_summaries"]["5m"]["future_return_null_rows"] == 5
    assert diagnostics["horizon_summaries"]["5m"]["direction_null_rows"] == 5
    assert diagnostics["horizon_summaries"]["5m"]["triple_barrier_null_rows"] == 5
    assert set(diagnostics["horizon_summaries"]["5m"]["triple_barrier_class_balance"]) == {"-1", "0", "1"}


def test_label_diagnostics_surface_constant_output_columns() -> None:
    pack = resolve_label_pack("core_tb_volscaled@1.0.0")
    label_df = pl.DataFrame(
        {
            "symbol": ["XAUUSD"] * 4,
            "timeframe": ["M1"] * 4,
            "time_utc": [dt.datetime(2026, 4, 2, 0, idx, tzinfo=UTC) for idx in range(4)],
            "future_return_5m": [0.1, 0.2, 0.3, 0.4],
            "direction_5m": [1, 1, 1, 1],
            "triple_barrier_5m": [0, 0, 0, 0],
            "mae_5m": [0.02, 0.02, 0.02, 0.02],
            "mfe_5m": [0.03, 0.03, 0.03, 0.03],
            "future_return_15m": [0.1, 0.2, 0.3, None],
            "direction_15m": [1, 1, 0, None],
            "triple_barrier_15m": [1, 0, -1, None],
            "mae_15m": [0.02, 0.03, 0.04, None],
            "mfe_15m": [0.03, 0.04, 0.05, None],
            "future_return_60m": [0.1, 0.0, -0.1, None],
            "direction_60m": [1, 0, -1, None],
            "triple_barrier_60m": [1, 0, -1, None],
            "mae_60m": [0.02, 0.03, 0.04, None],
            "mfe_60m": [0.03, 0.04, 0.05, None],
            "future_return_240m": [None, None, None, None],
            "direction_240m": [None, None, None, None],
            "triple_barrier_240m": [None, None, None, None],
            "mae_240m": [None, None, None, None],
            "mfe_240m": [None, None, None, None],
        }
    ).select(["symbol", "timeframe", "time_utc", *pack.output_columns])

    diagnostics = _label_manifest_diagnostics(label_df, pack)

    assert diagnostics["constant_output_columns"] == [
        "direction_5m",
        "mae_5m",
        "mfe_5m",
        "triple_barrier_5m",
    ]
