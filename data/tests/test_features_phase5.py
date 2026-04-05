"""Phase 5 trainability diagnostics and experiment metadata tests."""

from __future__ import annotations

import datetime as dt

import polars as pl

from mt5pipe.config.models import DatasetConfig
from mt5pipe.features.public import FeatureService, get_default_feature_specs
from mt5pipe.labels.public import LabelService, get_default_label_packs, resolve_label_pack


UTC = dt.timezone.utc


class _CatalogStub:
    def __init__(self) -> None:
        self.manifests: list[object] = []

    def register_artifact(self, manifest, manifest_uri: str, *, detail: str = "") -> None:
        self.manifests.append((manifest, manifest_uri, detail))


def _phase5_feature_frame(rows: int = 180) -> pl.DataFrame:
    times = [dt.datetime(2026, 4, 3, 0, 0, tzinfo=UTC) + dt.timedelta(minutes=i) for i in range(rows)]
    close = [3000.0 + i * 0.07 + ((i % 11) - 5) * 0.05 for i in range(rows)]
    tick_count = [max(1, 9 + (i % 7) - (4 if i % 29 == 0 else 0)) for i in range(rows)]
    dual_source_ticks = [max(0, count - (1 if i % 9 == 0 else 0)) for i, count in enumerate(tick_count)]
    dual_source_ratio = [round(dual_source_ticks[i] / count, 8) if count else 0.0 for i, count in enumerate(tick_count)]

    return pl.DataFrame(
        {
            "symbol": ["XAUUSD"] * rows,
            "timeframe": ["M1"] * rows,
            "time_utc": times,
            "open": close,
            "high": [value + 0.22 + (i % 5) * 0.01 for i, value in enumerate(close)],
            "low": [value - 0.18 - (i % 3) * 0.01 for i, value in enumerate(close)],
            "close": close,
            "tick_count": tick_count,
            "spread_mean": [0.08 + (i % 4) * 0.01 for i in range(rows)],
            "mid_return": [((i % 9) - 4) * 0.00012 for i in range(rows)],
            "realized_vol": [0.0004 + (i % 8) * 0.00005 for i in range(rows)],
            "source_count": [2 if i % 4 else 1 for i in range(rows)],
            "conflict_count": [2 if i % 31 == 0 else (1 if i % 17 == 0 else 0) for i in range(rows)],
            "dual_source_ticks": dual_source_ticks,
            "secondary_present_ticks": [min(count, dual_source_ticks[i] + (1 if i % 6 == 0 else 0)) for i, count in enumerate(tick_count)],
            "dual_source_ratio": dual_source_ratio,
            "_filled": [i % 41 == 0 for i in range(rows)],
        }
    )


def _phase5_label_frame(rows: int = 320, *, step: float = 0.02) -> pl.DataFrame:
    times = [dt.datetime(2026, 4, 3, 0, 0, tzinfo=UTC) + dt.timedelta(minutes=i) for i in range(rows)]
    base = [3000.0 + i * step for i in range(rows)]
    return pl.DataFrame(
        {
            "symbol": ["XAUUSD"] * rows,
            "timeframe": ["M1"] * rows,
            "time_utc": times,
            "open": base,
            "high": [value + 0.08 for value in base],
            "low": [value - 0.08 for value in base],
            "close": [value + 0.01 for value in base],
        }
    )


def test_stable_registry_exposes_experiment_metadata() -> None:
    stable_specs = [spec for spec in get_default_feature_specs() if spec.status == "stable"]
    stable_pack = resolve_label_pack("core_tb_volscaled@1.0.0")

    assert stable_specs
    assert all(spec.ablation_group for spec in stable_specs)
    assert all(spec.trainability_tags for spec in stable_specs)
    assert stable_pack.ablation_group == "core_nonhuman_targets"
    assert stable_pack.trainability_tags == [
        "multi_horizon",
        "multi_task",
        "strict_tail_nulls",
        "vol_scaled_barriers",
    ]
    assert stable_pack.target_groups == ["future_return", "direction", "triple_barrier", "excursion"]
    assert stable_pack.tail_policy == "strict_null"
    assert stable_pack.qa_policy_ref == "qa.label.default@1.0.0"


def test_feature_service_materializes_trainability_metadata(paths, store) -> None:
    frame = _phase5_feature_frame()
    spec = next(candidate for candidate in get_default_feature_specs() if candidate.family == "multiscale")
    catalog = _CatalogStub()
    service = FeatureService(paths, store, catalog, DatasetConfig())
    store.write(frame, paths.built_bars_file("XAUUSD", "M1", dt.date(2026, 4, 3)))

    result = service.materialize_features(
        symbol="XAUUSD",
        base_clock="M1",
        date_from=dt.date(2026, 4, 3),
        date_to=dt.date(2026, 4, 3),
        feature_specs=[spec],
        base_df=frame,
        state_artifact_id="state.XAUUSD.M1.demo",
        build_id="build.phase5.feature",
        dataset_spec_ref="xau_phase5@1.0.0",
        code_version="workspace-local",
    )

    metadata = result.artifacts[0].manifest.metadata
    diagnostics = metadata["trainability_diagnostics"]

    assert metadata["ablation_group"] == spec.ablation_group
    assert metadata["family_tags"] == spec.tags
    assert metadata["trainability_tags"] == spec.trainability_tags
    assert diagnostics["feature_key"] == spec.key
    assert diagnostics["row_count_total"] == frame.height
    assert diagnostics["warmup_excluded_rows"] == spec.warmup_rows - 1
    assert diagnostics["row_count_post_warmup"] == frame.height - (spec.warmup_rows - 1)
    assert diagnostics["complete_row_ratio_post_warmup"] is not None
    assert diagnostics["family_non_null_ratio_post_warmup"] is not None
    assert set(diagnostics["column_summaries"]) == set(spec.output_columns)
    assert "insufficient_rows_vs_warmup" not in diagnostics["warning_reasons"]


def test_label_service_materializes_trainability_and_target_distribution_metadata(paths, store) -> None:
    frame = _phase5_feature_frame(rows=320)
    pack = resolve_label_pack("core_tb_volscaled@1.0.0")
    catalog = _CatalogStub()
    service = LabelService(paths, store, catalog)
    store.write(frame, paths.built_bars_file("XAUUSD", "M1", dt.date(2026, 4, 3)))

    result = service.materialize_labels(
        symbol="XAUUSD",
        base_clock="M1",
        date_from=dt.date(2026, 4, 3),
        date_to=dt.date(2026, 4, 3),
        label_pack=pack,
        base_df=frame,
        state_artifact_id="state.XAUUSD.M1.demo",
        build_id="build.phase5.label",
        dataset_spec_ref="xau_phase5@1.0.0",
        code_version="workspace-local",
    )

    metadata = result.manifest.metadata
    diagnostics = metadata["label_diagnostics"]

    assert metadata["ablation_group"] == pack.ablation_group
    assert metadata["trainability_tags"] == pack.trainability_tags
    assert metadata["target_groups"] == pack.target_groups
    assert metadata["tail_policy"] == pack.tail_policy
    assert diagnostics["qa_policy_ref"] == pack.qa_policy_ref
    assert diagnostics["direction_threshold_bps"] == 0.0
    assert diagnostics["tail_mismatch_horizons"] == []
    assert set(diagnostics["target_distribution_summary"]["usable_rows_by_horizon"]) == {"5m", "15m", "60m", "240m"}
    assert set(diagnostics["target_distribution_summary"]["future_return_std_by_horizon"]) == {"5m", "15m", "60m", "240m"}
    assert "warning_reasons" in diagnostics


def test_label_service_respects_direction_threshold_parameter(paths, store) -> None:
    frame = _phase5_label_frame(step=0.01)
    base_pack = resolve_label_pack("core_tb_volscaled@1.0.0")
    pack = base_pack.model_copy(
        update={
            "parameters": {
                **base_pack.parameters,
                "direction_threshold_bps": 20.0,
            }
        },
        deep=True,
    )
    catalog = _CatalogStub()
    service = LabelService(paths, store, catalog)
    store.write(frame, paths.built_bars_file("XAUUSD", "M1", dt.date(2026, 4, 3)))

    result = service.materialize_labels(
        symbol="XAUUSD",
        base_clock="M1",
        date_from=dt.date(2026, 4, 3),
        date_to=dt.date(2026, 4, 3),
        label_pack=pack,
        base_df=frame,
        state_artifact_id="state.XAUUSD.M1.demo",
        build_id="build.phase5.threshold",
        dataset_spec_ref="xau_phase5@1.0.0",
        code_version="workspace-local",
    )

    diagnostics = result.manifest.metadata["label_diagnostics"]

    assert result.label_df["direction_5m"].drop_nulls().n_unique() == 1
    assert result.label_df["direction_5m"].drop_nulls()[0] == 0
    assert diagnostics["direction_threshold_bps"] == 20.0
    assert "5m" in diagnostics["degenerate_direction_horizons"]


def test_public_label_registry_defaults_remain_single_stable_pack() -> None:
    packs = get_default_label_packs()

    assert len([pack for pack in packs if pack.status == "stable"]) == 1
    assert packs[0].key == "core_tb_volscaled@1.0.0"
