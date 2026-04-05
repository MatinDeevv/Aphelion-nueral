"""Compiler-owned registry helpers for dataset-facing contracts.

This module intentionally depends only on cross-sector public models plus the
compiler/catalog sector. It provides:

- stable built-in feature specs / label packs for legacy-compatible flows
- catalog-aware selector resolution
- a clean path for artifact-backed feature specs registered by other sectors
"""

from __future__ import annotations

from fnmatch import fnmatch

from mt5pipe.catalog.sqlite import CatalogDB
from mt5pipe.features.public import FeatureSpec, get_default_feature_specs
from mt5pipe.labels.public import LabelPack, get_default_label_packs


def builtin_feature_specs() -> list[FeatureSpec]:
    """Return compiler-known stable feature specs.

    These are the existing stable core families that the compiler can resolve
    without requiring pre-registered catalog contracts. Additional feature
    families should be registered in the catalog by their producer and will be
    picked up automatically by the resolver below.
    """

    public_stable_specs = [
        spec
        for spec in get_default_feature_specs()
        if getattr(spec, "status", "stable") == "stable"
    ]
    if public_stable_specs:
        return public_stable_specs

    return [
        FeatureSpec(
            feature_name="cyclical_time",
            family="time",
            version="1.0.0",
            description="Hour-of-day and weekday cyclical encodings",
            input_contract="BuiltBar",
            input_clock="M1",
            output_clock="M1",
            builder_ref="mt5pipe.features.time:add_time_features",
            output_columns=["hour", "minute", "weekday", "time_sin", "time_cos", "weekday_sin", "weekday_cos"],
            dependencies=["time_utc"],
            missingness_policy="fail",
            qa_policy_ref="qa.feature.default@1.0.0",
            status="stable",
            tags=["core", "time"],
        ),
        FeatureSpec(
            feature_name="session_flags",
            family="session",
            version="1.0.0",
            description="Asia/London/NY session flags",
            input_contract="BuiltBar",
            input_clock="M1",
            output_clock="M1",
            builder_ref="mt5pipe.features.session:add_session_features",
            output_columns=["session_asia", "session_london", "session_ny", "session_overlap"],
            dependencies=["time_utc"],
            missingness_policy="fail",
            qa_policy_ref="qa.feature.default@1.0.0",
            status="stable",
            tags=["core", "session"],
        ),
        FeatureSpec(
            feature_name="spread_quality",
            family="quality",
            version="1.0.0",
            description="Relative spread, conflict ratio, broker diversity",
            input_contract="BuiltBar",
            input_clock="M1",
            output_clock="M1",
            builder_ref="mt5pipe.features.quality:add_spread_quality_features",
            output_columns=["relative_spread", "conflict_ratio", "broker_diversity"],
            dependencies=["spread_mean", "close", "conflict_count", "tick_count", "source_count"],
            missingness_policy="fail",
            qa_policy_ref="qa.feature.default@1.0.0",
            status="stable",
            tags=["core", "quality"],
        ),
        FeatureSpec(
            feature_name="standard_context",
            family="htf_context",
            version="1.0.0",
            description="Lagged higher-timeframe bar context from configured context clocks",
            input_contract="BuiltBar",
            input_clock="M1",
            output_clock="M1",
            builder_ref="mt5pipe.features.context:add_lagged_bar_features",
            output_columns=[
                "M5_open", "M5_high", "M5_low", "M5_close", "M5_tick_count", "M5_spread_mean", "M5_mid_return", "M5_realized_vol",
                "M15_open", "M15_high", "M15_low", "M15_close", "M15_tick_count", "M15_spread_mean", "M15_mid_return", "M15_realized_vol",
                "H1_open", "H1_high", "H1_low", "H1_close", "H1_tick_count", "H1_spread_mean", "H1_mid_return", "H1_realized_vol",
                "H4_open", "H4_high", "H4_low", "H4_close", "H4_tick_count", "H4_spread_mean", "H4_mid_return", "H4_realized_vol",
                "D1_open", "D1_high", "D1_low", "D1_close", "D1_tick_count", "D1_spread_mean", "D1_mid_return", "D1_realized_vol",
            ],
            dependencies=["time_utc"],
            missingness_policy="allow",
            qa_policy_ref="qa.feature.default@1.0.0",
            status="stable",
            tags=["core", "context"],
        ),
    ]


def builtin_label_packs() -> list[LabelPack]:
    """Return compiler-known stable label packs."""

    public_stable_packs = [
        pack
        for pack in get_default_label_packs()
        if getattr(pack, "status", "stable") == "stable"
    ]
    if public_stable_packs:
        return public_stable_packs

    horizons = [5, 15, 60, 240]
    output_columns: list[str] = []
    for horizon in horizons:
        output_columns.extend(
            [
                f"future_return_{horizon}m",
                f"direction_{horizon}m",
                f"triple_barrier_{horizon}m",
                f"mae_{horizon}m",
                f"mfe_{horizon}m",
            ]
        )

    return [
        LabelPack(
            label_pack_name="core_tb_volscaled",
            version="1.0.0",
            description="Future returns, direction, and vol-scaled triple-barrier labels",
            base_clock="M1",
            horizons_minutes=horizons,
            generator_refs=[
                "mt5pipe.features.labels:add_future_returns",
                "mt5pipe.features.labels:add_direction_labels",
                "mt5pipe.features.labels:add_triple_barrier_labels",
            ],
            parameters={
                "tp_bps": 50.0,
                "sl_bps": 50.0,
                "vol_scale_window": 60,
                "vol_multiplier": 2.0,
            },
            exclusions=["filled_rows"],
            purge_rows=max(horizons) + 1,
            output_columns=output_columns,
            status="stable",
        )
    ]


def register_builtin_contracts(catalog: CatalogDB) -> None:
    """Register compiler-known stable contracts in the catalog."""

    catalog.register_feature_specs(builtin_feature_specs())
    catalog.register_label_packs(builtin_label_packs())


def resolve_feature_selectors(
    selectors: list[str],
    *,
    catalog: CatalogDB,
    extra_specs: list[FeatureSpec] | None = None,
) -> list[FeatureSpec]:
    """Resolve selectors against built-ins, catalog contracts, and caller extras."""

    candidates = _feature_spec_index(catalog=catalog, extra_specs=extra_specs)
    resolved: list[FeatureSpec] = []
    seen: set[str] = set()

    for selector in selectors:
        current = selector.strip()
        if not current:
            continue

        matched = False
        for spec in candidates.values():
            short_ref = f"{spec.family}/{spec.feature_name}"
            if (
                current == spec.key
                or current == short_ref
                or current == f"{spec.family}/*"
                or fnmatch(spec.key, current)
                or fnmatch(short_ref, current)
            ):
                matched = True
                if spec.key not in seen:
                    resolved.append(spec)
                    seen.add(spec.key)

        if not matched:
            visible_selectors = sorted({spec.key for spec in candidates.values()})
            visible_families = sorted({spec.family for spec in candidates.values()})
            raise KeyError(
                "Feature selector "
                f"'{current}' did not match any registered feature spec. "
                "Register the feature contract in the compiler catalog or use a compiler-known stable selector. "
                f"Visible families: {', '.join(visible_families)}. "
                f"Visible specs: {', '.join(visible_selectors)}."
            )

    return resolved


def resolve_label_pack(
    ref: str,
    *,
    catalog: CatalogDB,
    extra_packs: list[LabelPack] | None = None,
) -> LabelPack:
    """Resolve a label pack against built-ins, catalog contracts, and caller extras."""

    candidates = _label_pack_index(catalog=catalog, extra_packs=extra_packs)
    current = ref.strip()
    for pack in candidates.values():
        if current in {pack.key, pack.label_pack_name, f"{pack.label_pack_name}:{pack.version}"}:
            return pack
    raise KeyError(
        f"Label pack '{ref}' was not found. Register it in the compiler catalog or use a compiler-known stable label pack."
    )


def feature_spec_index(
    *,
    catalog: CatalogDB,
    extra_specs: list[FeatureSpec] | None = None,
) -> dict[str, FeatureSpec]:
    """Return all compiler-visible feature specs indexed by key."""

    return _feature_spec_index(catalog=catalog, extra_specs=extra_specs)


def label_pack_index(
    *,
    catalog: CatalogDB,
    extra_packs: list[LabelPack] | None = None,
) -> dict[str, LabelPack]:
    """Return all compiler-visible label packs indexed by key."""

    return _label_pack_index(catalog=catalog, extra_packs=extra_packs)


def _feature_spec_index(
    *,
    catalog: CatalogDB,
    extra_specs: list[FeatureSpec] | None = None,
) -> dict[str, FeatureSpec]:
    index = {spec.key: spec for spec in builtin_feature_specs()}
    index.update({spec.key: spec for spec in catalog.list_feature_specs()})
    if extra_specs:
        index.update({spec.key: spec for spec in extra_specs})
    return index


def _label_pack_index(
    *,
    catalog: CatalogDB,
    extra_packs: list[LabelPack] | None = None,
) -> dict[str, LabelPack]:
    index = {pack.key: pack for pack in builtin_label_packs()}
    index.update({pack.key: pack for pack in catalog.list_label_packs()})
    if extra_packs:
        index.update({pack.key: pack for pack in extra_packs})
    return index
