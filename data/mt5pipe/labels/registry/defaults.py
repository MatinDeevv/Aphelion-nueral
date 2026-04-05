"""Default stable label packs."""

from __future__ import annotations

from mt5pipe.labels.registry.models import LabelPack


def get_default_label_packs() -> list[LabelPack]:
    horizons = [5, 15, 60, 240]
    output_columns: list[str] = []
    for h in horizons:
        output_columns.extend([
            f"future_return_{h}m",
            f"direction_{h}m",
            f"triple_barrier_{h}m",
            f"mae_{h}m",
            f"mfe_{h}m",
        ])

    return [
        LabelPack(
            label_pack_name="core_tb_volscaled",
            version="1.0.0",
            description="Multi-horizon future returns, direction, and vol-scaled triple-barrier labels",
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
                "direction_threshold_bps": 0.0,
            },
            exclusions=["filled_rows"],
            purge_rows=max(horizons) + 1,
            output_columns=output_columns,
            qa_policy_ref="qa.label.default@1.0.0",
            status="stable",
            ablation_group="core_nonhuman_targets",
            trainability_tags=["multi_horizon", "multi_task", "strict_tail_nulls", "vol_scaled_barriers"],
            target_groups=["future_return", "direction", "triple_barrier", "excursion"],
            tail_policy="strict_null",
        )
    ]


def resolve_label_pack(ref: str) -> LabelPack:
    for pack in get_default_label_packs():
        if ref in {pack.key, pack.label_pack_name, f"{pack.label_pack_name}:{pack.version}"}:
            return pack
    raise KeyError(f"Label pack '{ref}' was not found in the default registry")
