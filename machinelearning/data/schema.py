"""Column schema definitions for the Phase 6 machine learning data layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

SYMBOL_COLUMN = "symbol"
TIMEFRAME_COLUMN = "timeframe"
TIME_INDEX_COLUMN = "time_utc"
FILLED_FLAG_COLUMN = "_filled"
SYMBOL_INDEX_COLUMN = "symbol_idx"

RAW_BAR_METADATA_COLUMNS = [
    SYMBOL_COLUMN,
    TIMEFRAME_COLUMN,
    TIME_INDEX_COLUMN,
]

TIME_FAMILY_COLUMNS = [
    "hour",
    "minute",
    "weekday",
    "time_sin",
    "time_cos",
    "weekday_sin",
    "weekday_cos",
]

SESSION_FAMILY_COLUMNS = [
    "session_asia",
    "session_london",
    "session_ny",
    "session_overlap",
]

QUALITY_FAMILY_COLUMNS = [
    "relative_spread",
    "conflict_ratio",
    "broker_diversity",
]


def _build_htf_context_columns() -> list[str]:
    timeframes = ("M5", "M15", "H1", "H4", "D1")
    suffixes = (
        "open",
        "high",
        "low",
        "close",
        "spread_mean",
        "mid_return",
        "realized_vol",
    )
    return [f"{timeframe}_{suffix}" for timeframe in timeframes for suffix in suffixes]


HTF_CONTEXT_COLUMNS = _build_htf_context_columns()

DISAGREEMENT_FAMILY_COLUMNS = [
    "mid_divergence_proxy_bps",
    "disagreement_pressure_bps",
    "disagreement_zscore_60",
    "disagreement_burst_15",
]

EVENT_SHAPE_FAMILY_COLUMNS = [
    "tick_rate_hz",
    "interarrival_mean_ms",
    "burstiness_20",
    "silence_ratio_20",
    "direction_switch_rate_20",
    "path_efficiency_20",
    "tortuosity_20",
    "signed_run_length",
]

ENTROPY_FAMILY_COLUMNS = [
    "return_sign_shannon_entropy_30",
    "return_permutation_entropy_30",
    "return_sample_entropy_30",
    "volatility_approx_entropy_30",
]

MULTISCALE_FAMILY_COLUMNS = [
    "trend_alignment_5_15_60",
    "return_energy_ratio_5_60",
    "volatility_ratio_5_60",
    "range_expansion_ratio_15_60",
    "tick_intensity_ratio_5_60",
]

BASE_BAR_COLUMNS = [
    SYMBOL_COLUMN,
    TIMEFRAME_COLUMN,
    TIME_INDEX_COLUMN,
    "open",
    "high",
    "low",
    "close",
    "tick_count",
    "spread_mean",
    "mid_return",
    "realized_vol",
    "source_count",
    "conflict_count",
    "dual_source_ratio",
    "dual_source_ticks",
    "secondary_present_ticks",
]

FUTURE_KNOWN_COLUMNS = [
    "time_sin",
    "time_cos",
    "weekday_sin",
    "weekday_cos",
    "session_asia",
    "session_london",
    "session_ny",
    "session_overlap",
]

STATIC_COLUMNS = [SYMBOL_INDEX_COLUMN]

FEATURE_COLUMNS = (
    TIME_FAMILY_COLUMNS
    + SESSION_FAMILY_COLUMNS
    + QUALITY_FAMILY_COLUMNS
    + HTF_CONTEXT_COLUMNS
    + DISAGREEMENT_FAMILY_COLUMNS
    + EVENT_SHAPE_FAMILY_COLUMNS
    + ENTROPY_FAMILY_COLUMNS
    + MULTISCALE_FAMILY_COLUMNS
    + BASE_BAR_COLUMNS
)

PAST_OBSERVED_COLUMNS = [
    column
    for column in FEATURE_COLUMNS
    if column not in FUTURE_KNOWN_COLUMNS and column not in RAW_BAR_METADATA_COLUMNS
]

NORMALIZED_FEATURE_COLUMNS = PAST_OBSERVED_COLUMNS + FUTURE_KNOWN_COLUMNS

TARGET_HORIZONS_MINUTES = (5, 15, 60, 240)
CLASSIFICATION_TARGET_PREFIXES = ("direction", "triple_barrier")
REGRESSION_TARGET_PREFIXES = ("future_return", "mae", "mfe")
TARGET_COLUMN_ORDER = (
    "future_return",
    "direction",
    "triple_barrier",
    "mae",
    "mfe",
)


def make_target_column(prefix: str, horizon: int) -> str:
    """Return the canonical target column name for a family/horizon pair."""

    return f"{prefix}_{horizon}m"


def _build_target_columns(prefixes: tuple[str, ...]) -> list[str]:
    return [
        make_target_column(prefix, horizon)
        for horizon in TARGET_HORIZONS_MINUTES
        for prefix in prefixes
    ]


CLASSIFICATION_TARGET_COLUMNS = _build_target_columns(CLASSIFICATION_TARGET_PREFIXES)
REGRESSION_TARGET_COLUMNS = _build_target_columns(REGRESSION_TARGET_PREFIXES)
TARGET_COLUMNS = _build_target_columns(TARGET_COLUMN_ORDER)

DIRECTION_TARGET_COLUMNS = [
    column for column in CLASSIFICATION_TARGET_COLUMNS if column.startswith("direction_")
]
TRIPLE_BARRIER_TARGET_COLUMNS = [
    column
    for column in CLASSIFICATION_TARGET_COLUMNS
    if column.startswith("triple_barrier_")
]
FUTURE_RETURN_TARGET_COLUMNS = [
    column for column in REGRESSION_TARGET_COLUMNS if column.startswith("future_return_")
]
MAE_TARGET_COLUMNS = [
    column for column in REGRESSION_TARGET_COLUMNS if column.startswith("mae_")
]
MFE_TARGET_COLUMNS = [
    column for column in REGRESSION_TARGET_COLUMNS if column.startswith("mfe_")
]


def is_classification_target(column: str) -> bool:
    """Return whether a target column is categorical and needs class remapping."""

    return column in CLASSIFICATION_TARGET_COLUMNS


def is_regression_target(column: str) -> bool:
    """Return whether a target column is continuous and should preserve NaNs."""

    return column in REGRESSION_TARGET_COLUMNS


@dataclass(frozen=True)
class ColumnSchema:
    """Frozen schema for the Phase 6 parquet artifact columns."""

    past_observed: list[str]
    future_known: list[str]
    static: list[str]
    targets: list[str]
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "past_observed", list(self.past_observed))
        object.__setattr__(self, "future_known", list(self.future_known))
        object.__setattr__(self, "static", list(self.static))
        object.__setattr__(self, "targets", list(self.targets))

    @property
    def n_past(self) -> int:
        """Return the number of past-observed feature columns."""

        return len(self.past_observed)

    @property
    def n_future(self) -> int:
        """Return the number of future-known feature columns."""

        return len(self.future_known)

    @property
    def n_static(self) -> int:
        """Return the number of static feature columns."""

        return len(self.static)

    def validate_dataframe_columns(self, cols: Iterable[str]) -> list[str]:
        """Return required schema columns that are missing from a dataframe."""

        present = set(cols)
        required = self.past_observed + self.future_known + self.static + self.targets
        return [column for column in required if column not in present]


DEFAULT_SCHEMA = ColumnSchema(
    past_observed=PAST_OBSERVED_COLUMNS,
    future_known=FUTURE_KNOWN_COLUMNS,
    static=STATIC_COLUMNS,
    targets=TARGET_COLUMNS,
)

__all__ = [
    "BASE_BAR_COLUMNS",
    "CLASSIFICATION_TARGET_COLUMNS",
    "CLASSIFICATION_TARGET_PREFIXES",
    "ColumnSchema",
    "DEFAULT_SCHEMA",
    "DIRECTION_TARGET_COLUMNS",
    "DISAGREEMENT_FAMILY_COLUMNS",
    "ENTROPY_FAMILY_COLUMNS",
    "EVENT_SHAPE_FAMILY_COLUMNS",
    "FEATURE_COLUMNS",
    "FILLED_FLAG_COLUMN",
    "FUTURE_KNOWN_COLUMNS",
    "FUTURE_RETURN_TARGET_COLUMNS",
    "HTF_CONTEXT_COLUMNS",
    "MAE_TARGET_COLUMNS",
    "MFE_TARGET_COLUMNS",
    "MULTISCALE_FAMILY_COLUMNS",
    "NORMALIZED_FEATURE_COLUMNS",
    "PAST_OBSERVED_COLUMNS",
    "QUALITY_FAMILY_COLUMNS",
    "RAW_BAR_METADATA_COLUMNS",
    "REGRESSION_TARGET_COLUMNS",
    "REGRESSION_TARGET_PREFIXES",
    "SESSION_FAMILY_COLUMNS",
    "STATIC_COLUMNS",
    "SYMBOL_COLUMN",
    "SYMBOL_INDEX_COLUMN",
    "TARGET_COLUMNS",
    "TARGET_COLUMN_ORDER",
    "TARGET_HORIZONS_MINUTES",
    "TIME_FAMILY_COLUMNS",
    "TIMEFRAME_COLUMN",
    "TIME_INDEX_COLUMN",
    "TRIPLE_BARRIER_TARGET_COLUMNS",
    "is_classification_target",
    "is_regression_target",
    "make_target_column",
]
