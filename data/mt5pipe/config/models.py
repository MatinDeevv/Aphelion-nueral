"""Pydantic configuration models."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, SecretStr


class BrokerConfig(BaseModel):
    """Configuration for a single MT5 broker terminal."""

    broker_id: str = Field(..., description="Unique broker identifier, e.g. 'broker_a'")
    terminal_path: Path = Field(..., description="Path to MT5 terminal64.exe")
    login: int = Field(..., description="MT5 account login number")
    # Optional: if terminal is already open/logged in, pipeline can connect via session.
    password: SecretStr | None = Field(default=None, description="MT5 account password")
    server: str = Field(..., description="MT5 server name")
    timeout_ms: int = Field(default=30_000, description="Connection timeout in ms")
    priority: int = Field(default=0, description="Broker priority for merge (lower = higher priority)")
    symbol_map: dict[str, str] | None = Field(
        default_factory=dict,
        description="Map canonical symbol -> broker symbol, e.g. {'XAUUSD': 'XAUUSD.raw'}",
    )

    def resolve_symbol(self, canonical: str) -> str:
        """Return broker-specific symbol name."""
        mapping = self.symbol_map or {}
        return mapping.get(canonical, canonical)


class StorageConfig(BaseModel):
    """Storage paths and settings."""

    root: Path = Field(default=Path("local_data/pipeline_data"), description="Root pipeline data directory")
    checkpoint_db: str = Field(default="checkpoints.db", description="SQLite DB filename inside root")
    parquet_row_group_size: int = Field(default=500_000)
    compression: str = Field(default="zstd")


class BackfillConfig(BaseModel):
    """Backfill engine settings."""

    tick_chunk_hours: int = Field(default=4, description="Hours per tick backfill chunk")
    bar_chunk_days: int = Field(default=30, description="Days per bar backfill chunk")
    history_chunk_days: int = Field(default=90, description="Days per order/deal history chunk")
    max_retries: int = Field(default=3)
    retry_wait_seconds: float = Field(default=2.0)


class LiveConfig(BaseModel):
    """Live collection settings."""

    tick_poll_seconds: float = Field(default=0.25, description="Tick poll interval")
    snapshot_interval_seconds: int = Field(default=60, description="Snapshot poll interval")
    market_book_interval_seconds: int = Field(default=5, description="DOM poll interval")
    flush_interval_seconds: int = Field(default=30, description="Buffer flush to disk interval")
    buffer_max_rows: int = Field(default=50_000, description="Max buffered rows before force flush")
    warmup_seconds: int = Field(
        default=120,
        description="Duration in seconds for pipeline-managed live collection (0=infinite, blocks pipeline)",
    )


class MergeConfig(BaseModel):
    """Canonical merge settings."""

    bucket_ms: int = Field(default=100, description="Merge bucket window in milliseconds")
    max_spread_ratio: float = Field(default=0.005, description="Max bid-ask spread as fraction of mid")
    max_mid_diff_ratio: float = Field(default=0.002, description="Max inter-broker mid diff as fraction")
    freshness_weight: float = Field(default=0.4)
    spread_weight: float = Field(default=0.3)
    continuity_weight: float = Field(default=0.3)
    conflict_log_threshold: float = Field(default=0.001, description="Mid diff ratio to flag as conflict")
    diagnostics_near_miss_factor: int = Field(
        default=5,
        description="Near-miss window factor vs bucket_ms for timestamp alignment diagnostics",
    )
    min_dual_source_ratio: float = Field(
        default=0.001,
        description="Minimum acceptable share of canonical rows with secondary source populated",
    )
    hard_fail_on_low_dual_source: bool = Field(
        default=True,
        description="Fail merge command when dual-source participation is below min_dual_source_ratio",
    )


class BarConfig(BaseModel):
    """Bar builder settings."""

    timeframes: list[str] = Field(
        default=[
            "M1", "M2", "M3", "M4", "M5", "M6", "M10", "M12", "M15", "M20", "M30",
            "H1", "H2", "H3", "H4", "H6", "H8", "H12",
            "D1", "W1", "MN1",
        ]
    )


class DatasetConfig(BaseModel):
    """Dataset builder settings."""

    base_timeframe: str = Field(default="M1")
    context_timeframes: list[str] = Field(default=["M5", "M15", "H1", "H4", "D1"])
    horizons_minutes: list[int] = Field(default=[5, 15, 30, 60, 240])
    triple_barrier_tp_bps: float = Field(default=50.0, description="Take-profit in basis points (floor when vol-scaled)")
    triple_barrier_sl_bps: float = Field(default=50.0, description="Stop-loss in basis points (floor when vol-scaled)")
    triple_barrier_vol_lookback: int = Field(
        default=60,
        description="Rolling window (bars) for vol-scaled barriers. 0 disables vol-scaling.",
    )
    triple_barrier_vol_multiplier: float = Field(
        default=2.0,
        description="Sigma multiplier for vol-scaled barrier width",
    )
    train_ratio: float = Field(default=0.7)
    val_ratio: float = Field(default=0.15)
    test_ratio: float = Field(default=0.15)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    json_output: bool = Field(default=False)


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    brokers: dict[str, BrokerConfig] = Field(..., description="Broker configs keyed by broker_id")
    symbols: list[str] = Field(default=["XAUUSD"])
    storage: StorageConfig = Field(default_factory=StorageConfig)
    backfill: BackfillConfig = Field(default_factory=BackfillConfig)
    live: LiveConfig = Field(default_factory=LiveConfig)
    merge: MergeConfig = Field(default_factory=MergeConfig)
    bars: BarConfig = Field(default_factory=BarConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def get_broker(self, broker_id: str) -> BrokerConfig:
        if broker_id not in self.brokers:
            raise KeyError(f"Broker '{broker_id}' not found in config. Available: {list(self.brokers)}")
        return self.brokers[broker_id]

    def broker_ids(self) -> list[str]:
        return list(self.brokers.keys())
