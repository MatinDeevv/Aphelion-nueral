"""Ingestion checkpoint model for resumable backfill."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field

from mt5pipe.utils.time import utc_now


class IngestionCheckpoint(BaseModel):
    """Tracks progress of a backfill or live ingestion task."""

    broker_id: str
    symbol: str
    data_type: str = Field(description="'ticks', 'bars', 'history_orders', 'history_deals', etc.")
    timeframe: str = Field(default="", description="Timeframe for bars, empty for ticks")
    last_timestamp_utc: dt.datetime = Field(description="Last successfully ingested timestamp")
    last_time_msc: int = Field(default=0, description="Last tick time_msc for dedup")
    rows_ingested: int = Field(default=0)
    updated_at: dt.datetime = Field(default_factory=utc_now)
    status: str = Field(default="active", description="'active', 'completed', 'failed'")
