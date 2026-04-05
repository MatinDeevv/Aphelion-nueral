"""Tests for checkpoint resume logic."""

from __future__ import annotations

import datetime as dt

import pytest
import polars as pl

from mt5pipe.backfill.engine import BackfillEngine
from mt5pipe.config.models import BackfillConfig
from mt5pipe.ingestion.ticks import store_ticks_by_date
from mt5pipe.models.checkpoint import IngestionCheckpoint
from mt5pipe.storage.checkpoint_db import CheckpointDB


class TestCheckpointDB:
    def test_upsert_and_get(self, checkpoint_db: CheckpointDB) -> None:
        cp = IngestionCheckpoint(
            broker_id="broker_a",
            symbol="XAUUSD",
            data_type="ticks",
            last_timestamp_utc=dt.datetime(2024, 1, 15, tzinfo=dt.timezone.utc),
            rows_ingested=1000,
        )
        checkpoint_db.upsert_checkpoint(cp)

        result = checkpoint_db.get_checkpoint("broker_a", "XAUUSD", "ticks")
        assert result is not None
        assert result.broker_id == "broker_a"
        assert result.rows_ingested == 1000
        assert result.last_timestamp_utc.year == 2024

    def test_upsert_overwrites(self, checkpoint_db: CheckpointDB) -> None:
        cp1 = IngestionCheckpoint(
            broker_id="broker_a",
            symbol="XAUUSD",
            data_type="ticks",
            last_timestamp_utc=dt.datetime(2024, 1, 15, tzinfo=dt.timezone.utc),
            rows_ingested=1000,
        )
        checkpoint_db.upsert_checkpoint(cp1)

        cp2 = IngestionCheckpoint(
            broker_id="broker_a",
            symbol="XAUUSD",
            data_type="ticks",
            last_timestamp_utc=dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc),
            rows_ingested=5000,
        )
        checkpoint_db.upsert_checkpoint(cp2)

        result = checkpoint_db.get_checkpoint("broker_a", "XAUUSD", "ticks")
        assert result is not None
        assert result.rows_ingested == 5000
        assert result.last_timestamp_utc.month == 6

    def test_get_nonexistent_returns_none(self, checkpoint_db: CheckpointDB) -> None:
        result = checkpoint_db.get_checkpoint("nonexistent", "XAUUSD", "ticks")
        assert result is None

    def test_list_checkpoints(self, checkpoint_db: CheckpointDB) -> None:
        for broker in ["broker_a", "broker_b"]:
            cp = IngestionCheckpoint(
                broker_id=broker,
                symbol="XAUUSD",
                data_type="ticks",
                last_timestamp_utc=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
                rows_ingested=100,
            )
            checkpoint_db.upsert_checkpoint(cp)

        all_cps = checkpoint_db.list_checkpoints()
        assert len(all_cps) == 2

        broker_a_cps = checkpoint_db.list_checkpoints("broker_a")
        assert len(broker_a_cps) == 1

    def test_different_timeframes(self, checkpoint_db: CheckpointDB) -> None:
        for tf in ["M1", "M5", "H1"]:
            cp = IngestionCheckpoint(
                broker_id="broker_a",
                symbol="XAUUSD",
                data_type="bars",
                timeframe=tf,
                last_timestamp_utc=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
                rows_ingested=100,
            )
            checkpoint_db.upsert_checkpoint(cp)

        all_cps = checkpoint_db.list_checkpoints("broker_a")
        assert len(all_cps) == 3


class TestJobLog:
    def test_start_and_finish_job(self, checkpoint_db: CheckpointDB) -> None:
        job_id = checkpoint_db.start_job("backfill_ticks", "broker_a", "XAUUSD")
        assert job_id > 0

        checkpoint_db.finish_job(job_id, "completed", rows_processed=5000)
        # No exception = success


def test_store_ticks_by_date_reports_net_new_rows_and_registers_manifest(paths, store, checkpoint_db) -> None:
    date = dt.date(2026, 4, 2)
    times = [dt.datetime(2026, 4, 2, 0, minute, tzinfo=dt.timezone.utc) for minute in range(3)]
    first_batch = pl.DataFrame(
        {
            "broker_id": ["broker_a"] * 3,
            "symbol": ["XAUUSD"] * 3,
            "time_utc": times,
            "time_msc": [int(ts.timestamp() * 1000) for ts in times],
            "bid": [3000.0, 3000.1, 3000.2],
            "ask": [3000.2, 3000.3, 3000.4],
            "last": [0.0, 0.0, 0.0],
            "volume": [1.0, 1.0, 1.0],
            "volume_real": [0.0, 0.0, 0.0],
            "flags": [6, 6, 6],
            "ingest_ts": times,
        }
    )

    first_added = store_ticks_by_date(first_batch, "broker_a", "XAUUSD", paths, store, checkpoint_db)
    assert first_added == 3
    assert checkpoint_db.get_total_rows("broker_a", "ticks") == 3

    second_times = times + [dt.datetime(2026, 4, 2, 0, 3, tzinfo=dt.timezone.utc)]
    second_batch = pl.DataFrame(
        {
            "broker_id": ["broker_a"] * 4,
            "symbol": ["XAUUSD"] * 4,
            "time_utc": second_times,
            "time_msc": [int(ts.timestamp() * 1000) for ts in second_times],
            "bid": [3000.0, 3000.1, 3000.2, 3000.3],
            "ask": [3000.2, 3000.3, 3000.4, 3000.5],
            "last": [0.0, 0.0, 0.0, 0.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
            "volume_real": [0.0, 0.0, 0.0, 0.0],
            "flags": [6, 6, 6, 6],
            "ingest_ts": second_times,
        }
    )

    second_added = store_ticks_by_date(second_batch, "broker_a", "XAUUSD", paths, store, checkpoint_db)
    assert second_added == 1

    persisted = store.read(paths.raw_ticks_file("broker_a", "XAUUSD", date))
    assert persisted.height == 4
    assert checkpoint_db.get_total_rows("broker_a", "ticks") == 4


def test_backfill_ticks_for_day_gapfills_even_when_checkpoint_is_ahead(monkeypatch, paths, store, checkpoint_db) -> None:
    class FakeConn:
        broker_id = "broker_a"

        def ensure_connected(self) -> None:
            return None

    checkpoint_db.upsert_checkpoint(
        IngestionCheckpoint(
            broker_id="broker_a",
            symbol="XAUUSD",
            data_type="ticks",
            last_timestamp_utc=dt.datetime(2026, 4, 5, tzinfo=dt.timezone.utc),
            rows_ingested=100,
        )
    )

    fetch_calls: list[tuple[dt.datetime, dt.datetime]] = []

    def fake_fetch_ticks_chunk(conn, symbol, date_from, date_to):
        fetch_calls.append((date_from, date_to))
        ts = date_from
        return pl.DataFrame(
            {
                "broker_id": [conn.broker_id],
                "symbol": [symbol],
                "time_utc": [ts],
                "time_msc": [int(ts.timestamp() * 1000)],
                "bid": [3000.0],
                "ask": [3000.2],
                "last": [0.0],
                "volume": [1.0],
                "volume_real": [0.0],
                "flags": [6],
                "ingest_ts": [ts],
            }
        )

    monkeypatch.setattr("mt5pipe.backfill.engine.fetch_ticks_chunk", fake_fetch_ticks_chunk)

    engine = BackfillEngine(FakeConn(), paths, store, checkpoint_db, BackfillConfig(tick_chunk_hours=24))
    date = dt.date(2026, 4, 2)

    total_rows = engine.backfill_ticks_for_utc_day("XAUUSD", date)

    assert total_rows == 101
    assert fetch_calls
    persisted = store.read(paths.raw_ticks_file("broker_a", "XAUUSD", date))
    assert persisted.height == 1
    checkpoint = checkpoint_db.get_checkpoint("broker_a", "XAUUSD", "ticks")
    assert checkpoint is not None
    assert checkpoint.rows_ingested == 101
    assert checkpoint.last_timestamp_utc == dt.datetime(2026, 4, 5, tzinfo=dt.timezone.utc)
