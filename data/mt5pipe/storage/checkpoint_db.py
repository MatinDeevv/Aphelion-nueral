"""SQLite checkpoint / manifest database for ingestion state tracking."""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

from mt5pipe.models.checkpoint import IngestionCheckpoint
from mt5pipe.utils.logging import get_logger
from mt5pipe.utils.time import utc_now

log = get_logger(__name__)

_CREATE_CHECKPOINTS_SQL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    broker_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    data_type TEXT NOT NULL,
    timeframe TEXT NOT NULL DEFAULT '',
    last_timestamp_utc TEXT NOT NULL,
    last_time_msc INTEGER NOT NULL DEFAULT 0,
    rows_ingested INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    PRIMARY KEY (broker_id, symbol, data_type, timeframe)
)
"""

_CREATE_JOB_LOG_SQL = """
CREATE TABLE IF NOT EXISTS job_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_type TEXT NOT NULL,
    broker_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    rows_processed INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    metadata_json TEXT
)
"""

_CREATE_MANIFEST_SQL = """
CREATE TABLE IF NOT EXISTS file_manifest (
    path TEXT PRIMARY KEY,
    broker_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    data_type TEXT NOT NULL,
    date TEXT NOT NULL,
    rows INTEGER NOT NULL DEFAULT 0,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    checksum TEXT
)
"""


class CheckpointDB:
    """SQLite-backed checkpoint and manifest store."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), isolation_level="DEFERRED")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_tables()

    def _init_tables(self) -> None:
        with self._conn:
            self._conn.execute(_CREATE_CHECKPOINTS_SQL)
            self._conn.execute(_CREATE_JOB_LOG_SQL)
            self._conn.execute(_CREATE_MANIFEST_SQL)

    def close(self) -> None:
        self._conn.close()

    # --- Checkpoints ---

    def get_checkpoint(
        self, broker_id: str, symbol: str, data_type: str, timeframe: str = ""
    ) -> IngestionCheckpoint | None:
        row = self._conn.execute(
            "SELECT * FROM checkpoints WHERE broker_id=? AND symbol=? AND data_type=? AND timeframe=?",
            (broker_id, symbol, data_type, timeframe),
        ).fetchone()
        if row is None:
            return None
        return IngestionCheckpoint(
            broker_id=row["broker_id"],
            symbol=row["symbol"],
            data_type=row["data_type"],
            timeframe=row["timeframe"],
            last_timestamp_utc=dt.datetime.fromisoformat(row["last_timestamp_utc"]),
            last_time_msc=row["last_time_msc"],
            rows_ingested=row["rows_ingested"],
            updated_at=dt.datetime.fromisoformat(row["updated_at"]),
            status=row["status"],
        )

    def upsert_checkpoint(self, cp: IngestionCheckpoint) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT INTO checkpoints (broker_id, symbol, data_type, timeframe,
                   last_timestamp_utc, last_time_msc, rows_ingested, updated_at, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(broker_id, symbol, data_type, timeframe)
                   DO UPDATE SET
                       last_timestamp_utc=excluded.last_timestamp_utc,
                       last_time_msc=excluded.last_time_msc,
                       rows_ingested=excluded.rows_ingested,
                       updated_at=excluded.updated_at,
                       status=excluded.status
                """,
                (
                    cp.broker_id,
                    cp.symbol,
                    cp.data_type,
                    cp.timeframe,
                    cp.last_timestamp_utc.isoformat(),
                    cp.last_time_msc,
                    cp.rows_ingested,
                    cp.updated_at.isoformat(),
                    cp.status,
                ),
            )
        log.debug(
            "checkpoint_upsert",
            broker=cp.broker_id,
            symbol=cp.symbol,
            data_type=cp.data_type,
            last_ts=cp.last_timestamp_utc.isoformat(),
        )

    def list_checkpoints(self, broker_id: str | None = None) -> list[IngestionCheckpoint]:
        if broker_id:
            rows = self._conn.execute(
                "SELECT * FROM checkpoints WHERE broker_id=? ORDER BY symbol, data_type",
                (broker_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM checkpoints ORDER BY broker_id, symbol, data_type"
            ).fetchall()
        return [
            IngestionCheckpoint(
                broker_id=r["broker_id"],
                symbol=r["symbol"],
                data_type=r["data_type"],
                timeframe=r["timeframe"],
                last_timestamp_utc=dt.datetime.fromisoformat(r["last_timestamp_utc"]),
                last_time_msc=r["last_time_msc"],
                rows_ingested=r["rows_ingested"],
                updated_at=dt.datetime.fromisoformat(r["updated_at"]),
                status=r["status"],
            )
            for r in rows
        ]

    # --- Job log ---

    def start_job(
        self, job_type: str, broker_id: str, symbol: str, metadata_json: str = ""
    ) -> int:
        with self._conn:
            cursor = self._conn.execute(
                """INSERT INTO job_log (job_type, broker_id, symbol, started_at, status, metadata_json)
                   VALUES (?, ?, ?, ?, 'running', ?)""",
                (job_type, broker_id, symbol, utc_now().isoformat(), metadata_json),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def finish_job(
        self, job_id: int, status: str, rows_processed: int = 0, error_message: str = ""
    ) -> None:
        with self._conn:
            self._conn.execute(
                """UPDATE job_log SET finished_at=?, status=?, rows_processed=?, error_message=?
                   WHERE id=?""",
                (utc_now().isoformat(), status, rows_processed, error_message, job_id),
            )

    # --- File manifest ---

    def register_file(
        self,
        path: str,
        broker_id: str,
        symbol: str,
        data_type: str,
        date: str,
        rows: int,
        size_bytes: int,
        checksum: str = "",
    ) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT OR REPLACE INTO file_manifest
                   (path, broker_id, symbol, data_type, date, rows, size_bytes, created_at, checksum)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (path, broker_id, symbol, data_type, date, rows, size_bytes, utc_now().isoformat(), checksum),
            )

    def get_total_rows(self, broker_id: str | None = None, data_type: str | None = None) -> int:
        conditions: list[str] = []
        params: list[str] = []
        if broker_id:
            conditions.append("broker_id=?")
            params.append(broker_id)
        if data_type:
            conditions.append("data_type=?")
            params.append(data_type)
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        row = self._conn.execute(
            f"SELECT COALESCE(SUM(rows), 0) as total FROM file_manifest WHERE {where_clause}",
            params,
        ).fetchone()
        return row["total"] if row else 0
