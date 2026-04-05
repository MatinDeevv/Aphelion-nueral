"""Daily merge QA aggregation and synchronized coverage checks."""

from __future__ import annotations

import datetime as dt
import shutil
from dataclasses import dataclass
from typing import Any, Iterable

import polars as pl

from mt5pipe.config.models import MergeConfig
from mt5pipe.merge.canonical import merge_canonical_ticks
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths

UTC = dt.timezone.utc


class CoverageParityError(RuntimeError):
    """Raised when raw tick coverage exists for one broker but not the other."""


class MergeQaIntegrityError(RuntimeError):
    """Raised when stored merge artifacts are internally inconsistent."""


@dataclass(frozen=True)
class RawTickRangeStats:
    broker_id: str
    symbol: str
    days_requested: int
    days_written: int
    total_ticks_written: int
    first_timestamp: dt.datetime | None
    last_timestamp: dt.datetime | None
    covered_dates: tuple[dt.date, ...]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "broker_id": self.broker_id,
            "symbol": self.symbol,
            "days_requested": self.days_requested,
            "days_written": self.days_written,
            "total_ticks_written": self.total_ticks_written,
            "first_timestamp": self.first_timestamp.isoformat() if self.first_timestamp else "",
            "last_timestamp": self.last_timestamp.isoformat() if self.last_timestamp else "",
        }


def iter_utc_dates(start_date: dt.date, end_date: dt.date) -> list[dt.date]:
    """Return all UTC dates in an inclusive range."""
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    dates: list[dt.date] = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += dt.timedelta(days=1)
    return dates


def resolve_utc_day_bounds(
    date: dt.date,
    hours_start_utc: dt.time | None = None,
    hours_end_utc: dt.time | None = None,
) -> tuple[dt.datetime, dt.datetime]:
    """Resolve one UTC date into a [start, end) time window."""
    if (hours_start_utc is None) != (hours_end_utc is None):
        raise ValueError("hours_start_utc and hours_end_utc must both be set or both be omitted")

    day_start = dt.datetime.combine(date, dt.time(0, 0), tzinfo=UTC)
    if hours_start_utc is None:
        return day_start, day_start + dt.timedelta(days=1)

    if hours_start_utc >= hours_end_utc:
        raise ValueError("UTC intraday window must satisfy hours_start_utc < hours_end_utc")

    return (
        dt.datetime.combine(date, hours_start_utc, tzinfo=UTC),
        dt.datetime.combine(date, hours_end_utc, tzinfo=UTC),
    )


def _filter_frame_to_day_window(
    df: pl.DataFrame,
    date: dt.date,
    *,
    time_col: str,
    hours_start_utc: dt.time | None = None,
    hours_end_utc: dt.time | None = None,
) -> pl.DataFrame:
    if df.is_empty() or hours_start_utc is None:
        return df

    window_start, window_end = resolve_utc_day_bounds(date, hours_start_utc, hours_end_utc)
    return df.filter((pl.col(time_col) >= window_start) & (pl.col(time_col) < window_end))


def collect_raw_tick_range_stats(
    paths: StoragePaths,
    store: ParquetStore,
    broker_id: str,
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
    *,
    hours_start_utc: dt.time | None = None,
    hours_end_utc: dt.time | None = None,
) -> RawTickRangeStats:
    """Summarize raw tick coverage for a broker across a UTC date range."""
    dates = iter_utc_dates(start_date, end_date)
    total_ticks = 0
    covered_dates: list[dt.date] = []
    first_timestamp: dt.datetime | None = None
    last_timestamp: dt.datetime | None = None

    for date in dates:
        day_df = store.read_dir(paths.raw_ticks_dir(broker_id, symbol, date))
        day_df = _filter_frame_to_day_window(
            day_df,
            date,
            time_col="time_utc",
            hours_start_utc=hours_start_utc,
            hours_end_utc=hours_end_utc,
        )
        if day_df.is_empty():
            continue

        covered_dates.append(date)
        total_ticks += day_df.height

        day_first = day_df["time_utc"].min()
        day_last = day_df["time_utc"].max()
        if first_timestamp is None or day_first < first_timestamp:
            first_timestamp = day_first
        if last_timestamp is None or day_last > last_timestamp:
            last_timestamp = day_last

    return RawTickRangeStats(
        broker_id=broker_id,
        symbol=symbol,
        days_requested=len(dates),
        days_written=len(covered_dates),
        total_ticks_written=total_ticks,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        covered_dates=tuple(covered_dates),
    )


def assert_synchronized_raw_tick_coverage(
    paths: StoragePaths,
    store: ParquetStore,
    broker_a_id: str,
    broker_b_id: str,
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
    *,
    hours_start_utc: dt.time | None = None,
    hours_end_utc: dt.time | None = None,
) -> tuple[RawTickRangeStats, RawTickRangeStats]:
    """Fail if requested raw tick coverage exists for one broker but not the other."""
    stats_a = collect_raw_tick_range_stats(
        paths,
        store,
        broker_a_id,
        symbol,
        start_date,
        end_date,
        hours_start_utc=hours_start_utc,
        hours_end_utc=hours_end_utc,
    )
    stats_b = collect_raw_tick_range_stats(
        paths,
        store,
        broker_b_id,
        symbol,
        start_date,
        end_date,
        hours_start_utc=hours_start_utc,
        hours_end_utc=hours_end_utc,
    )

    dates_a = set(stats_a.covered_dates)
    dates_b = set(stats_b.covered_dates)
    missing_for_a = sorted(dates_b - dates_a)
    missing_for_b = sorted(dates_a - dates_b)

    if missing_for_a or missing_for_b:
        def _fmt(values: Iterable[dt.date]) -> str:
            return ", ".join(v.isoformat() for v in values)

        parts: list[str] = [
            f"Requested synchronized raw tick coverage is asymmetric for {symbol} "
            f"between {start_date.isoformat()} and {end_date.isoformat()}."
        ]
        if missing_for_a:
            parts.append(f"{broker_a_id} is missing dates present for {broker_b_id}: {_fmt(missing_for_a)}.")
        if missing_for_b:
            parts.append(f"{broker_b_id} is missing dates present for {broker_a_id}: {_fmt(missing_for_b)}.")
        raise CoverageParityError(" ".join(parts))

    return stats_a, stats_b


def _latest_row(df: pl.DataFrame) -> dict[str, Any] | None:
    if df.is_empty():
        return None
    sort_columns = [column for column in ["time_utc", "date"] if column in df.columns]
    if sort_columns:
        df = df.sort(sort_columns)
    return df.tail(1).row(0, named=True)


def _canonical_dual_rows(df: pl.DataFrame) -> int:
    if df.is_empty() or "source_secondary" not in df.columns:
        return 0
    return df.filter(pl.col("source_secondary") != "").height


def _canonical_conflicts(df: pl.DataFrame) -> int:
    if df.is_empty() or "conflict_flag" not in df.columns:
        return 0
    return int(df["conflict_flag"].sum())


def _gap_metrics(df: pl.DataFrame) -> tuple[int, float]:
    if df.is_empty() or df.height < 2 or "ts_msc" not in df.columns:
        return 0, 0.0

    gaps = (
        df.sort("ts_msc")
        .with_columns((pl.col("ts_msc").diff() / 60_000.0).alias("_gap_minutes"))
        .filter(pl.col("_gap_minutes") > 1.0)
    )
    if gaps.is_empty():
        return 0, 0.0

    return gaps.height, round(float(gaps["_gap_minutes"].max()), 6)


def _session_participation(df: pl.DataFrame) -> dict[str, int]:
    if df.is_empty() or "ts_utc" not in df.columns:
        return {
            "asia_rows": 0,
            "london_rows": 0,
            "ny_rows": 0,
            "overlap_rows": 0,
        }

    with_hours = df.with_columns(pl.col("ts_utc").dt.hour().alias("_hour"))
    return {
        "asia_rows": with_hours.filter((pl.col("_hour") >= 0) & (pl.col("_hour") < 8)).height,
        "london_rows": with_hours.filter((pl.col("_hour") >= 7) & (pl.col("_hour") < 16)).height,
        "ny_rows": with_hours.filter((pl.col("_hour") >= 13) & (pl.col("_hour") < 22)).height,
        "overlap_rows": with_hours.filter((pl.col("_hour") >= 13) & (pl.col("_hour") < 16)).height,
    }


def _validate_diagnostic_consistency(
    date: dt.date,
    diag_row: dict[str, Any] | None,
    canonical_rows: int,
    canonical_dual_rows: int,
    dual_source_ratio: float,
    conflicts: int,
) -> None:
    if diag_row is None:
        return

    diag_rows = int(diag_row.get("canonical_rows", 0))
    diag_dual = int(diag_row.get("canonical_dual_rows", 0))
    diag_conflicts = int(diag_row.get("conflicts", 0))
    diag_ratio = float(diag_row.get("dual_source_ratio", 0.0))

    if diag_rows != canonical_rows:
        raise MergeQaIntegrityError(
            f"Canonical row count mismatch for {date.isoformat()}: diagnostics={diag_rows}, actual={canonical_rows}."
        )
    if diag_dual != canonical_dual_rows:
        raise MergeQaIntegrityError(
            f"Canonical dual row mismatch for {date.isoformat()}: diagnostics={diag_dual}, actual={canonical_dual_rows}."
        )
    if abs(diag_ratio - dual_source_ratio) > 1e-8:
        raise MergeQaIntegrityError(
            f"Dual-source ratio mismatch for {date.isoformat()}: diagnostics={diag_ratio}, actual={dual_source_ratio}."
        )
    if diag_conflicts != conflicts:
        raise MergeQaIntegrityError(
            f"Conflict count mismatch for {date.isoformat()}: diagnostics={diag_conflicts}, actual={conflicts}."
        )


def build_daily_merge_qa_report(
    paths: StoragePaths,
    store: ParquetStore,
    broker_a_id: str,
    broker_b_id: str,
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
    *,
    expected_bucket_ms: int | None = None,
) -> pl.DataFrame:
    """Aggregate raw, canonical, and diagnostic data into one row per UTC day."""
    assert_synchronized_raw_tick_coverage(paths, store, broker_a_id, broker_b_id, symbol, start_date, end_date)

    rows: list[dict[str, Any]] = []
    for date in iter_utc_dates(start_date, end_date):
        raw_a = store.read_dir(paths.raw_ticks_dir(broker_a_id, symbol, date))
        raw_b = store.read_dir(paths.raw_ticks_dir(broker_b_id, symbol, date))
        canonical = store.read_dir(paths.canonical_ticks_dir(symbol, date))
        diag_row = _latest_row(store.read_dir(paths.merge_diagnostics_dir(symbol, date)))

        has_materialized_day = not raw_a.is_empty() or not raw_b.is_empty() or not canonical.is_empty()
        if diag_row is None and has_materialized_day:
            raise MergeQaIntegrityError(
                f"Merge diagnostics are missing for {symbol} on {date.isoformat()}."
            )

        canonical_rows = canonical.height
        canonical_dual_rows = _canonical_dual_rows(canonical)
        dual_source_ratio = round(
            (canonical_dual_rows / canonical_rows) if canonical_rows else 0.0,
            8,
        )
        conflicts = _canonical_conflicts(canonical)

        _validate_diagnostic_consistency(
            date,
            diag_row,
            canonical_rows,
            canonical_dual_rows,
            dual_source_ratio,
            conflicts,
        )

        sessions = _session_participation(canonical)
        gaps_gt_1m, max_gap_minutes = _gap_metrics(canonical)

        audit_values = dict(diag_row or {})
        audit_values.pop("time_utc", None)
        audit_values.pop("date", None)
        audit_values.pop("symbol", None)
        audit_values.pop("canonical_rows", None)
        audit_values.pop("canonical_dual_rows", None)
        audit_values.pop("dual_source_ratio", None)
        audit_values.pop("conflicts", None)

        bucket_ms = int(audit_values.get("bucket_ms", expected_bucket_ms or 0))
        row = {
            "time_utc": dt.datetime.combine(date, dt.time(0, 0), tzinfo=UTC),
            "date": date.isoformat(),
            "symbol": symbol,
            "broker_a_id": broker_a_id,
            "broker_b_id": broker_b_id,
            "bucket_ms": bucket_ms,
            "broker_a_tick_count": raw_a.height,
            "broker_b_tick_count": raw_b.height,
            "canonical_rows": canonical_rows,
            "canonical_dual_rows": canonical_dual_rows,
            "dual_source_ratio": dual_source_ratio,
            "conflicts": conflicts,
            **audit_values,
            **sessions,
            "gaps_gt_1m": gaps_gt_1m,
            "max_gap_minutes": max_gap_minutes,
        }
        rows.append(row)

    if not rows:
        return pl.DataFrame()

    return pl.DataFrame(rows).sort("time_utc")


def write_daily_merge_qa_report(
    report_df: pl.DataFrame,
    paths: StoragePaths,
    store: ParquetStore,
    symbol: str,
) -> int:
    """Persist the daily merge QA report into one partition per UTC day."""
    if report_df.is_empty():
        return 0

    total_written = 0
    for date_str in report_df["date"].unique().sort().to_list():
        date = dt.date.fromisoformat(date_str)
        day_df = report_df.filter(pl.col("date") == date_str)
        day_dir = paths.merge_qa_dir(symbol, date)
        if day_dir.exists():
            shutil.rmtree(day_dir)
        total_written += store.write(day_df, paths.merge_qa_file(symbol, date))
    return total_written


def assert_daily_merge_qa_exists(
    paths: StoragePaths,
    store: ParquetStore,
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
) -> None:
    """Ensure the daily QA report already exists for each requested day."""
    missing: list[str] = []
    for date in iter_utc_dates(start_date, end_date):
        if store.read_dir(paths.merge_qa_dir(symbol, date)).is_empty():
            missing.append(date.isoformat())

    if missing:
        raise MergeQaIntegrityError(
            "Daily merge QA report is required before running the bucket sweep. "
            f"Missing report dates for {symbol}: {', '.join(missing)}."
        )


def run_bucket_sweep(
    paths: StoragePaths,
    store: ParquetStore,
    broker_a_id: str,
    broker_b_id: str,
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
    cfg: MergeConfig,
    *,
    broker_a_priority: int = 0,
    broker_b_priority: int = 1,
    bucket_values: Iterable[int] = (50, 75, 100, 125),
) -> pl.DataFrame:
    """Evaluate alternate bucket sizes without mutating persisted merge outputs."""
    assert_daily_merge_qa_exists(paths, store, symbol, start_date, end_date)
    assert_synchronized_raw_tick_coverage(paths, store, broker_a_id, broker_b_id, symbol, start_date, end_date)

    rows: list[dict[str, Any]] = []
    for bucket_ms in sorted(set(int(v) for v in bucket_values)):
        sweep_cfg = cfg.model_copy(update={"bucket_ms": bucket_ms})
        prev_mid: float | None = None
        total_rows = 0
        total_dual_rows = 0
        total_near_miss_pairs = 0
        total_validation_reject_count = 0
        total_conflicts = 0

        current = start_date
        while current <= end_date:
            _result, prev_mid, diag = merge_canonical_ticks(
                broker_a_id,
                broker_b_id,
                symbol,
                current,
                paths,
                store,
                sweep_cfg,
                broker_a_priority,
                broker_b_priority,
                prev_mid,
                write_outputs=False,
            )
            total_rows += int(diag.get("canonical_rows", 0))
            total_dual_rows += int(diag.get("canonical_dual_rows", 0))
            total_near_miss_pairs += int(diag.get("near_miss_pairs", 0))
            total_validation_reject_count += int(diag.get("validation_reject_count", 0))
            total_conflicts += int(diag.get("conflicts", 0))
            current += dt.timedelta(days=1)

        rows.append({
            "bucket_ms": bucket_ms,
            "canonical_rows": total_rows,
            "canonical_dual_rows": total_dual_rows,
            "dual_source_ratio": round((total_dual_rows / total_rows) if total_rows else 0.0, 8),
            "near_miss_pairs": total_near_miss_pairs,
            "validation_reject_count": total_validation_reject_count,
            "conflicts": total_conflicts,
        })

    return pl.DataFrame(rows).sort("bucket_ms")


def format_daily_merge_qa_summary(report_df: pl.DataFrame) -> str:
    """Format the daily merge QA report for terminal output."""
    if report_df.is_empty():
        return "Daily merge QA report: no rows."

    symbol = report_df["symbol"][0]
    start_date = report_df["date"].min()
    end_date = report_df["date"].max()
    total_canonical = int(report_df["canonical_rows"].sum())
    total_dual = int(report_df["canonical_dual_rows"].sum())
    total_near = int(report_df["near_miss_pairs"].sum()) if "near_miss_pairs" in report_df.columns else 0
    total_rejects = int(report_df["validation_reject_count"].sum()) if "validation_reject_count" in report_df.columns else 0
    total_conflicts = int(report_df["conflicts"].sum()) if "conflicts" in report_df.columns else 0
    total_gap_events = int(report_df["gaps_gt_1m"].sum()) if "gaps_gt_1m" in report_df.columns else 0
    dual_ratio = (total_dual / total_canonical) if total_canonical else 0.0
    max_gap = float(report_df["max_gap_minutes"].max()) if "max_gap_minutes" in report_df.columns else 0.0

    lines = [
        f"Daily merge QA summary for {symbol} {start_date} -> {end_date}",
        (
            f"days={report_df.height} canonical_rows={total_canonical:,} "
            f"canonical_dual_rows={total_dual:,} dual_source_ratio={dual_ratio:.4f}"
        ),
        (
            f"near_miss_pairs={total_near:,} validation_reject_count={total_rejects:,} "
            f"conflicts={total_conflicts:,} gaps_gt_1m={total_gap_events:,} max_gap_minutes={max_gap:.2f}"
        ),
        "",
        "date         a_ticks    b_ticks    canonical    dual_rows   dual%   bucket_both   near_miss   rejects   gaps>1m   max_gap_m",
    ]

    for row in report_df.iter_rows(named=True):
        dual_pct = float(row.get("dual_source_ratio", 0.0)) * 100.0
        lines.append(
            f"{row['date']} "
            f"{int(row.get('broker_a_tick_count', 0)):>9,} "
            f"{int(row.get('broker_b_tick_count', 0)):>10,} "
            f"{int(row.get('canonical_rows', 0)):>12,} "
            f"{int(row.get('canonical_dual_rows', 0)):>11,} "
            f"{dual_pct:>7.2f} "
            f"{int(row.get('bucket_both', 0)):>13,} "
            f"{int(row.get('near_miss_pairs', 0)):>11,} "
            f"{int(row.get('validation_reject_count', 0)):>9,} "
            f"{int(row.get('gaps_gt_1m', 0)):>9,} "
            f"{float(row.get('max_gap_minutes', 0.0)):>10.2f}"
        )

    return "\n".join(lines)


def format_bucket_sweep_summary(report_df: pl.DataFrame) -> str:
    """Format bucket sweep metrics for terminal output."""
    if report_df.is_empty():
        return "Bucket sweep: no rows."

    lines = [
        "Bucket sweep summary",
        "bucket_ms   canonical_rows   dual_rows   dual%   near_miss   rejects   conflicts",
    ]
    for row in report_df.iter_rows(named=True):
        lines.append(
            f"{int(row['bucket_ms']):>9} "
            f"{int(row.get('canonical_rows', 0)):>16,} "
            f"{int(row.get('canonical_dual_rows', 0)):>11,} "
            f"{float(row.get('dual_source_ratio', 0.0)) * 100.0:>7.2f} "
            f"{int(row.get('near_miss_pairs', 0)):>11,} "
            f"{int(row.get('validation_reject_count', 0)):>9,} "
            f"{int(row.get('conflicts', 0)):>10,}"
        )

    return "\n".join(lines)
