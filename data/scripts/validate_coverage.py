"""Validate daily synchronized coverage from persisted merge QA parquet files."""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable


SCRIPT_PATH = Path(__file__).resolve()
DATA_ROOT = SCRIPT_PATH.parents[1]
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

import polars as pl

from mt5pipe.config.loader import load_config
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths


@dataclass(frozen=True, slots=True)
class CoverageDay:
    """Coverage metrics for one UTC date loaded from the persisted merge QA report."""

    date: dt.date
    broker_a_ticks: int
    broker_b_ticks: int
    canonical_rows: int
    dual_source_ratio: float
    gap_count: int

    @property
    def status(self) -> str:
        """Return GREEN/YELLOW/RED based on the persisted dual-source ratio."""

        if self.dual_source_ratio >= 0.15:
            return "GREEN"
        if self.dual_source_ratio >= 0.10:
            return "YELLOW"
        return "RED"


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    """Aggregate counts and date recommendations for a coverage range."""

    rows: list[CoverageDay]
    green_days: int
    yellow_days: int
    red_days: int
    recommended_date_from: dt.date | None
    recommended_date_to: dt.date | None
    estimated_rows_after_compilation: int

    @property
    def total_days(self) -> int:
        """Return the number of requested UTC dates represented in the report."""

        return len(self.rows)

    @property
    def eligible_days(self) -> int:
        """Return the number of GREEN or YELLOW dates."""

        return self.green_days + self.yellow_days


def iter_dates(start_date: dt.date, end_date: dt.date) -> Iterable[dt.date]:
    """Yield every UTC date in an inclusive range."""

    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    current = start_date
    while current <= end_date:
        yield current
        current += dt.timedelta(days=1)


def load_coverage_days(
    paths: StoragePaths,
    store: ParquetStore,
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
) -> list[CoverageDay]:
    """Load one coverage record per UTC date from persisted merge QA partitions."""

    rows: list[CoverageDay] = []
    for date in iter_dates(start_date, end_date):
        frame = store.read_dir(paths.merge_qa_dir(symbol, date))
        if frame.is_empty():
            rows.append(
                CoverageDay(
                    date=date,
                    broker_a_ticks=0,
                    broker_b_ticks=0,
                    canonical_rows=0,
                    dual_source_ratio=0.0,
                    gap_count=0,
                )
            )
            continue

        latest = _latest_row(frame)
        rows.append(
            CoverageDay(
                date=date,
                broker_a_ticks=int(latest.get("broker_a_tick_count", 0) or 0),
                broker_b_ticks=int(latest.get("broker_b_tick_count", 0) or 0),
                canonical_rows=int(latest.get("canonical_rows", 0) or 0),
                dual_source_ratio=float(latest.get("dual_source_ratio", 0.0) or 0.0),
                gap_count=int(
                    latest.get("gap_count", latest.get("gaps_gt_1m", 0)) or 0
                ),
            )
        )
    return rows


def build_coverage_summary(rows: list[CoverageDay]) -> CoverageSummary:
    """Aggregate coverage counts, recommended dates, and eligible row estimate."""

    green_days = sum(1 for row in rows if row.status == "GREEN")
    yellow_days = sum(1 for row in rows if row.status == "YELLOW")
    red_days = sum(1 for row in rows if row.status == "RED")
    eligible_rows = [row for row in rows if row.status in {"GREEN", "YELLOW"}]
    return CoverageSummary(
        rows=rows,
        green_days=green_days,
        yellow_days=yellow_days,
        red_days=red_days,
        recommended_date_from=eligible_rows[0].date if eligible_rows else None,
        recommended_date_to=eligible_rows[-1].date if eligible_rows else None,
        estimated_rows_after_compilation=sum(row.canonical_rows for row in eligible_rows),
    )


def format_coverage_report(summary: CoverageSummary) -> str:
    """Render a human-readable per-day coverage table plus a compact summary."""

    lines = [
        "date         canonical_rows   dual_source_ratio   gap_count   status",
    ]
    for row in summary.rows:
        lines.append(
            f"{row.date.isoformat()} "
            f"{row.canonical_rows:>15,} "
            f"{row.dual_source_ratio:>19.4f} "
            f"{row.gap_count:>11,} "
            f"{row.status}"
        )

    total_days = summary.total_days or 1
    lines.extend(
        [
            "",
            f"Total days: {summary.total_days}",
            f"GREEN: {summary.green_days} ({summary.green_days / total_days:.2%})",
            f"YELLOW: {summary.yellow_days} ({summary.yellow_days / total_days:.2%})",
            f"RED: {summary.red_days} ({summary.red_days / total_days:.2%}) - excluded from dataset",
            "Recommended date_from: "
            + (summary.recommended_date_from.isoformat() if summary.recommended_date_from else "N/A"),
            "Recommended date_to: "
            + (summary.recommended_date_to.isoformat() if summary.recommended_date_to else "N/A"),
            f"Estimated rows after compilation: ~{summary.estimated_rows_after_compilation:,}",
            f"Eligible GREEN/YELLOW days: {summary.eligible_days}",
        ]
    )
    return "\n".join(lines)


def _latest_row(frame: pl.DataFrame) -> dict[str, object]:
    if "time_utc" in frame.columns:
        frame = frame.sort("time_utc")
    elif "date" in frame.columns:
        frame = frame.sort("date")
    return frame.tail(1).row(0, named=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="date_from", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--symbol", required=True, help="Canonical symbol, e.g. XAUUSD")
    parser.add_argument(
        "--config",
        default=str(DATA_ROOT / "config" / "pipeline.yaml"),
        help="Path to pipeline.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config(Path(args.config))
    paths = StoragePaths(cfg.storage.root)
    store = ParquetStore(cfg.storage.compression, cfg.storage.parquet_row_group_size)
    start_date = dt.date.fromisoformat(args.date_from)
    end_date = dt.date.fromisoformat(args.date_to)
    rows = load_coverage_days(paths, store, args.symbol, start_date, end_date)
    summary = build_coverage_summary(rows)
    print(format_coverage_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
