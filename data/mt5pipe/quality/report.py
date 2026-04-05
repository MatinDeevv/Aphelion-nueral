"""Dataset quality report — comprehensive stats printed after build."""

from __future__ import annotations

import polars as pl

from mt5pipe.utils.logging import get_logger

log = get_logger(__name__)


def dataset_quality_report(df: pl.DataFrame, *, time_col: str = "time_utc") -> dict:
    """Generate a comprehensive quality report for a built dataset.

    Returns a dict with all metrics; also logs a human-readable summary.
    """
    n = len(df)
    n_cols = len(df.columns)
    report: dict = {
        "rows": n,
        "columns": n_cols,
    }

    if df.is_empty():
        return report

    # --- Null analysis ---
    null_counts = {}
    for col in df.columns:
        nc = df[col].null_count()
        if nc > 0:
            null_counts[col] = {"count": nc, "pct": round(nc / n * 100, 2)}
    report["null_columns"] = null_counts
    report["total_nulls"] = sum(v["count"] for v in null_counts.values())
    report["null_free"] = len(null_counts) == 0

    # --- Numeric stats ---
    numeric_cols = [c for c in df.columns if df[c].dtype in (pl.Float64, pl.Float32)]
    inf_cols = {}
    for col in numeric_cols:
        inf_count = df.filter(pl.col(col).is_infinite()).height
        if inf_count > 0:
            inf_cols[col] = inf_count
    report["inf_columns"] = inf_cols
    report["inf_free"] = len(inf_cols) == 0

    # --- Duplicate check ---
    if time_col in df.columns:
        n_unique_times = df[time_col].n_unique()
        report["duplicate_timestamps"] = n - n_unique_times
        report["time_range_start"] = str(df[time_col].min())
        report["time_range_end"] = str(df[time_col].max())
    else:
        report["duplicate_timestamps"] = 0

    # --- Constant columns (zero information) ---
    constant_cols = []
    for col in df.columns:
        if col in (time_col, "symbol", "timeframe"):
            continue
        if df[col].n_unique() <= 1:
            constant_cols.append(col)
    report["constant_columns"] = constant_cols

    # --- Feature distributions (key columns) ---
    key_cols = ["close", "mid_return", "realized_vol", "spread_mean", "relative_spread"]
    distributions = {}
    for col in key_cols:
        if col in df.columns and df[col].dtype in (pl.Float64, pl.Float32):
            s = df[col].drop_nulls()
            if len(s) > 0:
                distributions[col] = {
                    "mean": round(float(s.mean()), 8),
                    "std": round(float(s.std()), 8),
                    "min": round(float(s.min()), 8),
                    "max": round(float(s.max()), 8),
                    "median": round(float(s.median()), 8),
                    "q01": round(float(s.quantile(0.01)), 8),
                    "q99": round(float(s.quantile(0.99)), 8),
                }
    report["distributions"] = distributions

    # --- Label balance ---
    label_cols = [c for c in df.columns if c.startswith("direction_")]
    label_balance = {}
    for col in label_cols:
        s = df[col].drop_nulls()
        if len(s) > 0:
            vc = s.value_counts().sort("count", descending=True)
            label_balance[col] = {
                str(row[col]): row["count"]
                for row in vc.iter_rows(named=True)
            }
    report["label_balance"] = label_balance

    # --- Gap check (M1 bars should be roughly 1-min apart) ---
    if time_col in df.columns and len(df) > 1:
        sorted_df = df.sort(time_col)
        diffs = sorted_df.with_columns(
            pl.col(time_col).diff().dt.total_seconds().alias("_dt_seconds")
        ).filter(pl.col("_dt_seconds").is_not_null())

        if not diffs.is_empty():
            dt_col = diffs["_dt_seconds"]
            report["time_delta"] = {
                "mean_seconds": round(float(dt_col.mean()), 2),
                "median_seconds": round(float(dt_col.median()), 2),
                "min_seconds": round(float(dt_col.min()), 2),
                "max_seconds": round(float(dt_col.max()), 2),
                "std_seconds": round(float(dt_col.std()), 2),
            }

    # --- Filled bar ratio ---
    if "_filled" in df.columns:
        filled_count = df.filter(pl.col("_filled")).height
        report["filled_bars"] = filled_count
        report["filled_pct"] = round(filled_count / n * 100, 2)

    # --- Broker merge audit ---
    merge_audit: dict = {}
    if "source_count" in df.columns:
        sc = df["source_count"]
        multi = df.filter(pl.col("source_count") >= 2).height
        merge_audit["multi_source_bars"] = multi
        merge_audit["multi_source_pct"] = round(multi / n * 100, 2)
        merge_audit["mean_source_count"] = round(float(sc.mean()), 3)
    if "dual_source_ratio" in df.columns:
        dsr = df["dual_source_ratio"].drop_nulls()
        if len(dsr) > 0:
            merge_audit["mean_dual_source_ratio"] = round(float(dsr.mean()), 4)
    if "conflict_count" in df.columns:
        cc = df["conflict_count"]
        bars_with_conflicts = df.filter(pl.col("conflict_count") > 0).height
        merge_audit["bars_with_conflicts"] = bars_with_conflicts
        merge_audit["total_conflicts"] = int(cc.sum())
    if merge_audit:
        report["merge_audit"] = merge_audit

    # --- Overall score ---
    score = 100.0
    if report.get("total_nulls", 0) > 0:
        null_pct = report["total_nulls"] / (n * n_cols) * 100
        score -= min(null_pct * 5, 30)  # up to -30 for nulls
    if not report.get("inf_free", True):
        score -= 20
    if report.get("duplicate_timestamps", 0) > 0:
        score -= 10
    if len(constant_cols) > 0:
        score -= min(len(constant_cols) * 2, 10)
    report["quality_score"] = round(max(score, 0), 1)

    # Log summary
    log.info(
        "dataset_quality_report",
        rows=n,
        cols=n_cols,
        null_free=report["null_free"],
        inf_free=report["inf_free"],
        dup_timestamps=report["duplicate_timestamps"],
        constant_cols=len(constant_cols),
        quality_score=report["quality_score"],
    )

    return report


def format_quality_report(report: dict) -> str:
    """Format quality report as a human-readable string."""
    lines = []
    lines.append(f"Dataset Quality Report")
    lines.append(f"=" * 50)
    lines.append(f"Rows: {report.get('rows', 0):,}")
    lines.append(f"Columns: {report.get('columns', 0)}")
    lines.append(f"Quality Score: {report.get('quality_score', 0)}/100")
    lines.append("")

    if report.get("time_range_start"):
        lines.append(f"Time Range: {report['time_range_start']} -> {report['time_range_end']}")

    lines.append(f"Null-Free: {'yes' if report.get('null_free') else 'no'}")
    lines.append(f"Inf-Free: {'yes' if report.get('inf_free') else 'no'}")
    lines.append(f"Duplicate Timestamps: {report.get('duplicate_timestamps', 0)}")

    if report.get("filled_bars"):
        lines.append(f"Filled Bars: {report['filled_bars']} ({report.get('filled_pct', 0):.1f}%)")

    if report.get("constant_columns"):
        lines.append(f"Constant Columns (removed): {', '.join(report['constant_columns'])}")

    if report.get("null_columns"):
        lines.append("")
        lines.append("Columns with Nulls:")
        for col, info in list(report["null_columns"].items())[:10]:
            lines.append(f"  {col}: {info['count']:,} ({info['pct']:.1f}%)")

    if report.get("distributions"):
        lines.append("")
        lines.append("Key Distributions:")
        for col, stats in report["distributions"].items():
            lines.append(f"  {col}: mean={stats['mean']:.6f} std={stats['std']:.6f} "
                          f"[{stats['q01']:.6f}, {stats['q99']:.6f}]")

    if report.get("label_balance"):
        lines.append("")
        lines.append("Label Balance:")
        for col, counts in report["label_balance"].items():
            total = sum(counts.values())
            parts = [f"{k}:{v} ({v / total * 100:.0f}%)" for k, v in counts.items()]
            lines.append(f"  {col}: {' | '.join(parts)}")

    return "\n".join(lines)
