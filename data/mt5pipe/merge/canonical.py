"""Canonical dual-broker tick merger with quality scoring and conflict detection."""

from __future__ import annotations

import datetime as dt
import math
import shutil
from dataclasses import dataclass

import polars as pl

from mt5pipe.config.models import MergeConfig
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths
from mt5pipe.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class _BrokerQuote:
    """A single broker's quote at a point in time."""
    broker_id: str
    ts_msc: int
    bid: float
    ask: float
    last: float
    volume: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    def is_valid(self, max_spread_ratio: float) -> bool:
        if self.bid <= 0 or self.ask <= 0:
            return False
        if self.bid > self.ask:
            return False
        mid = self.mid
        if mid > 0 and (self.spread / mid) > max_spread_ratio:
            return False
        return True


def merge_canonical_ticks(
    broker_a_id: str,
    broker_b_id: str,
    symbol: str,
    date: dt.date,
    paths: StoragePaths,
    store: ParquetStore,
    cfg: MergeConfig,
    broker_a_priority: int = 0,
    broker_b_priority: int = 1,
    prev_canonical_mid: float | None = None,
    write_outputs: bool = True,
) -> tuple[pl.DataFrame, float | None, dict]:
    """Merge two broker raw tick streams into a canonical feed for a single date.
    
    Returns (canonical_df, last_canonical_mid) for continuity across dates.
    """
    # Load raw ticks
    dir_a = paths.raw_ticks_dir(broker_a_id, symbol, date)
    dir_b = paths.raw_ticks_dir(broker_b_id, symbol, date)

    df_a = store.read_dir(dir_a)
    df_b = store.read_dir(dir_b)

    if df_a.is_empty() and df_b.is_empty():
        empty_diag = {
            "date": date.isoformat(),
            "symbol": symbol,
            "bucket_ms": cfg.bucket_ms,
            "bucket_a_only": 0,
            "bucket_b_only": 0,
            "bucket_both": 0,
            "bucket_both_valid": 0,
            "bucket_both_downgraded_to_single": 0,
            "bucket_both_rejected": 0,
            "bucket_invalid_a": 0,
            "bucket_invalid_b": 0,
            "wallclock_overlap_minutes": 0,
            "median_offset_both_ms": 0.0,
            "p95_offset_both_ms": 0.0,
            "near_miss_pairs": 0,
            "validation_reject_count": 0,
            "canonical_rows": 0,
            "canonical_dual_rows": 0,
            "dual_source_ratio": 0.0,
            "conflicts": 0,
        }
        return pl.DataFrame(), prev_canonical_mid, empty_diag

    # Normalize columns
    for label, df in [("a", df_a), ("b", df_b)]:
        if not df.is_empty() and "time_msc" not in df.columns:
            log.warning("merge_missing_time_msc", broker=label)

    # Assign merge buckets
    bucket_ms = cfg.bucket_ms

    def _add_bucket(df: pl.DataFrame, broker_label: str) -> pl.DataFrame:
        if df.is_empty():
            return df
        return df.with_columns([
            (pl.col("time_msc") // bucket_ms * bucket_ms).alias("bucket_msc"),
            pl.lit(broker_label).alias("_broker_label"),
        ])

    df_a = _add_bucket(df_a, "a")
    df_b = _add_bucket(df_b, "b")

    # Get unique buckets from both
    all_buckets: set[int] = set()
    if not df_a.is_empty():
        all_buckets.update(df_a["bucket_msc"].unique().to_list())
    if not df_b.is_empty():
        all_buckets.update(df_b["bucket_msc"].unique().to_list())

    if not all_buckets:
        empty_diag = {
            "date": date.isoformat(),
            "symbol": symbol,
            "bucket_ms": cfg.bucket_ms,
            "bucket_a_only": 0,
            "bucket_b_only": 0,
            "bucket_both": 0,
            "bucket_both_valid": 0,
            "bucket_both_downgraded_to_single": 0,
            "bucket_both_rejected": 0,
            "bucket_invalid_a": 0,
            "bucket_invalid_b": 0,
            "wallclock_overlap_minutes": 0,
            "median_offset_both_ms": 0.0,
            "p95_offset_both_ms": 0.0,
            "near_miss_pairs": 0,
            "validation_reject_count": 0,
            "canonical_rows": 0,
            "canonical_dual_rows": 0,
            "dual_source_ratio": 0.0,
            "conflicts": 0,
        }
        return pl.DataFrame(), prev_canonical_mid, empty_diag

    sorted_buckets = sorted(all_buckets)

    # Build index for quick bucket lookup
    a_by_bucket: dict[int, pl.DataFrame] = {}
    b_by_bucket: dict[int, pl.DataFrame] = {}

    if not df_a.is_empty():
        for bucket_val in df_a["bucket_msc"].unique().to_list():
            a_by_bucket[bucket_val] = df_a.filter(pl.col("bucket_msc") == bucket_val)

    if not df_b.is_empty():
        for bucket_val in df_b["bucket_msc"].unique().to_list():
            b_by_bucket[bucket_val] = df_b.filter(pl.col("bucket_msc") == bucket_val)

    canonical_rows: list[dict] = []
    last_mid = prev_canonical_mid
    bucket_a_only = 0
    bucket_b_only = 0
    bucket_both = 0
    bucket_both_valid = 0
    bucket_both_downgraded_to_single = 0
    bucket_both_rejected = 0
    bucket_invalid_a = 0
    bucket_invalid_b = 0
    both_offsets_ms: list[int] = []

    for bucket in sorted_buckets:
        chunk_a = a_by_bucket.get(bucket)
        chunk_b = b_by_bucket.get(bucket)

        has_a = chunk_a is not None and not chunk_a.is_empty()
        has_b = chunk_b is not None and not chunk_b.is_empty()

        q_a = _best_quote_in_bucket(chunk_a, broker_a_id) if has_a else None
        q_b = _best_quote_in_bucket(chunk_b, broker_b_id) if has_b else None

        valid_a = q_a is not None and q_a.is_valid(cfg.max_spread_ratio)
        valid_b = q_b is not None and q_b.is_valid(cfg.max_spread_ratio)

        if has_a and not has_b:
            bucket_a_only += 1
        elif has_b and not has_a:
            bucket_b_only += 1
        elif has_a and has_b:
            bucket_both += 1
            if q_a is not None and q_b is not None:
                both_offsets_ms.append(abs(q_a.ts_msc - q_b.ts_msc))
            if valid_a and valid_b:
                bucket_both_valid += 1
            elif valid_a or valid_b:
                bucket_both_downgraded_to_single += 1
            else:
                bucket_both_rejected += 1

        if has_a and not valid_a:
            bucket_invalid_a += 1
        if has_b and not valid_b:
            bucket_invalid_b += 1

        if not valid_a and not valid_b:
            continue

        row = _resolve_bucket(
            q_a if valid_a else None,
            q_b if valid_b else None,
            broker_a_id,
            broker_b_id,
            broker_a_priority,
            broker_b_priority,
            cfg,
            last_mid,
            bucket,
            symbol,
        )
        if row is not None:
            canonical_rows.append(row)
            last_mid = (row["bid"] + row["ask"]) / 2.0

    near_miss_pairs = _estimate_near_miss_pairs(
        df_a,
        df_b,
        cfg.bucket_ms,
        cfg.diagnostics_near_miss_factor,
    )
    wallclock_overlap_minutes = _count_wallclock_overlap_minutes(df_a, df_b)
    validation_reject_count = bucket_invalid_a + bucket_invalid_b
    median_offset_both_ms = float(_median_or_zero(both_offsets_ms))
    p95_offset_both_ms = float(_percentile_or_zero(both_offsets_ms, 0.95))

    if not canonical_rows:
        diag = {
            "date": date.isoformat(),
            "symbol": symbol,
            "bucket_ms": cfg.bucket_ms,
            "bucket_a_only": bucket_a_only,
            "bucket_b_only": bucket_b_only,
            "bucket_both": bucket_both,
            "bucket_both_valid": bucket_both_valid,
            "bucket_both_downgraded_to_single": bucket_both_downgraded_to_single,
            "bucket_both_rejected": bucket_both_rejected,
            "bucket_invalid_a": bucket_invalid_a,
            "bucket_invalid_b": bucket_invalid_b,
            "wallclock_overlap_minutes": wallclock_overlap_minutes,
            "median_offset_both_ms": median_offset_both_ms,
            "p95_offset_both_ms": p95_offset_both_ms,
            "near_miss_pairs": near_miss_pairs,
            "validation_reject_count": validation_reject_count,
            "canonical_rows": 0,
            "canonical_dual_rows": 0,
            "dual_source_ratio": 0.0,
            "conflicts": 0,
        }
        if write_outputs:
            _write_day_diagnostics(paths, store, symbol, date, diag)
        return pl.DataFrame(), last_mid, diag

    result = pl.DataFrame(canonical_rows)

    # Sort by timestamp
    result = result.sort("ts_msc")

    if write_outputs:
        # Write to canonical partition
        path = paths.canonical_ticks_file(symbol, date)
        store.write(result, path)

    dual_rows = result.filter(pl.col("source_secondary") != "").height if "source_secondary" in result.columns else 0
    conflicts = int(result["conflict_flag"].sum()) if "conflict_flag" in result.columns else 0
    dual_ratio = (dual_rows / len(result)) if len(result) else 0.0
    diag = {
        "date": date.isoformat(),
        "symbol": symbol,
        "bucket_ms": cfg.bucket_ms,
        "bucket_a_only": bucket_a_only,
        "bucket_b_only": bucket_b_only,
        "bucket_both": bucket_both,
        "bucket_both_valid": bucket_both_valid,
        "bucket_both_downgraded_to_single": bucket_both_downgraded_to_single,
        "bucket_both_rejected": bucket_both_rejected,
        "bucket_invalid_a": bucket_invalid_a,
        "bucket_invalid_b": bucket_invalid_b,
        "wallclock_overlap_minutes": wallclock_overlap_minutes,
        "median_offset_both_ms": median_offset_both_ms,
        "p95_offset_both_ms": p95_offset_both_ms,
        "near_miss_pairs": near_miss_pairs,
        "validation_reject_count": validation_reject_count,
        "canonical_rows": len(result),
        "canonical_dual_rows": dual_rows,
        "dual_source_ratio": round(dual_ratio, 8),
        "conflicts": conflicts,
    }
    if write_outputs:
        _write_day_diagnostics(paths, store, symbol, date, diag)

    log.info(
        "canonical_merge_done",
        symbol=symbol,
        date=date.isoformat(),
        rows=len(result),
        conflicts=conflicts,
        bucket_both=bucket_both,
        both_valid=bucket_both_valid,
        both_downgraded_to_single=bucket_both_downgraded_to_single,
        both_rejected=bucket_both_rejected,
        wallclock_overlap_minutes=wallclock_overlap_minutes,
        median_offset_both_ms=diag["median_offset_both_ms"],
        p95_offset_both_ms=diag["p95_offset_both_ms"],
        validation_reject_count=validation_reject_count,
        near_miss_pairs=near_miss_pairs,
        dual_source_ratio=f"{dual_ratio:.6f}",
    )

    return result, last_mid, diag


def _median_or_zero(values: list[int]) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    n = len(v)
    m = n // 2
    if n % 2:
        return float(v[m])
    return float((v[m - 1] + v[m]) / 2.0)


def _percentile_or_zero(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    return float(pl.Series("v", values).quantile(q))


def _count_wallclock_overlap_minutes(df_a: pl.DataFrame, df_b: pl.DataFrame) -> int:
    """Count wall-clock minutes where both brokers have at least one raw tick."""
    if df_a.is_empty() or df_b.is_empty() or "time_msc" not in df_a.columns or "time_msc" not in df_b.columns:
        return 0
    a_minutes = set((df_a["time_msc"] // 60_000).to_list())
    b_minutes = set((df_b["time_msc"] // 60_000).to_list())
    return len(a_minutes.intersection(b_minutes))


def _estimate_near_miss_pairs(
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
    bucket_ms: int,
    factor: int,
) -> int:
    """Estimate how often streams are close but outside the merge bucket.

    A near miss means nearest timestamps are within factor*bucket_ms but
    farther apart than bucket_ms.
    """
    if df_a.is_empty() or df_b.is_empty() or "time_msc" not in df_a.columns or "time_msc" not in df_b.columns:
        return 0

    try:
        a = df_a.select(pl.col("time_msc")).sort("time_msc")
        b = df_b.select(pl.col("time_msc").alias("time_msc_b")).sort("time_msc_b")
        joined = a.join_asof(b, left_on="time_msc", right_on="time_msc_b", strategy="nearest")
        if joined.is_empty():
            return 0
        max_tol = max(bucket_ms * max(factor, 2), bucket_ms + 1)
        near = joined.with_columns(
            (pl.col("time_msc") - pl.col("time_msc_b")).abs().alias("_offset")
        ).filter(
            (pl.col("_offset") > bucket_ms) & (pl.col("_offset") <= max_tol)
        )
        return len(near)
    except Exception:
        return 0


def _write_day_diagnostics(
    paths: StoragePaths,
    store: ParquetStore,
    symbol: str,
    date: dt.date,
    diag: dict,
) -> None:
    diag_df = pl.DataFrame([
        {
            "time_utc": dt.datetime.combine(date, dt.time(0, 0), tzinfo=dt.timezone.utc),
            **diag,
        }
    ])
    diagnostics_dir = paths.merge_diagnostics_dir(symbol, date)
    if diagnostics_dir.exists():
        shutil.rmtree(diagnostics_dir)
    store.write(diag_df, paths.merge_diagnostics_file(symbol, date))


def _best_quote_in_bucket(df: pl.DataFrame | None, broker_id: str) -> _BrokerQuote | None:
    """Pick the last (freshest) tick in a bucket."""
    if df is None or df.is_empty():
        return None
    last = df.sort("time_msc").tail(1)
    return _BrokerQuote(
        broker_id=broker_id,
        ts_msc=last["time_msc"][0],
        bid=last["bid"][0],
        ask=last["ask"][0],
        last=last["last"][0] if "last" in last.columns else 0.0,
        volume=last["volume"][0] if "volume" in last.columns else 0.0,
    )


def _score_quote(
    quote: _BrokerQuote,
    other: _BrokerQuote | None,
    prev_mid: float | None,
    cfg: MergeConfig,
    bucket_msc: int,
) -> float:
    """Score a broker quote on freshness, spread, and continuity."""
    # Freshness: how close to the bucket end
    freshness = 1.0 - min(abs(quote.ts_msc - bucket_msc) / max(cfg.bucket_ms, 1), 1.0)

    # Spread: lower is better
    if quote.mid > 0:
        spread_ratio = quote.spread / quote.mid
        spread_score = max(0.0, 1.0 - spread_ratio / cfg.max_spread_ratio)
    else:
        spread_score = 0.0

    # Continuity: how close to previous canonical mid
    if prev_mid is not None and prev_mid > 0:
        diff_ratio = abs(quote.mid - prev_mid) / prev_mid
        continuity = max(0.0, 1.0 - diff_ratio / cfg.max_mid_diff_ratio)
    else:
        continuity = 0.5  # Neutral if no history

    return (
        cfg.freshness_weight * freshness
        + cfg.spread_weight * spread_score
        + cfg.continuity_weight * continuity
    )


def _resolve_bucket(
    q_a: _BrokerQuote | None,
    q_b: _BrokerQuote | None,
    broker_a_id: str,
    broker_b_id: str,
    priority_a: int,
    priority_b: int,
    cfg: MergeConfig,
    prev_mid: float | None,
    bucket_msc: int,
    symbol: str,
) -> dict | None:
    """Resolve a single merge bucket into a canonical tick."""
    # Single source
    if q_a is not None and q_b is None:
        return _make_canonical_row(
            chosen=q_a, other=None,
            broker_a_id=broker_a_id, broker_b_id=broker_b_id,
            mode="single", bucket_msc=bucket_msc, symbol=symbol,
            quality=_score_quote(q_a, None, prev_mid, cfg, bucket_msc),
            conflict=False,
        )

    if q_b is not None and q_a is None:
        return _make_canonical_row(
            chosen=q_b, other=None,
            broker_a_id=broker_a_id, broker_b_id=broker_b_id,
            mode="single", bucket_msc=bucket_msc, symbol=symbol,
            quality=_score_quote(q_b, None, prev_mid, cfg, bucket_msc),
            conflict=False,
        )

    # Both None — no data in this bucket
    if q_a is None and q_b is None:
        return None

    assert q_a is not None and q_b is not None

    # Both available — score and check for conflict
    score_a = _score_quote(q_a, q_b, prev_mid, cfg, bucket_msc)
    score_b = _score_quote(q_b, q_a, prev_mid, cfg, bucket_msc)

    mid_a = q_a.mid
    mid_b = q_b.mid
    avg_mid = (mid_a + mid_b) / 2.0
    mid_diff = abs(mid_a - mid_b)
    mid_diff_ratio = mid_diff / avg_mid if avg_mid > 0 else 0.0

    conflict = mid_diff_ratio > cfg.conflict_log_threshold
    spread_diff = abs(q_a.spread - q_b.spread)

    # Choose best
    if score_a > score_b:
        chosen, other = q_a, q_b
    elif score_b > score_a:
        chosen, other = q_b, q_a
    else:
        # Tie-break by priority
        if priority_a <= priority_b:
            chosen, other = q_a, q_b
        else:
            chosen, other = q_b, q_a

    mode = "best"
    if conflict:
        mode = "conflict"
        log.debug(
            "merge_conflict",
            symbol=symbol,
            bucket=bucket_msc,
            mid_diff=mid_diff,
            ratio=mid_diff_ratio,
        )

    return _make_canonical_row(
        chosen=chosen, other=other,
        broker_a_id=broker_a_id, broker_b_id=broker_b_id,
        mode=mode, bucket_msc=bucket_msc, symbol=symbol,
        quality=max(score_a, score_b),
        conflict=conflict,
        q_a=q_a, q_b=q_b,
        mid_diff=mid_diff, spread_diff=spread_diff,
    )


def _make_canonical_row(
    chosen: _BrokerQuote,
    other: _BrokerQuote | None,
    broker_a_id: str,
    broker_b_id: str,
    mode: str,
    bucket_msc: int,
    symbol: str,
    quality: float,
    conflict: bool,
    q_a: _BrokerQuote | None = None,
    q_b: _BrokerQuote | None = None,
    mid_diff: float = 0.0,
    spread_diff: float = 0.0,
) -> dict:
    ts_utc = dt.datetime.fromtimestamp(bucket_msc / 1000.0, tz=dt.timezone.utc)

    return {
        "ts_utc": ts_utc,
        "ts_msc": bucket_msc,
        "symbol": symbol,
        "bid": chosen.bid,
        "ask": chosen.ask,
        "last": chosen.last,
        "volume": chosen.volume,
        "source_primary": chosen.broker_id,
        "source_secondary": other.broker_id if other else "",
        "merge_mode": mode,
        "quality_score": round(quality, 6),
        "conflict_flag": conflict,
        "broker_a_bid": q_a.bid if q_a else 0.0,
        "broker_a_ask": q_a.ask if q_a else 0.0,
        "broker_b_bid": q_b.bid if q_b else 0.0,
        "broker_b_ask": q_b.ask if q_b else 0.0,
        "mid_diff": round(mid_diff, 6),
        "spread_diff": round(spread_diff, 6),
    }


def merge_canonical_date_range(
    broker_a_id: str,
    broker_b_id: str,
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
    paths: StoragePaths,
    store: ParquetStore,
    cfg: MergeConfig,
    broker_a_priority: int = 0,
    broker_b_priority: int = 1,
    write_outputs: bool = True,
) -> int:
    """Merge canonical ticks for a date range. Returns total rows."""
    total = 0
    prev_mid: float | None = None
    current = start_date
    total_days = max((end_date - start_date).days, 1)
    days_done = 0
    range_diags: list[dict] = []

    while current <= end_date:
        df, prev_mid, day_diag = merge_canonical_ticks(
            broker_a_id, broker_b_id, symbol, current,
            paths, store, cfg,
            broker_a_priority, broker_b_priority, prev_mid,
            write_outputs=write_outputs,
        )
        total += len(df)
        range_diags.append(day_diag)
        days_done += 1
        current += dt.timedelta(days=1)

        # Progress log every 30 days
        if days_done % 30 == 0:
            pct = min(100.0, days_done / total_days * 100)
            log.info(
                "canonical_merge_progress",
                symbol=symbol,
                days_done=days_done,
                total_days=total_days,
                progress=f"{pct:.1f}%",
                rows_so_far=total,
            )

    log.info(
        "canonical_merge_range_done",
        symbol=symbol,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        total=total,
    )

    if range_diags:
        total_rows = sum(int(d.get("canonical_rows", 0)) for d in range_diags)
        total_dual = sum(int(d.get("canonical_dual_rows", 0)) for d in range_diags)
        total_both = sum(int(d.get("bucket_both", 0)) for d in range_diags)
        total_downgraded = sum(int(d.get("bucket_both_downgraded_to_single", 0)) for d in range_diags)
        total_rejected = sum(int(d.get("bucket_both_rejected", 0)) for d in range_diags)
        total_near_miss = sum(int(d.get("near_miss_pairs", 0)) for d in range_diags)
        total_overlap_minutes = sum(int(d.get("wallclock_overlap_minutes", 0)) for d in range_diags)
        total_validation_reject = sum(int(d.get("validation_reject_count", 0)) for d in range_diags)
        p95_offsets = [float(d.get("p95_offset_both_ms", 0.0)) for d in range_diags]
        median_offsets = [float(d.get("median_offset_both_ms", 0.0)) for d in range_diags]
        dual_ratio = (total_dual / total_rows) if total_rows else 0.0
        mean_p95_offset = sum(p95_offsets) / len(p95_offsets) if p95_offsets else 0.0
        mean_median_offset = sum(median_offsets) / len(median_offsets) if median_offsets else 0.0

        log.info(
            "canonical_merge_diagnostics_summary",
            symbol=symbol,
            rows=total_rows,
            dual_rows=total_dual,
            dual_ratio=f"{dual_ratio:.6f}",
            bucket_both=total_both,
            both_downgraded_to_single=total_downgraded,
            both_rejected=total_rejected,
            wallclock_overlap_minutes=total_overlap_minutes,
            median_offset_both_ms=f"{mean_median_offset:.2f}",
            p95_offset_both_ms=f"{mean_p95_offset:.2f}",
            validation_reject_count=total_validation_reject,
            near_miss_pairs=total_near_miss,
            bucket_ms=cfg.bucket_ms,
        )

        # Hard assertion for bucket starvation: if both brokers clearly overlap in
        # wall-clock minutes but same-bucket co-presence is near zero.
        if total_overlap_minutes > 0 and total_both <= 1:
            raise RuntimeError(
                "Merge bucket starvation detected: both brokers have raw ticks in overlapping wall-clock minutes, "
                "but almost no same-bucket co-presence. "
                f"bucket_ms={cfg.bucket_ms}, median_offset_both_ms={mean_median_offset:.2f}, "
                f"p95_offset_both_ms={mean_p95_offset:.2f}, validation_reject_count={total_validation_reject}, "
                f"bucket_both={total_both}, wallclock_overlap_minutes={total_overlap_minutes}."
            )

        if cfg.hard_fail_on_low_dual_source and dual_ratio < cfg.min_dual_source_ratio:
            raise RuntimeError(
                "Dual-source participation below threshold: "
                f"dual_ratio={dual_ratio:.6f} < min_dual_source_ratio={cfg.min_dual_source_ratio:.6f}. "
                "Check timestamp alignment, bucket_ms, and per-day merge diagnostics under your storage root /merge_qa/ and /merge_diagnostics/."
            )
    return total
