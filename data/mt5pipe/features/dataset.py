"""Dataset assembly utilities and legacy compiler compatibility."""

from __future__ import annotations

import datetime as dt
from importlib import import_module
import shutil
import tempfile
from pathlib import Path
from typing import Any

import polars as pl

from mt5pipe.config.models import DatasetConfig
from mt5pipe.features.context import add_lagged_bar_features
from mt5pipe.features.labels import (
    add_direction_labels,
    add_future_returns,
    add_triple_barrier_labels,
)
from mt5pipe.features.quality import add_spread_quality_features
from mt5pipe.features.session import add_session_features
from mt5pipe.features.time import add_time_features
from mt5pipe.quality.cleaning import clean_dataset, validate_bars
from mt5pipe.quality.gaps import detect_gaps, fill_bar_gaps
from mt5pipe.storage.parquet_store import ParquetStore
from mt5pipe.storage.paths import StoragePaths
from mt5pipe.utils.logging import get_logger

log = get_logger(__name__)

_MISSING = object()
_COMPILER_COMPAT_BASE_TIMEFRAME = "M1"
_COMPILER_COMPAT_CONTEXT_TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]
_COMPILER_COMPAT_HORIZONS = [5, 15, 60, 240]
_COMPILER_COMPAT_FEATURE_SELECTORS = ["time/*", "session/*", "quality/*", "htf_context/*"]
_COMPILER_COMPAT_LABEL_PACK_REF = "core_tb_volscaled@1.0.0"
_COMPILER_COMPAT_STATE_VERSION_REF = "state.default@1.0.0"
_COMPILER_COMPAT_TRUTH_POLICY_REF = "truth.default@1.0.0"


def build_dataset(
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
    paths: StoragePaths,
    store: ParquetStore,
    cfg: DatasetConfig,
    dataset_name: str = "default",
) -> pl.DataFrame:
    """Build a complete model-ready dataset.

    When the legacy request maps cleanly to the compiler-era DatasetSpec contract,
    route through the compiler service and mirror the resulting split files back into
    the legacy dataset location. Otherwise, keep the original legacy implementation.
    """
    compatibility_reason = _compiler_compatibility_reason(cfg)
    if compatibility_reason is None:
        try:
            compiled = _build_dataset_via_compiler_compat(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                paths=paths,
                store=store,
                cfg=cfg,
                dataset_name=dataset_name,
            )
            if compiled is not None:
                return compiled
        except Exception as exc:
            log.warning("dataset_compiler_compat_failed", dataset=dataset_name, error=str(exc))
    else:
        log.info("dataset_compiler_compat_unavailable", dataset=dataset_name, reason=compatibility_reason)

    return _build_dataset_legacy(symbol, start_date, end_date, paths, store, cfg, dataset_name)


def _build_dataset_legacy(
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
    paths: StoragePaths,
    store: ParquetStore,
    cfg: DatasetConfig,
    dataset_name: str = "default",
) -> pl.DataFrame:
    """Original dataset builder implementation."""
    from mt5pipe.bars.builder import timeframe_to_seconds

    base_tf = cfg.base_timeframe
    base_bars = _load_bars_range(symbol, base_tf, start_date, end_date, paths, store)

    if base_bars.is_empty():
        log.warning("dataset_no_base_bars", symbol=symbol, tf=base_tf)
        return pl.DataFrame()

    log.info("dataset_base_loaded", symbol=symbol, rows=len(base_bars))

    base_bars = validate_bars(base_bars)
    log.info("dataset_bars_validated", rows=len(base_bars))

    tf_secs = timeframe_to_seconds(base_tf)
    gap_report = detect_gaps(base_bars, base_tf, tf_secs)
    if gap_report.missing_bars > 0:
        log.info(
            "dataset_gaps_detected",
            missing=gap_report.missing_bars,
            completeness=f"{gap_report.completeness_pct:.1f}%",
        )
        base_bars = fill_bar_gaps(base_bars, base_tf, tf_secs)
        log.info("dataset_gaps_filled", rows=len(base_bars))
    elif "_filled" not in base_bars.columns:
        base_bars = base_bars.with_columns(pl.lit(False).alias("_filled"))

    base_bars = add_time_features(base_bars)
    base_bars = add_session_features(base_bars)
    base_bars = add_spread_quality_features(base_bars)

    for htf in cfg.context_timeframes:
        htf_bars = _load_bars_range(symbol, htf, start_date, end_date, paths, store)
        if htf_bars.is_empty():
            continue
        htf_bars = validate_bars(htf_bars)
        htf_secs = timeframe_to_seconds(htf)
        base_bars = add_lagged_bar_features(
            base_bars,
            htf_bars,
            htf,
            bar_duration_seconds=htf_secs,
        )
        log.info("dataset_htf_joined", htf=htf, bar_dur_s=htf_secs, cols_added=True)

    base_bars = add_future_returns(base_bars, cfg.horizons_minutes)
    base_bars = add_direction_labels(base_bars, cfg.horizons_minutes)
    base_bars = add_triple_barrier_labels(
        base_bars,
        cfg.horizons_minutes,
        tp_bps=cfg.triple_barrier_tp_bps,
        sl_bps=cfg.triple_barrier_sl_bps,
        vol_scale_window=cfg.triple_barrier_vol_lookback,
        vol_multiplier=cfg.triple_barrier_vol_multiplier,
    )

    max_horizon = max(cfg.horizons_minutes)
    purge_rows = max_horizon + 1
    if len(base_bars) <= purge_rows:
        log.warning("dataset_too_short_for_labels", rows=len(base_bars), purge_needed=purge_rows)
        return pl.DataFrame()
    base_bars = base_bars.head(len(base_bars) - purge_rows)

    pre_filter = len(base_bars)
    if "_filled" in base_bars.columns:
        base_bars = base_bars.filter(~pl.col("_filled"))
    if "source_count" in base_bars.columns:
        base_bars = base_bars.filter(pl.col("source_count") > 0)
    filtered_out = pre_filter - len(base_bars)
    if filtered_out > 0:
        log.info("dataset_filled_rows_removed", removed=filtered_out, remaining=len(base_bars))

    base_bars, _clean_stats = clean_dataset(base_bars)

    embargo = max(cfg.horizons_minutes)
    n = len(base_bars)
    train_end = int(n * cfg.train_ratio)
    val_start = train_end + embargo
    val_end = int(n * (cfg.train_ratio + cfg.val_ratio))
    test_start = val_end + embargo

    train_df = base_bars.slice(0, train_end)
    val_df = base_bars.slice(val_start, max(0, val_end - val_start)) if val_start < n else pl.DataFrame()
    test_df = base_bars.slice(test_start, max(0, n - test_start)) if test_start < n else pl.DataFrame()

    log.info(
        "dataset_split_info",
        train=len(train_df),
        val=len(val_df),
        test=len(test_df),
        embargo_rows=embargo,
    )

    for split_name in ("train", "val", "test"):
        split_dir = paths.dataset_dir(dataset_name, split_name)
        if split_dir.exists():
            shutil.rmtree(split_dir)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if split_df.is_empty():
            continue
        path = paths.dataset_file(dataset_name, split_name)
        store.write(split_df, path)
        log.info("dataset_split_written", split=split_name, rows=len(split_df), path=str(path))

    return base_bars


def _resolve_attr(value: Any, path: str, default: Any = _MISSING) -> Any:
    current = value
    for part in path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
            continue
        if not hasattr(current, part):
            return default
        current = getattr(current, part)
    return current


def _first_present(value: Any, *paths: str, default: Any = None) -> Any:
    for path in paths:
        resolved = _resolve_attr(value, path, _MISSING)
        if resolved is not _MISSING:
            return resolved
    return default


def _compiler_compatibility_reason(cfg: DatasetConfig) -> str | None:
    if cfg.base_timeframe != _COMPILER_COMPAT_BASE_TIMEFRAME:
        return f"base_timeframe must be {_COMPILER_COMPAT_BASE_TIMEFRAME}"
    if list(cfg.context_timeframes) != _COMPILER_COMPAT_CONTEXT_TIMEFRAMES:
        return f"context_timeframes must be {_COMPILER_COMPAT_CONTEXT_TIMEFRAMES}"
    if list(cfg.horizons_minutes) != _COMPILER_COMPAT_HORIZONS:
        return f"horizons_minutes must be {_COMPILER_COMPAT_HORIZONS}"
    if cfg.triple_barrier_tp_bps != 50.0 or cfg.triple_barrier_sl_bps != 50.0:
        return "triple barrier bps settings must match the stable compiler label pack"
    if cfg.triple_barrier_vol_lookback != 60 or cfg.triple_barrier_vol_multiplier != 2.0:
        return "vol-scaled triple barrier settings must match the stable compiler label pack"
    return None


def _build_dataset_via_compiler_compat(
    *,
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
    paths: StoragePaths,
    store: ParquetStore,
    cfg: DatasetConfig,
    dataset_name: str,
) -> pl.DataFrame | None:
    try:
        service = import_module("mt5pipe.compiler.service")
    except ModuleNotFoundError:
        return None
    compile_fn = getattr(service, "compile_dataset_spec", None)
    if not callable(compile_fn):
        return None

    with tempfile.TemporaryDirectory(prefix="mt5pipe_dataset_compat_") as tmp_dir:
        spec_path = Path(tmp_dir) / "dataset_spec.yaml"
        _write_compiler_compat_spec(spec_path, symbol, start_date, end_date, cfg, dataset_name)
        result = compile_fn(spec_path, publish=False)

    artifact_id = str(_first_present(result, "artifact_id", "manifest.artifact_id", default=""))
    logical_name = str(
        _first_present(result, "manifest.logical_name", "logical_name", "spec.dataset_name", default=dataset_name)
    )
    if not artifact_id:
        raise ValueError("compiler compatibility build did not return an artifact_id")

    split_frames: list[pl.DataFrame] = []
    for split_name in ("train", "val", "test"):
        legacy_dir = paths.dataset_dir(logical_name, split_name)
        if legacy_dir.exists():
            shutil.rmtree(legacy_dir)

        artifact_dir = paths.compiler_dataset_dir(logical_name, artifact_id, split_name)
        split_df = store.read_dir(artifact_dir)
        if split_df.is_empty():
            continue

        store.write(split_df, paths.dataset_file(logical_name, split_name))
        split_frames.append(split_df)

    if not split_frames:
        raise FileNotFoundError(
            f"compiler compatibility build produced no split data for dataset={logical_name} artifact_id={artifact_id}"
        )

    dataset_df = pl.concat(split_frames, how="diagonal_relaxed")
    if "time_utc" in dataset_df.columns:
        dataset_df = dataset_df.sort("time_utc")

    log.info(
        "dataset_compiler_compat_succeeded",
        dataset=logical_name,
        artifact_id=artifact_id,
        rows=len(dataset_df),
    )
    return dataset_df


def _write_compiler_compat_spec(
    path: Path,
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
    cfg: DatasetConfig,
    dataset_name: str,
) -> None:
    version = f"legacy-compat-{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
    lines = [
        'schema_version: "1.0.0"',
        f'dataset_name: "{dataset_name}"',
        f'version: "{version}"',
        f'description: "Legacy dataset build compatibility spec for {dataset_name}"',
        "symbols:",
        f'  - "{symbol}"',
        f'date_from: "{start_date.isoformat()}"',
        f'date_to: "{end_date.isoformat()}"',
        f'base_clock: "{cfg.base_timeframe}"',
        f'state_version_ref: "{_COMPILER_COMPAT_STATE_VERSION_REF}"',
        "feature_selectors:",
    ]
    lines.extend(f'  - "{selector}"' for selector in _COMPILER_COMPAT_FEATURE_SELECTORS)
    lines.extend([
        f'label_pack_ref: "{_COMPILER_COMPAT_LABEL_PACK_REF}"',
        "filters:",
        '  - "exclude:filled_rows"',
        'split_policy: "temporal_holdout"',
        f"train_ratio: {cfg.train_ratio:.2f}",
        f"val_ratio: {cfg.val_ratio:.2f}",
        f"test_ratio: {cfg.test_ratio:.2f}",
        f"embargo_rows: {max(cfg.horizons_minutes)}",
        f'truth_policy_ref: "{_COMPILER_COMPAT_TRUTH_POLICY_REF}"',
        "publish_on_accept: false",
        "tags:",
        '  - "legacy-compat"',
        '  - "cli-build"',
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_bars_range(
    symbol: str,
    timeframe: str,
    start_date: dt.date,
    end_date: dt.date,
    paths: StoragePaths,
    store: ParquetStore,
) -> pl.DataFrame:
    """Load built bars for a date range."""
    frames: list[pl.DataFrame] = []
    current = start_date
    while current <= end_date:
        bar_dir = paths.built_bars_dir(symbol, timeframe, current)
        df = store.read_dir(bar_dir)
        if not df.is_empty():
            frames.append(df)
        current += dt.timedelta(days=1)

    if not frames:
        return pl.DataFrame()

    result = pl.concat(frames, how="diagonal_relaxed")
    return result.sort("time_utc")


def walk_forward_splits(
    df: pl.DataFrame,
    n_splits: int = 5,
    train_pct: float = 0.6,
    gap_pct: float = 0.02,
    embargo_rows: int = 240,
) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    """Generate walk-forward train/test splits for time-series cross-validation."""
    n = len(df)
    splits: list[tuple[pl.DataFrame, pl.DataFrame]] = []
    step = int(n * (1.0 - train_pct) / n_splits)
    gap = max(int(n * gap_pct), embargo_rows)

    for i in range(n_splits):
        test_start = int(n * train_pct) + i * step
        test_end = min(test_start + step, n)
        train_end = test_start - gap

        if train_end < 1 or test_start >= n:
            continue

        train = df.slice(0, train_end)
        test = df.slice(test_start, test_end - test_start)

        if train.is_empty() or test.is_empty():
            continue

        splits.append((train, test))

    return splits
