"""Path conventions for Hive-partitioned Parquet storage."""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path


class StoragePaths:
    """Builds filesystem paths for all data categories."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def _compact_name(value: str, *, prefix_len: int = 24) -> str:
        """Keep manifest-oriented path segments short and deterministic on Windows."""
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        safe_prefix = "".join(ch if ch.isalnum() or ch in {".", "_", "-"} else "_" for ch in value)
        safe_prefix = safe_prefix[:prefix_len].rstrip("._-")
        if not safe_prefix:
            safe_prefix = "artifact"
        return f"{safe_prefix}-{digest}"

    # --- Raw ticks ---
    def raw_ticks_dir(self, broker_id: str, symbol: str, date: dt.date) -> Path:
        return self.root / "raw_ticks" / f"broker={broker_id}" / f"symbol={symbol}" / f"date={date.isoformat()}"

    def raw_ticks_file(self, broker_id: str, symbol: str, date: dt.date, part: int = 0) -> Path:
        d = self.raw_ticks_dir(broker_id, symbol, date)
        return d / f"part-{part:05d}.parquet"

    # --- Native bars ---
    def native_bars_dir(self, broker_id: str, symbol: str, timeframe: str, date: dt.date) -> Path:
        return (
            self.root / "native_bars" / f"broker={broker_id}" / f"symbol={symbol}"
            / f"timeframe={timeframe}" / f"date={date.isoformat()}"
        )

    def native_bars_file(self, broker_id: str, symbol: str, timeframe: str, date: dt.date, part: int = 0) -> Path:
        return self.native_bars_dir(broker_id, symbol, timeframe, date) / f"part-{part:05d}.parquet"

    # --- Symbol metadata ---
    def symbol_metadata_dir(self, broker_id: str) -> Path:
        return self.root / "symbol_metadata" / f"broker={broker_id}"

    def symbol_metadata_file(self, broker_id: str, ts: dt.datetime) -> Path:
        return self.symbol_metadata_dir(broker_id) / f"{ts.strftime('%Y%m%d_%H%M%S')}.parquet"

    # --- Symbol universe ---
    def symbol_universe_dir(self, broker_id: str) -> Path:
        return self.root / "symbol_universe" / f"broker={broker_id}"

    def symbol_universe_file(self, broker_id: str, ts: dt.datetime) -> Path:
        return self.symbol_universe_dir(broker_id) / f"{ts.strftime('%Y%m%d_%H%M%S')}.parquet"

    # --- Market book ---
    def market_book_dir(self, broker_id: str, symbol: str, date: dt.date) -> Path:
        return self.root / "market_book" / f"broker={broker_id}" / f"symbol={symbol}" / f"date={date.isoformat()}"

    def market_book_file(self, broker_id: str, symbol: str, date: dt.date, part: int = 0) -> Path:
        return self.market_book_dir(broker_id, symbol, date) / f"part-{part:05d}.parquet"

    # --- Account state ---
    def account_state_dir(self, broker_id: str) -> Path:
        return self.root / "account_state" / f"broker={broker_id}"

    def account_state_file(self, broker_id: str, date: dt.date) -> Path:
        return self.account_state_dir(broker_id) / f"date={date.isoformat()}.parquet"

    # --- Terminal state ---
    def terminal_state_dir(self, broker_id: str) -> Path:
        return self.root / "terminal_state" / f"broker={broker_id}"

    def terminal_state_file(self, broker_id: str, date: dt.date) -> Path:
        return self.terminal_state_dir(broker_id) / f"date={date.isoformat()}.parquet"

    # --- Active orders ---
    def orders_active_dir(self, broker_id: str) -> Path:
        return self.root / "orders_active" / f"broker={broker_id}"

    def orders_active_file(self, broker_id: str, date: dt.date) -> Path:
        return self.orders_active_dir(broker_id) / f"date={date.isoformat()}.parquet"

    # --- Active positions ---
    def positions_active_dir(self, broker_id: str) -> Path:
        return self.root / "positions_active" / f"broker={broker_id}"

    def positions_active_file(self, broker_id: str, date: dt.date) -> Path:
        return self.positions_active_dir(broker_id) / f"date={date.isoformat()}.parquet"

    # --- Historical orders ---
    def history_orders_dir(self, broker_id: str) -> Path:
        return self.root / "history_orders" / f"broker={broker_id}"

    def history_orders_file(self, broker_id: str, date: dt.date) -> Path:
        return self.history_orders_dir(broker_id) / f"date={date.isoformat()}.parquet"

    # --- Historical deals ---
    def history_deals_dir(self, broker_id: str) -> Path:
        return self.root / "history_deals" / f"broker={broker_id}"

    def history_deals_file(self, broker_id: str, date: dt.date) -> Path:
        return self.history_deals_dir(broker_id) / f"date={date.isoformat()}.parquet"

    # --- Canonical ticks ---
    def canonical_ticks_dir(self, symbol: str, date: dt.date) -> Path:
        return self.root / "canonical_ticks" / f"symbol={symbol}" / f"date={date.isoformat()}"

    def canonical_ticks_file(self, symbol: str, date: dt.date, part: int = 0) -> Path:
        return self.canonical_ticks_dir(symbol, date) / f"part-{part:05d}.parquet"

    # --- Merge diagnostics ---
    def merge_diagnostics_dir(self, symbol: str, date: dt.date) -> Path:
        return self.root / "merge_diagnostics" / f"symbol={symbol}" / f"date={date.isoformat()}"

    def merge_diagnostics_file(self, symbol: str, date: dt.date, part: int = 0) -> Path:
        return self.merge_diagnostics_dir(symbol, date) / f"part-{part:05d}.parquet"

    # --- Daily merge QA reports ---
    def merge_qa_dir(self, symbol: str, date: dt.date) -> Path:
        return self.root / "merge_qa" / f"symbol={symbol}" / f"date={date.isoformat()}"

    def merge_qa_file(self, symbol: str, date: dt.date, part: int = 0) -> Path:
        return self.merge_qa_dir(symbol, date) / f"part-{part:05d}.parquet"

    # --- State artifacts ---
    def state_root(self, symbol: str, clock: str, state_version: str) -> Path:
        return (
            self.root
            / "state"
            / f"symbol={symbol}"
            / f"clock={clock}"
            / f"state_version={state_version}"
        )

    def state_artifact_root(self, symbol: str, clock: str, state_version: str, artifact_id: str) -> Path:
        return (
            self.root
            / "sa"
            / f"s={symbol}"
            / f"c={clock}"
            / f"v={self._compact_name(state_version, prefix_len=8)}"
            / f"a={self._compact_name(artifact_id, prefix_len=8)}"
        )

    def state_dir(self, symbol: str, clock: str, date: dt.date, state_version: str) -> Path:
        return self.state_root(symbol, clock, state_version) / f"date={date.isoformat()}"

    def state_artifact_dir(
        self,
        symbol: str,
        clock: str,
        date: dt.date,
        state_version: str,
        artifact_id: str,
    ) -> Path:
        return self.state_artifact_root(symbol, clock, state_version, artifact_id) / f"date={date.isoformat()}"

    def state_file(self, symbol: str, clock: str, date: dt.date, state_version: str, part: int = 0) -> Path:
        return self.state_dir(symbol, clock, date, state_version) / f"part-{part:05d}.parquet"

    def state_artifact_file(
        self,
        symbol: str,
        clock: str,
        date: dt.date,
        state_version: str,
        artifact_id: str,
        part: int = 0,
    ) -> Path:
        return self.state_artifact_dir(symbol, clock, date, state_version, artifact_id) / f"part-{part:05d}.parquet"

    # --- Rolling state windows ---
    def state_window_root(self, symbol: str, clock: str, state_version: str, window_size: str) -> Path:
        return (
            self.root
            / "state_windows"
            / f"symbol={symbol}"
            / f"clock={clock}"
            / f"state_version={state_version}"
            / f"window={window_size}"
        )

    def state_window_artifact_root(
        self,
        symbol: str,
        clock: str,
        state_version: str,
        window_size: str,
        artifact_id: str,
    ) -> Path:
        return (
            self.root
            / "swa"
            / f"s={symbol}"
            / f"c={clock}"
            / f"v={self._compact_name(state_version, prefix_len=8)}"
            / f"w={window_size}"
            / f"a={self._compact_name(artifact_id, prefix_len=8)}"
        )

    def state_window_dir(
        self,
        symbol: str,
        clock: str,
        date: dt.date,
        state_version: str,
        window_size: str,
    ) -> Path:
        return self.state_window_root(symbol, clock, state_version, window_size) / f"date={date.isoformat()}"

    def state_window_artifact_dir(
        self,
        symbol: str,
        clock: str,
        date: dt.date,
        state_version: str,
        window_size: str,
        artifact_id: str,
    ) -> Path:
        return self.state_window_artifact_root(
            symbol,
            clock,
            state_version,
            window_size,
            artifact_id,
        ) / f"date={date.isoformat()}"

    def state_window_file(
        self,
        symbol: str,
        clock: str,
        date: dt.date,
        state_version: str,
        window_size: str,
        part: int = 0,
    ) -> Path:
        return self.state_window_dir(symbol, clock, date, state_version, window_size) / f"part-{part:05d}.parquet"

    def state_window_artifact_file(
        self,
        symbol: str,
        clock: str,
        date: dt.date,
        state_version: str,
        window_size: str,
        artifact_id: str,
        part: int = 0,
    ) -> Path:
        return self.state_window_artifact_dir(
            symbol,
            clock,
            date,
            state_version,
            window_size,
            artifact_id,
        ) / f"part-{part:05d}.parquet"

    # --- Feature views ---
    def feature_view_dir(self, feature_key: str, clock: str, date: dt.date) -> Path:
        return (
            self.root
            / "feature_views"
            / f"feature={feature_key}"
            / f"clock={clock}"
            / f"date={date.isoformat()}"
        )

    def feature_artifact_root(self, feature_key: str, clock: str, artifact_id: str) -> Path:
        return (
            self.root
            / "fva"
            / f"f={self._compact_name(feature_key, prefix_len=8)}"
            / f"c={clock}"
            / f"a={self._compact_name(artifact_id, prefix_len=8)}"
        )

    def feature_artifact_dir(self, feature_key: str, clock: str, artifact_id: str, date: dt.date) -> Path:
        return self.feature_artifact_root(feature_key, clock, artifact_id) / f"date={date.isoformat()}"

    def feature_view_file(self, feature_key: str, clock: str, date: dt.date, part: int = 0) -> Path:
        return self.feature_view_dir(feature_key, clock, date) / f"part-{part:05d}.parquet"

    def feature_artifact_file(
        self,
        feature_key: str,
        clock: str,
        artifact_id: str,
        date: dt.date,
        part: int = 0,
    ) -> Path:
        return self.feature_artifact_dir(feature_key, clock, artifact_id, date) / f"part-{part:05d}.parquet"

    # --- Label views ---
    def label_view_dir(self, label_pack_key: str, clock: str, date: dt.date) -> Path:
        return (
            self.root
            / "label_views"
            / f"label_pack={label_pack_key}"
            / f"clock={clock}"
            / f"date={date.isoformat()}"
        )

    def label_artifact_root(self, label_pack_key: str, clock: str, artifact_id: str) -> Path:
        return (
            self.root
            / "lva"
            / f"l={self._compact_name(label_pack_key, prefix_len=8)}"
            / f"c={clock}"
            / f"a={self._compact_name(artifact_id, prefix_len=8)}"
        )

    def label_artifact_dir(self, label_pack_key: str, clock: str, artifact_id: str, date: dt.date) -> Path:
        return self.label_artifact_root(label_pack_key, clock, artifact_id) / f"date={date.isoformat()}"

    def label_view_file(self, label_pack_key: str, clock: str, date: dt.date, part: int = 0) -> Path:
        return self.label_view_dir(label_pack_key, clock, date) / f"part-{part:05d}.parquet"

    def label_artifact_file(
        self,
        label_pack_key: str,
        clock: str,
        artifact_id: str,
        date: dt.date,
        part: int = 0,
    ) -> Path:
        return self.label_artifact_dir(label_pack_key, clock, artifact_id, date) / f"part-{part:05d}.parquet"

    # --- Built bars ---
    def built_bars_dir(self, symbol: str, timeframe: str, date: dt.date) -> Path:
        return self.root / "bars" / f"symbol={symbol}" / f"timeframe={timeframe}" / f"date={date.isoformat()}"

    def built_bars_file(self, symbol: str, timeframe: str, date: dt.date, part: int = 0) -> Path:
        return self.built_bars_dir(symbol, timeframe, date) / f"part-{part:05d}.parquet"

    # --- Datasets ---
    def dataset_dir(self, name: str, split: str) -> Path:
        return self.root / "datasets" / f"name={name}" / f"split={split}"

    def dataset_file(self, name: str, split: str, part: int = 0) -> Path:
        return self.dataset_dir(name, split) / f"part-{part:05d}.parquet"

    # --- Compiler dataset artifacts ---
    def compiler_dataset_dir(self, dataset_name: str, artifact_id: str, split: str) -> Path:
        return (
            self.root
            / "datasets"
            / f"name={dataset_name}"
            / f"artifact={artifact_id}"
            / f"split={split}"
        )

    def compiler_dataset_file(self, dataset_name: str, artifact_id: str, split: str, part: int = 0) -> Path:
        return self.compiler_dataset_dir(dataset_name, artifact_id, split) / f"part-{part:05d}.parquet"

    # --- Experiment artifacts ---
    def experiment_root(self, experiment_name: str, artifact_id: str) -> Path:
        short_name = self._compact_name(experiment_name)
        short_artifact_id = self._compact_name(artifact_id, prefix_len=18)
        return (
            self.root
            / "experiments"
            / f"name={short_name}"
            / f"artifact={short_artifact_id}"
        )

    def experiment_predictions_file(self, experiment_name: str, artifact_id: str) -> Path:
        return self.experiment_root(experiment_name, artifact_id) / "predictions.parquet"

    def experiment_summary_file(self, experiment_name: str, artifact_id: str) -> Path:
        return self.experiment_root(experiment_name, artifact_id) / "summary.json"

    # --- Model artifacts ---
    def model_root(self, model_name: str, artifact_id: str) -> Path:
        short_name = self._compact_name(model_name)
        short_artifact_id = self._compact_name(artifact_id, prefix_len=18)
        return (
            self.root
            / "models"
            / f"name={short_name}"
            / f"artifact={short_artifact_id}"
        )

    def model_payload_file(self, model_name: str, artifact_id: str) -> Path:
        return self.model_root(model_name, artifact_id) / "model.json"

    # --- Compiler manifests ---
    def manifest_dir(self, artifact_kind: str, logical_name: str) -> Path:
        short_name = self._compact_name(logical_name)
        return self.root / "manifests" / f"kind={artifact_kind}" / f"name={short_name}"

    def manifest_file(self, artifact_kind: str, logical_name: str, manifest_id: str) -> Path:
        short_manifest_id = self._compact_name(manifest_id, prefix_len=18)
        return self.manifest_dir(artifact_kind, logical_name) / f"{short_manifest_id}.json"

    # --- Truth reports ---
    def truth_report_dir(self, artifact_id: str) -> Path:
        return self.root / "truth" / f"artifact={artifact_id}"

    def truth_report_file(self, artifact_id: str, report_id: str) -> Path:
        return self.truth_report_dir(artifact_id) / f"{report_id}.json"

    # --- Compiler catalog ---
    def catalog_db_path(self) -> Path:
        return self.root / "catalog" / "catalog.db"

    # --- SQLite checkpoint DB ---
    def checkpoint_db_path(self) -> Path:
        return self.root / "checkpoints.db"
