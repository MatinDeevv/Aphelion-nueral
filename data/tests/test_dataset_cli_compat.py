from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import polars as pl
from typer.testing import CliRunner

from mt5pipe.cli.app import app
from mt5pipe.compiler.manifest import load_dataset_spec
from mt5pipe.config.models import DatasetConfig
from mt5pipe.features.dataset import build_dataset


runner = CliRunner()
UTC = dt.timezone.utc


def _install_compiler_service_stub(monkeypatch):
    module = types.ModuleType("mt5pipe.compiler.service")
    monkeypatch.setitem(sys.modules, "mt5pipe.compiler.service", module)
    return module


def _make_trust_report(*, status: str = "accepted", score: float = 97.25) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        accepted_for_publication=status == "accepted",
        trust_score_total=score,
        coverage_score=99.0,
        leakage_score=98.0,
        feature_quality_score=97.0,
        label_quality_score=96.0,
        source_quality_score=95.0,
        lineage_score=94.0,
        decision_summary=(
            f"{status} for publication; total={score:.2f}"
            if status != "accepted"
            else f"accepted for publication; total={score:.2f}"
        ),
        warning_reasons=[],
        rejection_reasons=[],
        check_status_counts={"passed": 8, "warning": 0, "failed": 0},
        metrics={
            "dataset_quality": {
                "quality_score": score,
                "total_nulls": 0,
                "null_columns": {},
                "constant_columns": [],
                "expected_sparse_null_columns": {},
                "unexpected_null_columns": {},
                "slice_trivial_constant_columns": {},
                "blocking_constant_columns": {},
                "family_warning_summary": {},
            },
            "quality_caveat_summary": {
                "accepted_caveats": [],
                "green_blockers": [],
                "publication_blockers": [],
                "family_warning_summary": {},
            },
            "source_quality": {
                "merge_observability_source": "merge_qa",
                "merge_qa_days": 2.0,
                "state_quality_mean": score,
                "state_filled_ratio": 0.0,
                "dual_source_ratio_mean": 0.8,
                "merge_conflict_mean": 0.0,
            }
        },
        hard_failures=[],
        warnings=[],
    )


def _make_inspection(
    *,
    artifact_id: str,
    logical_name: str = "xau_m1_core",
    logical_version: str = "1.0.0",
    status: str = "published",
    schema_columns: list[str] | None = None,
    feature_refs: list[str] | None = None,
    split_rows: dict[str, int] | None = None,
    dataset_spec_ref: str = "xau_m1_core@1.0.0",
    label_pack_ref: str = "core_tb_volscaled@1.0.0",
    trust_report: SimpleNamespace | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        artifact_id=artifact_id,
        manifest_path=Path(f"local_data/pipeline_data/manifests/kind=dataset/name={logical_name}/manifest.json"),
        manifest=SimpleNamespace(
            artifact_id=artifact_id,
            logical_name=logical_name,
            logical_version=logical_version,
            status=status,
            dataset_spec_ref=dataset_spec_ref,
            feature_spec_refs=feature_refs or [
                "time.cyclical_time@1.0.0",
                "session.session_flags@1.0.0",
                "quality.spread_quality@1.0.0",
                "htf_context.standard_context@1.0.0",
            ],
            label_pack_ref=label_pack_ref,
            state_artifact_refs=["state.xauusd.m1.20260401"],
            parent_artifact_refs=[
                "state.xauusd.m1.20260401",
                "feature.time.20260401",
                "feature.session.20260401",
                "feature.quality.20260401",
            ],
            metadata={
                "time_range_start": "2026-04-01 00:00:00+00:00",
                "time_range_end": "2026-04-02 23:59:00+00:00",
                "split_row_counts": split_rows or {"train": 700, "val": 150, "test": 150},
                "schema_columns": schema_columns or [
                    "symbol",
                    "timeframe",
                    "time_utc",
                    "open",
                    "high",
                    "low",
                    "close",
                    "hour",
                    "session_london",
                    "relative_spread",
                    "M5_close",
                    "triple_barrier_60m",
                ],
            },
        ),
        trust_report=trust_report or _make_trust_report(status="accepted", score=97.25),
    )


def test_compile_dataset_cli_success(monkeypatch) -> None:
    compiler_service = _install_compiler_service_stub(monkeypatch)
    calls: list[tuple[str, bool]] = []

    def fake_compile(spec_path: str | Path, *, publish: bool = True):
        calls.append((str(spec_path), publish))
        return SimpleNamespace(
            artifact_id="dataset.xau_m1_core.20260404T190000Z.abc12345",
            spec=SimpleNamespace(dataset_name="xau_m1_core", version="1.0.0"),
            manifest=SimpleNamespace(
                artifact_id="dataset.xau_m1_core.20260404T190000Z.abc12345",
                logical_name="xau_m1_core",
                logical_version="1.0.0",
                status="published",
            ),
            manifest_path=Path("local_data/pipeline_data/manifests/kind=dataset/name=xau_m1_core/manifest.json"),
            trust_report=_make_trust_report(status="accepted", score=98.5),
        )

    compiler_service.compile_dataset_spec = fake_compile

    result = runner.invoke(app, ["dataset", "compile-dataset", "--spec", "config/datasets/xau_m1_core_v1.yaml"])

    assert result.exit_code == 0
    assert Path(calls[0][0]) == Path("config/datasets/xau_m1_core_v1.yaml")
    assert calls[0][1] is True
    assert "artifact_id: dataset.xau_m1_core.20260404T190000Z.abc12345" in result.stdout
    assert "logical: xau_m1_core@1.0.0" in result.stdout
    assert "status: published" in result.stdout
    assert "manifest_path:" in result.stdout
    assert "manifest.json" in result.stdout
    assert "trust_status: accepted" in result.stdout
    assert "trust_score_total: 98.50" in result.stdout
    assert "trust_decision: accepted for publication; total=98.50" in result.stdout
    assert 'trust_check_counts: {"failed": 0, "passed": 8, "warning": 0}' in result.stdout
    assert "trust_warning_reasons: []" in result.stdout
    assert "trust_rejection_reasons: []" in result.stdout
    assert 'quality_caveats: {"accepted_caveats": [], "green_blockers": [], "publication_blockers": []}' in result.stdout
    assert 'source_quality_metrics: {"dual_source_ratio_mean": 0.8, "merge_conflict_mean": 0.0, "merge_observability_source": "merge_qa", "merge_qa_days": 2.0, "state_filled_ratio": 0.0, "state_quality_mean": 98.5}' in result.stdout
    assert "published_ref: dataset://xau_m1_core@1.0.0" in result.stdout


def test_inspect_dataset_cli_resolution_path(monkeypatch, tmp_path: Path) -> None:
    compiler_service = _install_compiler_service_stub(monkeypatch)
    refs_seen: list[str] = []
    manifest_path = tmp_path / "manifest.json"

    def fake_inspect(ref: str):
        refs_seen.append(ref)
        return _make_inspection(artifact_id="dataset.inspect.001")

    compiler_service.inspect_artifact = fake_inspect

    for ref in ["dataset.inspect.001", "dataset://xau_m1_core@1.0.0", str(manifest_path)]:
        result = runner.invoke(app, ["dataset", "inspect-dataset", "--artifact", ref])
        assert result.exit_code == 0
        assert "artifact_id: dataset.inspect.001" in result.stdout
        assert "manifest_path:" in result.stdout
        assert "time_range: 2026-04-01 00:00:00+00:00 -> 2026-04-02 23:59:00+00:00" in result.stdout
        assert 'feature_families: ["htf_context", "quality", "session", "time"]' in result.stdout
        assert "label_pack: core_tb_volscaled@1.0.0" in result.stdout
        assert "schema_columns_count: 12" in result.stdout
        assert "trust_score_total: 97.25" in result.stdout
        assert "trust_decision: accepted for publication; total=97.25" in result.stdout
        assert 'trust_check_counts: {"failed": 0, "passed": 8, "warning": 0}' in result.stdout
        assert "trust_warning_reasons: []" in result.stdout
        assert "trust_rejection_reasons: []" in result.stdout
        assert 'quality_caveats: {"accepted_caveats": [], "green_blockers": [], "publication_blockers": []}' in result.stdout
        assert "quality_family_summary: {}" in result.stdout
        assert 'source_quality_metrics: {"dual_source_ratio_mean": 0.8, "merge_conflict_mean": 0.0, "merge_observability_source": "merge_qa", "merge_qa_days": 2.0, "state_filled_ratio": 0.0, "state_quality_mean": 97.25}' in result.stdout
        assert 'dataset_quality_alerts: {"blocking_constant_columns_sample": [], "constant_columns_sample": [], "expected_sparse_null_columns_sample": {}, "null_columns_sample": {}, "quality_score": 97.25, "slice_trivial_constant_columns_sample": [], "total_nulls": 0, "unexpected_null_columns_sample": {}}' in result.stdout
        assert '"state_artifacts": ["state.xauusd.m1.20260401"]' in result.stdout

    assert refs_seen == ["dataset.inspect.001", "dataset://xau_m1_core@1.0.0", str(manifest_path)]


def test_diff_dataset_cli_resolution_path(monkeypatch, tmp_path: Path) -> None:
    compiler_service = _install_compiler_service_stub(monkeypatch)
    calls: list[tuple[str, str]] = []

    left = _make_inspection(
        artifact_id="dataset.left.001",
        logical_version="1.0.0",
        schema_columns=["symbol", "time_utc", "hour", "relative_spread"],
        feature_refs=["time.cyclical_time@1.0.0", "quality.spread_quality@1.0.0"],
        split_rows={"train": 700, "val": 150, "test": 150},
        dataset_spec_ref="xau_m1_core@1.0.0",
    )
    right_trust_report = _make_trust_report(status="accepted", score=97.25)
    right_trust_report = SimpleNamespace(
        **{
            **right_trust_report.__dict__,
            "decision_summary": "accepted for publication with warnings; total=97.25, warnings=1",
            "warning_reasons": ["source_quality: score 72.00 is below preferred 75.00"],
            "check_status_counts": {"passed": 7, "warning": 1, "failed": 0},
            "metrics": {
                **right_trust_report.metrics,
                "quality_caveat_summary": {
                    "accepted_caveats": [
                        "expected sparse nulls: htf_context: expected alignment sparsity in H1_tick_count=10"
                    ],
                    "green_blockers": ["source_quality below preferred threshold (72.00 < 75.00)"],
                    "publication_blockers": [],
                    "family_warning_summary": {
                        "htf_context": ["expected alignment sparsity in H1_tick_count=10"],
                    },
                },
                "source_quality": {
                    "merge_observability_source": "merge_diagnostics",
                    "merge_diagnostics_days": 2.0,
                    "state_quality_mean": 72.0,
                    "state_filled_ratio": 0.0,
                    "diagnostic_dual_source_ratio_mean": 0.0,
                    "diagnostic_conflict_mean": 0.0,
                },
            },
        }
    )

    right = _make_inspection(
        artifact_id="dataset.right.002",
        logical_version="1.0.1",
        schema_columns=["symbol", "time_utc", "hour", "M5_close"],
        feature_refs=["time.cyclical_time@1.0.0", "htf_context.standard_context@1.0.0"],
        split_rows={"train": 720, "val": 140, "test": 140},
        dataset_spec_ref="xau_m1_core@1.0.1",
        trust_report=right_trust_report,
    )

    def fake_diff(left_ref: str, right_ref: str):
        calls.append((left_ref, right_ref))
        return SimpleNamespace(left=left, right=right)

    compiler_service.diff_artifacts = fake_diff

    result = runner.invoke(
        app,
        [
            "dataset",
            "diff-dataset",
            "--left",
            "dataset.left.001",
            "--right",
            str(tmp_path / "right_manifest.json"),
        ],
    )

    assert result.exit_code == 0
    assert calls == [("dataset.left.001", str(tmp_path / "right_manifest.json"))]
    assert "left_spec_ref: xau_m1_core@1.0.0" in result.stdout
    assert "right_spec_ref: xau_m1_core@1.0.1" in result.stdout
    assert 'feature_families_left: ["quality", "time"]' in result.stdout
    assert 'feature_families_right: ["htf_context", "time"]' in result.stdout
    assert 'feature_refs_added: ["htf_context.standard_context@1.0.0"]' in result.stdout
    assert 'feature_refs_removed: ["quality.spread_quality@1.0.0"]' in result.stdout
    assert 'split_row_deltas: {"test": -10, "train": 20, "val": -10}' in result.stdout
    assert 'schema_columns_added: ["M5_close"]' in result.stdout
    assert 'schema_columns_removed: ["relative_spread"]' in result.stdout
    assert "trust_status_left: accepted" in result.stdout
    assert "trust_status_right: accepted" in result.stdout
    assert "trust_decision_left: accepted for publication; total=97.25" in result.stdout
    assert "trust_decision_right: accepted for publication with warnings; total=97.25, warnings=1" in result.stdout
    assert 'trust_check_counts_left: {"failed": 0, "passed": 8, "warning": 0}' in result.stdout
    assert 'trust_check_counts_right: {"failed": 0, "passed": 7, "warning": 1}' in result.stdout
    assert 'quality_caveats_left: {"accepted_caveats": [], "green_blockers": [], "publication_blockers": []}' in result.stdout
    assert 'quality_caveats_right: {"accepted_caveats": ["expected sparse nulls: htf_context: expected alignment sparsity in H1_tick_count=10"], "green_blockers": ["source_quality below preferred threshold (72.00 < 75.00)"], "publication_blockers": []}' in result.stdout
    assert 'quality_family_summary_right: {"htf_context": "expected alignment sparsity in H1_tick_count=10"}' in result.stdout
    assert 'source_quality_metrics_right: {"diagnostic_conflict_mean": 0.0, "diagnostic_dual_source_ratio_mean": 0.0, "merge_diagnostics_days": 2.0, "merge_observability_source": "merge_diagnostics", "state_filled_ratio": 0.0, "state_quality_mean": 72.0}' in result.stdout
    assert 'trust_warning_reasons_added: ["source_quality: score 72.00 is below preferred 75.00"]' in result.stdout


def test_legacy_dataset_build_compatibility_facade(monkeypatch, paths, store) -> None:
    compiler_service = _install_compiler_service_stub(monkeypatch)
    compat_specs: list[Path] = []
    cfg = DatasetConfig(horizons_minutes=[5, 15, 60, 240])
    train_df = pl.DataFrame(
        {
            "symbol": ["XAUUSD", "XAUUSD"],
            "time_utc": [
                dt.datetime(2026, 4, 1, 0, 0, tzinfo=UTC),
                dt.datetime(2026, 4, 1, 0, 1, tzinfo=UTC),
            ],
            "feature_a": [1.0, 2.0],
        }
    )
    val_df = pl.DataFrame(
        {
            "symbol": ["XAUUSD"],
            "time_utc": [dt.datetime(2026, 4, 1, 1, 0, tzinfo=UTC)],
            "feature_a": [3.0],
        }
    )
    test_df = pl.DataFrame(
        {
            "symbol": ["XAUUSD"],
            "time_utc": [dt.datetime(2026, 4, 1, 2, 0, tzinfo=UTC)],
            "feature_a": [4.0],
        }
    )

    def fake_compile(spec_path: str | Path, *, publish: bool = True):
        compat_specs.append(Path(spec_path))
        spec = load_dataset_spec(Path(spec_path))
        assert publish is False
        assert spec.dataset_name == "legacy_core"
        assert spec.feature_selectors == ["time/*", "session/*", "quality/*", "htf_context/*"]
        assert spec.label_pack_ref == "core_tb_volscaled@1.0.0"
        assert spec.split_policy == "temporal_holdout"
        assert spec.publish_on_accept is False

        artifact_id = "dataset.legacy_core.20260404T190000Z.abc12345"
        store.write(train_df, paths.compiler_dataset_file("legacy_core", artifact_id, "train"))
        store.write(val_df, paths.compiler_dataset_file("legacy_core", artifact_id, "val"))
        store.write(test_df, paths.compiler_dataset_file("legacy_core", artifact_id, "test"))
        return SimpleNamespace(
            artifact_id=artifact_id,
            manifest=SimpleNamespace(logical_name="legacy_core"),
        )

    compiler_service.compile_dataset_spec = fake_compile

    combined = build_dataset(
        symbol="XAUUSD",
        start_date=dt.date(2026, 4, 1),
        end_date=dt.date(2026, 4, 2),
        paths=paths,
        store=store,
        cfg=cfg,
        dataset_name="legacy_core",
    )

    assert len(compat_specs) == 1
    assert len(combined) == 4
    assert store.read(paths.dataset_file("legacy_core", "train")).height == 2
    assert store.read(paths.dataset_file("legacy_core", "val")).height == 1
    assert store.read(paths.dataset_file("legacy_core", "test")).height == 1


def test_example_spec_loading() -> None:
    spec = load_dataset_spec(Path("config/datasets/xau_m1_core_v1.yaml"))

    assert spec.dataset_name == "xau_m1_core"
    assert spec.symbols == ["XAUUSD"]
    assert spec.base_clock == "M1"
    assert spec.feature_selectors == ["time/*", "session/*", "quality/*", "htf_context/*"]
    assert spec.label_pack_ref == "core_tb_volscaled@1.0.0"
    assert spec.split_policy == "temporal_holdout"


def test_phase3_example_spec_loading() -> None:
    spec = load_dataset_spec(Path("config/datasets/xau_m1_nonhuman_v1.yaml"))

    assert spec.dataset_name == "xau_m1_nonhuman"
    assert spec.symbols == ["XAUUSD"]
    assert spec.base_clock == "M1"
    assert spec.feature_selectors == [
        "time/*",
        "session/*",
        "quality/*",
        "htf_context/*",
        "disagreement/*",
        "event_shape/*",
        "entropy/*",
        "multiscale/*",
    ]
    assert spec.feature_artifact_refs == []
    assert spec.label_pack_ref == "core_tb_volscaled@1.0.0"
