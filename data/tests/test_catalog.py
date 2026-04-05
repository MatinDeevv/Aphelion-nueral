"""Tests for compiler catalog persistence and resolution."""

from __future__ import annotations

import datetime as dt

from mt5pipe.catalog.sqlite import CatalogDB
from mt5pipe.compiler.models import DatasetSpec, LineageManifest
from mt5pipe.features.registry.defaults import get_default_feature_specs
from mt5pipe.labels.registry.defaults import get_default_label_packs
from mt5pipe.truth.models import QaCheckResult, TrustReport


UTC = dt.timezone.utc


def test_catalog_registers_and_resolves_artifact(tmp_path) -> None:
    catalog = CatalogDB(tmp_path / "catalog.db")
    try:
        feature_specs = get_default_feature_specs()
        label_packs = get_default_label_packs()
        catalog.register_feature_specs(feature_specs)
        catalog.register_label_packs(label_packs)

        spec = DatasetSpec(
            dataset_name="xau_core",
            version="1.0.0",
            symbols=["XAUUSD"],
            date_from=dt.date(2026, 4, 1),
            date_to=dt.date(2026, 4, 2),
            base_clock="M1",
            state_version_ref="state.default@1.0.0",
            feature_selectors=["time/*", "session/*"],
            label_pack_ref="core_tb_volscaled@1.0.0",
            split_policy="temporal_holdout",
            embargo_rows=240,
            truth_policy_ref="truth.default@1.0.0",
        )
        catalog.register_dataset_spec(spec)
        catalog.start_build(spec.key, "workspace-local-no-git", "build.1")

        manifest = LineageManifest(
            manifest_id="manifest.dataset.xau_core.abc123",
            artifact_id="dataset.xau_core.abc123",
            artifact_kind="dataset",
            logical_name="xau_core",
            logical_version="1.0.0",
            artifact_uri="data/datasets/name=xau_core/artifact=dataset.xau_core.abc123",
            content_hash="abc123",
            build_id="build.1",
            created_at=dt.datetime.now(UTC),
            status="building",
            dataset_spec_ref=spec.key,
            state_artifact_refs=["state.XAUUSD.M1.123"],
            feature_spec_refs=[feature.key for feature in feature_specs[:2]],
            label_pack_ref=label_packs[0].key,
            code_version="workspace-local-no-git",
            merge_config_ref="merge.default@test",
            input_partition_refs=["data/bars/symbol=XAUUSD/timeframe=M1/date=2026-04-01"],
            parent_artifact_refs=["state.XAUUSD.M1.123"],
        )
        catalog.register_artifact(manifest, "manifest.json", detail="building candidate")
        manifest = manifest.model_copy(update={"status": "truth_pending"})
        catalog.register_artifact(manifest, "manifest.json", detail="awaiting truth")
        manifest = manifest.model_copy(update={"status": "published", "truth_report_ref": "trust.dataset.xau_core.abc123"})
        catalog.register_artifact(manifest, "manifest.json", detail="published artifact")
        catalog.upsert_alias("dataset://xau_core@1.0.0", manifest.artifact_id)

        report = TrustReport(
            report_id="trust.dataset.xau_core.abc123",
            artifact_id=manifest.artifact_id,
            artifact_kind="dataset",
            truth_policy_version="truth.default@1.0.0",
            status="accepted",
            accepted_for_publication=True,
            trust_score_total=95.0,
            coverage_score=100.0,
            leakage_score=100.0,
            feature_quality_score=90.0,
            label_quality_score=90.0,
            source_quality_score=90.0,
            lineage_score=100.0,
            generated_at=dt.datetime.now(UTC),
            checks=[QaCheckResult(check_name="coverage", status="passed", score=100.0)],
        )
        catalog.register_trust_report(report)
        catalog.finish_build("build.1", "published", artifact_id=manifest.artifact_id)

        resolved = catalog.resolve_artifact("dataset://xau_core@1.0.0")
        assert resolved is not None
        assert resolved.artifact_id == manifest.artifact_id
        assert resolved.status == "published"

        stored_spec = catalog.get_dataset_spec(spec.key)
        assert stored_spec is not None
        assert stored_spec.dataset_name == "xau_core"

        stored_feature = catalog.get_feature_spec(feature_specs[0].key)
        assert stored_feature is not None
        assert stored_feature.family == feature_specs[0].family
        assert any(spec.key == feature_specs[0].key for spec in catalog.list_feature_specs())

        stored_label_pack = catalog.get_label_pack(label_packs[0].key)
        assert stored_label_pack is not None
        assert stored_label_pack.label_pack_name == label_packs[0].label_pack_name
        assert any(pack.key == label_packs[0].key for pack in catalog.list_label_packs())

        stored_build = catalog.get_build_run("build.1")
        assert stored_build is not None
        assert stored_build.status == "published"

        inputs = catalog.list_artifact_inputs(manifest.artifact_id)
        assert any(record.input_kind == "dataset_spec" for record in inputs)
        assert any(record.input_kind == "feature_spec" for record in inputs)
        assert any(record.input_kind == "label_pack" for record in inputs)
        assert any(record.input_kind == "merge_config" for record in inputs)

        aliases = catalog.list_aliases(manifest.artifact_id)
        assert [alias.alias_key for alias in aliases] == ["dataset://xau_core@1.0.0"]

        history = catalog.get_artifact_status_history(manifest.artifact_id)
        assert [event.status for event in history] == ["building", "truth_pending", "published"]

        raw_report = catalog.get_trust_report_json(manifest.artifact_id)
        assert raw_report is not None
        assert "trust_score_total" in raw_report
    finally:
        catalog.close()
