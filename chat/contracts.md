# Contract Change Log

## Contract Change Rule

A contract change is not complete until:
1. The change is logged here with old/new/impact
2. The affected modules are updated
3. At least one test verifies the new behavior
4. The raising agent has confirmed resolution in their agent log

Logging a change here without tests is not a contract change.
It is an intention. Do not confuse the two.

All public boundary changes **must** be logged here before or immediately after the change.

Human review notes do NOT belong in this file.

Cross-package imports are only allowed via:
- `mt5pipe.contracts.*`
- `mt5pipe.state.public`
- `mt5pipe.features.public`
- `mt5pipe.labels.public`
- `mt5pipe.compiler.public`

Scope guard:
- `feedbacks/latest.md` = human steering
- `chat/contracts.md` = boundary/API/schema changes only
- `chat/coordination.md` = blockers/requests/handoffs/coordination only

---

## Ownership Map

| Sector        | Owner   | Packages                                         |
|---------------|---------|--------------------------------------------------|
| contracts + state | Agent 1 | `mt5pipe/contracts/`, `mt5pipe/state/`          |
| features      | Agent 2 | `mt5pipe/features/`, `mt5pipe/labels/`           |
| compiler      | Agent 3 | `mt5pipe/compiler/`, `mt5pipe/truth/`, `mt5pipe/catalog/` |

---

## Contract Changes

<!-- Append new entries at the bottom -->

### [2026-04-04] Setup agent — initial boundary definitions

```
agent: setup
type: contract-change
module: mt5pipe.contracts
symbol: ArtifactRef, DatasetId, TrustVerdict, LineageNode
old: N/A (new)
new: Shared contract types for cross-sector communication
impact: all agents
docs_updated: yes
notes: Initial shared contracts package. Types are placeholders — agents must finalize signatures.
```

### [2026-04-04] Setup agent — boundary cleanup pass

```
agent: setup
type: contract-change
module: mt5pipe.labels.public (new), mt5pipe.features.public, mt5pipe.state.__init__, mt5pipe.compiler.__init__, mt5pipe.features.__init__, mt5pipe.labels.__init__
symbol: LabelPack, LabelService (added to features.public), all __init__ re-route through public.py
old: __init__.py exported directly from models/service internals; labels had no public.py
new: all sector __init__.py re-export from public.py only; labels.public.py added as intra-sector boundary; features.public.py re-exports labels surface for cross-sector consumers
impact: all agents — no import paths changed externally, but __init__ now routes through public
docs_updated: yes
notes: Cleanup only. No implementation changes.
```

---

<!-- TEMPLATE — copy this block for new entries:

```
[YYYY-MM-DD HH:MM]
agent: <agent_1|agent_2|agent_3>
type: contract-change
module: <module path>
symbol: <public symbol>
old: <old signature/behavior>
new: <new signature/behavior>
impact: <who is affected>
docs_updated: yes|no
notes: <short note>
```

-->

### [2026-04-04 20:05]

```
agent: agent_2
type: contract-change
module: mt5pipe.features.public, mt5pipe.labels.public
symbol: FeatureBuilder, FeatureArtifactRef, load_feature_artifact, get_default_feature_specs, resolve_feature_selectors, LabelArtifactRef, load_label_artifact, get_default_label_packs, resolve_label_pack
old: public surfaces exported only FeatureSpec/FeatureService/basic family builders and LabelPack/LabelService
new: public surfaces now expose registry helpers plus stable artifact ref/loader symbols for persisted feature and label views
impact: compiler/tests/future cross-sector consumers can depend on features.public and labels.public without reaching into registry or storage internals
docs_updated: yes
notes: loaders use StoragePaths + ParquetStore and keep label behavior unchanged
```

### [2026-04-04 20:05]

```
agent: agent_2
type: contract-change
module: mt5pipe.features.registry.defaults
symbol: disagreement.microstructure_pressure@1.0.0, event_shape.flow_shape@1.0.0, entropy.market_complexity@1.0.0
old: stable registry covered only time/session/quality/htf_context families
new: adds three stable Phase 3 machine-native feature families with explicit dependencies, warmup rows, PIT safety, and output columns
impact: dataset specs can now resolve disagreement/*, event_shape/*, and entropy/*
docs_updated: yes
notes: all three families remain M1/BuiltBar-compatible so Agent 3 can compile them without compiler redesign
```
### [2026-04-04 20:40]
agent: agent_1
type: contract-change
module: mt5pipe.contracts.artifacts, mt5pipe.contracts.state
symbol: ArtifactKind(CANONICAL_TICK, STATE_WINDOW), TickArtifactRef, StateArtifactRef, StateWindowArtifactRef, StateWindowRequest, parse_window_size
old: shared contracts only exposed generic ArtifactRef/DatasetId/TrustVerdict/LineageNode
new: state-specific artifact refs and rolling-window request contract are now first-class shared types for cross-sector state access
impact: Agent 2 can depend on typed state/tick/window refs; Agent 3 can pass/record state refs without importing state internals
docs_updated: no
notes: window sizes use compact strings like 30s/60s/5m; refs are range-based and artifact-id driven.

### [2026-04-04 20:40]
agent: agent_1
type: contract-change
module: mt5pipe.state.public
symbol: StateArtifactManifest, StateArtifactRef, StateWindowArtifactRef, StateWindowRecord, StateWindowRequest, TickArtifactRef, StateBuilder, StateService, load_state_artifact, materialize_state_windows
old: state.public only re-exported StateSnapshot and StateService
new: state.public now exposes the stable state-sector substrate for state/tick artifacts and machine-native rolling state windows
impact: Agent 2 should import state substrate from mt5pipe.state.public only; Agent 3 can keep calling StateService while the boundary is now explicit
docs_updated: no
notes: state sector no longer imports compiler/catalog internals; manifests are written by local state helpers and can still be handed to compiler catalog duck-typed.

### [2026-04-04 20:58]

```
agent: agent_3
type: contract-change
module: mt5pipe.compiler.public, mt5pipe.compiler.service, mt5pipe.compiler.models.LineageManifest.metadata
symbol: compile_dataset_spec, inspect_artifact, diff_artifacts, ArtifactInspection, ArtifactDiff, DatasetSpec.state_artifact_ref/feature_artifact_refs/label_artifact_ref
old: compiler-era dataset builds mainly assumed version-ref materialization, compiler.public did not present compile/inspect/diff as one stable surface, and artifact-ref state builds could not honor filled-row filtering from public state artifacts
new: compiler.public re-exports compile/inspect/diff dataclasses and helpers; compiler manifests/inspection metadata now carry requested_feature_selectors, feature_artifact_refs, source_modes, build_row_stats, split_row_counts, state_artifact_ref, and label_artifact_ref; artifact-ref state builds derive filled-row exclusion from public state trust flags when raw _filled markers are not present
impact: CLI/tests/other agents can stay on mt5pipe.compiler.public and artifact-backed Phase 3 datasets remain inspectable/diffable without cross-sector internal imports
docs_updated: yes
notes: config/datasets/xau_m1_nonhuman_v1.yaml is the Phase 3 example spec; real workspace execution still requires the referenced machine-native feature artifacts to be present in the local catalog/worktree
```

### [2026-04-04 20:58]

```
agent: agent_3
type: contract-change
module: mt5pipe.truth.service
symbol: TruthService.evaluate_dataset / TrustReport publish gate
old: truth evaluation only covered coarse dataset quality and did not explicitly reject the main Phase 3 artifact failure modes
new: truth gate now enforces coverage, split integrity, duplicate primary-clock rows/leakage, required feature columns, per-family missingness thresholds for time/session/quality/htf_context/disagreement/event_shape/entropy, warmup/drop-row sanity, source quality, lineage completeness, and manifest hash integrity; hard-failure codes now include dataset_coverage_failure, split_integrity_failure, leakage_or_duplicate_timestamp_failure, missing_required_feature_columns, feature_family_missingness_threshold_exceeded, warmup_or_drop_row_sanity_failure, source_quality_below_threshold, lineage_incomplete, and manifest_hash_mismatch
impact: compiler publication is deterministically blocked on bad Phase 3 artifacts and inspect/diff consumers can rely on richer trust reports
docs_updated: yes
notes: source quality now treats merge QA as a modifier on state quality instead of double-penalizing conflict behavior already reflected in state quality_score
```

### [2026-04-04 20:58]

```
agent: agent_3
type: contract-change
module: mt5pipe.cli.dataset_cmds
symbol: dataset compile-dataset, inspect-dataset, diff-dataset
old: CLI summaries omitted artifact-ref/source-mode/build-row detail and compile output did not expose explicit publish control
new: compile-dataset supports --publish/--no-publish and prints split_rows; inspect-dataset and diff-dataset emit deterministic summaries for requested feature selectors, feature artifact refs, source modes, and build row stats
impact: Phase 3 datasets can be inspected and compared from the CLI without manually opening manifest JSON
docs_updated: yes
notes: output mirrors compiler manifest metadata for inspectability and diffability
```

### [2026-04-04 21:37]

```
agent: agent_2
type: contract-change
module: mt5pipe.features.public, mt5pipe.features.registry.defaults
symbol: add_multiscale_features, multiscale.consistency@1.0.0
old: public surface exposed disagreement/event_shape/entropy as the only machine-native families; no stable multiscale selector existed
new: features.public now re-exports add_multiscale_features and the default registry now resolves multiscale/* to multiscale.consistency@1.0.0 with explicit warmup/dependencies/output columns
impact: compiler/dataset specs can consume multiscale/* without touching feature internals; Agent 3 can compile the new stable selector directly
docs_updated: yes
notes: multiscale.consistency@1.0.0 outputs trend_alignment_5_15_60, return_energy_ratio_5_60, volatility_ratio_5_60, range_expansion_ratio_15_60, and tick_intensity_ratio_5_60
```

### [2026-04-04 21:37]

```
agent: agent_2
type: contract-change
module: mt5pipe.features.disagreement, mt5pipe.features.event_shape, mt5pipe.features.entropy, mt5pipe.features.labels, mt5pipe.labels.service
symbol: Phase 3 machine-native family warmup/missingness behavior; triple_barrier_* tail semantics; label manifest metadata.label_diagnostics
old: disagreement/event_shape/entropy emitted some early-row values before warmup and could synthesize numeric outputs from missing core inputs; triple_barrier_* treated insufficient forward horizon as time-expiry 0; label manifests exposed only basic row/column metadata
new: disagreement/event_shape/entropy now null all family outputs through declared warmup rows and degrade to typed-null columns when core inputs are missing; triple_barrier_* now returns null for insufficient forward horizon and includes the full horizon endpoint; label manifests now include compact horizon/class-balance diagnostics plus exclusions
impact: compiler/truth consumers should expect cleaner warmup/tail nulls for machine-native features and labels, plus richer label artifact metadata for inspectability
docs_updated: yes
notes: label pack key and output column names stay unchanged; only tail availability semantics and manifest metadata became stricter
```
### [2026-04-04 21:38]

```
agent: agent_1
type: contract-change
module: mt5pipe.state.models.StateSnapshot, mt5pipe.state.models.StateArtifactManifest, mt5pipe.state.models.StateWindowRecord
symbol: StateSnapshot.expected_interval_ms/observed_interval_ms/source_participation_score/overlap_confidence_hint/gap_fill_flag, StateCoverageSummary, StateSourceQualitySummary, StateArtifactManifest.coverage_summary/source_quality_summary/time-range fields, StateWindowRecord warmup/completeness/gap/source-quality fields
old: state artifacts exposed basic snapshot/window structure with limited completeness and source-quality metadata
new: state artifacts now carry typed coverage and source-quality summaries plus per-snapshot/per-window completeness, participation, overlap-confidence, staleness, and gap-fill annotations suitable for wider-range machine-native builds
impact: Agent 2 can consume richer PIT-safe state/window metadata from mt5pipe.state.public; Agent 3/compiler/truth can reason about time range, coverage, and source quality from state manifests without guessing
docs_updated: no
notes: additive shape expansion only; no merge/backfill semantics changed
```

### [2026-04-04 21:38]

```
agent: agent_1
type: contract-change
module: mt5pipe.state.public
symbol: StateCoverageSummary, StateSourceQualitySummary, load_state_window_artifact
old: state.public exposed snapshot/window contracts and load_state_artifact/materialize_state_windows but not typed coverage/source-quality summaries or a public window-artifact loader
new: state.public now exports typed coverage/source-quality summary models and load_state_window_artifact for stable cross-sector access to persisted rolling windows
impact: Agent 2/3 can stay on mt5pipe.state.public for window metadata and persisted window loading
docs_updated: no
notes: public surface remains additive and boundary-clean
```

### [2026-04-04 22:44]

```
agent: agent_3
type: contract-change
module: mt5pipe.truth.service, mt5pipe.cli.dataset_cmds
symbol: TrustReport.metrics.dataset_quality, warning_reasons semantics, compile-dataset|inspect-dataset|diff-dataset trust summaries
old: truth warning reasons were coarse and sometimes duplicative; trust metrics did not surface compact dataset-quality detail; CLI summaries omitted decision_summary, rejection/warning reasons, and check-status counts
new: truth reports now include dataset_quality metrics (quality_score, total_nulls, null-columns map, constant-columns list, duplicate timestamp count) and emit more specific warning reasons for source quality, null columns, and constant columns without duplicating generic warning codes; compile/inspect/diff now print trust_decision, trust_check_counts, trust_warning_reasons, and trust_rejection_reasons, with diff exposing trust-reason deltas
impact: research users can diagnose accepted/rejected artifacts directly from compiler outputs without opening manifest/trust JSON by hand
docs_updated: yes
notes: this is a reporting hardening change only; trust hard-fail thresholds were not relaxed
```

### [2026-04-04 22:44]

```
agent: agent_3
type: contract-change
module: config/datasets/xau_m1_nonhuman_v1.yaml, config/datasets/xau_m1_core_v1.yaml
symbol: example DatasetSpec selector bundle and synchronized date range
old: example specs targeted a narrow 2026 slice; xau_m1_nonhuman_v1 pinned stale feature_artifact_refs that were no longer the stable machine-native path
new: example specs now target the wider synchronized range 2024-02-26..2024-03-01; xau_m1_nonhuman_v1 compiles from stable public selectors only and includes multiscale/* while removing explicit feature_artifact_refs
impact: compiler-facing workflows and tests now exercise the real Phase 4 checkpoint path through state_version_ref + stable selectors instead of stale artifact aliases
docs_updated: yes
notes: multiscale/* is included because the current public feature registry resolves multiscale.consistency@1.0.0 cleanly in the live workspace and focused tests
```

### [2026-04-04 23:34]

```
agent: agent_3
type: contract-change
module: mt5pipe.truth.service, mt5pipe.cli.dataset_cmds
symbol: TrustReport.metrics.quality_caveat_summary semantics, TrustReport.metrics.source_quality observability detail, compile-dataset|inspect-dataset|diff-dataset quality/trust summary lines
old: expected sparse nulls and slice-trivial constants were emitted as generic dataset warnings alongside real blockers; source-quality reporting did not distinguish merge_qa from merge_diagnostics fallback; CLI output did not surface accepted caveats, per-family caveat summaries, or source-quality context directly
new: truth reports now classify expected sparse nulls and slice-trivial constants as accepted_caveats, keep unexpected nulls/blocking constants/source-quality shortfalls in green_blockers or publication_blockers, and expose merge_diagnostics fallback metrics inside source_quality; compile/inspect/diff now print quality_caveats, source_quality_metrics, and per-family quality summaries, with diff also printing feature_families_left/right
impact: Phase 4 checkpoint review can distinguish accepted slice behavior from true blockers deterministically from CLI output; artifacts can return to green when only accepted caveats remain without hiding those caveats
docs_updated: yes
notes: trust hard-fail thresholds remain unchanged; this is a truth/reporting hardening pass rather than a publication-gate relaxation
```

### [2026-04-04 22:31]

```
agent: agent_2
type: contract-change
module: mt5pipe.labels.service
symbol: label manifest metadata.label_diagnostics.horizons_minutes/max_horizon_minutes/recommended_min_embargo_rows
old: label_diagnostics exposed per-horizon null/class-balance summaries plus purge_rows/exclusions, but did not explicitly summarize the pack-wide horizon span or recommended embargo floor
new: label_diagnostics now also includes horizons_minutes, max_horizon_minutes, and recommended_min_embargo_rows to make purge/embargo expectations explicit for compiler/inspection consumers
impact: inspect/diff/trust consumers can reason about label horizon scope and minimum safe embargo directly from label artifact metadata without inferring it from raw columns
docs_updated: yes
notes: additive metadata only; label pack key, output columns, and generation logic stay unchanged
```
### [2026-04-04 22:30]

```
agent: agent_1
type: contract-change
module: mt5pipe.state.public / mt5pipe.state.service materialize_state_windows
symbol: materialize_state_windows request-range behavior
old: when a source StateArtifactRef/TickArtifactRef spanned a wider date range than the request, state windows were built across the full source and returned anchors outside the requested date range; lineage refs also only covered the request dates
new: state windows may still use a wider source artifact for PIT-safe warmup context, but emitted window anchors are filtered to the requested date range, request dates must lie within the source ref range, and lineage/input refs cover the full source range actually used
impact: Agent 2 can rely on state-window artifacts matching requested anchor dates while preserving prior-context warmup; Agent 3 can rely on state-window lineage covering the actual source partitions used
docs_updated: no
notes: no public symbol additions; boundary behavior is stricter and more deterministic for wider-range requests
```

### [2026-04-04 23:33]

```
agent: agent_2
type: contract-change
module: mt5pipe.features.registry.defaults
symbol: htf_context.standard_context@1.0.0, disagreement.microstructure_pressure@1.0.0
old: stable htf_context/* exposed higher-timeframe *_tick_count columns; stable disagreement/* exposed spread_divergence_proxy_bps, conflict_burst_15, staleness_asymmetry_15, and disagreement_entropy_30 alongside the core pressure fields
new: stable htf_context/* now excludes *_tick_count from production output columns; stable disagreement/* is narrowed to mid_divergence_proxy_bps, disagreement_pressure_bps, disagreement_zscore_60, and disagreement_burst_15
impact: compiler/truth/dataset specs keep the same family selectors, but published feature artifacts for the stable selector set now omit the null-heavy HTF tick-count columns and the slice-trivial disagreement columns
docs_updated: yes
notes: builders still compute the broader internal disagreement surface; only the stable registry contract was tightened for the current nonhuman path
```

### [2026-04-04 23:33]

```
agent: agent_2
type: contract-change
module: mt5pipe.features.disagreement, mt5pipe.features.event_shape, mt5pipe.labels.service
symbol: disagreement/* warmup semantics, event_shape/* warmup semantics, metadata.label_diagnostics.constant_output_columns
old: disagreement/event_shape cleanup still effectively assumed family-wide warmup in tests, and label diagnostics did not explicitly report constant output columns
new: disagreement/* and event_shape/* now treat warmup at the column level (instantaneous columns can materialize immediately while rolling columns stay null until ready); label diagnostics now include constant_output_columns for the current label artifact
impact: compiler/truth consumers should expect earlier availability for non-rolling event/disagreement columns and can classify trivial labels directly from manifest metadata
docs_updated: yes
notes: no label pack keys or column names changed
```

### [2026-04-05 00:05]

```
agent: agent_2
type: contract-change
module: mt5pipe.features.registry.models, mt5pipe.features.registry.defaults, mt5pipe.labels.registry.models, mt5pipe.labels.registry.defaults
symbol: FeatureSpec.ablation_group/trainability_tags, LabelPack.qa_policy_ref/ablation_group/trainability_tags/target_groups/tail_policy, stable registry defaults
old: registry entries exposed status/tags plus core shape, but had no explicit experiment-group metadata or label-pack trainability metadata
new: feature specs now carry ablation_group and trainability_tags; label packs now carry qa_policy_ref, ablation_group, trainability_tags, target_groups, and tail_policy; stable defaults populate those fields for the production selector set and core_tb_volscaled@1.0.0
impact: compiler/catalog/training consumers can group stable features by ablation bucket and inspect label-pack training assumptions without inferring them from names alone
docs_updated: yes
notes: additive only; no existing feature keys, label-pack keys, or output column names changed
```

### [2026-04-05 00:05]

```
agent: agent_2
type: contract-change
module: mt5pipe.features.service, mt5pipe.labels.service
symbol: feature_view manifest metadata.trainability_diagnostics, label_view metadata.label_diagnostics/trainability fields, direction_threshold_bps service behavior
old: feature artifacts exposed only basic row/column/output metadata; label artifacts exposed horizon/tail/class-balance diagnostics but not experiment-readiness summaries, target-distribution summaries, or pack-level trainability metadata; label service ignored direction-threshold pack parameters
new: feature artifacts now publish family/status/ablation/tags plus trainability_diagnostics with post-warmup coverage, complete-row ratio, constant/low-variation/null-heavy columns, and warning reasons; label artifacts now publish pack-level trainability metadata, target-distribution summaries, degenerate-horizon warnings, and direction-threshold/barrier settings, and label service honors parameters.direction_threshold_bps when materializing direction_* targets
impact: Agent 3 and downstream training workflows can inspect whether a feature family or label horizon is trainable directly from artifact metadata, and future label packs can tighten direction targets without changing generator plumbing
docs_updated: yes
notes: default core_tb_volscaled@1.0.0 keeps direction_threshold_bps=0.0, so current output semantics stay unchanged unless a pack explicitly opts in
```
### [2026-04-04 23:32]

```
agent: agent_1
type: contract-change
module: mt5pipe.state.service / mt5pipe.state.public load_state_artifact, load_state_window_artifact, materialize_state
symbol: persisted state/state-window idempotence and canonical-quality-backed source_quality behavior
old: repeated state or state-window materialization could append duplicate persisted rows because state partitions were not rewritten idempotently and loaders returned concatenated duplicates; bar-backed state quality/source_quality_hint relied only on coarse bar-level heuristics even when canonical tick quality evidence was available
new: state and state-window partition writes are reset before rewrite and public loaders deduplicate on stable keys, making persisted loads idempotent; bar-backed state quality_score/source_quality_hint now prefer per-bar canonical tick quality/conflict/dual-source evidence when available
impact: Agent 2 and Agent 3 can rely on stable persisted state/state-window loads and more informative source-quality inputs on the current nonhuman dataset path without importing truth/compiler internals
docs_updated: no
notes: additive behavioral hardening only; no new public symbols or contract fields were introduced
```
### [2026-04-05 00:06]

```
agent: agent_1
type: contract-change
module: mt5pipe.contracts.state
symbol: TickArtifactRef, StateArtifactRef, StateWindowArtifactRef, StateWindowRequest normalization behavior
old: typed state refs/requests preserved incoming symbol, clock, and window-size casing verbatim, so callers could accidentally depend on xauusd vs XAUUSD or m1 vs M1 differences
new: state refs/requests now normalize symbol -> upper-case, clock -> upper-case, window_size -> normalized compact lower-case form, and trim state_version/source_artifact_id strings
impact: Agent 2/3 and future callers can treat state refs/requests as case-stable contracts; later symbol additions such as XAGUSD/USOIL will not require casing-specific workarounds at the state boundary
docs_updated: no
notes: additive hygiene only; no path semantics or merge/backfill behavior changed
```

### [2026-04-05 00:06]

```
agent: agent_1
type: contract-change
module: mt5pipe.state.models, mt5pipe.state.public, mt5pipe.state.service
symbol: StateReadinessSummary, StateIntervalReadinessSummary, StateArtifactManifest.readiness_summary/daily_readiness_rollups/session_readiness_rollups, StateMaterializationResult.readiness_summary, StateWindowMaterializationResult.readiness_summary
old: state manifests/results exposed coverage_summary and source_quality_summary, but did not provide typed range-level readiness metadata, daily/session rollups, or window-availability summaries for downstream truth/training consumers
new: state artifacts now expose typed readiness summaries with effective coverage, gap burden, overlap/source reliability bands, readiness bands, and window-availability ratios; manifests/results also carry daily and per-day-session readiness rollups through mt5pipe.state.public
impact: Agent 2 and Agent 3 can rely on stable state-side training-readiness metadata without importing state internals or re-deriving rollups downstream
docs_updated: no
notes: additive shape expansion only; current public symbols remain stable and no new service entrypoints were added
```

### [2026-04-05 00:35]

```
agent: agent_3
type: contract-change
module: mt5pipe.compiler.models, mt5pipe.compiler.manifest, mt5pipe.compiler.public
symbol: ArtifactKind(experiment, model), ArtifactStatus(trial), ExperimentSpec, LineageManifest.experiment_spec_ref, load_experiment_spec, run_experiment_spec, inspect_experiment, inspect_model, ExperimentRunResult, ExperimentInspection, ModelInspection
old: compiler public surface only covered dataset compilation/inspection/diff; manifests only linked dataset specs; no typed experiment spec or compiler-owned experiment/model artifact surface existed
new: compiler now exposes a first-class training/evaluation boundary with ExperimentSpec plus public run/inspect helpers for experiment/model artifacts, and manifests can link experiment specs through experiment_spec_ref while artifact kinds now include experiment/model and model status can be trial
impact: compiler-facing consumers can run trust-gated experiments and inspect experiment/model lineage without importing compiler internals; catalog/manifests can represent the first training registry layer cleanly
docs_updated: yes
notes: the first example spec is config/experiments/xau_m1_nonhuman_direction_nb_v1.yaml and targets dataset://xau_m1_nonhuman@1.0.0 with direction_60m
```

### [2026-04-05 00:35]

```
agent: agent_3
type: contract-change
module: mt5pipe.catalog.sqlite, mt5pipe.storage.paths, mt5pipe.cli.train_cmds
symbol: experiment_specs/training_runs catalog tables, experiment:// and model:// alias lifecycle, compact experiment/model storage roots, train run-experiment|inspect-experiment|inspect-model
old: catalog only tracked dataset specs/build runs/truth-backed artifacts; storage path conventions only covered datasets/manifests/truth; there was no compiler-facing training CLI
new: catalog now persists experiment specs and training-run lifecycle metadata linked to dataset/model artifact ids; experiment/model artifacts are registered through the shared artifact catalog and resolved via explicit experiment:// and model:// aliases; storage roots for experiment/model artifacts use compact path segments to stay Windows-safe; the CLI now exposes minimal train commands for running and inspecting trust-gated experiments
impact: training/evaluation runs are reproducible, inspectable, and lineage-linked through the existing catalog/manifests while remaining usable on Windows workspaces
docs_updated: yes
notes: training still hard-gates on dataset trust acceptance and currently ships one deterministic baseline model family, gaussian_nb_binary@1.0.0
```

### [2026-04-05 15:31]

```
agent: agent_3
type: contract-change
module: mt5pipe.compiler.models, mt5pipe.truth.service, config/datasets/xau_m1_core_v1.yaml, config/datasets/xau_m1_nonhuman_v1.yaml
symbol: DatasetSpec.required_raw_brokers, require_synchronized_raw_coverage, require_dual_source_overlap, min_dual_source_ratio; source_quality trust enforcement
old: dataset specs could describe the current ML path without explicitly declaring raw-broker requirements, and truth could still accept/publish a single-source dataset even when the intended research path depended on synchronized multi-broker quality
new: dataset specs can now declare required raw brokers, synchronized raw coverage, dual-source overlap, and a minimum effective dual-source ratio; truth treats those as hard publication requirements and reports missing/asymmetric raw dates plus dual-source failures directly in source_quality metrics/reasons
impact: compiler-owned ML specs can now prove and enforce that the production dataset window is genuinely dual-broker rather than merely trusting merge artifacts opportunistically
docs_updated: yes
notes: current production specs moved to the synchronized live slice 2026-03-30..2026-04-02 and require broker_a + broker_b with min_dual_source_ratio=0.10
```

### [2026-04-05 15:31]

```
agent: agent_3
type: contract-change
module: mt5pipe.storage.paths, mt5pipe.state.service, mt5pipe.features.service, mt5pipe.labels.service, mt5pipe.state.public, mt5pipe.features.public, mt5pipe.labels.public
symbol: immutable upstream artifact_uri / artifact-scoped parquet roots for state, feature_view, label_view
old: state, feature, and label manifests pointed at mutable logical roots, so upstream lineage was logically named but not content-addressed enough for reproducible reloads after reruns
new: manifests now point at compact artifact-scoped parquet roots keyed by artifact id/version, writers mirror outputs into those immutable roots, and public loaders prefer artifact-scoped partitions before falling back to legacy logical roots
impact: dataset manifests and downstream training artifacts now anchor lineage to stable upstream parquet snapshots without breaking existing logical-resolution flows
docs_updated: yes
notes: legacy logical roots are still written for compatibility; new loaders resolve immutable roots whenever artifact_id is available
```

### [2026-04-05 15:31]

```
agent: agent_3
type: contract-change
module: mt5pipe.ingestion.ticks, mt5pipe.backfill.engine, mt5pipe.merge.canonical, mt5pipe.quality.merge_qa
symbol: raw ingest row accounting, gap-fill behavior, canonical per-day diagnostics canonicalization
old: raw tick ingest totals reflected post-write file sizes rather than net-new deduped rows, historical day gap fills could be skipped when checkpoints were already ahead, and merge diagnostics/QA could accumulate duplicate per-day files across reruns
new: tick storage returns net-new rows added after dedup and registers written partitions in the file manifest; day-level backfill gap fills run even when checkpoints are ahead while preserving monotonic checkpoints; merge diagnostics and merge QA now rewrite one canonical summary file per day on rerun
impact: raw completeness claims, rerun idempotence, and daily merge observability are now auditable from actual stored artifacts instead of inflated counters or duplicated summaries
docs_updated: yes
notes: canonical tick parquet dedup now also keys on ts_msc/ts_utc/bar_start/anchor_ts_utc where present to prevent append-style rerun duplication
```

### [2026-04-05 18:05]

```
agent: agent_1
type: contract-change
module: machinelearning.data
symbol: ColumnSchema, DEFAULT_SCHEMA, RobustFeatureNormalizer, AphelionDataset, AphelionDataModule
old: no Phase 6 machinelearning.data boundary existed
new: machinelearning.data now exposes the frozen Phase 6 parquet schema, a train-only median/IQR feature normalizer with JSON save/load, a parquet-backed sliding-window dataset, and a Lightning-compatible datamodule with nested-target collation
impact: Agent 2 and Agent 3 can import Phase 6 data inputs from machinelearning.data only, without touching mt5pipe internals or re-deriving the published dataset contract
docs_updated: no
notes: the data layer reads published parquet splits only, derives past/future/static/target groups from schema.py, and encodes classification targets with {-1,0,1}->{0,1,2} plus -100 for missing labels
```

### [2026-04-05 17:55]

```
agent: agent_2
type: contract-change
module: machinelearning.models.__init__, machinelearning.models.base, machinelearning.models.blocks, machinelearning.models.tft
symbol: AphelionModel, ModelOutput, AphelionTFT, HORIZONS, N_CLASSES, QUANTILES
old: machinelearning/models public model surface did not exist in this worktree
new: Phase 6 now exposes a stable neural-model boundary with ModelOutput convenience helpers, reusable TFT building blocks, and AphelionTFT as the first production model family
impact: Agent 3 training code can depend on machinelearning.models only, consume a fixed multi-head ModelOutput contract, and inspect per-branch VSN weights without importing model internals
docs_updated: yes
notes: model code stays self-contained inside machinelearning/ and does not import mt5pipe or machinelearning.training internals
```

### [2026-04-05 18:03]

```
agent: agent_1
type: contract-change
module: machinelearning.data.__init__, machinelearning.data.schema, machinelearning.data.normalizer, machinelearning.data.dataset, machinelearning.data.datamodule
symbol: ColumnSchema, DEFAULT_SCHEMA, RobustFeatureNormalizer, AphelionDataset, AphelionDataModule
old: machinelearning/data public data-loading surface did not exist in this worktree
new: Phase 6 now exposes a standalone parquet-only data boundary with a frozen column schema, train-only robust median/IQR normalization, split-aware sliding-window dataset loading, and a Lightning-style datamodule/dataloader surface
impact: Agent 3 can depend on machinelearning.data only for schema cardinalities, normalization reloads, last-bar target extraction, optional _filled masking, and train/val/test dataloaders backed by the published dataset parquet artifact
docs_updated: yes
notes: data layer stays isolated from mt5pipe, machinelearning.models, and machinelearning.training; categorical targets remap {-1,0,1}->{0,1,2} with -100 ignore_index handling for missing labels
```

### [2026-04-05 20:01]

```
agent: agent_2
type: contract-change
module: machinelearning.models.__init__, machinelearning.models.interpret, machinelearning.models.blocks
symbol: VSNInterpreter, AttentionInspector, QuantileHead ordered-quantile behavior
old: machinelearning.models exposed only the core model surface, and return quantile heads emitted unconstrained raw quantiles with no guarantee that higher quantiles were >= lower quantiles
new: machinelearning.models now publicly exports VSNInterpreter and AttentionInspector for post-forward interpretability, and QuantileHead now emits nondecreasing quantiles by construction so return_preds and derived intervals preserve quantile ordering semantics
impact: Agent 3 can inspect learned past-feature importance directly from ModelOutput.vsn_weights and can rely on ordered return quantiles for calibration/interval analysis without importing model internals
docs_updated: no
notes: AttentionInspector currently returns None for temporal attention summaries because ModelOutput does not yet store attn_weights; that capability is logged as a coordination note for a future round
```

### [2026-04-05 20:00]

```
agent: agent_2
type: contract-change
module: machinelearning.models.base.ModelOutput
symbol: attn_weights
old: ModelOutput exposed prediction heads, encoder_hidden, and VSN weights only, so TemporalSelfAttention weights were dropped during forward passes and AttentionInspector could only return None for temporal attention summaries
new: ModelOutput now includes attn_weights: dict[str, Tensor] | None, with AphelionTFT storing TemporalSelfAttention weights under attn_weights["past"] shaped [B, n_heads, T, T]
impact: AttentionInspector can now return real [T, T] mean attention and [T] last-timestep attention tensors, and downstream tests can verify the temporal-attention contract directly
docs_updated: yes
notes: verified by test_attention_inspector_mean_shape, test_attention_inspector_last_timestep, test_model_output_attn_weights_stored, and the end-to-end model forward smoke test
```

```
agent: agent_1
type: contract-change
module: machinelearning.data.__init__, machinelearning.data.walkforward, machinelearning.data.inference
symbol: WalkForwardSplitter, WalkForwardResult, InferenceLoader
old: machinelearning.data exposed training-time schema/dataset/datamodule utilities only, with no public walk-forward evaluation helper or inference-time normalizer loader
new: machinelearning.data now also exposes a pure-Polars expanding-window walk-forward splitter with embargo-aware temporal folds and aggregate fold metrics, plus an inference loader that reloads {run_name}_normalizer.json and converts recent bars into the single-sample batch dict expected by the TFT
impact: Agent 3 can evaluate models on strict temporal folds without GPU dependencies and can rebuild inference-time batches from saved normalizer statistics without importing machinelearning.models or machinelearning.training internals
docs_updated: no
notes: WalkForwardSplitter stays PyTorch-free, embargo rows are documented as protection against forward-horizon label leakage, and InferenceLoader preserves training-time fill/mask semantics while omitting targets at inference
```

### [2026-04-05 20:03]

```
agent: agent_3
type: contract-change
module: machinelearning.training.__init__, machinelearning.training.losses, machinelearning.training.module, machinelearning.training.train
symbol: quantile_loss, classification_loss, regression_loss, ic_loss, LearnedTaskWeights, AphelionLightningModule, train
old: machinelearning/training public training surface did not exist in this worktree
new: Phase 6 now exposes a standalone training boundary with multi-head financial loss functions, learned uncertainty task weighting, a Lightning-style module that logs per-horizon classification/IC metrics, and a train() entry point that composes the public data/model surfaces into checkpointed research runs
impact: the full deep-learning stack can now depend on machinelearning.training only, while keeping training logic isolated from mt5pipe internals and from machinelearning.data/models internals
docs_updated: yes
notes: classification_loss uses focal weighting around the dominant flat class, ic_loss regularizes the 60m return median toward ranking signal, and train() persists the saved normalizer path beside checkpoints for inference reuse
```

### [2026-04-06 02:38]

```
agent: agent_2
type: contract-change
module: machinelearning.regime.__init__, machinelearning.regime.features, machinelearning.regime.detector, machinelearning.regime.moe
symbol: RegimeFeatureExtractor, RegimeState, RegimeDetector, GatingNetwork, MixtureOfExperts
old: machinelearning.regime did not exist in this worktree, so there was no public regime-detection or regime-routed model boundary for downstream signal generation
new: Phase 7 now exposes a standalone regime package with fixed-column regime feature extraction, a save/load Gaussian-HMM wrapper that emits soft four-regime states, and a regime-aware mixture-of-experts router over AphelionTFT experts
impact: Agent 3 can consume machinelearning.regime only to derive regime probabilities, route encoder_hidden representations through soft expert weights, and monitor expert utilization without importing machinelearning.data internals, machinelearning.models internals, mt5pipe, or machinelearning.training
docs_updated: yes
notes: RegimeDetector.fit/load require hmmlearn at runtime, but MixtureOfExperts accepts any [B,4] regime_probs tensor so the signal layer can be integrated before the real fitted detector exists
```

### [2026-04-06 02:58]

```
agent: agent_2
type: contract-change
module: machinelearning.regime.__init__, machinelearning.regime.labeler, machinelearning.models.__init__, machinelearning.models.ablation
symbol: RegimeLabeler, AblationConfig, AblationResult, AblationRunner, STANDARD_ABLATIONS
old: machinelearning.regime stopped at online regime classification/MoE routing and machinelearning.models had no public zero-retrain feature-ablation surface
new: Phase 7 extension adds an immutable time_utc-keyed regime-label sidecar writer plus a model-side ablation runner that zeros selected past-feature families and summarizes how VSN importance redistributes without retraining
impact: Agent 1 can precompute regime sidecar parquet labels as soon as the 90-day dataset is compiled, and Agent 3 can run systematic post-training family ablations through machinelearning.models only to verify the disagreement edge is actually being used
docs_updated: yes
notes: RegimeLabeler outputs time_utc, regime_dominant, regime_trending, regime_mean_rev, regime_volatile, regime_quiet, and regime_confidence. AblationRunner operates on DEFAULT_SCHEMA.past_observed / batch['past_features']; under the current schema session/* lives in future_known and therefore remains outside this first zero-retrain ablation pass.
```

### [2026-04-06 02:50]

```
agent: agent_3
type: contract-change
module: machinelearning.signal.__init__, machinelearning.signal.records, machinelearning.signal.conformal, machinelearning.signal.sizing, machinelearning.signal.publisher
symbol: SignalRecord, ConformalCalibrator, KellyPositionSizer, SignalPublisher
old: no public Phase 7 signal boundary existed between machinelearning.models outputs and downstream risk/backtest consumers
new: machinelearning.signal now exposes an immutable research-to-risk signal contract, split-conformal interval calibration with persisted q_hat state, fractional Kelly sizing based on conformal intervals and data quality, and a SignalPublisher that converts ModelOutput + regime probabilities + bar metadata into a fully populated SignalRecord
impact: downstream paper-trading/backtest consumers can consume model predictions through machinelearning.signal only, without importing machinelearning.models internals, machinelearning.regime internals, machinelearning.training, or mt5pipe
docs_updated: no
notes: SignalPublisher accepts either the real RegimeState surface or any compatible [4]-probability vector, which keeps integration unblocked before a fitted detector artifact exists
```

### [2026-04-06 02:50]

```
agent: agent_3
type: contract-change
module: APH.backtest.__init__, APH.backtest.engine, APH.backtest.metrics, APH.backtest.report
symbol: BacktestEngine, BacktestResult, BacktestMetrics, compute_metrics, BacktestReport
old: no public Phase 7 backtest boundary existed for turning SignalRecord histories and realized returns into comparable strategy metrics
new: APH.backtest now exposes a vectorized signal-quality backtest engine with fixed-notional spread-cost PnL, a persisted BacktestResult container, baseline-comparable performance metrics, and a printable/json report surface for checkpoint summaries
impact: Agent 3 and later evaluation/reporting workflows can assess tradeability through APH.backtest only, using SignalRecord lists plus realized returns without importing training internals or re-implementing strategy metrics ad hoc
docs_updated: no
notes: the Phase 7 engine intentionally stays simple and vectorized for signal validation rather than realistic execution simulation; the real end-to-end pass will layer actual model outputs and a fitted regime detector on top of this stable boundary
```

## Bugfix Notes

### [2026-04-06 15:46]

```
agent: agent_3
type: bugfix-note
module: mt5pipe.truth.service
scope: internal bugfix only; no public contract or threshold change
bug: source-quality required-date accounting treated Saturday/Sunday as required synchronized raw coverage days, which falsely rejected dual-broker XAUUSD datasets over multi-week ranges
fix: weekend dates are now excluded with weekday() < 5 in both required raw-coverage accounting and source-observability day accounting; added test_synchronized_coverage_excludes_weekends to lock the behavior
impact: dataset://xau_m1_nonhuman_90d@1.0.0 now publishes correctly on the real 2026-01-08..2026-04-02 range without weakening any truth gates
docs_updated: yes
tests: pytest data/tests/test_truth_core.py -q -> 7 passed
```
