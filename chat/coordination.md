# Coordination Log

Use this file for blockers, requests, progress updates, and handoffs between agents.

Use this file to log human-feedback conflicts when `feedbacks/latest.md` conflicts with the active task/prompt.

Scope guard:
- `feedbacks/latest.md` = human steering
- `chat/contracts.md` = boundary/API/schema changes only
- `chat/coordination.md` = blockers/requests/handoffs/coordination only

## Open Items

This section tracks all cross-agent gaps that are open.
When an item is resolved, move it to ## Resolved with the fix date.
An item is not resolved until the fix is verified in tests.

Template:
[YYYY-MM-DD HH:MM]
item: <one-line description>
raised_by: agent_N
blocks: agent_N [if another agent is waiting on this]
resolution: <empty until resolved>

None.

## Resolved

[2026-04-06 12:00]
item: attention weights gap left unimplemented after model interpretability landed
raised_by: agent_2
blocks: agent_2
resolution: ModelOutput now stores attn_weights["past"], AphelionTFT forwards TemporalSelfAttention weights, AttentionInspector returns real tensors, and test coverage verifies the contract end to end.

[2026-04-06 12:00]
item: train.py DDP and BF16 path existed without independent verification or early artifact-dir validation
raised_by: agent_3
blocks: agent_3
resolution: train.py now exposes build_trainer() and validate_artifact_dir(), BF16 is GPU-gated, CPU fallback is explicit and logged, and targeted tests verify the configuration path without requiring GPUs.

---

## Active Blockers

_None_

---

## Log

<!-- Append new entries at the bottom -->

### [2026-04-04] Setup agent — structure ready

```
agent: setup
type: update
area: contracts
summary: Created boundary structure (contracts, public.py surfaces, internal/ packages). Ready for parallel development.
needs: Agent 1/2/3 to populate their sectors
files: mt5pipe/contracts/, mt5pipe/state/public.py, mt5pipe/features/public.py, mt5pipe/compiler/public.py, docs/parallel_dev_architecture.md
```

---

<!-- TEMPLATE — copy this block for new entries:

```
[YYYY-MM-DD HH:MM]
agent: <agent_1|agent_2|agent_3>
type: update|blocker|request|handoff
area: state|features|compiler|contracts
summary: <short summary>
needs: <optional>
files: <optional list>
```

-->
### [2026-04-04] Agent 2 - Phase 3 feature families ready

```
agent: agent_2
type: handoff
area: features
summary: Stable selectors disagreement/*, event_shape/*, and entropy/* are registered and tested. Public feature/label loaders and artifact refs are available via features.public and labels.public.
needs: Agent 3 can include these selectors in DatasetSpec builds once mt5pipe.compiler.service is restored/importable in this worktree.
files: mt5pipe/features/registry/defaults.py, mt5pipe/features/disagreement/, mt5pipe/features/event_shape/, mt5pipe/features/entropy/, mt5pipe/features/public.py, mt5pipe/labels/public.py, tests/test_features_phase3.py
```
### [2026-04-04] Agent 1 - compiler public import blocker

```
agent: agent_1
type: blocker
area: compiler
summary: state/contracts tests are green, but global boundary import test currently fails because mt5pipe.compiler.public imports mt5pipe.compiler.service and that module is absent in this worktree.
needs: Agent 3 to restore or re-export compiler service boundary symbols so mt5pipe.compiler.public is importable again.
files: mt5pipe/compiler/public.py
```
### [2026-04-04] Agent 1 - state substrate ready for feature work

```
agent: agent_1
type: handoff
area: state
summary: published typed TickArtifactRef/StateArtifactRef/StateWindowArtifactRef + StateWindowRequest in mt5pipe.contracts, and expanded mt5pipe.state.public with StateWindowRecord/load_state_artifact/materialize_state_windows. Rolling state windows and canonical tick state are implemented and tested.
needs: Agent 2 can build against mt5pipe.state.public now. Agent 3 only needs to restore compiler.public importability in this worktree.
files: mt5pipe/contracts/state.py, mt5pipe/state/public.py, mt5pipe/state/service.py, mt5pipe/state/models.py
```

### [2026-04-04] Agent 3 - Phase 3 compiler/truth handoff

```
agent: agent_3
type: handoff
area: compiler
summary: Compiler/truth/catalog/CLI path is green for artifact-backed Phase 3 datasets. Artifact-backed state reuse now honors exclude:filled_rows through public state trust flags, and focused tests cover compile success, truth rejection, inspect/diff, and catalog lifecycle.
needs: The real workspace still only exposes stable feature view directories under local_data/pipeline_data/feature_views and currently has no catalog.db. For config/datasets/xau_m1_nonhuman_v1.yaml to compile end-to-end outside the tests, Agent 2 machine-native feature artifacts (disagreement/event_shape/entropy) must exist and be registered under the refs used by the spec, or the feature registry implementation must be present in this worktree so the compiler can materialize them.
files: mt5pipe/compiler/service.py, mt5pipe/truth/service.py, mt5pipe/cli/dataset_cmds.py, config/datasets/xau_m1_nonhuman_v1.yaml, tests/test_compiler.py
```

### [2026-04-04] Agent 2 - Phase 4 feature and label handoff

```
agent: agent_2
type: handoff
area: features
summary: disagreement/*, event_shape/*, and entropy/* now enforce family-wide warmup nulls and typed-null degradation on missing core inputs. Added stable multiscale/* via multiscale.consistency@1.0.0. Label tail semantics are stricter: triple_barrier_* is null when the forward horizon is unavailable, and label manifests now carry horizon/class-balance diagnostics.
needs: Agent 3 can safely consume multiscale/* in DatasetSpec selectors and should expect label tail nulls plus metadata.label_diagnostics on label manifests/inspections.
files: mt5pipe/features/disagreement/builders.py, mt5pipe/features/event_shape/builders.py, mt5pipe/features/entropy/builders.py, mt5pipe/features/multiscale/, mt5pipe/features/registry/defaults.py, mt5pipe/features/public.py, mt5pipe/features/service.py, mt5pipe/features/labels.py, mt5pipe/labels/service.py, tests/test_features_phase4.py, tests/test_labels.py
```
### [2026-04-04 21:38] Agent 1 - Phase 4 state metadata handoff

```
agent: agent_1
type: handoff
area: state
summary: Phase 4 state artifacts now expose typed coverage/source-quality summaries plus per-snapshot and per-window completeness, gap-fill, staleness, participation, and overlap-confidence metadata. State sector also localized timeframe/weekend-gap helpers, so state no longer imports mt5pipe.bars or mt5pipe.quality internals.
needs: Agent 2 can consume new window metadata from mt5pipe.state.public only. Agent 3 can read manifest.coverage_summary/source_quality_summary and manifest time-range fields without deriving them downstream.
files: mt5pipe/state/models.py, mt5pipe/state/public.py, mt5pipe/state/service.py, mt5pipe/state/internal/bar_support.py, mt5pipe/state/internal/quality.py, mt5pipe/state/internal/windows.py, tests/test_state_windows.py, tests/test_state_boundaries.py
```

### [2026-04-04 22:44] Agent 3 - Phase 4 checkpoint note

```
agent: agent_3
type: handoff
area: compiler
summary: Stable selector-only Phase 4 checkpoint path is now locked in. config/datasets/xau_m1_nonhuman_v1.yaml compiles/publishes through state_version_ref + feature_selectors only, over 2024-02-26..2024-03-01, with multiscale/* included. Compiler/truth/catalog/CLI focused tests are green and inspect/diff now surface trust decision/check-count/reason summaries directly.
needs: Checkpoint is yellow rather than fully green because the live accepted artifact still reports source_quality=62.87 (< preferred 75.00), nulls in HTF/event-related columns, and several slice-specific constant columns. Agent 1 can improve the checkpoint most by raising state/source-quality on this synchronized range; Agent 2 can review whether the remaining constant/null machine-native columns are expected for the slice or should be tightened in the next phase.
files: config/datasets/xau_m1_nonhuman_v1.yaml, config/datasets/xau_m1_core_v1.yaml, mt5pipe/truth/service.py, mt5pipe/cli/dataset_cmds.py, tests/test_compiler.py, tests/test_truth_core.py, tests/test_dataset_cli_compat.py
```

### [2026-04-04 22:31] Agent 2 - checkpoint selector confirmation

```
agent: agent_2
type: handoff
area: features
summary: Added a checkpoint compile test through mt5pipe.compiler.public that publishes successfully with selectors time/session/quality/htf_context/disagreement/event_shape/entropy/multiscale on seeded XAUUSD M1 data. Label diagnostics now surface horizon span and recommended embargo floor explicitly.
needs: Agent 3 can treat disagreement/*, event_shape/*, entropy/*, and multiscale/* as stable on the registry-materialized path. For the current config/datasets/xau_m1_nonhuman_v1 artifact-ref path, multiscale/* is not required; keep it excluded unless you intentionally switch the spec to registry materialization.
files: tests/test_features_phase4_checkpoint.py, mt5pipe/labels/service.py, tests/test_features_phase4.py
```
### [2026-04-04 22:30] Agent 1 - Phase 4 continuation state-window range fix

```
agent: agent_1
type: handoff
area: state
summary: Audited the nonhuman dataset path and fixed the remaining state-side reliability issue in rolling windows. State windows now preserve prior-source warmup context but only emit anchors within the requested date range, and they reject requests that fall outside the source artifact range.
needs: Agent 2 can assume requested window artifacts no longer leak anchors from earlier source dates. Agent 3 can assume window lineage/input refs cover the full source range actually used for warmup.
files: mt5pipe/state/service.py, tests/test_state_windows.py
```

### [2026-04-04 23:34] Agent 3 - Phase 4 cleanup checkpoint

```
agent: agent_3
type: handoff
area: compiler
summary: Phase 4 cleanup is now green on the live workspace. xau_m1_nonhuman_v1 compiles, publishes, inspects, and diffs cleanly over 2024-02-26..2024-03-01 with no trust warnings. Truth now separates accepted caveats from real blockers, and CLI summaries surface those caveats plus source-quality context directly.
needs: No immediate blocker for Agent 1 or Agent 2. Remaining accepted caveats are slice-trivial constants only: base/source conflict_count-style columns, 2/3 quality outputs on this slice, and one 5m label column. Current source_quality is 77.2389, above the preferred 75.00 threshold, so it is no longer a checkpoint blocker.
files: mt5pipe/truth/service.py, mt5pipe/cli/dataset_cmds.py, tests/test_truth_core.py, tests/test_dataset_cli_compat.py, config/datasets/xau_m1_nonhuman_v1.yaml
verification: pytest tests/test_compiler.py tests/test_truth_core.py tests/test_catalog.py tests/test_dataset_cli_compat.py -q -> 18 passed; python -m mt5pipe.cli.app dataset compile-dataset --spec config/datasets/xau_m1_core_v1.yaml --publish -> published accepted 97.72; python -m mt5pipe.cli.app dataset compile-dataset --spec config/datasets/xau_m1_nonhuman_v1.yaml --publish -> published accepted 97.72; inspect/diff by dataset:// refs show zero warnings, source_quality=77.2389, and accepted_caveats only
```

### [2026-04-04 23:33] Agent 2 - stable selector cleanup handoff

```
agent: agent_2
type: handoff
area: features
summary: Cleaned the stable selector surface for the nonhuman path without changing family names. htf_context/* no longer publishes *_tick_count columns, disagreement/* now publishes only mid_divergence_proxy_bps, disagreement_pressure_bps, disagreement_zscore_60, and disagreement_burst_15, and event/disagreement warmup is now column-level rather than blanket family-wide.
needs: Agent 3 can keep using time/session/quality/htf_context/disagreement/event_shape/entropy/multiscale selectors as stable. Do not expect HTF tick-count columns in stable feature artifacts, and treat label manifest metadata.label_diagnostics.constant_output_columns as the source of truth for slice-trivial labels.
files: mt5pipe/features/registry/defaults.py, mt5pipe/features/disagreement/builders.py, mt5pipe/features/event_shape/builders.py, mt5pipe/labels/service.py, tests/test_features_phase3.py, tests/test_features_phase4.py
```

### [2026-04-05 00:05] Agent 2 - Phase 5 trainability metadata handoff

```
agent: agent_2
type: handoff
area: features
summary: Added experiment-readiness metadata without expanding the family surface. Stable FeatureSpec entries now expose ablation_group/trainability_tags, stable LabelPack entries now expose qa_policy_ref/ablation_group/trainability_tags/target_groups/tail_policy, feature-view manifests publish post-warmup trainability diagnostics, and label-view manifests publish target-distribution and degenerate-horizon diagnostics.
needs: Agent 3 can treat feature manifest metadata.trainability_diagnostics and label metadata.label_diagnostics as stable inspection surfaces. core_tb_volscaled@1.0.0 still has the same key/output columns, but label service now honors parameters.direction_threshold_bps if a future pack opts into thresholded direction targets.
files: mt5pipe/features/registry/models.py, mt5pipe/features/registry/defaults.py, mt5pipe/features/service.py, mt5pipe/labels/registry/models.py, mt5pipe/labels/registry/defaults.py, mt5pipe/labels/service.py, tests/test_features_phase5.py
```
### [2026-04-04 23:32] Agent 1 - Phase 4 cleanup state quality handoff

```
agent: agent_1
type: handoff
area: state
summary: Fixed the main avoidable state-side quality issue on the current nonhuman range. Persisted state/state-window artifacts are now rewritten idempotently and loaders deduplicate stable keys, while bar-backed state quality now uses canonical tick quality/conflict/dual-source evidence when available instead of only coarse bar heuristics.
needs: Agent 3 should note one external blocker remains outside state ownership: the current full compiler suite still trips in mt5pipe.truth.service on a missing _quality_caveat_summary helper. Focused state tests are green and live state materialization quality on 2024-02-26..2024-03-01 improved materially after the state cleanup.
files: mt5pipe/state/service.py, tests/test_state_windows.py
```
### [2026-04-05 00:06] Agent 1 - Phase 5 readiness metadata handoff

```
agent: agent_1
type: handoff
area: state
summary: Added typed range-level and interval-level readiness metadata to state artifacts without changing the service surface. State manifests/results now expose readiness_summary plus daily/session readiness rollups, and state refs/requests normalize symbol/clock casing so callers are not sensitive to xauusd vs XAUUSD or m1 vs M1.
needs: Agent 2 can consume manifest.readiness_summary and per-day-session rollups from mt5pipe.state.public when deciding which intervals are gap-heavy or low-overlap. Agent 3 can use the same summaries for truth/training-readiness reasoning and window-availability checks without recomputing them downstream.
files: mt5pipe/contracts/state.py, mt5pipe/state/models.py, mt5pipe/state/public.py, mt5pipe/state/internal/quality.py, mt5pipe/state/service.py, mt5pipe/state/internal/windows.py, tests/test_contracts.py, tests/test_state_windows.py
verification: pytest tests/test_contracts.py tests/test_state_windows.py tests/test_state_boundaries.py tests/test_schema.py -q -> 28 passed
```

### [2026-04-05 00:35] Agent 3 - Phase 5 checkpoint

```
agent: agent_3
type: handoff
area: compiler
summary: The first institutional training-readiness path is now live on top of the green Dataset OS. Trusted datasets are hard-gated by truth acceptance, run-experiment writes walk-forward + holdout evaluation artifacts, experiment/model manifests are cataloged with lineage back to dataset:// refs, and train inspect-experiment / inspect-model are working on the live workspace.
needs: Stable now: ExperimentSpec boundary, training-run registry, experiment/model alias lifecycle, compact Windows-safe experiment/model artifact storage, and the baseline gaussian_nb_binary@1.0.0 workflow on dataset://xau_m1_nonhuman@1.0.0. Still missing before institutional-scale training: richer model families, model/result diff tooling, broader experiment comparison workflows, and deeper evaluation diagnostics beyond the first balanced-accuracy baseline.
files: mt5pipe/compiler/models.py, mt5pipe/compiler/manifest.py, mt5pipe/compiler/public.py, mt5pipe/compiler/training.py, mt5pipe/catalog/models.py, mt5pipe/catalog/sqlite.py, mt5pipe/storage/paths.py, mt5pipe/cli/train_cmds.py, mt5pipe/cli/app.py, config/experiments/xau_m1_nonhuman_direction_nb_v1.yaml, tests/test_training_flow.py
verification: pytest tests/test_training_flow.py tests/test_compiler.py tests/test_catalog.py tests/test_dataset_cli_compat.py tests/test_boundary_imports.py -q -> 21 passed, 1 xfailed; python -m mt5pipe.cli.app train run-experiment --spec config/experiments/xau_m1_nonhuman_direction_nb_v1.yaml -> accepted experiment/model artifacts on dataset.xau_m1_nonhuman.1af51fbdf628 with walk_forward_balanced_accuracy_mean=0.5354 and holdout_balanced_accuracy=0.5096
```

### [2026-04-05 15:31] Agent 3 - Final hardening checkpoint

```
agent: agent_3
type: handoff
area: compiler/truth/catalog/cli/training
summary: Closed the five remaining hardening gaps for the production ML path. Production dataset specs now require synchronized broker_a+broker_b raw coverage plus dual-source overlap, truth enforces those requirements as hard publication gates, raw tick accounting reports net-new deduped rows, upstream state/feature/label manifests resolve immutable artifact-scoped parquet roots, and merge diagnostics/QA rewrite one canonical file per day on rerun.
needs: Agent 1 and Agent 2 do not need to change their public surfaces for this checkpoint. The current production slice is now 2026-03-30..2026-04-02 because that is the verified synchronized dual-broker window in the live workspace; future wider production windows should only be adopted once both raw brokers are backfilled and the truth source requirements stay green.
files: mt5pipe/compiler/models.py, mt5pipe/truth/service.py, mt5pipe/ingestion/ticks.py, mt5pipe/backfill/engine.py, mt5pipe/storage/parquet_store.py, mt5pipe/storage/paths.py, mt5pipe/state/service.py, mt5pipe/features/service.py, mt5pipe/features/artifacts.py, mt5pipe/labels/service.py, mt5pipe/labels/artifacts.py, mt5pipe/merge/canonical.py, mt5pipe/quality/merge_qa.py, config/datasets/xau_m1_core_v1.yaml, config/datasets/xau_m1_nonhuman_v1.yaml, tests/test_checkpoint.py, tests/test_canonical_merge.py, tests/test_truth_core.py, tests/test_artifact_lineage.py, tests/test_compiler.py
verification: pytest tests/test_checkpoint.py tests/test_canonical_merge.py tests/test_merge_qa.py tests/test_truth_core.py tests/test_artifact_lineage.py tests/test_compiler.py tests/test_dataset_cli_compat.py tests/test_training_flow.py tests/test_catalog.py tests/test_boundary_imports.py -q -> 57 passed, 1 xfailed; python -m mt5pipe.cli.app backfill sync-ticks --broker-a broker_a --broker-b broker_b --symbol XAUUSD --from 2026-03-30 --to 2026-04-03 -> broker_a ticks_in_range=747,476 rows_added=0, broker_b ticks_in_range=784,562 rows_added=0 on rerun; python -m mt5pipe.cli.app merge canonical --symbol XAUUSD --broker-a broker_a --broker-b broker_b --from 2026-03-30 --to 2026-04-02 -> canonical_dual_rows=215,777 dual_source_ratio=0.1639 validation_reject_count=0; python -m mt5pipe.cli.app dataset compile-dataset --spec config/datasets/xau_m1_nonhuman_v1.yaml --publish -> dataset.xau_m1_nonhuman.219cc1cdb344 published accepted 98.11 with synchronized_raw_coverage_ratio=1.0; python -m mt5pipe.cli.app train run-experiment --spec config/experiments/xau_m1_nonhuman_direction_nb_v1.yaml -> experiment/model artifacts linked to dataset://xau_m1_nonhuman@1.0.0
```

### [2026-04-05 17:47] Agent 2 - Phase 6 prompt alignment note

```
agent: agent_2
type: update
area: machinelearning
summary: Current prompt references APH/machinelearning/models, but this workspace exposes machinelearning/ at the repository root. Proceeding in root machinelearning/models, and only touching chat/contracts.md if the new machinelearning public boundary actually changes.
needs: Agent 1/3 should target root machinelearning/ paths in this worktree rather than APH-prefixed paths.
files: machinelearning/models/, chat/agent_2.md, chat/contracts.md
```

### [2026-04-05 17:56] Agent 2 - Phase 6 model handoff

```
agent: agent_2
type: handoff
area: machinelearning
summary: machinelearning.models is ready for Agent 3 integration. Public exports are AphelionModel, ModelOutput, and AphelionTFT; ModelOutput carries all four horizons across direction/tb/return/mae/mfe heads and exposes direction_probs/return_median/return_interval helpers plus vsn_weights["past"|"future"].
needs: Agent 3 can train against batch keys past_features/future_known/static/mask/time_idx and should treat machinelearning.models as the only model import surface. No mt5pipe or machinelearning.training imports exist in the model package.
files: machinelearning/models/__init__.py, machinelearning/models/base.py, machinelearning/models/blocks.py, machinelearning/models/tft.py, tests/test_ml_models.py
verification: python -m pytest tests/test_ml_models.py -q -> 3 passed
```

### [2026-04-05 20:01] Agent 2 - attention-inspection note for next round

```
agent: agent_2
type: handoff
area: machinelearning
summary: Added AttentionInspector as the model-layer interpretability hook, but current ModelOutput does not store temporal self-attention weights. The inspector therefore returns None for mean_attention/last_timestep_attention until that signal is exposed.
needs: If Agent 3 wants actual temporal attention plots after training, the next round should extend the model output contract and AphelionTFT forward path to persist TemporalSelfAttention attn_weights explicitly.
files: machinelearning/models/interpret.py, tests/test_ml_models.py
```

## Hardening Pass Complete

[2026-04-06]
gaps_closed: 6
tests_added: 18
tests_total: 64
open_items: 0
notes: All requested Phase 6 hardening gaps are closed and verified. The machinelearning suite now includes a bridge file so `pytest machinelearning/tests -q -p no:cacheprovider` exercises the full Phase 6 stack layout in this worktree, including the model contract tests that still live under the repository-level tests directory.
