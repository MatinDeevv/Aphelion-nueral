# Agent 3 — Compiler / Truth / Catalog

[Human feedback rule]

Treat `feedbacks/latest.md` as the highest-priority human steering note before new work begins.

Before starting any new work:
1. Check whether `feedbacks/latest.md` exists and is non-empty.
2. If it exists, read it fully before doing anything else.
3. In your agent log, record:
	- `feedback_read: yes`
	- `feedback_source: feedbacks/latest.md`
	- `feedback_summary: <1-3 lines>`
4. Follow the latest human review unless it directly conflicts with the active task/prompt.
5. If there is a conflict, log it in `chat/coordination.md` before proceeding.

Do not put human feedback notes into `chat/contracts.md`.
Use:
- `feedbacks/latest.md` for human review
- `chat/contracts.md` for boundary/API/schema changes
- `chat/coordination.md` for blockers, requests, handoffs, and coordination

[Agent feedback log snippet]
feedback_read: yes|no
feedback_source: feedbacks/latest.md
feedback_summary: <1-3 lines>

**Owns:** `mt5pipe/compiler/`, `mt5pipe/truth/`, `mt5pipe/catalog/`
**Boundary module:** `mt5pipe/compiler/public.py`
**Tests:** `tests/test_compiler.py`, `tests/test_truth_core.py`, `tests/test_catalog.py`

## Responsibility
- Dataset compilation pipeline (`DatasetCompiler`, `compile_dataset_spec`)
- Artifact lifecycle management and truth gate (`TruthService`)
- Artifact catalog (`CatalogDB`)
- CLI integration glue (dataset commands)

## Allowed imports from other sectors
- `mt5pipe.contracts.*`
- `mt5pipe.state.public` — for `StateSnapshot`, `StateService`
- `mt5pipe.features.public` — for `FeatureSpec`, `FeatureService`, `LabelPack`, `LabelService`

## Log

<!-- Append timestamped entries below as work progresses -->

### [2026-04-04 20:16:24 -04:00] agent_3 - start

```
agent: agent_3
type: update
area: compiler
summary: Started Phase 3 compiler/truth/catalog/CLI pass. Reading current compiler-facing codepaths and dataset/test layout before implementing xau_m1_nonhuman_v1 end-to-end compile support.
needs: Will adapt to Agent 1/2 boundary changes through public modules only.
files: chat/contracts.md, chat/coordination.md, chat/agent_3.md, mt5pipe/compiler/public.py, mt5pipe/state/public.py, mt5pipe/features/public.py, mt5pipe/labels/public.py
```

### [2026-04-04 20:58:00 -04:00] agent_3 - phase3 pass complete

feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: No active human review yet.

```
agent: agent_3
type: update
area: compiler/truth/catalog/cli
summary: Implemented artifact-ref-aware compiler integration, Phase 3 truth gating, richer inspect/diff CLI summaries, and added the xau_m1_nonhuman_v1 example spec plus compiler-facing integration tests. Artifact-backed state builds now honor exclude:filled_rows through public state trust flags and/or _filled without importing state internals.
needs: Real workspace execution of config/datasets/xau_m1_nonhuman_v1.yaml still depends on Agent 2 machine-native feature artifacts being present in the local catalog/worktree. The compiler path and tests are ready for that boundary through mt5pipe.features.public and artifact refs only.
files: mt5pipe/compiler/public.py, mt5pipe/compiler/service.py, mt5pipe/truth/service.py, mt5pipe/cli/dataset_cmds.py, config/datasets/xau_m1_nonhuman_v1.yaml, config/datasets/xau_m1_core_v1.yaml, tests/test_compiler.py, tests/test_truth_core.py, tests/test_catalog.py, tests/test_dataset_cli_compat.py, docs/parallel_dev_architecture.md, chat/contracts.md, chat/coordination.md
```

### [2026-04-04 21:23:51 -04:00] agent_3 - phase4 start

feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: Phase 3 is accepted; Phase 4 should freeze architecture and harden the machine-native dataset path over wider synchronized ranges. Trust/reporting needs to become stricter and more useful without weakening publication gates.
phase: Phase 4
area: compiler|truth|catalog|cli

```
agent: agent_3
type: update
area: compiler/truth/catalog/cli
summary: Starting Phase 4 hardening on the nonhuman dataset path with focus on wider-range compile reliability, stricter and clearer trust gating, reproducible artifact lifecycle behavior, and more useful inspect/diff research ergonomics.
needs: Will stay on public boundaries only and log any public-surface drift in contracts/docs/spec examples immediately.
files: feedbacks/latest.md, chat/contracts.md, chat/coordination.md, chat/agent_3.md, mt5pipe/state/public.py, mt5pipe/features/public.py, mt5pipe/labels/public.py, mt5pipe/compiler/public.py
```

### [2026-04-04 22:28:11 -04:00] agent_3 - phase4 continuation start

feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: Phase 4 should stay disciplined: freeze architecture, harden the current nonhuman dataset path, and make trust/reporting more informative without weakening gates. Wider synchronized-range reliability is the checkpoint standard.
phase: Phase 4 continuation
area: compiler|truth|catalog|cli

```
agent: agent_3
type: update
area: compiler/truth/catalog/cli
summary: Starting the Phase 4 checkpoint audit/finalization pass. I am re-validating xau_m1_nonhuman_v1 against only stable public selectors, then tightening compiler/truth/catalog/CLI behavior wherever the live artifact path still produces ambiguous failures or weak diagnostics.
needs: Will keep the spec disciplined to currently stable selectors only and will record any checkpoint blockers precisely if the path is not truly green.
files: feedbacks/latest.md, chat/contracts.md, chat/coordination.md, chat/agent_3.md, mt5pipe/state/public.py, mt5pipe/features/public.py, mt5pipe/labels/public.py, mt5pipe/compiler/public.py
```

### [2026-04-04 22:44:59 -04:00] agent_3 - phase4 checkpoint pass

feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: Phase 4 should close only if the nonhuman dataset path is boringly reliable over a wider synchronized range and trust/reporting is stricter and more useful without architecture churn.
phase: Phase 4 continuation
area: compiler|truth|catalog|cli

```
agent: agent_3
type: update
area: compiler/truth/catalog/cli
summary: Finalized the stable-selector Phase 4 checkpoint path. xau_m1_nonhuman_v1 now compiles from public selectors only over 2024-02-26..2024-03-01, includes multiscale/*, publishes correctly, inspects/diffs with deterministic trust summaries, and focused compiler/truth/catalog/CLI tests are green.
needs: Checkpoint status is yellow rather than fully green because the live accepted artifact still carries research warnings: source_quality=62.87 (< preferred 75), HTF/event nulls are still present in the artifact, and some slice-specific columns are constant. Compiler/truth/catalog behavior itself is stable.
files: mt5pipe/compiler/service.py, mt5pipe/truth/service.py, mt5pipe/cli/dataset_cmds.py, config/datasets/xau_m1_nonhuman_v1.yaml, config/datasets/xau_m1_core_v1.yaml, tests/test_compiler.py, tests/test_truth_core.py, tests/test_dataset_cli_compat.py, docs/parallel_dev_architecture.md, chat/contracts.md
verification: pytest tests/test_compiler.py tests/test_truth_core.py tests/test_catalog.py tests/test_dataset_cli_compat.py -q -> 16 passed; python -m mt5pipe.cli.app dataset compile-dataset --spec config/datasets/xau_m1_core_v1.yaml --publish -> published/accepted 96.28; python -m mt5pipe.cli.app dataset compile-dataset --spec config/datasets/xau_m1_nonhuman_v1.yaml --publish -> published/accepted 96.28; inspect/diff by dataset:// refs returned deterministic trust decision/check-count/reason summaries
```

### [2026-04-04 23:15:37 -04:00] agent_3 - phase4 cleanup start

feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: The cleanup pass should keep the current stable selector path and make the yellow issues more truthful, not quieter. The goal is a cleaner checkpoint answer around source quality, expected sparsity, and constant-column usefulness.
phase: Phase 4 cleanup
area: compiler|truth|catalog|cli

```
agent: agent_3
type: update
area: compiler/truth/catalog/cli
summary: Starting the Phase 4 cleanup pass as checkpoint owner. I am re-evaluating xau_m1_nonhuman_v1 on the current stable selector set and tightening truth/inspect/diff classification around weak source quality, expected sparse columns, and slice-trivial constants.
needs: Will keep the spec on the stable registry-materialized selector path confirmed by Agent 2 and will not weaken publication gates to chase a greener badge.
files: feedbacks/latest.md, chat/contracts.md, chat/coordination.md, chat/agent_3.md, mt5pipe/state/public.py, mt5pipe/features/public.py, mt5pipe/labels/public.py, mt5pipe/compiler/public.py
```

### [2026-04-04 23:34:05 -04:00] agent_3 - phase4 cleanup complete

feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: The cleanup pass should keep the stable selector path and make yellow issues more truthful instead of quieter. The goal is a stronger checkpoint answer around source quality, expected sparsity, and constant-column usefulness.
phase: Phase 4 cleanup
area: compiler|truth|catalog|cli

```
agent: agent_3
type: completion
area: compiler/truth/catalog/cli
summary: Reclassified expected sparse nulls and slice-trivial constants as accepted caveats, preserved unexpected nulls/blocking constants/source-quality shortfalls as explicit blockers, and surfaced the new truth summaries through compile-dataset, inspect-dataset, and diff-dataset. Fresh live workspace runs now show xau_m1_nonhuman_v1 as a green checkpoint rather than yellow.
needs: None for this checkpoint. Remaining caveats are accepted slice-trivial constants only, and current source_quality is 77.2389 so it is no longer below the preferred threshold.
files: mt5pipe/truth/service.py, mt5pipe/cli/dataset_cmds.py, tests/test_truth_core.py, tests/test_dataset_cli_compat.py, chat/contracts.md, chat/coordination.md, docs/parallel_dev_architecture.md
verification: pytest tests/test_compiler.py tests/test_truth_core.py tests/test_catalog.py tests/test_dataset_cli_compat.py -q -> 18 passed; python -m mt5pipe.cli.app dataset compile-dataset --spec config/datasets/xau_m1_core_v1.yaml --publish -> dataset.xau_m1_core.01db30739cac published accepted 97.72; python -m mt5pipe.cli.app dataset compile-dataset --spec config/datasets/xau_m1_nonhuman_v1.yaml --publish -> dataset.xau_m1_nonhuman.1af51fbdf628 published accepted 97.72; inspect/diff on dataset://xau_m1_nonhuman@1.0.0 show trust_warning_reasons=[], source_quality=77.2389, accepted_caveats only
```

### [2026-04-04 23:53:37 -04:00] agent_3 - phase5 start

feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: Freeze the architecture that worked, keep the compiler/trust path as the center of gravity, and expand only where it improves research-grade artifact quality. Phase 5 training readiness needs disciplined lineage, reproducibility, and trust-gated usage rather than a kitchen-sink research layer.
phase: Phase 5
area: compiler|truth|catalog|cli|training

```
agent: agent_3
type: update
area: compiler/truth/catalog/cli/training
summary: Starting Phase 5 by designing a minimal but real training/evaluation path on top of accepted dataset artifacts. I am auditing the existing catalog/compiler surface first so the experiment registry, model registry, and CLI training workflow reuse the green Dataset OS rather than bypassing it.
needs: Will keep training strictly trust-gated, lineage-linked, and inspectable; any public compiler, registry, manifest, trust, or CLI behavior changes will be logged in contracts/docs immediately.
files: feedbacks/latest.md, chat/contracts.md, chat/coordination.md, chat/agent_3.md, mt5pipe/state/public.py, mt5pipe/features/public.py, mt5pipe/labels/public.py, mt5pipe/compiler/public.py
```

### [2026-04-05 00:35:38 -04:00] agent_3 - phase5 complete

feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: Freeze the architecture that worked, keep compiler/trust as the center of gravity, and add only disciplined training-readiness layers with full lineage and hard trust gates. Phase 5 should end with one real institutional training path, not placeholders.
phase: Phase 5
area: compiler|truth|catalog|cli|training

```
agent: agent_3
type: completion
area: compiler/truth/catalog/cli/training
summary: Implemented the first trust-gated institutional training path on top of accepted dataset artifacts. The compiler boundary now includes ExperimentSpec plus experiment/model inspection helpers, the catalog tracks experiment specs and training runs, experiment/model artifacts are registered through the shared manifest system, and the CLI exposes run-experiment / inspect-experiment / inspect-model with deterministic summaries.
needs: Phase 5 is stable for one baseline model path only. Future phases should add richer model families, comparison tooling, and deeper evaluation diagnostics without weakening the trusted-dataset gate or splitting artifact lineage across parallel registries.
files: mt5pipe/compiler/models.py, mt5pipe/compiler/manifest.py, mt5pipe/compiler/public.py, mt5pipe/compiler/training.py, mt5pipe/catalog/models.py, mt5pipe/catalog/sqlite.py, mt5pipe/storage/paths.py, mt5pipe/cli/train_cmds.py, mt5pipe/cli/app.py, config/experiments/xau_m1_nonhuman_direction_nb_v1.yaml, tests/test_training_flow.py, docs/parallel_dev_architecture.md, chat/contracts.md, chat/coordination.md
verification: pytest tests/test_training_flow.py tests/test_compiler.py tests/test_catalog.py tests/test_dataset_cli_compat.py tests/test_boundary_imports.py -q -> 21 passed, 1 xfailed; python -m mt5pipe.cli.app dataset inspect-dataset --artifact dataset://xau_m1_nonhuman@1.0.0 -> accepted trusted dataset artifact 97.72; python -m mt5pipe.cli.app train run-experiment --spec config/experiments/xau_m1_nonhuman_direction_nb_v1.yaml -> experiment.xau_m1_nonhuman_direction_nb.402c9aa8664d accepted, model.xau_m1_nonhuman_direction_nb.710a3e456ec2 accepted, walk_forward_balanced_accuracy_mean=0.5354, holdout_balanced_accuracy=0.5096; inspect-experiment / inspect-model by alias returned linked dataset/model/run summaries successfully
```

### [2026-04-05 14:45:50 -04:00] agent_3 - final hardening start

feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: Phase 3/4/5 feedback still points to the same discipline: no architecture churn, stricter trust, wider synchronized coverage, and artifact lineage you can actually audit. This pass is focused on closing the remaining proof gaps rather than expanding capability.
phase: Phase 5 hardening
area: compiler|truth|catalog|cli|training

```
agent: agent_3
type: update
area: compiler/truth/catalog/cli/training
summary: Starting the final end-to-end hardening pass to turn the pipeline claim from yellow to green. I am fixing synchronized dual-broker production coverage, making truth enforce multi-broker requirements, tightening raw ingest accounting, hardening upstream artifact identity, and deduplicating merge diagnostics before re-running the live dataset and training path.
needs: Will preserve the current research/training workflow and stay on public boundaries only; any public compiler/truth/catalog/CLI behavior changes will be logged in contracts/docs immediately.
files: feedbacks/latest.md, chat/contracts.md, chat/coordination.md, chat/agent_3.md, mt5pipe/state/public.py, mt5pipe/features/public.py, mt5pipe/labels/public.py, mt5pipe/compiler/public.py
```

### [2026-04-05 15:31:12 -04:00] agent_3 - final hardening complete

feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: The last pass should close proof gaps, not add scope: enforce synchronized dual-broker requirements, make raw accounting auditable, make upstream artifact lineage reproducible, and prove the current production path live.
phase: Phase 5 hardening
area: compiler|truth|catalog|cli|training

```
agent: agent_3
type: completion
area: compiler/truth/catalog/cli/training
summary: Fixed the remaining end-to-end proof gaps and re-ran the live production path on a synchronized dual-broker slice. Production DatasetSpecs now declare required brokers + synchronized overlap requirements, truth rejects/publishes accordingly, raw tick reruns report net-new rows_added=0, state/feature/label manifests resolve immutable artifact-scoped parquet roots, merge diagnostics/QA stay one-file-per-day on rerun, and the rebuilt xau_m1_nonhuman_v1 artifact compiles, inspects, diffs, and feeds the training path successfully.
needs: The strong claim is now green for the defined production slice 2026-03-30..2026-04-02. The main operational caveat is scope, not correctness: future production ranges must still be synchronized/backfilled on both brokers before these source requirements will pass.
files: mt5pipe/compiler/models.py, mt5pipe/truth/service.py, mt5pipe/cli/dataset_cmds.py, mt5pipe/ingestion/ticks.py, mt5pipe/backfill/engine.py, mt5pipe/storage/parquet_store.py, mt5pipe/storage/paths.py, mt5pipe/state/service.py, mt5pipe/features/service.py, mt5pipe/features/artifacts.py, mt5pipe/labels/service.py, mt5pipe/labels/artifacts.py, mt5pipe/merge/canonical.py, mt5pipe/quality/merge_qa.py, config/datasets/xau_m1_core_v1.yaml, config/datasets/xau_m1_nonhuman_v1.yaml, tests/test_checkpoint.py, tests/test_canonical_merge.py, tests/test_truth_core.py, tests/test_artifact_lineage.py, tests/test_compiler.py, tests/test_training_flow.py
verification: pytest tests/test_checkpoint.py tests/test_canonical_merge.py tests/test_merge_qa.py tests/test_truth_core.py tests/test_artifact_lineage.py tests/test_compiler.py tests/test_dataset_cli_compat.py tests/test_training_flow.py tests/test_catalog.py tests/test_boundary_imports.py -q -> 57 passed, 1 xfailed; python -m mt5pipe.cli.app backfill sync-ticks --broker-a broker_a --broker-b broker_b --symbol XAUUSD --from 2026-03-30 --to 2026-04-03 -> rows_added=0 on rerun with ticks_in_range 747,476 / 784,562; python -m mt5pipe.cli.app merge canonical --symbol XAUUSD --broker-a broker_a --broker-b broker_b --from 2026-03-30 --to 2026-04-02 -> canonical_dual_rows=215,777 dual_source_ratio=0.1639; python -m mt5pipe.cli.app dataset compile-dataset --spec config/datasets/xau_m1_nonhuman_v1.yaml --publish -> dataset.xau_m1_nonhuman.219cc1cdb344 published accepted 98.11; python -m mt5pipe.cli.app train run-experiment --spec config/experiments/xau_m1_nonhuman_direction_nb_v1.yaml -> experiment.xau_m1_nonhuman_direction_nb.b859ed294f94 and model.xau_m1_nonhuman_direction_nb.5f192c5412f2 linked to the trusted dataset alias
```

### [2026-04-05 20:03:38 -04:00] agent_3 - Phase 6 training complete

feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: Phase 6 is the self-contained deep-learning stack in machinelearning/, with Agent 3 owning training/ against the published dataset artifact and the 0.5354 walk-forward balanced-accuracy baseline. The training layer should optimize for real alpha signal, not generic loss minimization, and stay entirely off mt5pipe internals.
phase: Phase 6
area: machinelearning|training

```
agent: agent_3
type: completion
area: machinelearning|training
summary: Delivered the Phase 6 training sector with NaN-safe quantile/regression losses, focal classification loss, 60m IC regularization, homoscedastic learned task weighting, a Lightning-style AphelionLightningModule, and a reproducible train() entry point plus CLI under machinelearning/training/.
update: module.py now trains all 20 horizon/head combinations, logs per-task losses plus direction accuracy, balanced accuracy, and IC by horizon, and optimizes model parameters together with learned task-weight uncertainty terms. train.py wires the public AphelionDataModule and AphelionTFT surfaces into checkpointing, early stopping, W&B config logging, normalizer persistence, and a CPU-safe fallback trainer configuration when GPUs are unavailable.
update: cleaned the requested probe noise from machinelearning/probe_dir/ and machinelearning/test_write_probe.txt, and added synthetic coverage for losses, learned task weights, training_step, and optimizer/scheduler wiring in machinelearning/tests/test_ml_training.py.
needs: Code path is ready for real training, but I did not run an end-to-end Lightning + W&B experiment in this local environment because lightning and wandb are not installed here and the workspace is CPU-only. The package imports cleanly and the full machinelearning unit suite is green.
files: machinelearning/training/__init__.py, machinelearning/training/losses.py, machinelearning/training/module.py, machinelearning/training/train.py, machinelearning/tests/test_ml_training.py, chat/agent_3.md, chat/contracts.md
verification: pytest machinelearning/tests/test_ml_training.py -q -> 10 passed in 3.53s; pytest machinelearning/tests -q -> 25 passed in 3.55s; python import smoke for machinelearning.training -> AphelionLightningModule and train imported successfully; compileall over machinelearning/training and machinelearning/tests/test_ml_training.py -> True/True

[2026-04-06 12:00]
phase: Phase 6 hardening
area: machinelearning/training
fixes_applied:
	- refactored trainer construction into build_trainer() with explicit GPU-only DDP/BF16 behavior and a defensive CPU fp32 fallback
	- added validate_artifact_dir() so bad parquet roots fail immediately before training setup
	- added integration smoke coverage for data -> model -> training execution on synthetic parquet splits
tests_added: 5
verification: pytest machinelearning/tests -q -p no:cacheprovider -> 64 passed

### [2026-04-06 02:50:52 -04:00] agent_3 - Phase 7 signal/backtest complete

feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: feedbacks/latest.md is stale and still points to the completed Phase 6 training layer. I logged that conflict in chat/coordination.md and followed the active Phase 7 prompt, keeping the work isolated to machinelearning/signal, APH/backtest, synthetic tests, and chat logs.
phase: Phase 7
area: machinelearning/signal | APH/backtest

```
agent: agent_3
type: completion
area: machinelearning/signal | APH/backtest
summary: Delivered the Phase 7 signal-to-PnL layer. machinelearning.signal now provides immutable SignalRecord contracts, split conformal interval calibration, fractional Kelly position sizing, and a SignalPublisher that converts ModelOutput + regime probabilities + bar metadata into fully calibrated, fully sized trading signals. APH.backtest now provides a vectorized backtest engine, baseline-comparable metrics, and a printable/json report surface for evaluating whether those signals make money after simple spread costs.
update: The signal layer was kept intentionally boundary-clean: no mt5pipe imports and no machinelearning.training internals. SignalPublisher accepts the real RegimeState surface from machinelearning.regime but also works with any [4]-probability vector so downstream integration is not blocked on the fitted detector artifact.
update: Added synthetic contract coverage for conformal guarantees, Kelly sizing behavior, SignalRecord gating, SignalPublisher assembly, vectorized backtest PnL/cost handling, balanced accuracy range, and BacktestReport serialization/printing.
needs: Ready to receive real signals once Agent 1 delivers the 90-day dataset and Agent 2 delivers a fitted RegimeDetector. I did not run the prompt’s real end-to-end dataset -> regime -> TFT -> conformal -> signals -> backtest pass yet because Agent 1 has not logged the required Phase 7 dataset completion/handoff in chat/coordination.md.
files: machinelearning/signal/__init__.py, machinelearning/signal/records.py, machinelearning/signal/conformal.py, machinelearning/signal/sizing.py, machinelearning/signal/publisher.py, APH/backtest/__init__.py, APH/backtest/engine.py, APH/backtest/metrics.py, APH/backtest/report.py, machinelearning/tests/test_ml_signal.py, machinelearning/tests/test_ml_backtest.py, chat/contracts.md, chat/coordination.md, chat/agent_3.md
verification: python -m pytest machinelearning/tests/test_ml_signal.py -q -p no:cacheprovider -> 10 passed in 4.33s; python -m pytest machinelearning/tests/test_ml_backtest.py -q -p no:cacheprovider -> 7 passed in 3.24s; python -m pytest machinelearning/tests -q -p no:cacheprovider -> 91 passed, 1 skipped in 10.94s; compileall on machinelearning/signal and APH/backtest -> True/True; public import smoke for machinelearning.signal, APH.backtest, APH.backtest.metrics, and APH.backtest.report -> ok
```

### [2026-04-06 15:46 -04:00] agent_3 - Phase 7 quantitative validation

feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: feedbacks/latest.md still points at the completed Phase 6 stack, but the active task is Phase 7 quantitative validation on the published 90-day dataset. I followed the explicit Phase 7 prompt, fixed the blocking truth bug first, and kept all public APIs unchanged while validating the CPU research stack end to end.
phase: Phase 7 quantitative validation
area: truth/compiler + regime + signal + backtest

truth_gate_fix:
  bug: weekend dates were treated as required synchronized raw coverage days
  fix: weekday() < 5 filtering in source-quality required raw coverage and observability-day accounting
  tests: pytest data/tests/test_truth_core.py -q -> 7 passed

dataset:
  ref: dataset://xau_m1_nonhuman_90d@1.0.0
  artifact_id: dataset.xau_m1_nonhuman_90d.20de720fb54e
  trust_score: 97.74
  total_rows: 83269
  train/val/test: 58624/12322/12323
  source_quality: 77.3858
  warnings: none

nb_baseline_90d:
  walk_forward_balanced_acc: 0.493726
  holdout_balanced_acc: 0.499796
  vs_4day_baseline: worse
  analysis: The wider 90-day surface materially weakens the gaussian_nb signal relative to the 4-day checkpoint (walk-forward -0.0417, holdout -0.0098), which is informative rather than alarming. The small production slice was flattering the NB baseline; the wider dataset is the right surface for regime-aware TFT research.

regime_detector:
  fitted: yes
  saved: local_data/regime_detector_90d.pkl
  regime_distribution:
    trending: 15787 bars (26.9%)
    mean_reverting: 15304 bars (26.1%)
    volatile: 1627 bars (2.8%)
    quiet: 25906 bars (44.2%)
  notes: Distribution stayed below the >60% dominance warning threshold and did not require a refit above n_iter=200. hmmlearn is not installable on this local Python 3.14/MSVC-free Windows environment, so the fit used the internal sklearn GaussianMixture fallback without changing the public RegimeDetector API.

conformal_calibration:
  q_hat: 0.001764
  empirical_coverage: 0.9001
  target_coverage: 0.90
  saved: local_data/conformal_calibrator_90d.pkl

backtest_nb_signals:
  n_signals: 12323
  n_actionable: 0
  win_rate: 0.0000
  sharpe_ratio: 0.0000
  max_drawdown: 0.0000
  ic_60m: -0.0007
  balanced_accuracy: 0.3333
  vs_baseline_0.5354: -0.2021
  saved: local_data/backtest_nb_90d.json
  notes: The saved backtest/report path is valid, but the placeholder conformal intervals widened across zero on every NB signal, so Kelly sizing and SignalRecord.is_actionable() correctly gated all trades off. This is a conservative floor and pipeline-validation checkpoint, not evidence of a tradable NB stack.

regime_stratified_performance:
  trending: 1796 signals, win_rate=0.0000, IC=-0.0286
  mean_reverting: 9609 signals, win_rate=0.0000, IC=0.0409
  volatile: 66 signals, win_rate=0.0000, IC=0.0982
  quiet: 852 signals, win_rate=0.0000, IC=-0.0072

waiting_for:
  GCP VM - to run TFT training at full scale.
  All artifacts, calibrators, and fitted detectors are saved locally and ready to be loaded once training completes.

verification:
  - pytest data/tests/test_truth_core.py -q -> 7 passed
  - python -m pytest machinelearning/tests/test_ml_regime.py -q -p no:cacheprovider -> 14 passed, 1 skipped
  - python -m mt5pipe.cli.app dataset compile-dataset --config data/config/pipeline.yaml --spec data/config/datasets/xau_m1_nonhuman_90d_v1.yaml --publish -> published accepted 97.74
  - python -m mt5pipe.cli.app train run-experiment --spec data/config/experiments/xau_m1_nonhuman_90d_direction_nb_v1.yaml -> walk_forward_balanced_accuracy_mean=0.493726, holdout_balanced_accuracy=0.499796
  - local artifacts saved: local_data/regime_detector_90d.pkl, local_data/conformal_calibrator_90d.pkl, local_data/backtest_nb_90d.json, tmp/agent3_phase7_logs/phase7_quant_validation_summary.json
