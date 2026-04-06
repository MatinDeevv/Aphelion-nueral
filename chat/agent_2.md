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

[2026-04-04T19:06:28.4687860-04:00]
owner: mt5pipe/cli/dataset_cmds.py, mt5pipe/cli/app.py (registration only if needed), mt5pipe/features/dataset.py, config/datasets/xau_m1_core_v1.yaml, tests/* (CLI + compatibility + spec path)
plan: switch dataset CLI to service-level compiler APIs, add stable output formatting + failure handling, add legacy build compatibility facade when compiler/spec path is safe, harden example DatasetSpec and interface tests.

[2026-04-04T19:08:16.8631103-04:00]
update: CLI will prefer compiler.service compile_dataset_spec/inspect_artifact/diff_artifacts and fall back to the existing DatasetCompiler adapter only if the new functions are absent.
update: legacy build compatibility path will route through compiler only for compiler-compatible M1/default stable configs; it will use publish=false and mirror split outputs back to legacy datasets/name=*/split=* paths.

[2026-04-04T19:14:34.9594404-04:00]
update: compile-dataset/inspect-dataset/diff-dataset now format stable key-value output from service-level results and return explicit non-zero errors on rejection/exceptions.
update: legacy build now generates a temporary DatasetSpec with stable selectors (time/session/quality/htf_context) and mirrors compiler artifact splits back into legacy dataset paths when compile_dataset_spec is available.

[2026-04-04T19:16:09.9298113-04:00]
blocker-for-agent-a: mt5pipe/compiler/service.py is still absent in this worktree. My wrappers now handle the missing-module case, but they are ready to bind to compile_dataset_spec/inspect_artifact/diff_artifacts as soon as that module lands.
update: focused interface tests pass for the stubbed service contract and legacy compatibility facade.

[2026-04-04T19:16:50.1786266-04:00]
blocker-for-agent-a: verified current mt5pipe/compiler/service.py contains DatasetCompiler but not compile_dataset_spec/inspect_artifact/diff_artifacts yet. My CLI wrappers still fall back cleanly; once those functions exist they will be used first.

[2026-04-04T20:16:14.2124821-04:00]
agent: agent_2
area: features|labels
summary: starting Phase 3 feature-family work; auditing public surfaces, registry defaults, builder architecture, and PIT-safe alignment against mt5pipe.state.public only.
needs: none
files: mt5pipe/features/, mt5pipe/labels/, tests/*feature* tests/*label* registry/public surfaces

[2026-04-04T20:30:19.0936436-04:00]
agent: agent_2
area: features|labels
summary: shipped stable Phase 3 families disagreement/*, event_shape/*, entropy/*; expanded features.public and labels.public with registry helpers plus artifact ref/load helpers; labels remain compatible.
needs: Agent 3 can compile selectors disagreement/*, event_shape/*, entropy/* once compiler module is restored in this worktree.
files: mt5pipe/features/public.py, mt5pipe/features/service.py, mt5pipe/features/registry/defaults.py, mt5pipe/features/disagreement/, mt5pipe/features/event_shape/, mt5pipe/features/entropy/, mt5pipe/features/artifacts.py, mt5pipe/features/types.py, mt5pipe/labels/public.py, mt5pipe/labels/artifacts.py, tests/test_features_phase3.py, chat/contracts.md, docs/parallel_dev_architecture.md
selector-notes: disagreement.microstructure_pressure@1.0.0 -> outputs mid_divergence_proxy_bps, spread_divergence_proxy_bps, disagreement_pressure_bps, disagreement_zscore_60, conflict_burst_15, disagreement_burst_15, staleness_asymmetry_15, disagreement_entropy_30; event_shape.flow_shape@1.0.0 -> outputs tick_rate_hz, interarrival_mean_ms, burstiness_20, silence_ratio_20, direction_switch_rate_20, signed_run_length, path_efficiency_20, tortuosity_20; entropy.market_complexity@1.0.0 -> outputs return_sign_shannon_entropy_30, return_permutation_entropy_30, return_sample_entropy_30, volatility_approx_entropy_30.
blocker: tests/test_compiler.py and tests/test_boundary_imports.py::test_public_modules_importable are blocked by missing mt5pipe.compiler.service in the current tree, not by feature-sector failures.

[2026-04-04T21:25:36.4537479-04:00]
agent: agent_2
feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: Freeze the working Phase 3 boundaries, harden the nonhuman dataset path, and expand features in a controlled way. For Agent 2 specifically: improve registry quality, PIT safety, family-level tests, and label maturity without exploding feature scope.
phase: Phase 4
area: features|labels

[2026-04-04T21:37:37.2561916-04:00]
agent: agent_2
area: features|labels
summary: hardened disagreement/*, event_shape/*, and entropy/* so all family outputs stay null through warmup and degrade to typed-null columns when core inputs are missing; added one new stable family multiscale/*.
update: multiscale.consistency@1.0.0 is registered and exported via features.public as add_multiscale_features; selector outputs are trend_alignment_5_15_60, return_energy_ratio_5_60, volatility_ratio_5_60, range_expansion_ratio_15_60, tick_intensity_ratio_5_60.
update: triple_barrier_* now returns null when the forward horizon is unavailable and treats the horizon endpoint as included; label manifests now carry compact horizon/class-balance diagnostics plus exclusions metadata.
tests: pytest tests/test_features_phase3.py tests/test_features_phase4.py tests/test_labels.py tests/test_compiler.py tests/test_boundary_imports.py tests/test_truth_core.py -q; pytest tests/test_contracts.py -q
result: 54 passed, 1 xfailed

[2026-04-04T22:27:08.4389065-04:00]
agent: agent_2
feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: Freeze the current Phase 3/4 shape, harden the existing nonhuman dataset path, and avoid broad feature expansion. For Agent 2 that means reliability, PIT safety, registry clarity, and label maturity for the current machine-native families.
phase: Phase 4 continuation
area: features|labels

[2026-04-04T22:31:40.1910047-04:00]
agent: agent_2
area: features|labels
summary: finished the continuation pass without adding more families. Added checkpoint coverage proving the stable selector set time/session/quality/htf_context/disagreement/event_shape/entropy/multiscale compiles cleanly through mt5pipe.compiler.public on seeded M1 data.
update: label artifact diagnostics now include horizons_minutes, max_horizon_minutes, and recommended_min_embargo_rows so purge/embargo expectations are explicit alongside per-horizon null/class-balance summaries.
safe-selectors: disagreement/*, event_shape/*, entropy/*, multiscale/* are now checkpoint-stable for compile on the registry-materialized path; no further family expansion was introduced.
tests: pytest tests/test_features_phase4_checkpoint.py tests/test_features_phase4.py tests/test_features_phase3.py tests/test_labels.py tests/test_compiler.py tests/test_boundary_imports.py tests/test_truth_core.py tests/test_contracts.py -q
result: 55 passed, 1 xfailed

[2026-04-04T23:15:35.8155548-04:00]
agent: agent_2
feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: Freeze the current Phase 4 surface and focus on making the nonhuman path boringly reliable. For Agent 2 that means cleaning null-heavy or trivial stable columns, tightening warmup semantics, and keeping label compatibility honest.
phase: Phase 4 cleanup
area: features|labels

[2026-04-04T23:33:12.6400777-04:00]
agent: agent_2
area: features|labels
summary: completed the cleanup pass without adding families. Stable htf_context/* now excludes HTF *_tick_count outputs, stable disagreement/* is narrowed to the four production-ready pressure columns, and event/disagreement warmup is enforced per column rather than blanket family-wide.
update: label artifact diagnostics now include constant_output_columns so slice-trivial labels stay visible instead of silently blending into the stable path.
safe-selectors: time/*, session/*, quality/*, htf_context/*, disagreement/*, event_shape/*, entropy/*, multiscale/* remain stable for the nonhuman checkpoint path.
demoted-from-stable: htf_context/* -> M5_tick_count M15_tick_count H1_tick_count H4_tick_count D1_tick_count; disagreement/* -> spread_divergence_proxy_bps conflict_burst_15 staleness_asymmetry_15 disagreement_entropy_30
tests: pytest tests/test_features_phase3.py tests/test_features_phase4.py tests/test_features_phase4_checkpoint.py tests/test_labels.py tests/test_compiler.py tests/test_boundary_imports.py tests/test_truth_core.py tests/test_contracts.py -q
result: 59 passed, 1 xfailed

[2026-04-05T00:05:43.1234689-04:00]
agent: agent_2
area: features|labels
summary: Phase 5 landed as a trainability/metadata pass, not a new-family pass. Stable feature specs now expose ablation_group/trainability_tags, core_tb_volscaled@1.0.0 now exposes explicit training assumptions, feature-view manifests publish post-warmup trainability diagnostics, and label-view manifests publish target-distribution plus degenerate-horizon diagnostics.
update: label service now honors parameters.direction_threshold_bps for direction_* generation, but the default stable pack keeps that at 0.0 so current outputs remain backward-compatible unless a future pack opts in.
stable-training-surface: feature manifest metadata.trainability_diagnostics; label manifest metadata.label_diagnostics with target_distribution_summary, degenerate_*_horizons, tail_mismatch_horizons, and pack-level trainability metadata.
tests: pytest tests/test_features_phase5.py tests/test_features_phase4_checkpoint.py tests/test_features_phase4.py tests/test_features_phase3.py tests/test_labels.py tests/test_compiler.py tests/test_truth_core.py tests/test_boundary_imports.py tests/test_contracts.py -q
result: 67 passed, 1 xfailed


[2026-04-04T23:53:17.5707081-04:00]
agent: agent_2
feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: Freeze the stable feature surface and improve trainability, label maturity, and artifact evaluability without broad feature expansion. For Agent 2 this means stronger diagnostics, clearer metadata, and cleaner training assumptions on the existing families.
phase: Phase 5
area: features|labels

[2026-04-05T17:47:33.3315005-04:00]
agent: agent_2
feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: Phase 6 is the self-contained deep-learning stack in machinelearning/, evaluated against the 0.5354 walk-forward balanced-accuracy baseline. My scope is models/ only, with machinelearning boundary logging kept local and no mt5pipe internals touched.
phase: Phase 6
area: machinelearning|models

[2026-04-05T17:56:28.6835810-04:00]
agent: agent_2
area: machinelearning|models
summary: Completed the Phase 6 model-sector deliverable. Added the AphelionModel/ModelOutput boundary, reusable TFT building blocks, and AphelionTFT with multi-horizon direction/triple-barrier/quantile/MAE/MFE heads plus VSN interpretability weights for past and future inputs.
verification: python -m pytest tests/test_ml_models.py -q -> 3 passed in 27.93s; python compileall over machinelearning/ and tests/ -> clean
notes: Installed torch 2.11.0 into the local Python 3.14 environment to run the model tests in this workspace.
[2026-04-05T20:01:13-04:00]
agent: agent_2
area: machinelearning|models
summary: Hardened the Phase 6 model suite and added the first post-training interpretability utilities. tests/test_ml_models.py now covers individual block shapes, invalid head configuration, encoder_hidden and VSN-weight shapes, ordered quantiles, interval monotonicity, probability normalization, all-horizon presence, per-head gradient isolation, and lightweight interpretability checks.
fix: QuantileHead previously emitted unconstrained raw quantiles, so higher quantiles could be below lower quantiles. I changed it to emit a base quantile plus positive cumulative increments, making return_preds monotonic along the quantile axis by construction.
update: Added machinelearning.models.interpret with VSNInterpreter for past-feature importance summaries and AttentionInspector for temporal-attention inspection. Because ModelOutput still does not store attention weights, AttentionInspector currently returns None for mean_attention/last_timestep_attention; I logged that as a coordination note rather than expanding the output contract in this round.
verification: python -m pytest tests/test_ml_models.py -q -p no:cacheprovider -> 21 passed in 2.91s; python public-surface smoke import for AphelionTFT/VSNInterpreter/AttentionInspector/ModelOutput -> ok
