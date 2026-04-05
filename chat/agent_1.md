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

[2026-04-04T19:06:03.0507479-04:00]
owner: mt5pipe/compiler/, mt5pipe/catalog/, mt5pipe/truth/, mt5pipe/storage/paths.py (compiler/catalog/truth paths only), tests/test_catalog.py, tests/test_compiler.py, tests/test_contracts.py
plan: audit current core services, harden compile_dataset_spec/inspect_artifact/diff_artifacts, enforce artifact status lifecycle + publish gate, expand catalog/query tests, avoid CLI/config/legacy-builder files owned by Agent B.
[2026-04-04T19:09:19.1508947-04:00]
update: adding stable core entrypoints compile_dataset_spec(spec_path, publish=True), inspect_artifact(ref), diff_artifacts(left_ref, right_ref). Under the hood I�m tightening catalog lookup + truth-gated lifecycle, but not changing CLI files or spec ownership.
[2026-04-04T19:23:30-04:00]
update: traced remaining core failures to Windows manifest sidecar path length (>260 chars) for feature/label/state artifacts. Fix will stay inside compiler/storage path generation; stable service APIs remain unchanged.
[2026-04-04T19:24:40-04:00]
update: core service path is green. compile_dataset_spec/inspect_artifact/diff_artifacts are implemented and stable. Also shortened manifest sidecar storage paths to avoid Windows path-length failures during state/feature/label/dataset artifact writes.
[2026-04-04T19:25:00-04:00]
owner: mt5pipe/contracts/, mt5pipe/state/, tests for contracts/state/public boundary
plan: harden state-side artifact refs and window contracts, add machine-native state/window materialization + public exports, keep cross-sector boundary clean for Agent 2/3.
[2026-04-04T20:40:00-04:00]
update: published new shared state contracts in mt5pipe.contracts.state and expanded mt5pipe.state.public. Stable imports now include TickArtifactRef, StateArtifactRef, StateWindowArtifactRef, StateWindowRequest, StateWindowRecord, load_state_artifact, and materialize_state_windows.
update: state sector no longer imports compiler/catalog internals. StateService keeps compiler compatibility for materialize_state(...) but now also supports canonical tick state + rolling state-window artifacts.
[2026-04-04T21:05:00-04:00]
feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: Phase 4 should freeze the architecture and harden the nonhuman dataset path with better coverage intelligence, stronger state-window reliability, and richer source-quality metadata. Focus stays on state quality and wider-range reliability, not feature creativity or architecture churn.
phase: Phase 4
area: state
[2026-04-04T21:38:00-04:00]
update: hardened state artifacts for Phase 4 wider-range reliability. StateSnapshot now carries expected/observed interval metadata, source participation score, overlap confidence hint, and explicit gap_fill_flag; StateArtifactManifest now carries typed coverage_summary/source_quality_summary plus symbol/clock/time-range metadata.
update: rolling state windows now persist warmup/completeness/gap/fill/source-quality summaries and state.public exports StateCoverageSummary, StateSourceQualitySummary, and load_state_window_artifact. I also localized timeframe/weekend-gap helpers into state internals so the state sector no longer depends on mt5pipe.bars or mt5pipe.quality internals.
handoff: Agent 2 can rely on per-window fields warmup_satisfied, warmup_missing_rows, completeness, filled_row_count, gap_count, max_gap_ms, source_participation_score_mean, overlap_confidence_mean, and source_quality_hint_mean. Agent 3 can rely on manifest.coverage_summary/source_quality_summary plus time_range_start_utc/time_range_end_utc for artifact reasoning.
[2026-04-04T22:26:40-04:00]
feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: Phase 4 should stay disciplined: freeze the architecture, harden the nonhuman dataset path, and improve coverage/state quality without feature sprawl or contract churn. My continuation work is limited to state reliability, wider-range window stability, and source-quality metadata that Truth and features can trust.
phase: Phase 4 continuation
area: state
[2026-04-04T22:30:00-04:00]
update: audited the current nonhuman dataset path against state-only risks. No remaining state-side blocker showed up in the compiler-backed nonhuman tests; the one real reliability issue was state-window behavior when a wider source artifact was used for a narrower request.
update: materialize_state_windows now enforces that request dates lie within the source ref range, uses the full source range for PIT-safe warmup context, filters emitted anchors back to the requested date range, and records lineage/input refs for the full source range actually used.
handoff: boundary changed only in behavior, not symbols. Agent 2 can now assume state-window artifacts match requested anchor dates while still preserving prior-day warmup context. Agent 3 can now assume state-window manifests/input refs cover the actual source partitions used for those windows.
[2026-04-04T23:15:18-04:00]
feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: Phase 4 cleanup should stay narrow: improve state-side quality and metadata so the current nonhuman dataset path is more trustworthy without architecture churn. I am focusing on state source-quality signals, overlap/staleness/conflict annotations, and avoidable null propagation into the current dataset path.
phase: Phase 4 cleanup
area: state
[2026-04-04T23:32:00-04:00]
update: cleaned up state-side artifact reliability for the current nonhuman path. State partition writes are now idempotent and public state/window loaders deduplicate persisted rows on stable keys, so repeated materialization no longer inflates persisted artifacts.
update: bar-backed state quality/source_quality_hint now prefer canonical tick quality evidence when it exists, instead of relying only on coarse single-source bar heuristics. This materially improves the state-side source_quality inputs available to downstream truth without changing public symbols.
handoff: no public boundary symbols changed. Agent 2/3 can treat persisted state/state-window loads as idempotent and can rely on higher-fidelity state quality signals on the current synchronized range.
[2026-04-04T23:36:00-04:00]
feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: The latest human guidance still emphasizes discipline over churn: harden coverage quality, source-quality signals, and wider-range reliability while keeping the compiler/truth path central. For Phase 5 I am applying that same discipline to state/contracts by improving readiness metadata and reducing avoidable symbol-specific assumptions without expanding scope.
phase: Phase 5
area: state|contracts
[2026-04-05T00:06:00-04:00]
update: Phase 5 state/contracts hardening landed. Typed state refs/requests now normalize symbol and clock casing, and state manifests/results now carry typed readiness summaries plus daily/session readiness rollups for downstream truth/training use.
update: readiness metadata stays inside the existing state surface instead of adding new services. The new summaries cover effective coverage, gap burden, source/overlap quality bands, readiness bands, and window-availability ratios on state-window artifacts.
handoff: Agent 2 can rely on manifest.readiness_summary + daily/session rollups from mt5pipe.state.public. Agent 3 can use the same fields for training-readiness/truth reasoning without re-deriving day/session quality rollups downstream.
[2026-04-05T17:48:58-04:00]
feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: Phase 6 is the standalone deep-learning stack under machinelearning/, with Agent 1 owning data/. The scope is parquet-only ingestion from dataset://xau_m1_nonhuman@1.0.0, no mt5pipe internals, and the published baseline remains the floor for downstream training work.
phase: Phase 6
area: machinelearning/data
plan: Build the schema, robust feature normalizer, parquet-backed dataset, and Lightning datamodule under machinelearning/data/, add synthetic tests under machinelearning/tests, and log the machinelearning.data boundary without touching mt5pipe.
[2026-04-05T18:05:00-04:00]
update: Phase 6 data layer landed under machinelearning/data. The package now exposes the frozen dataset schema, a train-only robust median/IQR normalizer with JSON round-tripping, a parquet-backed sliding-window dataset, and a Lightning-compatible datamodule that reads split=train|val|test parquet roots only.
update: dataset behavior matches the published contract: missing feature columns warn and zero-fill, feature columns forward-fill without touching targets, symbol_idx defaults to 0 when absent, classification targets remap {-1,0,1}->{0,1,2} with -100 for missing labels, and targets always come from the last bar in each window.
handoff: Agent 2 and Agent 3 can import from machinelearning.data only. Tests live in machinelearning/tests/test_ml_data.py and use synthetic parquet splits, so model/training work can develop against the package surface without requiring the live dataset artifact in this worktree.
verification: python -m pytest machinelearning/tests/test_ml_data.py -q -p no:cacheprovider -> 9 passed; python import smoke for machinelearning.data -> ok
[2026-04-05T18:03:05-04:00]
feedback_read: yes
feedback_source: feedbacks/latest.md
feedback_summary: Phase 6 stays fully inside machinelearning/, with Agent 1 responsible for a parquet-only data layer over dataset://xau_m1_nonhuman@1.0.0. The deliverable is the schema + normalization + dataset/dataloader substrate that Agent 3 can train against without importing mt5pipe internals.
phase: Phase 6 completion
area: machinelearning/data
update: built the public machinelearning.data surface with ColumnSchema/DEFAULT_SCHEMA, a median+IQR RobustFeatureNormalizer, a parquet-backed AphelionDataset, and an AphelionDataModule that fits on the train split only and can save the normalizer JSON for inference reuse.
update: dataset loading now concatenates parquet parts, sorts by time_utc, zero-fills missing feature/static columns with warnings, preserves missing targets as typed nulls, derives mask=False from optional _filled rows, remaps categorical targets {-1,0,1}->{0,1,2} with -100 for missing labels, and takes all targets from the last bar of each context window only.
verification: pytest tests/test_ml_data.py -q -> 9 passed in 2.25s; python import smoke with machinelearning on sys.path -> ColumnSchema, DEFAULT_SCHEMA, RobustFeatureNormalizer, AphelionDataset, AphelionDataModule imported successfully; compileall on data/ and tests/test_ml_data.py -> True/True
handoff: Agent 3 can depend on machinelearning.data only, rely on DEFAULT_SCHEMA.n_past=75, DEFAULT_SCHEMA.n_future=8, DEFAULT_SCHEMA.n_static=1, and load/save the normalizer JSON identically for training and inference.
