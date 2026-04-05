# Parallel Development Architecture

## Overview

The codebase is organized into **3 sectors** with strict import boundaries, enabling 3 agents to work in parallel without merge conflicts or coupling regressions.

The top-level package is `mt5pipe/`. In architecture discussions, "pipeline" maps to `mt5pipe/`.

---

## Sectors

| Sector    | Owner   | Packages                                              |
|-----------|---------|-------------------------------------------------------|
| **State**     | Agent 1 | `mt5pipe/contracts/`, `mt5pipe/state/`               |
| **Features**  | Agent 2 | `mt5pipe/features/`, `mt5pipe/labels/`               |
| **Compiler**  | Agent 3 | `mt5pipe/compiler/`, `mt5pipe/truth/`, `mt5pipe/catalog/` |

### Agent 1 — State + Contracts
- **Owns:** `mt5pipe/contracts/`, `mt5pipe/state/`
- **Boundary module:** `mt5pipe/state/public.py`
- **Tests:** `tests/test_schema.py`, state-related tests
- **Responsibility:** Shared contract types, state materialization, state models

### Agent 2 — Features
- **Owns:** `mt5pipe/features/`, `mt5pipe/labels/`
- **Boundary module:** `mt5pipe/features/public.py`
- **Tests:** `tests/test_labels.py`, feature-related tests
- **Responsibility:** Feature builders, feature registry, label generation, label registry
- **Public surface:** registry helpers, artifact ref/load helpers, and stable family builders are re-exported from `mt5pipe.features.public` and `mt5pipe.labels.public`
- **Stable machine-native families:** `disagreement/*`, `event_shape/*`, `entropy/*`, `multiscale/*`
- **Stable selector cleanup:** stable `htf_context/*` excludes higher-timeframe `*_tick_count` columns; stable `disagreement/*` is limited to `mid_divergence_proxy_bps`, `disagreement_pressure_bps`, `disagreement_zscore_60`, and `disagreement_burst_15`
- **Experiment metadata note:** stable feature specs carry `ablation_group` and `trainability_tags`; stable label packs carry `qa_policy_ref`, `ablation_group`, `trainability_tags`, `target_groups`, and `tail_policy`
- **Artifact diagnostics note:** feature manifests include `metadata.trainability_diagnostics` with post-warmup coverage/stability summaries, and label manifests include richer `metadata.label_diagnostics` with target-distribution summaries, degenerate-horizon warnings, explicit embargo/tail policy, and `constant_output_columns`; insufficient forward-horizon triple-barrier rows are explicitly null

### Agent 3 — Compiler
- **Owns:** `mt5pipe/compiler/`, `mt5pipe/truth/`, `mt5pipe/catalog/`
- **Boundary module:** `mt5pipe/compiler/public.py`
- **Tests:** `tests/test_compiler.py`, `tests/test_truth_core.py`, `tests/test_catalog.py`
- **Responsibility:** Dataset compilation, truth gate, artifact catalog, CLI integration
- **Public surface:** `DatasetSpec` supports version refs plus explicit artifact refs, and `mt5pipe.compiler.public` re-exports compile/inspect/diff helpers for deterministic dataset artifact workflows
- **Truth/CLI note:** Trust reports now carry `score_breakdown`, `decision_summary`, `warning_reasons`, `rejection_reasons`, and `check_status_counts`; `compile-dataset`, `inspect-dataset`, and `diff-dataset` surface those fields directly for research use
- **Phase 5 training note:** `mt5pipe.compiler.public` also re-exports `ExperimentSpec`, `run_experiment_spec`, `inspect_experiment`, and `inspect_model`. Experiment/model artifacts live under compact experiment/model storage roots, are cataloged through the same manifest system as datasets, and are only trainable from dataset artifacts whose trust reports are accepted.
- **Experiment workflow:** the first institutional path is `config/experiments/xau_m1_nonhuman_direction_nb_v1.yaml`, which trains `gaussian_nb_binary@1.0.0` on `dataset://xau_m1_nonhuman@1.0.0` using walk-forward-plus-holdout evaluation and registers both `experiment://...` and `model://...` aliases.

---

## Neutral Packages (shared, no single owner)

These packages are **not** sector-gated. Any sector may import from them directly:

- `mt5pipe/models/` — domain models (ticks, bars, market, etc.)
- `mt5pipe/config/` — pipeline configuration
- `mt5pipe/storage/` — Parquet I/O, paths, checkpoints
- `mt5pipe/quality/` — QA / cleaning / gap detection
- `mt5pipe/merge/` — canonical tick merge
- `mt5pipe/mt5/` — MT5 connection layer
- `mt5pipe/ingestion/` — raw data ingestion
- `mt5pipe/backfill/` — backfill engine
- `mt5pipe/bars/` — bar builder
- `mt5pipe/live/` — live data collector
- `mt5pipe/cli/` — CLI commands
- `mt5pipe/utils/`, `mt5pipe/tools/`

---

## Import Rules

```
┌──────────────────────────────────────────────┐
│              ALLOWED IMPORTS                  │
│                                              │
│   Any module → mt5pipe.contracts.*           │
│   Any module → mt5pipe.state.public          │
│   Any module → mt5pipe.features.public       ││   Any module → mt5pipe.labels.public         │   Any module → mt5pipe.compiler.public       │
│   Any module → neutral packages              │
│   Intra-sector → anything within own sector  │
│                                              │
│              FORBIDDEN                       │
│                                              │
│   state → mt5pipe.features.builder           │
│   state → mt5pipe.compiler.service           │
│   features → mt5pipe.state.service           │
│   features → mt5pipe.compiler.models         │
│   compiler → mt5pipe.state.models            │
│   ... (any direct cross-sector internal)     │
└──────────────────────────────────────────────┘
```

**Rule:** Cross-sector imports may ONLY target:
1. `mt5pipe.contracts.*`
2. `mt5pipe.<sector>.public` (`state.public`, `features.public`, `labels.public`, `compiler.public`)

This is enforced by `tests/test_boundary_imports.py`.

---

## Coordination System

### `feedbacks/latest.md`
- **Purpose:** Highest-priority human steering note before new work begins
- **Rule:** Every agent must read this file before starting any new task
- **Archive:** Prior notes move to `feedbacks/archive/`

### Human Feedback Flow

Before starting any new work:
1. Check whether `feedbacks/latest.md` exists and is non-empty.
2. If it exists, read it fully before doing anything else.
3. In your agent log, record:
	- `feedback_read: yes`
	- `feedback_source: feedbacks/latest.md`
	- `feedback_summary: <1-3 lines>`
4. Follow the latest human review unless it directly conflicts with the active task/prompt.
5. If there is a conflict, log it in `chat/coordination.md` before proceeding.

Treat `feedbacks/latest.md` as the highest-priority human steering note before new work begins.

### `chat/agent_1.md`, `chat/agent_2.md`, `chat/agent_3.md`
- **Purpose:** Per-agent work logs: ownership declaration, plans, in-progress updates
- **Rule:** Each agent logs their own work here; other agents may read but not overwrite

### `chat/contracts.md`
- **Purpose:** Log all public boundary changes (new exports, signature changes, schema changes)
- **Rule:** Any change to a `public.py` or `mt5pipe/contracts/` file MUST be logged here
- **Format:** Structured entries with agent name, module, old/new signature, impact
- **Not for:** Human feedback notes, blockers, requests, or handoffs

### `chat/coordination.md`
- **Purpose:** Blockers, requests, progress updates, handoffs between agents
- **Rule:** Use this for any cross-agent communication
- **Format:** Structured entries with agent name, type, area, summary
- **Includes:** Human-feedback conflict logs when latest feedback conflicts with active prompt/task

---

## Package Structure

```
mt5pipe/
├── contracts/          ← Shared types (Agent 1 owns, all read)
│   ├── __init__.py
│   ├── artifacts.py    ← ArtifactRef, ArtifactKind
│   ├── dataset.py      ← DatasetId, DatasetSplitKind, DATASET_JOIN_KEYS
│   ├── trust.py        ← TrustVerdict
│   └── lineage.py      ← LineageNode
├── state/              ← Agent 1
│   ├── __init__.py
│   ├── public.py       ← BOUNDARY: StateSnapshot, StateService
│   ├── internal/
│   ├── models.py
│   └── service.py
├── features/           ← Agent 2
│   ├── __init__.py
│   ├── public.py       ← BOUNDARY: FeatureSpec, FeatureService, builders
│   ├── internal/
│   ├── builder.py
│   ├── service.py
│   ├── registry/
│   └── ...
├── labels/             ← Agent 2
│   ├── public.py       ← BOUNDARY: LabelPack, LabelService
│   └── ...
├── compiler/           ← Agent 3
│   ├── __init__.py
│   ├── public.py       ← BOUNDARY: DatasetSpec, LineageManifest, DatasetCompiler
│   ├── internal/
│   ├── models.py
│   ├── service.py
│   └── manifest.py
├── truth/              ← Agent 3
│   └── ...
├── catalog/            ← Agent 3
│   └── ...
└── [neutral packages]
```

---

## Migration Notes

The existing codebase has direct cross-sector imports that predate this boundary setup. For example, `compiler.service` imports from `state.service` directly. These will be migrated incrementally:

1. New code MUST use boundary modules
2. Existing imports will be migrated to use `public.py` + `contracts/`
3. `tests/test_boundary_imports.py` tracks violations (currently `xfail`)
4. Target: zero violations

---

## Workflow

1. Pick your sector's files
2. Read `feedbacks/latest.md` first and log `feedback_read`, `feedback_source`, `feedback_summary` in your agent file
3. If feedback conflicts with the active task/prompt, log conflict in `chat/coordination.md` before coding
4. Import only from contracts + other sectors' public.py
5. Log boundary changes in `chat/contracts.md`
6. Post blockers/updates/handoffs in `chat/coordination.md`
7. Run `pytest tests/test_boundary_imports.py` to check violations

---

## Phase 4 Truth Interpretation

- Trust warnings are reserved for real blockers to a green checkpoint: unexpected nulls, blocking constant columns, family missingness pressure, label issues, or source quality below preferred research comfort.
- Expected sparse nulls, such as HTF alignment sparsity or declared warmup sparsity, remain visible in `TrustReport.metrics.quality_caveat_summary.accepted_caveats` and in CLI `quality_caveats` / `quality_family_summary` output, but they do not create warning reasons by themselves.
- Slice-trivial constants that are expected for a synchronized slice remain visible as accepted caveats. Only blocking constant columns stay in the warning path.
- Source-quality reporting may fall back to `merge_diagnostics` when formal `merge_qa` artifacts are absent. `inspect-dataset` and `diff-dataset` surface this deterministically through `source_quality_metrics`.

---

## Phase 5 Hardening Notes

- Production `DatasetSpec` files can now declare `required_raw_brokers`, `require_synchronized_raw_coverage`, `require_dual_source_overlap`, and `min_dual_source_ratio`. These are compiler-owned spec expectations and are enforced by truth as hard publication requirements.
- The current production ML specs, [config/datasets/xau_m1_core_v1.yaml](/C:/Users/marti/Downloads/Datapipe/config/datasets/xau_m1_core_v1.yaml) and [config/datasets/xau_m1_nonhuman_v1.yaml](/C:/Users/marti/Downloads/Datapipe/config/datasets/xau_m1_nonhuman_v1.yaml), now target the verified synchronized live slice `2026-03-30` through `2026-04-02`.
- Raw tick ingest accounting now reports net-new rows added after dedup rather than the final row count of rewritten parquet partitions. Rerunning a synchronized backfill over a complete slice should show `rows_added=0`.
- State, feature, and label manifests now point at compact artifact-scoped parquet roots keyed by artifact id/version. Public loaders prefer those immutable roots when an artifact id is available, then fall back to legacy logical roots for compatibility.
- Canonical merge diagnostics and daily merge QA rewrite one canonical file per date on rerun. Downstream source-quality summaries should therefore see one authoritative daily row rather than duplicate rerun artifacts.
