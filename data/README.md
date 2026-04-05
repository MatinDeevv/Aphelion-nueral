# MT5Pipe

MT5Pipe is a research-grade MetaTrader 5 market data system that now operates at two levels:

1. As a production ingestion and market-data pipeline.
2. As a Phase 1 dataset compiler with explicit contracts, lineage, artifact manifests, and truth-gated publication.

In plain language: this repository no longer stops at "download ticks and save parquet." It is designed to capture broker reality, fuse it into canonical market representations, build deterministic research views, manufacture labels, and publish versioned dataset artifacts only if they pass hard quality gates.

## What This Repository Is

At the operational layer, MT5Pipe:

- connects to one or more Windows MT5 terminals
- backfills or streams ticks, bars, account state, history, and snapshots
- stores all outputs in partitioned parquet under a local storage root
- merges raw multi-broker ticks into a canonical stream
- builds bars across configured timeframes

At the research layer, MT5Pipe now also:

- materializes compiler-era `state`, `feature_view`, `label_view`, and `dataset` artifacts
- registers them in a SQLite metadata catalog
- emits immutable manifests with upstream lineage
- evaluates candidate dataset artifacts with a truth gate
- publishes only accepted artifacts through logical dataset aliases

This means the system now sits between a market-data pipeline and a compact dataset operating system.

## Current Scope

What is production-ready in this repository today:

- MT5 historical and live ingestion
- resumable backfill with checkpointing
- synchronized dual-broker raw tick backfill
- canonical merge with preserved audit columns
- daily merge QA reporting and optional bucket sweeps
- bar building from canonical ticks
- legacy dataset building
- Phase 1 compiler-backed dataset builds with manifests, catalog, and trust reports

What is intentionally not implemented yet:

- exotic bar families like volume, imbalance, or range bars
- complex event ingestion and event-state modeling
- large feature libraries beyond the Phase 1 core feature families
- a full replay / stress-testing lab
- distributed object storage or an external metadata service

## Conceptual Model

The guiding model for this repository is:

```text
MT5 terminals
  -> raw immutable market capture
  -> canonical broker fusion
  -> built bars and market state
  -> point-in-time-safe features
  -> future-looking labels
  -> compiler dataset artifact
  -> truth-gated publication
```

Or, in the actual compiler-first vocabulary used by the repo:

```text
raw -> canonical -> bars -> state -> features -> labels -> dataset -> trust -> published artifact
```

This distinction matters:

- raw data is the record of observed broker outputs
- canonical data is the system's best aligned representation of market quotes
- bars are deterministic time aggregations of canonical ticks
- state is the machine-readable market snapshot used by the compiler
- features are point-in-time-safe transformations
- labels are future-looking targets with explicit purge rules
- datasets are versioned, lineage-tracked research artifacts

## Research Philosophy

This codebase is opinionated in a way that matters for real ML research:

- UTC everywhere
- append-only raw capture
- deterministic transforms from immutable upstream partitions
- explicit point-in-time boundaries
- explicit separation between features and future-looking labels
- temporal-only dataset splitting
- artifact lineage as a first-class output
- hard QA gates before publication

The repository is trying to answer the question:

> "Can we manufacture a trustworthy research dataset from broker-native reality, while preserving enough provenance to reproduce, inspect, compare, and reject artifacts rigorously?"

That is the purpose of the compiler, manifest, catalog, and truth layers.

## End-to-End Architecture

```text
                        +--------------------+
                        |   MT5 Terminal A   |
                        +--------------------+
                                  |
                        +--------------------+
                        |   MT5 Terminal B   |
                        +--------------------+
                                  |
                                  v
                    +---------------------------+
                    |   Ingestion / Backfill    |
                    | raw ticks, bars, history  |
                    +---------------------------+
                                  |
                                  v
                    +---------------------------+
                    | Canonical Merge + QA      |
                    | aligned broker fusion     |
                    +---------------------------+
                                  |
                                  v
                    +---------------------------+
                    | Bar Builder               |
                    | multi-timeframe bars      |
                    +---------------------------+
                                  |
                                  v
                    +---------------------------+
                    | State Engine              |
                    | StateSnapshot artifacts   |
                    +---------------------------+
                                  |
                 +----------------+----------------+
                 |                                 |
                 v                                 v
       +-----------------------+         +-----------------------+
       | Feature Fabric        |         | Label Factory         |
       | feature_view artifacts|         | label_view artifacts  |
       +-----------------------+         +-----------------------+
                 |                                 |
                 +----------------+----------------+
                                  |
                                  v
                    +---------------------------+
                    | Dataset Compiler          |
                    | dataset artifacts         |
                    +---------------------------+
                                  |
                                  v
                    +---------------------------+
                    | Truth Layer               |
                    | publish / reject decision |
                    +---------------------------+
                                  |
                                  v
                    +---------------------------+
                    | Catalog + Manifest Layer  |
                    | aliases, lineage, inspect |
                    +---------------------------+
```

## Subsystem Map

### Core Python Packages

| Package | Role |
| --- | --- |
| `mt5pipe/backfill/` | Historical backfill engine and synchronized dual-broker raw tick backfill |
| `mt5pipe/live/` | Live market collection loop |
| `mt5pipe/merge/` | Canonical tick merge logic |
| `mt5pipe/bars/` | Bar construction from canonical ticks |
| `mt5pipe/state/` | Compiler-era state artifact materialization |
| `mt5pipe/features/` | Feature builders, feature packages, dataset helpers |
| `mt5pipe/labels/` | Label registry and label-view materialization |
| `mt5pipe/compiler/` | Dataset spec loading, manifests, compiler service |
| `mt5pipe/truth/` | Truth reports and publication gating |
| `mt5pipe/catalog/` | SQLite metadata catalog |
| `mt5pipe/quality/` | Cleaning, gap detection, merge QA, dataset quality metrics |
| `mt5pipe/storage/` | Storage path conventions, parquet store, checkpoint DB |
| `mt5pipe/cli/` | Typer CLI entrypoints |
| `mt5pipe/tools/` | Operator-facing TUI / orchestrator tooling |

### Feature Package Layout

The feature layer was reorganized to be easier to reason about and extend:

| Package | Responsibility |
| --- | --- |
| `mt5pipe/features/time/` | cyclical time-of-day and weekday features |
| `mt5pipe/features/session/` | Asia, London, New York, and overlap flags |
| `mt5pipe/features/quality/` | spread- and conflict-derived quality signals |
| `mt5pipe/features/context/` | higher-timeframe lagged context joins |

`mt5pipe/features/builder.py` remains as a compatibility facade so older imports still work.

## Data Contracts

Phase 1 introduced typed contracts for every compiler stage.

### `StateSnapshot`

Defined in `mt5pipe/state/models.py`.

Purpose:

- represent a machine-readable market state row at a specific compiler clock
- preserve bid/ask/mid/spread semantics
- retain provenance and trust flags

Important fields:

- `state_version`
- `symbol`
- `ts_utc`, `ts_msc`
- `clock`
- `window_start_utc`, `window_end_utc`
- `bid`, `ask`, `mid`, `spread`
- `source_count`
- `merge_mode`
- `conflict_flag`
- `quality_score`
- `session_code`
- `provenance_refs`

Important invariants:

- `bid > 0`, `ask > 0`
- `bid <= ask`
- `mid == (bid + ask) / 2`
- `spread == ask - bid`
- `ts_utc` must lie inside the snapshot window

### `FeatureSpec`

Defined in `mt5pipe/features/registry/models.py`.

Purpose:

- declare a feature as a registered, versioned, point-in-time-safe contract

Important fields:

- `family`
- `feature_name`
- `version`
- `builder_ref`
- `output_columns`
- `dependencies`
- `point_in_time_safe`
- `missingness_policy`
- `qa_policy_ref`
- `status`

The default registry currently includes:

- `time.cyclical_time@1.0.0`
- `session.session_flags@1.0.0`
- `quality.spread_quality@1.0.0`
- `htf_context.standard_context@1.0.0`

### `LabelPack`

Defined in `mt5pipe/labels/registry/models.py`.

Purpose:

- declare a bundle of future-looking targets with shared horizons, exclusions, and purge rules

The default pack is:

- `core_tb_volscaled@1.0.0`

It materializes:

- future returns
- direction labels
- vol-scaled triple-barrier labels
- MAE / MFE

at horizons:

- `5m`
- `15m`
- `60m`
- `240m`

### `DatasetSpec`

Defined in `mt5pipe/compiler/models.py`.

Purpose:

- define a dataset build as a reproducible compiler input

Important fields:

- `dataset_name`
- `version`
- `symbols`
- `date_from`, `date_to`
- `base_clock`
- `state_version_ref`
- `feature_selectors`
- `label_pack_ref`
- `filters`
- `split_policy`
- `embargo_rows`
- `truth_policy_ref`

Current important limitation:

- Phase 1 currently supports exactly one symbol per `DatasetSpec`

### `LineageManifest`

Defined in `mt5pipe/compiler/models.py`.

Purpose:

- make every compiled artifact reproducible and inspectable

It records:

- immutable `artifact_id`
- logical name and version
- physical artifact URI
- content hash
- build ID
- linked dataset spec
- state artifact refs
- feature spec refs
- label pack ref
- input partition refs
- parent artifact refs

### `TrustReport`

Defined in `mt5pipe/truth/models.py`.

Purpose:

- provide the publication decision for a candidate artifact

It includes:

- `trust_score_total`
- `coverage_score`
- `leakage_score`
- `feature_quality_score`
- `label_quality_score`
- `source_quality_score`
- `lineage_score`
- `hard_failures`
- `warnings`
- per-check results

Publication is not advisory. It is gated.

## Storage Model

The active storage root now defaults to:

```text
local_data/pipeline_data/
```

This is configured in:

- `config/pipeline.yaml`
- `mt5pipe/config/models.py`

### Why `local_data/` Exists

The repository intentionally separates code from local artifacts:

- `local_data/pipeline_data/` - active storage root used by the pipeline
- `local_data/clean_data/` - curated exports you want separated from pipeline internals
- `local_data/archive_data/` - scratch or archived local files

Only `local_data/README.md` is tracked by git. The rest is intentionally ignored.

### Storage Path Taxonomy

All storage path conventions live in `mt5pipe/storage/paths.py`.

Key artifact families:

| Artifact Family | Example |
| --- | --- |
| Raw ticks | `raw_ticks/broker=broker_a/symbol=XAUUSD/date=2026-04-01/part-00000.parquet` |
| Native bars | `native_bars/broker=broker_a/symbol=XAUUSD/timeframe=M1/date=2026-04-01/part-00000.parquet` |
| Canonical ticks | `canonical_ticks/symbol=XAUUSD/date=2026-04-01/part-00000.parquet` |
| Merge diagnostics | `merge_diagnostics/symbol=XAUUSD/date=2026-04-01/part-00000.parquet` |
| Daily merge QA | `merge_qa/symbol=XAUUSD/date=2026-04-01/part-00000.parquet` |
| Built bars | `bars/symbol=XAUUSD/timeframe=M1/date=2026-04-01/part-00000.parquet` |
| State artifacts | `state/symbol=XAUUSD/clock=M1/state_version=state.default@1.0.0/date=2026-04-01/part-00000.parquet` |
| Feature views | `feature_views/feature=time.cyclical_time@1.0.0/clock=M1/date=2026-04-01/part-00000.parquet` |
| Label views | `label_views/label_pack=core_tb_volscaled@1.0.0/clock=M1/date=2026-04-01/part-00000.parquet` |
| Compiler dataset artifacts | `datasets/name=xau_m1_core/artifact=<artifact_id>/split=train/part-00000.parquet` |
| Manifests | `manifests/kind=dataset/name=xau_m1_core/<manifest_id>.json` |
| Truth reports | `truth/artifact=<artifact_id>/<report_id>.json` |
| Catalog DB | `catalog/catalog.db` |

## Metadata Catalog

The compiler metadata catalog lives in SQLite and is implemented in `mt5pipe/catalog/sqlite.py`.

This is not the same thing as the ingestion checkpoint database.

The compiler catalog tracks:

- registered feature specs
- registered label packs
- registered dataset specs
- build runs
- artifact records
- artifact inputs
- trust reports
- per-check QA results
- logical artifact aliases

This allows:

- immutable artifact lookup
- logical alias resolution like `dataset://xau_m1_core@1.0.0`
- inspection of build provenance
- dataset diffing across versions or artifacts

## Canonical Merge and Merge QA

The canonical merge logic lives in `mt5pipe/merge/canonical.py`.

Important design choice:

- merge logic itself was intentionally not redesigned during the Phase 1 compiler work

The repository now also includes a stronger merge-quality workflow:

- synchronized dual-broker raw tick backfill
- explicit overlap validation
- daily merge QA output
- optional bucket sweeps after QA generation

Daily merge QA currently reports:

- raw tick counts by broker
- canonical rows
- dual-source participation
- bucket counts
- near-miss estimates
- validation rejects
- offset medians and tails
- session participation
- gap counts and max gap length

The truth layer can then use merge QA outputs as part of source-quality scoring.

## State Engine

The state engine lives in `mt5pipe/state/service.py`.

In Phase 1, state artifacts are built from already constructed bars rather than directly from tick-by-tick order-book state. This is deliberate: it creates a stable first compiler contract without redesigning the canonical merge pipeline.

The state engine currently:

- loads built bars across a requested date range
- validates and gap-fills bar series where appropriate
- derives bid/ask/mid/spread semantics
- assigns session codes
- computes a lightweight quality score
- emits `StateSnapshot` rows
- writes state partitions and a state manifest

This gives the compiler a formal base artifact instead of directly wiring the legacy dataset builder into the final dataset step.

## Feature Fabric

Feature materialization lives in `mt5pipe/features/service.py`.

Each feature family is selected through registry resolution, then materialized into its own first-class feature-view artifact.

The current Phase 1 feature families are intentionally small and conservative:

- `time/*`
- `session/*`
- `quality/*`
- `htf_context/*`

The higher-timeframe context features are joined in a point-in-time-safe way. They use lagged closed bars rather than current in-progress higher-timeframe bars, which is one of the core anti-leakage guarantees in this repository.

## Label Factory

Label materialization lives in `mt5pipe/labels/service.py`.

The default label pack builds:

- future returns
- direction labels
- triple-barrier labels
- MAE / MFE

Labels are materialized after state and features. The compiler then purges trailing rows according to `label_pack.purge_rows` so the final published dataset does not include rows whose future horizon is incomplete.

This separation is important:

- features must be available at prediction time
- labels intentionally use future data
- the compiler keeps those responsibilities separate

## Dataset Compiler

The compiler lives in `mt5pipe/compiler/service.py`.

The compiler is responsible for:

1. loading a `DatasetSpec`
2. registering the spec in the catalog
3. materializing state artifacts
4. materializing feature-view artifacts
5. materializing label-view artifacts
6. joining them into a candidate dataset frame
7. applying dataset filters
8. cleaning and selecting published columns
9. splitting the dataset temporally
10. writing dataset artifact partitions
11. writing a lineage manifest
12. invoking the truth layer
13. publishing aliases only if accepted

Current split policies:

- `temporal_holdout`
- `walk_forward`

Current compiler behavior:

- no random shuffling
- no cross-sectional multi-symbol builds yet
- embargo rows are enforced
- published aliases are only written on acceptance

## Truth Layer

The truth layer lives in `mt5pipe/truth/service.py`.

This is one of the most important architectural upgrades in the repository.

The truth layer checks:

- dataset coverage and empty splits
- duplicate timestamps and PIT safety
- feature contract completeness
- label contract completeness
- source quality from merge QA
- lineage completeness

Its scoring model currently combines:

- coverage
- leakage
- feature quality
- label quality
- source quality
- lineage quality

Publication requires:

- no hard failures
- total trust score above threshold
- coverage score above threshold
- leakage score equal to `100`
- lineage score equal to `100`

This means a dataset can be built but still rejected from publication.

## CLI Overview

All commands are exposed through:

```bash
mt5pipe --help
```

### Backfill

```bash
mt5pipe backfill ticks --broker broker_a --symbol XAUUSD --from 2026-04-01 --to 2026-04-04
mt5pipe backfill bars --broker broker_a --symbol XAUUSD --timeframe M5 --from 2026-04-01 --to 2026-04-04
mt5pipe backfill history-orders --broker broker_a --from 2026-04-01 --to 2026-04-04
mt5pipe backfill history-deals --broker broker_a --from 2026-04-01 --to 2026-04-04
mt5pipe backfill symbol-metadata --broker broker_a
```

### Synchronized Dual-Broker Tick Backfill

```bash
mt5pipe backfill sync-ticks \
  --broker-a broker_a \
  --broker-b broker_b \
  --symbol XAUUSD \
  --from 2026-04-01 \
  --to 2026-04-04
```

Optional liquid-hours-only window:

```bash
mt5pipe backfill sync-ticks \
  --broker-a broker_a \
  --broker-b broker_b \
  --symbol XAUUSD \
  --from 2026-04-01 \
  --to 2026-04-04 \
  --hours-start 06:00 \
  --hours-end 22:00
```

### Live Collection

```bash
mt5pipe live collect --broker broker_a --symbol XAUUSD --duration 0
```

Optional market-book collection is enabled by default and can be disabled with CLI flags if needed.

### Canonical Merge and QA

```bash
mt5pipe merge canonical \
  --broker-a broker_a \
  --broker-b broker_b \
  --symbol XAUUSD \
  --from 2026-04-01 \
  --to 2026-04-04
```

```bash
mt5pipe merge qa-report \
  --broker-a broker_a \
  --broker-b broker_b \
  --symbol XAUUSD \
  --from 2026-04-01 \
  --to 2026-04-04
```

```bash
mt5pipe merge bucket-sweep \
  --broker-a broker_a \
  --broker-b broker_b \
  --symbol XAUUSD \
  --from 2026-04-01 \
  --to 2026-04-04 \
  --buckets 50,75,100,125
```

### Bar Building

```bash
mt5pipe bars build --symbol XAUUSD --timeframe M5 --from 2026-04-01 --to 2026-04-04
```

If `--timeframe` is omitted, all configured timeframes are built.

### Legacy Dataset Build

The legacy path still exists and is intentionally preserved:

```bash
mt5pipe dataset build --symbol XAUUSD --from 2026-04-01 --to 2026-04-30 --name default
```

This is useful for continuity, but the compiler path is the more important research interface going forward.

### Compiler Dataset Build

```bash
mt5pipe dataset compile-dataset --spec config/datasets/xau_m1_core_v1.yaml
```

### Inspect a Published or Materialized Artifact

```bash
mt5pipe dataset inspect-dataset --artifact dataset://xau_m1_core@1.0.0
```

You can also inspect by:

- immutable artifact ID
- manifest file path
- logical alias

### Diff Two Dataset Artifacts

```bash
mt5pipe dataset diff-dataset \
  --left dataset://xau_m1_core@1.0.0 \
  --right <other-artifact-or-manifest>
```

### Status and Validation

```bash
mt5pipe status show
mt5pipe status validate
mt5pipe validate-storage
```

### Operator TUI

This repository includes a richer operator-facing TUI / orchestrator:

```bash
RUN_ME.bat
```

Or directly:

```bash
python -m mt5pipe.tools.super_pipeline_tui --interactive
```

Console script:

```bash
mt5pipe-super --from 2026-04-01 --to 2026-04-30
```

## Example Dataset Spec

The default Phase 1 example is:

`config/datasets/xau_m1_core_v1.yaml`

It currently defines:

- symbol: `XAUUSD`
- date range: `2026-04-01` to `2026-04-30`
- base clock: `M1`
- state version: `state.default@1.0.0`
- feature selectors: `time/*`, `session/*`, `quality/*`
- label pack: `core_tb_volscaled@1.0.0`
- filter: `exclude:filled_rows`
- split policy: `temporal_holdout`
- embargo: `240` rows

This is the current reference configuration for compiler-backed datasets.

## Installation

### Requirements

- Windows
- Python `>= 3.11`
- installed MT5 terminal(s)
- `MetaTrader5` Python package support

### Install

```bash
pip install -e ".[dev]"
```

Core dependencies:

| Package | Purpose |
| --- | --- |
| `MetaTrader5` | MT5 terminal connectivity |
| `polars` | dataframe engine |
| `pyarrow` | parquet I/O |
| `pydantic` | config and contract validation |
| `typer` | CLI surface |
| `rich` | terminal rendering |
| `structlog` | structured logging |
| `tenacity` | retry behavior |
| `PyYAML` | config/spec loading |

## Configuration

Primary runtime config:

- `config/pipeline.yaml`

Secrets template:

- `config/example_secrets.env`

Typical workflow:

1. Copy `config/example_secrets.env` to your own env file.
2. Update broker credentials and terminal paths.
3. Adjust symbol maps if broker symbols differ.
4. Keep `storage.root` pointed at `local_data/pipeline_data`.

## Reproducibility and Artifact Identity

The repository now distinguishes between:

- logical dataset identity
- immutable artifact identity

Examples:

- logical alias: `dataset://xau_m1_core@1.0.0`
- latest alias: `dataset://xau_m1_core:latest`
- immutable artifact: generated content-addressed artifact ID

This matters because:

- research should refer to immutable artifacts
- humans often want stable logical dataset names
- the catalog can map the latter to the former

## Testing

The test suite covers:

- bar building correctness
- canonical merge scoring and diagnostics
- checkpoint persistence
- contract validation
- compiler artifact generation
- catalog resolution
- label logic
- anti-leakage behavior
- merge QA aggregation
- dataset cleaning and quality reporting

Run:

```bash
pytest
```

Or:

```bash
python -m pytest
```

## Important Operational Constraints

### MT5 API Constraint

The MT5 Python API is effectively process-global. In practice this means one terminal connection per process, which is why this repository structures multi-broker work carefully rather than pretending the API is truly multi-tenant.

### Canonical Merge Scope

The canonical merge layer is working and validated for synchronized overlap windows, but historical dataset coverage is still an operational problem, not a solved theoretical one. Coverage quality should always be checked through merge QA and source-quality metrics.

### Compiler Phase 1 Scope

Phase 1 is intentionally modest in semantics even though the architecture is much stronger:

- state is derived from bars, not a full event-state graph
- feature families are narrow and conservative
- label packs are limited
- the truth gate is real, but still compact

That is a feature, not a bug. The repository now has the contracts and scaffolding needed to grow without degenerating into script sprawl.

## Suggested Research Workflow

For a disciplined research loop:

1. Backfill both brokers over the same UTC range.
2. Run canonical merge.
3. Generate and inspect daily merge QA.
4. Build bars for the target clocks.
5. Compile a dataset from a `DatasetSpec`.
6. Inspect the resulting manifest and truth report.
7. Compare dataset artifacts with `diff-dataset` when changing specs.
8. Train only against immutable artifact IDs, not mutable folder names.

That workflow gives you:

- provenance
- PIT safety
- comparable experiments
- dataset-level QA discipline

## Repository Layout

```text
Datapipe/
  config/
    datasets/
    example_secrets.env
    pipeline.yaml
  local_data/
    README.md
  mt5pipe/
    backfill/
    bars/
    catalog/
    cli/
    compiler/
    config/
    features/
      context/
      quality/
      registry/
      session/
      time/
    ingestion/
    labels/
      registry/
    live/
    merge/
    models/
    mt5/
    quality/
    state/
    storage/
    tools/
    truth/
    utils/
  scripts/
  tests/
  RUN_ME.bat
  sxcas.py
  pyproject.toml
```

## Bottom Line

MT5Pipe is no longer just a broker data downloader.

It is a broker-aware market-data capture system with:

- canonical multi-broker fusion
- deterministic bar construction
- compiler-era state, feature, and label artifacts
- manifest-based lineage
- catalog-backed artifact resolution
- trust-gated dataset publication

That is the key mental model for the repository going forward.

## License

MIT
