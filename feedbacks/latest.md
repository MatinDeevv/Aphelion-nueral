Phase 5 Complete. Phase 6 Start.

Status

Phase 5 is accepted as complete. The production path is green:
- dataset://xau_m1_nonhuman@1.0.0 published, trust 98.11
- 215,777 canonical dual-source rows, dual_source_ratio=0.1639
- gaussian_nb_binary@1.0.0 baseline: walk_forward_balanced_accuracy=0.5354,
  holdout_balanced_accuracy=0.5096

That baseline is now the floor. Everything in Phase 6 is evaluated
against it. If you cannot beat 0.5354 walk-forward balanced accuracy,
you have not added value.

What Phase 6 is

Phase 6 is the deep learning stack in APH/machinelearning/.

This is a separate package from mt5pipe. It does NOT touch mt5pipe
internals. It reads from published dataset artifacts only — through
the parquet files that the compiler already produced. It does not
re-implement anything the DataPipeline already owns.

The center of gravity for Phase 6 is:
  APH/machinelearning/
    data/       Agent 1
    models/     Agent 2
    training/   Agent 3

The goal is a Temporal Fusion Transformer trained on
dataset://xau_m1_nonhuman@1.0.0, producing calibrated multi-horizon
signals that demonstrably beat the gaussian_nb_binary baseline.

What Phase 6 must not do

Do not re-implement the DataPipeline. Do not import mt5pipe internals.
Do not touch chat/contracts.md for mt5pipe boundary changes unless
something in mt5pipe actually needs to change. The machinelearning/
package is self-contained and reads parquet only.

Do not add complexity you cannot evaluate. Every architectural decision
must be justified against the baseline. If it does not move the IC or
balanced accuracy, it is not in scope.

Coordination discipline

Same rules as always:
  feedbacks/latest.md     human steering — read before starting
  chat/contracts.md       machinelearning/ boundary changes only
  chat/coordination.md    blockers / handoffs between agents
  chat/agent_{1,2,3}.md  per-agent logs

Agent 3 owns training/ and is the primary quant researcher on this
phase. Agents 1 and 2 build what Agent 3 needs to train.

Frozen from Phase 5

Do not change:
  - mt5pipe package structure
  - DatasetSpec / ExperimentSpec / TrustReport contracts
  - The canonical dataset artifact and its published alias
  - Coordination file discipline

Phase 6 acceptance criteria

Phase 6 is complete only when:
  1. machinelearning/ is a working importable Python package
  2. AphelionTFT trains end-to-end on the published parquet splits
  3. val/ic_60m > 0 (the model has extracted some signal)
  4. walk-forward balanced accuracy on direction_60m > 0.5354
  5. The training run is logged to W&B with full hyperparameter config
  6. A trained model artifact is saved to APH/checkpoints/
  7. The normalizer is saved alongside the checkpoint for inference use

Required behavior before new work starts

All agents must:
  - Read this file before starting Phase 6 work
  - Summarize in their agent log what they are acting on
  - Log any conflict with their current prompt in chat/coordination.md