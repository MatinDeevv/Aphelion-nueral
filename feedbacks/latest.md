Phase 6 Complete. Phase 7 Start.

Status

Phase 6 is accepted as complete.
  - 64 passing tests, zero failures
  - Full stack: data → TFT → training → evaluation
  - Attention weights, conformal-ready quantile heads,
    walk-forward splitter, inference loader all hardened
  - Coordination and contract discipline verified

Phase 7 has four simultaneous goals:

  1. Expand the dataset — wider synchronized history
  2. Regime detection + MoE gating
  3. Conformal prediction + position sizing interface
  4. Backtest engine

These are coordinated across 3 agents. The dependency
order matters:

  dataset width → regime detection → MoE gating
  TFT quantile outputs → conformal → position sizing
  signals + sizing → backtest

Agent ownership:
  Agent 1: DataPipeline — dataset expansion
  Agent 2: machinelearning/regime/ — regime + MoE
  Agent 3: machinelearning/signal/ and APH/backtest/

Context that must not be lost:

  Baseline to beat: walk_forward_balanced_accuracy = 0.5354
  Live feed: both brokers running, synchronized
  Production slice: 2026-03-30..2026-04-02 (4 days)
  Target: expand to at least 90 days of synchronized history
  Purpose: paper trading first, then live

What Phase 7 must not do:

  - Do not touch the Phase 6 machinelearning/ stack internals
    unless a specific cross-phase contract requires it
  - Do not retrain the TFT until Agent 1 has delivered
    the wider dataset and it passes the truth gate
  - Do not weaken the DataPipeline truth gate to
    make the wider range compile faster
  - Do not build a live execution layer yet —
    paper trading first means signal generation and
    backtest only, no order submission

Coordination discipline: same rules as always.
  feedbacks/latest.md        human steering
  chat/contracts.md          boundary changes only
  chat/coordination.md       blockers and handoffs
  chat/agent_{1,2,3}.md      per-agent logs

Phase 7 acceptance criteria:

  1. At least 60 days of dual-broker synchronized
     history compiled and truth-gated
  2. HMM regime detector running on real feature data
  3. MoE gating layer tested against TFT encoder output
  4. Conformal prediction intervals on TFT quantile outputs
  5. SignalRecord contract defined and implemented
  6. Kelly-based position sizing consuming SignalRecords
  7. Vectorized backtest engine running on historical signals
  8. Backtest produces Sharpe, max drawdown, win rate
     that are comparable to the gaussian_nb baseline

All agents must read this file before starting.