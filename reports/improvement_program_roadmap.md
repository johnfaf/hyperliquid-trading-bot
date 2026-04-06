# Improvement Program Roadmap

This branch is the staged upgrade path for making the bot more measurable, more selective, and safer to scale.

## Phase 1: Live Ledger Foundation

- Status: shipped on `codex/live-ledger-foundation`
- Persist live equity snapshots across cycles.
- Persist live position snapshots across cycles, including zero-position batches.
- Persist exchange fill events idempotently for lifetime realized PnL and fee tracking.
- Expose live-ledger summaries to the dashboard and API.
- Add regression tests so the new live ledger does not silently drift.

## Phase 2: Expected-Value Decision Engine

- Status: shipped on `codex/live-ledger-foundation`
- Replace pure composite-score ranking with expected net value ranking.
- Model fees, slippage, funding, churn cost, and regime-adjusted expectancy.
- Add source, coin, and regime calibration before final ranking.

## Phase 3: Research And Feature Store

- Status: shipped on `codex/live-ledger-foundation`
- Persist every cycle's candidate set, rejections, selected trades, and realized outcomes.
- Store regime, cross-venue, options-flow, polymarket, and arena context with each decision.
- Make every live and paper decision replayable for later training and audit.

## Phase 4: Portfolio Construction And Sizing

- Status: shipped on `codex/live-ledger-foundation`
- Move from per-trade sizing to portfolio-aware sizing.
- Add cluster exposure, side exposure, BTC beta, and volatility targeting controls.
- Make stops, targets, and time-in-trade exits adaptive instead of mostly fixed.

## Phase 5: Execution Quality

- Status: shipped on `codex/live-ledger-foundation`
- Push all exchange traffic through shared infra and execution telemetry.
- Track realized slippage, fill quality, maker/taker mix, and rejection reasons by source.
- Feed execution quality back into source scoring and expected-value penalties.

## Phase 6: Experiment Discipline

- Status: shipped on `codex/live-ledger-foundation`
- Add benchmark and challenger packs for strategy changes.
- Require out-of-sample comparisons before promoting new ranking logic.
- Publish decision funnels, source attribution, and divergence between paper, shadow, and live.

## Phase 7: Adaptive Learning And Promotion

- Status: shipped on `codex/live-ledger-foundation`
- Build drift-aware source-health profiles from realized paper/live outcomes.
- Feed adaptive health, calibration, and drift back into the main decision engine.
- Review arena agents continuously so degraded champions are demoted and recovered agents can return.
- Expose source health, arena review actions, and adaptive learning state in reporting and the dashboard.

## Branch Operating Rule

Each phase should be landed in a reviewable commit with tests before moving to the next phase.
