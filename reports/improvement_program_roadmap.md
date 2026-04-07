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

## Phase 8: Adaptive Execution Policy

- Status: shipped on `codex/live-ledger-foundation`
- Add maker/taker-aware execution policy recommendations before final expected-value ranking.
- Route low-urgency live entries through bounded ALO maker orders with safe market fallback.
- Persist execution-policy metadata end to end across paper, mirror, live, reporting, and dashboard telemetry.

## Phase 9: Source Budget Governance

- Status: shipped on `codex/live-ledger-foundation`
- Scale source-level capital using adaptive-learning health plus recent attribution and trade outcomes.
- Cap warming and caution sources to smaller portfolio slices before execution, instead of only ranking them lower.
- Block degraded sources outright and expose source-budget runtime state in reporting and the dashboard.

## Phase 10: Cross-Source Confluence And Conflict Gating

- Status: shipped on `codex/live-ledger-foundation`
- Build per-cycle same-coin confluence maps across the actual candidate set before final ranking.
- Reward candidates that have distinct-source agreement on the same coin and side.
- Penalize or block candidates when strong same-coin source conflict exists across the candidate set.

## Phase 11: Context Performance Gating

- Status: shipped on `codex/live-ledger-foundation`
- Score candidates using recent realized performance in the specific `source x coin x side x regime` context, not just source-wide quality.
- Block contexts that have enough sample size and are still materially underperforming after costs.
- Fall back from exact regime matches to mixed-regime context history when exact context is too sparse, while keeping the minimum sample gate intact.

## Phase 12: Calibration-Aware Confidence Gating

- Status: shipped on `codex/live-ledger-foundation`
- Feed recorded confidence-vs-outcome calibration back into the main decision engine before adaptive/context scoring.
- Adjust raw candidate confidence using the observed calibration curve for the source when enough history exists.
- Block sources that remain materially miscalibrated after enough samples, and reward candidates whose confidence has proven reliable.

## Phase 13: Trade Memory Decision Feedback

- Status: shipped on `codex/live-ledger-foundation`
- Use similar historical setups directly inside the main decision engine instead of only at paper-execution time.
- Reward candidates whose nearest historical setup cluster has strong realized outcomes after enough samples.
- Block candidates whose similar setup cluster is explicitly rated `avoid`, so repeated bad patterns are filtered before execution.

## Phase 14: Live-vs-Paper Divergence Control

- Status: shipped on `codex/live-ledger-foundation`
- Promote passive divergence reporting into a runtime control layer instead of leaving it as reporting-only telemetry.
- Block or taper new entries when paper, shadow, and live drift exceeds configured thresholds globally or by source.
- Expose divergence runtime state in the decision engine, source allocator, reporting cycle, and dashboard.

## Phase 15: Adaptive Capital Governor

- Status: shipped on `codex/live-ledger-foundation`
- Convert recent paper and live portfolio quality into a global risk budget instead of treating every cycle as full-risk by default.
- Taper or block new entries when drawdown, rolling return quality, source-health degradation, or runtime divergence indicate the system should derisk.
- Expose the capital posture in source allocation, decision ranking, reporting, and the dashboard so risk-off states are visible instead of silent.

## Branch Operating Rule

Each phase should be landed in a reviewable commit with tests before moving to the next phase.
