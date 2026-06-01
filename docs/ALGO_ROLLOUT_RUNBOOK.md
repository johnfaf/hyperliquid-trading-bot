# Decision-core rebuild — rollout runbook

Six flag-gated upgrades to the decision core (PRs #72–#77). **Every flag
defaults OFF**, so merging changed nothing live; each is enabled per-deployment
via a Railway environment variable. This runbook is the safe order to turn them
on, what to watch, and how to roll back.

## Why these exist (one-paragraph context)

The bot is a trader-following system whose downstream machinery (calibration,
EV gate, agent scoring, Kelly) is well-built but **data-starved**: outcomes
fragment across `(source|side|regime)` cells, so confidence collapsed to a flat
~0.50 cold-start prior and every gate went blunt. The rebuild makes the engine
**concentrate capital, leverage, and live risk on *proven* edge** once data
exists. The keystone is calibration (#2); everything else consumes its output.

## Flag reference

| # / PR | Flag (Railway env) | Default | What it does | Depends on |
|---|---|---|---|---|
| #2 / 72 | `CALIBRATION_HIERARCHICAL_ENABLED` | `false` | Thin calibration cells borrow strength from pooled parents (empirical-Bayes) so confidence becomes real instead of the 0.50 cap | — (keystone) |
| | `CALIBRATION_HIERARCHICAL_SHRINKAGE` | `10.0` | Beta-binomial K toward parent (higher = more conservative) | #2 |
| | `CALIBRATION_HIERARCHICAL_BLEND_N` | `10.0` | How fast the EB rate overrides raw confidence | #2 |
| #6 / 75 | `FIREWALL_SHADOW_MODE_FRACTION` | `0.0` | Fraction of rejected signals logged as shadows + marked to market → feeds calibration/EV the data they're missing | — |
| #3 / 73 | `DECISION_EV_FIRST_ENABLED` | `false` | Rank candidates by net-of-cost EV first (composite is the tie-breaker) + require `DECISION_MIN_EV_R` | #2 (uses calibrated conf) |
| | `DECISION_EV_COST_R` / `DECISION_MIN_EV_R` | `0.12` / `0.0` | Round-trip cost (R) / minimum EV (R) to qualify | #3 |
| #7 / 77 | `PORTFOLIO_NET_EXPOSURE_CAP_ENABLED` | `false` | Cap concurrent same-side (correlated) positions | — |
| | `PORTFOLIO_MAX_SAME_SIDE_POSITIONS` | `3` | The cap (existing open + new) | #7 |
| #4 / 76 | `COPY_TRADER_EDGE_RANK_ENABLED` | `false` | Rank copy wallets by *shrunk* win-rate edge (not raw PnL); drop sub-min | — |
| | `COPY_TRADER_EDGE_SHRINKAGE` / `COPY_TRADER_MIN_SHRUNK_EDGE` | `20.0` / `0.50` | Shrinkage K / minimum shrunk edge to copy | #4 |
| #5 / 74 | `LEVERAGE_EDGE_PROPORTIONAL_ENABLED` | `false` | Leverage scales with calibrated edge (unproven → 1×), not the followed trader's habit | #2 |
| | `LEVERAGE_EDGE_MIN_CONF` / `LEVERAGE_EDGE_FULL_CONF` | `0.50` / `0.65` | 1× below min; ramps to max at full | #5 |
| #1 / 75 | `LIVE_MIRROR_REQUIRE_PROVEN_ENABLED` | `false` | Only mirror to LIVE if the source has proven evidence; unproven keep paper-trading | #2, #6 |
| | `LIVE_MIRROR_PROVEN_MIN_SAMPLES` / `LIVE_MIRROR_PROVEN_MIN_EDGE` | `30` / `0.50` | Samples + edge required to risk live capital | #1 |

## Phased rollout (do NOT enable everything at once)

These levers only bite once the bot has **forward data** — the keystone
calibration and the "proven" gate are inert on a cold account. Enable the
foundation, let weeks of outcomes accumulate, then enable the consumers.

### Phase 0 — Foundation + data (enable now, no live behaviour change)
```
CALIBRATION_HIERARCHICAL_ENABLED=true
FIREWALL_SHADOW_MODE_FRACTION=0.5        # log half of rejected signals as shadows
```
- **Effect:** confidence starts differentiating as data accrues; shadow outcomes
  feed calibration 5–10× faster without risking capital.
- **Watch (1–2 weeks):** calibration ECE/sample sizes climbing; confidence
  spread widening off 0.50. No P&L change expected.

### Phase 1 — Selection + risk (conservative, can enable after Phase 0 warms)
```
DECISION_EV_FIRST_ENABLED=true
PORTFOLIO_NET_EXPOSURE_CAP_ENABLED=true   # PORTFOLIO_MAX_SAME_SIDE_POSITIONS=3
COPY_TRADER_EDGE_RANK_ENABLED=true
```
- **Effect:** strictly more selective ranking; fewer correlated stacked
  positions; copy concentrates on robustly-good wallets. Never loosens.
- **Watch:** trade count down modestly; blended win rate up; no large drawdowns.

### Phase 2 — Live-money-sensitive (enable LAST, once calibration is warm)
```
LEVERAGE_EDGE_PROPORTIONAL_ENABLED=true
LIVE_MIRROR_REQUIRE_PROVEN_ENABLED=true
```
- **Effect:** unproven sources drop to 1× and stop being live-mirrored; only
  proven sources earn leverage + live capital. **On a cold account these force
  ~everything to 1× / paper-only — safe but inert until data exists.** That's
  why they go last.
- **Watch:** live mirrors only for proven buckets; leverage concentrated on
  high-confidence sources; the 5× strategy/momentum bleed (walk-forward:
  −$350/7) eliminated from live.

## Monitoring

- `/api/live_ready` — must stay `ready: true`.
- `/api/why_not_trading` (admin) — confirm the new gate reasons appear and look
  sane (e.g. "unproven source", "correlation cap", "below min EV").
- Calibration ECE + per-source sample sizes (rising = the foundation working).
- Trade count + blended win rate + max per-trade loss (the risk levers should
  shrink the tail).

## Rollback

Every change is a single Railway env flip back to `false` (or the default),
which takes effect on the next restart — **no code deploy, no PR**. Because all
defaults are OFF, "set the var back to false" fully reverts any phase
independently. Start with the most recently enabled flag if you see regressions.

## Validation before enabling

Run the deterministic replay with the flags set in env and compare to baseline:
```
# baseline
python scripts/run_replay.py --start <s> --end <e> --step 4h --coins <...> \
  --cache-db data/candle_cache.db --strategy-snapshot data/pool.json \
  --fills-db data/fills.db --run-id stack_off
# treatment (flags ON via env), same window/run-id stack_on
python scripts/analyze_replay_pnl.py --replay-db data/replay_stack_on.db
```
Note: the replay starts with **cold calibration**, so #2/#1 are inert there;
the offline-measurable effects are #5 (leverage de-risk), #7 (correlation cap),
#4 (copy re-rank), #3 (EV gate). Warm-calibration validation requires forward
live/shadow data.
