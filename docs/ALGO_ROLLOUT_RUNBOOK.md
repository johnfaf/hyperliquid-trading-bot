# Decision-core rebuild — rollout runbook

Two flag-gated tranches: the decision-core rebuild (PRs #72–#77) and the
signal-quality layer (PRs #79–#89). **Every flag defaults OFF/0/neutral**, so
merging changed nothing live; each is enabled per-deployment via a Railway
environment variable. This runbook is the safe order to turn them on, what to
watch, and how to roll back. Exact copy-pasteable commands are in
[Exact Railway commands](#exact-railway-commands-operator-runbook).

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

### Signal-quality layer (PRs #79–#89)

A second tranche of flag-gated upgrades that improve **which signals are trusted**
before the decision core ranks them. All default OFF/0/neutral; most are
**observe-first** (inert until the dashboard shows the source/sub-book has cleared
its sample floor). Watch the data accrue on the **Sources** tab (`Signal IC` and
`Copy sub-book edges` panels, added in PR #89) before enabling.

| Sig / PR | Flag (Railway env) | Default | What it does | Precondition to enable |
|---|---|---|---|---|
| #3 / 85 | `FIREWALL_CONFLUENCE_MIN_CONFIRMATIONS` | `0` | Non-copy entry needs ≥N independent confirmations (options / volume / regime / cross-venue). copy-exempt | Phase 1; start at `2` |
| #4 / 86 | `REGIME_ROUTING_ENABLED` (+ `REGIME_ROUTING_MIN_COMPAT=0.3`) | `false` | Drop strategies whose regime compatibility < min (emit-where-it-works, not just pause) | Phase 1 |
| #5 / 87 | `COPY_SUBBOOK_EDGE_ENABLED` (+ `COPY_SUBBOOK_MIN_SAMPLES=8`, `COPY_SUBBOOK_MIN_EDGE=0.50`) | `false` | Gate copy per (wallet, coin, side) sub-book; drop proven-flat/negative books. Unmeasured books still pass | `Copy sub-book edges` panel shows ≥1 book cleared n≥8 |
| #8 / 76 (PR-B) | `COPY_FORWARD_EDGE_ENABLED` (+ `…_LOOKBACK_DAYS=60`, `…_HALF_LIFE_DAYS=14`) | `false` | Rank copy wallets by recency-weighted forward edge, not all-time PnL (survivorship-resistant) | After sub-book data accrues |
| #6 / 88 | `MICROSTRUCTURE_IC_WEIGHT_ENABLED` (+ `MICROSTRUCTURE_IC_MIN_N=20`, `_GAIN=2.5`, `_MIN_WEIGHT=0.25`, `_MAX_WEIGHT=1.5`) | `false` | Scale a non-copy source's confidence by its measured IC. copy-exempt; neutral until a source clears `MIN_N` | `Signal IC` panel shows several sources graded `PREDICTIVE`/`NEGATIVE` at n≥20 |
| #7 / 81 | `CROSS_SECTIONAL_ENABLED` (+ `CROSS_SECTIONAL_TOP_K=3`, `_LOOKBACK=24`) | `false` | Inject a market-neutral basket: long top-K / short bottom-K by trailing momentum | Optional; independent of calibration |
| #8 / 82 (PR-A) | `META_LABEL_ENABLED` (+ `META_LABEL_MIN_MULT=0.25`, `_MAX_MULT=1.5`) | `false` | Scale position size by the trade's calibrated win-prob (meta-label) | **Phase 2** — live-money-sensitive, needs warm calibration |
| #8 / 84 (PR-C) | `ML_REQUIRE_OOS_BEATS_BASELINE` (+ `ML_OOS_MIN_MARGIN=0.0`, `ML_OOS_EMBARGO=1`) | `false` | Disable ML models that don't beat a majority-class baseline out-of-sample (purged walk-forward) | Anytime; tightens ML trust |

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

## Exact Railway commands (operator runbook)

> **These are for the operator to run — they were NOT applied by tooling.**
> Enabling a flag is a live, real-money change on the ~$91 wallet. Run them
> yourself (CLI below, or the Railway dashboard → Variables), one phase at a
> time, and watch a full restart cycle before the next phase.

**Current live state** (read 2026-06-13, secrets excluded): only
`FIREWALL_SHADOW_MODE_FRACTION=0.10` is set; every decision-core and
signal-quality flag is unset and running its OFF/neutral default.

### Step 0 — deploy `main` as-is (no flag changes)
Merging the signal-quality PRs changed nothing live. Deploy first so the new
**measurement loops** (IC, copy-slippage, sub-book outcomes) start logging
forward data and the new **Sources** panels populate. Nothing to set.

### Phase 0 — foundation + data (safe to run now; zero behaviour change)
```bash
railway variables --set "CALIBRATION_HIERARCHICAL_ENABLED=true"
railway variables --set "FIREWALL_SHADOW_MODE_FRACTION=0.5"   # currently 0.10
```
Confidence starts differentiating off the 0.50 floor; shadow outcomes feed
calibration ~5× faster without risking capital. **Let this run 1–2 weeks.**

### Phase 1 — selective tightening (after Phase 0 warms; never loosens)
Enable the data-independent selectors first:
```bash
railway variables --set "DECISION_EV_FIRST_ENABLED=true"
railway variables --set "PORTFOLIO_NET_EXPOSURE_CAP_ENABLED=true"
railway variables --set "COPY_TRADER_EDGE_RANK_ENABLED=true"
railway variables --set "REGIME_ROUTING_ENABLED=true"
railway variables --set "FIREWALL_CONFLUENCE_MIN_CONFIRMATIONS=2"
```
Then, **gated on the Sources tab** showing real forward data:
```bash
# when Copy sub-book edges shows ≥1 book that cleared n≥8:
railway variables --set "COPY_SUBBOOK_EDGE_ENABLED=true"
railway variables --set "COPY_FORWARD_EDGE_ENABLED=true"
# when Signal IC shows several sources graded PREDICTIVE/NEGATIVE at n≥20:
railway variables --set "MICROSTRUCTURE_IC_WEIGHT_ENABLED=true"
```

### Phase 2 — live-money-sensitive (LAST, once calibration is warm)
```bash
railway variables --set "LEVERAGE_EDGE_PROPORTIONAL_ENABLED=true"
railway variables --set "LIVE_MIRROR_REQUIRE_PROVEN_ENABLED=true"
railway variables --set "META_LABEL_ENABLED=true"
```

### Optional / independent (any time)
```bash
railway variables --set "CROSS_SECTIONAL_ENABLED=true"          # market-neutral basket
railway variables --set "ML_REQUIRE_OOS_BEATS_BASELINE=true"    # tighten ML trust
```

**After each command:** `/api/live_ready` must stay `ready: true`; check the
Sources `Signal IC` / `Copy sub-book edges` panels and `/api/why_not_trading`
for sane new gate reasons. **Rollback** = set the flag back to `false`/default
(takes effect next restart; no deploy).

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
