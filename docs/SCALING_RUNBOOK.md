# Live-Trading Scaling Runbook

Graduated ramp-up from canary to full-scale trading on Hyperliquid.
Replaces the single `LIVE_CANARY_MODE` flag with a **five-rung ladder**
driven by one env var, `LIVE_TIER`, and a set of operator-verified
advance gates.

> **Principle.** Capital exposure only grows when the operator decides
> it should, on evidence, one rung at a time. The bot never
> auto-advances tiers based on its own reading of P&L. Every rung
> change is a deliberate restart with an explicit env-var flip.

---

## The ladder

| Tier | Label    | Max order | Max daily loss | Max position | Signals/day | Kelly dampen | Min days before advance |
|------|----------|-----------|----------------|--------------|-------------|--------------|-------------------------|
| T0   | canary   | $25       | $50            | $50          | 25          | 0.5          | 2                       |
| T1   | seed     | $50       | $100           | $150         | 40          | 0.65         | 3                       |
| T2   | growth   | $100      | $250           | $500         | 60          | 0.8          | 5                       |
| T3   | scale    | $250      | $600           | $1,500       | 100         | 0.9          | 7                       |
| T4   | full     | (env)     | (env)          | (env)        | (env)       | 1.0          | --                      |

`(env)` = tier imposes no override; limit comes entirely from
`LIVE_MAX_ORDER_USD` / `HL_MAX_DAILY_LOSS` / `LIVE_MAX_POSITION_SIZE_USD` /
`LIVE_CANARY_MAX_SIGNALS_PER_DAY` env vars.

### Safety invariant: **downward-only**

Tiers can only *tighten* limits below the operator's env-var configuration.
A tier never silently expands exposure.

- If you set `LIVE_MAX_ORDER_USD=500` and `LIVE_TIER=T2` (cap $100), the
  effective per-order cap is **$100** (tier wins, it's tighter).
- If you set `LIVE_MAX_ORDER_USD=15` and `LIVE_TIER=T3` (cap $250), the
  effective per-order cap is **$15** (operator config wins, it's tighter).
- If you set `LIVE_MAX_ORDER_USD=500` and `LIVE_TIER=T4`, the tier
  applies no override — effective cap is **$500**.

To advance to a higher dollar ceiling you must raise **both** the env
var and the tier. Either alone is a no-op in the expansion direction.

---

## Starting state

On `main` today the bot defaults to:

```env
LIVE_CANARY_MODE=true
LIVE_CANARY_MAX_ORDER_USD=25.0
LIVE_CANARY_MAX_SIGNALS_PER_DAY=25
```

This is effectively **T0 canary**. Keep it that way for your first
live session.

---

## Advancement checklist

Before flipping `LIVE_TIER` from `Tn` to `Tn+1`, every item below must
be green. If any is yellow/red, **stay on the current tier** and log why.

### 1. Time in tier
- Current tier has been the active tier for **at least the
  `min_days_recommended` count** for that tier (see table above).
- No partial days — each counted day must include a full trading
  session with the bot online.

### 2. Profit & loss
- Cumulative realized P&L over the tier's window is **>= $0**.
- **No single-day loss** exceeded 60% of the tier's `max_daily_loss`.
  (E.g. on T1 ($100 daily cap) no day may have lost more than $60.)
- **Max drawdown within the window <= 25%** of cumulative deposits.
- `daily_pnl` field in `/healthz` reviewed at end-of-day on each of the
  last N days.

### 3. Kill-switch events
- **Zero kill-switch activations** in the tier's window.
- If the kill switch fired (manual, daily-loss, or external file), the
  tier clock resets to 0 — you do not advance that day.
- Verify via `kill_switch_active=false` and `kill_switch_reason=""` in
  `/healthz` across the window.

### 4. Order quality
- **Fill ratio >= 95%**: `fills_today / orders_today >= 0.95` averaged
  across the window.
- **Rejection rate sane**: `entry_metrics.rejected_firewall +
  rejected_kill_switch + rejected_canary_cap + rejected_source_cap` is
  < 10% of `attempted_entry_signals`. Higher means signals are poor
  quality or guards are mis-tuned — don't scale a broken pipeline.
- **No emergency close retries** exceeded their retry budget. Check
  logs for `_emergency_close_retries` exhausted.

### 5. Margin & balance
- Wallet margin usage never exceeded **60%** during the window.
- `free_margin > 0` every minute — no `free_margin_blocked_since` gaps
  longer than 60s.
- Deposits cleared and reflected before advancement.

### 6. Infrastructure
- No Railway restarts caused by the bot itself (OOM, crash). Planned
  deploys are fine.
- `bot_backup.json` was successfully rehydrated at least once across a
  restart during the window (proves the atomic-write path works).
- Postgres dual-write mirror caught up — run
  `SELECT COUNT(*) FROM paper_trades WHERE closed_at > NOW() - INTERVAL '1 day'`
  on both SQLite and PG and compare.

---

## How to advance

1. **Snapshot current state** — save output of `/healthz` and the day's
   `daily_pnl` to a dated note.
2. **Update env vars.** In Railway (or your deployment target):
   ```env
   LIVE_TIER=T2        # ← pick the next rung
   ```
   Do not raise `LIVE_MAX_ORDER_USD` or similar until you reach T4.
   Tiers T0-T3 enforce their own ceilings anyway.
3. **Restart the service.** The tier is read at init, not hot-reloaded.
4. **Confirm the new tier in the first `/healthz` poll:**
   ```json
   {
     "active_tier": {
       "name": "T2",
       "label": "growth",
       "max_order_usd": 100.0,
       ...
     },
     "max_order_usd": 100.0,  ← must match tier
     "kelly_dampen": 0.8
   }
   ```
   If any field doesn't match the tier table above, **stop**: something
   is overriding the tier (usually an older env var still set lower).
5. **Watch the first 3 trades.** They should size up to the new cap
   (subject to Kelly / margin / firewall), no crashes, no rejects from
   the caps you just relaxed.

---

## How to retreat

If the new tier shows trouble within the first few sessions:

1. Set `LIVE_TIER` back to the previous rung.
2. Restart.
3. **Do not advance again for twice the normal window.** If T1 → T2
   failed and T2 ran for 1 day before rollback, serve another 6 days on
   T1 (= 2x the T1 `min_days_recommended`) before re-attempting T2.
4. Log the failure reason — which gate tripped? Which env var
   countermanded the tier?

---

## Reaching T4 (full scale)

T4 disables all tier overrides. At that point the *only* caps are:

- `LIVE_MAX_ORDER_USD` (per-order notional)
- `HL_MAX_DAILY_LOSS` / `LIVE_MAX_DAILY_LOSS` (daily loss)
- `LIVE_MAX_POSITION_SIZE_USD` (per-coin notional)
- `LIVE_CANARY_MAX_SIGNALS_PER_DAY` (or 0 for unlimited)
- `KELLY_MULTIPLIER` (sizing aggressiveness)
- The firewall's cooldown, same-side limit, and drawdown cap
- The kill switch (manual + daily-loss auto)

This is the configuration a production book operates under. **Do not
flip straight from T0 to T4.** Walk the ladder; each tier validates
one more dimension of safety (size handling, fee drag, slippage,
regime response) at higher stakes.

Recommended T4 starting envelope for a retail operator:
```env
LIVE_TIER=T4
LIVE_MAX_ORDER_USD=500
LIVE_MAX_POSITION_SIZE_USD=3000
HL_MAX_DAILY_LOSS=1000
LIVE_CANARY_MAX_SIGNALS_PER_DAY=150
KELLY_MULTIPLIER=0.25
```

Raise these env vars over further weeks/months, not days. Each raise is
an informal T5/T6/etc. — run the same advancement checklist, just
against env-var changes rather than a tier flip.

---

## Quick reference: env-var recipes

### Start fresh on T0 (default)
```env
LIVE_TIER=T0
# or just omit — the legacy LIVE_CANARY_MODE still works
```

### Move to T1 after T0 checklist passes
```env
LIVE_TIER=T1
```

### Full scale
```env
LIVE_TIER=T4
LIVE_MAX_ORDER_USD=500        # operator-defined ceiling
HL_MAX_DAILY_LOSS=1000
LIVE_MAX_POSITION_SIZE_USD=3000
```

### Emergency brake (without deleting env vars)
```env
LIVE_TIER=T0                  # immediately caps everything to $25 / $50
```
Restart → your next order will be $25 or less, period.

---

## Monitoring

`/healthz` exposes the following tier-relevant fields:

| Field                            | Meaning                                      |
|----------------------------------|----------------------------------------------|
| `active_tier.name`               | T0..T4 (null if no LIVE_TIER set)            |
| `active_tier.label`              | canary / seed / growth / scale / full        |
| `active_tier.max_order_usd`      | Tier's own order cap (null = no override)    |
| `kelly_dampen`                   | Effective sizing multiplier                  |
| `max_order_usd`                  | Final resolved cap after tier + env          |
| `max_position_size`              | Final resolved per-coin cap                  |
| `max_daily_loss` (via daily_pnl_limit) | Final resolved loss cap                 |
| `canary_mode`                    | True for T0-T3, False for T4 or no tier      |
| `kill_switch_active` / `kill_switch_reason` | Current halt state                |
| `entry_metrics.*`                | Why signals were accepted/rejected today     |

The tier fields are derived once at init — they do not change unless
the process restarts, which is intentional (tier changes are explicit
deploys, not runtime toggles).
