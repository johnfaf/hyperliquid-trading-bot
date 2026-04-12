# Recommended Live Config: Sub-$100 Canary

Date: 2026-04-11

This profile is based on the current live rollout evidence:

- live runtime is stable
- latest logs are clean apart from expected canary warnings
- the live account was recently observed around `$51` in exposure guardrail logs
- `copy_trade` is the weakest source in the current realized sample
- short trades are the main drag on performance
- Hyperliquid minimum notional is effectively `$11`
- current canary cap is `$25`, which is enough for normal entries but not for many crash-de-risked entries

## Recommendation

Keep the bot in a small canary posture until we have a better realized sample from `strategy` and `options_flow`.

Do not raise sizing yet just to force more crash-regime trades through. That would mainly re-enable weak short exposure before we have evidence it deserves more capital.

## Recommended Env Profile

```env
LIVE_CANARY_MODE=true
LIVE_MAX_ORDER_USD=25
LIVE_CANARY_MAX_ORDER_USD=25
LIVE_CANARY_MAX_SIGNALS_PER_DAY=8
LIVE_MAX_DAILY_LOSS_USD=8
LIVE_MAX_ORDERS_PER_SOURCE_PER_DAY=1

FIREWALL_CANARY_MODE=true
FIREWALL_CANARY_MAX_POSITIONS=2
PORTFOLIO_HARD_MAX_POSITIONS=2

SHORT_HARDENING_ENABLED=true
SHORT_HARDENING_MIN_CLOSED_TRADES=12
SHORT_HARDENING_DEGRADE_WIN_RATE=0.45
SHORT_HARDENING_BLOCK_WIN_RATE=0.35
SHORT_HARDENING_BLOCK_NET_PNL=-1.0
SHORT_HARDENING_CONFIDENCE_MULTIPLIER=0.85
SHORT_HARDENING_SIZE_MULTIPLIER=0.60

COPY_TRADER_ENABLED=true
COPY_TRADER_MAX_CONCURRENT_TRADES=1
COPY_TRADER_MAX_NEW_TRADES_PER_CYCLE=1
COPY_TRADER_AUTO_PAUSE_MIN_CLOSED_TRADES=6
COPY_TRADER_AUTO_PAUSE_DEGRADE_WIN_RATE=0.40
COPY_TRADER_AUTO_PAUSE_BLOCK_WIN_RATE=0.25
COPY_TRADER_AUTO_PAUSE_BLOCK_NET_PNL=-25

SOURCE_POLICY_ENABLED=true
SOURCE_POLICY_KEEP_TOP_N=2
SOURCE_POLICY_WARMUP_MAX_SIGNALS_PER_DAY=1
SOURCE_POLICY_DEGRADED_MAX_SIGNALS_PER_DAY=1
SOURCE_POLICY_WARMUP_SIZE_MULTIPLIER=0.75
SOURCE_POLICY_DEGRADED_SIZE_MULTIPLIER=0.60
SOURCE_POLICY_WARMUP_MIN_CONFIDENCE=0.45
SOURCE_POLICY_DEGRADED_MIN_CONFIDENCE=0.55
```

## Why These Values

### `LIVE_MAX_ORDER_USD=25`

Keep this where it is for now. It is safely above the exchange minimum, but still small enough to limit damage while the source mix is still being proven.

### `LIVE_MAX_DAILY_LOSS_USD=8`

The code default of `$100` is too high for a wallet around `$51`. A canary wallet should stop well before a large fraction of the account is gone. `$8` is still meaningful room for a noisy session, but small enough to preserve capital.

### `FIREWALL_CANARY_MAX_POSITIONS=2` and `PORTFOLIO_HARD_MAX_POSITIONS=2`

With sub-$100 capital, more positions mostly create dust, minimum-order friction, and operational noise. Two positions are enough to test the live path without over-fragmenting capital.

### `LIVE_MAX_ORDERS_PER_SOURCE_PER_DAY=1`

This keeps any one source from dominating a small account before it earns that right.

### Copy-trade settings

The current realized copy-trade sample is poor enough that it should be treated as a constrained experiment, not a core capital consumer.

- max `1` concurrent copy trade
- max `1` new copy trade per cycle
- auto-pause after at least `6` closed copy trades if the realized sample is still bad

That keeps copy-trading available for re-validation, but prevents it from taking over the book.

## When To Raise Size

Do not raise the canary cap until all three are true:

1. `strategy` and `options_flow` have closed-trade history with non-negative net PnL
2. min-order rejects stay low in the dashboard
3. short-side guardrail is no longer blocking or degrading most short setups

The first meaningful cap raise is `LIVE_MAX_ORDER_USD=60`. That is the point where crash-de-risked trades can still clear the `$11` exchange minimum more often.

## Not Recommended Yet

- `LIVE_MAX_ORDER_USD=90-100`
- disabling short hardening
- re-enabling unrestricted copy-trade flow
- increasing max positions above `2`

Those changes would add risk faster than they add information.
