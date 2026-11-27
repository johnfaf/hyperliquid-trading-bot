# SPY multi-session pullback holdout

**Decision: do not promote.** The rule found only nine eligible signal dates
and lost money on resolved trades in every economic scenario. One exit remained
unresolved in every scenario; the realized subtotal is not total strategy P&L.

## Fixed test

- SPY, US regular-session calendar, September 2, 2025 through May 29, 2026
  (187 sessions). Warm-up starts May 1, 2025 and is not scored. This is a
  different scored window from the June-August 2026 range-reversal study and the
  September-November 2026 opening-range study. It is not a test on a different
  instrument or market.
- On a completed daily bar, require SPY to be above its 50-session simple
  moving average, that average to be above its level 10 sessions earlier,
  and the close to reclaim its 10-session average after the prior close was
  at or below its then-current 10-session average. Stop reference: lowest low
  of the last four completed sessions. Target reference: signal close plus
  2.5 times the initial stop distance.
- First possible entry is after 9:35 a.m. New York time in the next session.
  Use the first eligible historical IEX ask in the bounded observation window,
  with modeled slippage and fees. Reject prices more than 35 basis points
  above the signal close or below 1.5:1 estimated reward-to-risk after costs.
  One position, capped at $75 including entry fees on a $1,000 account. A
  position remains open for at most ten completed holding sessions before the
  model requests an exit; it can stay unresolved beyond that point.
- A completed close at or beyond a stop or target, or the tenth holding
  session's close, schedules a sale at the next morning's eligible bid.
  There is no fill at the trigger price, no same-session sleeve reuse after
  exit, and no forward-filled exit price. Unresolved exits block later entries.
  Signals too near the end to allow the full planned hold are not eligible.
- Base uses 5 basis points of slippage and 2 of fees per leg; stress uses
  12 and 6. Combined also delays eligible quotes from second 2 to second 20,
  triples observed spread, requires depth of at least five on both sides,
  and applies a further 35-basis-point adverse exit-price haircut. The quote
  window ends at second 30 and accepts at most a 10-basis-point spread.
  The exit outage is a separate forced-failure test.
- Compare to unchanged cash and the same $75 sleeve passively holding SPY,
  using the first and last session's opening quote windows with the same
  scenario costs. Price return only; no dividends or idle-cash yield. Raw
  daily prices matched split-adjusted prices on the tested interval. This is
  single-venue quote modeling, not broker fill proof or intraday drawdown.
- The VPS runner pinned dates, feed and source hash in an isolated SQLite
  ledger before requesting historical data. Completed historical bars determine
  which potential exit quote windows to download; the simulator still
  schedules each actual exit chronologically, independently of this download
  list. All 20 requested windows were captured.

## Results

| Scenario | Closed (wins) | Missed entries | Blocked signals | Unresolved exits | Realized subtotal | Passive SPY | Close-mark drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 5 (3) | 1 | 2 | 1 | -$1.01 | +$13.97 | $6.13 |
| Stress | 4 (2) | 2 | 2 | 1 | -$3.53 | +$13.78 | $6.77 |
| Combined | 3 (0) | 3 | 2 | 1 | -$4.35 | Unmeasured | $6.20 |
| Forced exit outage | 0 (0) | 2 | 6 | 1 | $0.00 | Unmeasured | $0.80 |

Cash P&L is $0 in every case. Drawdown is measured at session closes through
the unresolved exit, not through the end of the holdout. In the base run,
three time exits made $2.42 combined; two stop exits lost $3.43. On February
25, 2026, the scheduled time-limit sale had 4,196 quotes, but none met the
spread limit even in the base case. Combined passive SPY is unmeasured because
the September 2 entry window had no quote with enough displayed depth after
the modeled delay.

The fixed screen required at least 20 resolved trades in each economic scenario,
zero unresolved exits, at most $10 in close-mark drawdown, and net P&L after
removing the biggest winner above both cash and passive SPY. This test failed
the screen. The result rejects promotion of this rule under this protocol;
it does not establish that every pullback strategy loses money.

Before testing another overnight signal, define and measure an exit fallback
separately: a bounded retry window, explicit allowed execution costs, and an
alert when the exposure remains open. Validate that policy against both
single-venue and routing-relevant data where available, then evaluate it on
fresh dates. Do not swap in a more favorable fill for this replay. This study
creates no broker orders, has no ongoing schedule, and grants neither paper
nor live approval.
