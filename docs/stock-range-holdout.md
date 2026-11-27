# SPY range-reversal holdout

The 27 November range rebound was a discovery example, not evidence used
to qualify either strategy. The new holdout is **1 June through 31 August
2026**, separate from the September-November opening-range replay. The
runner pins the source and date range before it fetches market data.

## Fixed rules

Both rules observe the same opening range from the first six completed
five-minute SPY bars. Each can take at most one long or short candidate
per session during the first 90 minutes after the open. The two books
are separate comparisons against $1,000 in unchanged cash, not two ways
to hold simultaneous positions in the bot.

- **Long rebound:** The bar touches the bottom quarter of the range, then
  closes higher than the previous bar and between the lower-quarter mark
  and the midpoint. Disqualify all later rebound attempts after the first
  completed close below the range low. Stop at the signal bar low; target the opening
  range high.
- **Failed retest short:** After the first close below the range low,
  require one of the next three completed bars to reach that low from
  below and close below it. A close back inside first invalidates the
  attempt. Stop at the retest bar high; target one opening-range width
  below the low.

The stop and target are checked on completed bars, not assumed to execute
at the printed level. Time exit is 30 minutes after the signal. Use the
first usable quote 2-30 seconds after each decision. A candidate needs
1.5:1 or better estimated reward to risk at the modeled entry after
costs, and cannot chase by more than 15 bps. Maximum notional including
modeled entry fees and short carry is $75 per book.

## Costs and checks

Base costs are 5 bps adverse slippage and 2 bps fee per leg. Stress uses
12 and 6. Both use IEX bid/ask and displayed size. The combined case also
waits 20 seconds, widens the spread threefold, requires five displayed
size units, and applies a 35 bps adverse exit gap. The additional modeled
short carry haircut is 2 bps base or 5 bps stress per round trip. These
are test assumptions, not actual borrowing or broker rates.

An exit outage leaves the position unresolved, prevents a later entry
from using the same sleeve, and makes total P&L unmeasurable. The
economics screen requires at least 90% usable sessions, 20 closes
in each scored scenario, positive net without the best winner, zero
unresolved exits, and at most $10 realized drawdown. A pass is a reason
to collect fresh forward evidence, not permission to place orders.

Short results are hypothetical price-side comparisons. No historical
borrow or eligibility was verified. Neither rule is wired into the
order worker. Historical bars can be corrected later, and IEX data is
not proof of a broker fill. VPS receipts are kept in
`/var/lib/hyperliquid-stock-range-holdout/2026-06-to-08-v1`.

## Results, 27 November 2026

All 64 scheduled sessions had usable data. Neither rule cleared the
economics screen.

| Rule | Signals | Base closes | Base net | Stress closes |
| --- | ---: | ---: | ---: | ---: |
| Long rebound | 27 | 2 | -$0.45 | 0 |
| Failed retest short | 7 | 1 | +$0.12 | 0 |
| Cash | 0 | 0 | $0.00 | 0 |

Under stress, all 34 candidates had a usable quote within the chase
limit, but none met the cost-adjusted 1.5:1 reward-to-risk threshold.
In the combined case the chase guard rejected seven first, and the
reward-to-risk guard rejected the other 27. No stressed fills means
there is no stressed return evidence, not that the strategy broke even.
The only base short fill was its only profitable trade; removing it
eliminates the entire gain. These results do not justify releasing
either rule or retuning the thresholds on the same window.
