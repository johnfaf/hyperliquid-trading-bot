# SPY paper entry lab: 27 November 2026

## Today's comparison

Cutoff registered at 17:45 UK time. All rules use only completed 5-minute
Alpaca IEX bars within their specified window. The entry window ends at
16:00 UK time (90 minutes after the regular open), so later bars cannot
create an entry. The range from the first 30 minutes was **$720.03 to
$726.05**. The highest of the 12 subsequent eligible closes was **$725.61**.

| Rule | Qualification | Result |
| --- | --- | --- |
| Opening-range breakout | First completed close crossing above the opening high | No breakout |
| Opening-range retest | After a breakout, one of the next three completed bars touches the high and closes back above it, without first closing below the range midpoint | No initial breakout |
| VWAP reclaim | Previous close at or below running IEX VWAP, followed by a close above VWAP and above the opening-range midpoint | No qualifying reclaim |
| Cash | No position | No modeled P&L or execution costs |

No candidate order or fill was created. Cash was the result under the
predeclared rules, not a recommendation based on a hindsight low. The
same $1,000 book, $75 maximum sleeve, midpoint exit and 30-minute
maximum observation horizon apply to all three rules. No new rule
qualifies for paper activation from this single session.

## Original breakout: unchanged 61-session holdout

All 29 original signals were run through a second set of execution
assumptions. None of these assumptions changes which signal fired.

| Scenario | Matched closes | Missed entries | Unresolved exits | Net |
| --- | ---: | ---: | ---: | ---: |
| Base, 5 bps slippage and 2 bps fee per leg | 29 | 0 | 0 | -$3.61 |
| Stress, 12 bps slippage and 6 bps fee per leg | 28 | 1 | 0 | -$8.19 |
| Stress plus 20 seconds of delay | 25 | 4 | 0 | -$7.08 |
| Stress plus triple the quoted spread | 27 | 2 | 0 | -$7.94 |
| Stress with at least five displayed size units | 28 | 1 | 0 | -$8.19 |
| Stress plus a 35 bps adverse exit gap | 28 | 1 | 0 | -$15.51 |
| Combined delay, spread, depth and exit gap | 25 | 4 | 0 | -$13.79 |
| Stress with all exit quotes unavailable | 0 | 1 | 28 | Unmeasured |

A scenario with a smaller loss and more missed entries does not show
that delay or wider spreads improve the strategy; the filled-trade
sample changed. The outage scenario keeps fills unresolved and never
invents an exit price. Results exclude hosting costs, are simulated
from one feed, and do not prove broker fills or a trading edge.

## Reproduction

`scripts/run_stock_entry_lab.py` pins the requested date, cutoff and
source before collecting data. It stores the exact rows separately
from the production journals. The original holdout remains read-only.
On the VPS, the 27 November run is in
`/var/lib/hyperliquid-stock-entry-lab/2026-11-27-1745`.
No order endpoint, strategy enablement, or worker changes are involved.
