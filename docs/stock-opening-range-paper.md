# SPY opening-range paper pilot

The existing daily SPY order worker records the two completed closes and
their respective 20-session averages. Its current-cycle outcome and most
recent measured daily inputs are separate: a closed market, missing quote,
blocked entry or absent crossover is not a rejected order.

## Held-out replay

The fixed rule uses the first six completed regular-session five-minute
bars to define the high and low. From then until 90 minutes after regular
open, take the first completed close crossing above the range high. Buy
once per session, long only. Exit after a completed close at or below the
range midpoint, or 30 minutes after the signal, whichever comes first.
No extended hours, overnight carry is intended. A real paper limit order
can remain unfilled, including on exit; the order journal remains the
authority in that case.

For the replay, the bid/ask quote must come 2 to 30 seconds after the
completed bar's decision time, with displayed size, at most 10 bps spread
and at most 15 bps unfavorable price movement from the signal close for
entry. Both cost scenarios use the same observations and the same $75
maximum unlevered sleeve on a $1,000 reference book. Base slippage and
fees are 5 and 2 bps per leg; stress uses 12 and 6. Unfilled entries and
unresolved exits are counted separately; hypothetical exits are not
manufactured. Quotes are IEX observations, not guaranteed broker fills.
The replay measures execution costs, not the cost of hosting the bot.

Register `2026-09-01` through `2026-11-27` (end exclusive) before collecting
the holdout. The tool freezes source fingerprint and session dates, saves
the private input receipts, and only calls the read-only Alpaca calendar
and historical market endpoints. Never use the held-out period to select
new thresholds. Evaluate materially changed strategies on a fresh window.

## Paper release

The report needs at least 90% usable sessions, 20 actual modeled closes
in both cost scenarios, no unresolved exits, a positive stressed net
even without the best winner, and no more than $10 of realized marked
drawdown in the stress case. A pass would allow only a supervised
paper pilot, never live trading or automatic promotion. It still needs
explicit `STOCK_ORB_PAPER_ENABLE=true` for the stock worker and a fresh,
root-owned, non-writable-by-others report at
`/opt/hyperliquid-stock-paper/intraday-proof.json`.

Daily SPY has priority. Both strategies use the same Alpaca Paper
execution owner and verified account pin, with one shared inventory
slot and the existing $75 cap. The new pilot keeps its signal and exit
plan in a separate SQLite file beside the order journal. Removing
pilot admission permission blocks new pilot entries; known pilot
holdings still have a verified exit path. Do not stop order polling or
clear the journals to recover from a missing acknowledgement.

## Held-out result, 27 November 2026

Alpaca IEX supplied usable inputs for 61 of 61 scheduled sessions.
The fixed rule generated 29 signals. All historical quote pages in each
bounded window were read, and no completed close used a same-bar fill.

| Execution costs | Closes | Missed entries | Unresolved exits | Net | Without best winner |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 29 | 0 | 0 | -$3.61 | -$3.75 |
| Stress | 28 | 1 | 0 | -$8.19 | -$8.20 |

The rule did not qualify. The paper admission override was not enabled.
Private immutable inputs are in the replay-v1 holdout ledger on the VPS.
