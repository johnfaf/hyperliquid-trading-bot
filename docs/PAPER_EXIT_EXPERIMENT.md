# BTC paper exit experiment

This research tool does not place orders, modify the live bot, or change the frozen
$1,000 [paper pilot](PAPER_PILOT.md). It compares exits offline using the pilot's
recorded entries and one common clocked market recording. The portfolio may take
different future entries if its exits change; this matched-entry study intentionally
does not model that effect. Its drawdown metric is per trade, not portfolio drawdown.

## Collect and compare

```powershell
python scripts/paper_exit_experiment.py capture --market-db <DIR>/btc-paper-exit-market.sqlite3 --seconds 3600
python scripts/paper_exit_experiment.py compare --market-db <DIR>/btc-paper-exit-market.sqlite3 --pilot-db <PILOT_SNAPSHOT> --from <UTC_START> --until <UTC_END>
```

Supply full ISO timestamps with an explicit timezone (for example
`2026-11-15T00:00:00Z`). Use one collector per market database. Capture at least
16 minutes before the first measured entry and until all compared positions close
plus five minutes. Use a consistent copy of the pilot SQLite database if copying
it to another machine. Comparison opens both inputs read-only and writes JSON
to stdout. No account address, secret, Telegram integration, or order endpoint
is needed.

For a sustained recording, use `capture --continuous` and the `status`
subcommand against its market database. The VPS companion image in
`deploy/paper-pilot/exit-collector.Dockerfile` layers these research files
onto the pinned paper-pilot image. Mount only a new writable directory
at `/capture`; don't mount the pilot database or publish network ports.
`status` exits nonzero if no usable quote was recorded in the last 30
seconds. Container restart policy should be configured at launch; an
unhealthy Docker status alone does not restart a running container.

The collector samples [public Hyperliquid streams](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions)
every five seconds: mark, best bid/ask and top bid size, five-level book imbalance,
signed trade notional over the last minute, funding rate, oracle price and open
interest. Use the newest valid top quote from the two book feeds; depth
freshness is tracked separately and stale depth is not substituted. Book
freshness is measured from exchange time; asset context has no
exchange timestamp in the published feed, so its age is measured from local
receipt. Reconnects discard cached state. Missing observations are stored as
such, and incomplete minute buckets reset the trailing volatility warmup.

## Frozen comparison

| Arm | Checks | Exit policy |
| --- | --- | --- |
| Fixed 60s | At a 60-second cadence | Original stop, target and holding limit |
| Fixed 5s | At a five-second cadence | Same rules, on the common recording |
| Trailing 5s | At a five-second cadence | Same original stop/target; trail once the modeled executable gain clears +1R after estimated costs |

The trail distance is the larger of twice the mean true range of 14 complete
one-minute buckets or 0.35% of entry. A bucket requires at least 10 valid
samples and no observation gap over 10 seconds; an explicit missing
sample makes it incomplete. A change requires three distinct,
recent book observations with adequate top-bid size and spread within 20 bps,
plus known funding estimates to date. A new stop must be above the old stop
and below the current mark. Never widen the stop. The first target stays fixed.
Funding/open interest, volume imbalance and returns are recorded for analysis;
sentiment and headlines do not change the rule. The thresholds are research
parameters, not validated settings or a recommendation to trade.

All three arms use mark-price trigger checks and a conservative modeled bid
fill less 5 bps of extra impact; 4.5 bps taker fees apply per leg. Insufficient
top-bid size delays the modeled exit and leaves it pending. Real order behavior
can differ, especially in gaps and at the exchange's [trigger and execution boundary](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/take-profit-and-stop-loss-orders-tp-sl).
The entry price is shared across arms, and the 60-second arm emulates timing;
its results are not a reconstruction of the deployed pilot's actual paper exits.

Review paired after-cost P&L, peak profit giveback, worst single-trade drawdown,
and the best executable rebound in the five minutes after exit. The rebound
is an early-exit clue, not a realizable strategy. Aggregates exclude cohorts
with any unclosed arm, a material quote gap (over 15 seconds), or missing
funding estimates. Per-entry results retain these cases. Report exclusion and
rebound coverage; even short gaps and five-second sampling can omit price moves.
Collect a sufficiently broad sample and preregister acceptance criteria before
considering dynamic take-profit, adding data sources, or changing any deployment.
