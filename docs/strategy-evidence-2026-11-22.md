# Strategy evidence, shadow replay, and execution reconciliation

Crypto stays paper-only and the stock pilot stays shadow-only. This release
does not place orders, change signers, reset the ledger, or change the frozen
exit and absorption studies.

## Evidence gate

Research shows a 30-day scorecard for observed paper cohorts. New approved
paper entries record their canonical source, entry route, strategy instance
and rule version at entry. The version uses the build SHA and a hash of
recorded strategy parameters and selected paper cost and risk controls.
This is conservative grouping, not a claim that changing upstream models
or market inputs are frozen. Older trades get no retroactive version.

Review requires uniquely linked, completed paper decisions with valid
prices, a recorded exit reason and consistent modeled cost accounting.
Synthetic or ambiguous closes stay excluded. Paper's best-effort funding
aggregate cannot prove settlement coverage; trades spanning a funding
hour are excluded until that coverage can be demonstrated.

The score requires 30 verified closes across seven UTC close dates, no
excluded closes in the cohort, positive net, positive net with an extra
16 bps round trip, and positive stressed net excluding the largest winner.
Passing means ready for human review, not a proven edge or funding approval.

Both live entry adapters enforce an independent hold, including mirrored,
direct, and reversal entries. It leaves exits and protective actions
available. Legacy promotion and bootstrap flags are diagnostic only.
No approval or unlock mechanism is provided in this release.

## Frozen BTC screen

One candidate, no parameter search: the prior frozen failed-breakout rule
on BTC 15-minute bars, next-bar fills, 5% of an isolated $1,000 book per
position at 1x, with a 2% fixed stop, 4% target, and 18-hour max hold.
Base: 5 bps slippage (spread proxy included) and 4.5 bps fees per leg.
Stress: 12 bps and 6 bps respectively. Both use public hourly funding.
Benchmarks: cash and a passive long perpetual with the same initial BTC
allocation. Passive stays exposed longer. Candle data does not provide
actual spread, depth or intrabar order.

| UTC window | Natural / total closes | Base net | Stress net | Cash | Passive base net |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oct 26 to Nov 16 13:00 | 24 / 25 | -$3.62 | -$5.42 | $0 | -$10.31 |
| Nov 16 13:15 to Nov 22 00:00 | 9 / 9 | -$3.46 | -$4.01 | $0 | -$3.20 |

The later window starts one bar after the previous archive. It is a
retrospective date holdout, not prospective execution. The candidate failed
against cash in both windows and against passive in the later window.
Do not promote or tune it against that holdout. A predeclared November 23
to November 30 diagnostic is available only after that window is complete.

    python -m scripts.fetch_hyperliquid_replay_market --days 42 --out MARKET.sqlite3
    python -m scripts.run_strategy_shadow --market MARKET.sqlite3 --output NEW_REPORT_DIR
    python -m scripts.run_strategy_shadow --market LATER.sqlite3 --output NEW_FORWARD_DIR --prospective

The runner records the input SHA-256 and a trace for both scenarios. It opens
the public capture read-only and disables networking during evaluation.

## Historical executions

Overlapping CSV exports are counted separately and their shared signatures
are reported. Do not subtract a fee again when the reported P&L already
includes it. Exports without exact order/fill IDs, verified timezone, or
funding coverage cannot attribute losses to decisions or establish full
account P&L. Keep account-level inputs and reports outside version control.

    python -m scripts.reconcile_hyperliquid_execution --csv EXPORT.csv --out EXPORT_AUDIT.json
    python -m scripts.import_hyperliquid_fills FILLS.json --ledger FILLS.sqlite3 --account PUBLIC_ADDRESS
    python -m scripts.reconcile_hyperliquid_execution --fills FILLS.sqlite3 --account PUBLIC_ADDRESS --orders ORDERS_SNAPSHOT.sqlite3 --decisions BOT_SNAPSHOT.db --out EXECUTION_AUDIT.json

Supply unaggregated venue fills with time, tid, and oid. A successful link
requires venue order ID in the matching account's journal and deterministic
client order ID in the decision ledger, with consistent symbol, side, time
and filled quantity. Full, partial, conflicting, and unmatched records
remain separate. No match by coin/time proximity. Fee totals are separated
by token and are not subtracted again from the venue-reported P&L. Check
funding, transfers, coverage, and the independent account statement separately.
