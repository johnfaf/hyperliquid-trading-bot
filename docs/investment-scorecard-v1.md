# Investment research protocol v1

Registered 23 November 2026. This protocol is a review of research evidence,
not a trading strategy release or authorization to spend capital.

## Fixed evaluation

- Crypto's existing BTC 24-hour channel candidate remains frozen from 22
  November to 20 December 2026. It fills at the next modeled 15-minute open,
  sizes at 5% of a $1,000 reference account without leverage, and uses its
  original stop, target and time exit. The existing cost assumptions and
  separate cash and equal-starting-sleeve passive BTC results are unchanged.
- After the study ends, look for at least 30 natural closes. Both base and
  stressed results must beat cash and the passive sleeve after expenses and
  stay above cash without the largest winner. Missing real operating costs
  leave the result unmeasured unless market costs alone already fail the
  test. A pass is a screen, not investment evidence sufficient for approval.
  Require an independently reproduced, longer prospective evaluation under
  different market conditions before proposing a supervised pilot.
- Log every tested variation, including unsuccessful ones. The retrospective
  range-shock screen skips new BTC entries when the last completed candle
  range is at least three times the median of the prior 96 closed bars. It
  uses the original frozen fill and exit model. It was an exploratory
  diagnostic on already seen data, not another untouched holdout. Six
  signals were screened out without changing any executed trades. The
  base and stressed results remained negative in the earlier window; the
  later window had no trades. Do not promote this filter.
- Stocks remain a separate $1,000 reference book under the existing Alpaca
  equity trend hypothesis. Shadow decisions and quotes are not order fills
  or strategy returns. Review the schedule, benchmark and cost model
  independently. Do not transfer or optimize crypto parameters on stocks.

## Accounting and risk verification

The completed paper week is Monday-to-Monday UTC. Its investment memo is
stored once on the VPS, after a ten-minute settlement buffer. Earlier
reports are not overwritten. A cumulative BTC study figure has its own
evaluation timestamp and is not presented as P&L within the paper week.
Verified historical wallet attribution is still missing.

The reporter saves the first complete daily SQLite accounting checkpoint.
Consecutive daily checkpoints compare independent visible changes in cash,
account P&L, and number of wins and closes to the underlying paper ledger.
The first checkpoint is a baseline; no prior cash is inferred. A flat day
without trades is not execution evidence. This is an internal paper ledger
check, not independent venue or custody verification.

The separate stock worker uses paper credentials, a scoped read-only
journal, and broker GETs only. It checks exact order and fill identities,
quantity, price, and resulting inventory. A connected flat account has
insufficient fill evidence. Read failures and partial data cannot pass.
This worker never submits or cancels orders. A supervised disposable paper
account with before/after cash and forced network failures is a separate
drill. Do not use a live account or change the shadow strategy setting to
complete it.

Open-position protection checks report missing, stale, or breached marks.
They do not claim a modeled stop was accepted or filled on a venue.
Offline tests cover gaps, stale quotes, unknown acknowledgements,
duplicated and partial fills, and the durable entry halt. Keep exits
available when making an entry halt. Before any pilot, prove limits and
protective exits on the target venue and require another reviewer to
approve an explicit loss budget. Do not increase size to satisfy a broker
minimum if it would violate that budget.

## Operator inputs

Operating costs are unmeasured until the operator supplies current monthly
hosting and data costs in USD. The root-owned dashboard-only optional
environment file is `/etc/hyperliquid-bot/investment-review.env`; set
`INVESTMENT_COST_ARGS="--monthly-ops-usd 10.00"` with the actual total,
not the example. The current BTC comparison uses the latest recorded
expenses. The next weekly memo captures the assumptions then in force;
existing memos remain unchanged. Expenses are prorated using 12 months per 365 days and applied once
to the reference strategy book; separate alternatives are not added
together. Do not add any brokerage secrets to this file.

The weekly memo and ongoing audits have no path to order authorization.
Production code still holds live entries independently of this report.

The completed-week **Copy on vs off** panel removes only new copy entries
opened in the weekly window. Existing positions stay in both books.
Retained trades use identical recorded sizing, fills, and costs; both
sides apply the same added 16 bps round-trip stress using leveraged
notional. Closes at or beyond the cutoff are unscored. This is attribution
on recorded fills, not a fresh execution replay: changed approvals,
replacement trades, and portfolio reallocation are not measured.
