# VPS safety rollout: 22 November 2026

This is a paper/shadow safety release, not approval to trade real money.
No credentials, trading history, positions, or frozen research rules are reset.

## Six work items

1. Restore the existing deployment key and verify the VPS. Deploy a pinned Git
   revision only after tests, with SQLite backups and an image rollback.
2. Fix the funding-cost guard: positive funding is paid by longs; negative
   funding is paid by shorts. Convert the native hourly context value to the
   eight-hour fraction expected by the existing policy thresholds.
3. Resolve live-mirror promotion evidence using the exact route/type/wallet
   identity. Reject conflicting source keys and strategy IDs, rather than
   borrowing Arena/options-flow evidence from a generic strategy cohort.
4. Apply final firewall validation after sizing and confidence adjustments to
   liquidation, options-flow, and Arena entries. Use the approved leverage,
   size, and protection. Never use paper capital or an empty position list as
   fallback for unavailable live data. Apply planned-stop sizing to all three
   direct paper routes, as already done by the ordinary and copy routes.
5. Persist account-scoped marked equity, daily baselines, peaks, and drawdowns.
   Enable atomic paper entry checks on the VPS, including a combined open
   mark-to-stop budget. Keep exits and idempotent replay available while entries
   are paused. Show equity risk in Trade Review, Stock Pilot, and Settings.
6. Exercise restart, ambiguous submission, partial-fill deduplication, missing
   quotes, stale observations, and competing-process budget tests. Restore
   immutable VPS backups into separate scratch files and verify their integrity
   and critical risk/execution evidence without connecting to an exchange.

## Paper settings

| Setting | VPS default |
| --- | --- |
| `PAPER_PORTFOLIO_GUARD_ENABLED` | `true` in deployment overlay |
| `PAPER_MAX_PLANNED_STOP_LOSS_PCT` | 0.0025 (0.25% per entry) |
| `PAPER_MAX_OPEN_STOP_RISK_PCT` | 0.02 (2% combined open stop risk) |
| `PAPER_DAILY_EQUITY_LOSS_PCT` | 0.03 (3% of daily baseline) |
| `PAPER_MAX_EQUITY_DRAWDOWN_PCT` | 0.10 (10% below observed peak) |

Actual existing environment overrides remain authoritative. Ordinary historical
replays keep the portfolio guard disabled unless explicitly configured, because
they do not necessarily contain an equity observation stream. No future marks
or observations are used by point-in-time risk queries.

Paper risk observations are in the sibling `bot.risk.sqlite3`; stock risk is
account-scoped within the existing stock order journal. Missing/stale/incomplete
equity blocks new guarded entries. Existing positions without measurable
protection also block further entries; they are not silently closed or erased.
Stop risk includes modeled slippage and fees but is not a guarantee against gaps.

The initial equity capture starts with this release. Reports must not invent
historical equity drawdown. A weekly drawdown is only shown with a sufficiently
complete observation window; realized closed-trade drawdown is labeled separately.

## Operations

- `/api/execution-mode` is a fast, nonsensitive runtime mode attestation. It does
  not run the heavy database audit or replace full live-readiness checks.
- The public absorption collector inherited an HTTP health check from its
  dependency image despite having no HTTP server. Replace that check with
  freshness of both BTC and ETH ready frames, using
  `deploy/repair-absorption-healthcheck.sh FULL_DEPLOYMENT_SHA`.
- The collector repair backs up its database, retains a stopped rollback
  container, and preserves the exact image ID and frozen research revision.
  One public-feed reconnect creates a new, honestly recorded capture session.
  The health probe monitors freshness; Docker does not restart a container just
  because it is unhealthy. The existing collector reconnect loop is unchanged.
- Run `scripts/verify_sqlite_recovery.py --revision FULL_DEPLOYMENT_SHA` as an
  administrator on the VPS after the rollout. This uses only pre-rollout
  backups and temporary restored copies. It never overwrites production files.
- Rollout failure restores the previous image, never automatically restores an
  old trading database over newer state. Backups remain restricted to root.

## Limits and remaining live prerequisites

Offline recovery simulations do not establish actual exchange behavior or
strategy profitability. This release does not send live or supervised paper
orders. Stocks remain in shadow mode.

Before enabling real money, independently validate completed-trade, net-of-cost
promotion evidence (not generated-signal counts), all direct-live promotion
paths, actual venue stop acknowledgements, order-rounding/minimums, exchange
outage behavior, and a supervised minimal-risk round trip. In particular, exact
source identity alone does not prove a profitable out-of-sample edge.

Hyperliquid funding reference:
https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding
