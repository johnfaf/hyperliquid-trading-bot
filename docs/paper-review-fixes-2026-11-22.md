# Paper admission and reporting fixes

## Changes

- Arena must have a firewall, a positive measured account balance, and an approved
  entry in both paper and live modes. Live validation never uses a paper balance
  fallback. Sizing uses the approved fraction and leverage, and the paper stop
  budget includes modeled slippage and fees. Dynamic sizing cannot increase the
  risk-policy allocation.
- Arena's persisted source is `arena_champion:<strategy_type>`, distinct from
  ordinary strategy signals. Entry records capture approval, source, stop budget,
  and a scorer signal ID so subsequent outcomes can update that source's policy.
- Paper admission checks and insertion share a database transaction. SQLite uses
  `BEGIN IMMEDIATE`; PostgreSQL uses a transaction advisory lock. Opposing entries
  are rejected regardless of route. Idempotent retries still return the original
  trade ID. New asset and side fields are normalized before storage.
- Source-policy lookup failures, malformed allocations, and zero allocations
  reject entries. An untracked source receives the existing warmup limits instead
  of an unlimited full-size allowance. Existing pause/degrade policies remain.
- Coverage distinguishes non-error final dispositions from errors and exposes
  the latest 20 error references, routes, symbols and recorded reasons. A final
  disposition does not assert successful execution.
- Trade review adds dated windows, a completed UTC calendar week, entry-route
  grouping, profit factor, mean return, win rate, realized drawdown, and net
  without the largest winner. Evidence labels cannot grant live approval.
- Current whole-book balance, unrealized P&L and marked equity are separate from
  period realized returns. Marks come from observed exit-check quotes, expire
  after 15 minutes, and never fall back to entry prices for missing symbols.
  Marked equity is before future exit costs. Historical equity drawdown is not
  reconstructed from closed trades.
- The host publisher archives one fixed Monday-to-Monday paper report per week,
  after a 10-minute settlement allowance. Valid archives are reused unchanged;
  invalid existing archives require inspection rather than silent replacement.
  Archives include constituent clean trades, route totals, fee completeness and
  excluded-record counts. Late ledger corrections do not rewrite an archive.
- `/research` shows the independent BTC/ETH absorption scorecard, its observation
  dates, freshness, sample coverage and stressed matched outcomes. It also shows
  the archived paper week. Authentication is required. Missing evidence is not
  displayed as a zero return or a successful test.
- Stock evidence uses the injectable clock for default capture timestamps and
  coverage windows, preserving replay-time semantics. The PostgreSQL schema
  guard exempts specific direct-SQLite journal module/table pairs, not generic
  table names throughout the app.

## Verification

- Full regression suite: 1,997 passed, 18 skipped.
- Latest focused checks, including invalid Arena quotes, stock replay timestamps
  and reporting: 189 passed.
- SQLite admission tested with two competing processes; exactly one opposing
  entry is admitted. Policy errors and zero allocations reject entries.
- Playwright checked trade review, completed-week view, research and coverage at
  1440px, 390px and 320px, with screenshots and no page-level horizontal overflow.
- Reporter dependencies import successfully from an isolated staged release.
  Python compilation and rollout shell syntax checks pass.
- No live-money or target-VPS runtime test was performed. PostgreSQL concurrency
  was not exercised locally.

## Deployment

Use `deploy/rollout-vps-paper.sh <full-git-sha>` from an unpacked release on the
VPS. Its existing backup, paper-only, signer-absence and private-dashboard checks
remain in force. It installs the weekly reporting module and, when the absorption
ledger exists, the structured research reporter. The reporter has a separate
release directory; the frozen collector and study execution are not restarted.

Weekly reports are private JSON archives under
`/var/lib/hyperliquid-dashboard/copy-reports/weekly/`. Research reports remain in
`/var/lib/hyperliquid-absorption/reports/`.

The configured administrative SSH key was rejected on 22 November 2026.
GitHub publication does not mean these changes are running on the VPS. Restore
deployment access, run the rollout, then inspect `/decision-coverage`,
`/trades?scope=all`, and `/research` through the private dashboard tunnel.

## Limits

- No historical trade, database, stop evidence or existing position is deleted,
  reconstructed or forcibly closed by these changes.
- Old stops without trigger/fill evidence remain unverified. Prior recorded
  errors cannot be declared resolved without checking their causes and new
  runtime observations.
- No live flag, signer, broker credential, stock pilot mode, or live-capital allocation
  is changed. Research success and small profitable cohorts are not proof of an
  investable edge.
- Runtime rollout and PostgreSQL concurrency must be verified on the target host;
  local regression checks exercise SQLite, simulated execution and dashboard UI.
