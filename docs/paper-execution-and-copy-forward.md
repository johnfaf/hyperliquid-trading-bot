# Paper execution and forward copy checks

The stock execution scorecard follows automatic paper reservations by exact
client order ID to broker IDs, quantities and fill timestamps. A reservation
is not proof of submission or acceptance. A filled status without the
matching fill quantity needs review. Supervised drills and operator orders
do not count as natural orders. Round-trip counts are withheld if an operator
intervention makes natural-only inventory ambiguous. The dated independent
broker audit is reported separately. Working orders age at 30 minutes;
unknown or unresolved orders age at 10 minutes. Alerts do not resend or cancel.
A session with no new strategy fills is not execution evidence.

SPY forward recovery alerts after the oldest unprotected capture is 12 hours
old. A current backup does not alert merely because the study is idle. The
existing Windows encrypted pull and isolated restore are unchanged; confirm
that a new receipt covers the current capture count. Alerts retry after
unconfirmed delivery and suppress repeats for 24 hours after confirmation.

## Paired copy experiment

- Study window: 28 days from the next UTC hour after registration. The timer
  skips missed slots instead of replaying them from later quotes. The model
  and capture script are pinned together and checked at every run.
- Shared inputs: the first ten minutes of each hour from the live decision
  journal, observed once. Only BTC, ETH and SOL; known named sources and raw
  confidence in `[0.65, 1]`. No outcomes, realized prices, retrospective
  filters or account activity enter the allocator. This is a diagnostic on
  observed candidates, not a full counterfactual of the upstream bot.
- Rank by confidence, then recorded time and hashed decision ID. One
  unlevered simulated position per hour per book. Both books start with
  $1,000, can deploy at most $75 including entry fees, and stop entering
  for the remainder of a UTC day after $10 in realized losses. The copy-off
  book can reallocate a slot to the best eligible non-copy candidate.
- After the input capture at :10, collect a shared Hyperliquid top-of-book
  snapshot at :15 for entry and :40 for time exit. Require a later exchange
  timestamp, bounded spread, sufficient displayed top-level size, and at
  least $10 modeled notional for an entry. Buy at ask and sell at bid,
  with another 5 bps adverse slippage and 4.5 bps modeled fees per leg
  (stress: 12 and 8 bps). No hourly funding boundary is crossed. Exit timing
  is fixed; no stop, priority, extra depth or intrahour drawdown is modeled.
- Missing capture or insufficient exit depth invalidates the affected
  paired hour for that fee scenario, never only the worse book. These hours
  are not counted as victories or trading opportunities. Do not use
  partial results to change strategy or promote live execution. The
  exploratory summary needs at least 90% scored hours and 20 different
  allocations per scenario after the full 28 days.

The study only reads the production decision ledger with a read-only SQLite
connection and public market data. It writes its own ledger under
`/var/lib/hyperliquid-copy-forward`, with model receipts in `trace.json`.
It does not alter the frozen SPY or BTC study, production filters, or
broker execution. The prior weekly fixed-fill attribution remains separate.
