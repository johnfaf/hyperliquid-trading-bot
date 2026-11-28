# Stock exit order fault drill

This is an **offline fixture**, not an Alpaca Paper order test. It runs the
stock worker's `submit_once`, `reconcile`, and SQLite order journal against
a separate, deterministic venue fixture. Each observation runs in a new
process. The fixture persists its order and fill records before suppressing
the acknowledgment or exiting the worker. There are no network calls,
broker orders, Telegram notifications, or changes to the trading workers.

## Run

Use a dedicated empty state directory. No credentials are needed:

```bash
mkdir -p /tmp/stock-exit-order-drill
python -I -B scripts/stock_exit_order_drill.py --state-dir /tmp/stock-exit-order-drill
```

Each run retains its own trace, SQLite journals, and venue snapshots.
`latest.json` is the latest **successful** fixture receipt; also check
the current command's exit status. A failed run does not overwrite an
earlier success. Neither a pass nor a retained report grants paper or
live approval. The VPS fixture runs with networking disabled and was
verified on November 28, 2026.

## Verified Outcomes

| Injected fault | Expected and observed result |
|---|---|
| Sell accepted, acknowledgment lost | Recovery looks up the original client ID. An absent lookup does not justify another submission. |
| Partial fill, cancel pending, then a late fill | An initial 0.10-share sell fills 0.03, then 0.02 more before cancellation. No replacement while it is working or while the canceled order's filled quantity, fill journal, and position disagree. |
| Worker exits after the replacement fills, before saving the response | One replacement for the reconciled 0.05 shares. Restart finds the filled order by its reserved ID and verifies flat inventory. Exactly two fixture sell submissions in total. |
| Unknown sell or pending cancellation at deadline | No replacement. One persisted unresolved-exposure alert per scenario (0.10 and 0.07 shares remaining). A repeat observation cannot duplicate the alert, and a late confirmation does not reopen the retry window. |
| Two callers race for the same replacement | A focused test uses two SQLite connections to verify that only one caller can reserve the ID and submit. |

The fixture drives cancellation status from the outside; it does not send a
cancel request. It does not prove actual broker status timing, executed
prices, alert delivery, or real-world partial-fill behavior. It is also
separate from the [SPY exit quote replay](stock-exit-retry-holdout.md):
an eligible quote is not permission to resubmit a working order.

## Before Activation

Use a separately bound Alpaca Paper account and the
[supervised paper procedure](isolated-paper-drill.md). Capture the broker's
acknowledgment, fill activity, order history, and resulting position by
client ID. Do not force a partial fill by retrying a working paper order.
If no partial occurs, leave that broker behavior unverified. Keep automatic
retries off until the broker evidence and unresolved-position alert delivery
are verified independently.
