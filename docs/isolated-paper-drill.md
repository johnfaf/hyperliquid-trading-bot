# Isolated paper broker drill

This drill uses a new, empty Alpaca Paper account. Do not reset, reuse or
delete the stock worker's existing account. Confirm that only the paper
endpoint and paper credentials are used. Paper fills are simulated.

Create the new account from the paper account selector in Alpaca. On the
VPS, provision these three values in the root-only
`/etc/hyperliquid-stock-drill/paper.env`:

```dotenv
DRILL_ALPACA_KEY=dedicated_paper_key
DRILL_ALPACA_SECRET=dedicated_paper_secret
DRILL_ALPACA_ACCOUNT_ID=dedicated_paper_api_account_id
```

Use a protected editor, not command-line parameters, Git or chat. The
separate `excluded.env` is generated from existing stock-worker journal
account IDs. Do not fill that file by guessing. Refresh it by rerunning
the installation step after changing the worker's broker account. The
service refuses to run with no exclusion list or when the API reports the
same account as the worker. The drill journal lives in
`/var/lib/hyperliquid-stock-drill` and is not shared with the strategy.

## Supervised sequence

1. Start `hyperliquid-stock-paper-drill@preflight.service` and check the
   journal message. It is read-only on the broker. It checks the account,
   empty order and fill history, market clock, and quote.
2. In the indicated open session, start
   `hyperliquid-stock-paper-drill@entry.service` once. The order is a
   fractional limit capped at $10 simulated notional. The test withholds
   the acknowledgement after sending it once. Treat the response as
   unresolved and do not resend. A day limit order can remain open.
3. Start the distinct `hyperliquid-stock-paper-drill@recover.service`.
   It reopens the durable journal, queries by the original client ID,
   and verifies the unique entry and exact fills. Poll reconciliation,
   not submission. If the order is rejected, partial and canceled, or
   ambiguous, stop the drill and handle cleanup in Alpaca Paper. Do not
   reset the journal to make a pass. A canceled partial fill still
   represents simulated inventory; confirm its disposition.
4. Start `hyperliquid-stock-paper-drill@exit.service` only after confirmed
   recovery. It sends one limit sell for the exact verified inventory.
   Start `hyperliquid-stock-paper-drill@verify.service` after it fills.
   Verification requires two exact order IDs, matched buy and sell fills,
   zero remaining inventory or open orders, and a broker cash difference
   within $0.01 of the cash movement implied by those fills.

None of these services is enabled by a timer. A verified receipt is
separate from local fault fixtures, the weekly frozen memo, and strategy
returns. It does not authorize money or qualify a live broker pilot.
