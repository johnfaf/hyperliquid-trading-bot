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

## Sell Acknowledgment Follow-up

After a verified first round trip, the VPS can reuse that **dedicated**
account without resetting its history. The follow-up copies the old
journal into a separate v2 journal from an immutable, root-only SQLite
snapshot. It rejects extra orders, accounts on the stock-worker exclusion
list, fill discrepancies, and inventory that differs from the journal.
Do not point the follow-up at the stock worker's account or database.

1. Start `hyperliquid-stock-exit-verification@preflight.service` and check
   its journal. The broker checks are read-only, even outside market hours.
   A closed market does not prevent identity and history verification.
2. During a supervised open session, start
   `hyperliquid-stock-exit-verification@entry.service` once. This reserves
   and sends a single SPY limit buy for at most $10 of **paper** notional.
3. Start `hyperliquid-stock-exit-verification@exit.service` only after the
   entry fills. It reconciles and sends one sell for the exact inventory,
   then intentionally withholds its acknowledgment. A repeat invocation
   cannot submit a second sell.
4. Start `hyperliquid-stock-exit-verification@verify.service`. It looks
   up the sell by its original client order ID. Reissue only **verify**
   while it is working or waiting for fills. If the order stays open,
   is canceled partially filled, or leaves a discrepancy, stop and
   reconcile manually in Alpaca Paper. No replacement or working-order
   cancellation is sent by this service.

A successful receipt requires two new filled orders, zero remaining
inventory, no open orders, and matching fill and cash records within one
cent. A single reported sell fill leaves broker partial fills unverified.
Multiple reported fills show a split execution, not that the cancellation
race from the [offline drill](stock-exit-order-drill.md) occurred.
Neither a successful receipt nor the test notification enables retries.

### Test Alert

`scripts/stock_exit_test_alert.py` sends the labeled
`STOCK-EXIT-TEST-01` fixture message to the configured Telegram chat.
It must be invoked deliberately with `--send-test`, a dedicated state
directory, and a root-managed configuration containing only the Telegram
bot token and chat ID. It may instead read the isolated systemd credential
named `telegram`. It reserves the test in SQLite before posting to
Telegram; a repeat process with the same state will not re-send, even if
the previous delivery is uncertain. Do not clear state to force delivery.

An accepted Telegram response confirms the API accepted the message,
not that it was seen on a device. This test describes a simulated 0.07 SPY
position and never places an order. There is no recurring alert job.
