# Automatic stock and crypto paper execution

Crypto runs with `BOT_RUN_MODE=paper`, no trading signer, and both live venue
flags off. Its normal strategy, market, entry and portfolio checks still
decide whether a simulated entry is eligible. No signals is a valid state.

For stocks, the scheduled strategy uses a long-only SPY crossover of the
20-session moving average, completed-session bars and a fresh bounded-spread
quote. It places fractional day limit orders for up to $75 per entry on
`https://paper-api.alpaca.markets`. An accepted order is not a fill, and
ambiguous responses are reconciled by client order ID before another entry.
No entry occurs outside regular hours. A no-signal, closed-session or
risk rejection is a valid outcome, not a reason to manufacture a trade.

## Stock activation

After installing the stock release and systemd unit, leave
`STOCK_PAPER_SEND_ORDERS=false` in `/etc/hyperliquid-stock-paper/paper.env`.
The independent stock study and reconciliation worker share this read-only
configuration. For the stock order worker only, preflight writes an override
containing the explicit paper opt-in and the verified account fingerprint:

```sh
sudo systemd-run --quiet --wait --collect --pipe \
  -p EnvironmentFile=/etc/hyperliquid-stock-paper/paper.env \
  -p EnvironmentFile=/etc/hyperliquid-stock-drill/paper.env \
  -p 'UnsetEnvironment=DRILL_ALPACA_KEY DRILL_ALPACA_SECRET' \
  /usr/bin/python3 -I -B /opt/hyperliquid-stock-paper/current/scripts/check_stock_paper_readonly.py \
  --db /var/lib/hyperliquid-stock-paper/orders.sqlite3 \
  --arm-output /etc/hyperliquid-stock-paper/execution.env
sudo systemctl start hyperliquid-stock-paper.service
```

Preflight cannot submit an order. It requires a flat, matched account with
no open or unresolved orders and an identity distinct from the isolated
drill account. The worker refuses to trade if its account later differs
from the fingerprint. Leave this override out of Git, backups shared
outside the host, and the environment of the frozen study.

To stop new entries while allowing a held position to exit, run the worker
with its normal service environment and `--action pause`; `--action resume`
clears that pause. On a flat and reconciled account, moving
`execution.env` out of the configuration directory also restores shadow-only
operation. Do not disable polling or discard the journal while there are
open, partial, or unresolved orders.

Verify **Paper mode** and the decision journal on **Stock Pilot**.
Compare actual stock fills in Alpaca's **Paper** account and crypto simulated
fills in **Trades**; the crypto exchange account intentionally receives none.
The frozen forward studies do not consume these broker fills.
