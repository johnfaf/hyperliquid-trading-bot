# Execution rollout: crypto and US stocks

## Current boundary

The VPS deployment in this repository is paper-only. The rollout refuses a
running bot that requests live trading or has a signer. The frozen BTC pilot,
collector, and backup implementation are not changed by this rollout.

- The crypto order path persists a submission before sending it. A missing
  reply blocks new entries across that account until order lookup resolves
  it. A filled status requires matching venue fills, deduplicated by trade
  ID. A negative lookup alone does not clear the block. Reduce-only
  protection and manual venue exits remain available. Existing trade and
  balance reconciliation remains responsible for the full live book.
- Entry client IDs derive from decision IDs, not the retry timestamp. A
  resting maker order is not replaced with a market buy after a timeout or
  unconfirmed cancel. Preserve the execution journal together with the
  trading database when backing up, migrating or rolling back a live system.
- The stock worker cannot use a live trading URL. It reserves orders before
  submitting, imports confirmed paper fills, compares them to positions and
  open orders, and stops on discrepancies. Never reset the broker paper
  account while reusing its old ledger.

## Stock paper onboarding

1. Create a **dedicated, empty Alpaca paper account** and reset its simulated
   balance to $1,000. Do not use live keys, trade manually in it, or share
   credentials in chat. Set the root-owned file
   `/etc/hyperliquid-stock-paper/paper.env` (mode `0600`) with
   `STOCK_PAPER_KEY`, `STOCK_PAPER_SECRET`, and
   `STOCK_PAPER_SEND_ORDERS=false`.
2. Inspect `journalctl -u hyperliquid-stock-paper.service`. Verify account,
   calendar, data entitlements and shadow intents during a US session.
   The example is a single-symbol, long-only daily SMA20 crossover, using
   completed split-adjusted bars and fresh regular-session quotes.
3. Enable **paper** submissions with `STOCK_PAPER_SEND_ORDERS=true` only
   after shadow checks. The worker allows one position, at most $75 for a
   new entry, and uses broker cash/buying power. A $10 fall in daily paper
   equity blocks new entries and requests an exit of the tracked position
   during the permitted session. This is not a guaranteed loss maximum.
4. `run_stock_paper.py --action pause` blocks new entries persistently;
   `--action flatten` pauses and requests a sell of the verified position
   when open orders and fills reconcile. Both use the same environment and
   service ledger; do not run another writer against that account.
   Flatten is not guaranteed to fill. Use the broker console for manual
   intervention if the worker cannot reconcile or the market is closed.

The prototype does not book dividend cash flows or automate split/symbol
migrations: a position change without matching fills blocks the worker.
IEX quotes are not consolidated market data; do not treat simulated fills
or paper benchmark results as live execution evidence.

## Evidence before live

1. **Execution:** In a dedicated testnet or paper account, check a lost
   response, restart, partial fill, repeated fill, cancel/fill race, manual
   intervention, missing protection, and unavailable broker. Require zero
   duplicate orders and zero unexplained positions. Drill pause, flatten and
   incident recovery. Do not clear an unresolved record without external
   evidence and operator review.
2. **Strategy:** Freeze the rules and the prospective test window before
   observing results. Compare net performance to cash and a buy-and-hold
   reference where appropriate. Include actual subscriptions, spread, fees,
   modeled slippage and funding where applicable. Record drawdown and sample
   size. A small or inconclusive sample is not a pass.
3. **Supervised live trial:** Only after both reviews pass, confirm venue
   eligibility, account permissions, and any regulatory requirements before
   configuring a tightly limited live account. Use one strategy, review
   the first orders, and verify protective orders and statement
   reconciliation. The agent cannot set its own limits or authorize
   deployment. The stock worker in this release has no live mode.

## VPS releases

Ship a clean Git archive of the tested commit and run
`deploy/rollout-vps-paper.sh FULL_GIT_SHA` on the VPS. It preserves the
site's compose files and local notification overlay, stages a new image,
takes a checked SQLite backup, verifies paper-only health and rolls back the
image on a failed startup. It then refreshes the paper report jobs and stock
timer without provisioning keys. Do not use this procedure to arm live
trading.

The private dashboard tunnel still targets VPS port `8080`. The refreshed
Overview shows paper performance, decisions and service status; Settings
shows effective risk settings and stock-paper readiness. Settings are
read-only and never display signing keys. The legacy dashboard, including its
options and stress views, is retained on VPS loopback port `8082` for a
separate administrative SSH tunnel. Neither dashboard port is public.
