# Isolated $1,000 paper pilot

The full bot, its database and its dashboard are not needed to run the pilot.
This runner uses the patched simulator and public Hyperliquid market data only.
It does not load a signer, route orders to an exchange, or expose a port.

## Frozen study

- One BTC perpetual contract market, long-only; no borrowed leverage, one position, at most $75 simulated notional. No copy trades, strategy selection, adaptive sizing or model retraining.
- Enter on a cross above a 24-hour simple moving average, based on contiguous, fully closed one-hour candles. Only consider the signal in the first five minutes of the next hour, using a fresh midpoint from the order book.
- Stop at 2% below simulated entry, target 4% above it, or exit after 24 hours; check once a minute. A gap or missing data can delay fills or increase simulated losses. No missing quote is replaced with a fictitious exit.
- Assume taker fees of 4.5 bps and modeled adverse slippage of 5 bps each way. Hourly funding is estimated. Spread is not independently charged; queue, latency and size impact are not calibrated. These are assumptions, not live execution results.
- Track marked equity, drawdown, a flat-cash reference, and a BTC price-only reference on the same maximum initial allocation. The BTC reference holds continuously (unlike the strategy) and has no costs, so is not an exposure-matched benchmark. A controlled replay with an untouched holdout and adverse costs remains separate work.

## Operations

Use `docs/audit-paper-1000.env.example` with a dedicated `/data` volume, no inherited production env, and no published ports. Start with `python -B scripts/run_paper_pilot.py`; `--once` is a one-cycle smoke check. Do not run a second copy against the same database. The pilot refuses an existing database unless it carries the identical persisted study spec. Changes to the strategy or costs require a new database and study record. To stop, stop the pilot process only; do not reset the database.

Generate a report using `python -B scripts/daily_paper_report.py --db PATH --day YYYY-MM-DD`. Omit the date for the previous UTC day or use `--all-history` for a preserved historical copy. Metrics show the fraction of trades with available cost data; synthetic reconciliation closes are reported as exclusions, never counted as actual fills.

The optional VPS systemd units in `deploy/paper-pilot` expect the pilot data in `/srv/hyperliquid-audit-pilot/data`, the report script in `tools`, and a private writable `reports` directory under the same prefix. They write a report for the previous UTC day at 00:10 UTC. They don't send messages or control the bot. Keep backups independently and record the release revision and profile used for each book.
