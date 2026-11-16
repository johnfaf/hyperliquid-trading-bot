# Decision evidence and paper review

## What the dashboard counts

- Crypto **Trade review > Decision Coverage** counts registered cycles and candidate rosters for ranked strategies, copy, liquidation reversals, options flow and Arena. Raw scanner prefilters and market data that never reached those routes do not have an expected count. An empty roster is different from a route that did not report. No rows before capture means unmeasured, not 100% coverage.
- The candidate ticket is fixed within a cycle. A terminal result requires an explicit disposition or matching paper-ledger evidence. A snapshot saying approved or containing an unmatched trade ID is not a proven execution. The unresolved table retains the candidate identifier and the newest recorded checkpoint. Quote age is unknown when the feed did not supply it.
- A paper intent is written before insertion in the trading ledger. Matching the exact ID in the paper ledger resolves it after a restart, even if the separate decision snapshot was never updated. An unmatched intent is not an instruction to retry. An unavailable evidence store blocks new crypto entries within the tracked cycle; it does not prevent position management or reduce-only orders.
- Hyperliquid recovers ambiguous orders by client order ID, the returned venue order ID and matching fill IDs. An unknown response is not proof that nothing executed. Do not bypass the account pause or clear a pending order because the local ledger is empty. Stock Pilot separately records scheduled process starts, results and overdue unfinished runs; it does not make stock performance out of shadow decisions.

The on-host crypto SQLite journal sits alongside the paper ledger and order journal on the persistent volume. Rollout snapshots any existing journals separately. These are not a coordinated point-in-time restore or off-host disaster recovery. Do not restore one journal in isolation and resume orders without reconciling it with the ledger and venue.

## Replay, then prospective review

Run from the repository root against a read-only SQLite ledger or a protected snapshot. Save the public candle cache with the original report so the hypothetical price paths can be rerun. Keep ledger data and credentials out of Git.

```shell
python -m scripts.run_decision_evidence_backtest --db /path/to/bot.db \
  --start 2026-11-14T00:00:00Z --split 2026-11-15T12:00:00Z --end 2026-11-16T10:00:00Z \
  --save-candles /private/decision-candles.json --output /private/decision-evidence.md
```

Actual paper fills, matched copy exclusions, and hypothetical future paths are separate. The matched diagnostic skips a position when same-asset, same-direction copy exposure already existed in the observed book; it does not roll forward the hypothetical book or reinvest freed capital. Forward prices begin at the next 15-minute candle, use no fill or exit assumptions, and are not trade-level P&L. A chronological split of already-observed history is not independent prospective evidence.

The copy shadow protocol is registered with the build revision when each cycle starts. Once a **new** period has at least 14 elapsed days on one known revision, rerun the command with that period and `--prospective`. This read-only screen requires at least 30 matched clean closes, 95% terminal coverage, no unfinished cycles or unmatched paper intents, and positive improvement without worse realized drawdown after an additional 16 bps cost on the recorded entry notional. Missing or mixed-revision data blocks the screen. A passing screen only requests operator review of a paper change; it does not modify the active policy or authorize any live order. Do not pick a different threshold or date range after viewing the result and call that a holdout.
