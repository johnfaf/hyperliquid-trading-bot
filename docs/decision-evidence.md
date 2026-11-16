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

## Daily report and alerts

The existing `hyperliquid-dashboard-status.timer` publishes the latest comparison to **Trade review > Copy trades**. Around 00:10 UTC it archives the previous completed UTC day and sends one Telegram digest through the configured bot. The VPS keeps private Markdown and JSON in `/var/lib/hyperliquid-dashboard/copy-reports`. The total for all clean copy closes is separate from the matched comparison, which requires entry-time tags. A day with no tagged closes is unmeasured, not break-even. The report excludes tainted or synthetic closes and displays a separate additional 16 bps cost scenario when entry notional is available for every matched close.

The same job alerts on missing or late decision coverage, failed cycles, unresolved evaluations and paper intents older than 20 minutes, unresolved venue submissions, and daily report failure. It retries unsuccessful deliveries and limits repeat alerts for the same reason to once every six hours. It does not pause or modify trading. Use [Decision Coverage](/decision-coverage) to inspect pending candidates. An absent live order journal remains unmeasured, not a confirmed zero. Today's P&L is not reported as a completed daily result.

## Faster historical iteration

Use historical replay to reject weak ideas quickly, without lowering the prospective review standard. The offline selector below fits one choice only: keep the recorded policy or skip same-direction overlaps. Freeze dates and thresholds before examining results. It reconstructs older overlap from the observed book, delivering close events no earlier than their timestamp; actual entry tags take precedence when available. It does not use future prices, final trade outcomes, or mutable decision scores as entry features.

```shell
python -m scripts.run_copy_walk_forward --db /path/to/bot.db \
  --start 2026-11-14T00:00:00Z --test-start 2026-11-15T00:00:00Z \
  --test-start 2026-11-16T00:00:00Z --end 2026-11-16T12:30:00Z
```

Each fold learns only from earlier trades that closed before its start, with an extra one-hour embargo. The selected rule stays frozen throughout that test fold. Later P&L is used only to score the frozen selection; trades whose outcomes cross the scoring cutoff are counted as unscored. Never randomize time order or use a later exchange price to populate an entry-time feature. Legacy reconstructed trades are historical diagnostics, not a full execution backtest or independent prospective evidence. Resolve missing data and retain the next unseen period for confirmation. This tool writes no model to the running bot.
