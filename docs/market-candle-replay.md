# Fast-forward Hyperliquid candle research

The capture uses public Hyperliquid info requests only. It fixes the universe at download time: ten active native crypto perpetuals by 24-hour USD notional. This is not historical market selection. It archives closed 15-minute OHLCV, actual hourly funding rates and their raw times, ranking weights, capture timestamp, and any failures. A missing candle or funding hour blocks that asset from the comparison; the BTC/ETH baseline requires both complete.

From the repo root, choose new output paths in a research directory outside Git:

```shell
python -m scripts.fetch_hyperliquid_replay_market --days 42 --out /private/market.sqlite3
python -m scripts.run_hyperliquid_exit_replay --market /private/market.sqlite3 \
  --report /private/exit-test.md --trace /private/exit-test.jsonl
```

The runner uses the actual Alpha Arena `trend_following` entry generator, not a stored decision label. It gives the generator only completed candles, leaves sentiment/copy data and the full live firewall out, and uses a temporary SQLite configuration with outbound HTTP disabled. An additional container `--network none` is appropriate on the VPS. It never writes a policy artifact or enables execution.

On the VPS, run `bash deploy/run-vps-market-replay.sh MARKET_SQLITE NEW_REPORT_DIR` from an unpacked private Git release. The helper uses the dependencies from the installed image in a separate container with networking disabled, an ephemeral `/tmp`, and no access to the trading database or environment file. It does not restart the bot.

The frozen comparison tests fixed 2% price stops, 2x and 3x Wilder ATR(14) sized stops, and a 2x variant that can tighten from the next bar after a completed close in profit by at least 1R. Stops are bounded, the target stays twice the initial risk, and positions are closed after 18 hours at the next executable open if still active. Fills use the bar after the signal with adverse slippage; an opening gap through the stop fills at the gap, and if neither order was triggered at the open, a two-touch bar takes the stop first. Fees and historical funding are applied. These are simulated fills, not measured market execution.

Selection only sees the expanding BTC/ETH training window, with a one-bar gap before each later window. A candidate must have enough closed trades, improve stressed net P&L into positive territory, and not worsen stressed marked drawdown. The other eight assets never train the selector and are evaluated as an equal-sleeve retrospective robustness check. The final window is separate from its calibration; once its scores are inspected, new changes need a fresh holdout. The baseline and every variant are reported alongside the selected policy. Traces record every forward decision, identified by asset/fold/policy, and all simulated entries, exits and stop changes at base costs.

Use the capture and replay to screen exit ideas, not to infer live P&L or the profitability of the whole bot. Candle prices do not contain the order book or the exact path within a bar, and volume rank comes from the capture date. Keep the live and paper bot untouched while evaluating the results.

## Test whether entries have an edge

```shell
python -m scripts.run_hyperliquid_entry_replay --market /private/market.sqlite3 \
  --report /private/entry-test.md --trace /private/entry-pairs.jsonl
```

The primary screen is BTC/ETH with a fixed four-hour hold; one hour and the other eight markets are diagnostics. Exits and position sizing are intentionally out of scope. Controls are earlier no-signal moments on the same market, regime, and UTC session, matched by ATR within a factor of 1.5. Copy the signal direction to the control and charge exactly the same fee/slippage model, plus each leg's actual historical funding. Match before calculating returns, require the control to have already completed, and avoid overlapping/reused controls. Report net bps per traded dollar and absolute net on every eligible signal, not just the matched subset.

Repeated training evaluates a small fixed menu of confidence, alignment and volatility filters on two past weeks only. It can reward only a filter with a minimum sample and positive stressed net in both weeks. The worst of the two weeks is the first ranking criterion. It freezes the choice before the next forward week with a full-bar embargo, and falls back to cash when no candidate is viable. The paired trace identifies both the unfiltered diagnostic and the frozen selected filter. Reusing an inspected period or changing the candidate menu after seeing its score voids the holdout; no result can promote a policy automatically.

For true forward confirmation, record a freeze time before seeing outcomes and rerun on a later capture with `--forward-from` and that UTC hour. The runner requires at least 14 days of earlier control context and at least one completed seven-day post-cutoff window. On the VPS, append `entry` to the isolated helper's two arguments, optionally followed by `--forward-from` and the recorded cutoff. This is research only: fast iterations can reject bad filters quickly, but this is not a way to turn an old period into independent evidence.
