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
