#!/usr/bin/env python3
"""Backfill the candle cache with extra coins for multi-coin backtests/replay.

The replay harness (scripts/run_replay.py) and the bg-auto-backtest learning
loop can only evaluate coins that exist in the candle cache -- historically
just BTC/ETH, so every strategy/copy signal on SOL/HYPE/ZEC/etc. skipped with
"no valid mid price."  This fetches 1h (or any timeframe) candles for a coin
list over a window into the cache, so strategy AND copy-trade backtests cover
the coins the bot actually trades.

Examples
--------
    # The default crypto-perp set the copy traders use, into the live cache:
    python scripts/backfill_candle_cache.py --start 2026-03-01 --end 2026-04-23 \\
        --cache-dir /data

    # A custom list/timeframe:
    python scripts/backfill_candle_cache.py --coins SOL,HYPE,ZEC --timeframe 1h \\
        --start 2026-03-01 --end 2026-04-23
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Crypto perps the tracked copy-traders trade with meaningful volume (the
# xyz:/para:/@NNN HL equity/index/spot instruments are intentionally excluded:
# the candle API doesn't serve them and the perp strategies don't trade them).
DEFAULT_COINS = (
    "SOL,HYPE,MON,ZEC,DOGE,ALGO,ZRO,XRP,FARTCOIN,HBAR,TAO,TRUMP,"
    "XPL,MOODENG,MORPHO,BNB,PENGU,IP,MET"
)


def _default_start() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%d")


def _default_end() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def backfill(coins, timeframe, start, end, cache_dir):
    """Fetch+cache each coin; return {coin: count|'ERR:..'}.  Never raises."""
    from src.backtest.data_fetcher import DataFetcher

    fetcher = DataFetcher(cache_dir=cache_dir)
    results = {}
    for coin in coins:
        try:
            candles = fetcher.fetch_candles(coin, timeframe, start=start, end=end)
            results[coin] = len(candles)
            print(f"  {coin:<10} {len(candles)} candles")
        except Exception as exc:  # one bad coin must not abort the batch
            results[coin] = f"ERR:{str(exc)[:60]}"
            print(f"  {coin:<10} ERROR: {str(exc)[:80]}")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--coins", default=DEFAULT_COINS,
                    help="Comma-separated coins (default: the copy-trader perp set)")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--start", default=_default_start(), help="YYYY-MM-DD (default 60d ago)")
    ap.add_argument("--end", default=_default_end(), help="YYYY-MM-DD (default today)")
    ap.add_argument("--cache-dir", default="data",
                    help="Dir holding candle_cache.db (use /data for the live cache)")
    args = ap.parse_args(argv)

    coins = [c.strip() for c in args.coins.split(",") if c.strip()]
    print(f"Backfilling {len(coins)} coins {args.timeframe} {args.start}..{args.end} "
          f"-> {args.cache_dir}/candle_cache.db")
    results = backfill(coins, args.timeframe, args.start, args.end, args.cache_dir)
    ok = sum(1 for v in results.values() if isinstance(v, int))
    print(f"done: {ok}/{len(coins)} coins backfilled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
