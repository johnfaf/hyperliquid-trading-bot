#!/usr/bin/env python3
"""Run cross-coin strategy sweeps and walk-forward parameter research.

Roadmap coverage:
  9. Re-run the 12-strategy sweep over the full 10-coin universe.
 10. Walk-forward optimize RSI + mean_reversion on trailing windows and test OOS.

By default this reads the local candle cache only. Add --fetch-missing if you
want the script to call Hyperliquid for missing candles.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.data_fetcher import DataFetcher, TIMEFRAME_MS, _parse_date  # noqa: E402
from src.backtest.research_suite import (  # noqa: E402
    DEFAULT_10_COIN_UNIVERSE,
    DEFAULT_12_STRATEGIES,
    DEFAULT_WALK_FORWARD_GRIDS,
    run_cross_coin_strategy_sweep,
    run_walk_forward_parameter_optimization,
)  # noqa: E402


def _load_candles(
    *,
    cache_dir: str,
    coins: list[str],
    timeframe: str,
    start: str,
    end: str,
    fetch_missing: bool,
) -> dict:
    fetcher = DataFetcher(cache_dir=cache_dir)
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"unsupported timeframe {timeframe!r}")
    if fetch_missing:
        return {
            coin: fetcher.fetch_candles(coin, timeframe, start=start, end=end, use_cache=True)
            for coin in coins
        }
    start_ms = _parse_date(start)
    end_ms = _parse_date(end)
    return {
        coin: fetcher._get_cached(coin, timeframe, start_ms, end_ms)
        for coin in coins
    }


def _print_summary(report: dict) -> None:
    print()
    print("=" * 78)
    print("  CANDLE RESEARCH SUMMARY")
    print("=" * 78)
    print(f"  Universe: {', '.join(report['config']['coins'])}")
    print(f"  Window:   {report['config']['start']} -> {report['config']['end']}")
    print()
    print("  Cross-coin strategy ranking:")
    rows = sorted(
        report["cross_coin_sweep"]["per_strategy"].items(),
        key=lambda item: (
            item[1].get("survives_cross_coin", False),
            item[1].get("total_pnl", 0.0),
            item[1].get("avg_sharpe", 0.0),
        ),
        reverse=True,
    )
    for strategy, metrics in rows[:12]:
        print(
            f"    {strategy:<18} pnl={metrics['total_pnl']:>10.2f} "
            f"trades={metrics['total_trades']:>5} "
            f"coins={metrics['coins_with_trades']}/{metrics['coins_tested']} "
            f"survives={metrics['survives_cross_coin']}"
        )
    print()
    print("  Walk-forward OOS:")
    for item in report["walk_forward"]:
        agg = item["aggregate"]
        print(
            f"    {item['coin']:<6} {item['strategy']:<15} "
            f"folds={agg['fold_count']:>3} pnl={agg['test_total_pnl']:>10.2f} "
            f"trades={agg['test_trades']:>5} stable={agg['stable_params']}"
        )
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--coins", default=",".join(DEFAULT_10_COIN_UNIVERSE))
    parser.add_argument("--strategies", default=",".join(DEFAULT_12_STRATEGIES))
    parser.add_argument("--fetch-missing", action="store_true")
    parser.add_argument("--min-candles", type=int, default=50)
    parser.add_argument("--walk-forward-coins", default="BTC,ETH,SOL")
    parser.add_argument("--train-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--report-out", default="reports/candle_research.json")
    args = parser.parse_args(argv)

    coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    candles = _load_candles(
        cache_dir=args.cache_dir,
        coins=coins,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        fetch_missing=args.fetch_missing,
    )
    sweep = run_cross_coin_strategy_sweep(
        candles,
        strategies=strategies,
        min_candles=args.min_candles,
    )

    wf_coins = [c.strip().upper() for c in args.walk_forward_coins.split(",") if c.strip()]
    walk_forward = []
    for coin in wf_coins:
        coin_candles = candles.get(coin, [])
        if len(coin_candles) < args.min_candles:
            continue
        for strategy in ("rsi", "mean_reversion"):
            walk_forward.append(asdict(run_walk_forward_parameter_optimization(
                coin_candles,
                coin=coin,
                strategy=strategy,
                param_grid=DEFAULT_WALK_FORWARD_GRIDS[strategy],
                train_days=args.train_days,
                test_days=args.test_days,
                step_days=args.step_days,
                min_train_candles=args.min_candles,
                min_test_candles=max(10, args.min_candles // 2),
            )))

    report = {
        "config": {
            "cache_dir": args.cache_dir,
            "timeframe": args.timeframe,
            "start": args.start,
            "end": args.end,
            "coins": coins,
            "strategies": strategies,
            "fetch_missing": args.fetch_missing,
        },
        "cross_coin_sweep": asdict(sweep),
        "walk_forward": walk_forward,
    }

    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    _print_summary(report)
    print(f"\nReport written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
