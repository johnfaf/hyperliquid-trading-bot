"""Run candle research reports, including regime-gated strategy enablement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.backtest.candle_backtester import CandleBacktestConfig
from src.backtest.data_fetcher import DataFetcher
from src.backtest.research_suite import (
    DEFAULT_10_COIN_UNIVERSE,
    DEFAULT_12_STRATEGIES,
    run_regime_conditional_strategy_enablement,
)


def _split_csv(value: str) -> list[str]:
    return [part.strip().upper() for part in str(value or "").split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coins", default=",".join(DEFAULT_10_COIN_UNIVERSE))
    parser.add_argument("--strategies", default=",".join(DEFAULT_12_STRATEGIES))
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--segment-candles", type=int, default=168)
    parser.add_argument("--min-segment-candles", type=int, default=60)
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--position-size-pct", type=float, default=0.05)
    parser.add_argument("--max-leverage", type=float, default=3.0)
    parser.add_argument("--report-out", default="reports/candle_research.json")
    args = parser.parse_args()

    fetcher = DataFetcher(cache_dir=args.cache_dir)
    candle_sets = {}
    for coin in _split_csv(args.coins):
        candles = fetcher.fetch_candles(
            coin,
            args.timeframe,
            start=args.start,
            end=args.end,
            use_cache=True,
        )
        if candles:
            candle_sets[coin] = candles

    cfg = CandleBacktestConfig(
        initial_balance=args.initial_balance,
        position_size_pct=args.position_size_pct,
        max_leverage=args.max_leverage,
    )
    report = run_regime_conditional_strategy_enablement(
        candle_sets,
        strategies=[s.strip() for s in args.strategies.split(",") if s.strip()],
        config=cfg,
        segment_candles=args.segment_candles,
        min_segment_candles=args.min_segment_candles,
    ).to_dict()

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary = report.get("summary", {})
    print(
        "regime research: "
        f"always=${summary.get('always_on_pnl', 0):+.2f}, "
        f"gated=${summary.get('regime_gated_pnl', 0):+.2f}, "
        f"delta=${summary.get('pnl_delta', 0):+.2f}"
    )
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

