#!/usr/bin/env python3
"""Run the automated offline backtest/improvement loop.

Examples:
    python scripts/run_auto_backtest_loop.py --once
    python scripts/run_auto_backtest_loop.py --interval-seconds 21600

The loop is offline-only. It records learning candidates and reports for
operator review; it does not deploy policies or mutate live trading settings.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.boot import init_database, setup_logging  # noqa: E402
from src.learning.auto_backtest_loop import (  # noqa: E402
    AutoBacktestConfig,
    AutoBacktestLoop,
)


def _build_config(args: argparse.Namespace) -> AutoBacktestConfig:
    cfg = AutoBacktestConfig.from_env()
    overrides = {}
    for key in (
        "interval_seconds",
        "startup_delay_seconds",
        "dataset_limit",
        "reports_dir",
        "coins",
        "live_db",
        "cache_dir",
        "cache_db",
        "timeframe",
        "replay_window_days",
        "candle_research_days",
        "command_timeout_s",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if args.no_offline_learning:
        overrides["run_offline_learning"] = False
    if args.no_replay_validation:
        overrides["run_replay_validation"] = False
    if args.no_candle_research:
        overrides["run_candle_research"] = False
    if args.fetch_missing:
        overrides["candle_fetch_missing"] = True
    if args.allow_network_replay:
        overrides["replay_allow_network"] = True
    if args.manual_approval:
        overrides["manual_approval"] = True
    return replace(cfg, enabled=True, **overrides)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--interval-seconds", type=int)
    parser.add_argument("--startup-delay-seconds", type=int)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--reports-dir")
    parser.add_argument("--coins")
    parser.add_argument("--live-db")
    parser.add_argument("--cache-dir")
    parser.add_argument("--cache-db")
    parser.add_argument("--timeframe")
    parser.add_argument("--replay-window-days", type=int)
    parser.add_argument("--candle-research-days", type=int)
    parser.add_argument("--command-timeout-s", type=int)
    parser.add_argument("--no-offline-learning", action="store_true")
    parser.add_argument("--no-replay-validation", action="store_true")
    parser.add_argument("--no-candle-research", action="store_true")
    parser.add_argument("--fetch-missing", action="store_true")
    parser.add_argument("--allow-network-replay", action="store_true")
    parser.add_argument(
        "--manual-approval",
        action="store_true",
        help="Only marks eligible packages as manually approved; still does not deploy live config.",
    )
    args = parser.parse_args(argv)

    logger = setup_logging()
    init_database(logger)
    config = _build_config(args)
    loop = AutoBacktestLoop(config)

    if args.once:
        result = loop.run_cycle()
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str))
        return 0 if result.status in {"completed", "completed_with_skips"} else 1

    logging.getLogger(__name__).info(
        "Starting auto-backtest loop interval=%ss reports_dir=%s",
        config.interval_seconds,
        config.reports_dir,
    )
    loop.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
