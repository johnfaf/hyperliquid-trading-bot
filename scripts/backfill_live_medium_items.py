"""Backfill remaining medium-severity deployed DB audit items.

This script reuses the app's existing historical-source and regime-refresh
logic, but it must choose the target SQLite DB before importing app modules.
That keeps the DB path explicit without monkey-patching the global database
module in a process that might already be running worker threads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fetch_candles(cache_dir: str) -> dict[str, Any]:
    from src.backtest.data_fetcher import DataFetcher

    fetcher = DataFetcher(cache_dir=cache_dir)
    refreshed = []
    for coin in ("BTC", "ETH"):
        candles = fetcher.fetch_candles(coin, "1h", use_cache=True)
        refreshed.append({"coin": coin, "candles": len(candles)})
    return {"cache_dir": cache_dir, "refreshed": refreshed}


def run(db_path: str, candle_cache_dir: str) -> dict[str, Any]:
    _configure_target_db(db_path)

    from src.data import db_audit

    actions: list[db_audit.DbRepairAction] = []
    pre_audit = db_audit.run_db_audit(include_code_scan=False)
    db_audit._repair_historical_sources(actions)
    stale_non_active = list(
        (pre_audit.checks.get("regime_history", {}) or {}).get("stale_other", [])
    )
    db_audit._repair_non_active_regime_history(actions, stale_non_active)
    candle_result = _fetch_candles(candle_cache_dir)
    post_audit = db_audit.run_db_audit(include_code_scan=False)

    return {
        "db_path": db_path,
        "candle_cache_dir": candle_cache_dir,
        "pre_audit": pre_audit.to_dict(block_severity="high"),
        "post_audit": post_audit.to_dict(block_severity="high"),
        "actions": [action.to_dict() for action in actions],
        "candle_cache": candle_result,
    }


def _configure_target_db(db_path: str) -> None:
    loaded = [name for name in ("config", "src.data.database", "src.data.db.router") if name in sys.modules]
    if loaded:
        raise RuntimeError(
            "Target DB must be configured before app database modules are imported; "
            f"already loaded: {', '.join(loaded)}"
        )
    os.environ["HL_BOT_DB"] = str(Path(db_path).resolve())
    os.environ["DB_BACKEND"] = "sqlite"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill deployed medium-severity DB items.")
    parser.add_argument(
        "--db-path",
        default=str(os.environ.get("HL_BOT_DB", "") or "/data/bot.db"),
        help="SQLite DB path (default: HL_BOT_DB or /data/bot.db)",
    )
    parser.add_argument(
        "--candle-cache-dir",
        default=None,
        help="Directory containing candle_cache.db (default: DB directory)",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args(argv)

    db_path = str(args.db_path)
    candle_cache_dir = args.candle_cache_dir or str(Path(db_path).resolve().parent)
    result = run(db_path, candle_cache_dir)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"db_path: {result['db_path']}")
        print(f"candle_cache_dir: {result['candle_cache_dir']}")
        print(f"pre_findings: {result['pre_audit']['finding_count']}")
        print(f"post_findings: {result['post_audit']['finding_count']}")
        print(json.dumps(result["candle_cache"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
