"""Backfill remaining medium-severity deployed DB audit items.

This script reuses the app's existing historical-source and regime-refresh
logic, but patches database access to a direct SQLite connection with an
aggressive busy timeout. It is intended for one-off maintenance on a deployed
volume when the main app process is alive and routed DB writes are too noisy.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.backtest.data_fetcher import DataFetcher
from src.data import database as db
from src.data import db_audit


@contextmanager
def _sqlite_ctx(path: str, *, for_read: bool = False):
    conn = sqlite3.connect(path, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        if not for_read:
            conn.commit()
    finally:
        conn.close()


class _PatchedDb:
    def __init__(self, path: str):
        self.path = path
        self._orig_get_connection = db.get_connection
        self._orig_get_backend_name = db.get_backend_name
        self._orig_get_db_path = db.get_db_path

    def __enter__(self):
        db.get_connection = lambda for_read=False: _sqlite_ctx(self.path, for_read=for_read)
        db.get_backend_name = lambda: "sqlite"
        db.get_db_path = lambda: self.path
        return self

    def __exit__(self, exc_type, exc, tb):
        db.get_connection = self._orig_get_connection
        db.get_backend_name = self._orig_get_backend_name
        db.get_db_path = self._orig_get_db_path


def _fetch_candles(cache_dir: str) -> dict[str, Any]:
    fetcher = DataFetcher(cache_dir=cache_dir)
    refreshed = []
    for coin in ("BTC", "ETH"):
        candles = fetcher.fetch_candles(coin, "1h", use_cache=True)
        refreshed.append({"coin": coin, "candles": len(candles)})
    return {"cache_dir": cache_dir, "refreshed": refreshed}


def run(db_path: str, candle_cache_dir: str) -> dict[str, Any]:
    actions: list[db_audit.DbRepairAction] = []
    with _PatchedDb(db_path):
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
