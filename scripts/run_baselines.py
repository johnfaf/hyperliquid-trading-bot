#!/usr/bin/env python3
"""Generate investor baseline benchmarks from existing bot data."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.investor_evidence import (  # noqa: E402
    build_baselines,
    parse_window,
    render_baseline_markdown,
    utc_now_slug,
    write_json,
)


def _sqlite_conn(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _default_out() -> Path:
    return Path("reports") / "baselines" / f"baselines_{utc_now_slug()}.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="90d", help="Lookback window, e.g. 30d, 12w, 90")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Benchmark account capital")
    parser.add_argument("--db-path", help="Optional SQLite DB path. Defaults to configured bot DB.")
    parser.add_argument("--candle-cache-db", default="data/candle_cache.db",
                        help="SQLite candle cache for BTC/ETH buy-and-hold baselines")
    parser.add_argument("--out", default=str(_default_out()), help="Markdown output path")
    parser.add_argument("--json-out", help="Optional JSON report output path")
    parser.add_argument("--random-wallets", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fee-bps", type=float, default=4.5)
    parser.add_argument("--funding-hourly-rate", type=float, default=0.0000125)
    args = parser.parse_args(argv)

    if args.db_path:
        db_cm = _sqlite_conn(args.db_path)
    else:
        from src.data import database as db

        db_cm = db.get_connection(for_read=True)

    candle_conn = None
    candle_path = Path(args.candle_cache_db)
    if candle_path.exists():
        candle_conn = _sqlite_conn(str(candle_path))

    try:
        with db_cm as conn:
            report = build_baselines(
                conn,
                candle_conn=candle_conn,
                starting_capital=args.capital,
                window_days=parse_window(args.window),
                random_wallets=args.random_wallets,
                seed=args.seed,
                fee_bps=args.fee_bps,
                funding_hourly_rate=args.funding_hourly_rate,
            )
    finally:
        if candle_conn is not None:
            candle_conn.close()

    markdown = render_baseline_markdown(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    if args.json_out:
        write_json(args.json_out, report)
    else:
        write_json(out.with_suffix(".json"), report)
    print(f"Wrote baseline report: {out}")
    print(json.dumps({
        "out": str(out),
        "trades": report.get("bot_metrics", {}).get("trades", 0),
        "period_start": report.get("period_start"),
        "period_end": report.get("period_end"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
