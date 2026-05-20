#!/usr/bin/env python3
"""Generate signed live evidence packs for T+30/T+60/T+90 style reviews."""
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
    build_live_evidence_pack,
    parse_window,
    utc_now_slug,
    write_json,
)


def _sqlite_conn(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _default_out_dir() -> Path:
    return Path("reports") / "evidence" / f"live_pack_{utc_now_slug()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="90d", help="Lookback window or T horizon")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--db-path", help="Optional SQLite DB path. Defaults to configured bot DB.")
    parser.add_argument("--out-dir", default=str(_default_out_dir()))
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    if args.db_path:
        db_cm = _sqlite_conn(args.db_path)
    else:
        from src.data import database as db

        db_cm = db.get_connection(for_read=True)

    with db_cm as conn:
        pack = build_live_evidence_pack(
            conn,
            args.out_dir,
            window_days=parse_window(args.window),
            starting_capital=args.capital,
            num_trials=args.num_trials,
        )
    write_json(args.json_out or Path(args.out_dir) / "live_evidence_pack.json", pack)
    print(json.dumps({
        "report_path": pack.get("report_path"),
        "signature_path": pack.get("signature_path"),
        "source_sha256": pack.get("source_sha256"),
        "signed": pack.get("signature", {}).get("signed"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
