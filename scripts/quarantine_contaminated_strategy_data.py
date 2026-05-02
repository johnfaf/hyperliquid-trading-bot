#!/usr/bin/env python3
"""Quarantine fixture/invalid strategy-source data in the configured DB."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import database as db  # noqa: E402


def _preview() -> dict:
    summary = {
        "mode": "dry_run",
        "invalid_traders": [],
        "invalid_golden_wallets": [],
        "invalid_strategies": [],
    }
    if db.table_exists("traders"):
        with db.get_connection(for_read=True) as conn:
            rows = conn.execute("SELECT address FROM traders WHERE active = ?", (True,)).fetchall()
        summary["invalid_traders"] = [
            str(row["address"] if hasattr(row, "keys") else row[0] or "").strip()
            for row in rows
            if not db.is_valid_trader_address(str(row["address"] if hasattr(row, "keys") else row[0] or "").strip())
        ]
    if db.table_exists("golden_wallets"):
        with db.get_connection(for_read=True) as conn:
            rows = conn.execute(
                "SELECT address FROM golden_wallets WHERE is_golden = ? OR connected_to_live = ?",
                (True, True),
            ).fetchall()
        summary["invalid_golden_wallets"] = [
            str(row["address"] if hasattr(row, "keys") else row[0] or "").strip()
            for row in rows
            if not db.is_valid_trader_address(str(row["address"] if hasattr(row, "keys") else row[0] or "").strip())
        ]
    if db.table_exists("strategies"):
        with db.get_connection(for_read=True) as conn:
            rows = conn.execute("SELECT * FROM strategies WHERE active = ?", (True,)).fetchall()
        for row in rows:
            strategy = dict(row)
            reason = db.strategy_quarantine_reason(strategy)
            if reason:
                summary["invalid_strategies"].append({
                    "id": strategy.get("id"),
                    "name": strategy.get("name"),
                    "reason": reason,
                })
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quarantine fixture/invalid strategy-source data.")
    parser.add_argument("--apply", action="store_true", help="Apply quarantine updates. Default is dry-run preview.")
    args = parser.parse_args(argv)

    if args.apply:
        summary = db.quarantine_contaminated_runtime_data()
        summary["mode"] = "apply"
    else:
        summary = _preview()
        summary["hint"] = "No changes committed; re-run with --apply to quarantine these rows."
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
