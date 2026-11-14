#!/usr/bin/env python3
"""Import unaggregated Hyperliquid JSON fills into a dedicated offline ledger."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from contextlib import closing
from decimal import Decimal, InvalidOperation
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS hyperliquid_fills (
    account TEXT NOT NULL,
    coin TEXT NOT NULL,
    time_ms INTEGER NOT NULL,
    tid INTEGER NOT NULL,
    oid INTEGER NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (account, coin, time_ms, tid, oid)
)
"""


def _fill_record(account: str, fill: dict) -> tuple:
    if not isinstance(fill, dict):
        raise ValueError("expected a raw Hyperliquid fill object")
    ids = tuple(fill.get(key) for key in ("time", "tid", "oid"))
    if any(type(value) is not int or not 0 <= value < 2**63 for value in ids) or ids[0] == 0:
        raise ValueError("each fill needs exchange time, tid and oid; use unaggregated JSON, not summary CSV")
    coin = fill.get("coin")
    if not isinstance(coin, str) or not coin.strip() or fill.get("side") not in {"A", "B"}:
        raise ValueError("each fill needs its venue asset ID and side")
    for field in ("px", "sz", "fee", "closedPnl", "startPosition"):
        value = fill.get(field)
        if not isinstance(value, str):
            raise ValueError(f"expected decimal string for fill field {field}")
        try:
            number = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"invalid fill field {field}") from exc
        if not number.is_finite() or (field in {"px", "sz"} and number <= 0):
            raise ValueError(f"invalid fill field {field}")
    return (account, coin, *ids, json.dumps(fill, sort_keys=True, allow_nan=False))


def import_fills(path: Path, account: str, raw_fills: list) -> tuple[int, int]:
    account = account.lower()
    if not re.fullmatch(r"0x[0-9a-f]{40}", account):
        raise ValueError("account must be the public Hyperliquid account address")
    if not isinstance(raw_fills, list):
        raise ValueError("input must be an array of unaggregated fills")
    records = [_fill_record(account, fill) for fill in raw_fills]
    added = 0
    with closing(sqlite3.connect(path)) as conn:
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            existing_tables = {name for (name,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )}
            if existing_tables - {"hyperliquid_fills"}:
                raise ValueError("write only to a dedicated Hyperliquid fill ledger")
            conn.execute(SCHEMA)
            for record in records:
                inserted = conn.execute(
                    "INSERT INTO hyperliquid_fills VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                    record,
                )
                if inserted.rowcount:
                    added += 1
                    continue
                previous = conn.execute(
                    "SELECT payload FROM hyperliquid_fills WHERE account=? AND coin=? AND time_ms=? AND tid=? AND oid=?",
                    record[:-1],
                ).fetchone()
                if previous != (record[-1],):
                    raise ValueError(f"conflicting fill record for {record[1:-1]}; import rolled back")
    return added, len(records) - added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Raw JSON array; obtain with aggregateByTime=false")
    parser.add_argument("--ledger", type=Path, required=True, help="Separate SQLite ledger, never the bot database")
    parser.add_argument("--account", required=True, help="Public source account address; do not supply any signing key")
    args = parser.parse_args()
    try:
        fills = json.loads(args.input.read_text(encoding="utf-8-sig"))
        added, duplicates = import_fills(args.ledger, args.account, fills)
    except (OSError, sqlite3.Error, ValueError) as exc:
        parser.exit(1, f"Fill import failed: {exc}\n")
    print(f"Imported {added} fills; {duplicates} identical records already present. Not reconciled to a statement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
