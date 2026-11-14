#!/usr/bin/env python3
"""Check an independently restored SQLite research ledger without changing it."""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path


def verify_backup(path: Path, expected_sha256: str) -> None:
    path = path.resolve(strict=True)
    if any(path.with_name(path.name + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
        raise ValueError("restore must be a standalone SQLite snapshot, not a copy of an active database")
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != expected_sha256.lower():
        raise ValueError("restored snapshot does not match the recorded SHA-256")

    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)) as conn:
        if conn.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("SQLite integrity check failed")
        if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("SQLite foreign key check failed")
        account = conn.execute(
            "SELECT total_trades, winning_trades, total_pnl FROM paper_account WHERE id = 1"
        ).fetchone()
        trades = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0), "
            "COALESCE(SUM(pnl), 0) FROM paper_trades WHERE status = 'closed'"
        ).fetchone()
    if not account or account[:2] != trades[:2] or abs(float(account[2]) - float(trades[2])) > 0.01:
        raise ValueError("paper ledger does not match account totals; preserve it and investigate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--sha256", required=True, help="Expected digest from the original backup manifest")
    args = parser.parse_args()
    try:
        verify_backup(args.snapshot, args.sha256)
    except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
        parser.exit(1, f"Backup check failed: {exc}\n")
    print("Backup check passed: digest, integrity, foreign keys and paper account totals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
