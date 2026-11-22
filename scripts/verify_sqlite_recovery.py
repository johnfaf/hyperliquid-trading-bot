"""Restore immutable rollout backups into scratch files; never contact a venue."""
from argparse import ArgumentParser
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import sqlite3
from tempfile import TemporaryDirectory

CRITICAL = {"paper_account", "paper_trades", "orders", "fills", "pauses", "account_equity_samples", "evidence",
            "stock_cycles", "stock_cycle_attempts"}


def fingerprint(conn, table):
    name = '"'+table.replace('"', '""')+'"'
    digest, count = sha256(), 0
    for row in conn.execute("SELECT * FROM "+name):
        digest.update(repr(tuple(row)).encode("utf-8"))
        digest.update(b"\n")
        count += 1
    return count, digest.hexdigest()


def verify_backup(source, scratch):
    source, scratch = Path(source).resolve(), Path(scratch).resolve()
    if source == scratch or scratch.exists():
        raise ValueError("restore destination must be a new scratch file")
    with closing(sqlite3.connect(source.as_uri()+"?mode=ro", uri=True)) as original, closing(sqlite3.connect(scratch)) as restored:
        original.backup(restored, pages=1024, sleep=.05)
        if restored.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("restored database failed integrity check")
        tables = {row[0] for row in original.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        copies = {row[0] for row in restored.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables != copies:
            raise ValueError("restored schema mismatch")
        result = {}
        for table in sorted(tables & CRITICAL):
            expected = fingerprint(original, table)
            if fingerprint(restored, table) != expected:
                raise ValueError("restored risk or execution evidence mismatch")
            result[table] = {"rows": expected[0], "sha256": expected[1]}
        return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--backups", type=Path, default=Path("/srv/hyperliquid-bot/rollbacks"))
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.revision):
        parser.error("a full Git revision is required")
    root = args.backups.resolve(strict=True)
    if not (root / f"bot-pre-{args.revision}.db").is_file():
        parser.error("the pre-rollout bot backup is required")
    needed = sum(path.stat().st_size for path in root.glob(f"*-pre-{args.revision}.db"))
    if shutil.disk_usage(root).free < needed + 1024**3:
        parser.error("insufficient free space for isolated restore and reserve")
    results = {}
    with TemporaryDirectory(prefix="sqlite-recovery-", dir=root) as scratch:
        for kind in ("bot", "decisions", "orders", "risk", "stock"):
            source = root / f"{kind}-pre-{args.revision}.db"
            if source.exists():
                results[kind] = verify_backup(source, Path(scratch) / f"{kind}.db")
    print(json.dumps({"state": "verified_offline_restore", "revision": args.revision, "ledgers": results}, sort_keys=True))


if __name__ == "__main__":
    main()
