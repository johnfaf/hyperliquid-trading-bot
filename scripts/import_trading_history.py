#!/usr/bin/env python3
"""Archive Hyperliquid fill CSVs and optional paper-ledger sessions offline."""
from argparse import ArgumentParser
from contextlib import closing
import csv
import os
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfoNotFoundError

from src.backtest.history_archive import (
    FORMAT_VERSION, SCHEMA, SESSION, import_fills, import_paper_session, quality_report,
)


def exclusive(path):
    return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)


def create_archive(csv_paths, sessions, timezone_name, output, report):
    output, report = output.resolve(), report.resolve()
    if output == report or output.exists() or report.exists():
        raise ValueError("archive and report must be distinct new paths; nothing is overwritten")
    for path in (*csv_paths, *sessions.values()):
        if not path.is_file():
            raise ValueError(f"input not found: {path.name}")
    created = []
    try:
        fd = exclusive(output)
        created.append(output)
        os.close(fd)
        with closing(sqlite3.connect(output)) as db:
            db.execute("PRAGMA foreign_keys=ON")
            db.executescript(SCHEMA)
            with db:
                fills = import_fills(db, csv_paths, timezone_name)
                quality = {name: import_paper_session(db, name, path) for name, path in sessions.items()}
                db.execute("INSERT INTO archive_format VALUES (?)", (FORMAT_VERSION,))
            if db.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValueError("archive failed SQLite integrity check")
        fd = exclusive(report)
        created.append(report)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write(quality_report(fills, quality))
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return fills, quality


def session_arg(value):
    name, separator, path = value.partition("=")
    if not separator or not SESSION.fullmatch(name) or not path:
        raise ValueError("use NAME=PATH with a short letters/numbers/hyphens session name")
    return name, Path(path).resolve()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, action="append", default=[], help="One export; repeatable")
    parser.add_argument("--csv-dir", type=Path, help="Import trade_history*.csv from this directory")
    parser.add_argument("--csv-timezone", help="IANA timezone of the original exports, if confirmed; e.g. Europe/London")
    parser.add_argument("--ledger", action="append", default=[], metavar="NAME=PATH", help="Read-only paper ledger snapshot; repeatable")
    parser.add_argument("--out", type=Path, required=True, help="New offline archive SQLite file")
    parser.add_argument("--report", type=Path, required=True, help="New Markdown data-quality report")
    args = parser.parse_args()
    try:
        extra = []
        if args.csv_dir is not None:
            if not args.csv_dir.is_dir():
                raise ValueError("CSV directory not found")
            extra = sorted(args.csv_dir.glob("trade_history*.csv"))
            if not extra:
                raise ValueError("CSV directory has no trade_history exports")
        csv_paths = list(dict.fromkeys(p.resolve() for p in (*args.csv, *extra)))
        pairs = [session_arg(s) for s in args.ledger]
        sessions = dict(pairs)
        if len(pairs) != len(sessions) or len(set(sessions.values())) != len(sessions):
            raise ValueError("paper ledger names and snapshot paths must each be unique")
        if not csv_paths and not sessions:
            raise ValueError("provide a CSV export, directory or paper ledger snapshot")
        create_archive(csv_paths, sessions, args.csv_timezone, args.out, args.report)
    except (ValueError, OSError, csv.Error, sqlite3.Error, ZoneInfoNotFoundError) as exc:
        parser.exit(2, f"Historical import stopped: {exc}\n")
    print("Offline archive and data-quality report created. No trading data was changed.")


if __name__ == "__main__":
    main()
