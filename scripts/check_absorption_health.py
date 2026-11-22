"""Read-only public feed liveness probe, independent of frozen research rules."""
from argparse import ArgumentParser
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sqlite3


def healthy(path, *, now, max_age_seconds=120):
    try:
        with closing(sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True, timeout=2)) as conn:
            stamps = [conn.execute("SELECT at FROM frames WHERE coin=? AND state='ready' ORDER BY at DESC LIMIT 1", (coin,)).fetchone()
                      for coin in ("BTC", "ETH")]
        return all(row and 0 <= now.timestamp()-row[0]/1000 <= max_age_seconds for row in stamps)
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return False


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    args = parser.parse_args()
    passed = healthy(args.db, now=datetime.now(timezone.utc))
    print("public feed fresh" if passed else "public feed missing or stale")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
