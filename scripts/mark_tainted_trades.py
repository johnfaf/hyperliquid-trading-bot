"""One-time migration: flag pre-reconciler-fix tainted paper trades.

Background
----------
The reconciler in ``src/core/live_execution.py:sync_shadow_book_to_live``
used to force-close every paper trade that had no matching live
position, at the *current adverse mid-price*.  But ~89 % of paper
trades were never attempted on the live exchange (promotion-gate
blocked, bootstrap-deferred copies, untagged sources), so the
"missing live position" was by design — and the reconciler was
silently bleeding ~3 % of paper PnL on every cycle.

PR 'reconciler-skip-non-mirrored' fixed the bug going forward.  But
the historical trade history is poisoned: 381 of 485 recent closes
were tainted reconciliation kills (-$735 net), and the firewall's
recent-loss gates + agent_scorer's per-source policy now read those
artefacts as "this strategy/source loses money" → block every new
signal from that source → no trades → calibration never improves.

This script tags every historical pre-fix tainted close with::

    metadata.tainted = true
    metadata.taint_reason = "reconciler_kill_pre_fix"

so the analytics gates (``_is_trade_tainted`` in
``src/analysis/trade_analytics.py``) can exclude them.  Behaviour is
idempotent — re-running is a no-op.

Usage
-----
    # SQLite (default DB path)
    python scripts/mark_tainted_trades.py

    # Postgres
    DATABASE_URL=postgresql://... python scripts/mark_tainted_trades.py

    # Dry run (count only, no writes)
    python scripts/mark_tainted_trades.py --dry-run

Safety
------
- Idempotent: only updates rows where ``metadata.tainted`` is not
  already set.
- Read-only by default with ``--dry-run``.
- Backup the DB first if you're paranoid.

Env opt-out for live analytics
------------------------------
After this migration, the firewall ignores tainted trades.  If you
want to compare the pre- and post-fix windows (forensics), set
``ANALYTICS_INCLUDE_TAINTED=1`` temporarily.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Iterable

logger = logging.getLogger("mark_tainted_trades")


# Pattern that identifies a tainted reconciliation close in the legacy
# data: close_reason='live_reconciled_closed' AND the trade was never
# mirrored to live (no live_mirror_status in metadata).  These two
# conditions together capture the exact bug that produced the -$735
# bleed; live_reconciled_closed alone is too broad (some are legit
# manual closes on properly-mirrored trades).
TAINT_REASON_KEY = "reconciler_kill_pre_fix"


def _connect_postgres(url: str):
    import psycopg2  # type: ignore
    return psycopg2.connect(url)


def _connect_sqlite(path: str):
    import sqlite3
    return sqlite3.connect(path)


def _run_postgres(conn, *, dry_run: bool) -> tuple[int, int]:
    """Return (n_examined, n_updated)."""
    cur = conn.cursor()
    # All closed trades whose close_reason is reconciler-driven AND
    # were never mirrored to live AND haven't already been tagged.
    cur.execute(
        """
        SELECT id, metadata
        FROM paper_trades
        WHERE status = 'closed'
          AND (
              metadata->>'close_reason' = 'live_reconciled_closed'
              OR metadata->>'reconciliation_reason' = 'live_reconciled_closed'
          )
          AND (metadata->>'live_mirror_status') IS NULL
          AND (metadata->>'tainted') IS DISTINCT FROM 'true'
        """
    )
    rows = cur.fetchall()
    n_examined = len(rows)
    n_updated = 0
    for trade_id, raw_meta in rows:
        meta = raw_meta if isinstance(raw_meta, dict) else json.loads(raw_meta or "{}")
        meta = dict(meta or {})
        meta["tainted"] = True
        meta["taint_reason"] = TAINT_REASON_KEY
        if dry_run:
            continue
        cur.execute(
            "UPDATE paper_trades SET metadata = %s::jsonb WHERE id = %s",
            (json.dumps(meta), trade_id),
        )
        n_updated += 1
    if not dry_run:
        conn.commit()
    return n_examined, n_updated


def _run_sqlite(conn, *, dry_run: bool) -> tuple[int, int]:
    cur = conn.cursor()
    # SQLite uses JSON1 extension; metadata is stored as TEXT.
    cur.execute(
        """
        SELECT id, metadata FROM paper_trades
        WHERE status = 'closed'
          AND (
              json_extract(metadata, '$.close_reason') = 'live_reconciled_closed'
              OR json_extract(metadata, '$.reconciliation_reason') = 'live_reconciled_closed'
          )
          AND json_extract(metadata, '$.live_mirror_status') IS NULL
          AND (
              json_extract(metadata, '$.tainted') IS NULL
              OR json_extract(metadata, '$.tainted') != 1
          )
        """
    )
    rows = cur.fetchall()
    n_examined = len(rows)
    n_updated = 0
    for trade_id, raw_meta in rows:
        meta = json.loads(raw_meta or "{}") if isinstance(raw_meta, str) else dict(raw_meta or {})
        meta["tainted"] = True
        meta["taint_reason"] = TAINT_REASON_KEY
        if dry_run:
            continue
        cur.execute(
            "UPDATE paper_trades SET metadata = ? WHERE id = ?",
            (json.dumps(meta), trade_id),
        )
        n_updated += 1
    if not dry_run:
        conn.commit()
    return n_examined, n_updated


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change, but make no DB writes.",
    )
    parser.add_argument(
        "--sqlite-path", default=None,
        help="Override the SQLite DB path.  Defaults to data/bot.db or "
             "the SQLITE_DB_PATH env var.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pg_url = os.environ.get("DATABASE_URL", "").strip()
    if pg_url.startswith("postgres"):
        logger.info("Connecting to Postgres (DATABASE_URL set)")
        conn = _connect_postgres(pg_url)
        n_examined, n_updated = _run_postgres(conn, dry_run=args.dry_run)
        conn.close()
    else:
        sqlite_path = (
            args.sqlite_path
            or os.environ.get("SQLITE_DB_PATH", "")
            or "data/bot.db"
        )
        if not os.path.exists(sqlite_path):
            logger.error("SQLite DB not found at %s", sqlite_path)
            return 1
        logger.info("Connecting to SQLite at %s", sqlite_path)
        conn = _connect_sqlite(sqlite_path)
        n_examined, n_updated = _run_sqlite(conn, dry_run=args.dry_run)
        conn.close()

    if args.dry_run:
        logger.info(
            "DRY RUN: %d trade(s) would be marked tainted (taint_reason=%s)",
            n_examined, TAINT_REASON_KEY,
        )
    else:
        logger.info(
            "Marked %d / %d examined trade(s) as tainted (taint_reason=%s)",
            n_updated, n_examined, TAINT_REASON_KEY,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
