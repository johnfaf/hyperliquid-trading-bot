"""VACUUM the runtime SQLite DB at /data/bot.db.

Background
----------
The bot's runtime DB at ``/data/bot.db`` accumulates orphan rows over
time -- mostly ``strategy_scores`` history for retired strategies.
The cleanup CLI ``scripts/trim_strategies_per_wallet.py`` and direct
``DELETE`` operations remove the logical rows, but SQLite keeps the
freed pages on a freelist instead of returning them to the OS.

On the 2026-05-25 production incident the runtime DB held ~398K
``strategy_scores`` rows, of which 95% were dead history.  Deleting
the dead rows brought the row count down to ~21K, but the DB FILE
still occupied the original page count.  ``PRAGMA integrity_check``
in ``run_db_audit`` walks every page (live + freelist) so the boot
audit continued to hang.

VACUUM rewrites the entire SQLite file, dropping the freelist pages.
After VACUUM the DB file size on disk reflects only the live rows,
and PRAGMA-style full scans complete in seconds.

Cost
----
VACUUM acquires an EXCLUSIVE lock on the database for the duration.
On a bot.db of a few hundred MB this typically takes 1-5 minutes.
ANY OTHER WRITER WILL BLOCK during VACUUM.  For the live trading
bot, that means trading-cycle writes, ws_position_monitor writes,
decision_journal writes all queue until VACUUM completes.

Recommended usage
-----------------
Run during a planned maintenance window:
  1. Scale the bot to zero (``railway down`` or via the dashboard)
     so no writers contend with VACUUM.
  2. ``railway ssh "python scripts/vacuum_runtime_db.py"``
  3. Scale back up.

Or, accept brief disruption and let it run live -- this script
reports the before/after sizes and elapsed time so you can decide
whether the disruption was acceptable.

Safety
------
The script:
  - Logs the DB file size before / after
  - Sets a busy_timeout so the EXCLUSIVE lock acquisition waits up
    to ``--busy-timeout-ms`` (default 30s) before failing
  - ``--dry-run`` reports the expected work without doing anything
    (uses PRAGMA freelist_count to estimate page reclaim)
  - Catches operational errors and reports them without crashing

Usage
-----
    # Show what would be reclaimed (no writes)
    python scripts/vacuum_runtime_db.py --dry-run

    # Actually VACUUM (waits up to 30s for the exclusive lock)
    python scripts/vacuum_runtime_db.py

    # Wait longer for the exclusive lock on a busy DB
    python scripts/vacuum_runtime_db.py --busy-timeout-ms 120000
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

# Make ``src`` importable when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402  -- needs sys.path mutation first


def _db_file_size_bytes(db_path: str) -> int:
    """Return the size in bytes of the SQLite DB file, 0 if missing."""
    try:
        return os.path.getsize(db_path)
    except OSError:
        return 0


def _human_bytes(n: int) -> str:
    """Format a byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def inspect_db(db_path: str) -> dict:
    """Return a snapshot of useful PRAGMA counters before/after VACUUM.

    Used in both --dry-run mode (just inspection) and as the before/
    after snapshot for the real VACUUM run.
    """
    snapshot = {
        "db_path": db_path,
        "file_size_bytes": _db_file_size_bytes(db_path),
        "page_count": 0,
        "page_size": 0,
        "freelist_count": 0,
    }
    try:
        # Open with a short busy_timeout for the inspection -- if we
        # can't even read PRAGMAs because of a writer, defer.
        conn = sqlite3.connect(db_path, timeout=5.0)
        try:
            for pragma in ("page_count", "page_size", "freelist_count"):
                row = conn.execute(f"PRAGMA {pragma}").fetchone()
                snapshot[pragma] = int(row[0]) if row and row[0] is not None else 0
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        snapshot["error"] = str(exc)
    return snapshot


def run_vacuum(db_path: str, busy_timeout_ms: int = 30_000) -> dict:
    """Execute ``VACUUM`` on the given DB file.

    Returns a dict with before/after snapshots, elapsed seconds, and
    an ``error`` key when something went wrong.
    """
    before = inspect_db(db_path)
    if "error" in before:
        return {"before": before, "after": before, "elapsed_s": 0.0,
                "error": f"Pre-VACUUM inspection failed: {before['error']}"}

    result: dict = {"before": before}
    started = time.time()
    try:
        # Open with a longer busy_timeout so we wait for live writers
        # to release their locks rather than failing on the first
        # contention.  ``isolation_level=None`` puts us in autocommit
        # mode, which is required for VACUUM (it can't run inside a
        # transaction).
        conn = sqlite3.connect(
            db_path,
            timeout=busy_timeout_ms / 1000.0,
            isolation_level=None,
        )
        try:
            conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
            conn.execute("VACUUM")
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        result["elapsed_s"] = round(time.time() - started, 2)
        result["after"] = inspect_db(db_path)
        result["error"] = str(exc)
        return result

    result["elapsed_s"] = round(time.time() - started, 2)
    result["after"] = inspect_db(db_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report freelist + file size, do not VACUUM",
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Override DB path (default: config.DB_PATH or /data/bot.db)",
    )
    parser.add_argument(
        "--busy-timeout-ms", type=int, default=30_000,
        help="Wait up to N ms for the EXCLUSIVE lock (default 30000 = 30s)",
    )
    args = parser.parse_args()

    db_path = args.db_path or getattr(config, "DB_PATH", "/data/bot.db")
    print(f"Target DB: {db_path}")

    snapshot = inspect_db(db_path)
    if "error" in snapshot:
        print(f"ERROR: cannot inspect DB: {snapshot['error']}", file=sys.stderr)
        return 2

    page_size = snapshot["page_size"] or 0
    page_count = snapshot["page_count"] or 0
    freelist = snapshot["freelist_count"] or 0
    live_pages = max(0, page_count - freelist)
    reclaim_estimate = freelist * page_size
    on_disk = snapshot["file_size_bytes"]

    print(f"  on-disk size:    {_human_bytes(on_disk)} ({on_disk:,} bytes)")
    print(f"  page size:       {page_size:,} bytes")
    print(f"  total pages:     {page_count:,}")
    print(f"  live pages:      {live_pages:,}")
    print(f"  freelist pages:  {freelist:,}  ({100 * freelist / max(page_count, 1):.1f}%)")
    print(f"  reclaim est.:    {_human_bytes(reclaim_estimate)} ({reclaim_estimate:,} bytes)")
    print()

    if args.dry_run:
        print("Dry-run: VACUUM would reclaim the freelist pages above. "
              "Re-run without --dry-run to apply.")
        return 0

    if freelist == 0:
        print("No freelist pages to reclaim -- VACUUM would only rewrite "
              "live pages.  Exiting without VACUUM.")
        return 0

    print(f"Running VACUUM (busy_timeout={args.busy_timeout_ms} ms) ...")
    result = run_vacuum(db_path, busy_timeout_ms=args.busy_timeout_ms)

    after = result.get("after", {})
    elapsed = result.get("elapsed_s", 0.0)
    if "error" in result:
        print(f"VACUUM FAILED after {elapsed}s: {result['error']}",
              file=sys.stderr)
        return 3

    after_size = after.get("file_size_bytes", 0)
    reclaimed = on_disk - after_size
    print()
    print(f"VACUUM complete in {elapsed}s")
    print(f"  before:  {_human_bytes(on_disk)} ({on_disk:,} bytes)")
    print(f"  after:   {_human_bytes(after_size)} ({after_size:,} bytes)")
    print(f"  reclaimed: {_human_bytes(reclaimed)} ({reclaimed:,} bytes)")
    print(f"  pages:   {page_count:,} -> {after.get('page_count', 0):,}")
    print(f"  freelist now: {after.get('freelist_count', 0):,}")
    print()
    print("Safe to flip DB_SAFE_AUTO_REPAIR_ON_BOOT and BOOT_DB_AUDIT_SKIP "
          "back to default once this completes -- the boot-time scans now "
          "have a normal page count to walk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
