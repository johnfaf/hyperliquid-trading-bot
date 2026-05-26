"""Retroactively prune polymarket history tables.

Background
----------
The bot's signal generation only consults RECENT polymarket data
(last few hours for regime sentiment, last 24-48h for activity
scoring).  Older rows in ``polymarket_price_points`` and
``polymarket_market_snapshots`` are pure storage cost.

Production state on 2026-05-25:
  * /data/bot.db file size      ~15.4 GB
  * polymarket_price_points     2,673,890 rows
  * polymarket_market_snapshots 1,736,353 rows

These two tables alone dominate the DB size and were the underlying
cause of the boot-hang incident (PRAGMA integrity_check walks every
page, and 15 GB = many millions of pages).  PRs #27/#28 made boot
resilient to DB size via background daemon threads, but the
underlying bloat remained.

This script catches the historical rows up to the new retention
contract: keep only rows within the last ``--retention-days`` days
(default 30, matching the runtime POLYMARKET_HISTORY_RETENTION_DAYS
env var default).

Companion: ``src/data/polymarket_history.py::prune_polymarket_history``
runs the same prune opportunistically after every polymarket scan
so the bloat doesn't accumulate again.

Output
------
- Stdout: counts of rows deleted from each table
- Optional JSON dump under ``data/prune_polymarket_<ts>.json``

Safety
------
- Default is ``--dry-run`` (no writes)
- Batched DELETE in chunks of 5000 by default (configurable) so the
  write lock is released between batches; live writers compete
  normally
- Only the high-frequency history tables are pruned.  Dimension
  tables (polymarket_markets, polymarket_tokens, polymarket_trades)
  are NOT touched -- they're low-volume and the bot reads them to
  build full token chains.

Usage
-----
    # Show what would be pruned (no writes)
    python scripts/prune_polymarket_history.py --dry-run

    # Apply 30-day retention
    python scripts/prune_polymarket_history.py --apply

    # More aggressive (7-day retention)
    python scripts/prune_polymarket_history.py --apply --retention-days 7

    # Smaller batches if the DB is contended
    python scripts/prune_polymarket_history.py --apply --batch-size 1000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make ``src`` importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import database as db
from src.data.polymarket_history import prune_polymarket_history


def _table_counts() -> dict:
    """Return current row counts for the polymarket tables we care about."""
    counts = {}
    with db.get_connection(for_read=True) as conn:
        for tbl in (
            "polymarket_price_points",
            "polymarket_market_snapshots",
            "polymarket_markets",
            "polymarket_tokens",
            "polymarket_trades",
        ):
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                counts[tbl] = int(n)
            except Exception:
                counts[tbl] = None
    return counts


def _old_row_counts(retention_days: int) -> dict:
    """Estimate how many rows older than the cutoff exist (read-only)."""
    cutoff_ms = int(time.time() * 1000) - retention_days * 86_400_000
    counts = {}
    with db.get_connection(for_read=True) as conn:
        try:
            counts["price_points_old"] = int(
                conn.execute(
                    "SELECT COUNT(*) FROM polymarket_price_points "
                    "WHERE timestamp_ms < ?",
                    (cutoff_ms,),
                ).fetchone()[0]
            )
        except Exception:
            counts["price_points_old"] = None
        try:
            counts["snapshots_old"] = int(
                conn.execute(
                    "SELECT COUNT(*) FROM polymarket_market_snapshots "
                    "WHERE observed_at_ms < ?",
                    (cutoff_ms,),
                ).fetchone()[0]
            )
        except Exception:
            counts["snapshots_old"] = None
    counts["cutoff_ms"] = cutoff_ms
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retention-days", type=int, default=30,
        help="Keep rows from the last N days (default: 30)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually delete (default: dry-run shows the plan only)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Force dry-run mode (default behaviour when --apply is absent)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5000,
        help="DELETE batch size, lower = less contention (default: 5000)",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to write a JSON dump (default: data/prune_polymarket_<ts>.json)",
    )
    args = parser.parse_args()

    if args.retention_days < 0:
        print("ERROR: --retention-days must be >= 0", file=sys.stderr)
        return 2

    apply_changes = bool(args.apply) and not bool(args.dry_run)

    before = _table_counts()
    print("Before:")
    for tbl, n in before.items():
        print(f"  {tbl:<32} {n:>12,}" if isinstance(n, int) else f"  {tbl:<32} ERR")
    print()

    old = _old_row_counts(args.retention_days)
    cutoff_iso = datetime.fromtimestamp(
        old["cutoff_ms"] / 1000.0, tz=timezone.utc,
    ).isoformat()
    print(f"Retention: keep rows >= {cutoff_iso}")
    pp_old = old.get("price_points_old")
    sn_old = old.get("snapshots_old")
    print(f"  price_points older than cutoff: "
          f"{pp_old:,}" if isinstance(pp_old, int) else "  price_points older: ERR")
    print(f"  snapshots older than cutoff:    "
          f"{sn_old:,}" if isinstance(sn_old, int) else "  snapshots older: ERR")
    print()

    if not apply_changes:
        print("Dry-run -- no rows deleted.  Re-run with --apply to prune.")
        plan = {
            "applied": False,
            "retention_days": args.retention_days,
            "cutoff_iso": cutoff_iso,
            "before": before,
            "estimate_to_delete": old,
        }
    else:
        print(f"Pruning (batch_size={args.batch_size}) ...")
        started = time.time()
        result = prune_polymarket_history(
            args.retention_days, batch_size=args.batch_size,
        )
        elapsed = time.time() - started
        after = _table_counts()
        print(f"Pruned in {elapsed:.1f}s:")
        print(f"  price_points deleted:    {result['price_points_deleted']:,}")
        print(f"  market_snapshots deleted: {result['snapshots_deleted']:,}")
        print()
        print("After:")
        for tbl, n in after.items():
            print(f"  {tbl:<32} {n:>12,}" if isinstance(n, int) else f"  {tbl:<32} ERR")
        plan = {
            "applied": True,
            "retention_days": args.retention_days,
            "cutoff_iso": cutoff_iso,
            "before": before,
            "after": after,
            "deleted": result,
            "elapsed_s": round(elapsed, 2),
        }
        print()
        print("Run scripts/vacuum_runtime_db.py after this to reclaim "
              "the freed pages from disk.")

    # JSON dump for audit trail
    output_path = args.output_json or f"data/prune_polymarket_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(plan, fh, indent=2, sort_keys=True)
        print(f"Plan written to {output_path}")
    except OSError as exc:
        print(f"WARN: could not write {output_path}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
