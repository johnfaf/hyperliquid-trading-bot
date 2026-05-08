"""
Surgical DB Cleanup
====================
Targeted cleanup of stale rows that are poisoning the firewall's adaptive
gates (paper_trades from before recent fixes) and the strategy table
(inactive rows from old discovery runs that no longer match the new
MAX_ACTIVE_STRATEGIES cap).

What this DOES NOT touch:
  - regime_history (XGBoost training labels)
  - shadow_trades (RL sizer training data)
  - trader_discovery (~1259 tracked traders, expensive to rediscover)
  - calibration_records (51 sources of attribution)
  - active strategies
  - bot_state (kill switch, dedup cache)
  - paper_account balance

What this DOES touch:
  - paper_trades.status = 'closed' AND closed_at < (now - PAPER_TRADES_DAYS)
  - strategies.active = 0 AND COALESCE(updated_at, discovered_at) < (now - STRATEGIES_DAYS)

Usage:
    # Dry-run (default) — counts only, no deletes:
    python scripts/db_cleanup_surgical.py

    # Actually delete:
    python scripts/db_cleanup_surgical.py --apply

    # Customise retention:
    python scripts/db_cleanup_surgical.py --apply --paper-days 21 --strategies-days 30

LIVE CAPITAL SAFETY:
    Refuses to run with --apply when LIVE_TRADING_ENABLED is true unless
    --i-understand-the-risks is also set. Even then, it ONLY deletes rows
    older than the retention window — open paper positions and recent
    closed rows are preserved untouched.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make sure we can import the bot's modules.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import src.data.database as db  # noqa: E402


def _count_paper_trades_to_delete(cutoff_iso: str) -> int:
    with db.get_connection(for_read=True) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS c FROM paper_trades
            WHERE status = 'closed' AND closed_at IS NOT NULL AND closed_at < ?
            """,
            (cutoff_iso,),
        ).fetchone()
    if row is None:
        return 0
    try:
        return int(row["c"]) if hasattr(row, "keys") else int(row[0])
    except (TypeError, ValueError):
        return 0


def _count_strategies_to_delete(cutoff_iso: str) -> int:
    with db.get_connection(for_read=True) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS c FROM strategies
            WHERE COALESCE(active, 0) = 0
              AND COALESCE(NULLIF(updated_at, ''), discovered_at) IS NOT NULL
              AND COALESCE(NULLIF(updated_at, ''), discovered_at) < ?
            """,
            (cutoff_iso,),
        ).fetchone()
    if row is None:
        return 0
    try:
        return int(row["c"]) if hasattr(row, "keys") else int(row[0])
    except (TypeError, ValueError):
        return 0


def _delete_paper_trades(cutoff_iso: str) -> int:
    with db.get_connection() as conn:
        cur = conn.execute(
            """
            DELETE FROM paper_trades
            WHERE status = 'closed' AND closed_at IS NOT NULL AND closed_at < ?
            """,
            (cutoff_iso,),
        )
        return int(getattr(cur, "rowcount", 0) or 0)


def _delete_inactive_strategies(cutoff_iso: str) -> int:
    with db.get_connection() as conn:
        cur = conn.execute(
            """
            DELETE FROM strategies
            WHERE COALESCE(active, 0) = 0
              AND COALESCE(NULLIF(updated_at, ''), discovered_at) IS NOT NULL
              AND COALESCE(NULLIF(updated_at, ''), discovered_at) < ?
            """,
            (cutoff_iso,),
        )
        return int(getattr(cur, "rowcount", 0) or 0)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Surgical DB cleanup")
    parser.add_argument(
        "--paper-days",
        type=int,
        default=21,
        help="Delete closed paper_trades older than this many days (default: 21)",
    )
    parser.add_argument(
        "--strategies-days",
        type=int,
        default=30,
        help="Delete inactive strategies older than this many days (default: 30)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete. Without this, the script reports counts only.",
    )
    parser.add_argument(
        "--i-understand-the-risks",
        action="store_true",
        help="Allow --apply while LIVE_TRADING_ENABLED is true",
    )
    args = parser.parse_args(argv[1:])

    now = datetime.now(timezone.utc)
    paper_cutoff = (now - timedelta(days=args.paper_days)).isoformat()
    strategies_cutoff = (now - timedelta(days=args.strategies_days)).isoformat()

    paper_count = _count_paper_trades_to_delete(paper_cutoff)
    strategies_count = _count_strategies_to_delete(strategies_cutoff)

    print("=" * 64)
    print(" SURGICAL DB CLEANUP")
    print("=" * 64)
    print(f" Paper trades cutoff:      {paper_cutoff}  ({args.paper_days}d)")
    print(f" Strategies cutoff:        {strategies_cutoff}  ({args.strategies_days}d)")
    print(f" Closed paper_trades to delete:        {paper_count:,}")
    print(f" Inactive strategies to delete:        {strategies_count:,}")
    print()

    if not args.apply:
        print(" [DRY-RUN] No deletes performed. Re-run with --apply to commit.")
        return 0

    live_enabled = bool(getattr(config, "LIVE_TRADING_ENABLED", False))
    if live_enabled and not args.i_understand_the_risks:
        print(
            " REFUSED: LIVE_TRADING_ENABLED is true. Re-run with both "
            "--apply and --i-understand-the-risks to proceed."
        )
        return 2

    print(" --apply set. Deleting...")
    deleted_paper = _delete_paper_trades(paper_cutoff)
    deleted_strategies = _delete_inactive_strategies(strategies_cutoff)

    print()
    print(f" Deleted {deleted_paper:,} closed paper_trades older than {args.paper_days}d")
    print(f" Deleted {deleted_strategies:,} inactive strategies older than {args.strategies_days}d")
    print()
    print(" Untouched: regime_history, shadow_trades, trader_discovery,")
    print("            calibration_records, active strategies, bot_state,")
    print("            paper_account.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
