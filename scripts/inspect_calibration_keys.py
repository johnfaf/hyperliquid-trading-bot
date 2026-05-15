#!/usr/bin/env python3
# ruff: noqa: E402
"""
Calibration key health inspector  (READ-ONLY)
=============================================
Answers one question: are the calibration_records / decision_outcomes
rows usefully keyed by (source, side, regime), or are they mostly
legacy ``strategy:unknown`` / ``strategy:untagged`` debt from before
the source_key fix?

If most rows are legacy-unknown, the EV gate stays on the 0.50
cold-start prior for a long time because the well-keyed buckets start
from zero. If the distribution is varied, the data is just
accumulating and the right move is to wait.

Usage (run from the repo root, in the SAME environment as the bot so
it hits the same database -- see the bottom of this file for Railway
instructions):

    python scripts/inspect_calibration_keys.py

This script never writes. It only runs SELECT/COUNT queries.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data import database as db

_LEGACY_MARKERS = ("unknown", "untagged", ":", "")  # ":" bare prefix etc.


def _is_legacy(source_key: str) -> bool:
    k = str(source_key or "").strip().lower()
    if not k:
        return True
    if k in ("unknown", "strategy:unknown", "strategy:untagged", "strategy:"):
        return True
    if k.endswith(":unknown") or k.endswith(":untagged"):
        return True
    return False


def _table_exists(conn, name: str) -> bool:
    try:
        backend = getattr(conn, "backend", "sqlite")
        if backend == "postgres":
            row = conn.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_name = ?",
                (name,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (name,),
            ).fetchone()
        return row is not None
    except Exception:
        try:
            conn.execute(f"SELECT 1 FROM {name} LIMIT 1").fetchone()
            return True
        except Exception:
            return False


def _dump_distribution(conn, table: str, key_col: str = "source_key") -> None:
    print(f"\n{'='*64}\n{table}  (key column: {key_col})\n{'='*64}")
    if not _table_exists(conn, table):
        print(f"  [table '{table}' does not exist in this database]")
        return
    try:
        total = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()
        total = (total["c"] if isinstance(total, dict) else total[0]) or 0
    except Exception as exc:
        print(f"  [count failed: {exc}]")
        return
    if total == 0:
        print("  [0 rows]")
        return
    print(f"  total rows: {total}")

    try:
        rows = conn.execute(
            f"SELECT {key_col} AS k, COUNT(*) AS n FROM {table} "
            f"GROUP BY {key_col} ORDER BY n DESC LIMIT 40"
        ).fetchall()
    except Exception as exc:
        print(f"  [group-by failed: {exc}]")
        return

    legacy = 0
    keyed = 0
    print(f"\n  {'rows':>7}  source_key")
    print(f"  {'-'*7}  {'-'*48}")
    for r in rows:
        k = r["k"] if isinstance(r, dict) else r[0]
        n = r["n"] if isinstance(r, dict) else r[1]
        flag = "LEGACY" if _is_legacy(k) else "keyed "
        if _is_legacy(k):
            legacy += n
        else:
            keyed += n
        print(f"  {n:>7}  [{flag}] {k}")

    shown = legacy + keyed
    print(f"\n  shown rows: {shown}  ->  legacy/unknown: {legacy}"
          f"  ({(legacy/shown*100 if shown else 0):.0f}%)"
          f"   well-keyed: {keyed}  ({(keyed/shown*100 if shown else 0):.0f}%)")
    if shown and legacy / shown >= 0.6:
        print("\n  VERDICT: mostly LEGACY-UNKNOWN debt. The EV gate's per-bucket"
              "\n           lookups (source|side|regime) will keep returning the"
              "\n           0.50 cold-start prior because the well-keyed buckets"
              "\n           start from zero. A one-time prune of the legacy rows"
              "\n           (or just accepting a long cold-start) is the call.")
    elif shown:
        print("\n  VERDICT: distribution is reasonably keyed. Data is accumulating"
              "\n           normally -- the right move is to let it run until"
              "\n           per-bucket sample sizes cross the 30-outcome threshold.")


def main() -> None:
    print("DB backend:", getattr(config, "DB_BACKEND", "?"),
          "| postgres_dsn_source:", getattr(config, "POSTGRES_DSN_SOURCE", "") or "(none)")
    try:
        with db.get_connection(for_read=True) as conn:
            _dump_distribution(conn, "calibration_records", "source_key")
            _dump_distribution(conn, "decision_outcomes", "source_key")
            _dump_distribution(conn, "agent_scores", "source_key")
            # decision_snapshots is the upstream source of truth for keying
            _dump_distribution(conn, "decision_snapshots", "source_key")
    except Exception as exc:
        print(f"\n[ERROR] could not open the bot database: {exc}")
        print("Make sure you run this in the SAME environment as the bot "
              "(same DB_BACKEND / POSTGRES_DSN env vars).")
        sys.exit(1)


if __name__ == "__main__":
    main()
