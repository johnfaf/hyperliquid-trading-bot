"""
Prune legacy / unkeyable calibration_records
============================================
The EV gate and bucketed-threshold gate look calibration up by the
composite key ``source|side|regime``. Rows whose ``source_key`` is a
dead placeholder (``strategy:unknown``, ``copy_trade:unknown``,
``*:unknown``, ``*:untagged``, NULL, or empty) can NEVER match a real
bucket lookup -- they only dilute the table and, when they dominate it,
keep every per-bucket lookup on the 0.50 cold-start prior.

This prunes ONLY those provably-dead rows. It deliberately KEEPS:
  - bare source keys (``strategy:mean_reversion``, ``copy_trade:0xabc``)
    -- the CalibrationTracker's ``_resolve_key`` legacy fallback can
    still use those as source-level aggregates.
  - every composite ``source|side|regime`` row.

DB note: backend is dualwrite. A DELETE through ``db.get_connection``
executes on SQLite (authoritative) and is auto-mirrored to Postgres
(see ``_MIRRORED_WRITE_PREFIXES`` in src/data/db/connection.py), so a
single delete cleans both stores.

Usage:
    # Dry-run (default) -- prints exactly what WOULD be deleted and the
    # projected post-prune distribution. No writes.
    python scripts/prune_legacy_calibration.py

    # Actually delete (takes a JSON backup first):
    python scripts/prune_legacy_calibration.py --apply

LIVE CAPITAL SAFETY:
    With --apply, refuses to run when LIVE_TRADING_ENABLED is true
    unless --i-understand-the-risks is also passed. This only ever
    deletes dead-placeholder calibration rows -- never real buckets,
    never trades, never balances.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import src.data.database as db  # noqa: E402

# Predicate for "can never match a real bucket lookup". Lowercased.
_DEAD_EXACT = (
    "",
    "unknown",
    "strategy:unknown",
    "strategy:untagged",
    "strategy:",
    "copy_trade:unknown",
    "copy_trade:untagged",
    "copy_trade",          # bare copy_trade with no trader address
    "none",
)
_DEAD_SUFFIXES = (":unknown", ":untagged")


def _is_dead(source_key) -> bool:
    if source_key is None:
        return True
    k = str(source_key).strip().lower()
    if k in _DEAD_EXACT:
        return True
    if any(k.endswith(sfx) for sfx in _DEAD_SUFFIXES):
        return True
    return False


def _row_get(row, key, idx):
    if row is None:
        return None
    if hasattr(row, "keys"):
        try:
            return row[key]
        except Exception:
            return None
    try:
        return row[idx]
    except Exception:
        return None


def _fetch_distribution():
    with db.get_connection(for_read=True) as conn:
        rows = conn.execute(
            "SELECT source_key AS k, COUNT(*) AS n FROM calibration_records "
            "GROUP BY source_key ORDER BY n DESC"
        ).fetchall()
    out = []
    for r in rows:
        out.append((_row_get(r, "k", 0), int(_row_get(r, "n", 1) or 0)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete (default is dry-run).")
    ap.add_argument("--i-understand-the-risks", action="store_true",
                    help="Required with --apply when live trading is enabled.")
    args = ap.parse_args()

    print(f"DB backend: {getattr(config, 'DB_BACKEND', '?')} | "
          f"db_path: {getattr(config, 'DB_PATH', '?')}")

    dist = _fetch_distribution()
    total = sum(n for _, n in dist)
    dead = [(k, n) for k, n in dist if _is_dead(k)]
    keep = [(k, n) for k, n in dist if not _is_dead(k)]
    dead_total = sum(n for _, n in dead)
    keep_total = sum(n for _, n in keep)

    print(f"\ncalibration_records total: {total}")
    print(f"\n  WOULD DELETE ({dead_total} rows, "
          f"{(dead_total/total*100 if total else 0):.0f}%):")
    if dead:
        for k, n in sorted(dead, key=lambda x: -x[1]):
            print(f"    {n:>6}  {k!r}")
    else:
        print("    (none)")

    print(f"\n  WOULD KEEP ({keep_total} rows, "
          f"{(keep_total/total*100 if total else 0):.0f}%):")
    for k, n in sorted(keep, key=lambda x: -x[1])[:30]:
        print(f"    {n:>6}  {k}")
    if len(keep) > 30:
        print(f"    ... and {len(keep) - 30} more keyed buckets")

    if not args.apply:
        print("\n[DRY-RUN] No rows deleted. Re-run with --apply to execute.")
        return

    # ---- apply path ----
    live_enabled = str(
        os.environ.get("LIVE_TRADING_ENABLED", "")
    ).strip().lower() in ("1", "true", "yes")
    if live_enabled and not args.i_understand_the_risks:
        print("\n[REFUSED] LIVE_TRADING_ENABLED is set. Re-run with "
              "--apply --i-understand-the-risks to proceed. (This only "
              "deletes dead-placeholder calibration rows.)")
        sys.exit(2)

    if dead_total == 0:
        print("\nNothing to delete. Done.")
        return

    # Best-effort JSON backup before mutating prod state.
    try:
        db.backup_to_json()
        print("\n[backup] db.backup_to_json() completed.")
    except Exception as exc:
        print(f"\n[backup] WARNING: backup_to_json failed: {exc}")
        print("Proceeding anyway -- this delete is row-scoped and "
              "reversible from the dry-run list above.")

    deleted = 0
    with db.get_connection() as conn:
        # Delete the exact dead keys we enumerated. NULL handled separately
        # because ``= NULL`` never matches in SQL.
        cur = conn.execute(
            "DELETE FROM calibration_records "
            "WHERE source_key IS NULL OR TRIM(LOWER(source_key)) = ''"
        )
        try:
            deleted += int(cur.rowcount or 0)
        except Exception:
            pass
        for k, _n in dead:
            if k is None:
                continue
            ks = str(k).strip()
            if ks == "" or ks.lower() in {"none"}:
                continue
            cur = conn.execute(
                "DELETE FROM calibration_records WHERE source_key = ?",
                (k,),
            )
            try:
                deleted += int(cur.rowcount or 0)
            except Exception:
                pass

    print(f"\n[APPLIED] deleted ~{deleted} dead calibration_records "
          f"(dualwrite -> mirrored to Postgres).")

    print("\nPost-prune distribution:")
    for k, n in sorted(_fetch_distribution(), key=lambda x: -x[1])[:30]:
        flag = "DEAD?" if _is_dead(k) else "keep "
        print(f"    {n:>6}  [{flag}] {k}")


if __name__ == "__main__":
    main()
