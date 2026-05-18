"""
Prune low-evidence "junk" traders from the active set
=====================================================
The traders dashboard and the copy pool draw from ``active = 1`` rows.
Past discovery cycles (and an upstream profitability-filter schema bug,
now fixed) left many rows with no actionable history -- the classic
"100% winrate / 0% ROI" accounts: a couple of lucky closes, or zero
realized PnL.  They are NOT bots; they just have nothing to copy.

This deactivates ONLY rows that are currently ``active = 1`` and fail
``db.trader_meets_evidence_bar`` (i.e. < TRADER_MIN_CLOSED_TRADES
realized closed trades, OR degenerate ``$0 pnl / 0% roi``).  It does
NOT touch:
  - rows that clear the evidence bar (real, copyable traders)
  - rows already inactive (bots / quarantined addresses)

Crucially it tags ``metadata.status = "low_evidence"`` so they are
NOT treated as bots: ``db.get_known_bot_addresses`` skips that tag, so
the next discovery cycle re-evaluates them and any that build a real
track record return to the active/copyable set automatically. Fully
reversible (the dashboard "restore" control flips ``active`` back too).

DB note: backend is dualwrite.  An UPDATE through ``db.get_connection``
executes on SQLite (authoritative) and is mirrored to Postgres -- the
same path ``db.mark_trader_inactive`` uses in production -- so a single
run cleans both stores.

Usage:
    # Dry-run (default): prints exactly what WOULD be deactivated. No writes.
    python scripts/prune_low_evidence_traders.py

    # Actually deactivate (takes a JSON backup first):
    python scripts/prune_low_evidence_traders.py --apply

LIVE CAPITAL SAFETY:
    With --apply, refuses to run when LIVE_TRADING_ENABLED is true
    unless --i-understand-the-risks is also passed.  This only ever
    flips an account out of the *copy candidate* set (reversibly) --
    it never touches trades, positions, or balances.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import src.data.database as db  # noqa: E402


def _reason(row, min_closed: int) -> str:
    try:
        tc = int(row.get("trade_count", 0) or 0)
    except (TypeError, ValueError):
        tc = 0
    try:
        pnl = float(row.get("total_pnl", 0) or 0)
    except (TypeError, ValueError):
        pnl = 0.0
    try:
        roi = float(row.get("roi_pct", 0) or 0)
    except (TypeError, ValueError):
        roi = 0.0
    if tc < min_closed:
        return f"only {tc} closed trades (< {min_closed})"
    if pnl == 0.0 and roi == 0.0:
        return f"degenerate stats ($0 pnl / 0% roi over {tc} trades)"
    return "fails evidence bar"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Actually deactivate (default is dry-run).")
    ap.add_argument("--i-understand-the-risks", action="store_true",
                    help="Required with --apply when live trading is enabled.")
    args = ap.parse_args()

    min_closed = int(getattr(config, "TRADER_MIN_CLOSED_TRADES", 10))
    print(f"DB backend: {getattr(config, 'DB_BACKEND', '?')} | "
          f"db_path: {getattr(config, 'DB_PATH', '?')}")
    print(f"Evidence bar: >= {min_closed} closed trades AND non-zero pnl/roi\n")

    try:
        active = db.get_active_traders()
    except Exception as exc:
        print(f"[ERROR] get_active_traders failed: {exc}")
        sys.exit(1)

    failing = []
    for row in active:
        if not isinstance(row, dict):
            continue
        try:
            ok = db.trader_meets_evidence_bar(row, min_closed)
        except Exception:
            ok = True  # fail-safe: never prune on predicate error
        if not ok:
            failing.append(row)

    total_active = len(active)
    print(f"active traders: {total_active}")
    print(f"  clear evidence bar (KEEP):    {total_active - len(failing)}")
    print(f"  fail evidence bar (DEACTIVATE): {len(failing)}\n")

    if failing:
        print("WOULD DEACTIVATE (tag metadata.status=low_evidence):")
        for row in sorted(failing, key=lambda r: int(r.get("trade_count", 0) or 0)):
            addr = str(row.get("address") or "")
            print(f"    {addr[:12]}  tc={int(row.get('trade_count', 0) or 0):>4}  "
                  f"pnl={float(row.get('total_pnl', 0) or 0):>10.2f}  "
                  f"roi={float(row.get('roi_pct', 0) or 0):>7.2f}%  "
                  f"wr={float(row.get('win_rate', 0) or 0):>5.2f}  "
                  f"-- {_reason(row, min_closed)}")
    else:
        print("    (none -- active set already clean)")

    if not args.apply:
        print("\n[DRY-RUN] No rows changed. Re-run with --apply to execute.")
        return

    # ---- apply path ----
    live_enabled = str(
        os.environ.get("LIVE_TRADING_ENABLED", "")
    ).strip().lower() in ("1", "true", "yes")
    if live_enabled and not args.i_understand_the_risks:
        print("\n[REFUSED] LIVE_TRADING_ENABLED is set. Re-run with "
              "--apply --i-understand-the-risks to proceed. (This only "
              "deactivates non-copyable trader rows -- reversibly -- and "
              "never touches trades/positions/balances.)")
        sys.exit(2)

    if not failing:
        print("\nNothing to deactivate. Done.")
        return

    try:
        db.backup_to_json()
        print("\n[backup] db.backup_to_json() completed.")
    except Exception as exc:
        print(f"\n[backup] WARNING: backup_to_json failed: {exc}")
        print("Proceeding anyway -- this is a reversible, row-scoped "
              "active-flag flip (see the dry-run list above).")

    now = datetime.now(timezone.utc).isoformat()
    updated = 0
    with db.get_connection() as conn:
        for row in failing:
            addr = str(row.get("address") or "").strip()
            if not addr:
                continue
            cur = conn.execute(
                "SELECT metadata FROM traders WHERE address = ?", (addr,)
            ).fetchone()
            try:
                raw_meta = cur["metadata"] if cur is not None else None
            except Exception:
                raw_meta = None
            meta = {}
            if raw_meta:
                try:
                    parsed = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
                    if isinstance(parsed, dict):
                        meta = parsed
                except (TypeError, ValueError):
                    meta = {}
            meta["status"] = "low_evidence"
            meta["low_evidence_reason"] = _reason(row, min_closed)
            meta["low_evidence_at"] = now
            conn.execute(
                "UPDATE traders SET active = ?, last_updated = ?, metadata = ? "
                "WHERE address = ?",
                (False, now, json.dumps(meta), addr),
            )
            updated += 1

    print(f"\n[APPLIED] deactivated {updated} low-evidence traders "
          f"(active=0, metadata.status=low_evidence; dualwrite -> "
          f"mirrored to Postgres).")
    print("They remain re-discoverable: get_known_bot_addresses() skips "
          "the low_evidence tag, so discovery re-evaluates them and any "
          "that build a real track record return automatically.")


if __name__ == "__main__":
    main()
