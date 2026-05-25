"""Retroactively trim the strategies table to top-N per source_wallet.

Background
----------
PR #24 capped ``StrategyIdentifier.identify_strategies`` to top-N
strategies per trader (default 2), preventing future discoveries
from bloating the table.  But it does NOT trim the existing rows.
The 2026-05-24 discovery cycle had already saved 1817 strategies for
~775 humans (2.3 per trader average).  That bloat causes the
boot-time DB safe-repair AND db_audit to hang for 10+ minutes,
requiring the temporary ``DB_SAFE_AUTO_REPAIR_ON_BOOT=false`` and
``BOOT_DB_AUDIT_SKIP=true`` env-var bypasses (PR #23).

What this script does
---------------------
For each ``source_wallet`` in the ``strategies`` table:
  1. Load all active rows
  2. Sort by ``current_score`` DESC (highest-scoring first)
  3. Keep top-N (default 2, matching the runtime cap)
  4. Mark the rest ``active=false`` -- preserves history for
     forensics, hides them from ``get_active_strategies()``
  5. Optionally delete their orphan ``strategy_scores`` rows to
     reclaim space (the strategy rows themselves stay)

Output
------
- Stdout: a markdown report showing per-wallet before/after counts
  and which strategy types survived
- JSON dump in ``data/trim_strategies_<ts>.json`` for the operator

Safety
------
- ``--dry-run`` (default): reports what would change, writes nothing
- ``--apply``: actually deactivates rows
- Deactivation is reversible: re-running the runtime cap on those
  wallets would re-promote the same surviving strategies, so this
  matches the bot's go-forward behaviour.

Usage
-----
    # Show what would change (no DB writes)
    python scripts/trim_strategies_per_wallet.py --dry-run

    # Actually apply with the runtime cap (top-2)
    python scripts/trim_strategies_per_wallet.py --apply

    # More aggressive trim (top-1 per wallet)
    python scripts/trim_strategies_per_wallet.py --apply --cap 1

    # Also purge orphan strategy_scores rows
    python scripts/trim_strategies_per_wallet.py --apply --purge-scores
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make ``src`` importable when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import database as db


def _extract_source_wallet(strategy: Dict) -> Optional[str]:
    """Return the lowercase source wallet for grouping.

    Tries the ``parameters.source_wallet`` JSON field first; falls
    back to parsing the strategy ``name`` (format
    ``<type>_<addr8>``).
    """
    params_raw = strategy.get("parameters")
    if isinstance(params_raw, str):
        try:
            params = json.loads(params_raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            params = {}
    elif isinstance(params_raw, dict):
        params = params_raw
    else:
        params = {}

    src = params.get("source_wallet")
    if src:
        return str(src).strip().lower()

    # Fallback: the name encodes the 8-char prefix of the wallet
    # (e.g. "momentum_long_0xabcdef0").  This is lossy but enough
    # to group strategies from the same wallet together.
    name = strategy.get("name") or ""
    if "_" in name:
        addr_suffix = name.rsplit("_", 1)[-1].strip().lower()
        if addr_suffix.startswith("0x"):
            return addr_suffix
    return None


def plan_trim(strategies: List[Dict], cap: int) -> Tuple[List[int], List[int], Dict[str, Dict]]:
    """Decide which strategy ids to keep vs deactivate.

    Returns (keep_ids, deactivate_ids, per_wallet_report).
    """
    by_wallet: Dict[Optional[str], List[Dict]] = defaultdict(list)
    no_wallet: List[Dict] = []
    for s in strategies:
        wallet = _extract_source_wallet(s)
        if wallet:
            by_wallet[wallet].append(s)
        else:
            no_wallet.append(s)

    keep_ids: List[int] = []
    deactivate_ids: List[int] = []
    report: Dict[str, Dict] = {}

    for wallet, group in by_wallet.items():
        # Sort by current_score DESC; rows without a score fall last.
        group_sorted = sorted(
            group,
            key=lambda s: float(s.get("current_score") or 0.0),
            reverse=True,
        )
        keepers = group_sorted[:cap]
        droppers = group_sorted[cap:]
        for s in keepers:
            try:
                keep_ids.append(int(s["id"]))
            except (KeyError, TypeError, ValueError):
                continue
        for s in droppers:
            try:
                deactivate_ids.append(int(s["id"]))
            except (KeyError, TypeError, ValueError):
                continue
        report[wallet] = {
            "before": len(group),
            "after": len(keepers),
            "dropped": len(droppers),
            "kept_types": [s.get("strategy_type") for s in keepers],
            "dropped_types": [s.get("strategy_type") for s in droppers],
            "kept_scores": [round(float(s.get("current_score") or 0.0), 4) for s in keepers],
            "dropped_scores": [round(float(s.get("current_score") or 0.0), 4) for s in droppers],
        }

    # Strategies with no resolvable source_wallet stay as-is (we
    # don't have a way to group them safely).  Log them in the
    # report so the operator knows they were skipped.
    if no_wallet:
        report["__no_wallet__"] = {
            "before": len(no_wallet),
            "after": len(no_wallet),
            "dropped": 0,
            "note": (
                "Strategies without a parseable source_wallet were not "
                "grouped; they remain active.  Add a source_wallet field "
                "to their parameters JSON for future runs."
            ),
        }

    return keep_ids, deactivate_ids, report


def apply_trim(
    deactivate_ids: List[int],
    *,
    purge_scores: bool = False,
) -> Dict[str, int]:
    """Set ``active=false`` for the given ids, optionally purge scores.

    Returns a dict of counts: ``{"deactivated": N, "scores_purged": M}``.
    """
    counts = {"deactivated": 0, "scores_purged": 0}
    if not deactivate_ids:
        return counts

    with db.get_connection() as conn:
        # Batch the UPDATE in chunks of 500 to keep the SQL parameter
        # list under SQLite's default limit (999).
        for i in range(0, len(deactivate_ids), 500):
            chunk = deactivate_ids[i:i + 500]
            placeholders = ",".join("?" * len(chunk))
            cur = conn.execute(
                f"UPDATE strategies SET active = ? WHERE id IN ({placeholders})",
                (False, *chunk),
            )
            counts["deactivated"] += int(getattr(cur, "rowcount", 0) or 0)

        if purge_scores:
            # Optional: drop strategy_scores history for the deactivated
            # strategies.  Reclaims space but loses the historical
            # score trajectory.  Default OFF (keep scores).
            for i in range(0, len(deactivate_ids), 500):
                chunk = deactivate_ids[i:i + 500]
                placeholders = ",".join("?" * len(chunk))
                try:
                    cur = conn.execute(
                        f"DELETE FROM strategy_scores WHERE strategy_id IN ({placeholders})",
                        chunk,
                    )
                    counts["scores_purged"] += int(getattr(cur, "rowcount", 0) or 0)
                except Exception:
                    # The scores table may not exist on some backends;
                    # tolerate it.
                    pass

    # Note: we deliberately don't touch ``last_scored`` or
    # ``current_score`` -- they remain at their last-computed values
    # so an operator can re-examine the deactivation decision later.
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cap", type=int, default=2,
        help="Strategies to keep per wallet (default 2, matches runtime cap)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually deactivate (default: dry-run shows the plan only)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Force dry-run mode (default behaviour when --apply is absent)",
    )
    parser.add_argument(
        "--purge-scores", action="store_true",
        help="Also delete strategy_scores rows for deactivated strategies",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to write a JSON dump of the plan/result (default: auto-named under data/)",
    )
    args = parser.parse_args()

    if args.cap < 1:
        print("ERROR: --cap must be >= 1", file=sys.stderr)
        return 2

    apply_changes = bool(args.apply) and not bool(args.dry_run)

    # Load every active strategy.  We deliberately do NOT use
    # get_active_strategies() because that filters out validated_only
    # entries; here we want the full active set so the cap is
    # symmetric with the runtime detector.
    with db.get_connection(for_read=True) as conn:
        rows = conn.execute(
            "SELECT * FROM strategies WHERE active = ?",
            (True,),
        ).fetchall()
    strategies = [dict(r) for r in rows]
    total = len(strategies)
    print(f"Loaded {total} active strategies from the runtime DB.")

    if total == 0:
        print("Nothing to trim.")
        return 0

    keep_ids, deactivate_ids, report = plan_trim(strategies, args.cap)

    # ── Stdout report ────────────────────────────────────────
    print()
    print(f"## Trim plan (cap={args.cap})")
    print(f"  Wallets covered:           {len([w for w in report if w != '__no_wallet__'])}")
    print(f"  Strategies kept (active):  {len(keep_ids)}")
    print(f"  Strategies to deactivate:  {len(deactivate_ids)}")
    print(f"  Mode:                      {'APPLY' if apply_changes else 'DRY-RUN (no writes)'}")
    print()

    multi_wallets = [
        (w, r) for w, r in report.items()
        if w != "__no_wallet__" and r["before"] > args.cap
    ]
    multi_wallets.sort(key=lambda kv: kv[1]["before"], reverse=True)
    if multi_wallets:
        print("## Top 10 wallets by drop count")
        print("  wallet                       before -> after  dropped_types")
        for wallet, r in multi_wallets[:10]:
            wallet_short = (wallet[:18] + "...") if len(wallet) > 21 else wallet
            print(
                f"  {wallet_short:<24}     {r['before']:>3} -> {r['after']:<3}    "
                f"{', '.join(r['dropped_types'][:5])}"
            )
        print()

    # ── Optional JSON dump ───────────────────────────────────
    output_path = args.output_json
    if output_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = f"data/trim_strategies_{ts}.json"
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump({
                "cap": args.cap,
                "total_input": total,
                "kept_count": len(keep_ids),
                "deactivated_count": len(deactivate_ids),
                "applied": apply_changes,
                "purge_scores": bool(args.purge_scores) and apply_changes,
                "per_wallet": report,
            }, fh, indent=2, sort_keys=True)
        print(f"Plan written to {output_path}")
    except OSError as exc:
        print(f"WARN: could not write {output_path}: {exc}", file=sys.stderr)

    # ── Apply ────────────────────────────────────────────────
    if apply_changes:
        print()
        print("## Applying trim ...")
        counts = apply_trim(
            deactivate_ids,
            purge_scores=bool(args.purge_scores),
        )
        print(f"  Deactivated: {counts['deactivated']}")
        print(f"  Scores purged: {counts['scores_purged']}")
        print()
        print(
            "After this trim, ``DB_SAFE_AUTO_REPAIR_ON_BOOT`` and "
            "``BOOT_DB_AUDIT_SKIP`` can be flipped back to default "
            "and the bot should boot through both steps cleanly."
        )
    else:
        print(
            "Dry-run complete. Re-run with --apply to actually "
            "deactivate the over-cap strategies."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
