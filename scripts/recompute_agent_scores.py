"""Rebuild agent_scores from authoritative paper_trades history.

Background
----------
Production audit (2026-05-29) found the ``agent_scores`` table's
running-total columns are decoupled from the actual trade outcomes:

    source_key                total_signals correct_signals total_pnl
    -------------------------------------------------------------------
    strategy:unknown                0           103          -303.77
    strategy:momentum_short         7             6          +468.90
    live_orphan                     0             5           +14.67
    ...12 rows with corr>total or stale accuracy

The decoupling happened because ``record_outcome`` was sometimes
called with signal_ids whose ``record_signal`` never ran for that
source_key, incrementing the columns without a matching
``trade_history`` entry.  ``_recalculate`` (which drives the source
allocator's dynamic_weight) reads ``trade_history``, so the columns
and the weight drifted apart -- momentum_short's allocator weight
(0.80) is based on 1 history entry while the column claims +$468 of
PnL across 6 wins.

This script repairs every source by replaying its REAL closed
paper_trades (excluding tainted / reconciler-close artefacts --
see src/analysis/trade_analytics._is_trade_tainted) through the
canonical ``AgentScorer.rebuild_source_from_trades`` so the columns,
trade_history, and dynamic_weight all agree.

Usage
-----
    python scripts/recompute_agent_scores.py            # SQLite default
    DATABASE_URL=postgres://... python scripts/recompute_agent_scores.py
    python scripts/recompute_agent_scores.py --dry-run  # report only

Idempotent: re-running produces the same result.  Safe to run any
time; the live process picks up the rebuilt rows on its next
restart (or immediately if it re-reads agent_scores).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("recompute_agent_scores")


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _source_key(meta: Dict[str, Any], trade: Dict[str, Any]) -> str:
    """Mirror the firewall/agent_scorer source-key derivation."""
    raw = (
        meta.get("source_key")
        or meta.get("source")
        or trade.get("source")
        or trade.get("strategy_type")
        or "unknown"
    )
    key = str(raw or "unknown").strip().lower() or "unknown"
    if key == "copy_trade":
        trader = str(meta.get("source_trader") or "").strip().lower()
        if trader:
            return f"copy_trade:{trader}"
    elif key == "strategy":
        st = str(meta.get("strategy_type") or trade.get("strategy_type") or "").strip().lower()
        if st:
            return f"strategy:{st}"
    return key


def _is_tainted(meta: Dict[str, Any]) -> bool:
    """Same predicate as src/analysis/trade_analytics._is_trade_tainted
    (kept inline so the script has no import-time side effects)."""
    if meta.get("tainted"):
        return True
    reason = str(
        meta.get("close_reason") or meta.get("reconciliation_reason") or ""
    ).strip().lower()
    return reason == "live_reconciled_closed"


def _load_closed_trades() -> List[Dict[str, Any]]:
    from src.data import database as db

    with db.get_connection(for_read=True) as conn:
        rows = conn.execute(
            "SELECT coin, side, pnl, closed_at, metadata "
            "FROM paper_trades WHERE status='closed' AND pnl IS NOT NULL "
            "ORDER BY closed_at ASC"
        ).fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            dict(row) if hasattr(row, "keys") else {
                "coin": row[0], "side": row[1], "pnl": row[2],
                "closed_at": row[3], "metadata": row[4],
            }
        )
    return out


def _group_clean_trades(trades: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for t in trades:
        meta = _parse_metadata(t.get("metadata"))
        if _is_tainted(meta):
            continue
        key = _source_key(meta, t)
        grouped[key].append({
            "pnl": float(t.get("pnl", 0.0) or 0.0),
            "return_pct": float(meta.get("return_pct", 0.0) or 0.0),
            "coin": t.get("coin", ""),
            "side": t.get("side", ""),
            "confidence": float(meta.get("confidence", 0.0) or 0.0),
            "timestamp": str(t.get("closed_at") or ""),
            "signal_id": meta.get("signal_id"),
        })
    return grouped


def _canonicalize_grouped_keys(
    grouped: Dict[str, List[Dict]]
) -> Dict[str, List[Dict]]:
    """Fold fragmented copy_trade keys into their canonical full-address key.

    The historical address-truncation bug stored the same trader under both a
    truncated key (``copy_trade:0x1ee7a73c``) and its full key
    (``copy_trade:0x1ee7a73cb5b0...``), splitting the stats so neither
    graduated from warmup.  A truncated key is merged into the UNIQUE full
    key that shares its prefix.  Truncated keys with no (or an ambiguous)
    full match are left untouched.  The merged-away truncated keys then fall
    into the recompute's phantom-reset set and zero out.
    """
    from src.signals.source_key import is_truncated_address

    prefix = "copy_trade:"

    def _addr(key: str) -> str:
        return key[len(prefix):] if key.startswith(prefix) else ""

    full_addrs = [
        a for a in (_addr(k) for k in grouped)
        if a.startswith("0x") and len(a) == 42
    ]

    remap: Dict[str, str] = {}
    for key in grouped:
        if not key.startswith(prefix) or not is_truncated_address(key):
            continue
        short = _addr(key)
        matches = [f for f in full_addrs if f.startswith(short)]
        if len(matches) == 1:
            remap[key] = prefix + matches[0]

    if not remap:
        return grouped

    merged: Dict[str, List[Dict]] = defaultdict(list)
    for key, trades in grouped.items():
        merged[remap.get(key, key)].extend(trades)
    for src, dst in remap.items():
        logger.info(
            "Merged fragmented key %s -> %s (%d trades consolidated)",
            src, dst, len(grouped[src]),
        )
    return dict(merged)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the rebuilt scores without writing.")
    parser.add_argument(
        "--keep-phantoms", action="store_true",
        help="Do NOT reset agent_scores rows that have no clean trades. "
             "By default such phantom rows (e.g. strategy:unknown with "
             "correct_signals=103 / total_signals=0, accumulated purely "
             "via the record_outcome-without-record_signal bug) are reset "
             "to zero so the scorecard reflects only real outcomes.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from src.signals.agent_scoring import AgentScorer

    trades = _load_closed_trades()
    grouped = _group_clean_trades(trades)
    grouped = _canonicalize_grouped_keys(grouped)
    logger.info(
        "Loaded %d closed trades -> %d sources after tainted filter + key merge",
        len(trades), len(grouped),
    )

    scorer = AgentScorer()

    # Phantom sources: rows already loaded into agent_scores (from the
    # DB at __init__) that have NO clean trades.  These are the rows the
    # record_outcome-without-record_signal bug polluted (corr>total,
    # stale accuracy) -- e.g. strategy:unknown, live_orphan,
    # strategy:copy_trade.  Reset them to zero unless --keep-phantoms.
    existing_keys = set(getattr(scorer, "scores", {}) or {})
    clean_keys = set(grouped.keys())
    phantom_keys = sorted(existing_keys - clean_keys)

    summary: List[Dict[str, Any]] = []
    for source_key, source_trades in sorted(grouped.items()):
        rebuilt = scorer.rebuild_source_from_trades(
            source_key, source_trades, persist=not args.dry_run,
        )
        summary.append({
            "source_key": source_key,
            "n": rebuilt.total_signals,
            "correct": rebuilt.correct_signals,
            "accuracy": round(rebuilt.accuracy, 3),
            "pnl": round(rebuilt.total_pnl, 2),
            "dynamic_weight": round(rebuilt.dynamic_weight, 3),
            "phantom_reset": False,
        })

    reset_count = 0
    if not args.keep_phantoms:
        for source_key in phantom_keys:
            rebuilt = scorer.rebuild_source_from_trades(
                source_key, [], persist=not args.dry_run,
            )
            reset_count += 1
            summary.append({
                "source_key": source_key,
                "n": rebuilt.total_signals,
                "correct": rebuilt.correct_signals,
                "accuracy": round(rebuilt.accuracy, 3),
                "pnl": round(rebuilt.total_pnl, 2),
                "dynamic_weight": round(rebuilt.dynamic_weight, 3),
                "phantom_reset": True,
            })
        logger.info(
            "Reset %d phantom source(s) with no clean trades: %s",
            reset_count, ", ".join(phantom_keys) or "(none)",
        )

    summary.sort(key=lambda r: -r["n"])
    mode = "DRY RUN (no writes)" if args.dry_run else "WROTE"
    logger.info("%s %d rebuilt source scores", mode, len(summary))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
