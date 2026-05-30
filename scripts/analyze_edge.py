"""Read-only edge analysis: which (source x side x regime) buckets make money?

Mines closed paper_trades (tainted/reconciler trades excluded), groups them by
source, side and regime, and ranks each bucket by realized expectancy with a
crude significance flag (t-stat of per-trade PnL).  The point is to decide
whether there is a *proven* edge worth gating live trading to -- rather than
mirroring everything that clears a conviction floor.

SELECT-only; safe to run against prod (railway ssh "cd /app && python
scripts/analyze_edge.py").  Use --min-trades N to set the significance floor.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def _parse_meta(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw or "{}")
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}
    return {}


def _is_tainted(meta: Dict[str, Any]) -> bool:
    if meta.get("tainted"):
        return True
    reason = str(
        meta.get("close_reason") or meta.get("reconciliation_reason") or ""
    ).strip().lower()
    return reason == "live_reconciled_closed"


def _source_key(meta: Dict[str, Any], side_fallback: str = "") -> str:
    from src.signals.source_key import copy_trade_source_key

    raw = str(meta.get("source_key") or meta.get("source") or "unknown").strip().lower()
    if raw == "copy_trade" or raw.startswith("copy_trade:"):
        addr = meta.get("source_trader") or (raw.split(":", 1)[1] if ":" in raw else "")
        return str(copy_trade_source_key(addr, strict=False))
    if raw == "strategy":
        st = str(meta.get("strategy_type") or "").strip().lower()
        return f"strategy:{st}" if st else "strategy:untagged"
    return raw or "unknown"


def _bucket_regime(regime: Any) -> str:
    r = str(regime or "").strip().lower()
    if not r:
        return "unknown"
    if r in {"trending_up", "bullish"}:
        return "bull"
    if r in {"trending_down", "crash", "volatile"}:
        return "bear"
    if r in {"ranging", "neutral", "low_liquidity"}:
        return "range"
    return r


def _stats(pnls: List[float]) -> Dict[str, float]:
    n = len(pnls)
    net = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    mean = net / n if n else 0.0
    if n > 1:
        var = sum((p - mean) ** 2 for p in pnls) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    tstat = (mean / (std / math.sqrt(n))) if (std > 0 and n > 1) else 0.0
    return {
        "n": n, "net": net, "wins": wins,
        "win_rate": (wins / n) if n else 0.0,
        "mean": mean, "tstat": tstat,
    }


def _verdict(s: Dict[str, float], min_n: int) -> str:
    if s["n"] < min_n:
        return "thin"
    if s["net"] > 0 and s["win_rate"] >= 0.50 and s["tstat"] >= 1.0:
        return "EDGE"
    if s["net"] < 0 and (s["win_rate"] < 0.45 or s["tstat"] <= -1.0):
        return "LOSER"
    return "flat"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-trades", type=int, default=8,
                    help="Minimum closed trades for a bucket to be 'proven'.")
    args = ap.parse_args(argv)

    from src.data.database import get_connection

    with get_connection(for_read=True) as conn:
        rows = conn.execute(
            "SELECT coin, side, pnl, metadata FROM paper_trades "
            "WHERE status='closed' AND pnl IS NOT NULL"
        ).fetchall()

    by_full: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    by_ss: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    by_src: Dict[str, List[float]] = defaultdict(list)
    kept = tainted = 0
    for r in rows:
        d = dict(r) if hasattr(r, "keys") else {"coin": r[0], "side": r[1], "pnl": r[2], "metadata": r[3]}
        meta = _parse_meta(d.get("metadata"))
        if _is_tainted(meta):
            tainted += 1
            continue
        kept += 1
        side = str(d.get("side") or meta.get("side") or "").strip().lower()
        src = _source_key(meta, side)
        regime = _bucket_regime(meta.get("regime"))
        pnl = float(d.get("pnl") or 0.0)
        by_full[(src, side, regime)].append(pnl)
        by_ss[(src, side)].append(pnl)
        by_src[src].append(pnl)

    print(f"closed trades: kept={kept} tainted_excluded={tainted} | min_trades={args.min_trades}\n")

    def _dump(title, groups, keyfmt):
        scored = [(k, _stats(v)) for k, v in groups.items()]
        for k, s in scored:
            s["verdict"] = _verdict(s, args.min_trades)
        scored.sort(key=lambda kv: (kv[1]["net"]), reverse=True)
        print(f"=== {title} (ranked by net PnL) ===")
        print(f"  {'bucket':<46} {'n':>3} {'wr':>5} {'net$':>9} {'mean$':>7} {'t':>6}  verdict")
        for k, s in scored:
            print(f"  {keyfmt(k):<46} {s['n']:>3} {s['win_rate']*100:>4.0f}% "
                  f"{s['net']:>8.2f} {s['mean']:>7.2f} {s['tstat']:>6.2f}  {s['verdict']}")
        edges = [k for k, s in scored if s["verdict"] == "EDGE"]
        print(f"  -> proven EDGE buckets: {len(edges)}\n")
        return edges

    edges_full = _dump("source x side x regime", by_full, lambda k: f"{k[0]}|{k[1]}|{k[2]}")
    _dump("source x side", by_ss, lambda k: f"{k[0]}|{k[1]}")
    _dump("source", {(k,): v for k, v in by_src.items()}, lambda k: k[0])

    print("=== suggested live allowlist (source|side|regime) ===")
    print(json.dumps([f"{a}|{b}|{c}" for (a, b, c) in edges_full], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
