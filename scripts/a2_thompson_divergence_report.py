"""A2: replay paper trades through the Thompson allocator and compare
its recommended allocation against the production AgentScorer weights.

What we want to know before flipping AGENT_BANDIT_ALLOCATOR_ENABLED:

* Would the bandit have routed capital meaningfully differently from
  the current static weighted_accuracy × recent_pnl scorer?
* Is the divergence biased toward sources the static scorer is
  under-weighting (good — that's the exploration story) or toward
  sources the scorer rightly *avoided* (bad — would risk capital)?

Loads paper_trades over a window, replays per-source binary outcomes
through a fresh ThompsonAllocator (chronologically), then for each
source prints:

* trades_observed (n closed)
* win_rate (raw, ignoring fees)
* thompson_posterior_mean (the bandit's point estimate)
* thompson_wilson_lower_95 (conservative bound — for low-N sources)
* current static dynamic_weight (from agent_scores table)
* divergence = |thompson - static|

Output: markdown to stdout + JSON dump for record keeping.

Usage::

    python scripts/a2_thompson_divergence_report.py [--days 14]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import database as db
from src.signals.bandit_allocator import ThompsonAllocator


@dataclass
class TradeRow:
    closed_at_ts: float
    source_key: str
    pnl: float


@dataclass
class SourceReport:
    source_key: str
    n_trades: int
    wins: int
    win_rate: float
    thompson_mean: float
    thompson_wilson: float
    static_weight: float
    divergence: float


# ── DB load ──────────────────────────────────────────────────────────


def load_trades(days: int) -> List[TradeRow]:
    """Pull closed paper_trades with a derivable source_key.

    Source key is read from metadata.source_trader (copy_trade) or
    metadata.strategy_type, falling back to "unknown".
    """
    sql = """
        SELECT EXTRACT(EPOCH FROM closed_at) AS ts,
               COALESCE(
                   metadata->>'source_key',
                   CASE
                     WHEN metadata->>'source_trader' IS NOT NULL
                       THEN 'copy_trade:' || (metadata->>'source_trader')
                     WHEN metadata->>'strategy_type' IS NOT NULL
                       THEN 'strategy:' || (metadata->>'strategy_type')
                     ELSE 'unknown'
                   END
               ) AS source_key,
               COALESCE(pnl, 0) AS pnl
          FROM paper_trades
         WHERE status = 'closed'
           AND closed_at >= NOW() - INTERVAL %s
         ORDER BY closed_at ASC
    """
    out: List[TradeRow] = []
    with db.get_connection(for_read=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (f"{days} days",))
            for ts, src, pnl in cur.fetchall():
                out.append(TradeRow(
                    closed_at_ts=float(ts or 0.0),
                    source_key=str(src or "unknown"),
                    pnl=float(pnl or 0.0),
                ))
    return out


def load_static_weights() -> Dict[str, float]:
    """Read current production dynamic_weight per source from agent_scores."""
    out: Dict[str, float] = {}
    with db.get_connection(for_read=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT source_key, dynamic_weight FROM agent_scores")
            for src, w in cur.fetchall():
                out[str(src)] = float(w or 0.0)
    return out


# ── Replay ──────────────────────────────────────────────────────────


def replay(trades: List[TradeRow]) -> Tuple[ThompsonAllocator, Dict[str, Tuple[int, int]]]:
    """Run the bandit forward in chronological order. Return the
    final allocator + per-source (n, wins) tally."""
    alloc = ThompsonAllocator()
    counts: Dict[str, Tuple[int, int]] = {}
    for t in trades:
        won = t.pnl > 0
        alloc.update(t.source_key, won=won, now_ts=t.closed_at_ts or None)
        n, w = counts.get(t.source_key, (0, 0))
        counts[t.source_key] = (n + 1, w + (1 if won else 0))
    return alloc, counts


def build_report(
    trades: List[TradeRow],
    static_weights: Dict[str, float],
) -> List[SourceReport]:
    alloc, counts = replay(trades)
    out: List[SourceReport] = []
    for src, (n, wins) in counts.items():
        arm = alloc._arms.get(src)  # internal — for read-only report
        if arm is None:
            continue
        thompson_mean = arm.posterior_mean()
        thompson_wilson = arm.wilson_lower_95()
        static = static_weights.get(src, 0.5)  # treat unknown as neutral
        # Compare bandit's mean to the static dynamic_weight scaled into
        # the [0, 1] win-prob space. Static weight is already in [0, 1].
        out.append(SourceReport(
            source_key=src,
            n_trades=n,
            wins=wins,
            win_rate=wins / max(n, 1),
            thompson_mean=thompson_mean,
            thompson_wilson=thompson_wilson,
            static_weight=static,
            divergence=abs(thompson_mean - static),
        ))
    out.sort(key=lambda r: -r.divergence)
    return out


# ── Render ──────────────────────────────────────────────────────────


def render(reports: List[SourceReport]) -> str:
    out: List[str] = []
    n_sources = len(reports)
    n_significant = sum(1 for r in reports if r.divergence > 0.15)
    out.append(f"# A2 — Thompson vs. static AgentScorer divergence ({n_sources} sources)\n")
    out.append(f"Sources with |Δ| > 0.15: **{n_significant}** "
               f"(meaningful re-allocation if bandit were live).\n")
    out.append("| source | n | wins | win_rate | thompson μ | wilson_low95 | static | |Δ| |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in reports[:50]:  # cap output
        out.append(
            f"| `{r.source_key}` | {r.n_trades} | {r.wins} | "
            f"{r.win_rate:.2f} | {r.thompson_mean:.3f} | "
            f"{r.thompson_wilson:.3f} | {r.static_weight:.3f} | "
            f"{r.divergence:.3f} |"
        )
    if len(reports) > 50:
        out.append(f"\n...{len(reports) - 50} more rows truncated; see JSON dump.")
    out.append("")
    return "\n".join(out)


# ── Entrypoint ──────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    trades = load_trades(args.days)
    static = load_static_weights()
    reports = build_report(trades, static)

    print(render(reports))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = Path(args.out_dir) / f"a2_divergence_{int(time())}.json"
    out_path.write_text(json.dumps([r.__dict__ for r in reports], indent=2))
    print(f"\nFull report dumped to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
