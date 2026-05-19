"""A5 retro dry-run: would the DSR + paired-SPRT gate have changed
which sources/strategies were considered "promotable" over the last
N days?

Loads closed paper_trades, groups them by source_key (or strategy_id
if the source is "unknown"), constructs a returns series per source,
and runs the Deflated Sharpe Ratio against a configurable champion
baseline (default: the equal-weight portfolio across all sources).
Pairs each candidate against the champion via SPRT.

Output:

* Per-candidate row: n_trades, raw_sharpe, deflated_sharpe,
  dsr_significant_95, sprt_decision.
* Summary row counting how many would PASS (DSR sig AND
  SPRT ACCEPT) vs the (typically larger) set the production
  promotion code currently lets through.

Usage::

    python scripts/a5_promotion_dry_run.py [--days 30] [--mde 0.001]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import database as db
from src.learning.promotion_stats import (
    deflated_sharpe,
    sharpe_ratio,
    sprt_pair,
)


@dataclass
class Returns:
    source: str
    series: List[float]


@dataclass
class CandidateReport:
    source: str
    n_trades: int
    sharpe: float
    deflated_sharpe: float
    dsr_p_value: float
    dsr_significant_95: bool
    sprt_decision: str
    sprt_llr: float
    would_promote: bool


def load_returns(days: int) -> List[Returns]:
    """Build per-source per-period returns from paper_trades.

    "Return" here is the trade's PnL divided by the trade's entry
    notional (size * entry_price * leverage^-1 -- i.e. the margin),
    so different size buckets are commensurate.
    """
    sql = """
        SELECT COALESCE(
                   metadata->>'source_key',
                   CASE
                     WHEN metadata->>'source_trader' IS NOT NULL
                       THEN 'copy_trade:' || (metadata->>'source_trader')
                     WHEN metadata->>'strategy_type' IS NOT NULL
                       THEN 'strategy:' || (metadata->>'strategy_type')
                     ELSE 'unknown'
                   END
               ) AS source_key,
               COALESCE(pnl, 0)             AS pnl,
               COALESCE(entry_price, 0)     AS entry_price,
               COALESCE(size, 0)            AS size,
               COALESCE(leverage, 1)        AS leverage
          FROM paper_trades
         WHERE status = 'closed'
           AND closed_at >= NOW() - INTERVAL %s
         ORDER BY closed_at ASC
    """
    by_source: Dict[str, List[float]] = {}
    with db.get_connection(for_read=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (f"{days} days",))
            for src, pnl, entry, size, lev in cur.fetchall():
                margin = float(entry or 0) * float(size or 0) / max(float(lev or 1), 1)
                if margin > 0:
                    ret = float(pnl or 0) / margin
                else:
                    ret = 0.0
                by_source.setdefault(str(src), []).append(ret)
    return [Returns(source=s, series=series) for s, series in by_source.items()]


def equal_weight_champion(per_source: List[Returns]) -> List[float]:
    """Build a "champion" return series by averaging across sources at
    each chronological index. Used as the SPRT baseline.
    """
    if not per_source:
        return []
    max_n = max(len(r.series) for r in per_source)
    out: List[float] = []
    for i in range(max_n):
        slice_returns = [r.series[i] for r in per_source if i < len(r.series)]
        if slice_returns:
            out.append(sum(slice_returns) / len(slice_returns))
    return out


def build_report(
    per_source: List[Returns],
    *,
    num_trials: int,
    mde: float,
    min_n: int = 8,
) -> List[CandidateReport]:
    champion = equal_weight_champion(per_source)
    reports: List[CandidateReport] = []
    for cand in per_source:
        n = len(cand.series)
        if n < min_n:
            continue
        raw_sr = sharpe_ratio(cand.series)
        dsr = deflated_sharpe(cand.series, num_trials=num_trials)
        # SPRT: pair the candidate against the champion at indices both have
        sprt = sprt_pair(
            cand.series,
            champion[:n],
            alpha=0.05, beta=0.05, mde=mde,
        )
        would_promote = bool(dsr.significant_at_95 and sprt.decision == "ACCEPT")
        reports.append(CandidateReport(
            source=cand.source,
            n_trades=n,
            sharpe=raw_sr,
            deflated_sharpe=dsr.deflated_sharpe,
            dsr_p_value=dsr.p_value,
            dsr_significant_95=dsr.significant_at_95,
            sprt_decision=sprt.decision,
            sprt_llr=sprt.log_likelihood_ratio,
            would_promote=would_promote,
        ))
    reports.sort(key=lambda r: -r.deflated_sharpe)
    return reports


def render(reports: List[CandidateReport]) -> str:
    out: List[str] = []
    n_promote = sum(1 for r in reports if r.would_promote)
    out.append(f"# A5 — retro promotion-gate dry-run "
               f"({len(reports)} eligible candidates, "
               f"{n_promote} would PASS DSR + SPRT)\n")
    out.append("| source | n | sharpe | DSR | DSR_sig | SPRT | promote? |")
    out.append("|---|---:|---:|---:|:---:|:---:|:---:|")
    for r in reports[:40]:
        out.append(
            f"| `{r.source}` | {r.n_trades} | {r.sharpe:.3f} | "
            f"{r.deflated_sharpe:.3f} | "
            f"{'YES' if r.dsr_significant_95 else 'no'} | "
            f"{r.sprt_decision} | "
            f"{'**PROMOTE**' if r.would_promote else '-'} |"
        )
    if len(reports) > 40:
        out.append(f"\n...{len(reports) - 40} more rows truncated; see JSON dump.")
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--mde", type=float, default=0.001,
                        help="Minimum detectable per-trade edge for SPRT.")
    parser.add_argument("--min-n", type=int, default=8,
                        help="Skip candidates with fewer than this many trades.")
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    per_source = load_returns(args.days)
    # num_trials = total number of sources we ever evaluated in window.
    # This is the canonical Lopez de Prado interpretation: it's the
    # multi-testing exposure of the candidate-pool we picked from.
    num_trials = len(per_source)

    reports = build_report(
        per_source, num_trials=num_trials, mde=args.mde, min_n=args.min_n,
    )
    print(render(reports))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = Path(args.out_dir) / f"a5_dry_run_{int(time())}.json"
    out_path.write_text(json.dumps([r.__dict__ for r in reports], indent=2))
    print(f"\nFull report dumped to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
