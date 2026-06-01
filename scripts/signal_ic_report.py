#!/usr/bin/env python
"""Per-source Information Coefficient report -- which signal sources actually
predict. Reads calibration_records (predicted_confidence vs realized pnl) and
ranks every source by IC so dead/anti-predictive sources can be pruned.

    python scripts/signal_ic_report.py [--db data/bot.db] [--min-n 10]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analysis.signal_ic import load_records, compute_source_ic  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=None, help="sqlite DB (default: config.DB_PATH)")
    p.add_argument("--min-n", type=int, default=10)
    args = p.parse_args(argv)

    db = args.db
    if db is None:
        import config
        db = config.DB_PATH

    rows = load_records(db)
    ic = compute_source_ic(rows, min_n=args.min_n)
    print(f"SIGNAL IC REPORT  db={db}  sources={len(ic)}  records={len(rows)}  min_n={args.min_n}")
    print(f"{'source':<40}{'n':>6}{'IC':>9}{'mean_ret':>11}  verdict")
    for src, d in sorted(ic.items(), key=lambda kv: (kv[1]['ic'] is None, -(kv[1]['ic'] or 0.0))):
        icv = f"{d['ic']:+.3f}" if d['ic'] is not None else "--"
        print(f"{src[:39]:<40}{d['n']:>6}{icv:>9}{d['mean_return']:>11.3f}  {d['verdict']}")

    print("\nverdict counts:", dict(Counter(d['verdict'] for d in ic.values())))
    cut = [s for s, d in ic.items() if d['verdict'] in ('noise', 'negative')]
    print(f"cut candidates ({len(cut)}):", cut[:20])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
