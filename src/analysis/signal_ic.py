"""Information Coefficient (IC) measurement per signal source.

The bot runs ~15 signal sources but never measured which actually *predict*.
IC -- the rank correlation between a source's pre-trade signal (predicted
confidence) and the realized outcome -- is the cleanest "does this source carry
alpha?" metric. This module computes it per source from the calibration_records
the bot already writes (source_key, predicted_confidence, pnl).

Observe-only: this reports IC; pruning/gating on it is downstream. No new deps
(Spearman implemented directly).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple


def _avg_ranks(values: Sequence[float]) -> List[float]:
    """1-based average ranks (ties share the mean rank)."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_ic(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation in [-1, 1], or None if undefined (n<3 or one
    side has no variance -- e.g. a source whose confidence is pinned flat)."""
    n = len(xs)
    if n < 3 or len(ys) != n:
        return None
    rx, ry = _avg_ranks(list(xs)), _avg_ranks(list(ys))
    mx, my = sum(rx) / n, sum(ry) / n
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return sxy / math.sqrt(sxx * syy)


def _verdict(ic: Optional[float], n: int, min_n: int, band: float) -> str:
    if n < min_n:
        return "insufficient"
    if ic is None:
        return "flat"          # no signal variance to correlate
    if ic <= -band:
        return "negative"      # actively anti-predictive -> candidate to cut/flip
    if ic < band:
        return "noise"         # ~no edge -> candidate to cut
    return "predictive"        # carries alpha -> keep/lean on


def compute_source_ic(rows: List[Tuple[str, float, float]],
                      min_n: int = 10, band: float = 0.05) -> Dict[str, Dict]:
    """Per-source IC from (source_key, predicted_score, realized_return) rows.

    Returns {source: {n, ic, mean_return, verdict}}. ``band`` is the |IC|
    deadzone that separates "noise" from a real (positive or negative) signal.
    """
    by: Dict[str, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
    for src, pred, real in rows:
        if not src:
            continue
        try:
            p, r = float(pred), float(real)
        except (TypeError, ValueError):
            continue
        by[src][0].append(p)
        by[src][1].append(r)

    out: Dict[str, Dict] = {}
    for src, (preds, reals) in by.items():
        n = len(preds)
        ic = spearman_ic(preds, reals) if n >= min_n else None
        out[src] = {
            "n": n,
            "ic": round(ic, 4) if ic is not None else None,
            "mean_return": round(sum(reals) / n, 4) if n else 0.0,
            "verdict": _verdict(ic, n, min_n, band),
        }
    return out


def ic_weight(ic, n, *, min_n: float = 20.0, gain: float = 2.5,
              min_weight: float = 0.25, max_weight: float = 1.5) -> float:
    """Map a source's measured IC + sample size to a confidence/size WEIGHT
    (signal #6). Observe-first: insufficient evidence (``n < min_n`` or no IC)
    -> neutral 1.0; positive IC scales up toward ``max_weight``, negative IC
    down toward ``min_weight``. ``weight = clamp(1 + ic*gain, min, max)``."""
    try:
        if ic is None or n is None or float(n) < float(min_n):
            return 1.0
        ic_c = max(-1.0, min(float(ic), 1.0))
    except (TypeError, ValueError):
        return 1.0
    return float(max(min_weight, min(1.0 + ic_c * gain, max_weight)))


def source_ic_weight(source_key: str, db_path: str, *, min_n: float = 20.0,
                     gain: float = 2.5, min_weight: float = 0.25,
                     max_weight: float = 1.5) -> float:
    """Confidence weight for one source from its measured IC over
    calibration_records. Observe-first: neutral 1.0 when the source has < min_n
    outcomes. Heavy callers should cache -- this reads the DB."""
    try:
        rows = [r for r in load_records(db_path) if r and r[0] == source_key]
        ic_map = compute_source_ic(rows, min_n=int(min_n))
        d = ic_map.get(source_key)
        if not d:
            return 1.0
        return ic_weight(d.get("ic"), d.get("n"), min_n=min_n, gain=gain,
                         min_weight=min_weight, max_weight=max_weight)
    except Exception:
        return 1.0


def load_records(db_path: str) -> List[Tuple[str, float, float]]:
    """Load (source_key, predicted_confidence, pnl) from calibration_records."""
    import sqlite3
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.execute(
            "SELECT source_key, predicted_confidence, pnl FROM calibration_records "
            "WHERE pnl IS NOT NULL AND predicted_confidence IS NOT NULL"
        )
        return [(r[0], r[1], r[2]) for r in cur.fetchall()]
    except Exception:
        return []
    finally:
        conn.close()
