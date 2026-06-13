"""Forward / recency-weighted edge for survivorship-resistant trader ranking
(signal #8, PR-B).

Ranking copy-source wallets by all-time PnL/win-rate rewards luck and dead
regimes -- a wallet that was great six months ago keeps its slot. This weights
RECENT outcomes more (exponential decay by age) and shrinks small samples toward
0.5, so a wallet has to KEEP winning to stay promoted. It's the same shrinkage
idea as the calibration ladder, applied across time instead of across keys.

Pure + observe-first: returns (edge, effective_n); the copy ranker consumes it
behind a flag. No new deps.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple


def recency_weighted_edge(
    outcomes: Sequence[Tuple[float, float]], *,
    half_life_days: float = 14.0, shrinkage: float = 10.0, prior: float = 0.5,
) -> Tuple[float, float]:
    """Recency-weighted, shrunk win-rate from ``(age_days, win)`` outcomes
    (``win`` in {0,1}).

    Each outcome is weighted ``0.5 ** (age_days / half_life_days)`` so recent
    results dominate; the weighted rate is then shrunk toward ``prior`` by
    ``shrinkage`` (beta-binomial). Returns ``(edge, effective_n)`` where
    ``effective_n`` is the summed decay weight (how much *recent* evidence backs
    the estimate). Empty -> ``(prior, 0.0)``.
    """
    hl = max(1e-9, float(half_life_days))
    sum_w = 0.0
    sum_win = 0.0
    for age, win in outcomes or ():
        try:
            a = max(0.0, float(age))
            wv = float(win)
        except (TypeError, ValueError):
            continue
        w = math.pow(0.5, a / hl)
        sum_w += w
        sum_win += w * wv
    if sum_w <= 0:
        return float(prior), 0.0
    edge = (sum_win + shrinkage * prior) / (sum_w + shrinkage)
    return float(edge), float(sum_w)


def wallet_recent_outcomes(db_path: str, wallet_address: str, now_ms: float,
                           *, lookback_days: float = 60.0):
    """``[(age_days, win)]`` from a wallet's recent CLOSED fills (closed_pnl != 0)
    in ``wallet_fills``. win = 1.0 if closed_pnl > 0 else 0.0. Read-only;
    returns [] on any error so the caller can fall back to the all-time edge."""
    cutoff = float(now_ms) - float(lookback_days) * 86_400_000.0
    out = []
    try:
        import sqlite3
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cur = conn.execute(
                "SELECT time_ms, closed_pnl FROM wallet_fills "
                "WHERE LOWER(wallet_address) = LOWER(?) AND time_ms >= ? "
                "AND closed_pnl IS NOT NULL AND closed_pnl != 0",
                (str(wallet_address), cutoff),
            )
            for t_ms, pnl in cur.fetchall():
                try:
                    age = max(0.0, (float(now_ms) - float(t_ms)) / 86_400_000.0)
                    out.append((age, 1.0 if float(pnl) > 0 else 0.0))
                except (TypeError, ValueError):
                    continue
        finally:
            conn.close()
    except Exception:
        return []
    return out


def wallet_subbook_outcomes(db_path: str, wallet_address: str, coin: str,
                            side: str, now_ms: float, *,
                            lookback_days: float = 90.0):
    """``[(age_days, win)]`` for one wallet's (coin, side) SUB-BOOK from recent
    closed fills -- a wallet may be great at ETH longs and bad at alt shorts, so
    copy should be gated per sub-book, not per wallet. Read-only; [] on error."""
    cutoff = float(now_ms) - float(lookback_days) * 86_400_000.0
    side_n = str(side or "").strip().lower()
    if side_n in ("buy",):
        side_n = "long"
    elif side_n in ("sell",):
        side_n = "short"
    out = []
    try:
        import sqlite3
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cur = conn.execute(
                "SELECT time_ms, closed_pnl FROM wallet_fills "
                "WHERE LOWER(wallet_address) = LOWER(?) AND UPPER(coin) = UPPER(?) "
                "AND LOWER(side) = ? AND time_ms >= ? "
                "AND closed_pnl IS NOT NULL AND closed_pnl != 0",
                (str(wallet_address), str(coin), side_n, cutoff),
            )
            for t_ms, pnl in cur.fetchall():
                try:
                    age = max(0.0, (float(now_ms) - float(t_ms)) / 86_400_000.0)
                    out.append((age, 1.0 if float(pnl) > 0 else 0.0))
                except (TypeError, ValueError):
                    continue
        finally:
            conn.close()
    except Exception:
        return []
    return out


def subbook_edge_table(db_path: str, now_ms: float, *, lookback_days: float = 90.0,
                       half_life_days: float = 14.0, shrinkage: float = 10.0,
                       prior: float = 0.5, min_samples: int = 1, limit: int = 100):
    """Every (wallet, coin, side) SUB-BOOK with its recency-weighted edge from
    recent closed ``wallet_fills``, sorted by edge desc -- the observability view
    behind the per-sub-book copy gate (signal #5).

    Returns ``[{wallet, coin, side, n, edge, eff_n}]``; ``n`` is the raw outcome
    count and ``eff_n`` the summed decay weight (recent evidence). Read-only;
    ``[]`` on any error so the dashboard degrades gracefully.
    """
    cutoff = float(now_ms) - float(lookback_days) * 86_400_000.0
    groups: dict = {}
    try:
        import sqlite3
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cur = conn.execute(
                "SELECT LOWER(wallet_address), UPPER(coin), LOWER(side), time_ms, "
                "closed_pnl FROM wallet_fills "
                "WHERE time_ms >= ? AND closed_pnl IS NOT NULL AND closed_pnl != 0",
                (cutoff,),
            )
            for wallet, coin, side, t_ms, pnl in cur.fetchall():
                try:
                    age = max(0.0, (float(now_ms) - float(t_ms)) / 86_400_000.0)
                    win = 1.0 if float(pnl) > 0 else 0.0
                except (TypeError, ValueError):
                    continue
                groups.setdefault((wallet, coin, side), []).append((age, win))
        finally:
            conn.close()
    except Exception:
        return []

    out = []
    for (wallet, coin, side), outcomes in groups.items():
        if len(outcomes) < int(min_samples):
            continue
        edge, eff_n = recency_weighted_edge(
            outcomes, half_life_days=half_life_days, shrinkage=shrinkage, prior=prior)
        out.append({
            "wallet": wallet, "coin": coin, "side": side,
            "n": len(outcomes), "edge": round(float(edge), 4),
            "eff_n": round(float(eff_n), 3),
        })
    out.sort(key=lambda r: (-r["edge"], -r["n"]))
    return out[: max(1, int(limit or 100))]
