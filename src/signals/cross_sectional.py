"""Cross-sectional (market-neutral) momentum signal.

Per-coin directional bets leave the book implicitly long-beta -- all crypto
longs are correlated, so N longs are ~one big beta bet. This ranks the coin
universe by a cross-sectional factor (trailing momentum) and goes LONG the
top-K / SHORT the bottom-K, so the book is ~beta-neutral and isolates *relative*
alpha. Cross-sectional momentum is a documented crypto edge.

Flag-gated (CROSS_SECTIONAL_ENABLED, default OFF). v1: momentum factor,
equal-weight, K longs + K shorts. Pure functions so the ranking/selection is
unit-testable without the data plane; the cycle supplies {coin: recent_closes}.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


def momentum_score(closes: Sequence[float], lookback: int) -> Optional[float]:
    """Trailing return over ``lookback`` bars: close[-1] / close[-1-lookback] - 1.
    None if there isn't enough history or the base price is non-positive."""
    if lookback <= 0 or not closes or len(closes) <= lookback:
        return None
    try:
        base = float(closes[-1 - lookback])
        last = float(closes[-1])
    except (TypeError, ValueError, IndexError):
        return None
    if base <= 0 or last <= 0:
        return None
    return last / base - 1.0


def rank_and_select(scores: Dict[str, float], top_k: int) -> Tuple[List[str], List[str]]:
    """Return ``(longs, shorts)``: the ``top_k`` highest-momentum coins long and
    the ``top_k`` lowest short. Market-neutral by construction (equal counts,
    disjoint sets); K is shrunk so longs and shorts never overlap when the
    universe is smaller than ``2*top_k``."""
    ranked = [c for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
    k = max(0, min(int(top_k), len(ranked) // 2))
    if k == 0:
        return [], []
    return ranked[:k], ranked[-k:]


def _signal(coin: str, side: str, confidence: float, score: float) -> Dict:
    """Decision-pipeline synthetic-strategy shape (matches options-flow/polymarket
    injection in trading_cycle Phase 4)."""
    return {
        "id": None,
        "name": f"xsect_{coin}_{side}",
        "strategy_type": "cross_sectional",
        "trader_address": "cross_sectional",
        "current_score": confidence,
        "confidence": confidence,
        "direction": side,
        "side": side,
        "source": "cross_sectional",
        "parameters": {"coins": [coin], "direction": side},
        "metrics": {},
        "metadata": {"momentum": round(float(score), 6)},
    }


def generate_cross_sectional_signals(
    coin_closes: Dict[str, Sequence[float]], *,
    top_k: int = 3, lookback: int = 24, confidence: float = 0.5,
) -> List[Dict]:
    """Rank coins by trailing momentum and emit a market-neutral basket: long the
    top-K, short the bottom-K. Returns [] if fewer than 2 coins have a score."""
    scores: Dict[str, float] = {}
    for coin, closes in (coin_closes or {}).items():
        s = momentum_score(closes, lookback)
        if s is not None:
            scores[coin] = s
    if len(scores) < 2:
        return []
    longs, shorts = rank_and_select(scores, top_k)
    out: List[Dict] = []
    out += [_signal(c, "long", confidence, scores[c]) for c in longs]
    out += [_signal(c, "short", confidence, scores[c]) for c in shorts]
    return out
