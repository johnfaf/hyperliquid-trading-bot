"""Cross-sectional market-neutral momentum signal (signal #7).

Rank the coin universe by trailing momentum, long the top-K / short the bottom-K
so the book is beta-neutral. Pure-function tests for ranking, selection, and
market-neutrality.
"""
from __future__ import annotations

from src.signals.cross_sectional import (
    generate_cross_sectional_signals,
    momentum_score,
    rank_and_select,
)


def test_momentum_score_trailing_return():
    closes = [100.0] * 24 + [110.0]   # len 25, lookback 24 -> base 100, last 110
    assert abs(momentum_score(closes, 24) - 0.10) < 1e-9


def test_momentum_score_guards():
    assert momentum_score([1, 2], 5) is None            # too little history
    assert momentum_score([0.0, 0.0, 5.0], 2) is None   # base price <= 0
    assert momentum_score([], 24) is None


def test_rank_and_select_is_market_neutral():
    scores = {"A": 0.5, "B": 0.3, "C": 0.1, "D": -0.1, "E": -0.4}
    longs, shorts = rank_and_select(scores, top_k=2)
    assert longs == ["A", "B"]
    assert shorts == ["D", "E"]
    assert set(longs).isdisjoint(shorts)
    assert len(longs) == len(shorts) == 2


def test_rank_shrinks_k_to_keep_sets_disjoint():
    # 3 coins, top_k=3 -> k = min(3, 3//2=1) = 1 long + 1 short, never overlapping
    longs, shorts = rank_and_select({"A": 1.0, "B": 0.0, "C": -1.0}, top_k=3)
    assert len(longs) == 1 and len(shorts) == 1
    assert set(longs).isdisjoint(shorts)


def test_generate_emits_neutral_basket():
    coin_closes = {c: [100.0] * 24 + [100.0 * (1 + m)]
                   for c, m in [("A", 0.2), ("B", 0.1), ("C", 0.0), ("D", -0.1), ("E", -0.2)]}
    sigs = generate_cross_sectional_signals(coin_closes, top_k=2, lookback=24)
    longs = [s for s in sigs if s["side"] == "long"]
    shorts = [s for s in sigs if s["side"] == "short"]
    assert len(longs) == len(shorts) == 2                      # market-neutral
    assert {s["parameters"]["coins"][0] for s in longs} == {"A", "B"}
    assert {s["parameters"]["coins"][0] for s in shorts} == {"D", "E"}
    assert all(s["strategy_type"] == "cross_sectional" for s in sigs)


def test_generate_empty_when_too_few_scored():
    assert generate_cross_sectional_signals({"A": [100.0] * 25}, top_k=2, lookback=24) == []
