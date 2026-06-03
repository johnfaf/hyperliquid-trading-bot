"""Forward / recency-weighted edge for trader discovery (signal #8, PR-B)."""
from __future__ import annotations

import sqlite3

from src.learning.forward_edge import recency_weighted_edge, wallet_recent_outcomes

_DAY = 86_400_000.0


def test_empty_returns_prior():
    assert recency_weighted_edge([], prior=0.5) == (0.5, 0.0)


def test_recent_wins_push_edge_up_and_more_evidence_less_shrinkage():
    e2, n2 = recency_weighted_edge([(0.0, 1.0)] * 2, half_life_days=14, shrinkage=10)
    e20, n20 = recency_weighted_edge([(0.0, 1.0)] * 20, half_life_days=14, shrinkage=10)
    assert 0.5 < e2 < e20          # more recent wins -> further above the 0.5 prior
    assert n20 > n2


def test_recent_losses_dominate_old_wins():
    # 5 fresh losses + 5 wins from ~100 days ago -> recency must drag edge < 0.5
    outcomes = [(0.0, 0.0)] * 5 + [(100.0, 1.0)] * 5
    edge, _ = recency_weighted_edge(outcomes, half_life_days=14, shrinkage=10)
    assert edge < 0.45, edge


def test_wallet_recent_outcomes_filters(tmp_path):
    db = str(tmp_path / "wf.db")
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE wallet_fills (wallet_address TEXT, time_ms REAL, closed_pnl REAL)")
    now = 1_000_000 * _DAY  # arbitrary 'now' in ms
    rows = [
        ("0xAAA", now - 1 * _DAY, 5.0),     # recent win
        ("0xAAA", now - 2 * _DAY, -3.0),    # recent loss
        ("0xAAA", now - 200 * _DAY, 9.0),   # too old (beyond 60d lookback)
        ("0xAAA", now - 1 * _DAY, 0.0),     # zero pnl (an open, not an outcome)
        ("0xAAA", now - 1 * _DAY, None),    # null pnl
        ("0xBBB", now - 1 * _DAY, 7.0),     # different wallet
    ]
    c.executemany("INSERT INTO wallet_fills VALUES (?,?,?)", rows)
    c.commit()
    c.close()

    outs = wallet_recent_outcomes(db, "0xaaa", now, lookback_days=60)
    wins = sorted(w for _, w in outs)
    assert len(outs) == 2 and wins == [0.0, 1.0]   # only recent, non-zero, this wallet
    assert wallet_recent_outcomes(db, "0xNOPE", now) == []
