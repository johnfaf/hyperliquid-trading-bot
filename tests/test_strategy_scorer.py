from __future__ import annotations

from contextlib import contextmanager

import config

from src.analysis.strategy_scorer import StrategyScorer
from src.data import database as db


def test_score_all_strategies_batches_persistence(monkeypatch):
    scorer = StrategyScorer()
    monkeypatch.setattr(config, "MIN_ACTIVE_STRATEGIES", 1)
    monkeypatch.setattr(config, "MAX_ACTIVE_STRATEGIES", 10)
    monkeypatch.setattr(config, "MIN_STRATEGY_SCORE", 0.5)
    strategies = [
        {
            "id": 1,
            "name": "alpha_btc",
            "strategy_type": "momentum",
            "trade_count": 20,
            "win_rate": 0.60,
            "total_pnl": 1200.0,
            "sharpe_ratio": 1.4,
        },
        {
            "id": 2,
            "name": "bravo_eth",
            "strategy_type": "reversion",
            "trade_count": 15,
            "win_rate": 0.45,
            "total_pnl": 100.0,
            "sharpe_ratio": 0.8,
        },
    ]

    monkeypatch.setattr(db, "get_active_strategies", lambda: strategies)
    monkeypatch.setattr(db, "get_strategy_score_history", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        scorer,
        "score_strategy",
        lambda strategy: {
            "composite": 0.90 if strategy["id"] == 1 else 0.20,
            "pnl_score": 0.8,
            "win_rate_score": 0.7,
            "sharpe_score": 0.6,
            "consistency_score": 0.5,
            "risk_adj_score": 0.4,
        },
    )

    connection_entries = {"count": 0}
    executed = []
    research_cycles = []

    class _Cursor:
        rowcount = 1

    class _Conn:
        def execute(self, sql, params=()):
            executed.append((" ".join(sql.split()), tuple(params or ())))
            return _Cursor()

    @contextmanager
    def _ctx(*, for_read: bool = False):
        assert for_read is False
        connection_entries["count"] += 1
        yield _Conn()

    monkeypatch.setattr(db, "get_connection", _ctx)
    monkeypatch.setattr(
        db,
        "log_research_cycle",
        lambda **kwargs: research_cycles.append(kwargs),
    )

    results = scorer.score_all_strategies()

    inserts = [sql for sql, _params in executed if sql.startswith("INSERT INTO strategy_scores")]
    updates = [sql for sql, _params in executed if sql.startswith("UPDATE strategies")]

    assert connection_entries["count"] == 1
    assert len(inserts) == 2
    assert len(updates) == 2
    assert results[0]["strategy_id"] == 1
    assert results[0]["active"] is True
    assert results[1]["active"] is False
    assert research_cycles[0]["strategies_updated"] == 2


def test_strategy_read_helpers_use_read_connections(monkeypatch):
    observed = []

    class _Cursor:
        def fetchall(self):
            return []

        def fetchone(self):
            return None

    class _Conn:
        def execute(self, _sql, _params=()):
            return _Cursor()

    @contextmanager
    def _ctx(*, for_read: bool = False):
        observed.append(for_read)
        yield _Conn()

    monkeypatch.setattr(db, "get_connection", _ctx)

    assert db.get_active_strategies() == []
    assert db.get_strategy(7) is None
    assert db.get_strategy_score_history(7) == []
    assert observed == [True, True, True]
