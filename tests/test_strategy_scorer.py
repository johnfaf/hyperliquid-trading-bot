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
    monkeypatch.setattr(config, "STRATEGY_RECOVERY_TARGET_ACTIVE_VALID", 2)
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
    monkeypatch.setattr(
        db,
        "get_strategy_runtime_status",
        lambda: {"total": 2, "active_valid": 2, "inactive_valid": 0, "invalid_reasons": {}},
    )
    monkeypatch.setattr(
        db,
        "quarantine_contaminated_runtime_data",
        lambda: {"invalid_strategies": []},
    )
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


def test_score_all_strategies_recovers_valid_inactive_when_active_set_empty(monkeypatch):
    scorer = StrategyScorer()
    monkeypatch.setattr(config, "MIN_ACTIVE_STRATEGIES", 1)
    monkeypatch.setattr(config, "MAX_ACTIVE_STRATEGIES", 10)
    monkeypatch.setattr(config, "MIN_STRATEGY_SCORE", 0.1)
    monkeypatch.setattr(config, "STRATEGY_RECOVERY_TARGET_ACTIVE_VALID", 1)

    recovered_strategy = {
        "id": 9,
        "name": "recovered_btc",
        "strategy_type": "momentum_long",
        "trade_count": 12,
        "win_rate": 0.58,
        "total_pnl": 250.0,
        "sharpe_ratio": 1.1,
    }
    calls = {"get_active": 0, "recover": 0}

    recovered_flag = {"done": False}

    def fake_get_active():
        calls["get_active"] += 1
        return [recovered_strategy] if recovered_flag["done"] else []

    def fake_recover(limit):
        calls["recover"] += 1
        assert limit == 1
        recovered_flag["done"] = True
        return [recovered_strategy]

    monkeypatch.setattr(db, "get_active_strategies", fake_get_active)
    monkeypatch.setattr(
        db,
        "get_strategy_runtime_status",
        lambda: {"total": 1, "active_valid": 0, "inactive_valid": 1, "invalid_reasons": {}},
    )
    monkeypatch.setattr(db, "recover_valid_inactive_strategies", fake_recover)
    monkeypatch.setattr(
        db,
        "quarantine_contaminated_runtime_data",
        lambda: {"invalid_strategies": []},
    )
    monkeypatch.setattr(db, "get_strategy_score_history", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        scorer,
        "score_strategy",
        lambda _strategy: {
            "composite": 0.5,
            "pnl_score": 0.5,
            "win_rate_score": 0.5,
            "sharpe_score": 0.5,
            "consistency_score": 0.5,
            "risk_adj_score": 0.5,
            "win_rate_pvalue": None,
            "significance_penalty": 1.0,
        },
    )

    persisted = []

    class _Cursor:
        rowcount = 1

    class _Conn:
        def execute(self, sql, params=()):
            persisted.append((" ".join(sql.split()), tuple(params or ())))
            return _Cursor()

    @contextmanager
    def _ctx(*, for_read: bool = False):
        yield _Conn()

    monkeypatch.setattr(db, "get_connection", _ctx)
    monkeypatch.setattr(db, "log_research_cycle", lambda **_kwargs: None)

    results = scorer.score_all_strategies()

    assert calls == {"get_active": 1, "recover": 1}
    assert results[0]["strategy_id"] == 9
    assert results[0]["breakdown"]["win_rate_pvalue"] is None
    assert any(sql.startswith("INSERT INTO strategy_scores") for sql, _ in persisted)


def test_improvement_report_no_strategies_has_concrete_health(monkeypatch):
    scorer = StrategyScorer()
    monkeypatch.setattr(db, "get_active_strategies", lambda: [])
    monkeypatch.setattr(
        db,
        "get_strategy_runtime_status",
        lambda: {
            "total": 4,
            "active_valid": 0,
            "inactive_valid": 2,
            "invalid_reasons": {"missing_source_wallet": 2},
        },
    )

    report = scorer.generate_improvement_report()

    assert report["health"] == "degraded_no_valid_active_strategies"
    assert report["recoverable_inactive_strategies"] == 2


def test_improvement_report_does_not_degrade_on_inactive_invalid_bloat(monkeypatch):
    scorer = StrategyScorer()
    monkeypatch.setattr(config, "MIN_ACTIVE_STRATEGIES", 5)
    monkeypatch.setattr(config, "MAX_STRATEGIES_PER_CYCLE", 15)
    monkeypatch.setattr(config, "STRATEGY_RECOVERY_TARGET_ACTIVE_VALID", 15)
    strategies = [
        {"id": idx, "current_score": 0.5, "name": f"s{idx}", "strategy_type": "momentum"}
        for idx in range(15)
    ]

    monkeypatch.setattr(db, "get_active_strategies", lambda: strategies)
    monkeypatch.setattr(
        db,
        "get_strategy_runtime_status",
        lambda: {
            "total": 1621,
            "active_valid": 15,
            "active_invalid": 0,
            "inactive_valid": 1100,
            "inactive_invalid": 506,
            "invalid_reasons": {
                "fixture_or_demo_strategy": 5,
                "missing_source_wallet": 500,
                "synthetic_placeholder_metrics": 1,
            },
        },
    )
    monkeypatch.setattr(
        scorer,
        "get_strategy_trend",
        lambda _strategy_id: {"trend": "stable", "momentum": 0, "current_score": 0.5},
    )

    report = scorer.generate_improvement_report()

    assert report["health"] == "neutral"
    assert report["data_hygiene_warnings"]["synthetic_placeholder_metrics"] == 1
