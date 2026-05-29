"""Tests for AgentScorer.rebuild_source_from_trades + recompute script."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def scorer(monkeypatch):
    from src.signals.agent_scoring import AgentScorer
    s = AgentScorer()
    monkeypatch.setattr(s, "_save_score", lambda source_key: None)
    return s


# ── rebuild_source_from_trades ──────────────────────────────────


def test_rebuild_sets_consistent_totals(scorer):
    trades = [
        {"pnl": 5.0, "coin": "BTC", "side": "long"},
        {"pnl": -2.0, "coin": "ETH", "side": "short"},
        {"pnl": 3.0, "coin": "SOL", "side": "long"},
    ]
    score = scorer.rebuild_source_from_trades("strategy:momentum_long", trades)
    assert score.total_signals == 3
    assert score.correct_signals == 2          # two positive pnl
    assert abs(score.total_pnl - 6.0) < 1e-9   # 5 - 2 + 3
    # The fundamental invariant that the prod data violated:
    assert score.total_signals >= score.correct_signals


def test_rebuild_overwrites_polluted_state(scorer):
    """Repro of the prod bug: a source whose columns claim wild values
    gets fully replaced by the authoritative trade list."""
    # Simulate polluted prior state (corr > total, inflated pnl).
    from src.signals.agent_scoring import SourceScore
    scorer.scores["strategy:momentum_short"] = SourceScore(
        source_key="strategy:momentum_short",
        total_signals=7, correct_signals=6, total_pnl=468.90, accuracy=1.0,
    )
    scorer._trade_history["strategy:momentum_short"] = [{"signal_id": "x", "pnl": 10.48, "correct": True}]

    # Authoritative reality: 3 trades, net -1.0, 1 win.
    real = [
        {"pnl": 2.0, "coin": "BTC", "side": "short"},
        {"pnl": -1.5, "coin": "BTC", "side": "short"},
        {"pnl": -1.5, "coin": "ETH", "side": "short"},
    ]
    score = scorer.rebuild_source_from_trades("strategy:momentum_short", real)
    assert score.total_signals == 3
    assert score.correct_signals == 1
    assert abs(score.total_pnl - (-1.0)) < 1e-9
    assert score.accuracy == pytest.approx(1 / 3, abs=0.01)
    # trade_history is rebuilt to match (not the stale 1-entry list).
    assert len(scorer._trade_history["strategy:momentum_short"]) == 3


def test_rebuild_empty_trades_is_safe(scorer):
    score = scorer.rebuild_source_from_trades("strategy:dead", [])
    assert score.total_signals == 0
    assert score.correct_signals == 0
    assert score.total_pnl == 0.0


def test_rebuild_dynamic_weight_uses_canonical_recalculate(scorer):
    """A strong winning source should land a high dynamic_weight; a
    losing one should land low -- proving _recalculate ran."""
    winners = [{"pnl": 5.0, "return_pct": 0.05} for _ in range(20)]
    losers = [{"pnl": -5.0, "return_pct": -0.05} for _ in range(20)]
    w = scorer.rebuild_source_from_trades("strategy:winner", winners)
    el = scorer.rebuild_source_from_trades("strategy:loser", losers)
    assert w.dynamic_weight > el.dynamic_weight
    assert w.accuracy == 1.0
    assert el.accuracy == 0.0


def test_rebuild_caps_history_at_200(scorer):
    trades = [{"pnl": 1.0} for _ in range(500)]
    scorer.rebuild_source_from_trades("strategy:busy", trades)
    assert len(scorer._trade_history["strategy:busy"]) == 200
    # But total_signals reflects the full input.
    assert scorer.scores["strategy:busy"].total_signals == 500


# ── recompute script helpers ───────────────────────────────────


@pytest.fixture
def script_mod():
    here = Path(__file__).resolve().parent.parent
    path = here / "scripts" / "recompute_agent_scores.py"
    spec = importlib.util.spec_from_file_location("recompute_agent_scores", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_excludes_tainted_and_reconciler(script_mod):
    trades = [
        {"coin": "BTC", "side": "short", "pnl": -5.0, "closed_at": "t1",
         "metadata": {"source_key": "strategy:momentum_short", "tainted": True}},
        {"coin": "BTC", "side": "short", "pnl": -9.0, "closed_at": "t2",
         "metadata": {"source_key": "strategy:momentum_short",
                      "close_reason": "live_reconciled_closed"}},
        {"coin": "BTC", "side": "short", "pnl": 3.0, "closed_at": "t3",
         "metadata": {"source_key": "strategy:momentum_short",
                      "close_reason": "take_profit"}},
    ]
    grouped = script_mod._group_clean_trades(trades)
    # Only the clean take_profit trade survives.
    assert "strategy:momentum_short" in grouped
    assert len(grouped["strategy:momentum_short"]) == 1
    assert grouped["strategy:momentum_short"][0]["pnl"] == 3.0


def test_script_source_key_derivation(script_mod):
    # copy_trade with full address
    addr = "0x" + "cd" * 20
    meta = {"source": "copy_trade", "source_trader": addr}
    assert script_mod._source_key(meta, {}) == f"copy_trade:{addr}"
    # strategy with strategy_type
    meta2 = {"source": "strategy", "strategy_type": "momentum_long"}
    assert script_mod._source_key(meta2, {}) == "strategy:momentum_long"
    # bare fallback
    assert script_mod._source_key({}, {}) == "unknown"


def test_script_tainted_predicate(script_mod):
    assert script_mod._is_tainted({"tainted": True})
    assert script_mod._is_tainted({"close_reason": "live_reconciled_closed"})
    assert script_mod._is_tainted({"reconciliation_reason": "live_reconciled_closed"})
    assert not script_mod._is_tainted({"close_reason": "take_profit"})
    assert not script_mod._is_tainted({})
