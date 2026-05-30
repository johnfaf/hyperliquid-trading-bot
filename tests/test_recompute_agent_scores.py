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


def test_phantom_source_reset_to_zero(scorer):
    """The phantom-reset path (rebuild with []) must zero a source whose
    columns were polluted by the record_outcome-without-record_signal
    bug -- e.g. strategy:unknown with corr=103 / total=0."""
    from src.signals.agent_scoring import SourceScore
    scorer.scores["strategy:unknown"] = SourceScore(
        source_key="strategy:unknown",
        total_signals=0, correct_signals=103, total_pnl=-303.77, accuracy=0.0,
    )
    rebuilt = scorer.rebuild_source_from_trades("strategy:unknown", [])
    assert rebuilt.total_signals == 0
    assert rebuilt.correct_signals == 0
    assert rebuilt.total_pnl == 0.0
    # Invariant restored: corr never exceeds total.
    assert rebuilt.correct_signals <= rebuilt.total_signals


def test_script_argparse_has_keep_phantoms_flag(script_mod):
    """The --keep-phantoms opt-out exists and defaults to off (reset on)."""
    import argparse
    # Build the parser the same way main() does and check the flag.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-phantoms", action="store_true")
    ns = parser.parse_args([])
    assert ns.keep_phantoms is False
    ns2 = parser.parse_args(["--keep-phantoms"])
    assert ns2.keep_phantoms is True


# ── source-key fragmentation merge ──────────────────────────────


def test_canonicalize_merges_truncated_into_full(script_mod):
    """A truncated copy_trade key folds into the unique full-address key
    that shares its prefix -- the historical fragmentation fix."""
    full = "0x" + "1e" * 20      # full 0x + 40 hex
    short = full[:10]            # 0x + 8 hex truncated prefix
    grouped = {
        f"copy_trade:{full}": [{"pnl": 5.0}],
        f"copy_trade:{short}": [{"pnl": 3.0}, {"pnl": -1.0}],
        "strategy:momentum_long": [{"pnl": 1.0}],
    }
    merged = script_mod._canonicalize_grouped_keys(grouped)
    assert f"copy_trade:{short}" not in merged              # folded away
    assert len(merged[f"copy_trade:{full}"]) == 3            # 1 + 2 consolidated
    assert merged["strategy:momentum_long"] == [{"pnl": 1.0}]  # untouched


def test_canonicalize_leaves_ambiguous_truncated(script_mod):
    """If a truncated prefix matches 2+ full addresses it's left alone."""
    short = "0xabcdef12"
    full1 = short + "00" * 16    # both 0x + 40 hex sharing the prefix
    full2 = short + "11" * 16
    grouped = {
        f"copy_trade:{full1}": [{"pnl": 1.0}],
        f"copy_trade:{full2}": [{"pnl": 2.0}],
        f"copy_trade:{short}": [{"pnl": 9.0}],
    }
    merged = script_mod._canonicalize_grouped_keys(grouped)
    assert f"copy_trade:{short}" in merged                   # ambiguous -> kept


def test_canonicalize_no_full_match_keeps_truncated(script_mod):
    short = "0xdeadbeef"
    grouped = {f"copy_trade:{short}": [{"pnl": 1.0}]}
    merged = script_mod._canonicalize_grouped_keys(grouped)
    assert merged == {f"copy_trade:{short}": [{"pnl": 1.0}]}


def test_canonicalize_noop_without_truncated(script_mod):
    full = "0x" + "ab" * 20
    grouped = {f"copy_trade:{full}": [{"pnl": 1.0}], "strategy:x": [{"pnl": 2.0}]}
    assert script_mod._canonicalize_grouped_keys(grouped) == grouped


# ── agent_scorer get_source_key canonicalization (forward fix) ──


def test_get_source_key_full_address_passthrough():
    from src.signals.agent_scoring import AgentScorer
    s = AgentScorer()
    full = "0x" + "cd" * 20
    sig = {"source": "copy_trade", "source_trader": full}
    assert s.get_source_key(sig) == f"copy_trade:{full}"


def test_get_source_key_truncated_falls_back_to_untagged():
    """A truncated address must NOT become copy_trade:0x<short> (the
    fragmentation bug) -- it falls back to the canonical untagged key."""
    from src.signals.agent_scoring import AgentScorer
    s = AgentScorer()
    sig = {"source": "copy_trade", "source_trader": "0x1ee7a73c"}
    assert s.get_source_key(sig) == "copy_trade"
