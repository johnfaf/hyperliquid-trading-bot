"""Phase 5: agent_scorer accounting invariants.

Production audit found 12 rows in agent_scores breaking basic
math: strategy:unknown had ``total_signals=0`` and
``correct_signals=103``; momentum_short had ``accuracy=100%`` even
though ``correct_signals=6, total_signals=7`` (math: 86%).  Root
cause: ``record_outcome`` was sometimes called for source_keys
whose ``record_signal`` had never run (live_orphan, mis-tagged
copy_trade flows, untagged strategies), creating a fresh score and
incrementing ``correct_signals`` without any history entry --
which then left ``_recalculate`` reading an empty completed-trades
set.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def scorer(monkeypatch):
    """A clean AgentScorer with the DB save path stubbed out."""
    from src.signals.agent_scoring import AgentScorer

    s = AgentScorer()

    # Strip the DB upsert so tests stay self-contained.
    monkeypatch.setattr(s, "_save_score", lambda source_key: None)
    return s


# ── Invariants ──────────────────────────────────────────────────


def test_record_outcome_without_prior_signal_keeps_n_ge_corr(scorer):
    """Prod bug repro: record_outcome on a fresh source_key would
    previously yield total_signals=0, correct_signals=1.  After the
    fix total_signals must be >= correct_signals."""
    src = "strategy:unknown"
    scorer.record_outcome(src, "sig-1", pnl=1.5)
    s = scorer.scores[src]
    assert s.correct_signals == 1
    assert s.total_signals >= s.correct_signals, (
        f"Invariant broken: corr={s.correct_signals} > total={s.total_signals}"
    )


def test_corr_gt_total_pattern_is_unreachable_after_fix(scorer):
    """Hammer 5 outcomes onto a fresh source -- the prod pattern
    (n=0, corr=103, strategy:unknown) should be impossible."""
    src = "live_orphan"
    for i in range(5):
        scorer.record_outcome(src, f"sig-{i}", pnl=1.0 if i % 2 else -1.0)
    s = scorer.scores[src]
    assert s.total_signals >= s.correct_signals
    assert s.total_signals >= 5  # at least one per synthesised entry


def test_accuracy_matches_correct_over_total(scorer):
    """Prod bug repro: momentum_short had acc=100% despite 6/7.
    After each outcome, accuracy must match the running ratio."""
    src = "strategy:momentum_short"
    # 6 wins, 1 loss
    for i in range(6):
        scorer.record_signal(src, {"coin": "BTC", "side": "short", "confidence": 0.5})
        scorer.record_outcome(src, _signal_id_for(scorer, src, i + 1), pnl=1.0)
    scorer.record_signal(src, {"coin": "BTC", "side": "short", "confidence": 0.5})
    scorer.record_outcome(src, _signal_id_for(scorer, src, 7), pnl=-1.0)

    s = scorer.scores[src]
    assert s.total_signals == 7
    assert s.correct_signals == 6
    # Accuracy must equal 6/7 ~= 0.857, not stuck at 100% or 0%.
    assert abs(s.accuracy - (6 / 7)) < 0.01, (
        f"accuracy={s.accuracy:.4f}, expected ~0.857"
    )


def test_record_outcome_matching_signal_id_still_works(scorer):
    """Back-compat: when signal_id matches a history entry, the old
    path is taken (no synthetic entry needed)."""
    src = "strategy:momentum_long"
    signal_id = scorer.record_signal(src, {"coin": "ETH", "side": "long", "confidence": 0.6})
    scorer.record_outcome(src, signal_id, pnl=2.0)
    s = scorer.scores[src]
    history = scorer._trade_history[src]
    assert len(history) == 1
    assert history[0]["pnl"] == 2.0
    assert history[0]["correct"] is True
    assert "synthetic_signal" not in history[0]
    assert s.total_signals == 1 and s.correct_signals == 1


def test_synthetic_entry_carries_close_metadata(scorer):
    """When we synthesise an entry from close_metadata, coin/side
    are preserved so per-side calibration paths still work."""
    src = "live_orphan"
    scorer.record_outcome(
        src, "missing-signal-id", pnl=0.5,
        close_metadata={"coin": "BTC", "side": "long", "confidence": 0.7},
    )
    h = scorer._trade_history[src][0]
    assert h["coin"] == "BTC"
    assert h["side"] == "long"
    assert h["confidence"] == 0.7
    assert h["synthetic_signal"] is True


# ── Helpers ─────────────────────────────────────────────────────


def _signal_id_for(scorer, src, n):
    """Return the n-th signal_id record_signal wrote for ``src``."""
    history = scorer._trade_history[src]
    return history[n - 1]["signal_id"]
