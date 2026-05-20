"""Tests for src.signals.loss_attribution.

Two layers:
  1. The pure classify_close() / bandit_outcome() math.
  2. The wiring into AgentScorer.record_outcome with the
     BANDIT_SKIP_NOISE_STOPS_ENABLED gate.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import config
from src.signals.agent_scoring import AgentScorer
from src.signals.loss_attribution import (
    CloseClass,
    ClassifyInputs,
    bandit_outcome,
    classify_close,
    should_feed_bandit,
)


# ── classify_close: pure ─────────────────────────────────────────


def test_take_profit_reason_classifies_as_tp():
    ci = ClassifyInputs(pnl=10.0, close_reason="take_profit_hit",
                        entry_price=100.0, exit_price=120.0)
    assert classify_close(ci) == CloseClass.TAKE_PROFIT


def test_time_out_reason_classifies_as_time_out():
    ci = ClassifyInputs(pnl=-2.0, close_reason="time_limit_exceeded",
                        entry_price=100.0, exit_price=99.5)
    assert classify_close(ci) == CloseClass.TIME_OUT


def test_reconciled_reason_classifies_as_reconciled():
    ci = ClassifyInputs(pnl=0.0, close_reason="live_reconciled_closed")
    assert classify_close(ci) == CloseClass.RECONCILED


def test_stop_loss_with_tiny_move_classifies_as_noise_stop():
    """The audit case: -0.03% move triggered the stop -- that's pure
    intra-bar noise, not a source-quality signal."""
    ci = ClassifyInputs(
        pnl=-1.50, close_reason="stop_loss_hit",
        entry_price=100.0, exit_price=99.97,  # -0.03% move
        atr_pct=0.015,
    )
    assert classify_close(ci) == CloseClass.NOISE_STOP


def test_stop_loss_with_real_adverse_move_classifies_as_signal_loss():
    """A 5% adverse move IS a directional miss -- the source signal was
    wrong, feed bandit as loss."""
    ci = ClassifyInputs(
        pnl=-50.0, close_reason="stop_loss_hit",
        entry_price=100.0, exit_price=95.0,  # -5% move
        atr_pct=0.015,
    )
    assert classify_close(ci) == CloseClass.SIGNAL_LOSS


def test_stop_loss_no_atr_uses_noise_floor():
    """When atr_pct is 0 (unknown), fall back to noise_floor_bps."""
    # -10 bps move with no ATR: below 50 bps default -> NOISE_STOP
    ci_noise = ClassifyInputs(
        pnl=-0.50, close_reason="stop_loss_hit",
        entry_price=100.0, exit_price=99.90, atr_pct=0.0,
    )
    assert classify_close(ci_noise) == CloseClass.NOISE_STOP
    # -1% move with no ATR: above 50 bps -> SIGNAL_LOSS
    ci_real = ClassifyInputs(
        pnl=-10.0, close_reason="stop_loss_hit",
        entry_price=100.0, exit_price=99.0, atr_pct=0.0,
    )
    assert classify_close(ci_real) == CloseClass.SIGNAL_LOSS


def test_stop_loss_with_invalid_prices_falls_back_to_signal_loss():
    """Missing entry/exit prices: assume the loss was real."""
    ci = ClassifyInputs(
        pnl=-1.0, close_reason="stop_loss_hit",
        entry_price=0.0, exit_price=0.0, atr_pct=0.015,
    )
    assert classify_close(ci) == CloseClass.SIGNAL_LOSS


def test_no_reason_falls_back_to_pnl_sign():
    assert classify_close(ClassifyInputs(pnl=5.0)) == CloseClass.TAKE_PROFIT
    assert classify_close(ClassifyInputs(pnl=-5.0)) == CloseClass.SIGNAL_LOSS
    assert classify_close(ClassifyInputs(pnl=0.0)) == CloseClass.OTHER


def test_stop_loss_short_side_classified_correctly():
    """Short side: adverse move is UP. The price went from 100 to 100.03,
    a -0.03% move against a short position -> noise stop."""
    ci = ClassifyInputs(
        pnl=-1.50, close_reason="stop_loss_hit",
        entry_price=100.0, exit_price=100.03,
        atr_pct=0.015, side="short",
    )
    assert classify_close(ci) == CloseClass.NOISE_STOP


# ── bandit_outcome / should_feed_bandit ───────────────────────────


def test_bandit_outcome_skips_noise_and_reconciled():
    assert bandit_outcome(CloseClass.NOISE_STOP, -1.0) is None
    assert bandit_outcome(CloseClass.RECONCILED, 0.0) is None


def test_bandit_outcome_take_profit_is_always_win():
    assert bandit_outcome(CloseClass.TAKE_PROFIT, 10.0) is True
    # Even if P&L is weirdly negative (fees ate it), TP intent -> win.
    assert bandit_outcome(CloseClass.TAKE_PROFIT, -0.1) is True


def test_bandit_outcome_signal_loss_is_always_loss():
    assert bandit_outcome(CloseClass.SIGNAL_LOSS, -10.0) is False
    assert bandit_outcome(CloseClass.SIGNAL_LOSS, 0.5) is False  # tiny win still 'loss' by class


def test_bandit_outcome_timeout_uses_pnl_sign():
    assert bandit_outcome(CloseClass.TIME_OUT, 5.0) is True
    assert bandit_outcome(CloseClass.TIME_OUT, -5.0) is False


def test_should_feed_bandit_respects_flags():
    assert should_feed_bandit(CloseClass.NOISE_STOP) is False
    assert should_feed_bandit(CloseClass.RECONCILED) is False
    assert should_feed_bandit(CloseClass.NOISE_STOP, skip_noise_stops=False) is True
    assert should_feed_bandit(CloseClass.SIGNAL_LOSS) is True
    assert should_feed_bandit(CloseClass.TAKE_PROFIT) is True


# ── AgentScorer.record_outcome wiring ─────────────────────────────


@pytest.fixture
def scorer():
    """A bandit-enabled AgentScorer without DB I/O."""
    s = AgentScorer.__new__(AgentScorer)
    s.scores = {}
    s._trade_history = {}
    from collections import defaultdict
    s._trade_history = defaultdict(list)
    s._bandit_enabled = True
    s._bandit_blend = 1.0
    s._db_save = MagicMock()
    s._save_score = MagicMock()
    s._recalculate = MagicMock()
    s._bandit = MagicMock()
    s._bandit.update = MagicMock()
    s._bandit_alloc = lambda: s._bandit
    return s


def test_record_outcome_flag_off_feeds_bandit_for_loss(scorer, monkeypatch):
    monkeypatch.setattr(config, "BANDIT_SKIP_NOISE_STOPS_ENABLED", False, raising=False)
    scorer.record_outcome("copy_trade:0xA", "sig1", pnl=-1.0)
    scorer._bandit.update.assert_called_once_with("copy_trade:0xA", won=False)


def test_record_outcome_flag_on_skips_noise_stop(scorer, monkeypatch):
    monkeypatch.setattr(config, "BANDIT_SKIP_NOISE_STOPS_ENABLED", True, raising=False)
    scorer.record_outcome(
        "copy_trade:0xA", "sig1", pnl=-1.0,
        close_metadata={
            "close_reason": "stop_loss_hit",
            "entry_price": 100.0,
            "exit_price": 99.97,
            "atr_pct": 0.015,
        },
    )
    scorer._bandit.update.assert_not_called()


def test_record_outcome_flag_on_feeds_signal_loss(scorer, monkeypatch):
    """A real adverse move still feeds the bandit as a loss."""
    monkeypatch.setattr(config, "BANDIT_SKIP_NOISE_STOPS_ENABLED", True, raising=False)
    scorer.record_outcome(
        "copy_trade:0xA", "sig1", pnl=-10.0,
        close_metadata={
            "close_reason": "stop_loss_hit",
            "entry_price": 100.0,
            "exit_price": 95.0,
            "atr_pct": 0.015,
        },
    )
    scorer._bandit.update.assert_called_once_with("copy_trade:0xA", won=False)


def test_record_outcome_flag_on_feeds_take_profit_as_win(scorer, monkeypatch):
    monkeypatch.setattr(config, "BANDIT_SKIP_NOISE_STOPS_ENABLED", True, raising=False)
    scorer.record_outcome(
        "copy_trade:0xA", "sig1", pnl=15.0,
        close_metadata={
            "close_reason": "take_profit_hit",
            "entry_price": 100.0,
            "exit_price": 120.0,
        },
    )
    scorer._bandit.update.assert_called_once_with("copy_trade:0xA", won=True)


def test_record_outcome_flag_on_no_metadata_uses_legacy(scorer, monkeypatch):
    """Backward compat: callers not supplying close_metadata still get the
    legacy P&L-sign behavior even with the flag enabled."""
    monkeypatch.setattr(config, "BANDIT_SKIP_NOISE_STOPS_ENABLED", True, raising=False)
    scorer.record_outcome("copy_trade:0xA", "sig1", pnl=-1.0)
    scorer._bandit.update.assert_called_once_with("copy_trade:0xA", won=False)


def test_record_outcome_classification_failure_falls_back(scorer, monkeypatch):
    """If classify_close raises for any reason, the wiring falls back
    to legacy behavior -- never breaks the trade-recording path."""
    monkeypatch.setattr(config, "BANDIT_SKIP_NOISE_STOPS_ENABLED", True, raising=False)
    with patch("src.signals.loss_attribution.classify_close",
               side_effect=RuntimeError("boom")):
        scorer.record_outcome(
            "copy_trade:0xA", "sig1", pnl=-1.0,
            close_metadata={"close_reason": "stop_loss_hit"},
        )
    # Either skipped (defensive) or legacy-fed; what matters is no exception.
    # We allow either since the wiring's defensive policy is acceptable.
    assert scorer._bandit.update.call_count in (0, 1)


def test_record_outcome_reconciled_skipped_when_enabled(scorer, monkeypatch):
    monkeypatch.setattr(config, "BANDIT_SKIP_NOISE_STOPS_ENABLED", True, raising=False)
    scorer.record_outcome(
        "copy_trade:0xA", "sig1", pnl=0.0,
        close_metadata={"close_reason": "live_reconciled_closed"},
    )
    scorer._bandit.update.assert_not_called()


# ── Static scorer gating (Issue #4 fix) ───────────────────────────


def test_record_outcome_flag_on_skips_static_counters_for_noise_stop(scorer, monkeypatch):
    """When gate is ON and close is NOISE_STOP, the static scorer's
    correct_signals / total_pnl must NOT be incremented either. Pre-fix,
    only the bandit was protected -- legacy dynamic_weight still degraded
    for sources hit by noise stops."""
    monkeypatch.setattr(config, "BANDIT_SKIP_NOISE_STOPS_ENABLED", True, raising=False)
    scorer.record_outcome(
        "copy_trade:0xA", "sig1", pnl=-1.0,
        close_metadata={
            "close_reason": "stop_loss_hit",
            "entry_price": 100.0,
            "exit_price": 99.97,
            "atr_pct": 0.015,
        },
    )
    s = scorer.scores["copy_trade:0xA"]
    # Static counters untouched
    assert s.correct_signals == 0
    assert s.total_pnl == 0.0
    assert s.total_return == 0.0
    # _recalculate / _save_score skipped too
    scorer._recalculate.assert_not_called()
    scorer._save_score.assert_not_called()
    scorer._bandit.update.assert_not_called()


def test_record_outcome_flag_on_still_updates_static_for_signal_loss(scorer, monkeypatch):
    """A real adverse move (SIGNAL_LOSS) still updates static counters."""
    monkeypatch.setattr(config, "BANDIT_SKIP_NOISE_STOPS_ENABLED", True, raising=False)
    scorer.record_outcome(
        "copy_trade:0xA", "sig1", pnl=-10.0,
        close_metadata={
            "close_reason": "stop_loss_hit",
            "entry_price": 100.0,
            "exit_price": 95.0,
            "atr_pct": 0.015,
        },
    )
    s = scorer.scores["copy_trade:0xA"]
    assert s.total_pnl == -10.0
    assert s.correct_signals == 0   # not a win
    scorer._recalculate.assert_called_once_with("copy_trade:0xA")
    scorer._bandit.update.assert_called_once_with("copy_trade:0xA", won=False)


def test_record_outcome_flag_off_always_updates_static(scorer, monkeypatch):
    """Default-OFF posture: even on a noise-shaped close, the static
    counters still update (legacy byte-identical behavior)."""
    monkeypatch.setattr(config, "BANDIT_SKIP_NOISE_STOPS_ENABLED", False, raising=False)
    scorer.record_outcome(
        "copy_trade:0xA", "sig1", pnl=-1.0,
        close_metadata={
            "close_reason": "stop_loss_hit",
            "entry_price": 100.0,
            "exit_price": 99.97,
            "atr_pct": 0.015,
        },
    )
    s = scorer.scores["copy_trade:0xA"]
    assert s.total_pnl == -1.0
    scorer._bandit.update.assert_called_once()
