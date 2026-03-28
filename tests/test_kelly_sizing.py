"""
Unit tests for Kelly Criterion position sizing.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.signals.kelly_sizing import KellySizer, SizingResult


def test_kelly_no_edge():
    """Kelly should return 0 when there's no edge (win rate too low)."""
    ks = KellySizer({"vol_adjusted_kelly": False})
    fraction = ks.calculate_kelly(win_rate=0.3, reward_risk_ratio=1.0)
    assert fraction == 0.0


def test_kelly_positive_edge():
    """Kelly should return positive fraction with a genuine edge."""
    ks = KellySizer({"kelly_multiplier": 0.5, "vol_adjusted_kelly": False})
    # 60% win rate, 1.5 R:R → good edge
    fraction = ks.calculate_kelly(win_rate=0.6, reward_risk_ratio=1.5)
    assert fraction > 0
    assert fraction < 1.0  # Should be < 1 with half-kelly


def test_kelly_quarter_kelly_smaller():
    """Quarter-Kelly should produce smaller fractions than half-Kelly."""
    half = KellySizer({"kelly_multiplier": 0.5, "vol_adjusted_kelly": False})
    quarter = KellySizer({"kelly_multiplier": 0.25, "vol_adjusted_kelly": False})

    f_half = half.calculate_kelly(0.6, 1.5)
    f_quarter = quarter.calculate_kelly(0.6, 1.5)

    assert f_quarter < f_half
    assert f_quarter == pytest.approx(f_half / 2, rel=0.01)


def test_sizing_insufficient_data():
    """Should return default sizing with insufficient trade history."""
    ks = KellySizer({"min_trades_for_kelly": 15, "vol_adjusted_kelly": False})
    result = ks.get_sizing("test_strategy", account_balance=10000)
    assert result.confidence == "insufficient"
    assert result.trades_used == 0
    assert result.position_usd > 0


def test_sizing_with_history():
    """Should compute proper Kelly with enough historical trades."""
    ks = KellySizer({"min_trades_for_kelly": 5, "vol_adjusted_kelly": False,
                     "kelly_multiplier": 0.5})

    # Simulate 20 winning trades and 10 losing ones
    for _ in range(20):
        ks.record_outcome("momentum_long", pnl=100,
                          entry_price=50000, size=0.01, leverage=3)
    for _ in range(10):
        ks.record_outcome("momentum_long", pnl=-80,
                          entry_price=50000, size=0.01, leverage=3)

    result = ks.get_sizing("momentum_long", account_balance=10000, signal_confidence=0.7)
    assert result.has_edge is True
    assert result.win_rate == pytest.approx(2/3, rel=0.01)
    assert result.position_pct > 0
    assert result.position_usd > 0


def test_sizing_caps():
    """Position size should be capped at max_position_pct."""
    ks = KellySizer({"max_position_pct": 0.08, "min_trades_for_kelly": 5,
                     "vol_adjusted_kelly": False, "kelly_multiplier": 1.0})

    # Very strong edge to push Kelly high
    for _ in range(50):
        ks.record_outcome("strong", pnl=500, entry_price=50000, size=0.01, leverage=3)
    for _ in range(5):
        ks.record_outcome("strong", pnl=-50, entry_price=50000, size=0.01, leverage=3)

    result = ks.get_sizing("strong", account_balance=10000, signal_confidence=1.0)
    assert result.position_pct <= 0.08


def test_record_keeps_last_200():
    """Should only keep the last 200 trades per strategy."""
    ks = KellySizer({"vol_adjusted_kelly": False})
    for i in range(250):
        ks.record_outcome("test", pnl=10, entry_price=100, size=1, leverage=1)

    assert len(ks._strategy_outcomes["test"]) == 200


def test_all_sizing_stats():
    """get_all_sizing_stats should report per-strategy stats."""
    ks = KellySizer({"min_trades_for_kelly": 5, "vol_adjusted_kelly": False})

    for _ in range(10):
        ks.record_outcome("strat_a", pnl=50, entry_price=100, size=1, leverage=1)
    for _ in range(5):
        ks.record_outcome("strat_a", pnl=-30, entry_price=100, size=1, leverage=1)

    stats = ks.get_all_sizing_stats()
    assert "strat_a" in stats
    assert stats["strat_a"]["trades"] == 15
    assert 0 < stats["strat_a"]["win_rate"] < 1
