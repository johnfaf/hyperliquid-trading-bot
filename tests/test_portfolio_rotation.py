"""
Unit tests for the portfolio rotation policy.
"""
from datetime import datetime, timedelta, timezone

from src.trading.portfolio_rotation import PortfolioRotationManager


class MockSide:
    def __init__(self, value: str):
        self.value = value


class MockSignal:
    def __init__(self, confidence: float, side: str = "long", source_accuracy: float = 0.0):
        self.coin = "BTC"
        self.side = MockSide(side)
        self.confidence = confidence
        self.source_accuracy = source_accuracy


def _open_trade(trade_id: int, confidence: float, minutes_ago: int, side: str = "long",
                coin: str = "ETH", current_price: float = 100.0, entry_price: float = 100.0):
    opened_at = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()
    return {
        "id": trade_id,
        "coin": coin,
        "side": side,
        "entry_price": entry_price,
        "current_price": current_price,
        "size": 1.0,
        "leverage": 1.0,
        "opened_at": opened_at,
        "metadata": {
            "confidence": confidence,
            "source_accuracy": 0.0,
        },
    }


def test_rotation_uses_reserved_slots_for_high_conviction_only():
    manager = PortfolioRotationManager({
        "target_positions": 8,
        "reserved_high_conviction_slots": 2,
    })
    open_positions = [_open_trade(i, confidence=0.55, minutes_ago=180, coin=f"C{i}") for i in range(6)]

    low_signal = MockSignal(confidence=0.60)
    low_decision = manager.decide(low_signal, open_positions)
    assert low_decision.action == "reject"
    assert "reserved slots" in low_decision.reason

    high_signal = MockSignal(confidence=0.82)
    high_decision = manager.decide(high_signal, open_positions)
    assert high_decision.action == "open"


def test_rotation_replaces_weak_stale_incumbent():
    manager = PortfolioRotationManager({
        "target_positions": 3,
        "reserved_high_conviction_slots": 0,
        "replacement_threshold": 0.10,
        "min_hold_minutes": 60,
    })
    open_positions = [
        _open_trade(1, confidence=0.28, minutes_ago=720, coin="ETH", current_price=98.0),
        _open_trade(2, confidence=0.60, minutes_ago=90, coin="SOL", current_price=104.0),
        _open_trade(3, confidence=0.58, minutes_ago=90, coin="ARB", current_price=103.0),
    ]

    decision = manager.decide(MockSignal(confidence=0.78), open_positions)
    assert decision.action == "replace"
    assert decision.replacement_trade_id == 1


def test_rotation_respects_min_hold_window():
    manager = PortfolioRotationManager({
        "target_positions": 2,
        "reserved_high_conviction_slots": 0,
        "min_hold_minutes": 120,
    })
    open_positions = [
        _open_trade(1, confidence=0.20, minutes_ago=30, coin="ETH", current_price=99.0),
        _open_trade(2, confidence=0.25, minutes_ago=45, coin="SOL", current_price=99.0),
    ]

    decision = manager.decide(MockSignal(confidence=0.95), open_positions)
    assert decision.action == "reject"
    assert "hold window" in decision.reason


def test_rotation_guardrail_blocks_immediate_round_trip_reentry():
    manager = PortfolioRotationManager({
        "target_positions": 2,
        "reserved_high_conviction_slots": 0,
        "replacement_threshold": 0.01,
        "round_trip_block_minutes": 60,
        "forced_exit_cooldown_minutes": 60,
    })
    victim = _open_trade(1, confidence=0.20, minutes_ago=180, coin="BTC", side="long")
    incumbent = _open_trade(2, confidence=0.55, minutes_ago=180, coin="ETH", side="long")
    manager.register_replacement(victim, new_coin="SOL", new_side="long")

    decision = manager.decide(MockSignal(confidence=0.95, side="short"), [victim, incumbent])
    assert decision.action == "reject"
    assert "cooldown" in decision.reason or "round-trip" in decision.reason


def test_rotation_guardrail_caps_replacements_per_hour():
    manager = PortfolioRotationManager({
        "target_positions": 2,
        "reserved_high_conviction_slots": 0,
        "replacement_threshold": 0.01,
        "max_replacements_per_hour": 1,
        "max_replacements_per_day": 10,
    })
    victim = _open_trade(1, confidence=0.10, minutes_ago=200, coin="ARB")
    manager.register_replacement(victim, new_coin="ETH", new_side="long")

    decision = manager.decide(MockSignal(confidence=0.95), [victim, _open_trade(2, confidence=0.5, minutes_ago=200)])
    assert decision.action == "reject"
    assert "cap hit" in decision.reason


def test_rotation_tracks_replacement_outcomes():
    manager = PortfolioRotationManager({})
    victim = _open_trade(7, confidence=0.2, minutes_ago=240, coin="ARB", side="short")
    now = datetime.now(timezone.utc)
    opened_at = (now - timedelta(minutes=25)).isoformat()
    closed_at = now.isoformat()

    manager.register_replacement(
        victim,
        new_coin="ETH",
        new_side="long",
        candidate_score=0.83,
        incumbent_score=0.27,
        reason="higher_edge",
        new_trade_id=777,
        closed_trade_event={"pnl": -3.5, "closed_at": closed_at},
    )
    manager.record_trade_close(
        {
            "trade_id": 777,
            "pnl": 5.25,
            "opened_at": opened_at,
            "closed_at": closed_at,
        }
    )

    stats = manager.get_stats()
    assert stats["replacement_outcomes_recorded"] == 1
    assert abs(stats["replacement_outcome_win_rate"] - 1.0) < 1e-9
    assert stats["replacement_events"], "replacement_events should include recent event telemetry"
    event = stats["replacement_events"][-1]
    assert event["reason"] == "higher_edge"
    assert abs(event["candidate_score"] - 0.83) < 1e-9
    assert abs(event["incumbent_score"] - 0.27) < 1e-9
    assert abs(event["victim_realized_pnl"] - (-3.5)) < 1e-9
    assert event["replacement_outcome"] == "win"
