from datetime import datetime, timedelta, timezone

from src.trading.regime_reversal_manager import (
    RegimeReversalConfig,
    RegimeReversalManager,
)


class _Forecaster:
    def __init__(self, payload):
        self.payload = payload

    def predict_regime(self, coin):
        return dict(self.payload)


def _position(side="long"):
    return {
        "coin": "BTC",
        "side": side,
        "entry_price": 100.0,
        "size": 1.0,
        "leverage": 5.0,
    }


def _trade():
    return {
        "id": 1,
        "coin": "BTC",
        "side": "long",
        "opened_at": (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat(),
    }


def test_opposite_regime_waits_for_confirmation_then_tightens():
    mgr = RegimeReversalManager(
        RegimeReversalConfig(
            confirm_cycles=2,
            min_confidence=0.70,
            close_enabled=False,
            tighten_enabled=True,
        )
    )
    forecaster = _Forecaster({"regime": "crash", "confidence": 0.80, "signal": -0.55})
    policy = {"stop_roe_pct": 0.05}

    first = mgr.evaluate(
        position=_position("long"),
        shadow_trades=[_trade()],
        policy=policy,
        current_price=101.0,
        current_r=0.2,
        forecaster=forecaster,
    )
    second = mgr.evaluate(
        position=_position("long"),
        shadow_trades=[_trade()],
        policy=policy,
        current_price=101.0,
        current_r=0.2,
        forecaster=forecaster,
    )

    assert first.action == "none"
    assert first.reason == "awaiting_confirmation"
    assert second.action == "tighten_stop"
    assert second.reverse_side == "short"
    assert second.stop_price is not None
    assert second.stop_price < 101.0


def test_close_and_reverse_requires_explicit_gates_and_higher_confidence():
    mgr = RegimeReversalManager(
        RegimeReversalConfig(
            confirm_cycles=1,
            close_enabled=True,
            reverse_enabled=True,
            reverse_on_crash=True,
            min_confidence=0.70,
            reverse_confidence=0.82,
        )
    )

    decision = mgr.evaluate(
        position=_position("long"),
        shadow_trades=[_trade()],
        policy={"stop_roe_pct": 0.05},
        current_price=99.0,
        current_r=-0.2,
        forecaster=_Forecaster({"regime": "crash", "confidence": 0.90, "signal": -0.75}),
    )

    assert decision.action == "close_and_reverse"
    assert decision.reason == "opposite_regime_confirmed_reverse"
    assert decision.reverse_side == "short"


def test_live_action_cooldown_downgrades_to_tighten_stop():
    now = datetime.now(timezone.utc)
    mgr = RegimeReversalManager(
        RegimeReversalConfig(
            confirm_cycles=1,
            close_enabled=True,
            reverse_enabled=True,
            min_confidence=0.70,
            reverse_confidence=0.82,
            cooldown_seconds=3600,
        )
    )
    mgr.mark_action("BTC", now=now)

    decision = mgr.evaluate(
        position=_position("short"),
        shadow_trades=[_trade()],
        policy={"stop_roe_pct": 0.05},
        current_price=100.0,
        current_r=-0.1,
        forecaster=_Forecaster({"regime": "bullish", "confidence": 0.95, "signal": 0.85}),
        now=now + timedelta(minutes=5),
    )

    assert decision.action == "tighten_stop"
    assert decision.reason == "live_action_blocked:cooldown"
    assert decision.metadata["live_action_blocked"] == "cooldown"


def test_aligned_or_low_confidence_regime_does_not_act():
    mgr = RegimeReversalManager(RegimeReversalConfig(confirm_cycles=1, min_confidence=0.70))

    aligned = mgr.evaluate(
        position=_position("long"),
        shadow_trades=[_trade()],
        policy={"stop_roe_pct": 0.05},
        current_price=100.0,
        current_r=0.0,
        forecaster=_Forecaster({"regime": "bullish", "confidence": 0.90, "signal": 0.7}),
    )
    weak = mgr.evaluate(
        position=_position("long"),
        shadow_trades=[_trade()],
        policy={"stop_roe_pct": 0.05},
        current_price=100.0,
        current_r=0.0,
        forecaster=_Forecaster({"regime": "crash", "confidence": 0.40, "signal": -0.7}),
    )

    assert aligned.action == "none"
    assert aligned.reason == "regime_aligned"
    assert weak.action == "none"
    assert weak.reason == "confidence_below_threshold"
