from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest

from src.backtest.risk_policy_walkforward import validate_risk_policy_walkforward
from src.signals.risk_policy import RiskPolicyEngine
from src.signals.signal_schema import SignalSide, SignalSource, TradeSignal
from src.trading.live_trader import LiveTrader


def _fake_live_credentials(self):
    self.signer = type("Signer", (), {"address": "0x1111111111111111111111111111111111111111"})()
    self.agent_wallet_address = self.signer.address
    self.public_address = "0x2222222222222222222222222222222222222222"
    self.status_reason = "credentials_loaded"


class _AllowAllFirewall:
    def validate(self, signal, **kwargs):
        return True, "ok"


def test_risk_policy_engine_adjusts_reward_and_time_limit_by_context():
    engine = RiskPolicyEngine()

    trend_signal = TradeSignal(
        coin="BTC",
        side=SignalSide.LONG,
        confidence=0.85,
        source=SignalSource.STRATEGY,
        reason="trend",
        leverage=3,
        source_accuracy=0.72,
        context={"atr_pct": 0.015, "expected_return": 0.30},
    )
    range_signal = TradeSignal(
        coin="BTC",
        side=SignalSide.SHORT,
        confidence=0.38,
        source=SignalSource.OPTIONS_FLOW,
        reason="range fade",
        leverage=3,
        source_accuracy=0.35,
        context={"atr_pct": 0.01, "expected_return": 0.03},
    )

    trend_policy = engine.resolve(
        trend_signal,
        regime_data={"regime": "trending_up", "confidence": 0.82},
        source_policy={"quality": 0.7, "status": "healthy"},
    )
    range_policy = engine.resolve(
        range_signal,
        regime_data={"regime": "ranging", "confidence": 0.74},
        source_policy={"quality": 0.35, "status": "degraded"},
    )

    assert trend_policy.reward_multiple > range_policy.reward_multiple
    assert trend_policy.time_limit_hours > range_policy.time_limit_hours
    assert trend_policy.stop_roe_pct > 0
    assert range_policy.stop_roe_pct > 0
    assert trend_policy.take_profit_roe_pct == pytest.approx(
        trend_policy.stop_roe_pct * trend_policy.reward_multiple,
        abs=5e-6,
    )


def _make_policy_signal(**kwargs):
    """Helper for H11 tests — neutral signal so the rr_mode's behavior is
    observable without regime/confidence noise."""
    sig = TradeSignal(
        coin="BTC",
        side=SignalSide.LONG,
        confidence=kwargs.get("confidence", 0.65),
        source=kwargs.get("source", SignalSource.STRATEGY),
        reason="rr-mode test",
        leverage=kwargs.get("leverage", 3),
        source_accuracy=kwargs.get("source_accuracy", 0.6),
        context=kwargs.get("context", {"atr_pct": 0.008, "expected_return": 0.12}),
    )
    if "stop_loss_pct" in kwargs:
        sig.risk.stop_loss_pct = kwargs["stop_loss_pct"]
        sig.risk.risk_basis = "roe"
    return sig


def test_risk_policy_engine_rejects_invalid_rr_mode():
    """H11: invalid rr_mode must fail loud at construction rather than
    silently falling back to a default — operators need feedback that
    their config typo won't deploy unchanged."""
    with pytest.raises(ValueError, match="Invalid rr_mode"):
        RiskPolicyEngine({"rr_mode": "not_a_mode"})


def test_risk_policy_engine_default_mode_is_dynamic_bounded():
    """H11: default rr_mode is dynamic_bounded (preserves legacy behavior
    for any caller that doesn't opt in to a specific mode)."""
    engine = RiskPolicyEngine()
    assert engine.rr_mode == "dynamic_bounded"


def test_risk_policy_engine_fixed_5r_targets_exactly_5r():
    """H11: fixed_5r mode must produce reward_multiple ≈ 5.0 regardless
    of regime/confidence/source-quality modifiers.  This is the canary-tier
    predictability guarantee: a TP placed at fixed 5R of the stop no matter
    what the dynamic adjustments would have preferred."""
    engine = RiskPolicyEngine({"rr_mode": "fixed_5r", "fixed_r_target": 5.0})
    # Use a signal that would *normally* earn a low reward_multiple
    # (range regime, weak source, low confidence): under fixed_5r, all
    # those dynamic knobs should be ignored.
    signal = _make_policy_signal(
        confidence=0.30,
        source_accuracy=0.30,
        context={"atr_pct": 0.008, "expected_return": 0.02},
    )
    policy = engine.resolve(
        signal,
        regime_data={"regime": "ranging", "confidence": 0.70},
        source_policy={"quality": 0.30, "status": "degraded"},
    )
    assert policy.rr_mode == "fixed_5r"
    assert policy.reward_multiple == pytest.approx(5.0, abs=1e-3), (
        f"fixed_5r must produce R≈5.0, got {policy.reward_multiple}"
    )
    # take_profit_roe_pct must equal stop_roe_pct * 5.0 (the core invariant)
    assert policy.take_profit_roe_pct == pytest.approx(
        policy.stop_roe_pct * 5.0, abs=5e-6
    )


def test_risk_policy_engine_hybrid_min_5r_floors_weak_signals():
    """H11: hybrid_min_5r mode runs the dynamic adjustments but floors
    the final R at the configured floor.  A weak signal that would
    normally earn ~2R under dynamic_bounded should come out at 5R under
    hybrid_min_5r."""
    dyn_engine = RiskPolicyEngine({"rr_mode": "dynamic_bounded"})
    hyb_engine = RiskPolicyEngine({"rr_mode": "hybrid_min_5r", "hybrid_min_r_floor": 5.0})

    weak_signal = _make_policy_signal(
        confidence=0.30,
        source_accuracy=0.30,
        context={"atr_pct": 0.008, "expected_return": 0.02},
    )
    regime_data = {"regime": "ranging", "confidence": 0.70}
    source_policy = {"quality": 0.30, "status": "degraded"}

    dyn_policy = dyn_engine.resolve(weak_signal, regime_data=regime_data, source_policy=source_policy)
    hyb_policy = hyb_engine.resolve(weak_signal, regime_data=regime_data, source_policy=source_policy)

    # dynamic_bounded should have produced low R (far below 5.0)
    assert dyn_policy.reward_multiple < 5.0
    # hybrid_min_5r must floor at 5.0 despite the same weak signal
    assert hyb_policy.rr_mode == "hybrid_min_5r"
    assert hyb_policy.reward_multiple == pytest.approx(5.0, abs=1e-3), (
        f"hybrid_min_5r floor must lift R to 5.0, got {hyb_policy.reward_multiple} "
        f"(dyn baseline was {dyn_policy.reward_multiple})"
    )


def test_risk_policy_engine_hybrid_min_5r_keeps_dynamic_upside_when_above_floor():
    """H11: hybrid_min_5r only *floors* — when the dynamic path produces
    an R ABOVE the floor, the hybrid mode should keep the dynamic value
    (it's the best of both worlds)."""
    # Raise max_reward_multiple so dynamic can legitimately exceed 5.0
    hyb_engine = RiskPolicyEngine({
        "rr_mode": "hybrid_min_5r",
        "hybrid_min_r_floor": 3.0,  # low floor so dynamic can exceed it
        "default_reward_multiple": 4.0,
        "max_reward_multiple": 7.0,
    })
    strong_signal = _make_policy_signal(
        confidence=0.90,
        source_accuracy=0.85,
        context={"atr_pct": 0.004, "expected_return": 0.30},
    )
    policy = hyb_engine.resolve(
        strong_signal,
        regime_data={"regime": "trending_up", "confidence": 0.85},
        source_policy={"quality": 0.80, "status": "healthy"},
    )
    # Dynamic path earns > 3.0 with these inputs, hybrid should preserve it
    assert policy.reward_multiple >= 3.0
    # And policy should surface the active mode for downstream auditing
    assert policy.rr_mode == "hybrid_min_5r"


def test_risk_policy_engine_caps_extreme_price_distance_even_with_low_leverage():
    engine = RiskPolicyEngine()
    signal = TradeSignal(
        coin="BTC",
        side=SignalSide.LONG,
        confidence=0.68,
        source=SignalSource.STRATEGY,
        reason="wide swing",
        leverage=1,
        source_accuracy=0.65,
        context={"atr_pct": 0.006, "expected_return": 0.12},
    )
    signal.risk.stop_loss_pct = 0.12
    signal.risk.take_profit_pct = 0.60
    signal.risk.reward_to_risk_ratio = 5.0
    signal.risk.risk_basis = "roe"

    policy = engine.resolve(
        signal,
        regime_data={"regime": "trending_up", "confidence": 0.72},
        source_policy={"quality": 0.62, "status": "healthy"},
    )

    assert policy.stop_roe_pct <= 0.025
    assert policy.take_profit_roe_pct <= 0.07
    assert policy.reward_multiple < 5.0


def test_live_manage_open_positions_trails_stop_using_shadow_policy(monkeypatch):
    updates = []
    cancellations = []
    placed = []

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)

    trader = LiveTrader(firewall=_AllowAllFirewall(), dry_run=False, max_order_usd=1_000_000)

    opened_at = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr(
        "src.trading.live_trader.db.get_open_paper_trades",
        lambda: [
            {
                "id": 1,
                "coin": "ETH",
                "side": "long",
                "entry_price": 100.0,
                "size": 1.0,
                "leverage": 5.0,
                "opened_at": opened_at,
                "metadata": {
                    "risk_policy": {
                        "stop_roe_pct": 0.05,
                        "take_profit_roe_pct": 0.25,
                        "reward_multiple": 5.0,
                        "time_limit_hours": 24.0,
                        "breakeven_at_r": 1.0,
                        "breakeven_buffer_roe_pct": 0.005,
                        "trail_after_r": 1.5,
                        "trailing_enabled": True,
                        "trailing_distance_roe_pct": 0.0375,
                    }
                },
            }
        ],
    )
    monkeypatch.setattr(
        trader,
        "get_positions",
        lambda **kw: [
            {
                "coin": "ETH",
                "side": "long",
                "size": 1.0,
                "szi": 1.0,
                "entry_price": 100.0,
                "entryPx": 100.0,
                "leverage": 5.0,
            }
        ],
    )
    monkeypatch.setattr(trader, "_get_mid_price", lambda coin: 102.0)
    monkeypatch.setattr(
        trader,
        "_cancel_protective_orders",
        lambda coin: cancellations.append(coin) or 2,
    )
    monkeypatch.setattr(
        trader,
        "_place_protective_orders_with_retries",
        lambda coin, close_side, size, sl_price, tp_price: (
            placed.append((coin, close_side, size, sl_price, tp_price)) or {"status": "success"},
            {"status": "success"},
            1,
        ),
    )
    monkeypatch.setattr(
        trader,
        "_update_shadow_trade_risk_levels",
        lambda trades, **kwargs: updates.append(kwargs),
    )

    summary = trader.manage_open_positions()

    assert summary["updated"] == 1
    assert summary["closed"] == 0
    assert cancellations == ["ETH"]
    assert placed
    _, close_side, size, stop_loss, take_profit = placed[0]
    assert close_side == "sell"
    assert size == 1.0
    assert stop_loss > 100.0
    assert take_profit == 105.0
    assert updates[0]["metadata_updates"]["risk_event"] == "trailing"


def test_live_manage_open_positions_closes_on_time_limit(monkeypatch):
    close_calls = []
    updates = []
    stale_opened_at = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)

    trader = LiveTrader(firewall=_AllowAllFirewall(), dry_run=False, max_order_usd=1_000_000)

    monkeypatch.setattr(
        "src.trading.live_trader.db.get_open_paper_trades",
        lambda: [
            {
                "id": 2,
                "coin": "BTC",
                "side": "short",
                "entry_price": 100.0,
                "size": 1.0,
                "leverage": 4.0,
                "opened_at": stale_opened_at,
                "metadata": {
                    "risk_policy": {
                        "stop_roe_pct": 0.04,
                        "take_profit_roe_pct": 0.20,
                        "reward_multiple": 5.0,
                        "time_limit_hours": 2.0,
                        "breakeven_at_r": 1.0,
                        "breakeven_buffer_roe_pct": 0.005,
                        "trail_after_r": 2.0,
                        "trailing_enabled": True,
                        "trailing_distance_roe_pct": 0.03,
                    }
                },
            }
        ],
    )
    monkeypatch.setattr(
        trader,
        "get_positions",
        lambda **kw: [
            {
                "coin": "BTC",
                "side": "short",
                "size": 1.0,
                "szi": -1.0,
                "entry_price": 100.0,
                "entryPx": 100.0,
                "leverage": 4.0,
            }
        ],
    )
    monkeypatch.setattr(trader, "_get_mid_price", lambda coin: 99.0)
    monkeypatch.setattr(
        trader,
        "close_position",
        lambda coin: close_calls.append(coin) or {"status": "success"},
    )
    monkeypatch.setattr(
        trader,
        "_update_shadow_trade_risk_levels",
        lambda trades, **kwargs: updates.append(kwargs),
    )

    summary = trader.manage_open_positions()

    assert summary["closed"] == 1
    assert close_calls == ["BTC"]
    assert updates[0]["metadata_updates"]["risk_event"] == "time_limit"


def test_validate_risk_policy_walkforward_scores_sources_and_windows():
    base_time = datetime(2026, 4, 1, tzinfo=timezone.utc)
    closed = []

    for idx in range(24):
        closed.append(
            {
                "status": "closed",
                "coin": "ETH",
                "side": "long",
                "entry_price": 100.0,
                "size": 1.0,
                "leverage": 2.0,
                "pnl": 3.0 if idx % 4 else 1.0,
                "opened_at": (base_time + timedelta(hours=idx)).isoformat(),
                "closed_at": (base_time + timedelta(hours=idx, minutes=30)).isoformat(),
                "metadata": {"source": "strategy"},
            }
        )

    for idx in range(18):
        closed.append(
            {
                "status": "closed",
                "coin": "BTC",
                "side": "short",
                "entry_price": 100.0,
                "size": 1.0,
                "leverage": 2.0,
                "pnl": -1.0 if idx % 2 else 0.2,
                "opened_at": (base_time + timedelta(days=2, hours=idx)).isoformat(),
                "closed_at": (base_time + timedelta(days=2, hours=idx, minutes=20)).isoformat(),
                "metadata": {"source": "copy_trade"},
            }
        )

    result = validate_risk_policy_walkforward(
        closed,
        min_train_trades=8,
        min_test_trades=4,
        max_windows=3,
    )

    assert result["summary"]["sources"] == 2
    strategy_row = next(row for row in result["by_source"] if row["source"] == "strategy")
    assert strategy_row["windows"]
    assert strategy_row["total"]["count"] == 24
