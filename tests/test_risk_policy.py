from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
    assert trend_policy.take_profit_roe_pct == round(
        trend_policy.stop_roe_pct * trend_policy.reward_multiple,
        6,
    )


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
        lambda: [
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
        lambda: [
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
