import sys
import types

import config
from src.signals.signal_schema import RiskParams, SignalSide, SignalSource, TradeSignal
from src.trading.lighter_live_trader import LighterLiveTrader


def _install_fake_lighter_sdk(monkeypatch, calls):
    class FakeSignerClient:
        ORDER_TYPE_MARKET = 1
        ORDER_TYPE_STOP_LOSS = 2
        ORDER_TYPE_TAKE_PROFIT = 4
        ORDER_TIME_IN_FORCE_IMMEDIATE_OR_CANCEL = 0
        ORDER_TIME_IN_FORCE_GOOD_TILL_TIME = 1
        NIL_TRIGGER_PRICE = 0
        DEFAULT_28_DAY_ORDER_EXPIRY = 123456

        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        async def create_order(self, **kwargs):
            calls.append(("create_order", kwargs))
            return {"ok": True}, "0xtx", None

    module = types.SimpleNamespace(SignerClient=FakeSignerClient)
    monkeypatch.setitem(sys.modules, "lighter", module)


def test_lighter_live_trader_fail_closed_without_enable(monkeypatch):
    monkeypatch.setattr(config, "LIGHTER_LIVE_TRADING_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "LIGHTER_LIVE_TRADING_DUAL_CONTROL_CONFIRM", False, raising=False)

    trader = LighterLiveTrader(
        dry_run=True,
        account_index=1,
        api_key_index=2,
        private_key="secret",
    )
    trader.market_adapter._reverse_symbol_map = {"BTC": "1"}
    trader.market_adapter._market_cache = {"1": {"size_decimals": 4, "price_decimals": 2}}
    monkeypatch.setattr(trader.market_adapter, "_ensure_markets_loaded", lambda: None)
    monkeypatch.setattr(trader, "_market_mid", lambda coin: 100.0)

    result = trader.place_market_order("BTC", "buy", 0.1, leverage=5)

    assert result["status"] == "dry_run"
    assert result["venue"] == "lighter"
    assert result["reason"] == "lighter_live_disabled_or_dry_run"


def test_lighter_execute_signal_places_entry_and_protective_orders(monkeypatch):
    calls = []
    _install_fake_lighter_sdk(monkeypatch, calls)
    monkeypatch.setattr(config, "LIGHTER_LIVE_TRADING_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "LIGHTER_LIVE_TRADING_DUAL_CONTROL_CONFIRM", True, raising=False)

    trader = LighterLiveTrader(
        dry_run=False,
        account_index=7,
        api_key_index=3,
        private_key="secret",
        max_order_usd=1_000,
    )
    trader.market_adapter._reverse_symbol_map = {"BTC": "1"}
    trader.market_adapter._market_cache = {"1": {"size_decimals": 4, "price_decimals": 2}}
    monkeypatch.setattr(trader.market_adapter, "_ensure_markets_loaded", lambda: None)
    monkeypatch.setattr(trader, "_market_mid", lambda coin: 100.0)

    signal = TradeSignal(
        coin="BTC",
        side=SignalSide.LONG,
        confidence=0.8,
        source=SignalSource.MANUAL,
        reason="test",
        entry_price=100.0,
        size=0.1,
        leverage=5,
        risk=RiskParams(stop_loss_pct=0.05, take_profit_pct=0.25, risk_basis="roe"),
    )

    result = trader.execute_signal(signal, bypass_firewall=True)

    assert result["status"] == "submitted"
    order_calls = [call for name, call in calls if name == "create_order"]
    assert len(order_calls) == 3
    assert order_calls[0]["order_type"] == 1
    assert order_calls[1]["order_type"] == 2
    assert order_calls[2]["order_type"] == 4
    assert order_calls[1]["reduce_only"] is True
    assert order_calls[2]["reduce_only"] is True


def test_lighter_execute_signal_runs_firewall_by_default(monkeypatch):
    calls = []
    _install_fake_lighter_sdk(monkeypatch, calls)
    monkeypatch.setattr(config, "LIGHTER_LIVE_TRADING_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "LIGHTER_LIVE_TRADING_DUAL_CONTROL_CONFIRM", True, raising=False)

    class _Firewall:
        def validate(self, *args, **kwargs):
            return False, "unit-test-block"

    trader = LighterLiveTrader(
        dry_run=False,
        account_index=7,
        api_key_index=3,
        private_key="secret",
        firewall=_Firewall(),
        max_order_usd=1_000,
    )
    monkeypatch.setattr(trader, "get_account_value", lambda: 1_000.0)
    monkeypatch.setattr(trader, "get_positions", lambda *_, **__: [])
    monkeypatch.setattr(trader, "_market_mid", lambda coin: 100.0)

    signal = TradeSignal(
        coin="BTC",
        side=SignalSide.LONG,
        confidence=0.8,
        source=SignalSource.MANUAL,
        reason="test",
        entry_price=100.0,
        size=0.1,
        leverage=5,
        risk=RiskParams(stop_loss_pct=0.05, take_profit_pct=0.25, risk_basis="roe"),
    )

    assert trader.execute_signal(signal) is None
    assert [name for name, _ in calls if name == "create_order"] == []
