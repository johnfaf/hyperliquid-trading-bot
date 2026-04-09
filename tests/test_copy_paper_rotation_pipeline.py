from datetime import datetime, timedelta, timezone

import config
import pytest

from src.trading.copy_trader import CopyTrader
from src.trading.paper_trader import PaperTrader
from src.trading.portfolio_rotation import RotationDecision


def _open_trade(
    trade_id: int,
    coin: str,
    side: str,
    *,
    strategy_id=None,
    confidence: float = 0.4,
    minutes_ago: int = 240,
):
    return {
        "id": trade_id,
        "coin": coin,
        "side": side,
        "strategy_id": strategy_id,
        "entry_price": 100.0,
        "current_price": 100.0,
        "size": 1.0,
        "leverage": 1.0,
        "opened_at": (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat(),
        "metadata": {"confidence": confidence, "source_accuracy": 0.0},
    }


def test_copy_trader_rotation_prescreen_bypasses_capacity(monkeypatch):
    account = {"balance": 10_000.0}
    open_trades = [
        _open_trade(i, f"C{i}", "long", confidence=0.2 + i * 0.01)
        for i in range(1, 9)
    ]
    firewall_calls = []
    closed = []

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            firewall_calls.append(kwargs)
            if kwargs.get("dry_run") and not kwargs.get("ignore_position_limit"):
                return False, "Max positions reached (8/8)"
            return True, "ok"

    monkeypatch.setattr("src.trading.copy_trader.db.get_paper_account", lambda: account)
    monkeypatch.setattr("src.trading.copy_trader.db.get_open_paper_trades", lambda: list(open_trades))
    monkeypatch.setattr("src.trading.copy_trader.hl.get_all_mids", lambda: {"BTC": 100.0})

    trader = CopyTrader(firewall=FakeFirewall())
    monkeypatch.setattr(trader, "_annotate_open_trades", lambda trades, mids: None)
    monkeypatch.setattr(
        trader.rotation_manager,
        "decide",
        lambda *args, **kwargs: RotationDecision(
            action="replace",
            reason="candidate beats incumbent",
            candidate_score=0.8,
            incumbent_score=0.2,
            replacement_trade_id=1,
        ),
    )
    monkeypatch.setattr(
        trader,
        "_open_copy_trade",
        lambda account, signal, positions: {
            "id": 999,
            "coin": signal["coin"],
            "side": signal["side"],
            "entry_price": signal["price"],
            "size": 1.0,
            "leverage": signal["leverage"],
            "opened_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    monkeypatch.setattr(
        trader,
        "_close_trade",
        lambda trade, exit_price, close_reason: closed.append((trade["id"], close_reason)) or {"id": trade["id"]},
    )

    executed = trader.execute_copy_signals(
        [
            {
                "type": "copy_open",
                "coin": "BTC",
                "side": "long",
                "price": 100.0,
                "leverage": 2,
                "confidence": 0.9,
                "source_trader": "0xabc",
            }
        ],
        regime_data={"overall_regime": "neutral"},
    )

    assert len(executed) == 1
    assert firewall_calls[0]["dry_run"] is True
    assert firewall_calls[0]["ignore_position_limit"] is True
    assert firewall_calls[0]["open_positions"] == []
    assert closed == [(1, "rotation_out:BTC")]


def test_copy_trader_crash_weight_preserves_strong_signal_above_firewall_floor():
    class CrashForecaster:
        def predict_regime(self, coin):
            return {"regime": "crash"}

    trader = CopyTrader(regime_forecaster=CrashForecaster())
    weighted = trader._apply_regime_weight({"confidence": 0.9}, "BTC")

    assert weighted["confidence"] == pytest.approx(0.54)
    assert weighted["confidence"] >= getattr(config, "FIREWALL_MIN_CONFIDENCE", 0.45)


def test_paper_trader_replaces_existing_strategy_position_instead_of_dropping(monkeypatch):
    account = {"balance": 10_000.0}
    open_trades = [
        _open_trade(11, "XRP", "long", strategy_id=1, confidence=0.3),
        _open_trade(12, "ETH", "long", strategy_id=2, confidence=0.7),
        _open_trade(13, "SOL", "long", strategy_id=3, confidence=0.7),
        _open_trade(14, "DOGE", "long", strategy_id=4, confidence=0.7),
        _open_trade(15, "AVAX", "long", strategy_id=5, confidence=0.7),
        _open_trade(16, "LINK", "long", strategy_id=6, confidence=0.7),
        _open_trade(17, "ARB", "long", strategy_id=7, confidence=0.7),
        _open_trade(18, "OP", "long", strategy_id=8, confidence=0.7),
    ]
    firewall_calls = []
    closed = []

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            firewall_calls.append(kwargs)
            if kwargs.get("dry_run") and not kwargs.get("ignore_position_limit"):
                return False, "Max positions reached (8/8)"
            return True, "ok"

    monkeypatch.setattr("src.trading.paper_trader.db.get_paper_account", lambda: account)
    monkeypatch.setattr("src.trading.paper_trader.db.get_open_paper_trades", lambda: list(open_trades))
    monkeypatch.setattr("src.trading.paper_trader.hl.get_all_mids", lambda: {"ETH": 2000.0, "XRP": 1.0})

    trader = PaperTrader(firewall=FakeFirewall())
    monkeypatch.setattr(trader, "_annotate_open_trades", lambda trades, mids: None)
    monkeypatch.setattr(
        trader,
        "_generate_signal",
        lambda strategy, mids, regime_data=None: {
            "coin": "ETH",
            "side": "short",
            "price": 2000.0,
            "size": 0.05,
            "leverage": 2,
            "stop_loss": 2100.0,
            "take_profit": 1800.0,
            "strategy_type": "momentum_short",
            "confidence": 0.8,
        },
    )
    monkeypatch.setattr(
        trader,
        "_close_trade",
        lambda trade, current_price, close_reason: closed.append((trade["id"], close_reason)) or {"id": trade["id"]},
    )
    monkeypatch.setattr(
        trader,
        "_execute_paper_trade",
        lambda account, strategy, signal: {
            "id": 999,
            "coin": signal["coin"],
            "side": signal["side"],
            "strategy_id": strategy["id"],
            "entry_price": signal["price"],
            "size": signal["size"],
            "leverage": signal["leverage"],
            "opened_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    executed = trader.execute_strategy_signals(
        [{"id": 1, "name": "Momentum", "current_score": 0.8}],
        regime_data={"overall_regime": "neutral"},
    )

    assert len(executed) == 1
    assert executed[0]["strategy_id"] == 1
    assert executed[0]["coin"] == "ETH"
    assert firewall_calls[0]["dry_run"] is True
    assert firewall_calls[0]["ignore_position_limit"] is True
    assert firewall_calls[0]["open_positions"] == []
    assert len(firewall_calls[1]["open_positions"]) == 7
    assert closed == [(11, "rotation_out:ETH")]
