import logging

import config
from src.core import boot
from src.core.live_execution import get_execution_open_positions, sync_shadow_book_to_live
from src.signals.signal_schema import TradeSignal, signal_from_execution_dict
from src.trading.live_trader import LiveTrader
from src.trading.portfolio_rotation import PortfolioRotationManager, RotationDecision


def test_shadow_mode_allows_missing_rotation_threshold_envs(monkeypatch):
    logger = logging.getLogger("test-shadow")
    monkeypatch.setattr(config, "ROTATION_ENGINE_ENABLED", True)
    monkeypatch.setattr(config, "ROTATION_REQUIRE_EXPLICIT_THRESHOLDS", True)
    monkeypatch.setattr(config, "ROTATION_DRY_RUN_TELEMETRY", True)
    monkeypatch.setattr(config, "ROTATION_SHADOW_MODE_DAYS", 7)

    required = [
        "PORTFOLIO_REPLACEMENT_THRESHOLD",
        "PORTFOLIO_MAX_REPLACEMENTS_PER_CYCLE",
        "PORTFOLIO_MAX_REPLACEMENTS_PER_HOUR",
        "PORTFOLIO_MAX_REPLACEMENTS_PER_DAY",
        "PORTFOLIO_FORCED_EXIT_COOLDOWN_MINUTES",
        "PORTFOLIO_ROUND_TRIP_BLOCK_MINUTES",
        "PORTFOLIO_MAX_COIN_EXPOSURE_PCT",
        "PORTFOLIO_MAX_SIDE_EXPOSURE_PCT",
        "PORTFOLIO_MAX_CLUSTER_EXPOSURE_PCT",
        "PORTFOLIO_TRANSACTION_COST_WEIGHT",
        "PORTFOLIO_CHURN_PENALTY",
        "PORTFOLIO_EXPECTED_SLIPPAGE_BPS",
    ]
    for env_name in required:
        monkeypatch.delenv(env_name, raising=False)

    boot.validate_operational_controls(logger)


def test_live_mode_still_requires_explicit_rotation_threshold_envs(monkeypatch):
    logger = logging.getLogger("test-live-thresholds")
    monkeypatch.setattr(config, "ROTATION_ENGINE_ENABLED", True)
    monkeypatch.setattr(config, "ROTATION_REQUIRE_EXPLICIT_THRESHOLDS", True)
    monkeypatch.setattr(config, "ROTATION_DRY_RUN_TELEMETRY", False)

    monkeypatch.delenv("PORTFOLIO_REPLACEMENT_THRESHOLD", raising=False)

    try:
        boot.validate_operational_controls(logger)
        assert False, "Expected RuntimeError when live mode lacks explicit thresholds"
    except RuntimeError as exc:
        assert "PORTFOLIO_REPLACEMENT_THRESHOLD" in str(exc)


def test_signal_from_execution_dict_preserves_trade_metadata():
    signal = signal_from_execution_dict(
        {
            "coin": "BTC",
            "side": "long",
            "confidence": 0.73,
            "entry_price": 101000,
            "strategy_id": 42,
            "strategy_type": "momentum_long",
            "position_pct": 0.06,
            "leverage": 3,
            "size": 0.15,
            "signal_id": "abc123",
            "source_accuracy": 0.61,
        }
    )

    assert signal.coin == "BTC"
    assert signal.side.value == "long"
    assert signal.strategy_id == 42
    assert signal.strategy_type == "momentum_long"
    assert signal.position_pct == 0.06
    assert signal.size == 0.15
    assert signal.signal_id == "abc123"
    assert signal.source_accuracy == 0.61


def test_shadow_mode_bypasses_rotation_reject_when_capacity_exists():
    manager = PortfolioRotationManager({"target_positions": 8})
    decision = RotationDecision(
        action="reject",
        reason="reserved slots held for stronger arrivals",
        candidate_score=0.55,
    )
    assert manager.should_bypass_reject_in_shadow_mode(decision, open_positions_count=6) is True
    assert manager.should_bypass_reject_in_shadow_mode(decision, open_positions_count=8) is False


def test_live_trader_coerces_execution_dict_and_uses_live_firewall_state(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            assert isinstance(signal, TradeSignal)
            assert signal.coin == "ETH"
            assert kwargs["open_positions"] == [{"coin": "BTC", "size": 0.1, "entry_price": 100000, "leverage": 2}]
            assert kwargs["account_balance"] == 2500.0
            return True, "ok"

    def fake_load_credentials(self):
        self.signer = type("Signer", (), {"address": "0x1111111111111111111111111111111111111111"})()
        self.agent_wallet_address = self.signer.address
        self.public_address = "0x2222222222222222222222222222222222222222"
        self.status_reason = "credentials_loaded"

    monkeypatch.setattr(LiveTrader, "_load_credentials", fake_load_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [
        {"coin": "BTC", "size": 0.1, "entry_price": 100000, "leverage": 2}
    ])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(LiveTrader, "place_market_order", lambda self, *args, **kwargs: {"status": "success"})
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(LiveTrader, "verify_fill", lambda self, *args, **kwargs: {"status": "verified"})
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(LiveTrader, "place_trigger_order", lambda self, *args, **kwargs: {"status": "success"})

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False)
    result = trader.execute_signal(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.1,
            "strategy_type": "momentum_long",
        }
    )

    assert result is not None
    assert result["status"] == "success"


def test_execution_open_positions_prefer_live_state_when_deployable():
    class FakeLiveTrader:
        def is_live_enabled(self):
            return True

        def is_deployable(self):
            return True

        def get_positions(self):
            return [{"coin": "BTC", "side": "long", "size": 0.25}]

    container = type("Container", (), {"live_trader": FakeLiveTrader()})()
    positions = get_execution_open_positions(container)
    assert positions == [{"coin": "BTC", "side": "long", "size": 0.25}]


def test_sync_shadow_book_closes_paper_trade_when_live_position_missing(monkeypatch):
    closed = []

    class FakeLiveTrader:
        def is_live_enabled(self):
            return True

        def is_deployable(self):
            return True

        def get_positions(self):
            return []

    class FakePaperTrader:
        def _close_trade(self, trade, current_price, close_reason):
            closed.append((trade["coin"], current_price, close_reason))
            return {"trade_id": trade["id"], "coin": trade["coin"], "reason": close_reason}

    container = type(
        "Container",
        (),
        {"live_trader": FakeLiveTrader(), "paper_trader": FakePaperTrader()},
    )()

    monkeypatch.setattr(
        "src.core.live_execution.db.get_open_paper_trades",
        lambda: [{"id": 7, "coin": "ETH", "side": "long", "entry_price": 2000.0}],
    )
    monkeypatch.setattr(
        "src.core.live_execution.get_all_mids",
        lambda: {"ETH": 2100.0},
    )

    reconciled = sync_shadow_book_to_live(container)
    assert reconciled == [{"trade_id": 7, "coin": "ETH", "reason": "live_reconciled_closed"}]
    assert closed == [("ETH", 2100.0, "live_reconciled_closed")]
