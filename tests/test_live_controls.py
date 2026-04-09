import logging
import os
import sqlite3
import sys
import tempfile
import types
from contextlib import contextmanager
from datetime import datetime, timezone

import config
import main
from src.core import boot
from src.core.api_manager import Priority
from src.core.cycles.reporting_cycle import run_reporting
from src.core.cycles.trading_cycle import (
    _execute_signal_live,
    _execute_lcrs_signals,
    _execute_options_flow_trades,
    _process_closed_trades,
    _run_alpha_arena,
)
from src.core.health_registry import SubsystemHealthRegistry, SubsystemState
from src.core.live_execution import (
    _rescale_size_for_live,
    get_execution_open_positions,
    mirror_executed_trades_to_live,
    sync_shadow_book_to_live,
)
from src.signals.decision_firewall import DecisionFirewall
from src.signals.signal_schema import (
    SignalSide,
    SignalSource,
    TradeSignal,
    signal_from_execution_dict,
)
from src.trading.live_trader import (
    HyperliquidSigner,
    LiveTrader,
    _hl_format_price,
    _hl_format_size,
)
from src.trading.portfolio_rotation import PortfolioRotationManager, RotationDecision


def _fake_live_credentials(self):
    self.signer = type("Signer", (), {"address": "0x1111111111111111111111111111111111111111"})()
    self.agent_wallet_address = self.signer.address
    self.public_address = "0x2222222222222222222222222222222222222222"
    self.status_reason = "credentials_loaded"


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


def test_default_rotation_controls_allow_built_in_thresholds(monkeypatch):
    logger = logging.getLogger("test-live-default-thresholds")
    monkeypatch.setattr(config, "ROTATION_ENGINE_ENABLED", True)
    monkeypatch.setattr(config, "ROTATION_REQUIRE_EXPLICIT_THRESHOLDS", False)
    monkeypatch.setattr(config, "ROTATION_DRY_RUN_TELEMETRY", False)

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


def test_health_registry_requires_dependency_ready_for_trading_safety():
    registry = SubsystemHealthRegistry()
    registry.register("strategy_scorer", affects_trading=True)

    assert registry.is_trading_safe("strategy_scorer") is False
    assert registry.is_all_trading_safe() is False

    registry.set_status(
        "strategy_scorer",
        SubsystemState.HEALTHY,
        dependency_ready=True,
        startup_status="READY",
    )

    assert registry.is_trading_safe("strategy_scorer") is True
    assert registry.is_all_trading_safe() is True


def test_run_reporting_skips_false_positive_stale_warning(monkeypatch):
    warnings = []
    container = type(
        "Container",
        (),
        {
            "reporter": None,
            "paper_trader": None,
            "shadow_tracker": None,
            "cross_venue_hedger": None,
            "scorer": None,
        },
    )()

    class FakeHealth:
        def check_stale(self, timeout_seconds=600):
            return {"live_trader": False, "paper_trader": False}

    monkeypatch.setattr("src.core.cycles.reporting_cycle._log_module_stats", lambda container: None)
    monkeypatch.setattr("src.core.cycles.reporting_cycle.logger.warning", lambda msg, *args: warnings.append(msg % args))

    run_reporting(container, cycle_count=1, health_registry=FakeHealth())

    assert warnings == []


def test_live_trader_coerces_execution_dict_and_uses_live_firewall_state(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            assert isinstance(signal, TradeSignal)
            assert signal.coin == "ETH"
            assert kwargs["open_positions"] == [{"coin": "BTC", "size": 0.1, "entry_price": 100000, "leverage": 2}]
            assert kwargs["account_balance"] == 2500.0
            return True, "ok"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
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

    # Disable the bootstrap $ cap so the test exercises the original sizing path.
    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
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


def test_live_trader_maps_long_to_buy_and_places_sell_protection(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    entry_calls = []
    trigger_calls = []

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(
        LiveTrader,
        "place_market_order",
        lambda self, coin, side, size, leverage=1, reduce_only=False: (
            entry_calls.append((coin, side, size, reduce_only)) or {"status": "success"}
        ),
    )
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(LiveTrader, "verify_fill", lambda self, *args, **kwargs: {"status": "verified"})
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(
        LiveTrader,
        "place_trigger_order",
        lambda self, coin, side, size, trigger_price, tp_or_sl="sl": (
            trigger_calls.append((coin, side, size, trigger_price, tp_or_sl)) or {"status": "success"}
        ),
    )

    # Disable the bootstrap $ cap so entry_calls reflects the requested 0.1 ETH.
    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
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
    assert entry_calls == [("ETH", "buy", 0.1, False)]
    assert [call[1] for call in trigger_calls] == ["sell", "sell"]
    assert [call[4] for call in trigger_calls] == ["sl", "tp"]


def test_live_trader_routes_info_calls_through_api_manager(monkeypatch):
    class FakeManager:
        def __init__(self):
            self.calls = []

        def post(self, payload, **kwargs):
            self.calls.append((payload, kwargs))
            if payload.get("type") == "meta":
                return {"universe": [{"name": "ETH"}]}
            if payload.get("type") == "allMids":
                return {"ETH": "2000.0"}
            raise AssertionError(f"unexpected payload: {payload}")

    manager = FakeManager()

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr("src.trading.live_trader.get_manager", lambda: manager)
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=True)
    price = trader._get_mid_price("ETH")

    assert price == 2000.0
    assert manager.calls[0][0]["type"] == "meta"
    assert manager.calls[1][0]["type"] == "allMids"
    assert manager.calls[1][1]["priority"] == Priority.HIGH


def test_post_order_routes_exchange_requests_through_api_manager(monkeypatch):
    class FakeManager:
        def __init__(self):
            self.calls = []

        def post(self, payload, **kwargs):
            self.calls.append((payload, kwargs))
            return {"status": "ok"}

    class FakeSigner:
        address = "0x1111111111111111111111111111111111111111"

        def sign_action(self, action, nonce, vault_address=None):
            return {"r": "0x1", "s": "0x2", "v": 27}

    manager = FakeManager()

    def _load_credentials(self):
        self.signer = FakeSigner()
        self.agent_wallet_address = self.signer.address
        self.public_address = "0x2222222222222222222222222222222222222222"
        self.status_reason = "credentials_loaded"

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr("src.trading.live_trader.get_manager", lambda: manager)
    monkeypatch.setattr(LiveTrader, "_load_credentials", _load_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False)
    result = trader._post_order({"type": "order", "orders": []})

    assert result["status"] == "ok"
    exchange_call = next(
        kwargs for payload, kwargs in manager.calls if kwargs.get("req_type") == "exchange"
    )
    assert exchange_call["endpoint_url"] == config.HYPERLIQUID_EXCHANGE_URL
    assert exchange_call["cache_response"] is False
    assert exchange_call["priority"] == Priority.CRITICAL


def test_live_trader_uses_verified_fill_size_for_protection(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    trigger_calls = []

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(LiveTrader, "place_market_order", lambda self, *args, **kwargs: {"status": "success"})
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(
        LiveTrader,
        "verify_fill",
        lambda self, *args, **kwargs: {"status": "verified", "size": 0.06, "partial_fill": True},
    )
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(
        LiveTrader,
        "place_trigger_order",
        lambda self, coin, side, size, trigger_price, tp_or_sl="sl": (
            trigger_calls.append((coin, side, size, trigger_price, tp_or_sl)) or {"status": "success"}
        ),
    )

    # Disable the bootstrap $ cap so fill verification preserves the requested 0.1 size.
    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
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
    assert abs(result["size"] - 0.06) < 1e-12
    assert abs(result["requested_size"] - 0.1) < 1e-12
    assert [call[2] for call in trigger_calls] == [0.06, 0.06]


def test_execute_signal_uses_submitted_entry_size_for_fill_verification(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    verify_sizes = []

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(
        LiveTrader,
        "place_market_order",
        lambda self, *args, **kwargs: {"status": "success", "submitted_size": 0.09},
    )
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(
        LiveTrader,
        "verify_fill",
        lambda self, coin, side, expected_size, timeout=10.0, poll_interval=1.0: (
            verify_sizes.append(expected_size) or {"status": "verified", "size": expected_size}
        ),
    )
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(LiveTrader, "place_trigger_order", lambda self, *args, **kwargs: {"status": "success"})

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
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
    assert verify_sizes == [0.09]
    assert abs(result["requested_size"] - 0.1) < 1e-12
    assert abs(result["submitted_size"] - 0.09) < 1e-12


def test_execute_signal_uses_exchange_reported_fill_size_for_verification(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    verify_sizes = []
    trigger_sizes = []

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(
        LiveTrader,
        "place_market_order",
        lambda self, *args, **kwargs: {
            "status": "ok",
            "submitted_size": 371.1539175295995,
            "exchange_reported_fill_size": 1.0,
        },
    )
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(
        LiveTrader,
        "verify_fill",
        lambda self, coin, side, expected_size, timeout=10.0, poll_interval=1.0: (
            verify_sizes.append(expected_size)
            or {"status": "verified", "size": expected_size, "partial_fill": True}
        ),
    )
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 0.0308)
    monkeypatch.setattr(
        LiveTrader,
        "place_trigger_order",
        lambda self, coin, side, size, trigger_price, tp_or_sl="sl": (
            trigger_sizes.append(size) or {"status": "success"}
        ),
    )

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
    result = trader.execute_signal(
        {
            "coin": "KAS",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 0.0308,
            "position_pct": 0.05,
            "leverage": 1,
            "size": 389.7116134060795,
            "strategy_type": "copy_long",
        }
    )

    assert result is not None
    assert result["status"] == "success"
    assert verify_sizes == [1.0]
    assert trigger_sizes == [1.0, 1.0]
    assert abs(result["requested_size"] - 389.7116134060795) < 1e-12
    assert abs(result["submitted_size"] - 371.1539175295995) < 1e-12
    assert result["exchange_reported_fill_size"] == 1.0


def test_live_trader_closes_position_when_protection_fails(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    trigger_results = iter(
        [
            {"status": "success"},
            {"status": "error", "message": "tp rejected"},
        ]
    )
    close_calls = []

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(LiveTrader, "place_market_order", lambda self, *args, **kwargs: {"status": "success"})
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(LiveTrader, "verify_fill", lambda self, *args, **kwargs: {"status": "verified"})
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(LiveTrader, "place_trigger_order", lambda self, *args, **kwargs: next(trigger_results))
    monkeypatch.setattr(
        LiveTrader,
        "close_position",
        lambda self, coin: close_calls.append(coin) or {"status": "success", "coin": coin},
    )

    # Disable the bootstrap $ cap so the sizing reaches place_trigger_order unmodified.
    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
    result = trader.execute_signal(
        {
            "coin": "ETH",
            "side": "short",
            "confidence": 0.75,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.1,
            "strategy_type": "momentum_short",
        }
    )

    assert result is not None
    assert result["status"] == "error"
    assert result["message"] == "protective_order_failed"
    assert close_calls == ["ETH"]


def test_verify_fill_waits_for_minimum_meaningful_fill(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    class FakeClock:
        def __init__(self):
            self.current = 0.0

        def time(self):
            return self.current

        def sleep(self, seconds):
            self.current += seconds

    positions = iter(
        [
            [{"coin": "ETH", "szi": 0.04}],
            [{"coin": "ETH", "szi": 0.06}],
        ]
    )
    clock = FakeClock()

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_positions", lambda self: next(positions))
    monkeypatch.setattr("src.trading.live_trader.time.time", clock.time)
    monkeypatch.setattr("src.trading.live_trader.time.sleep", clock.sleep)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False)
    result = trader.verify_fill("ETH", "buy", 0.1, timeout=1.0, poll_interval=0.1)

    assert result is not None
    assert result["status"] == "verified"
    assert result["partial_fill"] is True
    assert abs(result["size"] - 0.06) < 1e-12
    assert abs(result["position_size"] - 0.06) < 1e-12


def test_execute_lcrs_signals_live_path_executes_signal(monkeypatch):
    executed = []

    class FakeLiveTrader:
        def execute_signal(self, signal, bypass_firewall=False):
            executed.append((signal.coin, signal.side.value, bypass_firewall))
            return {"status": "success", "coin": signal.coin}

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    container = type(
        "Container",
        (),
        {
            "live_trader": FakeLiveTrader(),
            "firewall": FakeFirewall(),
            "kelly_sizer": None,
            "trade_memory": None,
            "llm_filter": None,
        },
    )()

    monkeypatch.setattr("src.core.cycles.trading_cycle.is_live_trading_active", lambda container: True)
    monkeypatch.setattr("src.core.cycles.trading_cycle.get_execution_open_positions", lambda container: [])
    monkeypatch.setattr("src.notifications.telegram_bot.is_configured", lambda: False)

    _execute_lcrs_signals(
        container,
        [
            {
                "coin": "ETH",
                "side": "long",
                "confidence": 0.7,
                "price": 2000.0,
                "leverage": 2,
                "stop_loss": 1950.0,
                "take_profit": 2100.0,
                "features": {"setup_type": "reversal"},
            }
        ],
        {"overall_regime": "neutral"},
    )

    assert executed == [("ETH", "long", True)]


def test_execute_options_flow_live_path_executes_signal(monkeypatch):
    executed = []

    class FakeLiveTrader:
        def execute_signal(self, signal, bypass_firewall=False):
            executed.append((signal.coin, signal.side.value, bypass_firewall))
            return {"status": "success", "coin": signal.coin}

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    class FakeAgentScorer:
        def record_signal(self, *args, **kwargs):
            return None

    container = type(
        "Container",
        (),
        {
            "live_trader": FakeLiveTrader(),
            "firewall": FakeFirewall(),
            "agent_scorer": FakeAgentScorer(),
            "options_scanner": type(
                "Scanner",
                (),
                {
                    "top_convictions": [
                        {
                            "ticker": "ETH",
                            "direction": "BULLISH",
                            "conviction_pct": 75,
                            "net_flow": 100000.0,
                            "total_prints": 3,
                        }
                    ]
                },
            )(),
        },
    )()

    monkeypatch.setattr("src.core.cycles.trading_cycle.is_live_trading_active", lambda container: True)
    monkeypatch.setattr("src.core.cycles.trading_cycle.get_execution_open_positions", lambda container: [])
    monkeypatch.setattr("src.data.hyperliquid_client.get_all_mids", lambda: {"ETH": "2000.0"})
    monkeypatch.setattr("src.notifications.telegram_bot.is_configured", lambda: False)

    _execute_options_flow_trades(container, {"overall_regime": "neutral"})

    assert executed == [("ETH", "long", False)]


def test_execute_options_flow_live_path_skips_unfunded_perps(monkeypatch):
    executed = []

    class FakeLiveTrader:
        max_position_size = 1000.0

        def get_account_value(self):
            return 0.0

        def execute_signal(self, signal, bypass_firewall=False):
            executed.append((signal.coin, signal.side.value, bypass_firewall))
            return {"status": "success", "coin": signal.coin}

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    container = type(
        "Container",
        (),
        {
            "live_trader": FakeLiveTrader(),
            "firewall": FakeFirewall(),
            "agent_scorer": None,
            "options_scanner": type(
                "Scanner",
                (),
                {
                    "top_convictions": [
                        {
                            "ticker": "ETH",
                            "direction": "BULLISH",
                            "conviction_pct": 75,
                            "net_flow": 100000.0,
                            "total_prints": 3,
                        }
                    ]
                },
            )(),
        },
    )()

    monkeypatch.setattr("src.core.cycles.trading_cycle.is_live_trading_active", lambda container: True)
    monkeypatch.setattr("src.core.cycles.trading_cycle.get_execution_open_positions", lambda container: [])
    monkeypatch.setattr("src.data.hyperliquid_client.get_all_mids", lambda: {"ETH": "2000.0"})
    monkeypatch.setattr("src.notifications.telegram_bot.is_configured", lambda: False)

    _execute_options_flow_trades(container, {"overall_regime": "neutral"})

    assert executed == []


def test_execute_options_flow_live_precomputes_size_for_firewall(monkeypatch):
    executed = []
    seen = {}

    class FakeLiveTrader:
        max_position_size = 1000.0

        def get_account_value(self):
            return 20.0

        def execute_signal(self, signal, bypass_firewall=False):
            executed.append((signal.coin, signal.side.value, bypass_firewall))
            return {"status": "success", "coin": signal.coin}

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            seen["size"] = signal.size
            seen["account_balance"] = kwargs.get("account_balance")
            return False, "risk_blocked"

    container = type(
        "Container",
        (),
        {
            "live_trader": FakeLiveTrader(),
            "firewall": FakeFirewall(),
            "agent_scorer": None,
            "options_scanner": type(
                "Scanner",
                (),
                {
                    "top_convictions": [
                        {
                            "ticker": "ETH",
                            "direction": "BULLISH",
                            "conviction_pct": 75,
                            "net_flow": 100000.0,
                            "total_prints": 3,
                        }
                    ]
                },
            )(),
        },
    )()

    monkeypatch.setattr("src.core.cycles.trading_cycle.is_live_trading_active", lambda container: True)
    monkeypatch.setattr("src.core.cycles.trading_cycle.get_execution_open_positions", lambda container: [])
    monkeypatch.setattr("src.data.hyperliquid_client.get_all_mids", lambda: {"ETH": "2000.0"})
    monkeypatch.setattr("src.notifications.telegram_bot.is_configured", lambda: False)

    _execute_options_flow_trades(container, {"overall_regime": "neutral"})

    assert executed == []
    assert seen["account_balance"] == 20.0
    assert abs(seen["size"] - 0.02) < 1e-12


def test_execute_signal_live_logs_warning_for_insufficient_margin(monkeypatch, caplog):
    class FakeLiveTrader:
        def execute_signal(self, signal, bypass_firewall=False):
            return {
                "status": "rejected",
                "reason": "exchange_inner_error",
                "errors": ["Insufficient margin to place order. asset=0"],
            }

    container = type("Container", (), {"live_trader": FakeLiveTrader()})()
    signal = TradeSignal(
        coin="ETH",
        side=SignalSide.LONG,
        confidence=0.8,
        source=SignalSource.STRATEGY,
        reason="test",
    )

    monkeypatch.setattr("src.core.cycles.trading_cycle.is_live_trading_active", lambda container: True)

    with caplog.at_level(logging.WARNING):
        result = _execute_signal_live(container, signal, "OPTIONS FLOW", bypass_firewall=False)

    assert result is None
    assert "insufficient margin" in caplog.text.lower()


def test_mirror_executed_trades_uses_account_value_without_wallet_alias():
    class FakeLiveTrader:
        def __init__(self):
            self.executed = []

        def is_live_enabled(self):
            return True

        def is_deployable(self):
            return True

        def get_account_value(self):
            return 250.0

        def execute_signal(self, signal, bypass_firewall=False):
            self.executed.append((signal.coin, signal.side.value, bypass_firewall))
            return {"status": "success"}

    trader = FakeLiveTrader()
    container = type("Container", (), {"live_trader": trader})()
    executed = [
        signal_from_execution_dict(
            {
                "coin": "ETH",
                "side": "long",
                "confidence": 0.7,
                "entry_price": 2000.0,
                "size": 0.01,
                "leverage": 2,
                "strategy_type": "mirror_test",
            }
        )
    ]

    mirror_executed_trades_to_live(
        container,
        executed,
        success_label="LIVE",
        skip_label="SKIP",
    )

    assert trader.executed == [("ETH", "long", True)]


def test_mirror_executed_trades_logs_warning_for_insufficient_margin(monkeypatch, caplog):
    class FakeLiveTrader:
        def __init__(self):
            self.executed = []
            self.max_order_usd = 25.0
            self.min_order_usd = 11.0

        def is_live_enabled(self):
            return True

        def is_deployable(self):
            return True

        def get_account_value(self):
            return 250.0

        def execute_signal(self, signal, bypass_firewall=False):
            self.executed.append((signal.coin, signal.side.value, bypass_firewall))
            return {
                "status": "rejected",
                "reason": "exchange_inner_error",
                "errors": ["Insufficient margin to place order. asset=4"],
            }

    monkeypatch.setattr("src.core.live_execution.db.get_paper_account", lambda: {"balance": 10000.0})
    monkeypatch.setattr("src.core.live_execution.get_all_mids", lambda: {"DYDX": "0.10403"})

    trader = FakeLiveTrader()
    container = type("Container", (), {"live_trader": trader})()
    executed = [
        signal_from_execution_dict(
            {
                "coin": "DYDX",
                "side": "long",
                "confidence": 0.8,
                "entry_price": 0.10403,
                "size": 100.0,
                "leverage": 5,
                "strategy_type": "mirror_test",
            }
        )
    ]

    with caplog.at_level(logging.WARNING):
        mirror_executed_trades_to_live(
            container,
            executed,
            success_label="LIVE COPY",
            skip_label="SKIP",
        )

    assert trader.executed == [("DYDX", "long", True)]
    assert "insufficient margin" in caplog.text.lower()
    assert "live copy failed" not in caplog.text.lower()


def test_execute_options_flow_paper_trade_preserves_precise_stops(monkeypatch):
    opened = {}
    price = 0.01234567

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    container = type(
        "Container",
        (),
        {
            "live_trader": None,
            "firewall": FakeFirewall(),
            "agent_scorer": None,
            "options_scanner": type(
                "Scanner",
                (),
                {
                    "top_convictions": [
                        {
                            "ticker": "FART",
                            "direction": "BULLISH",
                            "conviction_pct": 80,
                            "net_flow": 50000.0,
                            "total_prints": 2,
                        }
                    ]
                },
            )(),
        },
    )()

    monkeypatch.setattr("src.core.cycles.trading_cycle.is_live_trading_active", lambda container: False)
    monkeypatch.setattr("src.core.cycles.trading_cycle.get_execution_open_positions", lambda container: [])
    monkeypatch.setattr("src.data.hyperliquid_client.get_all_mids", lambda: {"FART": str(price)})
    monkeypatch.setattr("src.core.cycles.trading_cycle.db.get_paper_account", lambda: {"balance": 10000.0})
    monkeypatch.setattr(
        "src.core.cycles.trading_cycle.db.open_paper_trade",
        lambda **kwargs: opened.update(kwargs) or 1,
    )
    monkeypatch.setattr("src.notifications.telegram_bot.is_configured", lambda: False)

    _execute_options_flow_trades(container, {"overall_regime": "neutral"})

    assert abs(opened["stop_loss"] - (price * 0.95)) < 1e-12
    assert abs(opened["take_profit"] - (price * 1.10)) < 1e-12


def test_run_alpha_arena_live_path_executes_signal(monkeypatch):
    executed = []

    class FakeResponse:
        status_code = 200

        def json(self):
            return [
                {"o": "1.0", "h": "1.1", "l": "0.9", "c": "1.0", "v": "10"}
                for _ in range(60)
            ]

    class FakeLiveTrader:
        def execute_signal(self, signal, bypass_firewall=False):
            executed.append((signal.coin, signal.side.value, bypass_firewall, signal.position_pct))
            return {"status": "success", "coin": signal.coin}

    class FakeArena:
        def run_cycle(self, historical_candles=None):
            return None

        def get_stats(self):
            return {"active_agents": 1, "champions": 1, "total_arena_pnl": 0.0}

        def get_champion_signals(self, **kwargs):
            return [
                {
                    "coin": "FART",
                    "side": "long",
                    "confidence": 0.6,
                    "price": 0.01234567,
                    "agent_id": "a1",
                    "agent_name": "alpha",
                    "strategy_type": "arena_breakout",
                    "agent_fitness": 1.2,
                    "agent_elo": 1100,
                }
            ]

    container = type(
        "Container",
        (),
        {
            "arena": FakeArena(),
            "live_trader": FakeLiveTrader(),
        },
    )()

    monkeypatch.setattr("src.core.cycles.trading_cycle.is_live_trading_active", lambda container: True)

    # Mock the API manager to return candle data (arena now uses get_manager().post)
    fake_candles = [
        {"o": "1.0", "h": "1.1", "l": "0.9", "c": "1.0", "v": "10"}
        for _ in range(60)
    ]

    class FakeManager:
        def post(self, **kwargs):
            return fake_candles

    monkeypatch.setattr("src.core.api_manager.get_manager", lambda: FakeManager())

    _run_alpha_arena(container, {"overall_regime": "neutral"})

    assert executed == [("FART", "long", True, 0.03)]


def test_run_alpha_arena_paper_trade_preserves_precise_stops(monkeypatch):
    opened = {}
    price = 0.01234567

    class FakeResponse:
        status_code = 200

        def json(self):
            return [
                {"o": "1.0", "h": "1.1", "l": "0.9", "c": "1.0", "v": "10"}
                for _ in range(60)
            ]

    class FakeArena:
        def run_cycle(self, historical_candles=None):
            return None

        def get_stats(self):
            return {"active_agents": 1, "champions": 1, "total_arena_pnl": 0.0}

        def get_champion_signals(self, **kwargs):
            return [
                {
                    "coin": "FART",
                    "side": "short",
                    "confidence": 0.6,
                    "price": price,
                    "agent_id": "a1",
                    "agent_name": "alpha",
                    "strategy_type": "arena_breakout",
                    "agent_fitness": 1.2,
                    "agent_elo": 1100,
                }
            ]

    container = type(
        "Container",
        (),
        {
            "arena": FakeArena(),
            "live_trader": None,
        },
    )()

    monkeypatch.setattr("src.core.cycles.trading_cycle.is_live_trading_active", lambda container: False)
    monkeypatch.setattr("src.core.cycles.trading_cycle.db.get_paper_account", lambda: {"balance": 10000.0})
    monkeypatch.setattr(
        "src.core.cycles.trading_cycle.db.open_paper_trade",
        lambda **kwargs: opened.update(kwargs) or 1,
    )

    # Mock API manager (arena now uses get_manager().post instead of requests.post)
    fake_candles = [
        {"o": str(price), "h": str(price * 1.1), "l": str(price * 0.9),
         "c": str(price), "v": "10"} for _ in range(60)
    ]

    class FakeManager:
        def post(self, **kwargs):
            return fake_candles

    monkeypatch.setattr("src.core.api_manager.get_manager", lambda: FakeManager())

    _run_alpha_arena(container, {"overall_regime": "neutral"})

    assert abs(opened["stop_loss"] - (price * 1.05)) < 1e-12
    assert abs(opened["take_profit"] - (price * 0.90)) < 1e-12


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
    metadata_updates = []
    closed = []

    class FakeLiveTrader:
        def is_live_enabled(self):
            return True

        def is_deployable(self):
            return True

        def get_positions(self):
            return []

        def get_account_value(self):
            return 500.0  # Funded account — reconciliation should proceed

    container = type(
        "Container",
        (),
        {"live_trader": FakeLiveTrader(), "paper_trader": object()},
    )()

    monkeypatch.setattr(
        "src.core.live_execution.db.get_open_paper_trades",
        lambda: [{"id": 7, "coin": "ETH", "side": "long", "entry_price": 2000.0}],
    )
    monkeypatch.setattr(
        "src.core.live_execution.get_all_mids",
        lambda: {"ETH": 2100.0},
    )
    monkeypatch.setattr(
        "src.core.live_execution.db.update_paper_trade_metadata",
        lambda trade_id, meta: metadata_updates.append((trade_id, meta)),
    )
    monkeypatch.setattr(
        "src.core.live_execution.db.close_paper_trade",
        lambda trade_id, exit_price, pnl: closed.append((trade_id, exit_price, pnl)) or True,
    )

    reconciled = sync_shadow_book_to_live(container)
    assert len(reconciled) == 1
    assert reconciled[0]["trade_id"] == 7
    assert reconciled[0]["coin"] == "ETH"
    assert reconciled[0]["reason"] == "live_reconciled_closed"
    assert reconciled[0]["pnl"] == 0.0
    assert closed == [(7, 2100.0, 0.0)]
    assert metadata_updates[0][0] == 7
    assert metadata_updates[0][1]["synthetic_reconciliation"] is True
    assert metadata_updates[0][1]["reconciliation_reason"] == "live_reconciled_closed"


def test_process_closed_trades_skips_synthetic_reconciliation():
    class FakeArena:
        def __init__(self):
            self.calls = []

        def record_trade_for_strategy(self, *args):
            self.calls.append(args)

    class FakeKelly:
        def __init__(self):
            self.calls = []

        def record_outcome(self, **kwargs):
            self.calls.append(kwargs)

    class FakeAgentScorer:
        def __init__(self):
            self.calls = []

        def record_outcome(self, *args):
            self.calls.append(args)

    container = type(
        "Container",
        (),
        {
            "arena": FakeArena(),
            "kelly_sizer": FakeKelly(),
            "agent_scorer": FakeAgentScorer(),
            "shadow_tracker": None,
        },
    )()

    _process_closed_trades(
        container,
        [
            {
                "trade_id": 7,
                "coin": "ETH",
                "side": "long",
                "entry_price": 2000.0,
                "exit_price": 2100.0,
                "size": 0.1,
                "leverage": 2,
                "pnl": 0.0,
                "strategy_type": "momentum_long",
                "reason": "live_reconciled_closed",
                "metadata": {"synthetic_reconciliation": True},
            }
        ],
    )

    assert container.arena.calls == []
    assert container.kelly_sizer.calls == []
    assert container.agent_scorer.calls == []


def test_firewall_uses_explicit_live_positions_without_falling_back(monkeypatch):
    monkeypatch.setattr(
        "src.signals.decision_firewall.db.get_open_paper_trades",
        lambda: [{"coin": "BTC"}] * 10,
    )
    monkeypatch.setattr(
        "src.signals.decision_firewall.db.audit_log",
        lambda **kwargs: None,
    )

    firewall = DecisionFirewall(
        {
            "max_positions": 1,
            "funding_risk_enabled": False,
            "enable_predictive_derisk": False,
        }
    )
    signal = signal_from_execution_dict(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.1,
        }
    )

    passed, reason = firewall.validate(signal, open_positions=[], account_balance=5000.0)

    assert passed is True
    assert reason == "approved"


def test_firewall_daily_drawdown_uses_provided_account_balance(monkeypatch):
    monkeypatch.setattr(
        "src.signals.decision_firewall.db.audit_log",
        lambda **kwargs: None,
    )

    firewall = DecisionFirewall(
        {
            "daily_loss_limit_pct": 0.03,
            "funding_risk_enabled": False,
            "enable_predictive_derisk": False,
        }
    )
    firewall._daily_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    firewall.set_daily_losses(40.0)
    signal = signal_from_execution_dict(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.1,
        }
    )

    passed, reason = firewall.validate(signal, open_positions=[], account_balance=1000.0)

    assert passed is False
    assert "Daily loss limit hit" in reason


def test_positive_daily_pnl_does_not_trigger_kill_switch(monkeypatch):
    monkeypatch.setattr(LiveTrader, "_load_credentials", lambda self: None)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)

    trader = LiveTrader(
        firewall=DecisionFirewall(
            {
                "funding_risk_enabled": False,
                "enable_predictive_derisk": False,
            }
        ),
        dry_run=True,
    )
    trader.daily_pnl = 125.0

    assert trader.check_daily_loss() is False
    assert trader.kill_switch_active is False


def test_hyperliquid_signer_zero_pads_signature_components(monkeypatch):
    class FakeAccount:
        def sign_message(self, message):
            return type("Signed", (), {"r": 0x1234, "s": 0xABCD, "v": 27})()

    signer = HyperliquidSigner.__new__(HyperliquidSigner)
    signer.account = FakeAccount()

    # New signer uses eth_account.encode_typed_data (imported as
    # _encode_typed_data).  Replace it with a no-op so we do not need a
    # real eth_account install to verify zero-padding.
    monkeypatch.setattr(
        "src.trading.live_trader._encode_typed_data",
        lambda *args, **kwargs: object(),
        raising=False,
    )

    signature = signer.sign_action({"type": "noop"}, nonce=123)

    assert signature["r"].startswith("0x")
    assert signature["s"].startswith("0x")
    assert len(signature["r"]) == 66
    assert len(signature["s"]) == 66
    assert signature["r"].endswith("1234")
    assert signature["s"].endswith("abcd")


def test_hyperliquid_signer_includes_vault_address_in_hash():
    """Regression test: the old signer ignored the vault address when
    computing the signed hash, so agent-wallet orders signed one hash and
    Hyperliquid reconstructed a different one — ecrecover then returned a
    mystery address and the exchange replied "User or API Wallet 0x… does
    not exist".  The fix includes ``vault_address`` in the keccak
    preimage, so the same action + nonce with different vaults MUST
    produce different hashes.
    """
    action = {
        "type": "order",
        "orders": [{"a": 0, "b": True, "p": "50000", "s": "0.001", "r": False,
                    "t": {"limit": {"tif": "Ioc"}}}],
        "grouping": "na",
    }
    nonce = 1_700_000_000_000

    no_vault_hash = HyperliquidSigner._action_hash(action, None, nonce)
    with_vault_hash = HyperliquidSigner._action_hash(
        action, "0x1234567890abcdef1234567890abcdef12345678", nonce,
    )
    other_vault_hash = HyperliquidSigner._action_hash(
        action, "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd", nonce,
    )

    assert len(no_vault_hash) == 32
    assert len(with_vault_hash) == 32
    assert no_vault_hash != with_vault_hash
    assert with_vault_hash != other_vault_hash


def test_hyperliquid_signer_uses_agent_primary_type_and_exchange_domain():
    """Regression test: the signing payload must use ``primaryType='Agent'``
    with ``domain.name='Exchange'`` (Hyperliquid's real scheme), not the
    legacy ``HyperliquidTransaction`` struct that produced garbage
    ecrecover addresses.
    """
    captured = {}

    class FakeAccount:
        def sign_message(self, message):
            return type("Signed", (), {"r": 1, "s": 2, "v": 27})()

    def fake_encode(*args, **kwargs):
        captured["payload"] = kwargs.get("full_message") or (args[0] if args else None)
        return object()

    signer = HyperliquidSigner.__new__(HyperliquidSigner)
    signer.account = FakeAccount()

    import src.trading.live_trader as lt
    original = lt._encode_typed_data
    try:
        lt._encode_typed_data = fake_encode
        signer.sign_action({"type": "noop"}, nonce=1, vault_address=None)
    finally:
        lt._encode_typed_data = original

    payload = captured["payload"]
    assert payload is not None, "encode_typed_data was not called"
    assert payload["primaryType"] == "Agent"
    assert payload["domain"]["name"] == "Exchange"
    assert payload["domain"]["chainId"] == 1337
    assert payload["domain"]["verifyingContract"] == "0x" + "0" * 40
    assert "Agent" in payload["types"]
    assert {"name": "source", "type": "string"} in payload["types"]["Agent"]
    assert {"name": "connectionId", "type": "bytes32"} in payload["types"]["Agent"]
    assert payload["message"]["source"] == "a"
    assert isinstance(payload["message"]["connectionId"], (bytes, bytearray))
    assert len(payload["message"]["connectionId"]) == 32


def test_rescale_size_for_live_blocks_when_paper_balance_missing(monkeypatch):
    class FakeTrader:
        def get_account_value(self):
            return 2500.0

    monkeypatch.setattr("src.core.live_execution.db.get_paper_account", lambda: {"balance": 0})

    scaled = _rescale_size_for_live({"coin": "ETH", "size": 0.2}, FakeTrader())

    assert scaled is None


def test_rescale_size_for_live_enforces_max_order_usd_cap(monkeypatch):
    """After rescaling, notional must never exceed trader.max_order_usd."""
    class FakeTrader:
        max_order_usd = 3.0

        def get_account_value(self):
            return 10_000.0  # equal to paper — scale = 1.0

    monkeypatch.setattr("src.core.live_execution.db.get_paper_account", lambda: {"balance": 10_000.0})
    monkeypatch.setattr("src.core.live_execution.get_all_mids", lambda: {"ETH": 2000.0})

    # 0.5 ETH @ $2000 = $1000 notional, should be capped to $3 → 0.0015 ETH
    scaled = _rescale_size_for_live(
        {"coin": "ETH", "size": 0.5, "entry_price": 2000.0},
        FakeTrader(),
    )
    assert scaled is not None
    assert abs(scaled["size"] * 2000.0 - 3.0) < 1e-9


def test_rescale_size_for_live_floors_up_to_exchange_minimum(monkeypatch):
    """Small live wallet vs large paper book: proportional rescaling
    produces sub-minimum orders that Hyperliquid silently drops.  The
    rescaler should floor UP to the exchange minimum rather than skip,
    so the bot actually trades.  Regression test for the XRP copy-mirror
    scenario where 0.172 XRP @ $1.24 = $0.21 notional got skipped.
    """
    class FakeTrader:
        max_order_usd = 12.0
        min_order_usd = 11.0

        def get_account_value(self):
            return 12.44  # the real problem case

    monkeypatch.setattr("src.core.live_execution.db.get_paper_account",
                        lambda: {"balance": 10_041.0})
    monkeypatch.setattr("src.core.live_execution.get_all_mids",
                        lambda: {"XRP": 1.24})

    # 138.7 XRP × $1.24 = $172 paper notional, rescales to 0.172 XRP
    # (= $0.21) at scale factor 0.00124.  Should floor UP to ~$11 notional.
    scaled = _rescale_size_for_live(
        {"coin": "XRP", "size": 138.7, "entry_price": 1.24, "leverage": 5},
        FakeTrader(),
    )
    assert scaled is not None, "floor-up should succeed, not skip"
    notional = scaled["size"] * 1.24
    assert notional >= 11.0, f"notional ${notional:.2f} should be >= $11 min"
    assert notional <= 12.0, f"notional ${notional:.2f} should be <= $12 cap"


def test_rescale_size_for_live_skips_when_margin_exceeds_wallet_headroom(monkeypatch):
    """If even the minimum order's margin requirement exceeds 80% of the
    live wallet, skip instead of flooring up — the wallet is too small
    to safely carry any position at the configured leverage."""
    class FakeTrader:
        max_order_usd = 11.0
        min_order_usd = 11.0

        def get_account_value(self):
            return 5.0  # wallet too small for $11 notional at 1x

    monkeypatch.setattr("src.core.live_execution.db.get_paper_account",
                        lambda: {"balance": 10_000.0})
    monkeypatch.setattr("src.core.live_execution.get_all_mids",
                        lambda: {"BTC": 50_000.0})

    # 0.01 BTC × $50k = $500 paper, rescales to 0.00001 BTC = $0.25.
    # Min order $11 at 1x leverage = $11 margin > 80% of $5 wallet ($4).
    scaled = _rescale_size_for_live(
        {"coin": "BTC", "size": 0.01, "entry_price": 50_000.0, "leverage": 1},
        FakeTrader(),
    )
    assert scaled is None, "should skip: margin too large for wallet"


def test_execute_signal_bypass_firewall_skips_validation(monkeypatch):
    """When bypass_firewall=True, DecisionFirewall.validate must NOT be called."""
    validate_calls = []

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            validate_calls.append(signal.coin)
            return False, "Cooldown: ETH traded 58s ago"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(LiveTrader, "place_market_order", lambda self, *args, **kwargs: {"status": "success"})
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(LiveTrader, "verify_fill", lambda self, *args, **kwargs: {"status": "verified"})
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(LiveTrader, "place_trigger_order", lambda self, *args, **kwargs: {"status": "success"})

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)

    # Without bypass: firewall rejects, result is None
    r_strict = trader.execute_signal(
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
    assert r_strict is None
    assert validate_calls == ["ETH"]

    # With bypass: firewall is NOT called, trade proceeds
    validate_calls.clear()
    r_bypass = trader.execute_signal(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.1,
            "strategy_type": "momentum_long",
        },
        bypass_firewall=True,
    )
    assert r_bypass is not None
    assert r_bypass["status"] == "success"
    assert validate_calls == []


def test_execute_signal_caps_notional_at_max_order_usd(monkeypatch):
    """A large computed size must be shrunk so notional <= max_order_usd."""
    placed_sizes = []

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(
        LiveTrader,
        "place_market_order",
        lambda self, coin, side, size, leverage=1, reduce_only=False: (
            placed_sizes.append(size) or {"status": "success"}
        ),
    )
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(LiveTrader, "verify_fill", lambda self, *args, **kwargs: {"status": "verified"})
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(LiveTrader, "place_trigger_order", lambda self, *args, **kwargs: {"status": "success"})

    # Use a cap above the $10 exchange minimum so the test is not fighting
    # LiveTrader's min-notional auto-raise.  $15 is comfortably above the
    # $11 default min and proves the cap logic still shrinks large orders.
    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=15.0)
    result = trader.execute_signal(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.5,  # $1000 notional — should be capped to $15 → 0.0075 ETH
            "strategy_type": "momentum_long",
        }
    )

    assert result is not None
    assert result["status"] == "success"
    assert len(placed_sizes) == 1
    assert abs(placed_sizes[0] * 2000.0 - 15.0) < 1e-9


def test_live_trader_raises_cap_to_exchange_minimum(monkeypatch):
    """Regression test: LIVE_MAX_ORDER_USD below Hyperliquid's $10 minimum
    notional makes every live order silently fail (matching engine drops
    it, then fill verification times out as "FILL NOT VERIFIED").  The
    LiveTrader should raise the cap to the exchange minimum at startup
    with a warning rather than allow the impossible state."""
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=3.0)

    # Cap was below the $10 min — should have been raised to min (default 11)
    assert trader.max_order_usd >= 10.0
    assert trader.max_order_usd == trader.min_order_usd


def test_execute_signal_rejects_below_exchange_minimum(monkeypatch):
    """An order whose rescaled notional is under the exchange minimum
    must be rejected up front, not sent and then fail fill verification."""
    placed_sizes = []

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 20.0)
    monkeypatch.setattr(
        LiveTrader,
        "place_market_order",
        lambda self, coin, side, size, leverage=1, reduce_only=False: (
            placed_sizes.append(size) or {"status": "success"}
        ),
    )
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(LiveTrader, "verify_fill", lambda self, *a, **kw: {"status": "verified"})
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2.5)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=12.0)
    # Force a tiny order whose notional is well below $11: 0.172 * $2.50 = $0.43
    result = trader.execute_signal(
        {
            "coin": "XRP",
            "side": "short",
            "confidence": 0.8,
            "entry_price": 2.5,
            "position_pct": 0.01,
            "leverage": 5,
            "size": 0.172,
            "strategy_type": "momentum_short",
        }
    )

    # Should be rejected before hitting place_market_order
    assert result is None or result.get("status") in {"rejected", "error"}
    assert placed_sizes == []


def test_copy_trade_preserves_precise_stops_for_low_priced_assets(monkeypatch):
    opened = {}

    def fake_open_paper_trade(**kwargs):
        opened.update(kwargs)
        return 321

    monkeypatch.setitem(sys.modules, "numpy", types.SimpleNamespace(std=lambda values: 0.0))
    monkeypatch.setattr("src.trading.copy_trader.db.open_paper_trade", fake_open_paper_trade)

    from src.trading.copy_trader import CopyTrader

    trader = CopyTrader()
    price = 0.01234567
    leverage = 4
    trade = trader._open_copy_trade(
        {"balance": 10000.0},
        {
            "coin": "FART",
            "side": "long",
            "price": price,
            "leverage": leverage,
            "type": "copy_trade_signal",
            "confidence": 0.8,
            "source_trader": "0xabc",
        },
        [],
    )

    expected_stop_loss = price * (1 - 0.04 / leverage)
    expected_take_profit = price * (1 + 0.08 / leverage)

    assert trade is not None
    assert abs(opened["stop_loss"] - expected_stop_loss) < 1e-12
    assert abs(opened["take_profit"] - expected_take_profit) < 1e-12
    assert abs(trade["stop_loss"] - expected_stop_loss) < 1e-12
    assert abs(trade["take_profit"] - expected_take_profit) < 1e-12

    signal = signal_from_execution_dict(trade)
    assert abs(signal.risk.stop_loss_pct - (0.04 / leverage)) < 1e-12
    assert abs(signal.risk.take_profit_pct - (0.08 / leverage)) < 1e-12


def test_discovery_time_persistence_round_trips_through_db_context(monkeypatch):
    fd, raw_path = tempfile.mkstemp(prefix="bot_state_", suffix=".db", dir=os.getcwd())
    os.close(fd)
    db_path = os.path.abspath(raw_path)

    @contextmanager
    def fake_get_connection():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    monkeypatch.setattr(main.db, "get_connection", fake_get_connection)

    try:
        bot = main.HyperliquidResearchBot.__new__(main.HyperliquidResearchBot)
        bot._last_discovery = 12345.67

        bot._save_last_discovery_time()
        restored = bot._restore_last_discovery_time()

        assert restored == 12345.67
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


# ────────────────────────────────────────────────────────────────────────
# Hyperliquid wire-format regression tests
# ────────────────────────────────────────────────────────────────────────
#
# These lock in the exact price/size string encoding the exchange
# requires.  Violating the 5-sig-fig + (6 - szDecimals) decimal rules
# causes Hyperliquid to return
# ``{"status": "ok", "response": {..., "statuses": [{"error": "Order has
# invalid price."}]}}`` — the order never fills but looks accepted.

def test_hl_format_price_respects_five_sig_fig_limit():
    # HYPE at szDecimals=2 lets up to 4 decimals, but sig-fig cap is 5.
    # 33.95015 has 7 sig figs -> must round to 33.95.
    assert _hl_format_price(33.95015, 2) == "33.95"


def test_hl_format_price_strips_trailing_zeros():
    assert _hl_format_price(33.95, 2) == "33.95"
    assert _hl_format_price(0.23, 0) == "0.23"
    assert _hl_format_price(2163.8, 4) == "2163.8"


def test_hl_format_price_caps_high_sig_figs_for_large_numbers():
    # BTC at $67544.12345 -> 5 sig figs -> 67544
    assert _hl_format_price(67544.12345, 5) == "67544"


def test_hl_format_price_respects_szdecimals_constraint():
    # ETH szDecimals=4 -> max 2 decimal places (6-4).  2163.79123 -> 2163.8
    # (5 sig figs clamps first, then decimal limit).
    assert _hl_format_price(2163.79123, 4) == "2163.8"
    # BTC szDecimals=5 -> max 1 decimal place.  67544.12 would need to
    # be rounded to integer (1 decimal would be 6th sig fig).
    assert _hl_format_price(67544.12, 5) == "67544"


def test_hl_format_price_handles_non_finite_and_non_positive():
    assert _hl_format_price(0, 2) == "0"
    assert _hl_format_price(-1, 2) == "0"
    assert _hl_format_price(float("nan"), 2) == "0"
    assert _hl_format_price(float("inf"), 2) == "0"


def test_hl_format_size_rounds_to_sz_decimals():
    assert _hl_format_size(0.33580053, 2) == "0.34"
    assert _hl_format_size(49.27625500, 0) == "49"
    assert _hl_format_size(0.00017122, 5) == "0.00017"


def test_hl_format_size_strips_trailing_zeros():
    assert _hl_format_size(1.0, 4) == "1"
    assert _hl_format_size(1.5000, 4) == "1.5"
    assert _hl_format_size(1.50100, 4) == "1.501"


def test_is_order_result_success_detects_inner_error():
    """An outer ``status: ok`` with per-order ``error`` inside must be
    classified as failure, otherwise verify_fill waits pointlessly for a
    fill that will never come and logs FILL NOT VERIFIED even though the
    exchange actually rejected the order."""
    rejected_response = {
        "status": "ok",
        "response": {
            "type": "order",
            "data": {"statuses": [{"error": "Order has invalid price."}]},
        },
    }
    assert LiveTrader._is_order_result_success(rejected_response) is False


def test_is_order_result_success_allows_resting_and_filled():
    resting = {
        "status": "ok",
        "response": {
            "type": "order",
            "data": {"statuses": [{"resting": {"oid": 12345}}]},
        },
    }
    filled = {
        "status": "ok",
        "response": {
            "type": "order",
            "data": {
                "statuses": [
                    {"filled": {"oid": 12345, "totalSz": "1.0", "avgPx": "50000"}}
                ]
            },
        },
    }
    assert LiveTrader._is_order_result_success(resting) is True
    assert LiveTrader._is_order_result_success(filled) is True


def test_extract_inner_order_statuses_handles_missing_shape():
    assert LiveTrader._extract_inner_order_statuses(None) == []
    assert LiveTrader._extract_inner_order_statuses({}) == []
    assert LiveTrader._extract_inner_order_statuses({"status": "err"}) == []
    assert LiveTrader._extract_inner_order_statuses(
        {"status": "ok", "response": "not a dict"}
    ) == []


def test_extract_reported_fill_size_sums_filled_statuses():
    result = {
        "status": "ok",
        "response": {
            "type": "order",
            "data": {
                "statuses": [
                    {"filled": {"oid": 1, "totalSz": "1.0", "avgPx": "0.030813"}},
                    {"resting": {"oid": 2}},
                    {"filled": {"oid": 3, "totalSz": "2.5", "avgPx": "0.030900"}},
                ]
            },
        },
    }
    assert LiveTrader._extract_reported_fill_size(result) == 3.5
    assert LiveTrader._extract_reported_fill_size({"status": "ok"}) is None


def test_place_market_order_uses_wire_format_price_and_size(monkeypatch):
    """End-to-end: place_market_order must format price/size through
    _hl_format_* so the payload sent to _post_order has canonical strings.
    Regression test for the HYPE/TAO/ADA/ETH "Order has invalid price"
    burst.
    """
    captured_actions = []

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 33.95015)
    monkeypatch.setattr(
        LiveTrader,
        "_post_order",
        lambda self, action, dry_run_override=None: (
            captured_actions.append(action) or {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"resting": {"oid": 1}}]}}}
        ),
    )

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=15.0)
    trader.asset_index_map = {"HYPE": 42}
    trader.sz_decimals_map = {"HYPE": 2}

    result = trader.place_market_order("HYPE", "sell", 0.33580053448251734)
    assert result.get("status") == "ok"
    assert len(captured_actions) == 1
    order = captured_actions[0]["orders"][0]
    # Price must be canonical (5 sig figs, no trailing zeros).  Mid 33.95015
    # with 5% sell slippage = 32.2526425, rounds to 32.253 at 5 sig figs.
    assert order["p"] == "32.253", f"expected canonical price, got {order['p']!r}"
    # Size 0.33580053 at post-slippage price 32.2526425 = $10.83 notional,
    # which is below the $11 minimum — the floor-up logic inside
    # place_market_order bumps it to ~$11.22 / $32.2526425 ≈ 0.3479,
    # which rounds to "0.35" at szDecimals=2.  Critically the string is
    # canonical (no trailing zeros, 2 decimal places max).
    assert order["s"] == "0.35", f"expected canonical size, got {order['s']!r}"
    # Critical: NO trailing zeros, NO sig-fig overflow
    assert "." not in order["p"] or not order["p"].endswith("0")
    assert "." not in order["s"] or not order["s"].endswith("0")


def test_place_market_order_reports_submitted_and_exchange_fill_size(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 0.0308)
    monkeypatch.setattr(
        LiveTrader,
        "_post_order",
        lambda self, action, dry_run_override=None: {
            "status": "ok",
            "response": {
                "type": "order",
                "data": {
                    "statuses": [
                        {"filled": {"oid": 123, "totalSz": "1.0", "avgPx": "0.030813"}}
                    ]
                },
            },
        },
    )

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=12.0)
    trader.asset_index_map = {"KAS": 7}
    trader.sz_decimals_map = {"KAS": 0}

    result = trader.place_market_order("KAS", "buy", 389.7116134060795)

    assert result["submitted_size"] > 0
    assert result["submitted_size"] < result["requested_size"]
    assert result["submitted_notional"] <= 12.0
    assert result["exchange_reported_fill_size"] == 1.0
    assert result["wire_size"] == "371"


def test_post_order_promotes_inner_error_to_rejection(monkeypatch):
    """When Hyperliquid returns an outer ok with an inner error, _post_order
    must return a ``status: rejected`` dict so downstream callers stop
    treating it as success and waiting for a phantom fill."""

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "status": "ok",
                "response": {
                    "type": "order",
                    "data": {"statuses": [{"error": "Order has invalid price."}]},
                },
            }

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr("src.trading.live_trader.requests.post", lambda *a, **kw: FakeResponse())

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=15.0)

    # Give the trader a fake signer so _post_order reaches the network
    class FakeSigner:
        address = "0x1111111111111111111111111111111111111111"

        def sign_action(self, action, nonce, vault_address=None, expires_after=None):
            return {"r": "0x" + "0" * 64, "s": "0x" + "0" * 64, "v": 27}

    trader.signer = FakeSigner()
    trader.dry_run = False

    result = trader._post_order({"type": "order", "orders": [], "grouping": "na"})
    assert result.get("status") == "rejected"
    assert result.get("reason") == "exchange_inner_error"
    assert "Order has invalid price." in result.get("errors", [])


def test_rescale_size_for_live_uses_mid_price_not_signal_entry(monkeypatch):
    """If the signal's entry_price and the live mid drift apart, the
    rescale floor-up must target mid price so place_market_order (which
    uses mid) doesn't flip us back below the $11 minimum.  Regression
    test for the BTC/XRP below_exchange_minimum burst where entry was
    $67544 but mid was $64152 (~5% drop).
    """
    from src.core.live_execution import _rescale_size_for_live

    class FakeTrader:
        max_order_usd = 12.0
        min_order_usd = 11.0

        def get_account_value(self):
            return 12.44

    monkeypatch.setattr("src.core.live_execution.db.get_paper_account",
                        lambda: {"balance": 10_041.0})
    # Mid price is noticeably LOWER than signal entry
    monkeypatch.setattr("src.core.live_execution.get_all_mids",
                        lambda: {"BTC": 64152.0})

    scaled = _rescale_size_for_live(
        {"coin": "BTC", "size": 0.002679, "entry_price": 67544.0, "leverage": 5},
        FakeTrader(),
    )
    assert scaled is not None
    # Notional measured at the MID price must be above $11 (with headroom)
    notional_at_mid = scaled["size"] * 64152.0
    assert notional_at_mid >= 11.0, (
        f"notional at mid=${notional_at_mid:.2f} should clear $11 min "
        f"(size={scaled['size']})"
    )
    assert notional_at_mid <= 12.5, (
        f"notional at mid=${notional_at_mid:.2f} should stay under cap "
        f"(size={scaled['size']})"
    )


def test_coerce_float_handles_string_int_dict_and_none():
    """Hyperliquid returns numeric fields as strings, nested dicts, or
    None.  Plain float(x) crashes on dicts — the exact failure mode
    that produced 'float() argument must be a string or a real number,
    not dict' 72 times in one log and broke every verify_fill."""
    assert LiveTrader._coerce_float("1.5") == 1.5
    assert LiveTrader._coerce_float("") == 0.0
    assert LiveTrader._coerce_float(None) == 0.0
    assert LiveTrader._coerce_float(None, 99.0) == 99.0
    assert LiveTrader._coerce_float(42) == 42.0
    # The leverage dict shape from clearinghouseState
    assert LiveTrader._coerce_float({"type": "cross", "value": 5}) == 5.0
    assert LiveTrader._coerce_float({"type": "isolated", "value": 10}) == 10.0
    # cumFunding dict shape
    assert LiveTrader._coerce_float({"allTime": "0.123", "sinceOpen": "0.05"}) == 0.123
    # Unknown dict shape falls back to default
    assert LiveTrader._coerce_float({"foo": "bar"}, 77.0) == 77.0
    # Garbage
    assert LiveTrader._coerce_float("not a number", 99.0) == 99.0


def test_normalize_position_handles_leverage_dict():
    """_normalize_position must not crash when leverage is returned as a
    dict (the clearinghouseState wire format).  Regression test for the
    Railway crash chain:
      get_positions() crashed -> verify_fill returned None -> SL/TP
      placement skipped -> real positions left unprotected.
    """
    trader = LiveTrader.__new__(LiveTrader)
    pos_raw = {
        "position": {
            "coin": "ETH",
            "szi": "-0.0058",
            "entryPx": "2064.9",
            "unrealizedPnl": "0.12",
            "leverage": {"type": "cross", "value": 5},
            "positionValue": "11.98",
        },
        "type": "oneWay",
    }
    result = trader._normalize_position(pos_raw)
    assert result["coin"] == "ETH"
    assert result["szi"] == -0.0058
    assert result["size"] == 0.0058
    assert result["side"] == "short"
    assert result["entry_price"] == 2064.9
    assert result["leverage"] == 5.0
    # Also works for isolated leverage
    pos_raw["position"]["leverage"] = {"type": "isolated", "value": 10, "rawUsd": "1.5"}
    assert trader._normalize_position(pos_raw)["leverage"] == 10.0


def test_protect_orphaned_positions_places_sl_tp_for_unprotected(monkeypatch):
    """Positions with no reduce-only orders must get default SL/TP
    placed automatically on the next cycle.  This is the orphan
    recovery path for the crash that left ETH/BTC/HYPE/XRP/RESOLV
    unprotected in the latest Railway log."""
    placed_triggers: list = []

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(
        LiveTrader,
        "get_positions",
        lambda self: [
            {"coin": "ETH", "size": 0.0058, "szi": -0.0058, "side": "short", "entry_price": 2064.9, "entryPx": 2064.9},
            {"coin": "BTC", "size": 0.00018, "szi": -0.00018, "side": "short", "entry_price": 67401.0, "entryPx": 67401.0},
        ],
    )
    monkeypatch.setattr(LiveTrader, "get_open_orders", lambda self: [])  # no protection
    monkeypatch.setattr(
        LiveTrader,
        "place_trigger_order",
        lambda self, coin, side, size, trigger_price, tp_or_sl="sl": (
            placed_triggers.append({"coin": coin, "side": side, "size": size, "price": trigger_price, "type": tp_or_sl})
            or {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"resting": {"oid": len(placed_triggers)}}]}}}
        ),
    )

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=15.0)
    summary = trader.protect_orphaned_positions()

    assert summary["protected"] == 2
    assert summary["failed"] == 0
    assert summary["skipped"] == 0
    assert len(placed_triggers) == 4  # 2 positions × (sl + tp)

    eth_triggers = [t for t in placed_triggers if t["coin"] == "ETH"]
    assert len(eth_triggers) == 2
    # Short position → protection side is BUY
    assert all(t["side"] == "buy" for t in eth_triggers)
    sl = next(t for t in eth_triggers if t["type"] == "sl")
    tp = next(t for t in eth_triggers if t["type"] == "tp")
    # Short SL is ABOVE entry, TP is BELOW entry
    assert sl["price"] > 2064.9
    assert tp["price"] < 2064.9
    # 3% SL, 6% TP
    assert abs(sl["price"] - 2064.9 * 1.03) < 0.01
    assert abs(tp["price"] - 2064.9 * 0.94) < 0.01


def test_protect_orphaned_positions_skips_already_protected(monkeypatch):
    """Positions that already have reduce-only orders should be
    skipped — the method is safe to call on every cycle without
    stacking duplicate SL/TPs."""
    placed_triggers: list = []

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(
        LiveTrader,
        "get_positions",
        lambda self: [
            {"coin": "ETH", "size": 0.0058, "szi": -0.0058, "side": "short", "entry_price": 2064.9, "entryPx": 2064.9},
        ],
    )
    monkeypatch.setattr(
        LiveTrader,
        "get_open_orders",
        lambda self: [
            {"coin": "ETH", "reduceOnly": True, "orderType": "Stop Market"},
            {"coin": "ETH", "reduceOnly": True, "orderType": "Take Profit Market"},
        ],
    )
    monkeypatch.setattr(
        LiveTrader,
        "place_trigger_order",
        lambda self, *a, **kw: placed_triggers.append(a) or {"status": "ok"},
    )

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=15.0)
    summary = trader.protect_orphaned_positions()

    assert summary["protected"] == 0
    assert summary["skipped"] == 1
    assert summary["failed"] == 0
    assert len(placed_triggers) == 0  # nothing placed because already protected


def test_rescale_size_for_live_allows_1x_leverage_on_tight_wallet(monkeypatch):
    """1x leverage orders should not be blocked on a small wallet when
    the wallet actually has enough headroom.  Regression test for the
    KAS skip: $12.42 wallet at 1x leverage, $12 max order, $11 min.
    Old code checked min_order_usd > wallet * 0.80 = $9.94 and skipped;
    new code uses wallet * 0.95 * leverage = $11.80 as the cap and
    targets min(max_order_usd, headroom_target, budget) instead.
    """
    from src.core.live_execution import _rescale_size_for_live

    class FakeTrader:
        max_order_usd = 12.0
        min_order_usd = 11.0

        def get_account_value(self):
            return 12.42

    monkeypatch.setattr("src.core.live_execution.db.get_paper_account",
                        lambda: {"balance": 10_041.0})
    monkeypatch.setattr("src.core.live_execution.get_all_mids",
                        lambda: {"KAS": 0.1025})

    # 5846 KAS at paper, rescales to 7.23 KAS, way below $11 min
    scaled = _rescale_size_for_live(
        {"coin": "KAS", "size": 5846.43, "entry_price": 0.1025, "leverage": 1},
        FakeTrader(),
    )
    assert scaled is not None, "1x leverage should NOT skip when wallet fits"
    notional = scaled["size"] * 0.1025
    assert notional >= 11.0, f"notional ${notional:.2f} should clear $11 min"
    assert notional <= 12.0, f"notional ${notional:.2f} should stay ≤ max_order_usd"


def test_place_trigger_order_sends_wire_format_limit_and_trigger_price(monkeypatch):
    """Regression for the orphan-protection burst that left BTC/ETH/SOL/XRP/
    HYPE/RESOLV unprotected: place_trigger_order was sending ``p="0"`` which
    Hyperliquid rejects with "Order has invalid price" under a ``status: ok``
    wrapper.  The ``p`` field must be the slippage-padded limit cap in
    canonical wire format, and triggerPx must be canonical too.
    """
    captured_actions = []

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(
        LiveTrader,
        "_post_order",
        lambda self, action, dry_run_override=None: (
            captured_actions.append(action)
            or {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"resting": {"oid": 1}}]}}}
        ),
    )

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=15.0)
    trader.asset_index_map = {"ETH": 1, "BTC": 0}
    trader.sz_decimals_map = {"ETH": 4, "BTC": 5}

    # SHORT ETH stop loss: close with BUY at 2123.14 (3% above entry 2061.3).
    # p must be > trigger (buy direction) → 2123.14 * 1.05 = 2229.297, rounds
    # to 2229.3 at 5 sig figs.
    result = trader.place_trigger_order("ETH", "buy", 0.0174, 2123.139, tp_or_sl="sl")
    assert result.get("status") == "ok"
    assert len(captured_actions) == 1
    order = captured_actions[0]["orders"][0]

    # p must NOT be "0" — that was the bug
    assert order["p"] != "0", "place_trigger_order must not send p=0"
    # p is slippage-padded ABOVE trigger for a buy (closing a short)
    assert float(order["p"]) > 2123.139
    # p should be ~5% above trigger
    assert abs(float(order["p"]) - 2123.139 * 1.05) / (2123.139 * 1.05) < 0.01
    # triggerPx is canonical wire format
    assert order["t"]["trigger"]["triggerPx"] != "0"
    assert order["t"]["trigger"]["isMarket"] is True
    assert order["t"]["trigger"]["tpsl"] == "sl"
    # reduce-only flag is set
    assert order["r"] is True
    # size rounds to szDecimals=4: 0.0174 → "0.0174"
    assert order["s"] == "0.0174"

    # LONG take profit: close with SELL at 72000.  p must be < trigger for sell.
    captured_actions.clear()
    result = trader.place_trigger_order("BTC", "sell", 0.00036, 72000.0, tp_or_sl="tp")
    assert result.get("status") == "ok"
    order = captured_actions[0]["orders"][0]
    assert order["p"] != "0"
    assert float(order["p"]) < 72000.0  # slippage BELOW for sell
    assert abs(float(order["p"]) - 72000.0 * 0.95) / (72000.0 * 0.95) < 0.01
    assert order["t"]["trigger"]["tpsl"] == "tp"
    assert order["b"] is False  # sell


def test_post_order_does_not_cache_inner_error_rejections(monkeypatch):
    """When the exchange rejects an order at the inner level (e.g. "Order has
    invalid price"), the action_hash must NOT be cached — otherwise the next
    orphan-protection sweep that tries the same action is silently blocked by
    the dedup cache instead of being allowed to retry (after a fix) or to
    surface the same rejection again.
    """
    call_count = {"n": 0}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            call_count["n"] += 1
            return {
                "status": "ok",
                "response": {
                    "type": "order",
                    "data": {"statuses": [{"error": "Order has invalid price."}]},
                },
            }

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    class FakeSigner:
        address = "0x1111111111111111111111111111111111111111"

        def sign_action(self, action, nonce, vault_address=None, expires_after=None):
            return {"r": "0x" + "0" * 64, "s": "0x" + "0" * 64, "v": 27}

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr("src.trading.live_trader.requests.post", lambda *a, **kw: FakeResponse())

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=15.0)
    trader.signer = FakeSigner()
    trader.dry_run = False

    action = {"type": "order", "orders": [{"a": 0, "b": True, "p": "0", "s": "1", "r": False, "t": {"limit": {"tif": "Ioc"}}}], "grouping": "na"}

    # First call: hits the exchange, gets inner error
    r1 = trader._post_order(action)
    assert r1["status"] == "rejected"
    assert r1["reason"] == "exchange_inner_error"
    assert call_count["n"] == 1

    # Second call with the SAME action: must reach the exchange again
    # (not be silently blocked by the dedup cache)
    r2 = trader._post_order(action)
    assert r2["status"] == "rejected"
    assert r2["reason"] == "exchange_inner_error"
    assert call_count["n"] == 2, (
        "dedup cache must not block retries of inner-error rejections"
    )


def test_rescale_size_for_live_skips_when_wallet_cannot_fit_minimum(monkeypatch):
    """If wallet * 0.95 * leverage < min_order_usd, we truly cannot
    afford the exchange minimum at this leverage and must skip."""
    from src.core.live_execution import _rescale_size_for_live

    class FakeTrader:
        max_order_usd = 12.0
        min_order_usd = 11.0

        def get_account_value(self):
            return 10.0  # 10 * 0.95 * 1 = 9.50, below $11 min

    monkeypatch.setattr("src.core.live_execution.db.get_paper_account",
                        lambda: {"balance": 10_041.0})
    monkeypatch.setattr("src.core.live_execution.get_all_mids",
                        lambda: {"KAS": 0.1025})

    scaled = _rescale_size_for_live(
        {"coin": "KAS", "size": 5846.43, "entry_price": 0.1025, "leverage": 1},
        FakeTrader(),
    )
    assert scaled is None, "wallet too small to hold $11 min at 1x — must skip"


def test_load_asset_index_map_captures_sz_decimals(monkeypatch):
    """The meta endpoint returns szDecimals per coin — LiveTrader must
    cache it or else _hl_format_price emits uncapped decimals and the
    exchange rejects the order."""
    class FakeManager:
        def __init__(self):
            self.calls = []

        def post(self, payload, **kwargs):
            self.calls.append((payload, kwargs))
            if payload.get("type") == "meta":
                return {
                    "universe": [
                        {"name": "BTC", "szDecimals": 5, "maxLeverage": 50},
                        {"name": "ETH", "szDecimals": 4, "maxLeverage": 50},
                        {"name": "HYPE", "szDecimals": 2, "maxLeverage": 10},
                    ]
                }
            return {}

    manager = FakeManager()

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr("src.trading.live_trader.get_manager", lambda: manager)

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=True, max_order_usd=12.0)
    meta_call = next(payload for payload, kwargs in manager.calls if payload.get("type") == "meta")
    assert meta_call["type"] == "meta"
    assert trader.asset_index_map["BTC"] == 0
    assert trader.asset_index_map["ETH"] == 1
    assert trader.asset_index_map["HYPE"] == 2
    assert trader.sz_decimals_map["BTC"] == 5
    assert trader.sz_decimals_map["ETH"] == 4
    assert trader.sz_decimals_map["HYPE"] == 2


def test_live_trader_external_kill_switch_env_blocks_entries(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setenv("LIVE_EXTERNAL_KILL_SWITCH", "1")
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
    result = trader.execute_signal(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.1,
            "source": "copy_trade",
        },
        bypass_firewall=True,
    )

    assert result is None
    stats = trader.get_stats()
    assert stats["kill_switch_active"] is True
    assert stats["kill_switch_reason"] == "env:LIVE_EXTERNAL_KILL_SWITCH"


def test_live_trader_source_day_cap_blocks_second_entry(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.delenv("LIVE_EXTERNAL_KILL_SWITCH", raising=False)
    monkeypatch.setattr(config, "LIVE_MAX_ORDERS_PER_SOURCE_PER_DAY", 1)
    monkeypatch.setattr(config, "LIVE_CANARY_MODE", False)
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(LiveTrader, "place_market_order", lambda self, *args, **kwargs: {"status": "success"})
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(
        LiveTrader,
        "verify_fill",
        lambda self, *args, **kwargs: {"status": "verified", "size": 0.1},
    )
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(LiveTrader, "place_trigger_order", lambda self, *args, **kwargs: {"status": "success"})

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
    first = trader.execute_signal(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.1,
            "source": "copy_trade",
        }
    )
    second = trader.execute_signal(
        {
            "coin": "BTC",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 60000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.003,
            "source": "copy_trade",
        }
    )

    assert first is not None and first["status"] == "success"
    assert second is None
    stats = trader.get_stats()
    assert stats["source_orders_today"]["copy_trade"] == 1
    assert stats["max_orders_per_source_per_day"] == 1


def test_live_trader_canary_signal_cap_blocks_after_limit(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.delenv("LIVE_EXTERNAL_KILL_SWITCH", raising=False)
    monkeypatch.setattr(config, "LIVE_CANARY_MODE", True)
    monkeypatch.setattr(config, "LIVE_CANARY_MAX_SIGNALS_PER_DAY", 1)
    monkeypatch.setattr(config, "LIVE_CANARY_MAX_ORDER_USD", 25.0)
    monkeypatch.setattr(config, "LIVE_MAX_ORDERS_PER_SOURCE_PER_DAY", 0)
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(LiveTrader, "place_market_order", lambda self, *args, **kwargs: {"status": "success"})
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(
        LiveTrader,
        "verify_fill",
        lambda self, *args, **kwargs: {"status": "verified", "size": 0.01},
    )
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(LiveTrader, "place_trigger_order", lambda self, *args, **kwargs: {"status": "success"})

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
    first = trader.execute_signal(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.1,
            "source": "copy_trade",
        }
    )
    second = trader.execute_signal(
        {
            "coin": "BTC",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 60000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.003,
            "source": "golden_wallet",
        }
    )

    assert first is not None and first["status"] == "success"
    assert second is None
    stats = trader.get_stats()
    assert stats["canary_mode"] is True
    assert stats["total_entry_signals_today"] == 1
