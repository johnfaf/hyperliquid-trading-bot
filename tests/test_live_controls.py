import logging
import json
import os
import sqlite3
import sys
import tempfile
import threading
import types
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

import config
import main
from src.core import boot
from src.core.api_manager import Priority
from src.core.cycles.fast_cycle import check_file_kill_switch, run_fast_cycle
from src.core.cycles.reporting_cycle import run_reporting
from src.core.cycles.trading_cycle import (
    _execute_signal_live,
    _execute_lcrs_signals,
    _execute_options_flow_trades,
    _live_safety_stop_reason,
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
    RiskParams,
    SignalSide,
    SignalSource,
    TradeSignal,
    signal_from_execution_dict,
)
from src.signals.agent_scoring import AgentScorer, SourceScore
from src.trading.live_trader import (
    HyperliquidSigner,
    LiveTrader,
    _hl_format_price,
    _hl_format_size,
)
from src.trading.portfolio_rotation import PortfolioRotationManager, RotationDecision


@pytest.fixture(autouse=True)
def _isolate_live_kill_switch_state(tmp_path, monkeypatch):
    state_file = tmp_path / "live_kill_switch_state.json"
    monkeypatch.setenv("LIVE_KILL_SWITCH_STATE_FILE", str(state_file))
    monkeypatch.setattr(config, "LIVE_KILL_SWITCH_STATE_FILE", str(state_file), raising=False)
    monkeypatch.setattr(config, "LIVE_RISK_SIZING_ENABLED", False, raising=False)


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


def test_live_risk_sizing_targets_margin_at_stop(monkeypatch):
    monkeypatch.setattr(config, "LIVE_RISK_SIZING_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "LIVE_RISK_PER_TRADE_PCT", 0.01, raising=False)
    monkeypatch.setattr(config, "LIVE_MAX_MARGIN_PER_ORDER_PCT", 0.20, raising=False)
    monkeypatch.setattr(config, "LIVE_MIN_MARGIN_PER_ORDER_USD", 0.0, raising=False)
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)

    trader = LiveTrader(firewall=type("FW", (), {})(), dry_run=True, max_order_usd=1_000.0)
    monkeypatch.setattr(trader, "_get_mid_price", lambda coin: 100.0)
    monkeypatch.setattr(trader, "get_account_value", lambda: 100.0)
    monkeypatch.setattr(trader, "get_free_margin", lambda: 100.0)

    signal = TradeSignal(
        coin="ETH",
        side=SignalSide.LONG,
        confidence=0.80,
        source=SignalSource.STRATEGY,
        reason="test",
        risk=RiskParams(stop_loss_pct=0.10, take_profit_pct=0.50, risk_basis="roe"),
        leverage=5,
        size=0.01,
    )

    adjusted = trader._apply_risk_based_live_sizing(signal, source_policy={"size_multiplier": 1.0})

    assert round(adjusted.size * 100.0, 6) == 45.0
    assert round(adjusted.position_pct, 6) == 0.09
    sizing = trader.get_stats()["live_sizing"]
    assert sizing["reason"] == "risk_at_stop"
    assert sizing["target_margin_usd"] == 9.0
    assert sizing["target_notional_usd"] == 45.0


def test_live_risk_sizing_blocks_live_when_free_margin_unavailable(monkeypatch):
    monkeypatch.setattr(config, "LIVE_RISK_SIZING_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "LIVE_RISK_PER_TRADE_PCT", 0.01, raising=False)
    monkeypatch.setattr(config, "LIVE_MAX_MARGIN_PER_ORDER_PCT", 0.20, raising=False)
    monkeypatch.setattr(config, "LIVE_MIN_MARGIN_PER_ORDER_USD", 0.0, raising=False)
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)

    trader = LiveTrader(firewall=type("FW", (), {})(), dry_run=False, max_order_usd=1_000.0)
    monkeypatch.setattr(trader, "_get_mid_price", lambda coin: 100.0)
    monkeypatch.setattr(trader, "get_account_value", lambda: 100.0)
    monkeypatch.setattr(trader, "get_free_margin", lambda: None)

    signal = TradeSignal(
        coin="ETH",
        side=SignalSide.LONG,
        confidence=0.80,
        source=SignalSource.STRATEGY,
        reason="test",
        risk=RiskParams(stop_loss_pct=0.10, take_profit_pct=0.50, risk_basis="roe"),
        leverage=5,
        size=0.01,
    )

    adjusted = trader._apply_risk_based_live_sizing(signal, source_policy={"size_multiplier": 1.0})

    assert adjusted.size == 0.0
    assert adjusted.position_pct == 0.0
    sizing = trader.get_stats()["live_sizing"]
    assert sizing["reason"] == "free_margin_unavailable"
    assert sizing["blocked"] is True


def test_live_dual_control_requires_rolling_drawdown_cap(monkeypatch):
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "LIVE_TRADING_DUAL_CONTROL_CONFIRM", True, raising=False)
    monkeypatch.setenv("LIVE_MAX_DRAWDOWN_USD", "0")
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)

    trader = LiveTrader(firewall=type("FW", (), {})(), dry_run=False, max_order_usd=1_000.0)

    assert trader.dry_run is True
    assert trader.status_reason == "live_drawdown_cap_required"


def test_dynamic_source_policy_caps_active_sources():
    scorer = AgentScorer(
        {
            "policy_enabled": True,
            "policy_min_closed_trades": 3,
            "policy_keep_top_n": 5,
            "policy_pause_weight": 0.12,
            "policy_degrade_weight": 0.32,
            "policy_dynamic_caps_enabled": True,
            "policy_active_min_signals_per_day": 2,
            "policy_active_max_signals_per_day": 6,
            "policy_strong_min_closed_trades": 12,
            "policy_strong_win_rate": 0.55,
            "policy_strong_recent_pnl_floor": 0.0,
        }
    )
    source = "strategy:edge"
    scorer.scores = {}
    scorer._trade_history = {}
    scorer.scores[source] = SourceScore(
        source_key=source,
        total_signals=12,
        correct_signals=9,
        total_pnl=12.0,
        accuracy=0.75,
        weighted_accuracy=0.75,
        dynamic_weight=0.80,
    )
    scorer._trade_history[source] = [
        {
            "signal_id": f"s{i}",
            "pnl": 1.0,
            "correct": i < 9,
            "return_pct": 0.01,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
        for i in range(12)
    ]

    policy = scorer.get_source_policy(source)

    assert policy["status"] == "active"
    assert 2 < policy["max_signals_per_day"] <= 6
    assert policy["dynamic_cap_reason"].startswith("active_dynamic_edge=")


def test_live_source_cap_honors_static_ceiling_unless_expansion_enabled():
    trader = LiveTrader.__new__(LiveTrader)
    trader.max_orders_per_source_per_day = 3
    trader.dynamic_source_caps_allow_static_expansion = False

    policy = {"status": "active", "max_signals_per_day": 6}
    assert trader._effective_live_source_cap(policy) == 3

    trader.dynamic_source_caps_allow_static_expansion = True
    assert trader._effective_live_source_cap(policy) == 6
    assert trader._effective_live_source_cap({"status": "degraded", "max_signals_per_day": 2}) == 2


def test_audit_live_order_hygiene_stores_summary():
    trader = LiveTrader.__new__(LiveTrader)
    trader._state_lock = threading.RLock()
    trader._last_order_hygiene_audit = {}
    trader.protect_orphaned_positions = lambda: {
        "status": "ok",
        "protected": 1,
        "skipped": 0,
        "failed": 0,
        "stale_cancelled": 2,
    }

    summary = trader.audit_live_order_hygiene(repair=True)

    assert summary["audit"] == "live_order_hygiene"
    assert summary["repair"] is True
    assert summary["protected"] == 1
    assert trader._last_order_hygiene_audit["stale_cancelled"] == 2


def test_fast_cycle_runs_live_order_hygiene_on_cadence(monkeypatch):
    calls = []

    class FakePaper:
        def check_open_positions(self):
            return []

    class FakeLive:
        def snapshot_balance(self):
            pass

        def update_daily_pnl_from_fills(self):
            pass

        def manage_open_positions(self):
            return {"updated": 0, "closed": 0, "failed": 0}

        def audit_live_order_hygiene(self, repair=True):
            calls.append(repair)
            return {"status": "ok", "protected": 0, "failed": 0}

    container = types.SimpleNamespace(
        paper_trader=FakePaper(),
        live_trader=FakeLive(),
        copy_trader=None,
    )
    monkeypatch.setattr("src.core.cycles.fast_cycle.check_file_kill_switch", lambda _container: False)
    monkeypatch.setattr("src.core.cycles.fast_cycle.is_live_trading_active", lambda _container: True)
    monkeypatch.setattr("src.core.cycles.fast_cycle.sync_shadow_book_to_live", lambda _container: [])
    monkeypatch.setattr("src.core.cycles.fast_cycle._scan_whale_trades", lambda _container: None)
    monkeypatch.setenv("LIVE_ORDER_HYGIENE_AUDIT_INTERVAL_CYCLES", "5")

    run_fast_cycle(container, cycle_count=4)
    run_fast_cycle(container, cycle_count=5)

    assert calls == [True]


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
    monkeypatch.setattr(config, "LIVE_CANARY_MODE", False)

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
    monkeypatch.setattr(config, "LIVE_CANARY_MODE", False)

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


def test_execute_signal_keeps_existing_protection_before_new_fill_bracket(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    events = []
    trigger_sizes = []
    existing_positions = [
        {
            "coin": "ETH",
            "side": "long",
            "size": 0.005,
            "entry_price": 1900.0,
            "leverage": 2,
        }
    ]

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: list(existing_positions))
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(LiveTrader, "place_market_order", lambda self, *args, **kwargs: {"status": "success"})
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(
        LiveTrader,
        "verify_fill",
        lambda self, *args, **kwargs: {
            "status": "verified",
            "size": 0.1,
            "position_size": 0.105,
            "entry_price": 2000.0,
        },
    )
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(LiveTrader, "_cancel_protective_orders", lambda self, coin: events.append(("cancel", coin)) or 2)
    monkeypatch.setattr(
        LiveTrader,
        "place_trigger_order",
        lambda self, coin, side, size, trigger_price, tp_or_sl="sl": (
            events.append(("trigger", tp_or_sl, size))
            or trigger_sizes.append(size)
            or {"status": "success"}
        ),
    )
    monkeypatch.setattr(config, "LIVE_CANARY_MODE", False)

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
    assert ("cancel", "ETH") not in events
    assert [event[0] for event in events] == ["trigger", "trigger"]
    assert trigger_sizes == [0.1, 0.1]
    assert result["protected_size"] == 0.1


def test_execute_signal_protects_only_new_fill_when_existing_protection_not_visible(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    trigger_sizes = []
    existing_positions = [
        {
            "coin": "ETH",
            "side": "long",
            "size": 0.005,
            "entry_price": 1900.0,
            "leverage": 2,
        }
    ]

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: list(existing_positions))
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(LiveTrader, "place_market_order", lambda self, *args, **kwargs: {"status": "success"})
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(
        LiveTrader,
        "verify_fill",
        lambda self, *args, **kwargs: {
            "status": "verified",
            "size": 0.1,
            "position_size": 0.105,
            "entry_price": 2000.0,
        },
    )
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(LiveTrader, "_cancel_protective_orders", lambda self, coin: 0)
    monkeypatch.setattr(
        LiveTrader,
        "place_trigger_order",
        lambda self, coin, side, size, trigger_price, tp_or_sl="sl": (
            trigger_sizes.append(size) or {"status": "success"}
        ),
    )
    monkeypatch.setattr(config, "LIVE_CANARY_MODE", False)

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
    assert trigger_sizes == [0.1, 0.1]
    assert result["protected_size"] == 0.1


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
    monkeypatch.setattr(config, "LIVE_CANARY_MODE", False)

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
    """AUDIT M2 — SL succeeds, TP permanently fails.  Selective retry keeps
    the good SL across attempts; the caller cancels the surviving SL once
    before the emergency close so no orphan trigger is left resting."""

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    # Dispatch by leg so the new "only-retry-the-failed-leg" flow has the
    # right per-leg semantics (SL always ok, TP always rejected).
    def _fake_trigger(self, coin, side, size, trigger_price, tp_or_sl="sl"):
        if tp_or_sl == "sl":
            return {"status": "success"}
        return {"status": "error", "message": "tp rejected"}

    close_calls = []
    cancel_calls = []

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(LiveTrader, "place_market_order", lambda self, *args, **kwargs: {"status": "success"})
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(LiveTrader, "verify_fill", lambda self, *args, **kwargs: {"status": "verified"})
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(LiveTrader, "place_trigger_order", _fake_trigger)
    monkeypatch.setattr(
        LiveTrader,
        "cancel_all_orders",
        lambda self, coin=None: (cancel_calls.append(coin) or 1),
    )
    monkeypatch.setattr(
        LiveTrader,
        "close_position",
        lambda self, coin: close_calls.append(coin) or {"status": "success", "coin": coin},
    )
    monkeypatch.setattr("src.trading.live_trader.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.trading.live_trader.random.uniform", lambda *_args, **_kwargs: 0.0)

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
    assert result["protective_order_attempts"] == 3
    assert result["protective_legs_surviving"] == ["sl"]
    assert result["protective_legs_missing"] == ["tp"]
    assert close_calls == ["ETH"]
    # AUDIT M2: exactly ONE cancel_all_orders call — from the caller's
    # pre-close orphan cleanup (removing the surviving SL).  NO
    # between-retry cancels that would wipe the good SL during the
    # retry window.
    assert cancel_calls == ["ETH"]


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
    monkeypatch.setattr(LiveTrader, "get_positions", lambda self, **kw: next(positions))
    monkeypatch.setattr("src.trading.live_trader.time.time", clock.time)
    monkeypatch.setattr("src.trading.live_trader.time.sleep", clock.sleep)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False)
    result = trader.verify_fill("ETH", "buy", 0.1, timeout=1.0, poll_interval=0.1)

    assert result is not None
    assert result["status"] == "verified"
    assert result["partial_fill"] is True
    assert abs(result["size"] - 0.06) < 1e-12
    assert abs(result["position_size"] - 0.06) < 1e-12


def test_verify_fill_uses_position_delta_not_total_position(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_positions", lambda self, **kw: [{"coin": "ETH", "szi": 0.5}])

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False)

    assert trader.verify_fill("ETH", "buy", 0.1, blocking=False, baseline_position_size=0.5) is None


def test_cancel_all_orders_uses_hyperliquid_oid_fallback(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    cancelled = []
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(
        LiveTrader,
        "get_open_orders",
        lambda self, **kw: [
            {"coin": "ETH", "oid": 111},
            {"coin": "BTC", "order_id": 222},
            {"coin": "SOL", "id": 333},
        ],
    )
    monkeypatch.setattr(
        LiveTrader,
        "cancel_order",
        lambda self, coin, order_id: cancelled.append((coin, order_id)) or True,
    )

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False)

    assert trader.cancel_all_orders() == 3
    assert cancelled == [("ETH", 111), ("BTC", 222), ("SOL", 333)]


def test_get_open_orders_uses_frontend_open_orders(monkeypatch):
    class FakeManager:
        def __init__(self):
            self.calls = []

        def post(self, payload, **kwargs):
            self.calls.append((payload, kwargs))
            return [
                {
                    "coin": "ETH",
                    "isTrigger": True,
                    "isPositionTpsl": True,
                    "orderType": "Stop Market",
                    "triggerPx": "2407.8",
                    "reduceOnly": True,
                    "side": "B",
                    "sz": "0.0051",
                    "oid": 123,
                }
            ]

    trader = LiveTrader.__new__(LiveTrader)
    trader.public_address = "0x2222222222222222222222222222222222222222"
    trader.api_manager = FakeManager()
    trader._state_lock = threading.RLock()
    trader._last_order_visibility_status = {}

    orders = trader.get_open_orders(force_fresh=True)

    assert orders and orders[0]["oid"] == 123
    payload, kwargs = trader.api_manager.calls[0]
    assert payload["type"] == "frontendOpenOrders"
    assert kwargs["force_fresh"] is True
    assert trader._last_order_visibility_status["ok"] is True
    assert trader._last_order_visibility_status["order_count"] == 1


def test_free_margin_prefers_account_value_minus_used_margin_over_withdrawable():
    state = {
        "withdrawable": "25",
        "marginSummary": {"accountValue": "1000", "totalMarginUsed": "150"},
    }

    assert LiveTrader._extract_free_margin_from_state(state) == 850.0


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


def test_mirror_executed_trades_logs_warning_for_guardrail_skip(monkeypatch, caplog):
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
            return None  # guardrail skip path (e.g., source/day cap hit)

    monkeypatch.setattr("src.core.live_execution.db.get_paper_account", lambda: {"balance": 10000.0})
    monkeypatch.setattr("src.core.live_execution.get_all_mids", lambda: {"PAXG": "4734.45"})

    trader = FakeLiveTrader()
    container = type("Container", (), {"live_trader": trader})()
    executed = [
        signal_from_execution_dict(
            {
                "coin": "PAXG",
                "side": "long",
                "confidence": 0.63,
                "entry_price": 4734.45,
                "size": 0.04,
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

    assert trader.executed == [("PAXG", "long", True)]
    assert "blocked by live guardrails" in caplog.text.lower()
    assert "live copy failed" not in caplog.text.lower()


def test_mirror_executed_trades_preserves_paper_execution_order(monkeypatch):
    class FakeLiveTrader:
        def __init__(self):
            self.executed = []

        def is_live_enabled(self):
            return True

        def is_deployable(self):
            return True

        def get_account_value(self):
            return 100.0

        def execute_signal(self, signal, bypass_firewall=False):
            self.executed.append((signal.coin, signal.side.value, bypass_firewall))
            return {"status": "success"}

    # Keep candidate sizes as supplied so the margin-budget selection is deterministic.
    monkeypatch.setattr("src.core.live_execution._rescale_size_for_live", lambda trade, trader: trade)

    trader = FakeLiveTrader()
    container = type("Container", (), {"live_trader": trader})()
    executed = [
        signal_from_execution_dict(
            {
                "coin": "ETH",
                "side": "long",
                "confidence": 0.20,
                "entry_price": 100.0,
                "size": 0.60,   # margin $60 at 1x
                "leverage": 1,
                "strategy_type": "mirror_test",
            }
        ),
        signal_from_execution_dict(
            {
                "coin": "BTC",
                "side": "short",
                "confidence": 0.95,
                "entry_price": 100.0,
                "size": 0.50,   # margin $50 at 1x
                "leverage": 1,
                "strategy_type": "mirror_test",
            }
        ),
    ]

    mirror_executed_trades_to_live(
        container,
        executed,
        success_label="LIVE COPY",
        skip_label="SKIP",
    )

    # Margin budget is $95 (95% of $100), so only the first paper trade should mirror.
    assert trader.executed == [("ETH", "long", True)]


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

    assert abs(opened["stop_loss"] - (price * 0.975)) < 1e-12
    assert abs(opened["take_profit"] - (price * 1.125)) < 1e-12


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

    assert abs(opened["stop_loss"] - (price * 1.025)) < 1e-12
    assert abs(opened["take_profit"] - (price * 0.875)) < 1e-12


def test_run_alpha_arena_passes_multi_coin_candle_map(monkeypatch):
    seen = {}

    class FakeArena:
        def run_cycle(self, historical_candles=None):
            seen["historical"] = historical_candles

        def get_stats(self):
            return {"active_agents": 1, "champions": 1, "total_arena_pnl": 0.0}

        def get_champion_signals(self, **kwargs):
            seen["current"] = kwargs.get("current_candles")
            return []

    container = type(
        "Container",
        (),
        {
            "arena": FakeArena(),
            "live_trader": None,
        },
    )()

    monkeypatch.setattr("src.core.cycles.trading_cycle._ARENA_COIN_UNIVERSE", ["BTC", "ETH"])
    monkeypatch.setattr("src.core.cycles.trading_cycle._ARENA_MAX_COINS", 2)
    monkeypatch.setattr("src.core.cycles.trading_cycle.is_live_trading_active", lambda container: False)

    class FakeManager:
        def post(self, payload=None, **kwargs):
            coin = payload["req"]["coin"]
            return [
                {
                    "o": "100.0",
                    "h": "101.0",
                    "l": "99.0",
                    "c": "100.0",
                    "v": "10",
                    "t": 1_000 + idx,
                }
                for idx in range(60)
            ] if coin in {"BTC", "ETH"} else []

    monkeypatch.setattr("src.core.api_manager.get_manager", lambda: FakeManager())

    _run_alpha_arena(container, {"overall_regime": "neutral"})

    assert set(seen["historical"]) == {"BTC", "ETH"}
    assert set(seen["current"]) == {"BTC", "ETH"}


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
    alerts = []

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
    monkeypatch.setattr(
        "src.core.live_execution._notify_manual_close_detected",
        lambda trade, exit_price: alerts.append((trade["coin"], exit_price)),
    )

    reconciled = sync_shadow_book_to_live(container)
    assert len(reconciled) == 1
    assert reconciled[0]["trade_id"] == 7
    assert reconciled[0]["coin"] == "ETH"
    assert reconciled[0]["reason"] == "live_reconciled_closed"
    assert reconciled[0]["pnl"] == 0.0
    assert closed == [(7, 2100.0, 0.0)]
    assert alerts == [("ETH", 2100.0)]
    assert metadata_updates[0][0] == 7
    assert metadata_updates[0][1]["synthetic_reconciliation"] is True
    assert metadata_updates[0][1]["reconciliation_reason"] == "live_reconciled_closed"


def test_sync_shadow_book_creates_synthetic_trade_for_orphan_live_position(monkeypatch):
    opened = []
    audits = []

    class FakeLiveTrader:
        def is_live_enabled(self):
            return True

        def is_deployable(self):
            return True

        def get_positions(self):
            return [
                {
                    "coin": "BTC",
                    "side": "long",
                    "size": 0.15,
                    "szi": 0.15,
                    "entry_price": 64000.0,
                    "leverage": 3,
                }
            ]

        def get_account_value(self):
            return 500.0

    container = type(
        "Container",
        (),
        {"live_trader": FakeLiveTrader(), "paper_trader": object()},
    )()

    monkeypatch.setattr("src.core.live_execution.db.get_open_paper_trades", lambda: [])
    monkeypatch.setattr("src.core.live_execution.get_all_mids", lambda: {})
    monkeypatch.setattr("src.core.live_execution._paper_trade_id_for_client_order_id", lambda key: None)
    monkeypatch.setattr(
        "src.core.live_execution.db.open_paper_trade",
        lambda strategy_id, coin, side, entry_price, size, leverage=1,
               stop_loss=None, take_profit=None, metadata=None, **kw:
            opened.append(
                {
                    "strategy_id": strategy_id,
                    "coin": coin,
                    "side": side,
                    "entry_price": entry_price,
                    "size": size,
                    "leverage": leverage,
                    "metadata": metadata,
                    "idempotency_key": kw.get("idempotency_key"),
                }
            ) or 99,
    )
    monkeypatch.setattr(
        "src.core.live_execution.db.audit_log",
        lambda **kwargs: audits.append(kwargs),
    )

    reconciled = sync_shadow_book_to_live(container)

    assert reconciled == []
    assert len(opened) == 1
    assert opened[0]["coin"] == "BTC"
    assert opened[0]["side"] == "long"
    assert opened[0]["metadata"]["orphan_found"] is True
    assert opened[0]["metadata"]["reconciliation_reason"] == "orphan_found"
    assert opened[0]["metadata"]["source"] == "live_orphan"
    assert audits[0]["action"] == "orphan_found"


def test_sync_shadow_book_logs_idempotent_orphan_replay_as_info(monkeypatch, caplog):
    opened = []
    audits = []

    class FakeLiveTrader:
        def is_live_enabled(self):
            return True

        def is_deployable(self):
            return True

        def get_positions(self):
            return [
                {
                    "coin": "BTC",
                    "side": "long",
                    "size": 0.15,
                    "szi": 0.15,
                    "entry_price": 64000.0,
                    "leverage": 3,
                }
            ]

        def get_account_value(self):
            return 500.0

    container = type(
        "Container",
        (),
        {"live_trader": FakeLiveTrader(), "paper_trader": object()},
    )()

    monkeypatch.setattr("src.core.live_execution.db.get_open_paper_trades", lambda: [])
    monkeypatch.setattr("src.core.live_execution.get_all_mids", lambda: {})
    monkeypatch.setattr("src.core.live_execution._paper_trade_id_for_client_order_id", lambda key: 99)
    monkeypatch.setattr(
        "src.core.live_execution.db.open_paper_trade",
        lambda strategy_id, coin, side, entry_price, size, leverage=1,
               stop_loss=None, take_profit=None, metadata=None, **kw:
            opened.append(kw.get("idempotency_key")) or 99,
    )
    monkeypatch.setattr(
        "src.core.live_execution.db.audit_log",
        lambda **kwargs: audits.append(kwargs),
    )

    caplog.set_level(logging.INFO, logger="src.core.live_execution")
    reconciled = sync_shadow_book_to_live(container)

    assert reconciled == []
    assert opened == ["orphan:BTC:long:64000:0.15:3"]
    assert audits[0]["details"]["trade_id"] == 99
    assert "Synthetic paper trade already tracks orphan live position" in caplog.text
    assert "Created synthetic paper trade for orphan live position" not in caplog.text


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


def test_live_safety_stop_reports_actual_kill_switch_reason():
    trader = LiveTrader.__new__(LiveTrader)
    trader._state_lock = threading.Lock()
    trader.kill_switch_active = True
    trader._kill_switch_reason = "dualwrite_unhealthy:recent_failures=11:total_failed=11"
    trader.status_reason = "persisted_kill_switch"

    reason = trader.get_safety_stop_reason()

    assert reason == (
        "kill_switch_active:dualwrite_unhealthy:"
        "recent_failures=11:total_failed=11"
    )
    assert trader._safety_stop_rejection_code() == "kill_switch_active"
    assert _live_safety_stop_reason(trader) == reason
    assert "daily_loss" not in reason


def test_live_safety_stop_keeps_daily_loss_rejection_code():
    trader = LiveTrader.__new__(LiveTrader)
    trader._state_lock = threading.Lock()
    trader.kill_switch_active = True
    trader._kill_switch_reason = "daily_loss_limit:125.00>100.00"
    trader.status_reason = "daily_loss_limit_exceeded"

    assert trader.get_safety_stop_reason() == (
        "kill_switch_active:daily_loss_limit:125.00>100.00"
    )
    assert trader._safety_stop_rejection_code() == "daily_loss_exceeded"


def test_stale_dualwrite_kill_switch_auto_clears_when_healthy(monkeypatch, tmp_path):
    state_file = tmp_path / "kill_state.json"
    state_file.write_text(
        '{"active": true, "reason": "dualwrite_unhealthy:recent_failures=11:total_failed=11"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.trading.live_trader.db.get_backend_name",
        lambda: "dualwrite",
    )
    monkeypatch.setattr(
        "src.trading.live_trader.db.dualwrite_is_healthy",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        "src.data.db.postgres.check_health",
        lambda: True,
    )

    trader = LiveTrader.__new__(LiveTrader)
    trader.kill_switch_state_file = str(state_file)
    trader._state_lock = threading.Lock()
    trader.kill_switch_active = False
    trader._kill_switch_reason = ""
    trader.status_reason = ""
    trader._dualwrite_health_window_s = 300.0
    trader._dualwrite_health_max_failures = 5

    trader._load_persisted_kill_switch_state()

    assert trader.kill_switch_active is False
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert payload["active"] is False
    assert payload["reason"].startswith("auto_cleared:dualwrite_unhealthy")


def test_non_dualwrite_kill_switch_remains_sticky(monkeypatch, tmp_path):
    state_file = tmp_path / "kill_state.json"
    state_file.write_text(
        '{"active": true, "reason": "daily_loss_limit:125.00>100.00"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.trading.live_trader.db.get_backend_name",
        lambda: "dualwrite",
    )
    monkeypatch.setattr(
        "src.trading.live_trader.db.dualwrite_is_healthy",
        lambda **kwargs: True,
    )

    trader = LiveTrader.__new__(LiveTrader)
    trader.kill_switch_state_file = str(state_file)
    trader._state_lock = threading.Lock()
    trader.kill_switch_active = False
    trader._kill_switch_reason = ""
    trader.status_reason = ""
    trader._dualwrite_health_window_s = 300.0
    trader._dualwrite_health_max_failures = 5

    trader._load_persisted_kill_switch_state()

    assert trader.kill_switch_active is True
    assert trader._kill_switch_reason == "daily_loss_limit:125.00>100.00"
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert payload["active"] is True


def test_legacy_protective_churn_kill_switch_becomes_coin_quarantine(monkeypatch, tmp_path):
    state_file = tmp_path / "kill_state.json"
    state_file.write_text(
        '{"active": true, "reason": "protective_order_churn:TRUMP"}',
        encoding="utf-8",
    )
    bot_state = {}
    monkeypatch.setattr("src.trading.live_trader.db.set_bot_state", lambda key, value: bot_state.setdefault(key, value) or True)

    trader = LiveTrader.__new__(LiveTrader)
    trader.kill_switch_state_file = str(state_file)
    trader._state_lock = threading.Lock()
    trader.kill_switch_active = False
    trader._kill_switch_reason = ""
    trader.status_reason = ""
    trader._protective_churn_quarantined_coins = {}
    trader._protective_churn_trips = 0

    trader._load_persisted_kill_switch_state()

    assert trader.kill_switch_active is False
    assert trader._is_protective_churn_quarantined("TRUMP") is True
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert payload["active"] is False
    assert payload["reason"] == "coin_quarantine:protective_order_churn:TRUMP"
    assert "TRUMP" in bot_state[LiveTrader._PROTECTIVE_CHURN_QUARANTINE_STATE_KEY]["coins"]


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
    monkeypatch.setattr("src.core.live_execution.db.get_open_paper_trades", lambda: [])

    scaled = _rescale_size_for_live({"coin": "ETH", "size": 0.2}, FakeTrader())

    assert scaled is None


def test_rescale_size_for_live_enforces_max_order_usd_cap(monkeypatch):
    """After rescaling, notional must never exceed trader.max_order_usd
    EVEN when the fill slips to the SDK's default 5% market slippage (E6).
    The cap is computed against the slipped reference price, so the final
    mid-notional lands slightly under the cap — that's the point."""
    class FakeTrader:
        max_order_usd = 3.0

        def get_account_value(self):
            return 10_000.0  # equal to paper — scale = 1.0

    monkeypatch.setattr("src.core.live_execution.db.get_paper_account", lambda: {"balance": 10_000.0})
    monkeypatch.setattr("src.core.live_execution.db.get_open_paper_trades", lambda: [])
    monkeypatch.setattr("src.core.live_execution.get_all_mids", lambda: {"ETH": 2000.0})
    monkeypatch.setenv("LIVE_MARKET_SLIPPAGE_PCT", "0.05")

    # 0.5 ETH @ $2000 = $1000 notional. Cap=$3 with 5% buy-side slippage
    # means size = 3 / (2000 * 1.05) ≈ 0.001428… ETH.  Notional at slipped
    # price is $3 exactly; notional at mid is ≈ $2.857 (under the cap).
    scaled = _rescale_size_for_live(
        {"coin": "ETH", "size": 0.5, "side": "buy", "entry_price": 2000.0},
        FakeTrader(),
    )
    assert scaled is not None
    slipped_notional = scaled["size"] * 2000.0 * 1.05
    assert slipped_notional <= 3.0 + 1e-9
    # And the mid-notional should not be dramatically under cap either —
    # within one slippage-factor.
    assert scaled["size"] * 2000.0 >= 3.0 / 1.05 - 1e-9


def test_rescale_size_for_live_uses_free_margin_for_scale(monkeypatch):
    class FakeTrader:
        max_order_usd = 1_000.0
        min_order_usd = 11.0

        def get_account_value(self):
            return 1_000.0

        def get_free_margin(self):
            return 50.0

    monkeypatch.setattr("src.core.live_execution.db.get_paper_account", lambda: {"balance": 1_000.0})
    monkeypatch.setattr("src.core.live_execution.db.get_open_paper_trades", lambda: [])
    monkeypatch.setattr("src.core.live_execution.get_all_mids", lambda: {"ETH": 100.0})

    scaled = _rescale_size_for_live(
        {"coin": "ETH", "size": 10.0, "entry_price": 100.0, "leverage": 5},
        FakeTrader(),
    )

    assert scaled is not None
    assert abs(scaled["size"] - 0.5) < 1e-12


def test_rescale_size_for_live_deducts_paper_open_margin(monkeypatch):
    """H6: paper-to-live mirror must use paper *free* balance
    (= paper_balance - paper_margin_used) as the denominator, not the
    raw paper balance.  Without this deduction, live sizing silently
    oversizes by ``paper_balance / paper_free_balance`` whenever the
    paper book has open exposure — because ``live_free_margin`` on the
    live side already excludes locked margin.

    Setup: paper_balance=$10_000 with one open paper trade at
    size=100 @ $50 @ 5x leverage → notional=$5_000, margin=$1_000.
    paper_free_balance should be $10_000 - $1_000 = $9_000.
    live_free_margin=$9_000 → scale = 9000/9000 = 1.0, so a 1 ETH
    paper signal maps to 1 ETH live.  Without the deduction, scale
    would be 9000/10000 = 0.9 and we'd size to 0.9 ETH.
    """
    from src.core.live_execution import _rescale_size_for_live

    class FakeTrader:
        max_order_usd = 1_000_000.0  # out of the way
        min_order_usd = 1.0

        def get_account_value(self):
            return 9_000.0

        def get_free_margin(self):
            return 9_000.0

    monkeypatch.setattr("src.core.live_execution.db.get_paper_account",
                        lambda: {"balance": 10_000.0})
    monkeypatch.setattr(
        "src.core.live_execution.db.get_open_paper_trades",
        lambda: [
            {
                "id": 1,
                "coin": "BTC",
                "side": "long",
                "entry_price": 50.0,
                "size": 100.0,
                "leverage": 5,
            }
        ],
    )
    monkeypatch.setattr("src.core.live_execution.get_all_mids",
                        lambda: {"ETH": 3000.0})

    scaled = _rescale_size_for_live(
        {"coin": "ETH", "size": 1.0, "entry_price": 3000.0, "leverage": 5},
        FakeTrader(),
    )
    assert scaled is not None
    # scale = live_free_margin($9000) / paper_free_balance($9000) = 1.0
    assert abs(scaled["size"] - 1.0) < 1e-9, (
        f"expected size ≈ 1.0 with symmetric free-margin scaling, "
        f"got {scaled['size']}"
    )


def test_rescale_size_for_live_skips_when_paper_fully_levered(monkeypatch):
    """H6: when open paper exposure has locked up the entire paper
    balance, paper_free_balance is 0 and mirroring any *new* live trade
    would be a category error — the paper book has no free capacity
    to scale from.  Must skip."""
    from src.core.live_execution import _rescale_size_for_live

    class FakeTrader:
        max_order_usd = 1_000.0
        min_order_usd = 1.0

        def get_account_value(self):
            return 1_000.0

        def get_free_margin(self):
            return 1_000.0

    monkeypatch.setattr("src.core.live_execution.db.get_paper_account",
                        lambda: {"balance": 1_000.0})
    # Single paper trade locks up the entire balance as margin:
    # size=1000 * price=$5 / leverage=5 = $1000 margin.
    monkeypatch.setattr(
        "src.core.live_execution.db.get_open_paper_trades",
        lambda: [
            {
                "id": 1,
                "coin": "SOL",
                "side": "long",
                "entry_price": 5.0,
                "size": 1000.0,
                "leverage": 5,
            }
        ],
    )
    monkeypatch.setattr("src.core.live_execution.get_all_mids",
                        lambda: {"ETH": 3000.0})

    scaled = _rescale_size_for_live(
        {"coin": "ETH", "size": 1.0, "entry_price": 3000.0, "leverage": 5},
        FakeTrader(),
    )
    assert scaled is None, (
        "paper book fully-levered → no free capacity to scale from; must skip"
    )


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
    monkeypatch.setattr("src.core.live_execution.db.get_open_paper_trades", lambda: [])
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
    monkeypatch.setattr("src.core.live_execution.db.get_open_paper_trades", lambda: [])
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


def test_execute_signal_applies_kelly_dampen(monkeypatch):
    """H2 regression: ScalingTier.kelly_dampen must shrink both size AND
    position_pct before regime overlay + hard cap, otherwise the tier's
    soft-sizing dampener is dead code (exactly what the external audit
    caught on the first cut).
    """
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

    # T2 has kelly_dampen=0.8. Cap high enough not to interfere:
    # $40 raw notional * 0.8 = $32, below the $200 max_order_usd.
    monkeypatch.setenv("LIVE_TIER", "T2")
    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=200.0)
    assert trader.active_tier is not None
    assert trader.active_tier.name == "T2"
    assert trader.kelly_dampen == 0.8

    trader.execute_signal(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 1,
            "size": 0.02,  # $40 notional raw
            "strategy_type": "momentum_long",
        }
    )

    # Expect 0.02 * 0.8 = 0.016 ETH placed.
    assert len(placed_sizes) == 1
    assert abs(placed_sizes[0] - 0.016) < 1e-9, (
        f"Expected 0.016 ETH after 0.8 dampening, got {placed_sizes[0]}"
    )


def test_execute_signal_skips_kelly_dampen_at_t4(monkeypatch):
    """T4 has kelly_dampen=1.0 — size must NOT be dampened."""
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

    monkeypatch.setenv("LIVE_TIER", "T4")
    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=200.0)
    assert trader.active_tier.name == "T4"
    assert trader.kelly_dampen == 1.0

    trader.execute_signal(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 1,
            "size": 0.02,
            "strategy_type": "momentum_long",
        }
    )
    assert len(placed_sizes) == 1
    assert abs(placed_sizes[0] - 0.02) < 1e-9


def test_live_trader_raises_cap_to_exchange_minimum(monkeypatch):
    """Regression test (H11): LIVE_MAX_ORDER_USD below Hyperliquid's $10
    minimum notional is a FATAL startup misconfiguration in live mode —
    silently raising the cap would let the operator deploy with a number
    they didn't intend.  In live mode we now raise ValueError.  In dry-run
    mode we still raise-to-min so backtests/simulations don't crash.
    """
    import pytest

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)

    # Live mode: hard failure.
    with pytest.raises(ValueError, match="below"):
        LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=3.0)

    # Dry-run mode: fall back to the old raise-to-min behaviour.
    trader = LiveTrader(firewall=FakeFirewall(), dry_run=True, max_order_usd=3.0)
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


def test_execute_signal_top_tier_floorup_executes(monkeypatch):
    """High-confidence approved signals can be bumped just enough to clear the
    exchange minimum without relaxing any signal filters."""
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

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=15.0)
    result = trader.execute_signal(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.90,
            "entry_price": 2000.0,
            "position_pct": 0.02,
            "leverage": 2,
            "size": 0.005,  # $10 notional -> eligible for bounded top-tier floor-up
            "strategy_type": "momentum_long",
        }
    )

    assert result is not None
    assert result["status"] == "success"
    assert len(placed_sizes) == 1
    assert placed_sizes[0] * 2000.0 >= trader.min_order_usd
    assert trader.get_stats()["min_order_top_tier_floorups_today"] == 1
    assert trader.get_stats()["approved_but_not_executable_today"] == 0


def test_execute_signal_same_side_merge_executes_under_minimum(monkeypatch):
    """If we already hold the same coin and side, an undersized add-on can be
    lifted to the exchange minimum as a same-side merge instead of being dropped."""
    placed_sizes = []
    existing_positions = [
        {
            "coin": "ETH",
            "side": "long",
            "size": 0.25,
            "entry_price": 1900.0,
            "leverage": 2,
        }
    ]

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: list(existing_positions))
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

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=15.0)
    result = trader.execute_signal(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.50,
            "entry_price": 2000.0,
            "position_pct": 0.01,
            "leverage": 2,
            "size": 0.0025,  # $5 notional -> too small unless same-side merge is allowed
            "strategy_type": "momentum_long",
        }
    )

    assert result is not None
    assert result["status"] == "success"
    assert len(placed_sizes) == 1
    assert placed_sizes[0] * 2000.0 >= trader.min_order_usd
    stats = trader.get_stats()
    assert stats["min_order_same_side_merges_today"] == 1
    assert stats["min_order_top_tier_floorups_today"] == 0


def test_execute_signal_tracks_approved_but_not_executable(monkeypatch):
    """Approved signals that stay below the exchange minimum after bounded
    execution checks should be visible in telemetry instead of disappearing."""
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

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=15.0)
    result = trader.execute_signal(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.60,
            "entry_price": 2000.0,
            "position_pct": 0.01,
            "leverage": 2,
            "size": 0.002,  # $4 notional -> too far below min for top-tier floor-up
            "strategy_type": "momentum_long",
        }
    )

    assert result is None
    assert placed_sizes == []
    stats = trader.get_stats()
    assert stats["approved_but_not_executable_today"] == 1
    assert stats["min_order_floorups_today"] == 0


def test_execute_signal_live_mirror_refloors_after_tier_dampen(monkeypatch):
    """Paper-to-live mirrors can be shrunk below the exchange minimum by final
    live-only tier/regime modifiers.  Re-floor them at the last sizing gate so
    an already-approved mirror does not become approved-but-not-executable."""
    placed_sizes = []

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_positions", lambda self: [])
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

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=15.0)
    trader.kelly_dampen = 0.65
    signal = TradeSignal(
        coin="ETH",
        side=SignalSide.LONG,
        confidence=0.51,
        source=SignalSource.COPY_TRADE,
        reason="paper mirror",
        entry_price=2000.0,
        leverage=5,
        size=0.0056,  # $11.20 before T1 dampen, $7.28 after dampen
        context={"live_mirror": True},
    )

    result = trader.execute_signal(signal, bypass_firewall=True)

    assert result is not None
    assert result["status"] == "success"
    assert len(placed_sizes) == 1
    assert placed_sizes[0] * 2000.0 >= trader.min_order_usd
    stats = trader.get_stats()
    assert stats["min_order_mirror_floorups_today"] == 1
    assert stats["approved_but_not_executable_today"] == 0


def test_execute_signal_retries_protective_orders_before_succeeding(monkeypatch):
    """AUDIT M2 — both legs fail attempt 1, both succeed attempt 2.  No
    between-retry cancel_all_orders is expected anymore: the failed legs
    have no resting orders to wipe and the retry re-places just those
    legs on the next iteration."""
    placed_sizes = []
    cancel_calls = []
    trigger_results = iter(
        [
            {"status": "error", "message": "sl_failed"},
            {"status": "error", "message": "tp_failed"},
            {"status": "success"},
            {"status": "success"},
        ]
    )

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
    monkeypatch.setattr(LiveTrader, "place_trigger_order", lambda self, *args, **kwargs: next(trigger_results))
    monkeypatch.setattr(
        LiveTrader,
        "cancel_all_orders",
        lambda self, coin=None: (cancel_calls.append(coin) or 1),
    )
    monkeypatch.setattr("src.trading.live_trader.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.trading.live_trader.random.uniform", lambda *_args, **_kwargs: 0.0)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=15.0)
    result = trader.execute_signal(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.85,
            "entry_price": 2000.0,
            "position_pct": 0.02,
            "leverage": 2,
            "size": 0.01,
            "strategy_type": "momentum_long",
        }
    )

    assert result is not None
    assert result["status"] == "success"
    assert result["protective_order_attempts"] == 2
    # AUDIT M2: no between-retry cancel_all_orders call.  Both legs failed
    # on attempt 1 (no resting orders to wipe) and both succeeded on
    # attempt 2 so no caller-side orphan cleanup was needed either.
    assert cancel_calls == []


def test_execute_signal_closes_position_after_protective_retries_exhausted(monkeypatch):
    cancel_calls = []
    close_calls = []

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
        lambda self, coin, side, size, leverage=1, reduce_only=False: {"status": "success"},
    )
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(LiveTrader, "verify_fill", lambda self, *args, **kwargs: {"status": "verified"})
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(
        LiveTrader,
        "place_trigger_order",
        lambda self, *args, **kwargs: {"status": "error", "message": "trigger_failed"},
    )
    monkeypatch.setattr(
        LiveTrader,
        "cancel_all_orders",
        lambda self, coin=None: (cancel_calls.append(coin) or 1),
    )
    monkeypatch.setattr(
        LiveTrader,
        "close_position",
        lambda self, coin: (close_calls.append(coin) or {"status": "success"}),
    )
    monkeypatch.setattr("src.trading.live_trader.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.trading.live_trader.random.uniform", lambda *_args, **_kwargs: 0.0)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=15.0)
    result = trader.execute_signal(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.85,
            "entry_price": 2000.0,
            "position_pct": 0.02,
            "leverage": 2,
            "size": 0.01,
            "strategy_type": "momentum_long",
        }
    )

    assert result is not None
    assert result["status"] == "error"
    assert result["message"] == "protective_order_failed"
    assert result["protective_order_attempts"] == 3
    assert close_calls == ["ETH"]
    # AUDIT M2: both legs failed every attempt, so nothing to cancel
    # between retries and no surviving leg to cancel pre-close.
    assert cancel_calls == []
    assert result["protective_legs_surviving"] == []
    assert set(result["protective_legs_missing"]) == {"sl", "tp"}


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
    expected_take_profit = price * (1 + 0.20 / leverage)

    assert trade is not None
    assert abs(opened["stop_loss"] - expected_stop_loss) < 1e-12
    assert abs(opened["take_profit"] - expected_take_profit) < 1e-12
    assert abs(trade["stop_loss"] - expected_stop_loss) < 1e-12
    assert abs(trade["take_profit"] - expected_take_profit) < 1e-12

    signal = signal_from_execution_dict(trade)
    assert abs(signal.risk.stop_loss_pct - (0.04 / leverage)) < 1e-12
    assert abs(signal.risk.take_profit_pct - (0.20 / leverage)) < 1e-12


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


def test_check_file_kill_switch_cancels_live_orders_and_stops(tmp_path, monkeypatch):
    kill_file = tmp_path / "KILL_SWITCH"
    kill_file.write_text("kill", encoding="utf-8")

    class FakeLiveTrader:
        def __init__(self):
            self.kill_switch_active = False
            self._kill_switch_reason = ""
            self.status_reason = "live_ready"
            self.cancel_calls = 0

        def cancel_all_orders(self):
            self.cancel_calls += 1
            return 3

    live_trader = FakeLiveTrader()
    container = types.SimpleNamespace(live_trader=live_trader)
    monkeypatch.setenv("LIVE_EXTERNAL_KILL_SWITCH_FILE", str(kill_file))

    assert check_file_kill_switch(container) is True
    assert getattr(container, "_file_kill_switch_triggered", False) is True
    assert getattr(container, "_stop_requested", False) is True
    assert live_trader.kill_switch_active is True
    assert live_trader._kill_switch_reason == f"file:{kill_file}"
    assert live_trader.status_reason == "external_kill_switch"
    assert live_trader.cancel_calls == 1

    assert check_file_kill_switch(container) is True
    assert live_trader.cancel_calls == 1


def test_shutdown_cancels_live_orders_once():
    class FakeLiveTrader:
        def __init__(self):
            self.dry_run = False
            self.cancel_calls = 0

        def cancel_all_orders(self):
            self.cancel_calls += 1
            return 2

    bot = main.HyperliquidResearchBot.__new__(main.HyperliquidResearchBot)
    bot.container = types.SimpleNamespace(live_trader=FakeLiveTrader())
    bot.logger = logging.getLogger("test-shutdown-cancel")
    bot._shutdown_orders_cancelled = False

    bot._cancel_live_orders_for_shutdown("signal:test")
    bot._cancel_live_orders_for_shutdown("run_loop_exit")

    assert bot.container.live_trader.cancel_calls == 1


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
    assert len(captured_actions) == 2
    assert captured_actions[0]["type"] == "updateLeverage"
    assert captured_actions[0]["asset"] == 42
    assert captured_actions[0]["isCross"] is True
    assert captured_actions[0]["leverage"] == 1
    order = captured_actions[1]["orders"][0]
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


def test_place_market_order_rejects_when_leverage_update_fails(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    captured_actions = []

    def fake_post(self, action, dry_run_override=None):
        captured_actions.append(action)
        if action.get("type") == "updateLeverage":
            return {"status": "error", "message": "boom"}
        return {"status": "ok"}

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 100.0)
    monkeypatch.setattr(LiveTrader, "_post_order", fake_post)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=20.0)
    trader.asset_index_map = {"BTC": 0}
    trader.sz_decimals_map = {"BTC": 5}

    result = trader.place_market_order("BTC", "buy", 0.12, leverage=3)

    assert result["status"] == "rejected"
    assert result["reason"] == "leverage_update_failed"
    assert len(captured_actions) == 1
    assert captured_actions[0]["type"] == "updateLeverage"


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
    monkeypatch.setattr("src.core.live_execution.db.get_open_paper_trades", lambda: [])
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
        lambda self, **kw: [
            {"coin": "ETH", "size": 0.0058, "szi": -0.0058, "side": "short", "entry_price": 2064.9, "entryPx": 2064.9},
            {"coin": "BTC", "size": 0.00018, "szi": -0.00018, "side": "short", "entry_price": 67401.0, "entryPx": 67401.0},
        ],
    )
    monkeypatch.setattr(LiveTrader, "get_open_orders", lambda self, **kw: [])  # no protection
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
    # 3% SL, 15% TP fallback
    assert abs(sl["price"] - 2064.9 * 1.03) < 0.01
    assert abs(tp["price"] - 2064.9 * 0.85) < 0.01


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
        lambda self, **kw: [
            {"coin": "ETH", "size": 0.0058, "szi": -0.0058, "side": "short", "entry_price": 2064.9, "entryPx": 2064.9},
        ],
    )
    # P0-4: valid legs require side + triggerPx + adequate size, on the
    # correct side of entry for the position (short ETH → buy-side
    # protectors; SL above entry; TP below entry).
    monkeypatch.setattr(
        LiveTrader,
        "get_open_orders",
        lambda self, **kw: [
            {
                "coin": "ETH",
                "isTrigger": True,
                "isPositionTpsl": True,
                "reduceOnly": True,
                "orderType": "Stop Market",
                "side": "B",
                "sz": "0",
                "origSz": "0.0058",
                "triggerPx": "2126.8",
                "triggerCondition": "Price above 2126.8",
                "oid": 11,
            },
            {
                "coin": "ETH",
                "isTrigger": True,
                "isPositionTpsl": True,
                "reduceOnly": True,
                "orderType": "Take Profit Market",
                "side": "B",
                "sz": "0.0058",
                "triggerPx": "1755.2",
                "triggerCondition": "Price below 1755.2",
                "oid": 12,
            },
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


def test_protect_orphaned_positions_cancels_duplicate_frontend_tpsl_orders(monkeypatch):
    placed_triggers = []
    cancelled = []

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(
        LiveTrader,
        "get_positions",
        lambda self, **kw: [
            {
                "coin": "ETH",
                "size": 0.0058,
                "szi": -0.0058,
                "side": "short",
                "entry_price": 2064.9,
                "entryPx": 2064.9,
            },
        ],
    )
    monkeypatch.setattr(
        LiveTrader,
        "get_open_orders",
        lambda self, **kw: [
            {
                "coin": "ETH",
                "isTrigger": True,
                "isPositionTpsl": True,
                "reduceOnly": True,
                "orderType": "Stop Market",
                "side": "B",
                "sz": "0.0058",
                "triggerPx": "2126.8",
                "oid": 11,
            },
            {
                "coin": "ETH",
                "isTrigger": True,
                "isPositionTpsl": True,
                "reduceOnly": True,
                "orderType": "Stop Market",
                "side": "B",
                "sz": "0.0058",
                "triggerPx": "2126.8",
                "oid": 13,
            },
            {
                "coin": "ETH",
                "isTrigger": True,
                "isPositionTpsl": True,
                "reduceOnly": True,
                "orderType": "Take Profit Market",
                "side": "B",
                "sz": "0.0058",
                "triggerPx": "1755.2",
                "oid": 12,
            },
            {
                "coin": "ETH",
                "isTrigger": True,
                "isPositionTpsl": True,
                "reduceOnly": True,
                "orderType": "Take Profit Market",
                "side": "B",
                "sz": "0.0058",
                "triggerPx": "1755.2",
                "oid": 14,
            },
        ],
    )
    monkeypatch.setattr(
        LiveTrader,
        "cancel_order",
        lambda self, coin, oid: cancelled.append((coin, oid)) or True,
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
    assert summary["stale_cancelled"] == 2
    assert cancelled == [("ETH", 11), ("ETH", 12)]
    assert placed_triggers == []


def test_protect_orphaned_positions_churn_guard_blocks_repeated_brackets(monkeypatch):
    placed_triggers = []

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(
        LiveTrader,
        "get_positions",
        lambda self, **kw: [
            {
                "coin": "SOL",
                "size": 0.13,
                "szi": 0.13,
                "side": "long",
                "entry_price": 85.947,
                "entryPx": 85.947,
            },
        ],
    )
    monkeypatch.setattr(LiveTrader, "get_open_orders", lambda self, **kw: [])
    monkeypatch.setattr(
        LiveTrader,
        "place_trigger_order",
        lambda self, coin, side, size, trigger_price, tp_or_sl="sl": (
            placed_triggers.append((coin, side, size, trigger_price, tp_or_sl))
            or {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"resting": {"oid": len(placed_triggers)}}]}}}
        ),
    )

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=15.0)
    first = trader.protect_orphaned_positions()
    second = trader.protect_orphaned_positions()

    assert first["protected"] == 1
    assert second["protected"] == 0
    assert second["failed"] == 1
    assert second["churn_blocked"] == 1
    assert len(placed_triggers) == 2
    assert trader.kill_switch_active is False
    assert trader._is_protective_churn_quarantined("SOL") is True
    assert "SOL" in trader.get_stats()["protective_churn_quarantined_coins"]


def test_protective_churn_guard_persists_across_restart(monkeypatch):
    placed_triggers = []
    bot_state = {}

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    def _get_state(key, default=None):
        return bot_state.get(key, default)

    def _set_state(key, value):
        bot_state[key] = value
        return True

    monkeypatch.setattr("src.trading.live_trader.db.get_bot_state", _get_state)
    monkeypatch.setattr("src.trading.live_trader.db.set_bot_state", _set_state)
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(
        LiveTrader,
        "get_positions",
        lambda self, **kw: [
            {
                "coin": "SOL",
                "size": 0.13,
                "szi": 0.13,
                "side": "long",
                "entry_price": 85.947,
                "entryPx": 85.947,
            },
        ],
    )
    monkeypatch.setattr(LiveTrader, "get_open_orders", lambda self, **kw: [])
    monkeypatch.setattr(
        LiveTrader,
        "place_trigger_order",
        lambda self, coin, side, size, trigger_price, tp_or_sl="sl": (
            placed_triggers.append((coin, side, size, trigger_price, tp_or_sl))
            or {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"resting": {"oid": len(placed_triggers)}}]}}}
        ),
    )

    first_trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=15.0)
    first = first_trader.protect_orphaned_positions()
    second_trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=15.0)
    second = second_trader.protect_orphaned_positions()

    assert first["protected"] == 1
    assert second["protected"] == 0
    assert second["failed"] == 1
    assert second["churn_blocked"] == 1
    assert len(placed_triggers) == 2
    assert second_trader.kill_switch_active is False
    assert second_trader._is_protective_churn_quarantined("SOL") is True
    assert "SOL" in bot_state[LiveTrader._PROTECTIVE_CHURN_QUARANTINE_STATE_KEY]["coins"]


def test_execute_signal_rejects_only_quarantined_churn_coin(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=15.0)
    monkeypatch.setattr(trader, "_refresh_external_kill_switch", lambda: False)
    monkeypatch.setattr(trader, "check_daily_loss", lambda refresh_from_fills=True: False)
    monkeypatch.setattr(trader, "_get_source_policy", lambda signal: {})
    monkeypatch.setattr("src.trading.live_trader.db.set_bot_state", lambda *args, **kwargs: True)
    trader._quarantine_protective_churn_coin("TRUMP", "test")

    result = trader.execute_signal(
        TradeSignal(
            coin="TRUMP",
            side=SignalSide.LONG,
            confidence=0.80,
            source=SignalSource.STRATEGY,
            reason="test",
            risk=RiskParams(stop_loss_pct=0.10, take_profit_pct=0.50, risk_basis="roe"),
            leverage=2,
            size=0.01,
        ),
        bypass_firewall=True,
    )

    assert result is None
    assert trader.kill_switch_active is False
    assert trader._entry_metrics["rejected_protective_churn_coin"] == 1


def test_startup_reconcile_clears_restored_unresolved_dedup_markers(monkeypatch):
    now = datetime.now(timezone.utc).timestamp()
    bot_state = {
        LiveTrader._ORDER_DEDUP_STATE_KEY: {
            "h_timeout": {"ts": now, "payload": {"status": "timeout"}},
            "h_success": {"ts": now, "payload": {"status": "ok"}},
        }
    }

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    def _get_state(key, default=None):
        return bot_state.get(key, default)

    def _set_state(key, value):
        bot_state[key] = value
        return True

    monkeypatch.setattr("src.trading.live_trader.db.get_bot_state", _get_state)
    monkeypatch.setattr("src.trading.live_trader.db.set_bot_state", _set_state)
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_positions", lambda self, **kw: [])
    monkeypatch.setattr(LiveTrader, "get_open_orders", lambda self, **kw: [])

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=15.0)

    assert trader._order_dedup_reconcile_required is False
    assert trader._order_dedup_unresolved_restored == 0
    assert "h_timeout" not in trader._recent_order_hashes
    assert "h_success" in trader._recent_order_hashes
    assert "h_timeout" not in bot_state[LiveTrader._ORDER_DEDUP_STATE_KEY]
    assert "h_success" in bot_state[LiveTrader._ORDER_DEDUP_STATE_KEY]


def test_unresolved_dedup_marker_blocks_entries_until_safe_reconcile(monkeypatch):
    now = datetime.now(timezone.utc).timestamp()
    bot_state = {
        LiveTrader._ORDER_DEDUP_STATE_KEY: {
            "h_timeout": {"ts": now, "payload": {"status": "timeout"}},
        }
    }
    placed = []

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            raise AssertionError("firewall should not run before dedup reconcile clears")

    def _get_state(key, default=None):
        return bot_state.get(key, default)

    def _set_state(key, value):
        bot_state[key] = value
        return True

    monkeypatch.setattr("src.trading.live_trader.db.get_bot_state", _get_state)
    monkeypatch.setattr("src.trading.live_trader.db.set_bot_state", _set_state)
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_positions", lambda self, **kw: [])
    monkeypatch.setattr(
        LiveTrader,
        "get_open_orders",
        lambda self, **kw: [
            {
                "coin": "BTC",
                "oid": 123,
                "orderType": "Limit",
                "side": "buy",
                "sz": "0.001",
            }
        ],
    )
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self, **kw: None)
    monkeypatch.setattr(
        LiveTrader,
        "place_market_order",
        lambda self, *args, **kwargs: placed.append(args) or {"status": "success"},
    )

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
    result = trader.execute_signal(
        {
            "coin": "BTC",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 76000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.001,
            "strategy_type": "momentum_long",
        }
    )

    assert result is None
    assert trader._order_dedup_reconcile_required is True
    assert placed == []
    assert (
        trader.get_stats()["entry_metrics"]["rejected_order_dedup_reconcile_required"]
        == 1
    )


def test_classify_protective_leg_handles_close_long_short_side_strings():
    close_long = LiveTrader._classify_protective_leg(
        {
            "coin": "BTC",
            "isTrigger": True,
            "isPositionTpsl": True,
            "reduceOnly": True,
            "orderType": "Stop Market",
            "side": "Close Long",
            "origSz": "0.00015",
            "triggerPx": "73979",
            "oid": 101,
        }
    )
    close_short = LiveTrader._classify_protective_leg(
        {
            "coin": "ETH",
            "isTrigger": True,
            "isPositionTpsl": True,
            "reduceOnly": True,
            "orderType": "Take Profit Market",
            "direction": "Close Short",
            "origSz": "0.01",
            "triggerPx": "1800",
            "oid": 102,
        }
    )

    assert close_long["side"] == "sell"
    assert close_long["leg"] == "sl"
    assert close_short["side"] == "buy"
    assert close_short["leg"] == "tp"


def test_classify_protective_leg_infers_leg_from_trigger_condition():
    long_sl = LiveTrader._classify_protective_leg(
        {
            "coin": "BTC",
            "isTrigger": True,
            "isPositionTpsl": True,
            "reduceOnly": True,
            "side": "Close Long",
            "triggerCondition": "Price below 73979",
            "origSz": "0.00015",
            "triggerPx": "73979",
            "oid": 201,
        }
    )
    long_tp = LiveTrader._classify_protective_leg(
        {
            "coin": "BTC",
            "isTrigger": True,
            "isPositionTpsl": True,
            "reduceOnly": True,
            "side": "Close Long",
            "triggerCondition": "Price above 87707",
            "origSz": "0.00015",
            "triggerPx": "87707",
            "oid": 202,
        }
    )
    short_sl = LiveTrader._classify_protective_leg(
        {
            "coin": "ETH",
            "isTrigger": True,
            "isPositionTpsl": True,
            "reduceOnly": True,
            "direction": "Close Short",
            "triggerCondition": "Price above 2100",
            "origSz": "0.01",
            "triggerPx": "2100",
            "oid": 203,
        }
    )
    short_tp = LiveTrader._classify_protective_leg(
        {
            "coin": "ETH",
            "isTrigger": True,
            "isPositionTpsl": True,
            "reduceOnly": True,
            "direction": "Close Short",
            "triggerCondition": "Price below 1800",
            "origSz": "0.01",
            "triggerPx": "1800",
            "oid": 204,
        }
    )

    assert (long_sl["side"], long_sl["leg"]) == ("sell", "sl")
    assert (long_tp["side"], long_tp["leg"]) == ("sell", "tp")
    assert (short_sl["side"], short_sl["leg"]) == ("buy", "sl")
    assert (short_tp["side"], short_tp["leg"]) == ("buy", "tp")


def test_get_open_orders_uses_plain_open_orders_fallback_when_frontend_empty():
    class FakeApiManager:
        def __init__(self):
            self.calls = []

        def post(self, payload, **kwargs):
            self.calls.append(payload["type"])
            if payload["type"] == "frontendOpenOrders":
                return []
            return [
                {
                    "coin": "BTC",
                    "isTrigger": True,
                    "isPositionTpsl": True,
                    "reduceOnly": True,
                    "orderType": "Stop Market",
                    "side": "Close Long",
                    "origSz": "0.00015",
                    "triggerPx": "73979",
                    "oid": 101,
                }
            ]

    trader = LiveTrader.__new__(LiveTrader)
    trader.public_address = "0x2222222222222222222222222222222222222222"
    trader.api_manager = FakeApiManager()
    trader._state_lock = threading.Lock()
    trader._last_order_visibility_status = {}

    orders = trader.get_open_orders(force_fresh=True)

    assert trader.api_manager.calls == ["frontendOpenOrders", "openOrders"]
    assert len(orders) == 1
    assert orders[0]["oid"] == 101


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
    monkeypatch.setattr("src.core.live_execution.db.get_open_paper_trades", lambda: [])
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
    monkeypatch.setattr("src.core.live_execution.db.get_open_paper_trades", lambda: [])
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


def test_live_trader_persists_kill_switch_across_restart_and_daily_reset(tmp_path, monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    state_file = tmp_path / "sticky_kill.json"
    monkeypatch.setenv("LIVE_KILL_SWITCH_STATE_FILE", str(state_file))
    monkeypatch.setattr(config, "LIVE_KILL_SWITCH_STATE_FILE", str(state_file), raising=False)
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)

    first = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
    first.activate_kill_switch("operator_test", status_reason="manual_kill_switch")
    first.daily_reset_date = "2000-01-01"
    first._check_daily_reset()

    assert first.kill_switch_active is True
    assert first._kill_switch_reason == "operator_test"

    second = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
    stats = second.get_stats()
    assert stats["kill_switch_active"] is True
    assert stats["kill_switch_reason"] == "operator_test"


def test_daily_pnl_refresh_failure_fails_closed(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    class BadApiManager:
        def post(self, *args, **kwargs):
            raise RuntimeError("api down")

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
    trader.api_manager = BadApiManager()

    assert trader.update_daily_pnl_from_fills() is False
    assert trader.kill_switch_active is True
    assert trader._kill_switch_reason == "daily_pnl_refresh_failed"


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


def test_live_trader_source_cap_atomic_check_reserve_is_race_free(monkeypatch):
    """H10 (audit): the canary/source-cap check and the counter increment
    must happen atomically under ``_state_lock``.  Before H10, the old
    "check now, increment on success" pattern let N concurrent signals
    all read count=0 under max=1, all pass the cap check, and all place
    orders — overshooting the cap by the concurrency factor.

    This test exercises the reservation helper directly so it's
    deterministic (no cross-thread ordering assumptions): repeatedly
    reserving with max=3 should yield exactly 3 True followed by False.
    """
    monkeypatch.delenv("LIVE_EXTERNAL_KILL_SWITCH", raising=False)
    monkeypatch.setattr(config, "LIVE_MAX_ORDERS_PER_SOURCE_PER_DAY", 3)
    monkeypatch.setattr(config, "LIVE_CANARY_MODE", False)
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)

    trader = LiveTrader(firewall=None, dry_run=False, max_order_usd=1_000)

    results = [trader._reserve_entry_slot("copy_trade") for _ in range(5)]
    assert [r[0] for r in results] == [True, True, True, False, False], (
        f"expected 3 reservations then 2 rejections, got {results}"
    )
    # Reserved slots must show in the counter
    assert trader._source_orders_today["copy_trade"] == 3
    # Releasing rolls back the reservation
    trader._release_entry_slot("copy_trade")
    assert trader._source_orders_today["copy_trade"] == 2
    # Further releases remove the key entirely when it hits zero
    trader._release_entry_slot("copy_trade")
    trader._release_entry_slot("copy_trade")
    assert "copy_trade" not in trader._source_orders_today


def test_live_trader_source_cap_releases_reservation_on_entry_failure(monkeypatch):
    """H10: when place_market_order fails after the slot has been
    reserved, the counter must be released so the failed attempt does
    not permanently consume budget."""
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
    # Simulate exchange rejection - "insufficient margin" style failure
    monkeypatch.setattr(
        LiveTrader,
        "place_market_order",
        lambda self, *args, **kwargs: {"status": "rejected", "reason": "exchange_error"},
    )
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000)

    # First attempt fails → slot should be released
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
    # Returns the (rejected) order_result, not None
    assert first is not None
    assert first.get("status") == "rejected"
    # Critical: counter must have been released (budget intact)
    assert trader._source_orders_today.get("copy_trade", 0) == 0, (
        "failed entry must not permanently consume source-cap budget"
    )


def test_live_trader_source_cap_concurrent_signals_stay_within_cap(monkeypatch):
    """H10: fire 20 concurrent signals against a cap of 5 and verify
    that *at most* 5 succeed.  Without the atomic reserve, several
    threads can all pass the check and several orders can be placed
    past the cap.
    """
    import threading as _threading

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.delenv("LIVE_EXTERNAL_KILL_SWITCH", raising=False)
    monkeypatch.setattr(config, "LIVE_MAX_ORDERS_PER_SOURCE_PER_DAY", 5)
    monkeypatch.setattr(config, "LIVE_CANARY_MODE", False)
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(
        LiveTrader,
        "place_market_order",
        lambda self, *args, **kwargs: {"status": "success"},
    )
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(
        LiveTrader,
        "verify_fill",
        lambda self, *args, **kwargs: {"status": "verified", "size": 0.1},
    )
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(LiveTrader, "place_trigger_order", lambda self, *args, **kwargs: {"status": "success"})

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)

    results = []
    results_lock = _threading.Lock()
    start_barrier = _threading.Barrier(20)

    def _fire():
        start_barrier.wait()
        r = trader.execute_signal(
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
        with results_lock:
            results.append(r)

    threads = [_threading.Thread(target=_fire) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successes = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
    assert len(successes) <= 5, (
        f"H10 race guard: at most 5 of 20 concurrent signals should succeed, "
        f"got {len(successes)}"
    )
    # And the counter should match the number of successes exactly
    assert trader._source_orders_today.get("copy_trade", 0) == len(successes), (
        f"counter {trader._source_orders_today.get('copy_trade', 0)} must match "
        f"successes {len(successes)}"
    )


def test_live_trader_source_day_cap_scopes_copy_trades_per_trader(monkeypatch):
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
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0 if coin == "ETH" else 60000.0)
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
            "trader_address": "0xAAA",
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
            "trader_address": "0xBBB",
        }
    )

    assert first is not None and first["status"] == "success"
    assert second is not None and second["status"] == "success"
    stats = trader.get_stats()
    assert stats["source_orders_today"]["copy_trade:0xaaa"] == 1
    assert stats["source_orders_today"]["copy_trade:0xbbb"] == 1
    assert stats["total_entry_signals_today"] == 2


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


def test_post_order_uses_monotonic_nonce_with_same_millisecond(monkeypatch):
    nonces = []

    class FakeManager:
        def post(self, payload, **kwargs):
            return {"status": "ok"}

    class FakeSigner:
        address = "0x1111111111111111111111111111111111111111"

        def sign_action(self, action, nonce, vault_address=None):
            nonces.append(nonce)
            return {"r": "0x1", "s": "0x2", "v": 27}

    def _load_credentials(self):
        self.signer = FakeSigner()
        self.agent_wallet_address = self.signer.address
        self.public_address = "0x2222222222222222222222222222222222222222"
        self.status_reason = "credentials_loaded"

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr("src.trading.live_trader.get_manager", lambda: FakeManager())
    monkeypatch.setattr(LiveTrader, "_load_credentials", _load_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr("src.trading.live_trader.time.time", lambda: 1700000000.123)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False)
    trader._post_order({"type": "order", "orders": [{"id": 1}]})
    trader._post_order({"type": "order", "orders": [{"id": 2}]})

    assert len(nonces) == 2
    assert nonces[1] == nonces[0] + 1


def test_protect_orphaned_positions_aborts_when_open_orders_unavailable(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    trigger_calls = []

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(
        LiveTrader,
        "get_positions",
        lambda self, **kw: [
            {
                "coin": "ETH",
                "size": 0.1,
                "szi": 0.1,
                "side": "long",
                "entry_price": 2000.0,
                "entryPx": 2000.0,
            }
        ],
    )
    monkeypatch.setattr(LiveTrader, "get_open_orders", lambda self, **kw: None)
    monkeypatch.setattr(
        LiveTrader,
        "place_trigger_order",
        lambda self, *args, **kwargs: trigger_calls.append((args, kwargs)) or {"status": "success"},
    )

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=15.0)
    summary = trader.protect_orphaned_positions()

    assert summary["status"] == "degraded"
    assert summary["reason"] == "open_orders_unavailable"
    assert trigger_calls == []


def test_execute_signal_rejects_when_live_positions_are_unavailable(monkeypatch):
    class FakeFirewall:
        def __init__(self):
            self.calls = 0

        def validate(self, signal, **kwargs):
            self.calls += 1
            return True, "ok"

    firewall = FakeFirewall()

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self, trigger_check=True: None)

    trader = LiveTrader(firewall=firewall, dry_run=False, max_order_usd=1_000_000)
    result = trader.execute_signal(
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

    assert result is None
    assert firewall.calls == 0


def test_execute_signal_refreshes_daily_pnl_before_firewall(monkeypatch):
    class FakeFirewall:
        def __init__(self):
            self.calls = 0

        def validate(self, signal, **kwargs):
            self.calls += 1
            return True, "ok"

    firewall = FakeFirewall()

    def _refresh_daily_pnl(self, trigger_check=True):
        self.daily_pnl = -150.0

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", _refresh_daily_pnl)

    trader = LiveTrader(
        firewall=firewall,
        dry_run=False,
        max_daily_loss=100.0,
        max_order_usd=1_000_000,
    )
    result = trader.execute_signal(
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

    assert result is None
    assert firewall.calls == 0
    assert trader.kill_switch_active is True


# ─────────────────────────────────────────────────────────────────
# AUDIT M2 — selective protective-leg retry.  Previously, any single-
# leg failure caused ``cancel_all_orders`` to wipe BOTH legs before
# retrying.  The new logic re-places only the failed leg and leaves
# the good leg resting on the exchange across the retry window.
# ─────────────────────────────────────────────────────────────────


class _FakeFirewall:
    def validate(self, signal, **kwargs):
        return True, "ok"


def _build_trader_for_protective_tests(monkeypatch):
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    trader = LiveTrader(firewall=_FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
    # Tighten retries so tests execute quickly regardless of env defaults.
    trader._protective_order_retries = 3
    trader._protective_order_retry_delay_s = 0.0
    trader._protective_order_retry_jitter_s = 0.0
    return trader


def test_protective_retry_reuses_successful_sl_when_tp_fails_first(monkeypatch):
    """AUDIT M2: SL succeeds on attempt 1, TP fails; retry must NOT re-place SL."""
    trader = _build_trader_for_protective_tests(monkeypatch)

    tp_call_count = {"n": 0}
    sl_call_count = {"n": 0}

    def _fake_trigger(self, coin, side, size, trigger_price, tp_or_sl="sl"):
        if tp_or_sl == "sl":
            sl_call_count["n"] += 1
            return {"status": "success", "which": "sl"}
        # TP: fail first attempt, succeed second
        tp_call_count["n"] += 1
        if tp_call_count["n"] == 1:
            return {"status": "error", "message": "tp placement rejected"}
        return {"status": "success", "which": "tp"}

    cancel_calls = []

    def _fake_cancel_all(self, coin=None):
        cancel_calls.append(coin)
        return 0

    monkeypatch.setattr(LiveTrader, "place_trigger_order", _fake_trigger)
    monkeypatch.setattr(LiveTrader, "cancel_all_orders", _fake_cancel_all)

    sl_result, tp_result, attempts = trader._place_protective_orders_with_retries(
        coin="ETH",
        close_side="sell",
        size=0.1,
        sl_price=1900.0,
        tp_price=2100.0,
    )

    # SL should have been placed exactly ONCE (first attempt, succeeded).
    assert sl_call_count["n"] == 1, (
        f"SL was re-placed after success (saw {sl_call_count['n']} calls)"
    )
    # TP should have been placed TWICE (failed then succeeded).
    assert tp_call_count["n"] == 2
    assert trader._is_order_result_success(sl_result)
    assert trader._is_order_result_success(tp_result)
    assert attempts == 2
    # The between-retry cancel_all_orders must NOT have been called — it
    # would have wiped the good SL.
    assert cancel_calls == [], (
        f"Retry loop called cancel_all_orders (wipes good SL): {cancel_calls}"
    )


def test_protective_retry_reuses_successful_tp_when_sl_fails_first(monkeypatch):
    """AUDIT M2: TP succeeds on attempt 1, SL fails; retry must NOT re-place TP."""
    trader = _build_trader_for_protective_tests(monkeypatch)

    sl_call_count = {"n": 0}
    tp_call_count = {"n": 0}

    def _fake_trigger(self, coin, side, size, trigger_price, tp_or_sl="sl"):
        if tp_or_sl == "tp":
            tp_call_count["n"] += 1
            return {"status": "success", "which": "tp"}
        sl_call_count["n"] += 1
        if sl_call_count["n"] == 1:
            return {"status": "error", "message": "sl placement rejected"}
        return {"status": "success", "which": "sl"}

    cancel_calls = []

    def _fake_cancel_all(self, coin=None):
        cancel_calls.append(coin)
        return 0

    monkeypatch.setattr(LiveTrader, "place_trigger_order", _fake_trigger)
    monkeypatch.setattr(LiveTrader, "cancel_all_orders", _fake_cancel_all)

    sl_result, tp_result, attempts = trader._place_protective_orders_with_retries(
        coin="ETH",
        close_side="sell",
        size=0.1,
        sl_price=1900.0,
        tp_price=2100.0,
    )

    assert tp_call_count["n"] == 1
    assert sl_call_count["n"] == 2
    assert trader._is_order_result_success(sl_result)
    assert trader._is_order_result_success(tp_result)
    assert attempts == 2
    assert cancel_calls == []


def test_protective_retry_returns_immediately_when_both_succeed(monkeypatch):
    """Both legs succeed on attempt 1 → attempts=1, no cancel call."""
    trader = _build_trader_for_protective_tests(monkeypatch)

    place_calls = []
    cancel_calls = []

    def _fake_trigger(self, coin, side, size, trigger_price, tp_or_sl="sl"):
        place_calls.append(tp_or_sl)
        return {"status": "success"}

    monkeypatch.setattr(LiveTrader, "place_trigger_order", _fake_trigger)
    monkeypatch.setattr(
        LiveTrader,
        "cancel_all_orders",
        lambda self, coin=None: cancel_calls.append(coin) or 0,
    )

    sl_result, tp_result, attempts = trader._place_protective_orders_with_retries(
        coin="ETH",
        close_side="sell",
        size=0.1,
        sl_price=1900.0,
        tp_price=2100.0,
    )

    assert place_calls == ["sl", "tp"]
    assert attempts == 1
    assert cancel_calls == []
    assert trader._is_order_result_success(sl_result)
    assert trader._is_order_result_success(tp_result)


def test_protective_retry_exhausts_when_one_leg_never_succeeds(monkeypatch):
    """AUDIT M2: one leg permanently broken → returns last result, attempts==max,
    surviving leg is NOT re-placed after its first success."""
    trader = _build_trader_for_protective_tests(monkeypatch)
    trader._protective_order_retries = 3

    sl_call_count = {"n": 0}
    tp_call_count = {"n": 0}

    def _fake_trigger(self, coin, side, size, trigger_price, tp_or_sl="sl"):
        if tp_or_sl == "sl":
            sl_call_count["n"] += 1
            return {"status": "success", "which": "sl"}
        tp_call_count["n"] += 1
        return {"status": "error", "message": "tp always fails"}

    cancel_calls = []
    monkeypatch.setattr(LiveTrader, "place_trigger_order", _fake_trigger)
    monkeypatch.setattr(
        LiveTrader,
        "cancel_all_orders",
        lambda self, coin=None: cancel_calls.append(coin) or 0,
    )

    sl_result, tp_result, attempts = trader._place_protective_orders_with_retries(
        coin="ETH",
        close_side="sell",
        size=0.1,
        sl_price=1900.0,
        tp_price=2100.0,
    )

    # SL placed once, kept for all attempts.
    assert sl_call_count["n"] == 1
    # TP attempted once per iteration (3 iterations).
    assert tp_call_count["n"] == 3
    assert attempts == 3
    assert trader._is_order_result_success(sl_result)
    assert not trader._is_order_result_success(tp_result)
    # Still no between-retry cancel — SL survives.
    assert cancel_calls == []
