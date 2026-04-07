import importlib.util
import json
import logging
import os
import sqlite3
import sys
import tempfile
import types
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import config
import main
import src.analysis.daily_research_loop as daily_research_module
import src.analysis.shadow_certification as shadow_certification_module
from src.analysis.daily_research_loop import DailyResearchLoop
from src.analysis.shadow_certification import build_shadow_certification_report
from src.analysis.experiment_discipline import build_experiment_benchmark_pack
from src.core import boot
from src.core.incident_visibility import build_runtime_incident_report
from src.core.time_utils import utc_now, utc_now_iso, utc_now_naive
from src.core.api_manager import Priority
from src.core.cycles.reporting_cycle import run_reporting
from src.core.cycles.trading_cycle import (
    _apply_agent_scorer_weights,
    _execute_selected_decisions,
    _execute_lcrs_signals,
    _execute_options_flow_trades,
    _process_closed_trades,
    _run_alpha_arena,
    run_trading_cycle,
)
from src.core.health_registry import SubsystemHealthRegistry, SubsystemState
from src.core.live_execution import (
    _rescale_size_for_live,
    get_execution_open_positions,
    mirror_executed_trades_to_live,
    sync_shadow_book_to_live,
)
from src.data import database as db
from src.signals.adaptive_learning import AdaptiveLearningManager
from src.signals.alpha_arena import AgentStatus, ArenaAgent, CapitalAllocator
from src.signals.capital_governor import CapitalGovernor
from src.signals.decision_engine import DecisionEngine
from src.signals.decision_firewall import DecisionFirewall
from src.signals.divergence_controller import DivergenceController
from src.signals.execution_policy import ExecutionPolicyManager
from src.signals.portfolio_sizer import PortfolioSizer
from src.signals.source_allocator import SourceBudgetAllocator
from src.signals.signal_schema import RiskParams, SignalSide, SignalSource, TradeSignal, signal_from_execution_dict
from src.trading.live_trader import (
    HyperliquidSigner,
    LiveTrader,
    _hl_format_price,
    _hl_format_size,
)
from src.trading.portfolio_rotation import PortfolioRotationManager, RotationDecision
from src.ui import dashboard as dashboard_ui
from src.notifications import telegram_alerts


def _fake_live_credentials(self):
    self.signer = type("Signer", (), {"address": "0x1111111111111111111111111111111111111111"})()
    self.agent_wallet_address = self.signer.address
    self.public_address = "0x2222222222222222222222222222222222222222"
    self.status_reason = "credentials_loaded"


def _missing_live_credentials(self):
    self.signer = None
    self.agent_wallet_address = None
    self.public_address = None
    self.status_reason = "missing_agent_wallet_signer"


def _load_live_preflight_script():
    script_path = os.path.join(os.getcwd(), "scripts", "run_live_preflight.py")
    spec = importlib.util.spec_from_file_location("test_run_live_preflight", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_daily_research_script():
    script_path = os.path.join(os.getcwd(), "scripts", "run_daily_research_loop.py")
    spec = importlib.util.spec_from_file_location("test_run_daily_research_loop", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_shadow_certification_script():
    script_path = os.path.join(os.getcwd(), "scripts", "run_shadow_certification.py")
    spec = importlib.util.spec_from_file_location("test_run_shadow_certification", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_shadow_runtime_profile_allows_missing_live_envs(monkeypatch):
    logger = logging.getLogger("test-shadow-profile")
    monkeypatch.setattr(config, "RUNTIME_PROFILE", "shadow")
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", False)
    monkeypatch.setattr(config, "LIVE_PREFLIGHT_REQUIRED", False)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_GUARD_ENABLED", False)

    for env_name in (
        "LIVE_TRADING_ENABLED",
        "LIVE_MIN_ORDER_USD",
        "LIVE_MAX_ORDER_USD",
        "LIVE_MAX_DAILY_LOSS_USD",
        "LIVE_PREFLIGHT_REQUIRED",
        "LIVE_ACTIVATION_GUARD_ENABLED",
        "LIVE_ACTIVATION_APPROVED_AT",
        "LIVE_ACTIVATION_APPROVED_BY",
        "LIVE_ACTIVATION_MAX_AGE_HOURS",
        "HL_WALLET_MODE",
        "SECRET_MANAGER_PROVIDER",
    ):
        monkeypatch.delenv(env_name, raising=False)

    boot.validate_runtime_profile_controls(logger)


def test_live_runtime_profile_requires_explicit_live_envs(monkeypatch):
    logger = logging.getLogger("test-live-profile")
    monkeypatch.setattr(config, "RUNTIME_PROFILE", "live")
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_PREFLIGHT_REQUIRED", True)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_GUARD_ENABLED", True)

    for env_name in (
        "LIVE_TRADING_ENABLED",
        "LIVE_MIN_ORDER_USD",
        "LIVE_MAX_ORDER_USD",
        "LIVE_MAX_DAILY_LOSS_USD",
        "LIVE_PREFLIGHT_REQUIRED",
        "LIVE_ACTIVATION_GUARD_ENABLED",
        "LIVE_ACTIVATION_APPROVED_AT",
        "LIVE_ACTIVATION_APPROVED_BY",
        "LIVE_ACTIVATION_MAX_AGE_HOURS",
        "HL_WALLET_MODE",
        "SECRET_MANAGER_PROVIDER",
    ):
        monkeypatch.delenv(env_name, raising=False)

    try:
        boot.validate_runtime_profile_controls(logger)
        assert False, "Expected RuntimeError for incomplete live runtime profile envs"
    except RuntimeError as exc:
        assert "BOT_RUNTIME_PROFILE=live" in str(exc)
        assert "LIVE_MIN_ORDER_USD" in str(exc)


def test_live_runtime_profile_accepts_explicit_live_envs(monkeypatch):
    logger = logging.getLogger("test-live-profile-ready")
    monkeypatch.setattr(config, "RUNTIME_PROFILE", "live")
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_PREFLIGHT_REQUIRED", True)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_GUARD_ENABLED", True)

    env_values = {
        "LIVE_TRADING_ENABLED": "true",
        "LIVE_MIN_ORDER_USD": "11",
        "LIVE_MAX_ORDER_USD": "25",
        "LIVE_MAX_DAILY_LOSS_USD": "10",
        "LIVE_PREFLIGHT_REQUIRED": "true",
        "LIVE_ACTIVATION_GUARD_ENABLED": "true",
        "LIVE_ACTIVATION_APPROVED_AT": "2026-04-07T00:00:00Z",
        "LIVE_ACTIVATION_APPROVED_BY": "codex-test",
        "LIVE_ACTIVATION_MAX_AGE_HOURS": "24",
        "HL_WALLET_MODE": "agent_only",
        "SECRET_MANAGER_PROVIDER": "none",
    }
    for env_name, value in env_values.items():
        monkeypatch.setenv(env_name, value)

    boot.validate_runtime_profile_controls(logger)


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


def test_signal_from_execution_dict_builds_external_source_key():
    signal = signal_from_execution_dict(
        {
            "coin": "BTC",
            "side": "long",
            "confidence": 0.71,
            "entry_price": 101000,
            "strategy_type": "event_driven",
            "source": "polymarket",
        }
    )

    assert signal.source.value == "polymarket"
    assert signal.source_key == "polymarket:BTC"


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


def test_strategy_scorer_reports_warming_up_when_no_strategies(monkeypatch):
    from src.analysis.strategy_scorer import StrategyScorer

    monkeypatch.setattr(db, "get_active_strategies", lambda: [])

    report = StrategyScorer().generate_improvement_report()

    assert report["status"] == "no_strategies_tracked"
    assert report["health"] == "warming_up"
    assert report["message"] == "No strategies tracked yet"


def test_run_reporting_executes_daily_research_loop_on_daily_cadence(monkeypatch):
    runs = []
    infos = []
    container = type(
        "Container",
        (),
        {
            "reporter": None,
            "paper_trader": None,
            "shadow_tracker": None,
            "cross_venue_hedger": None,
            "scorer": None,
            "adaptive_learning": object(),
        },
    )()

    monkeypatch.setattr(config, "TRADING_CYCLE_INTERVAL", 3600)
    monkeypatch.setattr(config, "DAILY_RESEARCH_LOOP_ENABLED", True)
    monkeypatch.setattr("src.core.cycles.reporting_cycle._log_module_stats", lambda container: None)
    monkeypatch.setattr("src.core.cycles.reporting_cycle.backup_to_json", lambda: None)
    monkeypatch.setattr(
        "src.core.cycles.reporting_cycle.build_runtime_incident_report",
        lambda **kwargs: {"summary": {}, "incidents": []},
    )
    monkeypatch.setattr(
        "src.analysis.daily_research_loop.run_daily_research_loop",
        lambda **kwargs: runs.append(kwargs) or {
            "recommendation": {"action": "hold", "summary": "Daily review held baseline."},
            "benchmark": {
                "promotion_gate": {
                    "winner": "baseline_current",
                    "approved_profiles": [],
                }
            },
        },
    )
    monkeypatch.setattr(
        "src.core.cycles.reporting_cycle.logger.info",
        lambda msg, *args: infos.append(msg % args if args else msg),
    )

    run_reporting(container, cycle_count=24, health_registry=None)

    assert len(runs) == 1
    assert runs[0]["adaptive_learning"] is container.adaptive_learning
    assert runs[0]["cycle_count"] == 24
    assert runs[0]["force"] is True
    assert any("DailyResearch: hold" in entry for entry in infos)


def test_run_reporting_executes_shadow_certification_on_daily_cadence(monkeypatch):
    runs = []
    infos = []
    container = type(
        "Container",
        (),
        {
            "reporter": None,
            "paper_trader": None,
            "shadow_tracker": object(),
            "cross_venue_hedger": None,
            "scorer": None,
            "adaptive_learning": object(),
            "live_trader": object(),
            "divergence_controller": object(),
            "capital_governor": object(),
        },
    )()

    monkeypatch.setattr(config, "TRADING_CYCLE_INTERVAL", 3600)
    monkeypatch.setattr(config, "DAILY_RESEARCH_LOOP_ENABLED", False)
    monkeypatch.setattr(config, "SHADOW_CERTIFICATION_ENABLED", True)
    monkeypatch.setattr("src.core.cycles.reporting_cycle._log_module_stats", lambda container: None)
    monkeypatch.setattr("src.core.cycles.reporting_cycle.backup_to_json", lambda: None)
    monkeypatch.setattr(
        "src.core.cycles.reporting_cycle.build_runtime_incident_report",
        lambda **kwargs: {"summary": {}, "incidents": []},
    )
    monkeypatch.setattr(
        "src.analysis.shadow_certification.run_shadow_certification",
        lambda **kwargs: runs.append(kwargs) or {
            "status": "failed",
            "certified": False,
            "shadow_summary": {"total_trades": 12},
            "readiness_interruptions": {"count": 1},
        },
    )
    monkeypatch.setattr(
        "src.core.cycles.reporting_cycle.logger.info",
        lambda msg, *args: infos.append(msg % args if args else msg),
    )

    run_reporting(container, cycle_count=24, health_registry=None)

    assert len(runs) == 1
    assert runs[0]["shadow_tracker"] is container.shadow_tracker
    assert runs[0]["live_trader"] is container.live_trader
    assert any("ShadowCertification: failed" in entry for entry in infos)


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


def test_live_trader_preflight_reports_ready_when_account_is_funded(monkeypatch):
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_GUARD_ENABLED", False)

    class FakeManager:
        def __init__(self):
            self.calls = []

        def post(self, payload, **kwargs):
            self.calls.append(payload.get("type"))
            if payload.get("type") == "meta":
                return {"universe": [{"name": "BTC", "szDecimals": 5}]}
            if payload.get("type") == "clearinghouseState":
                return {"marginSummary": {"accountValue": "125.0"}, "assetPositions": []}
            if payload.get("type") == "spotClearinghouseState":
                return {"balances": [{"coin": "USDC", "total": "0"}]}
            raise AssertionError(f"unexpected payload: {payload}")

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    manager = FakeManager()
    monkeypatch.setattr("src.trading.live_trader.get_manager", lambda: manager)
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False)
    report = trader.run_preflight(force=True)

    assert report["deployable"] is True
    assert report["status"] == "ready"
    assert report["blocking_checks"] == []
    assert trader.is_deployable() is True
    assert trader.get_stats()["preflight"]["deployable"] is True


def test_live_trader_preflight_blocks_unfunded_account(monkeypatch):
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_GUARD_ENABLED", False)

    class FakeManager:
        def post(self, payload, **kwargs):
            if payload.get("type") == "meta":
                return {"universe": [{"name": "BTC", "szDecimals": 5}]}
            if payload.get("type") == "clearinghouseState":
                return {"marginSummary": {"accountValue": "0"}, "assetPositions": []}
            if payload.get("type") == "spotClearinghouseState":
                return {"balances": [{"coin": "USDC", "total": "0"}]}
            raise AssertionError(f"unexpected payload: {payload}")

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr("src.trading.live_trader.get_manager", lambda: FakeManager())
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False)
    report = trader.run_preflight(force=True)

    assert report["deployable"] is False
    assert "perps_buying_power" in report["blocking_checks"]
    assert trader.is_deployable() is False
    assert trader.get_stats()["status_reason"].startswith("preflight:")


def test_live_trader_preflight_blocks_when_account_state_is_unreachable(monkeypatch):
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_GUARD_ENABLED", False)

    class FakeManager:
        def post(self, payload, **kwargs):
            if payload.get("type") == "meta":
                return {"universe": [{"name": "BTC", "szDecimals": 5}]}
            if payload.get("type") == "clearinghouseState":
                return []
            if payload.get("type") == "spotClearinghouseState":
                return {"balances": [{"coin": "USDC", "total": "0"}]}
            raise AssertionError(f"unexpected payload: {payload}")

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr("src.trading.live_trader.get_manager", lambda: FakeManager())
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False)
    report = trader.run_preflight(force=True)

    assert report["deployable"] is False
    assert "account_state_accessible" in report["blocking_checks"]
    assert "perps_buying_power" in report["blocking_checks"]
    assert trader.is_deployable() is False


def test_live_trader_preflight_blocks_when_only_spot_wallet_is_funded(monkeypatch):
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_GUARD_ENABLED", False)

    class FakeManager:
        def post(self, payload, **kwargs):
            if payload.get("type") == "meta":
                return {"universe": [{"name": "BTC", "szDecimals": 5}]}
            if payload.get("type") == "clearinghouseState":
                return {"marginSummary": {"accountValue": "0"}, "assetPositions": []}
            if payload.get("type") == "spotClearinghouseState":
                return {"balances": [{"coin": "USDC", "total": "125"}]}
            raise AssertionError(f"unexpected payload: {payload}")

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr("src.trading.live_trader.get_manager", lambda: FakeManager())
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False)
    report = trader.run_preflight(force=True)

    assert report["deployable"] is False
    assert report["wallet_total"] == 125.0
    assert report["account_value"] == 0.0
    assert "perps_buying_power" in report["blocking_checks"]
    assert trader.is_deployable() is False


def test_live_trader_requires_fresh_activation_approval(monkeypatch):
    class FakeManager:
        def post(self, payload, **kwargs):
            if payload.get("type") == "meta":
                return {"universe": [{"name": "BTC", "szDecimals": 5}]}
            if payload.get("type") == "clearinghouseState":
                return {"marginSummary": {"accountValue": "250.0"}, "assetPositions": []}
            if payload.get("type") == "spotClearinghouseState":
                return {"balances": [{"coin": "USDC", "total": "50"}]}
            return {}

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_GUARD_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_APPROVED_AT", "")
    monkeypatch.setattr(config, "LIVE_ACTIVATION_APPROVED_BY", "")
    monkeypatch.setattr("src.trading.live_trader.get_manager", lambda: FakeManager())
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False)
    trader.run_preflight(force=True)
    activation = trader.evaluate_activation_guard()
    readiness = trader.get_live_readiness()

    assert activation["deployable"] is False
    assert "approved_at_present" in activation["blocking_checks"]
    assert readiness["deployable"] is False
    assert readiness["status_reason"].startswith("activation:")
    assert trader.is_deployable() is False


def test_live_trader_blocks_expired_activation_approval(monkeypatch):
    class FakeManager:
        def post(self, payload, **kwargs):
            if payload.get("type") == "meta":
                return {"universe": [{"name": "BTC", "szDecimals": 5}]}
            if payload.get("type") == "clearinghouseState":
                return {"marginSummary": {"accountValue": "250.0"}, "assetPositions": []}
            if payload.get("type") == "spotClearinghouseState":
                return {"balances": [{"coin": "USDC", "total": "50"}]}
            return {}

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    approved_at = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_GUARD_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_APPROVED_AT", approved_at)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_APPROVED_BY", "ops")
    monkeypatch.setattr(config, "LIVE_ACTIVATION_MAX_AGE_HOURS", 24.0)
    monkeypatch.setattr("src.trading.live_trader.get_manager", lambda: FakeManager())
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False)
    readiness = trader.get_live_readiness(force_preflight=True)

    assert readiness["deployable"] is False
    assert "approval_fresh" in readiness["activation_guard"]["blocking_checks"]
    assert readiness["status_reason"].startswith("activation:approval_fresh")
    assert trader.is_deployable() is False


def test_live_trader_accepts_fresh_activation_approval(monkeypatch):
    class FakeManager:
        def post(self, payload, **kwargs):
            if payload.get("type") == "meta":
                return {"universe": [{"name": "BTC", "szDecimals": 5}]}
            if payload.get("type") == "clearinghouseState":
                return {"marginSummary": {"accountValue": "250.0"}, "assetPositions": []}
            if payload.get("type") == "spotClearinghouseState":
                return {"balances": [{"coin": "USDC", "total": "50"}]}
            return {}

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    approved_at = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_GUARD_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_APPROVED_AT", approved_at)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_APPROVED_BY", "codex-test")
    monkeypatch.setattr(config, "LIVE_ACTIVATION_MAX_AGE_HOURS", 24.0)
    monkeypatch.setattr("src.trading.live_trader.get_manager", lambda: FakeManager())
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False)
    readiness = trader.get_live_readiness(force_preflight=True)

    assert readiness["deployable"] is True
    assert readiness["activation_guard"]["deployable"] is True
    assert readiness["status"] == "ready"
    assert trader.is_deployable() is True


def test_live_trader_warns_when_activation_is_near_expiry(monkeypatch):
    class FakeManager:
        def post(self, payload, **kwargs):
            if payload.get("type") == "meta":
                return {"universe": [{"name": "BTC", "szDecimals": 5}]}
            if payload.get("type") == "clearinghouseState":
                return {"marginSummary": {"accountValue": "250.0"}, "assetPositions": []}
            if payload.get("type") == "spotClearinghouseState":
                return {"balances": [{"coin": "USDC", "total": "50"}]}
            return {}

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    approved_at = (datetime.now(timezone.utc) - timedelta(hours=21.5)).isoformat()
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_GUARD_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_APPROVED_AT", approved_at)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_APPROVED_BY", "operator-b")
    monkeypatch.setattr(config, "LIVE_ACTIVATION_MAX_AGE_HOURS", 24.0)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_EXPIRY_WARNING_HOURS", 4.0)
    monkeypatch.setattr("src.trading.live_trader.get_manager", lambda: FakeManager())
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False)
    readiness = trader.get_live_readiness(force_preflight=True)
    activation = readiness["activation_guard"]

    assert readiness["deployable"] is True
    assert activation["deployable"] is True
    assert "approval_expiring_soon" in activation["warning_checks"]
    assert activation["hours_remaining"] < 4.0


def test_live_trader_stats_separate_live_requested_from_enabled(monkeypatch):
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_PREFLIGHT_REQUIRED", True)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_GUARD_ENABLED", True)
    monkeypatch.setattr(LiveTrader, "_load_credentials", _missing_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)

    trader = LiveTrader(firewall=DecisionFirewall({"min_confidence": 0.4}), dry_run=False, max_order_usd=1_000_000)
    stats = trader.get_stats()

    assert stats["live_requested"] is True
    assert stats["live_enabled"] is False
    assert stats["dry_run"] is True
    assert trader.is_live_requested() is True
    assert trader.is_live_enabled() is False


def test_execute_signal_blocks_when_activation_guard_is_not_ready(monkeypatch):
    class FakeManager:
        def post(self, payload, **kwargs):
            if payload.get("type") == "meta":
                return {"universe": [{"name": "BTC", "szDecimals": 5}]}
            if payload.get("type") == "clearinghouseState":
                return {"marginSummary": {"accountValue": "250.0"}, "assetPositions": []}
            if payload.get("type") == "spotClearinghouseState":
                return {"balances": [{"coin": "USDC", "total": "50"}]}
            return {}

    class CountingFirewall:
        def __init__(self):
            self.calls = 0

        def validate(self, signal, **kwargs):
            self.calls += 1
            return True, "ok"

    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_GUARD_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_APPROVED_AT", "")
    monkeypatch.setattr(config, "LIVE_ACTIVATION_APPROVED_BY", "")
    monkeypatch.setattr("src.trading.live_trader.get_manager", lambda: FakeManager())
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)

    firewall = CountingFirewall()
    trader = LiveTrader(firewall=firewall, dry_run=False)
    result = trader.execute_signal(
        {
            "coin": "BTC",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 100000.0,
            "position_pct": 0.05,
            "leverage": 2.0,
            "size": 0.01,
            "strategy_type": "momentum_long",
        }
    )

    assert result is None
    assert firewall.calls == 0


def test_execute_signal_blocks_when_preflight_is_not_ready(monkeypatch):
    class FakeManager:
        def post(self, payload, **kwargs):
            if payload.get("type") == "meta":
                return {"universe": [{"name": "BTC", "szDecimals": 5}]}
            if payload.get("type") == "clearinghouseState":
                return {"marginSummary": {"accountValue": "0"}, "assetPositions": []}
            if payload.get("type") == "spotClearinghouseState":
                return {"balances": [{"coin": "USDC", "total": "0"}]}
            return {}

    class CountingFirewall:
        def __init__(self):
            self.calls = 0

        def validate(self, signal, **kwargs):
            self.calls += 1
            return True, "ok"

    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", True)
    monkeypatch.setattr(config, "LIVE_PREFLIGHT_REQUIRED", True)
    monkeypatch.setattr(config, "LIVE_ACTIVATION_GUARD_ENABLED", False)
    monkeypatch.setattr("src.trading.live_trader.get_manager", lambda: FakeManager())
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)

    firewall = CountingFirewall()
    trader = LiveTrader(firewall=firewall, dry_run=False)
    result = trader.execute_signal(
        {
            "coin": "BTC",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 100000.0,
            "position_pct": 0.05,
            "leverage": 2.0,
            "size": 0.01,
            "strategy_type": "momentum_long",
        }
    )

    assert result is None
    assert firewall.calls == 0
    assert trader.get_live_readiness()["status_reason"].startswith("preflight:")


def test_run_live_preflight_main_returns_nonzero_when_not_certified(monkeypatch, capsys, tmp_path):
    script = _load_live_preflight_script()

    class FakeTrader:
        def __init__(self, firewall=None, dry_run=False, max_daily_loss=0.0, max_order_usd=0.0):
            self.firewall = firewall
            self.dry_run = dry_run

        def run_preflight(self, force=False):
            return {
                "status": "blocked",
                "deployable": False,
                "warning_checks": [],
                "blocking_checks": ["perps_buying_power"],
            }

        def evaluate_activation_guard(self):
            return {
                "status": "ready",
                "deployable": True,
                "approved_by": "ops",
                "warning_checks": [],
                "blocking_checks": [],
            }

        def get_live_readiness(self, force_preflight=False):
            return {
                "status": "blocked",
                "status_reason": "preflight:perps_buying_power",
                "deployable": False,
                "blocking_checks": ["preflight:perps_buying_power"],
            }

    monkeypatch.setattr(script, "setup_logging", lambda: logging.getLogger("preflight-script"))
    monkeypatch.setattr(script, "validate_dependencies", lambda logger: None)
    monkeypatch.setattr(script, "init_database", lambda logger: None)
    monkeypatch.setattr(script, "DecisionFirewall", lambda cfg: {"cfg": cfg})
    monkeypatch.setattr(script, "LiveTrader", FakeTrader)
    monkeypatch.setattr(script.config, "RUNTIME_PROFILE", "live")
    monkeypatch.setattr(script.config, "LIVE_TRADING_ENABLED", True)

    output_path = tmp_path / "preflight_blocked.json"
    rc = script.main(["--output", str(output_path)])
    captured = capsys.readouterr()

    assert rc == 1
    assert "Certification: FAIL" in captured.out
    with open(output_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["certified_for_live_entries"] is False
    assert report["live_readiness"]["status"] == "blocked"


def test_run_live_preflight_main_returns_zero_when_certified(monkeypatch, capsys, tmp_path):
    script = _load_live_preflight_script()

    class FakeTrader:
        def __init__(self, firewall=None, dry_run=False, max_daily_loss=0.0, max_order_usd=0.0):
            self.firewall = firewall
            self.dry_run = dry_run

        def run_preflight(self, force=False):
            return {
                "status": "ready",
                "deployable": True,
                "warning_checks": [],
                "blocking_checks": [],
            }

        def evaluate_activation_guard(self):
            return {
                "status": "ready",
                "deployable": True,
                "approved_by": "ops",
                "warning_checks": ["approval_expiring_soon"],
                "blocking_checks": [],
            }

        def get_live_readiness(self, force_preflight=False):
            return {
                "status": "ready",
                "status_reason": "ready",
                "deployable": True,
                "blocking_checks": [],
            }

    monkeypatch.setattr(script, "setup_logging", lambda: logging.getLogger("preflight-script"))
    monkeypatch.setattr(script, "validate_dependencies", lambda logger: None)
    monkeypatch.setattr(script, "init_database", lambda logger: None)
    monkeypatch.setattr(script, "DecisionFirewall", lambda cfg: {"cfg": cfg})
    monkeypatch.setattr(script, "LiveTrader", FakeTrader)
    monkeypatch.setattr(script.config, "RUNTIME_PROFILE", "live")
    monkeypatch.setattr(script.config, "LIVE_TRADING_ENABLED", True)

    output_path = tmp_path / "preflight_ready.json"
    rc = script.main(["--output", str(output_path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Certification: PASS" in captured.out
    with open(output_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["certified_for_live_entries"] is True
    assert report["live_readiness"]["status"] == "ready"


def test_run_daily_research_loop_script_writes_output(monkeypatch, capsys, tmp_path):
    script = _load_daily_research_script()
    output_path = tmp_path / "daily_research.json"

    monkeypatch.setattr(script.db, "init_db", lambda: None)
    monkeypatch.setattr(script, "build_adaptive_learning_config", lambda cfg: {"enabled": True})
    monkeypatch.setattr(script, "AgentScorer", lambda: object())
    monkeypatch.setattr(script, "CalibrationTracker", lambda: object())
    monkeypatch.setattr(script, "AdaptiveLearningManager", lambda *args, **kwargs: "adaptive-manager")
    monkeypatch.setattr(
        script,
        "run_daily_research_loop",
        lambda **kwargs: {
            "run_id": "daily-24",
            "status": "executed",
            "recommendation": {"action": "promote", "rollback_target_profile": "baseline_current"},
            "benchmark": {
                "promotion_gate": {
                    "winner": "challenger_execution_strict",
                    "approved_profiles": ["challenger_execution_strict"],
                }
            },
        },
    )

    rc = script.main(["--force", "--output", str(output_path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Recommendation: promote" in captured.out
    assert "Winner: challenger_execution_strict" in captured.out
    with open(output_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["run_id"] == "daily-24"


def test_run_shadow_certification_script_returns_nonzero_until_certified(monkeypatch, capsys, tmp_path):
    script = _load_shadow_certification_script()
    output_path = tmp_path / "shadow_certification.json"

    monkeypatch.setattr(script.db, "init_db", lambda: None)
    monkeypatch.setattr(script, "ShadowTracker", lambda: "shadow-tracker")
    monkeypatch.setattr(
        script,
        "run_shadow_certification",
        lambda **kwargs: {
            "run_id": "shadow-25",
            "status": "failed",
            "certified": False,
            "summary": "Shadow certification failed checks: slippage_drift_bps",
            "blocked_entry_reasons": {"capital_governor_guard": 4},
        },
    )

    rc = script.main(["--output", str(output_path)])
    captured = capsys.readouterr()

    assert rc == 1
    assert "Certified: False" in captured.out
    with open(output_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["run_id"] == "shadow-25"


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
        def get_weight(self, source_key):
            return 0.8

        def get_accuracy(self, source_key):
            return 0.7

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
                            "conviction_pct": 80,
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

    assert executed == [("ETH", "long", True)]


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


def test_execute_selected_decisions_routes_only_engine_choices(monkeypatch):
    paper_batches = []
    copy_batches = []
    mirrors = []
    lcrs_batches = []

    class FakePaperTrader:
        def execute_strategy_signals(self, strategies, **kwargs):
            paper_batches.append([strategy["name"] for strategy in strategies])
            return [{"coin": "BTC", "side": "long"}]

    class FakeCopyTrader:
        def execute_copy_signals(self, signals, regime_data=None):
            copy_batches.append([signal["coin"] for signal in signals])
            return [{"coin": signals[0]["coin"], "side": signals[0]["side"]}] if signals else []

    container = type(
        "Container",
        (),
        {
            "paper_trader": FakePaperTrader(),
            "copy_trader": FakeCopyTrader(),
            "exchange_agg": None,
            "options_scanner": None,
            "arena": None,
        },
    )()

    monkeypatch.setattr(
        "src.core.cycles.trading_cycle._execute_lcrs_signals",
        lambda container, signals, regime_data: lcrs_batches.append([signal["coin"] for signal in signals]),
    )
    monkeypatch.setattr(
        "src.core.cycles.trading_cycle.mirror_executed_trades_to_live",
        lambda container, trades, success_label, skip_label: mirrors.append((success_label, len(trades))),
    )
    monkeypatch.setattr("src.notifications.telegram_bot.is_configured", lambda: False)

    _execute_selected_decisions(
        container,
        [
            {"name": "ranked_strategy", "strategy_type": "trend_following"},
            {
                "name": "lcrs_eth",
                "_decision_route": "lcrs",
                "_lcrs_signal": {"coin": "ETH", "side": "long", "confidence": 0.7},
            },
            {
                "name": "copy_sol",
                "_decision_route": "copy_trade",
                "_copy_signal": {"coin": "SOL", "side": "short", "type": "copy_open"},
            },
        ],
        {"overall_regime": "neutral"},
    )

    assert paper_batches == [["ranked_strategy"]]
    assert lcrs_batches == [["ETH"]]
    assert copy_batches == [["SOL"]]
    assert mirrors == [("  LIVE", 1), ("  LIVE COPY", 1)]


def test_dashboard_live_account_summary_stays_separate_from_paper_metrics():
    class FakeLiveTrader:
        def get_stats(self):
            return {
                "live_enabled": True,
                "deployable": True,
                "dry_run": False,
                "status_reason": "ready",
                "preflight": {
                    "deployable": True,
                    "status": "ready",
                    "checked_at": "2026-04-05T12:00:00",
                    "blocking_checks": [],
                    "warning_checks": [],
                },
                "activation_guard": {
                    "deployable": True,
                    "status": "ready",
                    "approved_at": "2026-04-05T11:55:00+00:00",
                    "approved_by": "operator-a",
                    "expires_at": "2026-04-06T11:55:00+00:00",
                    "blocking_checks": [],
                    "warning_checks": ["approval_expiring_soon"],
                    "hours_remaining": 3.5,
                },
                "daily_pnl": 42.25,
                "daily_pnl_limit": 150.0,
                "orders_today": 7,
                "fills_today": 5,
                "wallet_balance": {
                    "perps_margin": 1200.0,
                    "spot_usdc": 50.0,
                    "total": 1250.0,
                    "timestamp": "2026-04-05T12:00:00",
                },
                "timestamp": "2026-04-05T12:00:00",
            }

        def snapshot_balance(self, log=False):
            return {
                "perps_margin": 1200.0,
                "spot_usdc": 50.0,
                "total": 1250.0,
                "timestamp": "2026-04-05T12:00:00",
            }

        def get_positions(self):
            return [
                {"coin": "BTC", "szi": 0.1, "unrealized_pnl": 15.5},
                {"coin": "ETH", "size": -0.2, "unrealizedPnl": -4.0},
                {"coin": "SOL", "szi": 0.0, "unrealized_pnl": 99.0},
            ]

    summary = dashboard_ui._build_live_account_summary(FakeLiveTrader())

    assert summary["available"] is True
    assert summary["live_enabled"] is True
    assert summary["deployable"] is True
    assert summary["dry_run"] is False
    assert summary["wallet_total"] == 1250.0
    assert summary["perps_margin"] == 1200.0
    assert summary["spot_usdc"] == 50.0
    assert summary["preflight_ready"] is True
    assert summary["preflight_status"] == "ready"
    assert summary["activation_ready"] is True
    assert summary["activation_status"] == "ready"
    assert summary["activation_approved_by"] == "operator-a"
    assert summary["activation_warning_checks"] == ["approval_expiring_soon"]
    assert summary["activation_hours_remaining"] == 3.5
    assert summary["realized_pnl_today"] == 42.25
    assert summary["open_unrealized_pnl"] == 11.5
    assert summary["open_positions"] == 2
    assert summary["orders_today"] == 7
    assert summary["fills_today"] == 5


def test_dashboard_live_account_summary_includes_ledger_metrics(monkeypatch):
    monkeypatch.setattr(
        dashboard_ui.db,
        "get_live_fill_summary",
        lambda public_address=None: {
            "fill_count": 3,
            "realized_pnl": 64.5,
            "fees_paid": 2.25,
            "last_fill_timestamp": "2026-04-05T12:02:00",
        },
    )
    monkeypatch.setattr(
        dashboard_ui.db,
        "get_latest_live_equity_snapshot",
        lambda public_address=None: {
            "timestamp": "2026-04-05T12:00:00",
            "perps_margin": 1300.0,
            "spot_usdc": 25.0,
            "total": 1325.0,
            "daily_realized_pnl": 18.0,
            "orders_today": 4,
            "fills_today": 3,
        },
    )
    monkeypatch.setattr(
        dashboard_ui.db,
        "get_latest_live_positions",
        lambda public_address=None: {
            "snapshot": {"position_count": 2},
            "positions": [
                {"coin": "BTC", "unrealized_pnl": 5.0},
                {"coin": "ETH", "unrealized_pnl": -2.0},
            ],
        },
    )

    class FakeLiveTrader:
        public_address = "0xabc"

        def get_stats(self):
            return {
                "live_enabled": True,
                "deployable": True,
                "dry_run": False,
                "status_reason": "ready",
                "daily_pnl": 18.0,
                "daily_pnl_limit": 150.0,
                "orders_today": 4,
                "fills_today": 3,
                "wallet_balance": {
                    "perps_margin": 1300.0,
                    "spot_usdc": 25.0,
                    "total": 1325.0,
                    "timestamp": "2026-04-05T12:00:00",
                },
                "timestamp": "2026-04-05T12:00:00",
            }

        def snapshot_balance(self, log=False):
            return {
                "perps_margin": 1300.0,
                "spot_usdc": 25.0,
                "total": 1325.0,
                "timestamp": "2026-04-05T12:00:00",
            }

        def get_positions(self):
            return [{"coin": "BTC", "szi": 0.2, "unrealized_pnl": 5.0}]

    summary = dashboard_ui._build_live_account_summary(FakeLiveTrader())

    assert summary["ledger_ready"] is True
    assert summary["lifetime_realized_pnl"] == 64.5
    assert summary["lifetime_fees_paid"] == 2.25
    assert summary["lifetime_fill_count"] == 3
    assert summary["last_fill_timestamp"] == "2026-04-05T12:02:00"
    assert summary["last_equity_timestamp"] == "2026-04-05T12:00:00"


def test_dashboard_experiment_discipline_metrics_include_report_and_shadow(monkeypatch):
    fd, raw_path = tempfile.mkstemp(prefix="experiment_report_", suffix=".json", dir=os.getcwd())
    os.close(fd)
    report_path = os.path.abspath(raw_path)

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": "2026-04-05T12:00:00",
                "cycle_count": 12,
                "in_sample_cycles": 8,
                "out_of_sample_cycles": 4,
                "promotion_gate": {
                    "baseline": "baseline_current",
                    "approved_profiles": ["challenger_execution_strict"],
                    "winner": "challenger_execution_strict",
                },
                "profiles": {
                    "baseline_current": {"out_of_sample": {"cycles": 4}, "promotion": {"approved": True}},
                    "challenger_execution_strict": {
                        "out_of_sample": {"cycles": 4, "avg_selected_ev_pct": 0.012},
                        "promotion": {"approved": True, "ev_delta_pct": 0.003},
                    },
                },
            },
            handle,
        )

    monkeypatch.setattr(config, "EXPERIMENT_BENCHMARK_REPORT_PATH", report_path)
    monkeypatch.setattr(
        dashboard_ui.db,
        "get_decision_funnel_summary",
        lambda limit_cycles=120: {"cycles": 4, "selected_count": 3},
    )
    monkeypatch.setattr(
        dashboard_ui.db,
        "get_source_attribution_summary",
        lambda limit_cycles=120, lookback_hours=168: [{"source_key": "strategy:trend", "selected_count": 2}],
    )
    monkeypatch.setattr(
        dashboard_ui.db,
        "get_runtime_divergence_summary",
        lambda lookback_hours=24: {"paper_live_open_gap": 1},
    )
    monkeypatch.setattr(
        dashboard_ui.db,
        "get_latest_daily_research_run",
        lambda: {"run_id": "daily-1", "recommendation": "promote", "winner_profile": "challenger_execution_strict"},
    )
    monkeypatch.setattr(
        dashboard_ui.db,
        "get_recent_daily_research_runs",
        lambda limit=10: [{"run_id": "daily-1"}, {"run_id": "daily-0"}],
    )
    monkeypatch.setattr(
        dashboard_ui.db,
        "get_daily_research_last_known_good",
        lambda: {"profile_name": "challenger_execution_strict"},
    )
    monkeypatch.setattr(
        dashboard_ui.db,
        "get_latest_shadow_certification_run",
        lambda: {"run_id": "shadow-1", "status": "failed", "certified": False},
    )
    monkeypatch.setattr(
        dashboard_ui.db,
        "get_recent_shadow_certification_runs",
        lambda limit=10: [{"run_id": "shadow-1"}, {"run_id": "shadow-0"}],
    )

    class FakeShadowTracker:
        def get_attribution(self, days=7):
            return {"strategy:trend": {"trades": 5, "pnl": 12.5}}

    metrics = dashboard_ui._build_experiment_discipline_metrics(FakeShadowTracker())

    assert metrics["decision_funnel"]["cycles"] == 4
    assert metrics["source_attribution"][0]["source_key"] == "strategy:trend"
    assert metrics["divergence"]["paper_live_open_gap"] == 1
    assert metrics["benchmark_pack"]["promotion_gate"]["winner"] == "challenger_execution_strict"
    assert metrics["daily_research_latest"]["run_id"] == "daily-1"
    assert metrics["daily_research_recent"][0]["run_id"] == "daily-1"
    assert metrics["daily_research_last_known_good"]["profile_name"] == "challenger_execution_strict"
    assert metrics["shadow_certification_latest"]["run_id"] == "shadow-1"
    assert metrics["shadow_certification_recent"][0]["run_id"] == "shadow-1"
    assert metrics["shadow_attribution"]["strategy:trend"]["trades"] == 5

    os.remove(report_path)


def test_daily_research_loop_persists_runs_and_last_known_good(monkeypatch):
    fd, raw_path = tempfile.mkstemp(prefix="daily_research_", suffix=".db", dir=os.getcwd())
    os.close(fd)
    db_path = os.path.abspath(raw_path)
    original_path = db._DB_PATH
    monkeypatch.setattr(db, "_DB_PATH", db_path)

    benchmark = {
        "promotion_gate": {
            "baseline": "baseline_current",
            "approved_profiles": ["challenger_execution_strict"],
            "winner": "challenger_execution_strict",
        },
        "profiles": {
            "baseline_current": {
                "out_of_sample": {"avg_selected_ev_pct": 0.011, "avg_selected_execution_cost_pct": 0.0025, "no_trade_rate": 0.40},
                "promotion": {"approved": True, "ev_delta_pct": 0.0, "execution_cost_delta_pct": 0.0, "no_trade_delta": 0.0},
                "overrides": {},
            },
            "challenger_execution_strict": {
                "out_of_sample": {"avg_selected_ev_pct": 0.016, "avg_selected_execution_cost_pct": 0.0018, "no_trade_rate": 0.35},
                "promotion": {"approved": True, "ev_delta_pct": 0.004, "execution_cost_delta_pct": -0.0007, "no_trade_delta": -0.05},
                "overrides": {"min_decision_score": 0.41},
            },
        },
    }

    class FakeAdaptive:
        def run_recalibration(self, cycle_count=None, force=False):
            return {"executed": True, "run_id": "recal-24"}

    monkeypatch.setattr(
        daily_research_module,
        "run_experiment_benchmark_pack",
        lambda limit_cycles, out_of_sample_ratio: benchmark,
    )

    try:
        db.init_db()
        loop = DailyResearchLoop(
            {
                "enabled": True,
                "interval_hours": 24.0,
                "benchmark_limit_cycles": 60,
                "out_of_sample_ratio": 0.30,
                "benchmark_report_path": "",
                "report_path": "",
                "rollback_ev_tolerance_pct": 0.0005,
            },
            adaptive_learning=FakeAdaptive(),
        )
        report = loop.run(cycle_count=96, force=True)
        latest = db.get_latest_daily_research_run()
        recent = db.get_recent_daily_research_runs(limit=10)
        last_known_good = db.get_daily_research_last_known_good()

        assert report["recommendation"]["action"] == "promote"
        assert latest["recommendation"] == "promote"
        assert latest["winner_profile"] == "challenger_execution_strict"
        assert latest["metadata"]["delta"]["winner_ev_delta_pct"] == 0.004
        assert recent[0]["run_id"] == latest["run_id"]
        assert last_known_good["profile_name"] == "challenger_execution_strict"
        assert last_known_good["overrides"]["min_decision_score"] == 0.41
    finally:
        monkeypatch.setattr(db, "_DB_PATH", original_path)
        if os.path.exists(db_path):
            os.remove(db_path)


def test_daily_research_loop_recommends_rollback_when_benchmark_regresses(monkeypatch):
    fd, raw_path = tempfile.mkstemp(prefix="daily_research_rollback_", suffix=".db", dir=os.getcwd())
    os.close(fd)
    db_path = os.path.abspath(raw_path)
    original_path = db._DB_PATH
    monkeypatch.setattr(db, "_DB_PATH", db_path)

    benchmark = {
        "promotion_gate": {
            "baseline": "baseline_current",
            "approved_profiles": [],
            "winner": "challenger_regressed",
        },
        "profiles": {
            "baseline_current": {
                "out_of_sample": {"avg_selected_ev_pct": 0.012, "avg_selected_execution_cost_pct": 0.0023, "no_trade_rate": 0.38},
                "promotion": {"approved": True, "ev_delta_pct": 0.0, "execution_cost_delta_pct": 0.0, "no_trade_delta": 0.0},
                "overrides": {},
            },
            "challenger_regressed": {
                "out_of_sample": {"avg_selected_ev_pct": 0.008, "avg_selected_execution_cost_pct": 0.0029, "no_trade_rate": 0.43},
                "promotion": {"approved": False, "ev_delta_pct": -0.0035, "execution_cost_delta_pct": 0.0006, "no_trade_delta": 0.05},
                "overrides": {"min_decision_score": 0.37},
            },
        },
    }

    monkeypatch.setattr(
        daily_research_module,
        "run_experiment_benchmark_pack",
        lambda limit_cycles, out_of_sample_ratio: benchmark,
    )

    try:
        db.init_db()
        db.save_daily_research_last_known_good(
            {
                "profile_name": "challenger_execution_strict",
                "overrides": {"min_decision_score": 0.41},
                "saved_at": "2026-04-06T00:00:00+00:00",
                "reason": "previous_promotion",
            }
        )
        loop = DailyResearchLoop(
            {
                "enabled": True,
                "interval_hours": 24.0,
                "benchmark_limit_cycles": 60,
                "out_of_sample_ratio": 0.30,
                "benchmark_report_path": "",
                "report_path": "",
                "rollback_ev_tolerance_pct": 0.0005,
            }
        )
        report = loop.run(cycle_count=97, force=True)
        latest = db.get_latest_daily_research_run()
        last_known_good = db.get_daily_research_last_known_good()

        assert report["recommendation"]["action"] == "rollback"
        assert report["recommendation"]["rollback_target_profile"] == "challenger_execution_strict"
        assert latest["recommendation"] == "rollback"
        assert last_known_good["profile_name"] == "challenger_execution_strict"
    finally:
        monkeypatch.setattr(db, "_DB_PATH", original_path)
        if os.path.exists(db_path):
            os.remove(db_path)


def test_shadow_certification_report_flags_failure_when_thresholds_are_breached(monkeypatch):
    monkeypatch.setattr(
        shadow_certification_module.db,
        "get_runtime_divergence_summary",
        lambda lookback_hours=24.0 * 7: {
            "paper_live_open_gap_ratio": 0.52,
            "realized_pnl_gap_ratio": 0.48,
        },
    )
    monkeypatch.setattr(
        shadow_certification_module.db,
        "get_decision_funnel_summary",
        lambda limit_cycles=10: {
            "blocker_mix": {"capital_governor_guard": 4, "net_ev<0.0015": 3},
        },
    )
    monkeypatch.setattr(
        shadow_certification_module.db,
        "get_source_attribution_summary",
        lambda limit_cycles=10, lookback_hours=24.0 * 7: [{"source_key": "options_flow:BTC", "selected_count": 4}],
    )
    monkeypatch.setattr(
        shadow_certification_module.db,
        "get_execution_quality_summary",
        lambda lookback_hours=24.0 * 7: {
            "avg_realized_slippage_bps": 4.5,
            "avg_expected_slippage_bps": 1.2,
        },
    )
    monkeypatch.setattr(
        shadow_certification_module.db,
        "get_execution_quality_by_source",
        lambda lookback_hours=24.0 * 7, limit=10: [{"source_key": "options_flow:BTC", "rejection_rate": 0.2}],
    )
    monkeypatch.setattr(
        shadow_certification_module.db,
        "get_capital_governor_summary",
        lambda lookback_hours=24.0 * 7: {"degraded_source_ratio": 0.42},
    )
    monkeypatch.setattr(
        shadow_certification_module.db,
        "get_latest_source_health_snapshot",
        lambda limit=25: {
            "snapshot": {"snapshot_id": "snap-25"},
            "profiles": [{"source_key": "options_flow:BTC", "status": "blocked", "health_score": 0.31}],
        },
    )
    monkeypatch.setattr(
        shadow_certification_module.db,
        "get_latest_daily_research_run",
        lambda: {"run_id": "daily-24", "recommendation": "hold"},
    )
    monkeypatch.setattr(
        shadow_certification_module.db,
        "get_recent_shadow_certification_runs",
        lambda limit=10: [
            {
                "timestamp": utc_now_iso(),
                "metadata": {"incident_keys": ["live_preflight_blocked"]},
            }
        ],
    )
    monkeypatch.setattr(
        shadow_certification_module,
        "build_runtime_incident_report",
        lambda **kwargs: {"summary": {}, "incidents": [{"key": "live_activation_blocked"}]},
    )

    class FakeShadowTracker:
        def get_summary(self, days=7):
            return {"total_trades": 15, "total_pnl": 42.0}

        def get_attribution(self, days=7):
            return {"options_flow:BTC": {"trades": 8, "pnl": 22.5}}

    report = build_shadow_certification_report(
        cfg={
            "enabled": True,
            "lookback_days": 7,
            "limit_cycles": 10,
            "report_path": "",
            "min_shadow_trades": 10,
            "max_open_gap_ratio": 0.40,
            "max_realized_pnl_gap_ratio": 0.45,
            "max_slippage_drift_bps": 2.0,
            "max_degraded_source_ratio": 0.35,
            "max_readiness_interruption_runs": 0,
        },
        shadow_tracker=FakeShadowTracker(),
    )

    assert report["status"] == "failed"
    assert report["certified"] is False
    assert "paper_live_open_gap_ratio" in report["summary"]
    assert report["blocked_entry_reasons"]["capital_governor_guard"] == 4
    assert report["readiness_interruptions"]["count"] == 2


def test_shadow_certification_report_warms_up_with_insufficient_trades(monkeypatch):
    monkeypatch.setattr(
        shadow_certification_module.db,
        "get_runtime_divergence_summary",
        lambda lookback_hours=24.0 * 7: {"paper_live_open_gap_ratio": 0.05, "realized_pnl_gap_ratio": 0.02},
    )
    monkeypatch.setattr(
        shadow_certification_module.db,
        "get_decision_funnel_summary",
        lambda limit_cycles=10: {"blocker_mix": {}},
    )
    monkeypatch.setattr(
        shadow_certification_module.db,
        "get_source_attribution_summary",
        lambda limit_cycles=10, lookback_hours=24.0 * 7: [],
    )
    monkeypatch.setattr(
        shadow_certification_module.db,
        "get_execution_quality_summary",
        lambda lookback_hours=24.0 * 7: {"avg_realized_slippage_bps": 1.0, "avg_expected_slippage_bps": 1.0},
    )
    monkeypatch.setattr(
        shadow_certification_module.db,
        "get_execution_quality_by_source",
        lambda lookback_hours=24.0 * 7, limit=10: [],
    )
    monkeypatch.setattr(
        shadow_certification_module.db,
        "get_capital_governor_summary",
        lambda lookback_hours=24.0 * 7: {"degraded_source_ratio": 0.10},
    )
    monkeypatch.setattr(
        shadow_certification_module.db,
        "get_latest_source_health_snapshot",
        lambda limit=25: {"snapshot": {}, "profiles": []},
    )
    monkeypatch.setattr(
        shadow_certification_module.db,
        "get_latest_daily_research_run",
        lambda: {},
    )
    monkeypatch.setattr(
        shadow_certification_module.db,
        "get_recent_shadow_certification_runs",
        lambda limit=10: [],
    )
    monkeypatch.setattr(
        shadow_certification_module,
        "build_runtime_incident_report",
        lambda **kwargs: {"summary": {}, "incidents": []},
    )

    class FakeShadowTracker:
        def get_summary(self, days=7):
            return {"total_trades": 4, "total_pnl": 8.0}

        def get_attribution(self, days=7):
            return {}

    report = build_shadow_certification_report(
        cfg={
            "enabled": True,
            "lookback_days": 7,
            "limit_cycles": 10,
            "report_path": "",
            "min_shadow_trades": 10,
            "max_open_gap_ratio": 0.40,
            "max_realized_pnl_gap_ratio": 0.45,
            "max_slippage_drift_bps": 2.0,
            "max_degraded_source_ratio": 0.35,
            "max_readiness_interruption_runs": 0,
        },
        shadow_tracker=FakeShadowTracker(),
    )

    assert report["status"] == "warming_up"
    assert report["certified"] is False
    assert "warming up" in report["summary"]


def test_live_ledger_helpers_persist_snapshots_and_deduplicate_fills(monkeypatch):
    fd, raw_path = tempfile.mkstemp(prefix="live_ledger_", suffix=".db", dir=os.getcwd())
    os.close(fd)
    db_path = os.path.abspath(raw_path)
    original_path = db._DB_PATH
    monkeypatch.setattr(db, "_DB_PATH", db_path)

    try:
        db.init_db()
        db.save_live_equity_snapshot(
            public_address="0xabc",
            perps_margin=1200.0,
            spot_usdc=50.0,
            total=1250.0,
            daily_realized_pnl=21.5,
            orders_today=7,
            fills_today=4,
            timestamp="2026-04-05T12:00:00",
        )
        db.save_live_position_snapshot(
            public_address="0xabc",
            positions=[{"coin": "BTC", "side": "long", "szi": 0.1, "entry_price": 100000.0, "unrealized_pnl": 15.0, "leverage": 2}],
            timestamp="2026-04-05T12:01:00",
        )
        db.save_live_position_snapshot(
            public_address="0xabc",
            positions=[],
            timestamp="2026-04-05T12:02:00",
        )
        db.upsert_live_fill_events(
            "0xabc",
            [
                {
                    "time": 1_775_437_200_000,
                    "coin": "BTC",
                    "dir": "Open Long",
                    "sz": "0.1",
                    "px": "100500",
                    "fee": "0.75",
                    "closedPnl": "-12.5",
                    "oid": "123",
                },
                {
                    "time": 1_775_437_200_000,
                    "coin": "BTC",
                    "dir": "Open Long",
                    "sz": "0.1",
                    "px": "100500",
                    "fee": "0.75",
                    "closedPnl": "-12.5",
                    "oid": "123",
                },
            ],
        )

        latest_equity = db.get_latest_live_equity_snapshot("0xabc")
        latest_positions = db.get_latest_live_positions("0xabc")
        fill_summary = db.get_live_fill_summary("0xabc")
        equity_curve = db.get_recent_live_equity_snapshots(10, "0xabc")

        assert latest_equity["total"] == 1250.0
        assert latest_equity["daily_realized_pnl"] == 21.5
        assert latest_positions["snapshot"]["position_count"] == 0
        assert latest_positions["positions"] == []
        assert fill_summary["fill_count"] == 1
        assert fill_summary["realized_pnl"] == -12.5
        assert fill_summary["fees_paid"] == 0.75
        assert len(equity_curve) == 1
    finally:
        monkeypatch.setattr(db, "_DB_PATH", original_path)
        if os.path.exists(db_path):
            os.remove(db_path)


def test_live_trader_persists_live_ledger_snapshots(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    saved_balance = []
    saved_positions = []
    saved_fills = []

    class FakeApiManager:
        def post(self, payload, priority=Priority.NORMAL, timeout=10):
            request_type = payload.get("type")
            if request_type == "clearinghouseState":
                return {"marginSummary": {"accountValue": "1200"}}
            if request_type == "spotClearinghouseState":
                return {"balances": [{"coin": "USDC", "total": "50"}]}
            if request_type == "userFills":
                return [
                    {
            "time": int(utc_now().timestamp() * 1000),
                        "coin": "BTC",
                        "dir": "Open Long",
                        "sz": "0.1",
                        "px": "100500",
                        "fee": "0.25",
                        "closedPnl": "5.5",
                        "oid": "123",
                    }
                ]
            return {}

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "check_daily_loss", lambda self: None)
    monkeypatch.setattr(
        LiveTrader,
        "get_account_state",
        lambda self: {
            "assetPositions": [
                {
                    "position": {
                        "coin": "BTC",
                        "szi": "0.1",
                        "entryPx": "100000",
                        "unrealizedPnl": "12.0",
                        "leverage": {"value": 2},
                    }
                }
            ]
        },
    )
    monkeypatch.setattr(
        "src.trading.live_trader.db.save_live_equity_snapshot",
        lambda **kwargs: saved_balance.append(kwargs) or 1,
    )
    monkeypatch.setattr(
        "src.trading.live_trader.db.save_live_position_snapshot",
        lambda **kwargs: saved_positions.append(kwargs) or "snapshot-1",
    )
    monkeypatch.setattr(
        "src.trading.live_trader.db.upsert_live_fill_events",
        lambda public_address, fills: saved_fills.append((public_address, fills)) or {"seen": len(fills)},
    )

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
    trader.api_manager = FakeApiManager()

    balance = trader.snapshot_balance(log=False)
    positions = trader.get_positions()
    trader.update_daily_pnl_from_fills()

    assert balance["total"] == 1250.0
    assert positions[0]["coin"] == "BTC"
    assert trader.fills_today == 1
    assert saved_balance
    assert saved_balance[-1]["total"] == 1250.0
    assert saved_positions
    assert saved_positions[-1]["positions"][0]["coin"] == "BTC"
    assert saved_fills
    assert saved_fills[-1][0] == "0x2222222222222222222222222222222222222222"


def test_execution_quality_helpers_round_trip_and_group_by_source(monkeypatch):
    fd, raw_path = tempfile.mkstemp(prefix="execution_quality_", suffix=".db", dir=os.getcwd())
    os.close(fd)
    db_path = os.path.abspath(raw_path)
    original_path = db._DB_PATH
    monkeypatch.setattr(db, "_DB_PATH", db_path)

    now = utc_now_iso()

    try:
        db.init_db()
        db.save_live_execution_event(
            "0xabc",
            timestamp=now,
            signal_id="sig-1",
            source="options_flow",
            source_key="options_flow:BTC",
            coin="BTC",
            side="short",
            status="success",
            execution_role="taker",
            requested_size=1.0,
            submitted_size=1.0,
            filled_size=0.8,
            requested_notional=100.0,
            submitted_notional=80.4,
            mid_price=100.0,
            execution_price=100.5,
            expected_slippage_bps=6.0,
            realized_slippage_bps=50.0,
            fill_ratio=0.8,
            protective_status="placed",
        )
        db.save_live_execution_event(
            "0xabc",
            timestamp=now,
            signal_id="sig-2",
            source="options_flow",
            source_key="options_flow:BTC",
            coin="BTC",
            side="short",
            status="rejected",
            requested_size=1.0,
            mid_price=100.0,
            rejection_reason="below_exchange_minimum_notional",
        )
        db.save_live_execution_event(
            "0xabc",
            timestamp=now,
            signal_id="sig-3",
            source="strategy",
            source_key="strategy:trend_following",
            coin="ETH",
            side="long",
            status="error",
            requested_size=2.0,
            mid_price=200.0,
            rejection_reason="protective_order_failed",
            protective_status="failed",
        )

        summary = db.get_execution_quality_summary("0xabc")
        source_rows = db.get_execution_quality_by_source("0xabc", limit=5)

        assert summary["total_events"] == 3
        assert summary["success_count"] == 1
        assert summary["rejection_count"] == 2
        assert summary["avg_realized_slippage_bps"] == 50.0
        assert summary["avg_fill_ratio"] == 0.8
        assert summary["protective_failure_count"] == 1
        assert source_rows[0]["source_key"] in {"options_flow:BTC", "strategy:trend_following"}
        btc_row = next(row for row in source_rows if row["source_key"] == "options_flow:BTC")
        assert btc_row["total_events"] == 2
        assert btc_row["rejection_rate"] == 0.5
    finally:
        monkeypatch.setattr(db, "_DB_PATH", original_path)
        if os.path.exists(db_path):
            os.remove(db_path)


def test_live_trader_records_execution_quality_events(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    saved_events = []

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "check_daily_loss", lambda self: False)
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 100.0)
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(
        LiveTrader,
        "place_market_order",
        lambda self, coin, side, size, leverage=1, reduce_only=False: {
            "status": "success",
            "requested_size": size,
            "requested_notional": size * 100.0,
            "submitted_size": size,
            "submitted_notional": size * 100.5,
            "mid_price": 100.0,
            "wire_price": 100.5,
            "exchange_reported_fill_size": size,
            "exchange_reported_fill_price": 100.5,
        },
    )
    monkeypatch.setattr(
        LiveTrader,
        "verify_fill",
        lambda self, *a, **kw: {"status": "verified", "size": 1.0, "partial_fill": False},
    )
    monkeypatch.setattr(LiveTrader, "place_trigger_order", lambda self, *a, **kw: {"status": "ok"})
    monkeypatch.setattr(
        "src.trading.live_trader.db.save_live_execution_event",
        lambda public_address, **event: saved_events.append((public_address, event)) or 1,
    )

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
    result = trader.execute_signal(
        {
            "coin": "BTC",
            "side": "long",
            "confidence": 0.82,
            "entry_price": 100.0,
            "size": 1.0,
            "leverage": 2,
            "strategy_type": "options_momentum",
            "source": "options_flow",
            "source_key": "options_flow:BTC",
            "metadata": {
                "expected_slippage_bps": 6.0,
                "route": "paper_strategy",
            },
        }
    )

    assert result["status"] == "success"
    assert saved_events
    public_address, event = saved_events[-1]
    assert public_address == "0x2222222222222222222222222222222222222222"
    assert event["source_key"] == "options_flow:BTC"
    assert event["status"] == "success"
    assert event["fill_ratio"] == 1.0
    assert event["realized_slippage_bps"] == 50.0
    assert event["expected_slippage_bps"] == 6.0


def test_decision_funnel_source_attribution_and_divergence_summaries(monkeypatch):
    fd, raw_path = tempfile.mkstemp(prefix="experiment_metrics_", suffix=".db", dir=os.getcwd())
    os.close(fd)
    db_path = os.path.abspath(raw_path)
    original_path = db._DB_PATH
    monkeypatch.setattr(db, "_DB_PATH", db_path)

    try:
        db.init_db()
        now = utc_now_iso()
        db.save_decision_research_snapshot(
            {
                "timestamp": now,
                "cycle_number": 10,
                "available_slots": 3,
                "candidate_count": 3,
                "qualified_count": 2,
                "executed_count": 1,
                "context": {"open_positions": [{"coin": "SOL", "side": "long"}]},
                "candidates": [
                    {
                        "rank": 1,
                        "status": "selected",
                        "name": "btc_options",
                        "source": "options_flow",
                        "source_key": "options_flow:BTC",
                        "strategy_type": "options_momentum",
                        "coin": "BTC",
                        "side": "short",
                        "route": "paper_strategy",
                        "composite_score": 0.83,
                        "confidence": 0.72,
                        "expected_value_pct": 0.018,
                        "execution_cost_pct": 0.002,
                        "blockers": [],
                        "score_breakdown": {},
                        "raw_candidate": {
                            "name": "btc_options",
                            "source": "options_flow",
                            "source_key": "options_flow:BTC",
                            "strategy_type": "options_momentum",
                            "current_score": 0.83,
                            "confidence": 0.72,
                            "entry_price": 100.0,
                            "stop_loss": 103.0,
                            "take_profit": 94.0,
                            "parameters": {"coins": ["BTC"]},
                            "metadata": {"source": "options_flow", "source_key": "options_flow:BTC"},
                        },
                    },
                    {
                        "rank": 2,
                        "status": "blocked",
                        "name": "trend_eth",
                        "source": "strategy",
                        "source_key": "strategy:trend_following",
                        "strategy_type": "trend_following",
                        "coin": "ETH",
                        "side": "long",
                        "route": "paper_strategy",
                        "composite_score": 0.42,
                        "confidence": 0.51,
                        "expected_value_pct": 0.0004,
                        "execution_cost_pct": 0.001,
                        "blockers": ["net_ev<0.0015"],
                        "score_breakdown": {},
                        "raw_candidate": {
                            "name": "trend_eth",
                            "source": "strategy",
                            "source_key": "strategy:trend_following",
                            "strategy_type": "trend_following",
                            "current_score": 0.42,
                            "confidence": 0.51,
                            "entry_price": 100.0,
                            "stop_loss": 97.0,
                            "take_profit": 101.0,
                            "parameters": {"coins": ["ETH"]},
                            "metadata": {"source": "strategy", "source_key": "strategy:trend_following"},
                        },
                    },
                    {
                        "rank": 3,
                        "status": "overflow",
                        "name": "sol_mean_revert",
                        "source": "strategy",
                        "source_key": "strategy:mean_reversion",
                        "strategy_type": "mean_reversion",
                        "coin": "SOL",
                        "side": "long",
                        "route": "paper_strategy",
                        "composite_score": 0.63,
                        "confidence": 0.66,
                        "expected_value_pct": 0.012,
                        "execution_cost_pct": 0.0014,
                        "blockers": [],
                        "score_breakdown": {},
                        "raw_candidate": {
                            "name": "sol_mean_revert",
                            "source": "strategy",
                            "source_key": "strategy:mean_reversion",
                            "strategy_type": "mean_reversion",
                            "current_score": 0.63,
                            "confidence": 0.66,
                            "entry_price": 100.0,
                            "stop_loss": 97.0,
                            "take_profit": 107.0,
                            "parameters": {"coins": ["SOL"]},
                            "metadata": {"source": "strategy", "source_key": "strategy:mean_reversion"},
                        },
                    },
                ],
            }
        )

        trade_id = db.open_paper_trade(
            strategy_id=1,
            coin="BTC",
            side="short",
            entry_price=100.0,
            size=1.0,
            metadata={"source": "options_flow", "source_key": "options_flow:BTC"},
        )
        db.close_paper_trade(trade_id, exit_price=95.0, pnl=5.0)
        db.open_paper_trade(
            strategy_id=2,
            coin="SOL",
            side="long",
            entry_price=50.0,
            size=2.0,
            metadata={"source": "strategy", "source_key": "strategy:mean_reversion"},
        )
        db.save_live_position_snapshot(
            public_address="0xabc",
            positions=[{"coin": "BTC", "side": "short", "szi": -1.0, "entry_price": 100.0}],
            timestamp=now,
        )
        db.save_live_execution_event(
            "0xabc",
            timestamp=now,
            signal_id="live-1",
            source="options_flow",
            source_key="options_flow:BTC",
            coin="BTC",
            side="short",
            status="success",
            filled_size=1.0,
            fill_ratio=1.0,
            realized_slippage_bps=9.0,
            protective_status="placed",
        )

        funnel = db.get_decision_funnel_summary(limit_cycles=10)
        attribution = db.get_source_attribution_summary(limit_cycles=10, lookback_hours=24)
        divergence = db.get_runtime_divergence_summary(lookback_hours=24)

        assert funnel["cycles"] == 1
        assert funnel["selected_count"] == 1
        assert funnel["blocked_count"] == 1
        assert funnel["overflow_count"] == 1
        assert funnel["blocker_mix"]["net_ev<0.0015"] == 1
        options_row = next(row for row in attribution if row["source_key"] == "options_flow:BTC")
        assert options_row["selected_count"] == 1
        assert options_row["paper_closed_count"] == 1
        assert options_row["live_events"] == 1
        assert divergence["shadow_selected_count"] == 1
        assert divergence["paper_open_count"] == 1
        assert divergence["live_open_positions"] == 1
    finally:
        monkeypatch.setattr(db, "_DB_PATH", original_path)
        if os.path.exists(db_path):
            os.remove(db_path)


def test_source_trade_outcome_summary_and_health_snapshot_round_trip(monkeypatch):
    fd, raw_path = tempfile.mkstemp(prefix="adaptive_learning_", suffix=".db", dir=os.getcwd())
    os.close(fd)
    db_path = os.path.abspath(raw_path)
    original_path = db._DB_PATH
    monkeypatch.setattr(db, "_DB_PATH", db_path)

    try:
        db.init_db()
        trade_id = db.open_paper_trade(
            strategy_id=1,
            coin="BTC",
            side="long",
            entry_price=100.0,
            size=1.0,
            metadata={"source": "polymarket", "source_key": "polymarket:BTC"},
        )
        db.close_paper_trade(trade_id, exit_price=107.0, pnl=7.0)
        with db.get_connection() as conn:
            conn.execute(
                "UPDATE paper_trades SET closed_at = ? WHERE id = ?",
            (utc_now_iso(), trade_id),
            )

        summary = db.get_source_trade_outcome_summary(lookback_hours=48)
        snapshot_id = db.save_source_health_snapshot(
            {
        "timestamp": utc_now_iso(),
                "metadata": {"summary": {"sources_tracked": 1}},
                "profiles": [
                    {
                        "source_key": "polymarket:BTC",
                        "source": "polymarket",
                        "status": "active",
                        "training_label": "promote",
                        "recommended_action": "increase weight and keep in main ranking set",
                        "health_score": 0.81,
                        "weight_multiplier": 1.12,
                        "confidence_multiplier": 1.05,
                        "sample_size": 6,
                        "recent_sample_size": 3,
                        "selection_count": 8,
                        "closed_trades": 1,
                        "recent_closed_trades": 1,
                        "win_rate": 1.0,
                        "recent_win_rate": 1.0,
                        "avg_return_pct": 0.07,
                        "recent_avg_return_pct": 0.07,
                        "realized_pnl": 7.0,
                        "calibration_ece": 0.03,
                        "drift_score": 0.01,
                        "live_success_rate": 1.0,
                        "live_rejection_rate": 0.0,
                        "metadata": {"selection_share": 1.0},
                    }
                ],
            }
        )
        latest = db.get_latest_source_health_snapshot(limit=5)

        assert summary[0]["source_key"] == "polymarket:BTC"
        assert summary[0]["closed_trades"] == 1
        assert summary[0]["win_rate"] == 1.0
        assert latest["snapshot"]["snapshot_id"] == snapshot_id
        assert latest["profiles"][0]["source_key"] == "polymarket:BTC"
        assert latest["profiles"][0]["training_label"] == "promote"
    finally:
        monkeypatch.setattr(db, "_DB_PATH", original_path)
        if os.path.exists(db_path):
            os.remove(db_path)


def test_context_trade_outcome_summary_groups_by_coin_side_and_regime(monkeypatch):
    fd, raw_path = tempfile.mkstemp(prefix="context_outcomes_", suffix=".db", dir=os.getcwd())
    os.close(fd)
    db_path = os.path.abspath(raw_path)
    original_path = db._DB_PATH
    monkeypatch.setattr(db, "_DB_PATH", db_path)

    try:
        db.init_db()
        trades = [
            {
                "coin": "BTC",
                "side": "long",
                "entry_price": 100.0,
                "exit_price": 106.0,
                "pnl": 6.0,
                "metadata": {
                    "source": "options_flow",
                    "source_key": "options_flow:BTC",
                    "regime": "trending_up",
                },
            },
            {
                "coin": "BTC",
                "side": "long",
                "entry_price": 100.0,
                "exit_price": 104.0,
                "pnl": 4.0,
                "metadata": {
                    "source": "options_flow",
                    "source_key": "options_flow:BTC",
                    "regime": "trending_up",
                },
            },
            {
                "coin": "BTC",
                "side": "long",
                "entry_price": 100.0,
                "exit_price": 97.0,
                "pnl": -3.0,
                "metadata": {
                    "source": "options_flow",
                    "source_key": "options_flow:BTC",
                    "regime": "ranging",
                },
            },
        ]

        closed_ids = []
        for item in trades:
            trade_id = db.open_paper_trade(
                strategy_id=1,
                coin=item["coin"],
                side=item["side"],
                entry_price=item["entry_price"],
                size=1.0,
                metadata=item["metadata"],
            )
            db.close_paper_trade(trade_id, exit_price=item["exit_price"], pnl=item["pnl"])
            closed_ids.append(trade_id)

        with db.get_connection() as conn:
            for trade_id in closed_ids:
                conn.execute(
                    "UPDATE paper_trades SET closed_at = ? WHERE id = ?",
            (utc_now_iso(), trade_id),
                )

        trending_up_rows = db.get_context_trade_outcome_summary(
            lookback_hours=48,
            source_key="options_flow:BTC",
            coin="BTC",
            side="long",
            regime="trending_up",
        )
        all_rows = db.get_context_trade_outcome_summary(
            lookback_hours=48,
            source_key="options_flow:BTC",
            coin="BTC",
            side="long",
        )

        assert len(trending_up_rows) == 1
        assert trending_up_rows[0]["closed_trades"] == 2
        assert trending_up_rows[0]["winning_trades"] == 2
        assert trending_up_rows[0]["regime"] == "trending_up"
        assert trending_up_rows[0]["avg_return_pct"] == 0.05
        assert len(all_rows) == 2
    finally:
        monkeypatch.setattr(db, "_DB_PATH", original_path)
        if os.path.exists(db_path):
            os.remove(db_path)


def test_adaptive_learning_manager_refreshes_profiles_and_reviews_arena(monkeypatch):
    fd, raw_path = tempfile.mkstemp(prefix="adaptive_manager_", suffix=".db", dir=os.getcwd())
    os.close(fd)
    db_path = os.path.abspath(raw_path)
    original_path = db._DB_PATH
    monkeypatch.setattr(db, "_DB_PATH", db_path)

    class FakeScorer:
        def get_all_scores(self):
            return [
                {
                    "source_key": "polymarket:BTC",
                    "total_signals": 14,
                    "accuracy": 0.72,
                    "weighted_accuracy": 0.75,
                    "sharpe": 1.3,
                    "dynamic_weight": 0.84,
                    "total_pnl": 42.0,
                },
                {
                    "source_key": "arena_champion:seed_breakout",
                    "total_signals": 12,
                    "accuracy": 0.33,
                    "weighted_accuracy": 0.31,
                    "sharpe": -0.4,
                    "dynamic_weight": 0.34,
                    "total_pnl": -21.0,
                },
            ]

    class FakeCalibration:
        def get_ece(self, source_key="global"):
            if source_key == "polymarket:BTC":
                return 0.04
            if source_key == "arena_champion:seed_breakout":
                return 0.27
            return 0.08

        def get_adjustment_factor(self, source_key, predicted_confidence):
            if source_key == "polymarket:BTC":
                return predicted_confidence * 1.05
            if source_key == "arena_champion:seed_breakout":
                return predicted_confidence * 0.78
            return predicted_confidence

        def get_all_stats(self):
            return {
                "global": {"total_records": 10},
                "polymarket:BTC": {"total_records": 8},
                "arena_champion:seed_breakout": {"total_records": 8},
            }

    class FakeAgent:
        def __init__(self):
            self.agent_id = "seed_breakout"
            self.name = "Seed_breakout"
            self.strategy_type = "breakout"
            self.status = "champion"
            self.total_trades = 14
            self.win_rate = 0.36
            self.sharpe_ratio = -0.22
            self.max_drawdown = 0.31

    class FakeArena:
        def __init__(self):
            self.agents = {"seed_breakout": FakeAgent()}
            self.saved = False

        def _save_agents(self):
            self.saved = True

    try:
        db.init_db()
        now = utc_now_iso()
        db.save_decision_research_snapshot(
            {
                "timestamp": now,
                "cycle_number": 22,
                "available_slots": 3,
                "candidate_count": 2,
                "qualified_count": 2,
                "executed_count": 2,
                "candidates": [
                    {
                        "rank": 1,
                        "status": "selected",
                        "name": "poly_btc",
                        "source": "polymarket",
                        "source_key": "polymarket:BTC",
                        "strategy_type": "event_driven",
                        "coin": "BTC",
                        "side": "long",
                        "route": "paper_strategy",
                        "composite_score": 0.82,
                        "confidence": 0.74,
                        "expected_value_pct": 0.018,
                        "execution_cost_pct": 0.001,
                        "blockers": [],
                        "score_breakdown": {},
                        "raw_candidate": {
                            "name": "poly_btc",
                            "source": "polymarket",
                            "source_key": "polymarket:BTC",
                            "strategy_type": "event_driven",
                            "current_score": 0.82,
                            "confidence": 0.74,
                            "entry_price": 100.0,
                            "stop_loss": 97.0,
                            "take_profit": 108.0,
                            "parameters": {"coins": ["BTC"]},
                            "metadata": {"source": "polymarket", "source_key": "polymarket:BTC"},
                        },
                    },
                    {
                        "rank": 2,
                        "status": "selected",
                        "name": "arena_breakout",
                        "source": "arena_champion",
                        "source_key": "arena_champion:seed_breakout",
                        "strategy_type": "breakout",
                        "coin": "BTC",
                        "side": "short",
                        "route": "paper_strategy",
                        "composite_score": 0.61,
                        "confidence": 0.62,
                        "expected_value_pct": 0.006,
                        "execution_cost_pct": 0.002,
                        "blockers": [],
                        "score_breakdown": {},
                        "raw_candidate": {
                            "name": "arena_breakout",
                            "source": "arena_champion",
                            "source_key": "arena_champion:seed_breakout",
                            "strategy_type": "breakout",
                            "current_score": 0.61,
                            "confidence": 0.62,
                            "entry_price": 100.0,
                            "stop_loss": 103.0,
                            "take_profit": 94.0,
                            "parameters": {"coins": ["BTC"]},
                            "metadata": {"source": "arena_champion", "source_key": "arena_champion:seed_breakout"},
                        },
                    },
                ],
            }
        )

        good_trade = db.open_paper_trade(
            strategy_id=1,
            coin="BTC",
            side="long",
            entry_price=100.0,
            size=1.0,
            metadata={"source": "polymarket", "source_key": "polymarket:BTC"},
        )
        bad_trade = db.open_paper_trade(
            strategy_id=2,
            coin="BTC",
            side="short",
            entry_price=100.0,
            size=1.0,
            metadata={"source": "arena_champion", "source_key": "arena_champion:seed_breakout"},
        )
        db.close_paper_trade(good_trade, exit_price=108.0, pnl=8.0)
        db.close_paper_trade(bad_trade, exit_price=103.0, pnl=-3.0)
        with db.get_connection() as conn:
            conn.execute(
                "UPDATE paper_trades SET closed_at = ? WHERE id = ?",
            (utc_now_iso(), good_trade),
            )
            conn.execute(
                "UPDATE paper_trades SET closed_at = ? WHERE id = ?",
            (utc_now_iso(), bad_trade),
            )

        db.save_live_execution_event(
            "0xabc",
            timestamp=now,
            source="arena_champion",
            source_key="arena_champion:seed_breakout",
            coin="BTC",
            side="short",
            status="rejected",
            fill_ratio=0.0,
            rejection_reason="adaptive_test",
            protective_status="failed",
        )

        arena = FakeArena()
        manager = AdaptiveLearningManager(
            {
                "enabled": True,
                "lookback_hours": 72,
                "recent_lookback_hours": 72,
                "min_closed_trades": 1,
                "min_recent_closed_trades": 1,
                "min_selected_candidates": 1,
                "caution_health_floor": 0.50,
                "promotion_health_floor": 0.68,
                "caution_drift_threshold": 0.12,
                "block_drift_threshold": 0.25,
                "scaled_promotion_closed_trades": 2,
                "scaled_promotion_recent_closed_trades": 1,
                "scaled_promotion_health_floor": 0.55,
                "scaled_promotion_recent_win_rate": 0.50,
                "full_promotion_closed_trades": 4,
                "full_promotion_recent_closed_trades": 1,
                "full_promotion_health_floor": 0.65,
                "full_promotion_recent_win_rate": 0.55,
                "full_promotion_recent_return_pct": 0.001,
                "promotion_confirm_runs": 1,
                "demotion_confirm_runs": 1,
                "recalibration_enabled": True,
                "recalibration_interval_hours": 0,
                "arena_min_trades": 5,
                "arena_max_drawdown": 0.25,
            },
            agent_scorer=FakeScorer(),
            calibration=FakeCalibration(),
            arena=arena,
        )

        stats = manager.refresh(force=True, cycle_count=96)
        poly = manager.get_source_profile("polymarket:BTC")
        arena_profile = manager.get_source_profile("arena_champion:seed_breakout")
        latest = db.get_latest_source_health_snapshot(limit=10)
        review_events = db.get_recent_arena_review_events(limit=10)

        assert stats["summary"]["sources_tracked"] >= 2
        assert poly["status"] == "active"
        assert poly["training_label"] == "promote"
        assert poly["promotion_stage"] == "full"
        assert stats["summary"]["promotion_stage_counts"]["full"] >= 1
        assert arena_profile["status"] in {"caution", "blocked"}
        assert arena.agents["seed_breakout"].status == "probation"
        assert arena.saved is True
        assert latest["profiles"]
        assert any(profile.get("promotion_stage") for profile in latest["profiles"])
        assert stats["recalibration"]["executed"] is True
        assert stats["recalibration"]["run_id"]
        assert review_events[0]["metrics"]["promotion_stage"] in {"trial", "blocked"}
        assert review_events[0]["agent_id"] == "seed_breakout"
    finally:
        monkeypatch.setattr(db, "_DB_PATH", original_path)
        if os.path.exists(db_path):
            os.remove(db_path)


def test_adaptive_learning_recalibration_requires_confirmation_and_persists_state(monkeypatch):
    fd, raw_path = tempfile.mkstemp(prefix="adaptive_recalibration_", suffix=".db", dir=os.getcwd())
    os.close(fd)
    db_path = os.path.abspath(raw_path)
    original_path = db._DB_PATH
    monkeypatch.setattr(db, "_DB_PATH", db_path)

    class FakeScorer:
        def get_all_scores(self):
            return [
                {
                    "source_key": "options_flow:BTC",
                    "total_signals": 24,
                    "accuracy": 0.72,
                    "weighted_accuracy": 0.75,
                    "dynamic_weight": 0.88,
                    "total_pnl": 120.0,
                }
            ]

    baseline_good = [
        {
            "source_key": "options_flow:BTC",
            "source": "options_flow",
            "closed_trades": 18,
            "winning_trades": 13,
            "win_rate": 0.72,
            "realized_pnl": 140.0,
            "avg_return_pct": 0.018,
        }
    ]
    recent_good = [
        {
            "source_key": "options_flow:BTC",
            "source": "options_flow",
            "closed_trades": 6,
            "winning_trades": 5,
            "win_rate": 0.83,
            "realized_pnl": 65.0,
            "avg_return_pct": 0.026,
        }
    ]
    baseline_soft = [
        {
            "source_key": "options_flow:BTC",
            "source": "options_flow",
            "closed_trades": 18,
            "winning_trades": 10,
            "win_rate": 0.56,
            "realized_pnl": 38.0,
            "avg_return_pct": 0.004,
        }
    ]
    recent_soft = [
        {
            "source_key": "options_flow:BTC",
            "source": "options_flow",
            "closed_trades": 6,
            "winning_trades": 3,
            "win_rate": 0.50,
            "realized_pnl": 6.0,
            "avg_return_pct": 0.0005,
        }
    ]
    current = {
        "baseline": baseline_good,
        "recent": recent_good,
    }

    monkeypatch.setattr(
        db,
        "get_source_attribution_summary",
        lambda **kwargs: [
            {
                "source_key": "options_flow:BTC",
                "source": "options_flow",
                "selected_count": 9,
                "live_events": 3,
                "live_success_rate": 1.0,
                "live_rejection_rate": 0.0,
            }
        ],
    )

    def fake_outcome_summary(lookback_hours):
        if float(lookback_hours) > 24 * 10:
            return current["baseline"]
        return current["recent"]

    monkeypatch.setattr(db, "get_source_trade_outcome_summary", fake_outcome_summary)

    try:
        db.init_db()
        manager = AdaptiveLearningManager(
            {
                "enabled": True,
                "lookback_hours": 24 * 30,
                "recent_lookback_hours": 24 * 3,
                "min_closed_trades": 1,
                "min_recent_closed_trades": 1,
                "min_selected_candidates": 1,
                "caution_health_floor": 0.30,
                "promotion_health_floor": 0.55,
                "caution_drift_threshold": 0.40,
                "block_drift_threshold": 0.80,
                "scaled_promotion_closed_trades": 4,
                "scaled_promotion_recent_closed_trades": 2,
                "scaled_promotion_health_floor": 0.45,
                "scaled_promotion_recent_win_rate": 0.60,
                "scaled_promotion_recent_return_pct": 0.001,
                "full_promotion_closed_trades": 10,
                "full_promotion_recent_closed_trades": 4,
                "full_promotion_health_floor": 0.50,
                "full_promotion_recent_win_rate": 0.70,
                "full_promotion_recent_return_pct": 0.01,
                "recalibration_enabled": True,
                "recalibration_interval_hours": 0,
                "promotion_confirm_runs": 2,
                "demotion_confirm_runs": 2,
                "transition_cooldown_hours": 0,
                "immediate_block_demotion": False,
            },
            agent_scorer=FakeScorer(),
        )

        first = manager.refresh(force=True, cycle_count=10)
        first_profile = manager.get_source_profile("options_flow:BTC")
        assert first["recalibration"]["executed"] is True
        assert first_profile["promotion_stage"] == "trial"
        assert first_profile["promotion_pending_stage"] == "full"
        assert first_profile["promotion_confirmed_runs"] == 1

        second = manager.refresh(force=True, cycle_count=11)
        second_profile = manager.get_source_profile("options_flow:BTC")
        assert second_profile["promotion_stage"] == "full"
        assert second["recalibration"]["promoted_count"] == 1

        current["baseline"] = baseline_soft
        current["recent"] = recent_soft

        manager.refresh(force=True, cycle_count=12)
        third_profile = manager.get_source_profile("options_flow:BTC")
        assert third_profile["promotion_stage"] == "full"
        assert third_profile["promotion_pending_stage"] == "trial"
        assert third_profile["promotion_confirmed_runs"] == 1

        fourth = manager.refresh(force=True, cycle_count=13)
        fourth_profile = manager.get_source_profile("options_flow:BTC")
        assert fourth_profile["promotion_stage"] == "trial"
        assert fourth["recalibration"]["demoted_count"] == 1

        promotion_states = db.get_adaptive_promotion_states(limit=10)
        recalibration_runs = db.get_recent_adaptive_recalibration_runs(limit=10)
        assert promotion_states[0]["applied_stage"] == "trial"
        assert promotion_states[0]["target_stage"] == "trial"
        assert recalibration_runs[0]["transition_count"] >= 1
        assert recalibration_runs[0]["run_id"]
    finally:
        monkeypatch.setattr(db, "_DB_PATH", original_path)
        if os.path.exists(db_path):
            os.remove(db_path)


def test_dashboard_adaptive_learning_metrics_include_runtime_and_snapshot(monkeypatch):
    monkeypatch.setattr(
        dashboard_ui.db,
        "get_latest_source_health_snapshot",
        lambda limit=12: {
            "snapshot": {"snapshot_id": "snap-1", "profile_count": 2},
            "profiles": [{"source_key": "polymarket:BTC", "status": "active"}],
        },
    )
    monkeypatch.setattr(
        dashboard_ui.db,
        "get_recent_arena_review_events",
        lambda limit=10: [{"agent_id": "seed_breakout", "action": "demote"}],
    )
    monkeypatch.setattr(
        dashboard_ui.db,
        "get_recent_adaptive_recalibration_runs",
        lambda limit=10: [{"run_id": "recal-1", "transition_count": 2}],
    )
    monkeypatch.setattr(
        dashboard_ui.db,
        "get_adaptive_promotion_states",
        lambda limit=12: [{"source_key": "polymarket:BTC", "applied_stage": "full"}],
    )

    class FakeAdaptive:
        def get_dashboard_payload(self, limit=12):
            return {"summary": {"sources_tracked": 2}, "profiles": [{"source_key": "options_flow:BTC"}]}

    metrics = dashboard_ui._build_adaptive_learning_metrics(FakeAdaptive())

    assert metrics["latest_snapshot"]["snapshot_id"] == "snap-1"
    assert metrics["profiles"][0]["source_key"] == "polymarket:BTC"
    assert metrics["runtime"]["summary"]["sources_tracked"] == 2
    assert metrics["recent_arena_reviews"][0]["agent_id"] == "seed_breakout"
    assert metrics["recent_recalibrations"][0]["run_id"] == "recal-1"
    assert metrics["promotion_states"][0]["applied_stage"] == "full"


def test_dashboard_source_allocator_metrics_include_runtime(monkeypatch):
    monkeypatch.setattr(
        dashboard_ui.db,
        "get_source_attribution_summary",
        lambda **kwargs: [{"source_key": "strategy:mean_reversion:BTC", "selected_count": 3}],
    )
    monkeypatch.setattr(
        dashboard_ui.db,
        "get_source_trade_outcome_summary",
        lambda **kwargs: [{"source_key": "strategy:mean_reversion:BTC", "realized_pnl": 42.0}],
    )

    class FakeAllocator:
        def get_dashboard_payload(self, limit=12):
            return {"profiles": [{"source_key": "strategy:mean_reversion:BTC", "status": "active"}]}

    metrics = dashboard_ui._build_source_allocator_metrics(FakeAllocator())

    assert metrics["runtime"]["profiles"][0]["status"] == "active"
    assert metrics["attribution"][0]["source_key"] == "strategy:mean_reversion:BTC"


def test_experiment_benchmark_pack_replays_profiles_and_builds_promotion_gate():
    cycles = []
    for idx in range(6):
        confidence = 0.68 if idx < 3 else 0.78
        cycles.append(
            {
                "timestamp": f"2026-04-05T1{idx}:00:00",
                "context": {
                    "regime_data": {"overall_regime": "neutral"},
                    "open_positions": [],
                    "kelly_summary": {},
                },
                "candidates": [
                    {
                        "status": "selected",
                        "raw_candidate": {
                            "name": f"good_{idx}",
                            "source": "strategy",
                            "source_key": "strategy:trend_following",
                            "strategy_type": "trend_following",
                            "current_score": 0.72,
                            "confidence": confidence,
                            "source_accuracy": 0.64,
                            "entry_price": 100.0,
                            "stop_loss": 97.0,
                            "take_profit": 108.0,
                            "parameters": {"coins": ["BTC"]},
                            "metadata": {"source": "strategy", "source_key": "strategy:trend_following"},
                        },
                    },
                    {
                        "status": "blocked",
                        "raw_candidate": {
                            "name": f"bad_{idx}",
                            "source": "options_flow",
                            "source_key": "options_flow:ETH",
                            "strategy_type": "options_momentum",
                            "current_score": 0.70,
                            "confidence": 0.52,
                            "source_accuracy": 0.55,
                            "entry_price": 100.0,
                            "stop_loss": 95.0,
                            "take_profit": 101.0,
                            "parameters": {"coins": ["ETH"]},
                            "metadata": {"source": "options_flow", "source_key": "options_flow:ETH"},
                        },
                    },
                ],
            }
        )

    report = build_experiment_benchmark_pack(
        cycles,
        profiles={
            "baseline_current": {},
            "challenger_selective": {
                "min_signal_confidence": 0.75,
                "min_expected_value_pct": 0.005,
            },
        },
        out_of_sample_ratio=0.5,
        min_oos_cycles=2,
        min_ev_delta_pct=0.0,
    )

    assert report["cycle_count"] == 6
    assert report["out_of_sample_cycles"] == 3
    assert "baseline_current" in report["profiles"]
    assert "challenger_selective" in report["profiles"]
    assert report["profiles"]["baseline_current"]["out_of_sample"]["cycles"] == 3
    assert "approved" in report["profiles"]["challenger_selective"]["promotion"]


def test_decision_engine_blocks_source_when_adaptive_profile_is_blocked():
    class FakeAdaptive:
        def get_source_profile(self, source_key="", source=""):
            assert source_key == "options_flow:BTC"
            return {
                "status": "blocked",
                "health_score": 0.22,
                "drift_score": 0.44,
                "weight_multiplier": 0.25,
                "confidence_multiplier": 0.70,
                "training_label": "freeze",
                "recommended_action": "disable source until retrained",
            }

    engine = DecisionEngine(
        {
            "adaptive_learning": FakeAdaptive(),
            "adaptive_learning_enabled": True,
            "adaptive_learning_block_on_status": True,
            "min_decision_score": 0.0,
            "min_signal_confidence": 0.4,
            "min_source_weight": 0.2,
            "min_expected_value_pct": -1.0,
        }
    )
    candidate = {
        "name": "blocked_flow",
        "source": "options_flow",
        "source_key": "options_flow:BTC",
        "strategy_type": "options_momentum",
        "current_score": 0.82,
        "confidence": 0.82,
        "source_accuracy": 0.88,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 107.0,
        "parameters": {"coins": ["BTC"]},
        "metadata": {"source": "options_flow", "source_key": "options_flow:BTC"},
    }

    selected = engine.decide([candidate], regime_data={"overall_regime": "neutral"}, open_positions=[], kelly_stats={})
    decision = engine.get_decision_history(limit=1)[0]
    blockers = engine._decision_blockers(candidate)

    assert selected == []
    assert decision["executed"] == 0
    assert candidate["_adaptive_learning"]["status"] == "blocked"
    assert candidate["_decision_confidence"] < 0.82
    assert "adaptive_blocked" in blockers


def test_decision_engine_prefers_fully_promoted_source():
    class FakeAdaptive:
        def get_source_profile(self, source_key="", source=""):
            if source_key == "options_flow:BTC":
                return {
                    "status": "active",
                    "health_score": 0.84,
                    "drift_score": 0.05,
                    "weight_multiplier": 1.02,
                    "confidence_multiplier": 1.01,
                    "training_label": "promote",
                    "recommended_action": "full capital access approved",
                    "promotion_stage": "full",
                    "promotion_score": 0.88,
                    "promotion_gate_passed": True,
                    "promotion_cap_pct": 0.32,
                    "metadata": {
                        "promotion_quality_multiplier": 1.08,
                        "promotion_confidence_multiplier": 1.04,
                    },
                }
            return {
                "status": "active",
                "health_score": 0.63,
                "drift_score": 0.08,
                "weight_multiplier": 0.96,
                "confidence_multiplier": 0.97,
                "training_label": "incubate",
                "recommended_action": "collect more evidence before promoting",
                "promotion_stage": "incubating",
                "promotion_score": 0.44,
                "promotion_gate_passed": False,
                "promotion_cap_pct": 0.08,
                "metadata": {
                    "promotion_quality_multiplier": 0.82,
                    "promotion_confidence_multiplier": 0.90,
                },
            }

    engine = DecisionEngine(
        {
            "adaptive_learning": FakeAdaptive(),
            "adaptive_learning_enabled": True,
            "min_decision_score": 0.0,
            "min_signal_confidence": 0.4,
            "min_source_weight": 0.2,
            "min_expected_value_pct": -1.0,
            "max_trades_per_cycle": 1,
            "execution_quality_enabled": False,
            "context_performance_enabled": False,
            "confluence_enabled": False,
            "calibration_enabled": False,
            "memory_enabled": False,
            "divergence_enabled": False,
            "capital_governor_enabled": False,
        }
    )
    btc = {
        "name": "btc_promoted",
        "source": "options_flow",
        "source_key": "options_flow:BTC",
        "strategy_type": "options_momentum",
        "current_score": 0.74,
        "confidence": 0.72,
        "source_accuracy": 0.70,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 107.0,
        "parameters": {"coins": ["BTC"], "direction": "long"},
        "metadata": {"source": "options_flow", "source_key": "options_flow:BTC"},
    }
    eth = {
        "name": "eth_incubating",
        "source": "options_flow",
        "source_key": "options_flow:ETH",
        "strategy_type": "options_momentum",
        "current_score": 0.74,
        "confidence": 0.72,
        "source_accuracy": 0.70,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 107.0,
        "parameters": {"coins": ["ETH"], "direction": "long"},
        "metadata": {"source": "options_flow", "source_key": "options_flow:ETH"},
    }

    selected = engine.decide([eth, btc], regime_data={"overall_regime": "neutral"}, open_positions=[], kelly_stats={})

    assert selected
    assert selected[0]["name"] == "btc_promoted"
    assert btc["_metadata_for_decision"]["promotion_stage"] == "full"
    assert eth["_metadata_for_decision"]["promotion_stage"] == "incubating"


def test_decision_engine_rewards_cross_source_same_coin_confluence():
    engine = DecisionEngine(
        {
            "w_score": 0.10,
            "w_regime": 0.05,
            "w_diversity": 0.05,
            "w_freshness": 0.0,
            "w_consensus": 0.0,
            "w_confidence": 0.05,
            "w_source_quality": 0.05,
            "w_confirmation": 0.0,
            "w_expected_value": 0.10,
            "w_confluence": 0.60,
            "confluence_enabled": True,
            "confluence_full_weight": 0.8,
            "min_decision_score": 0.0,
            "min_signal_confidence": 0.4,
            "min_source_weight": 0.2,
            "min_expected_value_pct": 0.0,
            "max_trades_per_cycle": 1,
        }
    )

    isolated = {
        "name": "isolated_sol",
        "source": "strategy",
        "source_key": "strategy:mean_reversion:SOL",
        "strategy_type": "mean_reversion",
        "current_score": 0.72,
        "confidence": 0.64,
        "source_accuracy": 0.62,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 107.0,
        "parameters": {"coins": ["SOL"], "direction": "long"},
        "metadata": {"source": "strategy", "source_key": "strategy:mean_reversion:SOL"},
    }
    btc_strategy = {
        "name": "btc_strategy",
        "source": "strategy",
        "source_key": "strategy:trend_following:BTC",
        "strategy_type": "trend_following",
        "current_score": 0.68,
        "confidence": 0.67,
        "source_accuracy": 0.64,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 107.0,
        "parameters": {"coins": ["BTC"], "direction": "long"},
        "metadata": {"source": "strategy", "source_key": "strategy:trend_following:BTC"},
    }
    btc_options = {
        "name": "btc_options",
        "source": "options_flow",
        "source_key": "options_flow:BTC",
        "strategy_type": "options_momentum",
        "current_score": 0.67,
        "confidence": 0.76,
        "source_accuracy": 0.75,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 107.0,
        "parameters": {"coins": ["BTC"], "direction": "long"},
        "metadata": {"source": "options_flow", "source_key": "options_flow:BTC"},
    }
    confluence_map = engine._build_source_confluence_map(
        [isolated, btc_strategy, btc_options],
        {"overall_regime": "neutral"},
    )
    isolated_breakdown = engine._compute_composite_score(
        dict(isolated),
        regime_data={"overall_regime": "neutral"},
        open_coins=set(),
        kelly_stats={},
        confluence_map=confluence_map,
    )

    selected = engine.decide(
        [isolated, btc_strategy, btc_options],
        regime_data={"overall_regime": "neutral"},
        open_positions=[],
        kelly_stats={},
    )

    assert selected
    assert selected[0]["name"] == "btc_strategy"
    assert selected[0]["_score_breakdown"]["confluence"] > isolated_breakdown["confluence"]
    assert selected[0]["_source_confluence"]["support_source_count"] == 1
    assert selected[0]["_source_confluence"]["conflict_source_count"] == 0


def test_decision_engine_blocks_underperforming_context(monkeypatch):
    def fake_context_trade_outcome_summary(**kwargs):
        if (
            kwargs.get("source_key") == "options_flow:BTC"
            and kwargs.get("coin") == "BTC"
            and kwargs.get("side") == "long"
            and kwargs.get("regime") == "trending_up"
        ):
            return [
                {
                    "source_key": "options_flow:BTC",
                    "source": "options_flow",
                    "coin": "BTC",
                    "side": "long",
                    "regime": "trending_up",
                    "closed_trades": 6,
                    "winning_trades": 1,
                    "losing_trades": 5,
                    "realized_pnl": -12.0,
                    "gross_profit": 2.0,
                    "gross_loss": 14.0,
                    "avg_return_pct": -0.025,
                    "win_rate": 0.1667,
                    "profit_factor": 0.1429,
                "last_closed_at": utc_now_iso(),
                }
            ]
        return []

    monkeypatch.setattr(
        "src.signals.decision_engine.db.get_context_trade_outcome_summary",
        fake_context_trade_outcome_summary,
    )

    engine = DecisionEngine(
        {
            "w_context": 0.60,
            "min_decision_score": 0.0,
            "min_signal_confidence": 0.4,
            "min_source_weight": 0.2,
            "min_expected_value_pct": -1.0,
            "execution_quality_enabled": False,
            "adaptive_learning_enabled": False,
            "context_performance_enabled": True,
            "context_performance_min_trades": 3,
            "context_performance_block_win_rate": 0.30,
            "context_performance_block_avg_return_pct": -0.01,
        }
    )

    candidate = {
        "name": "btc_options_flow",
        "source": "options_flow",
        "source_key": "options_flow:BTC",
        "strategy_type": "options_momentum",
        "current_score": 0.84,
        "confidence": 0.78,
        "source_accuracy": 0.74,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 108.0,
        "parameters": {"coins": ["BTC"], "direction": "long"},
        "metadata": {"source": "options_flow", "source_key": "options_flow:BTC"},
    }
    selected = engine.decide(
        [candidate],
        regime_data={"overall_regime": "trending_up"},
        open_positions=[],
        kelly_stats={},
    )
    blockers = engine._decision_blockers(candidate)

    assert selected == []
    assert candidate["_context_performance"]["blocked"] is True
    assert candidate["_context_performance"]["context_score"] < 0.5
    assert "context_underperforming" in blockers
    assert engine.get_stats()["total_context_blocks"] >= 1


def test_decision_engine_prefers_candidates_with_better_recent_context(monkeypatch):
    def fake_context_trade_outcome_summary(**kwargs):
        source_key = kwargs.get("source_key")
        regime = kwargs.get("regime")
        if source_key == "options_flow:BTC" and regime == "trending_up":
            return [
                {
                    "source_key": "options_flow:BTC",
                    "source": "options_flow",
                    "coin": "BTC",
                    "side": "long",
                    "regime": "trending_up",
                    "closed_trades": 8,
                    "winning_trades": 6,
                    "losing_trades": 2,
                    "realized_pnl": 18.0,
                    "gross_profit": 24.0,
                    "gross_loss": 6.0,
                    "avg_return_pct": 0.04,
                    "win_rate": 0.75,
                    "profit_factor": 4.0,
                "last_closed_at": utc_now_iso(),
                }
            ]
        if source_key == "strategy:trend_following:ETH" and regime == "trending_up":
            return [
                {
                    "source_key": "strategy:trend_following:ETH",
                    "source": "strategy",
                    "coin": "ETH",
                    "side": "long",
                    "regime": "trending_up",
                    "closed_trades": 6,
                    "winning_trades": 3,
                    "losing_trades": 3,
                    "realized_pnl": -1.0,
                    "gross_profit": 9.0,
                    "gross_loss": 10.0,
                    "avg_return_pct": -0.002,
                    "win_rate": 0.5,
                    "profit_factor": 0.9,
                "last_closed_at": utc_now_iso(),
                }
            ]
        return []

    monkeypatch.setattr(
        "src.signals.decision_engine.db.get_context_trade_outcome_summary",
        fake_context_trade_outcome_summary,
    )

    engine = DecisionEngine(
        {
            "w_score": 0.10,
            "w_regime": 0.0,
            "w_diversity": 0.0,
            "w_freshness": 0.0,
            "w_consensus": 0.0,
            "w_confidence": 0.05,
            "w_source_quality": 0.05,
            "w_confirmation": 0.0,
            "w_expected_value": 0.05,
            "w_confluence": 0.0,
            "w_context": 0.75,
            "min_decision_score": 0.0,
            "min_signal_confidence": 0.4,
            "min_source_weight": 0.2,
            "min_expected_value_pct": -1.0,
            "execution_quality_enabled": False,
            "adaptive_learning_enabled": False,
            "confluence_enabled": False,
        }
    )

    good_context = {
        "name": "good_context",
        "source": "options_flow",
        "source_key": "options_flow:BTC",
        "strategy_type": "options_momentum",
        "current_score": 0.70,
        "confidence": 0.70,
        "source_accuracy": 0.68,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 108.0,
        "parameters": {"coins": ["BTC"], "direction": "long"},
        "metadata": {"source": "options_flow", "source_key": "options_flow:BTC"},
    }
    weaker_context = {
        "name": "weaker_context",
        "source": "strategy",
        "source_key": "strategy:trend_following:ETH",
        "strategy_type": "trend_following",
        "current_score": 0.79,
        "confidence": 0.70,
        "source_accuracy": 0.68,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 108.0,
        "parameters": {"coins": ["ETH"], "direction": "long"},
        "metadata": {"source": "strategy", "source_key": "strategy:trend_following:ETH"},
    }

    good_breakdown = engine._compute_composite_score(
        dict(good_context),
        regime_data={"overall_regime": "trending_up"},
        open_coins=set(),
        kelly_stats=None,
    )
    weaker_breakdown = engine._compute_composite_score(
        dict(weaker_context),
        regime_data={"overall_regime": "trending_up"},
        open_coins=set(),
        kelly_stats=None,
    )
    selected = engine.decide(
        [weaker_context, good_context],
        regime_data={"overall_regime": "trending_up"},
        open_positions=[],
        kelly_stats={},
    )

    assert selected
    assert selected[0]["name"] == "good_context"
    assert good_breakdown["context"] > weaker_breakdown["context"]
    assert good_breakdown["total"] > weaker_breakdown["total"]


def test_decision_engine_blocks_poorly_calibrated_source():
    class FakeCalibration:
        def get_all_stats(self):
            return {
                "options_flow:BTC": {
                    "total_records": 36,
                    "ece": 0.26,
                    "calibration_quality": "poor",
                },
                "global": {
                    "total_records": 80,
                    "ece": 0.09,
                    "calibration_quality": "good",
                },
            }

        def get_adjustment_factor(self, source_key, predicted_confidence):
            if source_key == "options_flow:BTC":
                return 0.46
            return predicted_confidence

        def get_ece(self, source_key="global"):
            return self.get_all_stats().get(source_key, {}).get("ece")

    engine = DecisionEngine(
        {
            "w_calibration": 0.60,
            "min_decision_score": 0.0,
            "min_signal_confidence": 0.4,
            "min_source_weight": 0.2,
            "min_expected_value_pct": -1.0,
            "execution_quality_enabled": False,
            "adaptive_learning_enabled": False,
            "context_performance_enabled": False,
            "confluence_enabled": False,
            "calibration": FakeCalibration(),
            "calibration_enabled": True,
            "calibration_min_records": 20,
            "calibration_target_ece": 0.05,
            "calibration_max_ece": 0.20,
        }
    )

    candidate = {
        "name": "btc_options_flow",
        "source": "options_flow",
        "source_key": "options_flow:BTC",
        "strategy_type": "options_momentum",
        "current_score": 0.82,
        "confidence": 0.83,
        "source_accuracy": 0.74,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 108.0,
        "parameters": {"coins": ["BTC"], "direction": "long"},
        "metadata": {"source": "options_flow", "source_key": "options_flow:BTC"},
    }
    engine._compute_composite_score(
        dict(candidate),
        regime_data={"overall_regime": "trending_up"},
        open_coins=set(),
        kelly_stats=None,
    )

    selected = engine.decide(
        [candidate],
        regime_data={"overall_regime": "trending_up"},
        open_positions=[],
        kelly_stats={},
    )

    assert selected == []
    assert candidate["_calibration"]["blocked"] is True
    assert candidate["_decision_confidence"] == 0.46
    assert candidate["_calibration"]["calibration_score"] == 0.0
    assert candidate["_calibration"]["calibration_quality"] == "poor"
    assert "calibration_poor" in engine._decision_blockers(candidate)
    assert engine.get_stats()["total_calibration_blocks"] >= 1


def test_decision_engine_prefers_better_calibrated_source():
    class FakeCalibration:
        def get_all_stats(self):
            return {
                "options_flow:BTC": {
                    "total_records": 32,
                    "ece": 0.17,
                    "calibration_quality": "fair",
                },
                "strategy:trend_following:ETH": {
                    "total_records": 32,
                    "ece": 0.03,
                    "calibration_quality": "excellent",
                },
                "global": {
                    "total_records": 100,
                    "ece": 0.07,
                    "calibration_quality": "good",
                },
            }

        def get_adjustment_factor(self, source_key, predicted_confidence):
            if source_key == "options_flow:BTC":
                return 0.54
            if source_key == "strategy:trend_following:ETH":
                return 0.72
            return predicted_confidence

        def get_ece(self, source_key="global"):
            return self.get_all_stats().get(source_key, {}).get("ece")

    engine = DecisionEngine(
        {
            "w_score": 0.08,
            "w_regime": 0.0,
            "w_diversity": 0.0,
            "w_freshness": 0.0,
            "w_consensus": 0.0,
            "w_confidence": 0.05,
            "w_source_quality": 0.05,
            "w_confirmation": 0.0,
            "w_expected_value": 0.05,
            "w_context": 0.0,
            "w_confluence": 0.0,
            "w_calibration": 0.77,
            "min_decision_score": 0.0,
            "min_signal_confidence": 0.4,
            "min_source_weight": 0.2,
            "min_expected_value_pct": -1.0,
            "execution_quality_enabled": False,
            "adaptive_learning_enabled": False,
            "context_performance_enabled": False,
            "confluence_enabled": False,
            "calibration": FakeCalibration(),
            "calibration_enabled": True,
            "calibration_min_records": 20,
            "calibration_target_ece": 0.05,
            "calibration_max_ece": 0.22,
        }
    )

    weaker_calibrated = {
        "name": "weaker_calibrated",
        "source": "options_flow",
        "source_key": "options_flow:BTC",
        "strategy_type": "options_momentum",
        "current_score": 0.83,
        "confidence": 0.72,
        "source_accuracy": 0.70,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 108.0,
        "parameters": {"coins": ["BTC"], "direction": "long"},
        "metadata": {"source": "options_flow", "source_key": "options_flow:BTC"},
    }
    better_calibrated = {
        "name": "better_calibrated",
        "source": "strategy",
        "source_key": "strategy:trend_following:ETH",
        "strategy_type": "trend_following",
        "current_score": 0.78,
        "confidence": 0.72,
        "source_accuracy": 0.70,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 108.0,
        "parameters": {"coins": ["ETH"], "direction": "long"},
        "metadata": {"source": "strategy", "source_key": "strategy:trend_following:ETH"},
    }

    weak_breakdown = engine._compute_composite_score(
        dict(weaker_calibrated),
        regime_data={"overall_regime": "trending_up"},
        open_coins=set(),
        kelly_stats=None,
    )
    strong_breakdown = engine._compute_composite_score(
        dict(better_calibrated),
        regime_data={"overall_regime": "trending_up"},
        open_coins=set(),
        kelly_stats=None,
    )
    selected = engine.decide(
        [weaker_calibrated, better_calibrated],
        regime_data={"overall_regime": "trending_up"},
        open_positions=[],
        kelly_stats={},
    )

    assert selected
    assert selected[0]["name"] == "better_calibrated"
    assert strong_breakdown["calibration"] > weak_breakdown["calibration"]
    assert strong_breakdown["calibrated_confidence"] > weak_breakdown["calibrated_confidence"]
    assert strong_breakdown["total"] > weak_breakdown["total"]


def test_decision_engine_blocks_memory_avoid_setups():
    class FakeTradeMemory:
        def find_similar(self, features, coin=None, strategy_type=None, side=None, top_k=10, min_similarity=0.5):
            assert coin == "BTC"
            assert strategy_type == "options_momentum"
            assert side == "long"
            assert features["overall_score"] == 0.81
            return {
                "similar_trades": [{"trade_id": "a1"}, {"trade_id": "a2"}, {"trade_id": "a3"}],
                "total_found": 4,
                "win_rate": 0.25,
                "avg_pnl": -5.8,
                "avg_return": -0.041,
                "recommendation": "avoid",
                "reason": "Similar setups mostly lost money",
                "similarity_scores": [0.93, 0.88, 0.84, 0.79],
            }

    engine = DecisionEngine(
        {
            "w_memory": 0.60,
            "min_decision_score": 0.0,
            "min_signal_confidence": 0.35,
            "min_source_weight": 0.2,
            "min_expected_value_pct": -1.0,
            "execution_quality_enabled": False,
            "adaptive_learning_enabled": False,
            "context_performance_enabled": False,
            "confluence_enabled": False,
            "calibration_enabled": False,
            "trade_memory": FakeTradeMemory(),
            "memory_enabled": True,
            "memory_min_trades": 3,
            "memory_block_on_avoid": True,
        }
    )

    candidate = {
        "name": "avoid_setup",
        "source": "options_flow",
        "source_key": "options_flow:BTC",
        "strategy_type": "options_momentum",
        "current_score": 0.81,
        "confidence": 0.78,
        "source_accuracy": 0.72,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 108.0,
        "features": {"overall_score": 0.81, "trend_strength": 0.74, "volatility": 0.02},
        "parameters": {"coins": ["BTC"], "direction": "long"},
        "metadata": {"source": "options_flow", "source_key": "options_flow:BTC"},
    }
    breakdown = engine._compute_composite_score(
        dict(candidate),
        regime_data={"overall_regime": "trending_up"},
        open_coins=set(),
        kelly_stats=None,
    )

    selected = engine.decide(
        [candidate],
        regime_data={"overall_regime": "trending_up"},
        open_positions=[],
        kelly_stats={},
    )

    assert selected == []
    assert candidate["_trade_memory"]["blocked"] is True
    assert candidate["_trade_memory"]["recommendation"] == "avoid"
    assert breakdown["memory"] < 0.4
    assert "memory_avoid" in engine._decision_blockers(candidate)
    assert engine.get_stats()["total_memory_blocks"] >= 1


def test_decision_engine_prefers_candidates_with_stronger_trade_memory():
    class FakeTradeMemory:
        def find_similar(self, features, coin=None, strategy_type=None, side=None, top_k=10, min_similarity=0.5):
            if coin == "BTC":
                return {
                    "similar_trades": [{"trade_id": "btc-1"}, {"trade_id": "btc-2"}],
                    "total_found": 6,
                    "win_rate": 0.83,
                    "avg_pnl": 7.2,
                    "avg_return": 0.062,
                    "recommendation": "proceed",
                    "reason": "Historically strong setup cluster",
                    "similarity_scores": [0.92, 0.89, 0.86, 0.84, 0.82, 0.81],
                }
            return {
                "similar_trades": [{"trade_id": "eth-1"}, {"trade_id": "eth-2"}],
                "total_found": 5,
                "win_rate": 0.40,
                "avg_pnl": -0.4,
                "avg_return": -0.006,
                "recommendation": "caution",
                "reason": "Mixed results for similar setups",
                "similarity_scores": [0.90, 0.86, 0.82, 0.79, 0.76],
            }

    engine = DecisionEngine(
        {
            "w_score": 0.05,
            "w_regime": 0.0,
            "w_diversity": 0.0,
            "w_freshness": 0.0,
            "w_consensus": 0.0,
            "w_confidence": 0.03,
            "w_source_quality": 0.03,
            "w_confirmation": 0.0,
            "w_expected_value": 0.05,
            "w_context": 0.0,
            "w_confluence": 0.0,
            "w_calibration": 0.0,
            "w_memory": 0.84,
            "min_decision_score": 0.0,
            "min_signal_confidence": 0.35,
            "min_source_weight": 0.2,
            "min_expected_value_pct": -1.0,
            "execution_quality_enabled": False,
            "adaptive_learning_enabled": False,
            "context_performance_enabled": False,
            "confluence_enabled": False,
            "calibration_enabled": False,
            "trade_memory": FakeTradeMemory(),
            "memory_enabled": True,
            "memory_min_trades": 3,
            "memory_block_on_avoid": True,
        }
    )

    stronger_memory = {
        "name": "stronger_memory",
        "source": "options_flow",
        "source_key": "options_flow:BTC",
        "strategy_type": "options_momentum",
        "current_score": 0.74,
        "confidence": 0.70,
        "source_accuracy": 0.66,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 108.0,
        "features": {"overall_score": 0.74, "trend_strength": 0.81, "volatility": 0.018},
        "parameters": {"coins": ["BTC"], "direction": "long"},
        "metadata": {"source": "options_flow", "source_key": "options_flow:BTC"},
    }
    weaker_memory = {
        "name": "weaker_memory",
        "source": "strategy",
        "source_key": "strategy:trend_following:ETH",
        "strategy_type": "trend_following",
        "current_score": 0.81,
        "confidence": 0.70,
        "source_accuracy": 0.66,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 108.0,
        "features": {"overall_score": 0.81, "trend_strength": 0.58, "volatility": 0.021},
        "parameters": {"coins": ["ETH"], "direction": "long"},
        "metadata": {"source": "strategy", "source_key": "strategy:trend_following:ETH"},
    }

    strong_breakdown = engine._compute_composite_score(
        dict(stronger_memory),
        regime_data={"overall_regime": "trending_up"},
        open_coins=set(),
        kelly_stats=None,
    )
    weak_breakdown = engine._compute_composite_score(
        dict(weaker_memory),
        regime_data={"overall_regime": "trending_up"},
        open_coins=set(),
        kelly_stats=None,
    )
    selected = engine.decide(
        [weaker_memory, stronger_memory],
        regime_data={"overall_regime": "trending_up"},
        open_positions=[],
        kelly_stats={},
    )

    assert selected
    assert selected[0]["name"] == "stronger_memory"
    assert selected[0]["_trade_memory"]["recommendation"] == "proceed"
    assert strong_breakdown["memory"] > weak_breakdown["memory"]
    assert strong_breakdown["total"] > weak_breakdown["total"]


def test_decision_engine_prefers_higher_net_expectancy_over_raw_score():
    engine = DecisionEngine(
        {
            "w_score": 0.10,
            "w_regime": 0.05,
            "w_diversity": 0.05,
            "w_freshness": 0.0,
            "w_consensus": 0.0,
            "w_confidence": 0.15,
            "w_source_quality": 0.10,
            "w_confirmation": 0.05,
            "w_expected_value": 0.50,
            "min_decision_score": 0.0,
            "min_signal_confidence": 0.4,
            "min_source_weight": 0.3,
            "min_expected_value_pct": 0.0,
            "expected_slippage_bps": 3.0,
            "taker_fee_bps": 4.5,
            "churn_penalty_bps": 2.0,
        }
    )

    high_score_bad_ev = {
        "name": "bad_ev",
        "strategy_type": "trend_following",
        "current_score": 0.92,
        "confidence": 0.59,
        "source_accuracy": 0.55,
        "side": "long",
        "entry_price": 100.0,
        "stop_loss": 94.0,
        "take_profit": 101.0,
        "parameters": {"coins": ["BTC"]},
        "metadata": {"taker_fee_bps": 12.0, "expected_slippage_bps": 20.0},
    }
    lower_score_good_ev = {
        "name": "good_ev",
        "strategy_type": "trend_following",
        "current_score": 0.66,
        "confidence": 0.67,
        "source_accuracy": 0.64,
        "side": "long",
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 109.0,
        "parameters": {"coins": ["ETH"]},
        "metadata": {"taker_fee_bps": 4.5, "expected_slippage_bps": 3.0},
    }

    bad_breakdown = engine._compute_composite_score(
        dict(high_score_bad_ev), regime_data={"overall_regime": "neutral"}, open_coins=set(), kelly_stats=None
    )
    good_breakdown = engine._compute_composite_score(
        dict(lower_score_good_ev), regime_data={"overall_regime": "neutral"}, open_coins=set(), kelly_stats=None
    )
    selected = engine.decide([high_score_bad_ev, lower_score_good_ev], regime_data={"overall_regime": "neutral"})

    assert selected
    assert selected[0]["name"] == "good_ev"
    assert good_breakdown["net_expectancy_pct"] > bad_breakdown["net_expectancy_pct"]
    assert good_breakdown["expected_value"] > bad_breakdown["expected_value"]


def test_decision_engine_blocks_strong_same_coin_source_conflict():
    engine = DecisionEngine(
        {
            "confluence_enabled": True,
            "w_confluence": 0.40,
            "confluence_full_weight": 0.75,
            "confluence_conflict_block_threshold": 0.60,
            "confluence_conflict_floor": 0.35,
            "min_decision_score": 0.0,
            "min_signal_confidence": 0.4,
            "min_source_weight": 0.2,
            "min_expected_value_pct": 0.0,
        }
    )

    long_candidate = {
        "name": "btc_long_strategy",
        "source": "strategy",
        "source_key": "strategy:trend_following:BTC",
        "strategy_type": "trend_following",
        "current_score": 0.80,
        "confidence": 0.82,
        "source_accuracy": 0.76,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 107.0,
        "parameters": {"coins": ["BTC"], "direction": "long"},
        "metadata": {"source": "strategy", "source_key": "strategy:trend_following:BTC"},
    }
    short_candidate = {
        "name": "btc_short_poly",
        "source": "polymarket",
        "source_key": "polymarket:BTC",
        "strategy_type": "event_driven",
        "current_score": 0.79,
        "confidence": 0.84,
        "source_accuracy": 0.79,
        "entry_price": 100.0,
        "stop_loss": 103.0,
        "take_profit": 93.0,
        "parameters": {"coins": ["BTC"], "direction": "short"},
        "metadata": {"source": "polymarket", "source_key": "polymarket:BTC"},
    }

    selected = engine.decide(
        [long_candidate, short_candidate],
        regime_data={"overall_regime": "neutral"},
        open_positions=[],
        kelly_stats={},
    )

    assert selected == []
    assert "source_conflict" in engine._decision_blockers(long_candidate)
    assert "source_conflict" in engine._decision_blockers(short_candidate)
    assert engine.get_stats()["total_source_conflict_blocks"] >= 2


def test_decision_engine_blocks_negative_expected_value_candidates():
    engine = DecisionEngine(
        {
            "min_decision_score": 0.0,
            "min_signal_confidence": 0.4,
            "min_source_weight": 0.3,
            "min_expected_value_pct": 0.0015,
        }
    )

    candidate = {
        "name": "negative_ev",
        "strategy_type": "mean_reversion",
        "current_score": 0.8,
        "confidence": 0.55,
        "source_accuracy": 0.5,
        "side": "long",
        "entry_price": 100.0,
        "stop_loss": 94.0,
        "take_profit": 101.0,
        "parameters": {"coins": ["SOL"]},
        "metadata": {"taker_fee_bps": 15.0, "expected_slippage_bps": 25.0},
    }

    breakdown = engine._compute_composite_score(candidate, regime_data=None, open_coins=set(), kelly_stats=None)
    candidate["_score_breakdown"] = breakdown
    blockers = engine._decision_blockers(candidate)

    assert breakdown["net_expectancy_pct"] < 0.0015
    assert any(blocker.startswith("net_ev<") for blocker in blockers)


def test_decision_engine_uses_kelly_edge_to_improve_expected_value():
    engine = DecisionEngine(
        {
            "min_decision_score": 0.0,
            "min_signal_confidence": 0.4,
            "min_source_weight": 0.3,
            "min_expected_value_pct": 0.0,
        }
    )

    candidate = {
        "name": "kelly_candidate",
        "strategy_type": "trend_following",
        "source_key": "strategy:trend_following",
        "current_score": 0.7,
        "confidence": 0.62,
        "source_accuracy": 0.58,
        "side": "long",
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 106.0,
        "parameters": {"coins": ["BTC"]},
        "metadata": {},
    }

    no_kelly = engine._compute_composite_score(dict(candidate), regime_data=None, open_coins=set(), kelly_stats={})
    with_kelly = engine._compute_composite_score(
        dict(candidate),
        regime_data=None,
        open_coins=set(),
        kelly_stats={
            "strategy:trend_following": {
                "trades": 40,
                "win_rate": 0.71,
                "has_edge": True,
            }
        },
    )

    assert with_kelly["net_expectancy_pct"] > no_kelly["net_expectancy_pct"]
    assert with_kelly["expected_value"] > no_kelly["expected_value"]


def test_decision_engine_penalizes_sources_with_bad_execution_history(monkeypatch):
    def fake_execution_quality_summary(public_address=None, source_key=None, source=None, lookback_hours=None):
        if source_key == "options_flow:BTC":
            return {
                "total_events": 10,
                "rejection_rate": 0.5,
                "avg_fill_ratio": 0.62,
                "avg_realized_slippage_bps": 21.0,
                "protective_failure_rate": 0.2,
                "maker_ratio": 0.0,
            }
        return {"total_events": 0}

    monkeypatch.setattr(
        "src.signals.decision_engine.db.get_execution_quality_summary",
        fake_execution_quality_summary,
    )

    engine = DecisionEngine(
        {
            "w_score": 0.10,
            "w_regime": 0.05,
            "w_diversity": 0.05,
            "w_freshness": 0.0,
            "w_consensus": 0.0,
            "w_confidence": 0.15,
            "w_source_quality": 0.10,
            "w_confirmation": 0.05,
            "w_expected_value": 0.50,
            "min_decision_score": 0.0,
            "min_signal_confidence": 0.4,
            "min_source_weight": 0.3,
            "min_expected_value_pct": 0.0,
            "execution_quality_enabled": True,
            "execution_quality_min_events": 3,
        }
    )

    bad_execution = {
        "name": "bad_execution",
        "strategy_type": "options_momentum",
        "source": "options_flow",
        "source_key": "options_flow:BTC",
        "current_score": 0.76,
        "confidence": 0.67,
        "source_accuracy": 0.64,
        "side": "long",
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 108.0,
        "parameters": {"coins": ["BTC"]},
        "metadata": {},
    }
    clean_execution = {
        "name": "clean_execution",
        "strategy_type": "trend_following",
        "source": "strategy",
        "source_key": "strategy:trend_following",
        "current_score": 0.74,
        "confidence": 0.67,
        "source_accuracy": 0.64,
        "side": "long",
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 108.0,
        "parameters": {"coins": ["ETH"]},
        "metadata": {},
    }

    bad_breakdown = engine._compute_composite_score(
        dict(bad_execution), regime_data={"overall_regime": "neutral"}, open_coins=set(), kelly_stats=None
    )
    clean_breakdown = engine._compute_composite_score(
        dict(clean_execution), regime_data={"overall_regime": "neutral"}, open_coins=set(), kelly_stats=None
    )
    selected = engine.decide(
        [bad_execution, clean_execution],
        regime_data={"overall_regime": "neutral"},
    )

    assert selected
    assert selected[0]["name"] == "clean_execution"
    assert bad_breakdown["execution_quality_events"] == 10
    assert bad_breakdown["execution_penalty_bps"] > 0
    assert clean_breakdown["net_expectancy_pct"] > bad_breakdown["net_expectancy_pct"]


def test_decision_research_snapshot_round_trips(monkeypatch):
    fd, raw_path = tempfile.mkstemp(prefix="decision_research_", suffix=".db", dir=os.getcwd())
    os.close(fd)
    db_path = os.path.abspath(raw_path)
    original_path = db._DB_PATH
    monkeypatch.setattr(db, "_DB_PATH", db_path)

    try:
        db.init_db()
        research_id = db.save_decision_research_snapshot(
            {
                "timestamp": "2026-04-05T15:00:00",
                "cycle_number": 12,
                "regime": "trending_up",
                "available_slots": 5,
                "candidate_count": 2,
                "qualified_count": 1,
                "executed_count": 1,
                "long_score": 1.4,
                "short_score": 0.6,
                "market_bias": "long",
                "context": {
                    "regime_data": {"overall_regime": "trending_up"},
                    "open_positions": [{"coin": "SOL", "side": "long"}],
                },
                "candidates": [
                    {
                        "rank": 1,
                        "status": "selected",
                        "name": "good_ev",
                        "source": "strategy",
                        "source_key": "strategy:trend_following",
                        "strategy_type": "trend_following",
                        "coin": "BTC",
                        "side": "long",
                        "route": "paper_strategy",
                        "composite_score": 0.84,
                        "confidence": 0.66,
                        "expected_value_pct": 0.014,
                        "execution_cost_pct": 0.0011,
                        "blockers": [],
                        "score_breakdown": {"expected_value": 0.82},
                        "raw_candidate": {"name": "good_ev", "metadata": {"source": "strategy"}},
                    },
                    {
                        "rank": 2,
                        "status": "blocked",
                        "name": "bad_ev",
                        "source": "options_flow",
                        "source_key": "options_flow:ETH",
                        "strategy_type": "options_momentum",
                        "coin": "ETH",
                        "side": "short",
                        "route": "paper_strategy",
                        "composite_score": 0.28,
                        "confidence": 0.55,
                        "expected_value_pct": -0.002,
                        "execution_cost_pct": 0.004,
                        "blockers": ["net_ev<0.0015"],
                        "score_breakdown": {"expected_value": 0.12},
                        "raw_candidate": {"name": "bad_ev", "metadata": {"source": "options_flow"}},
                    },
                ],
            }
        )

        recent = db.get_recent_decision_research(limit=5, include_candidates=True)

        assert research_id > 0
        assert len(recent) == 1
        assert recent[0]["cycle_number"] == 12
        assert recent[0]["context"]["regime_data"]["overall_regime"] == "trending_up"
        assert len(recent[0]["candidates"]) == 2
        assert recent[0]["candidates"][0]["status"] == "selected"
        assert recent[0]["candidates"][1]["blockers"] == ["net_ev<0.0015"]
    finally:
        monkeypatch.setattr(db, "_DB_PATH", original_path)
        if os.path.exists(db_path):
            os.remove(db_path)


def test_decision_engine_persists_research_cycle(monkeypatch):
    saved_snapshots = []

    monkeypatch.setattr(
        "src.signals.decision_engine.db.save_decision_research_snapshot",
        lambda snapshot: saved_snapshots.append(snapshot) or 77,
    )

    engine = DecisionEngine(
        {
            "persist_research": True,
            "min_decision_score": 0.0,
            "min_signal_confidence": 0.4,
            "min_source_weight": 0.3,
            "min_expected_value_pct": 0.0,
        }
    )
    candidate = {
        "name": "persist_me",
        "strategy_type": "trend_following",
        "source": "strategy",
        "source_key": "strategy:trend_following",
        "current_score": 0.7,
        "confidence": 0.66,
        "source_accuracy": 0.62,
        "side": "long",
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 107.0,
        "parameters": {"coins": ["BTC"]},
        "metadata": {"source": "strategy"},
    }

    selected = engine.decide(
        [candidate],
        regime_data={"overall_regime": "neutral"},
        open_positions=[{"coin": "SOL", "side": "long"}],
        kelly_stats={"strategy:trend_following": {"trades": 25, "win_rate": 0.64, "has_edge": True}},
    )

    assert selected
    assert saved_snapshots
    assert saved_snapshots[0]["candidate_count"] == 1
    assert saved_snapshots[0]["candidates"][0]["status"] == "selected"
    assert saved_snapshots[0]["context"]["kelly_summary"]["strategy:trend_following"]["has_edge"] is True
    assert engine.get_stats()["last_research_cycle_id"] == 77


def test_get_v2_metrics_includes_persisted_decision_research(monkeypatch):
    class EmptyConn:
        def execute(self, query):
            class Result:
                def fetchall(self):
                    return []

            return Result()

    monkeypatch.setattr(
        dashboard_ui.db,
        "get_recent_decision_research",
        lambda limit=8, include_candidates=True: [
            {
                "id": 1,
                "cycle_number": 3,
                "market_bias": "long",
                "candidates": [{"name": "btc_long", "status": "selected"}],
            }
        ],
    )

    metrics = dashboard_ui._get_v2_metrics(EmptyConn())

    assert metrics["decision_research"][0]["cycle_number"] == 3
    assert metrics["decision_research"][0]["candidates"][0]["name"] == "btc_long"


def test_portfolio_sizer_reduces_size_and_adapts_exits_for_loaded_book():
    sizer = PortfolioSizer(
        {
            "min_position_pct": 0.01,
            "max_coin_exposure_pct": 0.45,
            "max_side_exposure_pct": 0.65,
            "max_cluster_exposure_pct": 0.55,
            "target_volatility_pct": 0.025,
        }
    )
    signal = TradeSignal(
        coin="SOL",
        side=SignalSide.LONG,
        confidence=0.72,
        source=SignalSource.STRATEGY,
        reason="loaded-book test",
        position_pct=0.08,
        leverage=3.0,
        risk=RiskParams(stop_loss_pct=0.01, take_profit_pct=0.02, time_limit_hours=24.0),
    )
    open_positions = [
        {
            "coin": "ETH",
            "side": "long",
            "size": 0.5,
            "entry_price": 3000.0,
            "leverage": 3.0,
        }
    ]
    decision = sizer.adjust_signal(
        signal,
        open_positions=open_positions,
        account_balance=10_000.0,
        regime_data={
            "overall_regime": "volatile",
            "strategy_guidance": {"size_modifier": 0.4},
            "per_coin": {"SOL": {"atr_pct": 0.05}},
        },
        features={"volatility": 0.05},
    )

    assert not decision.blocked
    assert 0.01 <= decision.position_pct < 0.08
    assert decision.stop_loss_pct > 0.01
    assert decision.time_limit_hours <= 12.0
    assert "cluster_exposure_cap" in decision.reasons


def test_portfolio_sizer_blocks_when_same_side_book_is_full():
    sizer = PortfolioSizer({"max_side_exposure_pct": 0.65})
    signal = TradeSignal(
        coin="ETH",
        side=SignalSide.LONG,
        confidence=0.7,
        source=SignalSource.STRATEGY,
        reason="headroom test",
        position_pct=0.05,
        leverage=2.0,
        risk=RiskParams(),
    )
    open_positions = [
        {
            "coin": "BTC",
            "side": "long",
            "size": 2.2,
            "entry_price": 1000.0,
            "leverage": 3.0,
        }
    ]

    decision = sizer.adjust_signal(
        signal,
        open_positions=open_positions,
        account_balance=10_000.0,
        regime_data={"overall_regime": "trending_up"},
    )

    assert decision.blocked is True
    assert "headroom" in decision.block_reason


def test_source_budget_allocator_blocks_blocked_source(monkeypatch):
    monkeypatch.setattr(
        "src.signals.source_allocator.db.get_source_trade_outcome_summary",
        lambda **kwargs: [
            {
                "source_key": "options_flow",
                "source": "options_flow",
                "closed_trades": 6,
                "winning_trades": 1,
                "realized_pnl": -250.0,
                "avg_return_pct": -0.03,
            }
        ],
    )
    monkeypatch.setattr(
        "src.signals.source_allocator.db.get_source_attribution_summary",
        lambda **kwargs: [
            {
                "source_key": "options_flow",
                "source": "options_flow",
                "selected_count": 8,
                "live_rejection_rate": 0.31,
                "live_avg_fill_ratio": 0.52,
            }
        ],
    )

    class FakeAdaptive:
        def get_source_profile(self, source_key="", source=""):
            return {
                "source_key": source_key or source,
                "source": source or "options_flow",
                "status": "blocked",
                "health_score": 0.21,
                "weight_multiplier": 0.25,
                "confidence_multiplier": 0.4,
            }

    allocator = SourceBudgetAllocator({"enabled": True}, adaptive_learning=FakeAdaptive())
    signal = TradeSignal(
        coin="BTC",
        side=SignalSide.SHORT,
        confidence=0.82,
        source=SignalSource.OPTIONS_FLOW,
        reason="blocked source budget test",
        position_pct=0.04,
        leverage=2.0,
        risk=RiskParams(),
    )

    decision = allocator.evaluate(
        signal,
        signal={"source": "options_flow", "source_key": "options_flow", "coin": "BTC"},
        open_positions=[],
        account_balance=10_000.0,
    )

    assert decision.blocked is True
    assert decision.block_reason == "adaptive_learning_blocked"
    assert decision.status == "blocked"


def test_source_budget_allocator_reduces_position_when_headroom_is_limited(monkeypatch):
    monkeypatch.setattr("src.signals.source_allocator.db.get_source_trade_outcome_summary", lambda **kwargs: [])
    monkeypatch.setattr("src.signals.source_allocator.db.get_source_attribution_summary", lambda **kwargs: [])

    class FakeAdaptive:
        def get_source_profile(self, source_key="", source=""):
            return {
                "source_key": source_key or source,
                "source": source or "copy_trade",
                "status": "caution",
                "health_score": 0.45,
                "weight_multiplier": 0.8,
                "confidence_multiplier": 0.85,
            }

    allocator = SourceBudgetAllocator(
        {
            "enabled": True,
            "caution_cap_pct": 0.10,
            "caution_multiplier": 0.5,
        },
        adaptive_learning=FakeAdaptive(),
    )
    signal = TradeSignal(
        coin="ETH",
        side=SignalSide.LONG,
        confidence=0.73,
        source=SignalSource.COPY_TRADE,
        reason="limited source headroom test",
        position_pct=0.05,
        leverage=2.0,
        risk=RiskParams(),
    )

    decision = allocator.apply_to_signal(
        signal,
        signal={"source": "copy_trade", "source_trader": "0xabc123", "coin": "ETH"},
        open_positions=[
            {
                "coin": "BTC",
                "entry_price": 1000.0,
                "size": 0.7,
                "metadata": {"source": "copy_trade", "source_key": "copy_trade:0xabc123", "position_pct": 0.07},
            }
        ],
        account_balance=10_000.0,
    )

    assert decision.blocked is False
    assert 0 < decision.position_pct <= 0.03
    assert signal.position_pct == decision.position_pct
    assert "source_budget_scaled" in decision.reasons


def test_paper_trader_syncs_adjusted_trade_signal_into_execution_payload():
    fake_features = types.ModuleType("src.analysis.features")
    fake_features.FeatureEngine = object
    sys.modules["src.analysis.features"] = fake_features
    fake_kelly = types.ModuleType("src.signals.kelly_sizing")
    fake_kelly.KellySizer = object
    sys.modules["src.signals.kelly_sizing"] = fake_kelly
    fake_trade_memory = types.ModuleType("src.trading.trade_memory")
    fake_trade_memory.TradeMemory = object
    sys.modules["src.trading.trade_memory"] = fake_trade_memory
    import importlib

    sys.modules.pop("src.trading.paper_trader", None)
    PaperTrader = importlib.import_module("src.trading.paper_trader").PaperTrader
    trader = PaperTrader()
    trade_signal = TradeSignal(
        coin="BTC",
        side=SignalSide.LONG,
        confidence=0.8,
        source=SignalSource.STRATEGY,
        reason="sync sizing",
        position_pct=0.02,
        leverage=2.0,
        risk=RiskParams(stop_loss_pct=0.01, take_profit_pct=0.03, time_limit_hours=18.0),
    )
    signal = {
        "coin": "BTC",
        "side": "long",
        "price": 100.0,
        "size": 8.0,
        "leverage": 2.0,
        "stop_loss": 95.0,
        "take_profit": 110.0,
    }

    assert trader._sync_signal_from_trade_signal(
        trade_signal,
        signal,
        account_balance=10_000.0,
    )
    assert abs(signal["size"] - 2.0) < 1e-9
    assert abs(signal["stop_loss"] - 99.0) < 1e-9
    assert abs(signal["take_profit"] - 103.0) < 1e-9
    assert signal["time_limit_hours"] == 18.0


def test_paper_trader_applies_source_budget_metadata():
    fake_features = types.ModuleType("src.analysis.features")
    fake_features.FeatureEngine = object
    sys.modules["src.analysis.features"] = fake_features
    fake_kelly = types.ModuleType("src.signals.kelly_sizing")
    fake_kelly.KellySizer = object
    sys.modules["src.signals.kelly_sizing"] = fake_kelly
    fake_trade_memory = types.ModuleType("src.trading.trade_memory")
    fake_trade_memory.TradeMemory = object
    sys.modules["src.trading.trade_memory"] = fake_trade_memory
    import importlib

    sys.modules.pop("src.trading.paper_trader", None)
    PaperTrader = importlib.import_module("src.trading.paper_trader").PaperTrader

    class FakeSourceAllocator:
        def apply_to_signal(self, trade_signal, **kwargs):
            trade_signal.position_pct = 0.012

            class Decision:
                blocked = False
                status = "caution"
                block_reason = ""

                @staticmethod
                def to_dict():
                    return {
                        "status": "caution",
                        "position_pct": 0.012,
                        "allocation_multiplier": 0.6,
                    }

            return Decision()

    trader = PaperTrader(source_allocator=FakeSourceAllocator())
    trade_signal = TradeSignal(
        coin="BTC",
        side=SignalSide.LONG,
        confidence=0.8,
        source=SignalSource.STRATEGY,
        reason="source budget sync",
        position_pct=0.02,
        leverage=2.0,
        risk=RiskParams(stop_loss_pct=0.01, take_profit_pct=0.03, time_limit_hours=18.0),
    )
    signal = {
        "coin": "BTC",
        "side": "long",
        "price": 100.0,
        "size": 8.0,
        "leverage": 2.0,
        "source": "strategy",
        "source_key": "strategy:momentum:BTC",
    }

    assert trader._apply_portfolio_sizing(
        trade_signal,
        signal,
        open_positions=[],
        account_balance=10_000.0,
        regime_data={"overall_regime": "trending_up"},
    )
    assert abs(signal["size"] - 1.2) < 1e-9
    assert signal["source_budget"]["status"] == "caution"
    assert signal["source_budget_status"] == "caution"


def test_check_open_positions_uses_trade_specific_time_limit(monkeypatch):
    fake_features = types.ModuleType("src.analysis.features")
    fake_features.FeatureEngine = object
    sys.modules["src.analysis.features"] = fake_features
    fake_kelly = types.ModuleType("src.signals.kelly_sizing")
    fake_kelly.KellySizer = object
    sys.modules["src.signals.kelly_sizing"] = fake_kelly
    fake_trade_memory = types.ModuleType("src.trading.trade_memory")
    fake_trade_memory.TradeMemory = object
    sys.modules["src.trading.trade_memory"] = fake_trade_memory
    import importlib

    sys.modules.pop("src.trading.paper_trader", None)
    PaperTrader = importlib.import_module("src.trading.paper_trader").PaperTrader
    monkeypatch.setattr(
        db,
        "get_paper_account",
        lambda: {"balance": 10_000.0, "total_pnl": 0.0, "total_trades": 0, "winning_trades": 0},
    )
    trader = PaperTrader()
    opened_at = (utc_now_naive() - timedelta(hours=8)).isoformat()
    trade = {
        "id": 7,
        "coin": "BTC",
        "side": "long",
        "entry_price": 100.0,
        "size": 1.0,
        "leverage": 2.0,
        "stop_loss": 90.0,
        "take_profit": 120.0,
        "opened_at": opened_at,
        "metadata": json.dumps({"time_limit_hours": 6.0}),
    }
    monkeypatch.setattr(db, "get_open_paper_trades", lambda: [trade])
    monkeypatch.setattr("src.trading.paper_trader.hl.get_all_mids", lambda: {"BTC": 100.0})

    closed_reasons = []

    def fake_close_trade(trade_obj, current_price, close_reason):
        closed_reasons.append(close_reason)
        return {"trade_id": trade_obj["id"], "reason": close_reason}

    monkeypatch.setattr(trader, "_close_trade", fake_close_trade)

    closed = trader.check_open_positions()

    assert closed_reasons == ["time_exit"]
    assert closed[0]["reason"] == "time_exit"


def test_run_trading_cycle_does_not_fall_through_to_standalone_options_flow(monkeypatch):
    class FakeScorer:
        def score_all_strategies(self):
            return []

        def get_top_strategies(self):
            return []

    class FakeOptionsScanner:
        top_convictions = [
            {
                "ticker": "ETH",
                "direction": "bullish",
                "conviction_pct": 82,
                "net_flow": 100000.0,
                "total_prints": 3,
            }
        ]

        def scan_flow(self):
            return {"unusual_prints": 1, "top_convictions": 1}

    class FakePaperTrader:
        def check_open_positions(self):
            return []

    class FakeDecisionEngine:
        def decide(self, strategies, **kwargs):
            return []

    container = type(
        "Container",
        (),
        {
            "live_trader": None,
            "scorer": FakeScorer(),
            "exchange_agg": None,
            "options_scanner": FakeOptionsScanner(),
            "regime_detector": None,
            "regime_strategy_filter": None,
            "signal_processor": None,
            "polymarket": None,
            "predictive_forecaster": None,
            "cross_venue_hedger": None,
            "liquidation_strategy": None,
            "paper_trader": FakePaperTrader(),
            "firewall": None,
            "position_monitor": None,
            "copy_trader": None,
            "arena": None,
            "decision_engine": FakeDecisionEngine(),
            "kelly_sizer": None,
            "multi_scanner": None,
            "agent_scorer": None,
            "calibration": None,
            "shadow_tracker": None,
        },
    )()

    monkeypatch.setattr("src.core.cycles.trading_cycle._run_liquidation_scan", lambda container, regime_data: [])
    monkeypatch.setattr("src.core.cycles.trading_cycle._run_alpha_arena", lambda container, regime_data: [])
    monkeypatch.setattr("src.core.cycles.trading_cycle._run_cross_venue_confirmation", lambda container, strategies: None)
    monkeypatch.setattr("src.core.cycles.trading_cycle.get_execution_open_positions", lambda container: [])
    monkeypatch.setattr("src.core.cycles.trading_cycle._execute_options_flow_trades", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("standalone options flow should not run")))
    monkeypatch.setattr("src.core.cycles.trading_cycle._execute_selected_decisions", lambda container, selected_strategies, regime_data: None)
    monkeypatch.setattr("src.notifications.telegram_bot.is_configured", lambda: False)

    run_trading_cycle(container, 1)


def test_run_trading_cycle_logs_pipeline_and_no_trade_diagnostics(monkeypatch):
    infos = []
    warnings = []

    class FakeScorer:
        def score_all_strategies(self):
            return []

        def get_top_strategies(self):
            return [
                {
                    "name": "base_btc",
                    "strategy_type": "trend_following",
                    "current_score": 0.81,
                    "confidence": 0.72,
                    "parameters": {"coins": ["BTC"]},
                    "metadata": {"source": "strategy"},
                }
            ]

    class FakeRegimeDetector:
        def get_market_regime(self):
            return {"overall_regime": "ranging", "overall_confidence": 0.6}

    class FakeRegimeFilter:
        def filter(self, strategies, regime_data):
            return []

    class FakeOptionsScanner:
        top_convictions = [
            {
                "ticker": "ETH",
                "direction": "bullish",
                "conviction_pct": 70,
                "net_flow": 75000.0,
                "total_prints": 2,
            }
        ]

        def scan_flow(self):
            return {"unusual_prints": 2, "top_convictions": 1}

    class FakePaperTrader:
        def check_open_positions(self):
            return []

    class FakeDecisionEngine:
        def decide(self, strategies, **kwargs):
            return []

        def get_last_cycle_summary(self):
            return {
                "source_counts": {},
                "blocker_counts": {},
                "no_trade_reason": "no candidates",
            }

    container = type(
        "Container",
        (),
        {
            "live_trader": None,
            "scorer": FakeScorer(),
            "exchange_agg": None,
            "options_scanner": FakeOptionsScanner(),
            "regime_detector": FakeRegimeDetector(),
            "regime_strategy_filter": FakeRegimeFilter(),
            "signal_processor": None,
            "polymarket": None,
            "predictive_forecaster": None,
            "cross_venue_hedger": None,
            "liquidation_strategy": None,
            "paper_trader": FakePaperTrader(),
            "firewall": None,
            "position_monitor": None,
            "copy_trader": None,
            "arena": None,
            "decision_engine": FakeDecisionEngine(),
            "kelly_sizer": None,
            "multi_scanner": None,
            "agent_scorer": None,
            "calibration": None,
            "shadow_tracker": None,
        },
    )()

    monkeypatch.setattr("src.core.cycles.trading_cycle._run_multi_exchange_scan", lambda container: ({}, []))
    monkeypatch.setattr("src.core.cycles.trading_cycle._run_liquidation_scan", lambda container, regime_data: [])
    monkeypatch.setattr("src.core.cycles.trading_cycle._gather_copy_trade_signals", lambda container: [])
    monkeypatch.setattr("src.core.cycles.trading_cycle._process_copy_trade_closures", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.core.cycles.trading_cycle._run_alpha_arena", lambda container, regime_data: [])
    monkeypatch.setattr("src.core.cycles.trading_cycle._run_cross_venue_confirmation", lambda container, strategies: None)
    monkeypatch.setattr("src.core.cycles.trading_cycle._apply_agent_scorer_weights", lambda container, strategies: None)
    monkeypatch.setattr("src.core.cycles.trading_cycle._apply_calibration_adjustments", lambda container, strategies: None)
    monkeypatch.setattr("src.core.cycles.trading_cycle._execute_selected_decisions", lambda container, selected_strategies, regime_data: None)
    monkeypatch.setattr("src.core.cycles.trading_cycle._collect_closed_trade_events", lambda container, closed: [])
    monkeypatch.setattr("src.core.cycles.trading_cycle._process_closed_trades", lambda container, closed: None)
    monkeypatch.setattr("src.core.cycles.trading_cycle.get_execution_open_positions", lambda container: [])
    monkeypatch.setattr("src.notifications.telegram_bot.is_configured", lambda: False)
    monkeypatch.setattr("src.core.cycles.trading_cycle.logger.info", lambda msg, *args: infos.append(msg % args if args else msg))
    monkeypatch.setattr("src.core.cycles.trading_cycle.logger.warning", lambda msg, *args: warnings.append(msg % args if args else msg))

    run_trading_cycle(container, 1)

    assert any("Regime filter removed all 1 base strategies" in entry for entry in warnings)
    assert any("Options flow not injected:" in entry for entry in infos)
    assert any("Candidate pipeline empty after upstream filters" in entry for entry in warnings)
    assert any("NoTradeSummary:" in entry for entry in infos)


def test_run_alpha_arena_returns_rankable_candidates(monkeypatch):
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
        },
    )()

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: FakeResponse())

    candidates = _run_alpha_arena(container, {"overall_regime": "neutral"})

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["source"] == "arena_champion"
    assert candidate["source_key"] == "arena_champion:a1"
    assert candidate["side"] == "long"
    assert candidate["parameters"]["coins"] == ["FART"]
    assert candidate["metadata"]["agent_name"] == "alpha"


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


def test_apply_agent_scorer_weights_uses_external_source_keys():
    class FakeAgentScorer:
        def get_weight(self, source_key):
            return {
                "polymarket:BTC": 0.9,
                "options_flow:ETH": 0.8,
                "arena_champion:a1": 0.7,
            }[source_key]

        def get_accuracy(self, source_key):
            return {
                "polymarket:BTC": 0.75,
                "options_flow:ETH": 0.65,
                "arena_champion:a1": 0.60,
            }[source_key]

    container = type("Container", (), {"agent_scorer": FakeAgentScorer()})()
    strategies = [
        {
            "strategy_type": "event_driven",
            "confidence": 0.6,
            "source": "polymarket",
            "parameters": {"coins": ["BTC"]},
        },
        {
            "strategy_type": "options_momentum",
            "confidence": 0.7,
            "source": "options_flow",
            "parameters": {"coins": ["ETH"]},
        },
        {
            "strategy_type": "arena_breakout",
            "confidence": 0.8,
            "source": "arena_champion",
            "agent_id": "a1",
            "parameters": {"coins": ["BTC"]},
            "metadata": {"agent_id": "a1"},
        },
    ]

    _apply_agent_scorer_weights(container, strategies)

    assert strategies[0]["source_key"] == "polymarket:BTC"
    assert strategies[0]["agent_scorer_weight"] == 0.9
    assert strategies[1]["source_key"] == "options_flow:ETH"
    assert strategies[1]["agent_scorer_weight"] == 0.8
    assert strategies[2]["source_key"] == "arena_champion:a1"
    assert strategies[2]["agent_scorer_weight"] == 0.7


def test_paper_trader_generate_signal_preserves_external_source_identity(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", types.SimpleNamespace(std=lambda values: 0.0))
    monkeypatch.setattr(
        "src.trading.paper_trader.db.get_paper_account",
        lambda: {"balance": 10000.0},
    )

    from src.trading.paper_trader import PaperTrader

    trader = PaperTrader()
    signal = trader._generate_signal(
        {
            "id": None,
            "name": "polymarket_btc_long",
            "strategy_type": "event_driven",
            "current_score": 0.72,
            "confidence": 0.72,
            "source": "polymarket",
            "parameters": {"coins": ["BTC"]},
            "metadata": {"source": "polymarket", "reason": "Odds moved sharply higher"},
        },
        {"BTC": 100000.0},
        {"overall_regime": "neutral", "overall_confidence": 0.2},
    )

    assert signal is not None
    assert signal["source"] == "polymarket"
    assert signal["source_key"] == "polymarket:BTC"
    assert signal["reason"] == "Odds moved sharply higher"


def test_process_closed_trades_routes_arena_outcomes_to_specific_agent():
    class FakeArena:
        def __init__(self):
            self.agent_calls = []
            self.strategy_calls = []

        def record_trade_result(self, agent_id, pnl, return_pct=0.0):
            self.agent_calls.append((agent_id, pnl, return_pct))

        def record_trade_for_strategy(self, *args):
            self.strategy_calls.append(args)

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
                "trade_id": 42,
                "coin": "BTC",
                "side": "long",
                "entry_price": 100000.0,
                "exit_price": 102000.0,
                "size": 0.01,
                "leverage": 2,
                "pnl": 40.0,
                "strategy_type": "arena_breakout",
                "signal_id": "arena-42",
                "metadata": {
                    "source": "arena_champion",
                    "source_key": "arena_champion:a1",
                    "agent_id": "a1",
                    "signal_id": "arena-42",
                },
            }
        ],
    )

    assert container.arena.agent_calls
    assert container.arena.agent_calls[0][0] == "a1"
    assert container.arena.strategy_calls == []
    assert container.kelly_sizer.calls[0]["strategy_key"] == "arena_champion:a1"
    assert container.agent_scorer.calls[0][0] == "arena_champion:a1"


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


def test_close_position_still_allows_reduce_only_exit_when_kill_switch_is_active(monkeypatch):
    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    class FakeManager:
        def post(self, payload, **kwargs):
            return {}

    captured = []

    monkeypatch.setattr("src.trading.live_trader.get_manager", lambda: FakeManager())
    monkeypatch.setattr(
        LiveTrader,
        "get_positions",
        lambda self: [{"coin": "BTC", "size": 0.25, "szi": 0.25}],
    )
    monkeypatch.setattr(
        LiveTrader,
        "place_market_order",
        lambda self, coin, side, size, reduce_only=False: captured.append(
            {
                "coin": coin,
                "side": side,
                "size": size,
                "reduce_only": reduce_only,
            }
        ) or {"status": "success", "coin": coin, "reduce_only": reduce_only},
    )

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False)
    trader.kill_switch_active = True

    result = trader.close_position("BTC")

    assert result["status"] == "success"
    assert captured == [
        {"coin": "BTC", "side": "sell", "size": 0.25, "reduce_only": True}
    ]


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


def test_execution_policy_manager_prefers_maker_for_low_urgency_source(monkeypatch):
    monkeypatch.setattr(
        "src.signals.execution_policy.db.get_execution_quality_summary",
        lambda **kwargs: {
            "total_events": 12,
            "avg_realized_slippage_bps": 2.2,
            "maker_ratio": 0.64,
            "avg_fill_ratio": 0.92,
            "rejection_rate": 0.03,
        },
    )

    manager = ExecutionPolicyManager({"enabled": True, "min_events": 3})
    recommendation = manager.recommend(
        strategy={
            "source": "strategy",
            "source_key": "strategy:momentum:ETH",
        },
        metadata={"expected_slippage_bps": 3.0},
        confidence=0.58,
        source_quality=0.71,
    )

    assert recommendation["execution_route"] == "maker_limit"
    assert recommendation["execution_role"] == "maker"
    assert recommendation["limit_tif"] == "Alo"
    assert recommendation["fallback_to_market"] is True
    assert recommendation["expected_slippage_bps"] < 3.0


def test_decision_engine_applies_execution_policy_metadata():
    class FakeExecutionPolicy:
        def recommend(self, **kwargs):
            return {
                "route": "maker_limit",
                "execution_route": "maker_limit",
                "execution_role": "maker",
                "expected_slippage_bps": 1.1,
                "maker_price_offset_bps": 1.8,
                "maker_timeout_seconds": 4.0,
                "fallback_to_market": True,
                "limit_tif": "Alo",
                "policy_reason": "maker_cost_saving",
                "urgency_score": 0.41,
            }

        def get_stats(self):
            return {"enabled": True}

    engine = DecisionEngine(
        {
            "execution_quality_enabled": False,
            "adaptive_learning_enabled": False,
            "execution_policy_enabled": True,
            "execution_policy": FakeExecutionPolicy(),
        }
    )

    strategy = {
        "name": "momentum_eth",
        "strategy_type": "momentum_long",
        "current_score": 0.71,
        "confidence": 0.69,
        "parameters": {"coins": ["ETH"], "direction": "long"},
        "metadata": {"source": "strategy", "source_key": "strategy:momentum_long:ETH"},
        "source": "strategy",
        "source_key": "strategy:momentum_long:ETH",
    }

    breakdown = engine._compute_composite_score(
        strategy,
        regime_data={"overall_regime": "trending_up", "overall_confidence": 0.8},
        open_coins=set(),
        kelly_stats={},
    )

    assert strategy["metadata"]["execution_route"] == "maker_limit"
    assert strategy["metadata"]["execution_role"] == "maker"
    assert strategy["_execution_policy"]["policy_reason"] == "maker_cost_saving"
    assert breakdown["execution_route"] == "maker_limit"
    assert breakdown["execution_policy_reason"] == "maker_cost_saving"
    assert breakdown["maker_price_offset_bps"] == 1.8


def test_execute_signal_uses_maker_limit_route_and_falls_back_to_market(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    limit_calls = []
    market_calls = []
    verify_calls = []
    trigger_calls = []

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(LiveTrader, "_compute_passive_limit_price", lambda self, coin, side, offset_bps: 1998.0)
    monkeypatch.setattr(
        LiveTrader,
        "place_limit_order",
        lambda self, coin, side, size, price, leverage=1, reduce_only=False, tif=None: (
            limit_calls.append((coin, side, size, price, tif))
            or {"status": "ok", "submitted_size": size, "mid_price": 2000.0, "wire_price": "1998"}
        ),
    )
    monkeypatch.setattr(
        LiveTrader,
        "place_market_order",
        lambda self, coin, side, size, leverage=1, reduce_only=False: (
            market_calls.append((coin, side, size, reduce_only))
            or {"status": "success", "submitted_size": size, "mid_price": 2000.0, "wire_price": "2001"}
        ),
    )
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    verify_results = iter(
        [
            None,
            {"status": "verified", "size": 0.1, "partial_fill": False},
        ]
    )
    monkeypatch.setattr(
        LiveTrader,
        "verify_fill",
        lambda self, coin, side, expected_size, timeout=10.0, poll_interval=1.0: (
            verify_calls.append((coin, side, expected_size, timeout))
            or next(verify_results)
        ),
    )
    monkeypatch.setattr(
        LiveTrader,
        "_cancel_open_entry_orders",
        lambda self, coin: {"attempted": 1, "cancelled": 1, "failed": 0, "remaining": []},
    )
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(
        LiveTrader,
        "place_trigger_order",
        lambda self, coin, side, size, trigger_price, tp_or_sl="sl": (
            trigger_calls.append((coin, side, size, tp_or_sl)) or {"status": "success"}
        ),
    )

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
    result = trader.execute_signal(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.67,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.1,
            "strategy_type": "momentum_long",
            "metadata": {
                "execution_route": "maker_limit",
                "execution_role": "maker",
                "maker_price_offset_bps": 1.5,
                "maker_timeout_seconds": 4.0,
                "fallback_to_market": True,
                "limit_tif": "Alo",
                "policy_reason": "maker_cost_saving",
            },
        }
    )

    assert result is not None
    assert result["status"] == "success"
    assert result["execution_route"] == "maker_limit_fallback_market"
    assert limit_calls == [("ETH", "buy", 0.1, 1998.0, "Alo")]
    assert market_calls == [("ETH", "buy", 0.1, False)]
    assert verify_calls[0] == ("ETH", "buy", 0.1, 4.0)
    assert verify_calls[1][0:3] == ("ETH", "buy", 0.1)
    assert [call[1] for call in trigger_calls] == ["sell", "sell"]


def test_mirror_executed_trades_to_live_preserves_execution_metadata(monkeypatch):
    captured = []

    class FakeTrader:
        def is_live_enabled(self):
            return True

        def is_deployable(self):
            return True

        def execute_signal(self, payload, bypass_firewall=False):
            captured.append((payload, bypass_firewall))
            return {"status": "success"}

    container = types.SimpleNamespace(live_trader=FakeTrader())
    trade = {
        "coin": "ETH",
        "side": "long",
        "confidence": 0.66,
        "size": 0.1,
        "entry_price": 2000.0,
        "metadata": {
            "execution_role": "maker",
            "upstream_metadata": {
                "execution_route": "maker_limit",
                "maker_price_offset_bps": 1.5,
                "policy_reason": "maker_cost_saving",
            },
        },
    }

    monkeypatch.setattr("src.core.live_execution._rescale_size_for_live", lambda trade, trader: trade)

    mirror_executed_trades_to_live(
        container,
        [trade],
        success_label="LIVE",
        skip_label="SKIP",
    )

    assert len(captured) == 1
    payload, bypass_firewall = captured[0]
    assert bypass_firewall is True
    assert isinstance(payload, dict)
    assert payload["metadata"]["upstream_metadata"]["execution_route"] == "maker_limit"


def test_dashboard_divergence_control_metrics_include_runtime(monkeypatch):
    monkeypatch.setattr(
        dashboard_ui.db,
        "get_runtime_divergence_summary",
        lambda lookback_hours=24: {
            "paper_live_open_gap_ratio": 0.42,
            "shadow_live_execution_gap_ratio": 0.18,
        },
    )

    class FakeController:
        def get_dashboard_payload(self, limit=12):
            return {
                "global": {"status": "caution", "reasons": ["global_open_gap_caution"]},
                "profiles": [{"scope_key": "options_flow:BTC", "status": "blocked"}],
            }

    metrics = dashboard_ui._build_divergence_control_metrics(FakeController())

    assert metrics["summary"]["paper_live_open_gap_ratio"] == 0.42
    assert metrics["runtime"]["global"]["status"] == "caution"
    assert metrics["runtime"]["profiles"][0]["scope_key"] == "options_flow:BTC"


def test_divergence_controller_blocks_on_large_runtime_gap(monkeypatch):
    monkeypatch.setattr(
        "src.signals.divergence_controller.db.get_runtime_divergence_summary",
        lambda lookback_hours=24: {
            "shadow_selected_count": 10,
            "paper_open_count": 6,
            "live_open_positions": 1,
            "live_execution_total": 2,
            "paper_live_open_gap_ratio": 0.83,
            "shadow_live_execution_gap_ratio": 0.8,
            "paper_live_realized_pnl_gap_ratio": 0.55,
            "live_rejection_rate": 0.31,
        },
    )
    monkeypatch.setattr(
        "src.signals.divergence_controller.db.get_source_attribution_summary",
        lambda **kwargs: [
            {
                "source_key": "options_flow:BTC",
                "source": "options_flow",
                "selected_count": 5,
                "live_events": 1,
                "paper_closed_count": 4,
                "live_rejection_rate": 0.12,
                "live_avg_fill_ratio": 0.82,
            }
        ],
    )

    controller = DivergenceController(
        {
            "enabled": True,
            "live_trading_enabled": True,
            "refresh_interval_seconds": 0.0,
            "min_live_events": 2,
            "source_min_selected": 2,
        }
    )
    assessment = controller.evaluate(source_key="options_flow:BTC", source="options_flow")

    assert assessment["status"] == "blocked"
    assert assessment["blocked"] is True
    assert any(
        reason in assessment["reasons"]
        for reason in (
            "global_open_gap_block",
            "global_execution_gap_block",
            "global_realized_pnl_gap_block",
            "global_live_rejection_block",
        )
    )


def test_divergence_controller_warms_up_when_only_shadow_history_exists(monkeypatch):
    monkeypatch.setattr(
        "src.signals.divergence_controller.db.get_runtime_divergence_summary",
        lambda lookback_hours=24: {
            "shadow_selected_count": 12,
            "paper_open_count": 0,
            "paper_recent_open_count": 0,
            "paper_recent_closed_count": 0,
            "live_open_positions": 6,
            "live_execution_total": 8,
            "paper_live_open_gap_ratio": 1.0,
            "shadow_live_execution_gap_ratio": 0.33,
            "paper_live_realized_pnl_gap_ratio": 0.88,
            "live_rejection_rate": 0.02,
        },
    )
    monkeypatch.setattr(
        "src.signals.divergence_controller.db.get_source_attribution_summary",
        lambda **kwargs: [],
    )

    controller = DivergenceController(
        {
            "enabled": True,
            "live_trading_enabled": True,
            "refresh_interval_seconds": 0.0,
            "min_live_events": 3,
            "source_min_selected": 2,
        }
    )
    assessment = controller.evaluate(source_key="options_flow:BTC", source="options_flow")

    assert assessment["status"] == "warming_up"
    assert assessment["blocked"] is False
    assert "insufficient_paper_history" in assessment["reasons"]
    assert "global_open_gap_block" not in assessment["reasons"]


def test_decision_engine_blocks_candidate_when_divergence_controller_blocked():
    class FakeController:
        def evaluate(self, source_key="", source="", strategy=None):
            return {
                "status": "blocked",
                "blocked": True,
                "multiplier": 0.0,
                "divergence_score": 0.0,
                "reasons": ["global_execution_gap_block"],
                "global": {"status": "blocked", "reasons": ["global_execution_gap_block"]},
                "source_profile": {"status": "healthy", "reasons": []},
            }

    engine = DecisionEngine(
        {
            "w_divergence": 0.60,
            "min_decision_score": 0.0,
            "min_signal_confidence": 0.4,
            "min_source_weight": 0.2,
            "min_expected_value_pct": -1.0,
            "execution_quality_enabled": False,
            "adaptive_learning_enabled": False,
            "context_performance_enabled": False,
            "confluence_enabled": False,
            "calibration_enabled": False,
            "memory_enabled": False,
            "divergence_controller": FakeController(),
            "divergence_enabled": True,
            "divergence_block_on_status": True,
        }
    )

    candidate = {
        "name": "divergence_blocked",
        "source": "options_flow",
        "source_key": "options_flow:BTC",
        "strategy_type": "options_momentum",
        "current_score": 0.82,
        "confidence": 0.79,
        "source_accuracy": 0.71,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 108.0,
        "parameters": {"coins": ["BTC"], "direction": "long"},
        "metadata": {"source": "options_flow", "source_key": "options_flow:BTC"},
    }

    selected = engine.decide(
        [candidate],
        regime_data={"overall_regime": "trending_up"},
        open_positions=[],
        kelly_stats={},
    )

    assert selected == []
    assert candidate["_divergence_control"]["status"] == "blocked"
    assert "divergence_guard" in engine._decision_blockers(candidate)
    assert engine.get_stats()["total_divergence_blocks"] >= 1


def test_source_budget_allocator_scales_position_when_divergence_is_caution(monkeypatch):
    monkeypatch.setattr("src.signals.source_allocator.db.get_source_trade_outcome_summary", lambda **kwargs: [])
    monkeypatch.setattr("src.signals.source_allocator.db.get_source_attribution_summary", lambda **kwargs: [])

    class FakeController:
        def evaluate(self, source_key="", source=""):
            return {
                "status": "caution",
                "blocked": False,
                "multiplier": 0.5,
                "divergence_score": 0.35,
                "reasons": ["global_execution_gap_caution"],
                "global": {"status": "caution"},
                "source_profile": {"status": "healthy"},
            }

        def get_dashboard_payload(self, limit=12):
            return {"global": {"status": "caution"}}

    allocator = SourceBudgetAllocator(
        {"enabled": True, "divergence_enabled": True},
        adaptive_learning=None,
        divergence_controller=FakeController(),
    )
    signal = TradeSignal(
        coin="BTC",
        side=SignalSide.LONG,
        confidence=0.8,
        source=SignalSource.OPTIONS_FLOW,
        reason="divergence caution scale test",
        position_pct=0.10,
        leverage=2.0,
        risk=RiskParams(),
    )

    decision = allocator.evaluate(
        signal,
        signal={"source": "options_flow", "source_key": "options_flow:BTC", "coin": "BTC"},
        open_positions=[],
        account_balance=10_000.0,
    )

    assert decision.blocked is False
    assert decision.position_pct < decision.original_position_pct
    assert decision.divergence_status == "caution"
    assert decision.divergence_multiplier == 0.5


def test_dashboard_capital_governor_metrics_include_runtime(monkeypatch):
    monkeypatch.setattr(
        dashboard_ui.db,
        "get_capital_governor_summary",
        lambda lookback_hours=24.0 * 14, health_limit=200: {
            "paper_current_drawdown_pct": 0.06,
            "live_current_drawdown_pct": 0.02,
        },
    )

    class FakeGovernor:
        def get_dashboard_payload(self):
            return {
                "runtime": {
                    "status": "caution",
                    "reasons": ["paper_drawdown_caution"],
                    "operator_risk_off_enabled": True,
                    "operator_risk_off_reason": "manual review",
                    "operator_risk_off_set_by": "ops",
                    "operator_risk_off_set_at": "2026-04-07T12:00:00+00:00",
                },
                "summary": {"paper_current_drawdown_pct": 0.06},
            }

    metrics = dashboard_ui._build_capital_governor_metrics(FakeGovernor())

    assert metrics["summary"]["paper_current_drawdown_pct"] == 0.06
    assert metrics["runtime"]["runtime"]["status"] == "caution"
    assert metrics["operator_override"]["enabled"] is True
    assert metrics["operator_override"]["reason"] == "manual review"


def test_runtime_incident_report_collects_blocking_states():
    class FakeLiveTrader:
        def get_stats(self):
            return {
                "live_enabled": True,
                "kill_switch_active": True,
                "status_reason": "kill_switch_active",
                "preflight": {
                    "deployable": False,
                    "status": "blocked",
                    "blocking_checks": ["account_reachability"],
                    "warning_checks": [],
                },
                "activation_guard": {
                    "deployable": False,
                    "status": "expired",
                    "approved_by": "ops",
                    "approved_at": "2026-04-07T00:00:00+00:00",
                    "expires_at": "2026-04-08T00:00:00+00:00",
                    "blocking_checks": ["approval_stale"],
                    "warning_checks": [],
                },
                "live_readiness": {
                    "status": "blocked",
                    "status_reason": "preflight:account_reachability",
                },
            }

    class FakeDivergence:
        def get_dashboard_payload(self, limit=8):
            return {
                "tracked_sources": 4,
                "global": {
                    "status": "blocked",
                    "reasons": ["global_execution_gap_block"],
                },
                "profiles": [{"scope_key": "options_flow:BTC", "status": "blocked"}],
            }

    class FakeGovernor:
        def get_dashboard_payload(self):
            return {
                "runtime": {
                    "status": "blocked",
                    "reasons": ["divergence_runtime_block"],
                    "metrics": {"live_current_drawdown_pct": 0.07},
                }
            }

    class FakeAdaptive:
        def get_dashboard_payload(self, limit=10):
            return {
                "recalibration": {
                    "run_id": "recal-22",
                    "demoted_count": 2,
                    "promoted_count": 1,
                    "transition_count": 3,
                }
            }

    report = build_runtime_incident_report(
        live_trader=FakeLiveTrader(),
        divergence_controller=FakeDivergence(),
        capital_governor=FakeGovernor(),
        adaptive_learning=FakeAdaptive(),
        max_items=12,
    )

    keys = {item["key"] for item in report["incidents"]}

    assert report["summary"]["critical_count"] >= 4
    assert report["summary"]["blocking_count"] >= 4
    assert "live_kill_switch_active" in keys
    assert "live_preflight_blocked" in keys
    assert "live_activation_blocked" in keys
    assert "divergence_runtime_blocked" in keys
    assert "capital_governor_blocked" in keys
    assert "adaptive_source_demotions:recal-22" in keys


def test_runtime_incident_report_labels_disabled_activation_guard_clearly():
    class FakeLiveTrader:
        def get_stats(self):
            return {
                "live_requested": True,
                "live_enabled": True,
                "kill_switch_active": False,
                "preflight": {
                    "deployable": True,
                    "status": "ready",
                    "blocking_checks": [],
                    "warning_checks": [],
                },
                "activation_guard": {
                    "deployable": True,
                    "status": "disabled",
                    "approved_by": "",
                    "approved_at": None,
                    "expires_at": None,
                    "blocking_checks": [],
                    "warning_checks": ["activation_guard_enabled"],
                },
                "live_readiness": {"status": "ready", "status_reason": "ready"},
            }

    report = build_runtime_incident_report(live_trader=FakeLiveTrader(), max_items=8)
    keys = {item["key"] for item in report["incidents"]}

    assert "live_activation_guard_disabled" in keys
    assert "live_activation_warning" not in keys


def test_dashboard_runtime_incident_metrics_include_summary():
    class FakeLiveTrader:
        def get_stats(self):
            return {
                "live_enabled": True,
                "kill_switch_active": False,
                "preflight": {
                    "deployable": True,
                    "status": "ready",
                    "blocking_checks": [],
                    "warning_checks": [],
                },
                "activation_guard": {
                    "deployable": True,
                    "status": "ready",
                    "approved_by": "ops",
                    "expires_at": "2026-04-08T00:00:00+00:00",
                    "hours_remaining": 2.5,
                    "blocking_checks": [],
                    "warning_checks": ["approval_expiring_soon"],
                },
                "live_readiness": {"status": "ready", "status_reason": "ready"},
            }

    class FakeDivergence:
        def get_dashboard_payload(self, limit=8):
            return {
                "tracked_sources": 3,
                "global": {"status": "caution", "reasons": ["global_open_gap_caution"]},
                "profiles": [{"scope_key": "polymarket:BTC", "status": "caution"}],
            }

    class FakeGovernor:
        def get_dashboard_payload(self):
            return {
                "runtime": {
                    "status": "caution",
                    "reasons": ["paper_drawdown_caution"],
                    "metrics": {"paper_current_drawdown_pct": 0.03},
                }
            }

    class FakeAdaptive:
        def get_dashboard_payload(self, limit=10):
            return {
                "recalibration": {
                    "run_id": "recal-23",
                    "demoted_count": 1,
                    "promoted_count": 0,
                    "transition_count": 1,
                }
            }

    metrics = dashboard_ui._build_runtime_incident_metrics(
        live_trader=FakeLiveTrader(),
        divergence_controller=FakeDivergence(),
        capital_governor=FakeGovernor(),
        adaptive_learning=FakeAdaptive(),
    )

    assert metrics["summary"]["overall_status"] == "warning"
    assert metrics["summary"]["total_incidents"] == 4
    assert metrics["incidents"][0]["severity"] in {"critical", "warning"}
    assert any(item["key"] == "live_activation_warning" for item in metrics["incidents"])


def test_runtime_incident_telegram_alerts_are_deduplicated(monkeypatch):
    sent_messages = []

    monkeypatch.setattr(config, "RUNTIME_INCIDENT_TELEGRAM_ENABLED", True)
    monkeypatch.setattr(config, "RUNTIME_INCIDENT_TELEGRAM_MAX_ALERTS", 3)
    monkeypatch.setattr(config, "RUNTIME_INCIDENT_TELEGRAM_COOLDOWN_MINUTES", 60.0)
    monkeypatch.setattr(telegram_alerts, "_RUNTIME_INCIDENT_ALERT_CACHE", {})
    monkeypatch.setattr(telegram_alerts, "is_configured", lambda: True)
    monkeypatch.setattr(
        telegram_alerts,
        "_send_message",
        lambda text, parse_mode="HTML", disable_preview=True: sent_messages.append(text) or True,
    )

    incident = {
        "key": "capital_governor_blocked",
        "severity": "critical",
        "source": "capital_governor",
        "status": "blocked",
        "blocking": True,
        "title": "Capital governor blocked entries",
        "summary": "Global capital posture is risk-off.",
    }

    assert telegram_alerts.send_runtime_incident_alerts([incident]) is True
    assert telegram_alerts.send_runtime_incident_alerts([incident]) is False
    assert len(sent_messages) == 1


def test_capital_governor_enters_risk_off_on_live_drawdown(monkeypatch):
    monkeypatch.setattr(
        "src.signals.capital_governor.db.get_capital_governor_summary",
        lambda lookback_hours=24.0 * 14: {
            "paper_closed_trades": 8,
            "paper_current_drawdown_pct": 0.06,
            "paper_sharpe": 0.12,
            "live_snapshot_count": 16,
            "live_current_drawdown_pct": 0.11,
            "live_sharpe": -0.31,
            "source_profile_count": 5,
            "degraded_source_ratio": 0.20,
            "blocked_source_ratio": 0.10,
        },
    )

    class FakeDivergence:
        def get_stats(self):
            return {"global_status": "healthy"}

    governor = CapitalGovernor(
        {
            "enabled": True,
            "refresh_interval_seconds": 0.0,
            "min_paper_trades": 3,
            "min_live_snapshots": 3,
            "min_source_profiles": 2,
        },
        divergence_controller=FakeDivergence(),
    )
    assessment = governor.evaluate(
        regime_data={"overall_regime": "choppy", "overall_confidence": 0.41}
    )

    assert assessment["status"] == "risk_off"
    assert assessment["blocked"] is False
    assert assessment["multiplier"] == 0.4
    assert "live_drawdown_risk_off" in assessment["reasons"]


def test_capital_governor_blocks_when_operator_risk_off_enabled(monkeypatch):
    monkeypatch.setenv("OPERATOR_RISK_OFF_ENABLED", "true")
    monkeypatch.setenv("OPERATOR_RISK_OFF_REASON", "manual review")
    monkeypatch.setenv("OPERATOR_RISK_OFF_SET_BY", "ops")
    monkeypatch.setenv("OPERATOR_RISK_OFF_SET_AT", "2026-04-07T12:00:00+00:00")
    monkeypatch.setattr(
        "src.signals.capital_governor.db.get_capital_governor_summary",
        lambda lookback_hours=24.0 * 14: {
            "paper_closed_trades": 10,
            "paper_current_drawdown_pct": 0.01,
            "paper_sharpe": 0.55,
            "live_snapshot_count": 20,
            "live_current_drawdown_pct": 0.01,
            "live_sharpe": 0.40,
            "source_profile_count": 5,
            "degraded_source_ratio": 0.05,
            "blocked_source_ratio": 0.0,
        },
    )

    governor = CapitalGovernor(
        {
            "enabled": True,
            "refresh_interval_seconds": 0.0,
            "min_paper_trades": 3,
            "min_live_snapshots": 3,
            "min_source_profiles": 1,
            "operator_risk_off_blocks": True,
        }
    )

    assessment = governor.evaluate()

    assert assessment["status"] == "blocked"
    assert assessment["blocked"] is True
    assert "operator_risk_off" in assessment["reasons"]
    assert assessment["operator_risk_off_enabled"] is True
    assert assessment["operator_risk_off_reason"] == "manual review"


def test_decision_engine_blocks_candidate_when_capital_governor_blocked():
    class FakeGovernor:
        def evaluate(self, regime_data=None):
            return {
                "status": "blocked",
                "blocked": True,
                "multiplier": 0.0,
                "capital_score": 0.0,
                "reasons": ["live_drawdown_block"],
                "metrics": {"live_current_drawdown_pct": 0.22},
                "divergence_status": "healthy",
            }

        def get_stats(self):
            return {"global_status": "blocked"}

    engine = DecisionEngine(
        {
            "w_capital_governor": 0.60,
            "min_decision_score": 0.0,
            "min_signal_confidence": 0.4,
            "min_source_weight": 0.2,
            "min_expected_value_pct": -1.0,
            "execution_quality_enabled": False,
            "adaptive_learning_enabled": False,
            "context_performance_enabled": False,
            "confluence_enabled": False,
            "calibration_enabled": False,
            "memory_enabled": False,
            "divergence_enabled": False,
            "capital_governor": FakeGovernor(),
            "capital_governor_enabled": True,
            "capital_governor_block_on_status": True,
        }
    )

    candidate = {
        "name": "capital_blocked",
        "source": "strategy",
        "source_key": "strategy:trend_following:BTC",
        "strategy_type": "trend_following",
        "current_score": 0.84,
        "confidence": 0.8,
        "source_accuracy": 0.74,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 108.0,
        "parameters": {"coins": ["BTC"], "direction": "long"},
        "metadata": {"source": "strategy", "source_key": "strategy:trend_following:BTC"},
    }

    selected = engine.decide(
        [candidate],
        regime_data={"overall_regime": "trending_up", "overall_confidence": 0.8},
        open_positions=[],
        kelly_stats={},
    )

    assert selected == []
    assert candidate["_capital_governor"]["status"] == "blocked"
    assert "capital_governor_guard" in engine._decision_blockers(candidate)
    assert engine.get_stats()["total_capital_blocks"] >= 1


def test_decision_engine_records_no_trade_summary_with_blockers(monkeypatch):
    logs = []
    monkeypatch.setattr(
        "src.signals.decision_engine.logger.info",
        lambda msg, *args: logs.append(msg % args if args else msg),
    )

    engine = DecisionEngine(
        {
            "min_decision_score": 0.0,
            "min_signal_confidence": 0.90,
            "min_source_weight": 0.2,
            "min_expected_value_pct": -1.0,
            "execution_quality_enabled": False,
            "adaptive_learning_enabled": False,
            "context_performance_enabled": False,
            "confluence_enabled": False,
            "calibration_enabled": False,
            "memory_enabled": False,
            "divergence_enabled": False,
            "capital_governor_enabled": False,
        }
    )

    candidate = {
        "name": "confidence_blocked",
        "source": "strategy",
        "source_key": "strategy:trend_following:BTC",
        "strategy_type": "trend_following",
        "current_score": 0.84,
        "confidence": 0.55,
        "source_accuracy": 0.74,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 108.0,
        "parameters": {"coins": ["BTC"], "direction": "long"},
        "metadata": {"source": "strategy", "source_key": "strategy:trend_following:BTC"},
    }

    selected = engine.decide(
        [candidate],
        regime_data={"overall_regime": "trending_up", "overall_confidence": 0.8},
        open_positions=[],
        kelly_stats={},
    )

    summary = engine.get_last_cycle_summary()

    assert selected == []
    assert summary["no_trade_reason"] == "blocked"
    assert summary["source_counts"]["strategy"] == 1
    assert summary["blocker_counts"]["confidence<0.90"] == 1
    assert any("WhyNoTrade:" in entry for entry in logs)


def test_decision_engine_blocks_candidate_when_operator_risk_off_is_active():
    class FakeGovernor:
        def evaluate(self, regime_data=None):
            return {
                "status": "blocked",
                "blocked": True,
                "multiplier": 0.0,
                "capital_score": 0.0,
                "reasons": ["operator_risk_off"],
                "metrics": {},
                "divergence_status": "healthy",
                "operator_risk_off_enabled": True,
                "operator_risk_off_reason": "manual review",
            }

        def get_stats(self):
            return {"global_status": "blocked"}

    engine = DecisionEngine(
        {
            "w_capital_governor": 0.60,
            "min_decision_score": 0.0,
            "min_signal_confidence": 0.4,
            "min_source_weight": 0.2,
            "min_expected_value_pct": -1.0,
            "execution_quality_enabled": False,
            "adaptive_learning_enabled": False,
            "context_performance_enabled": False,
            "confluence_enabled": False,
            "calibration_enabled": False,
            "memory_enabled": False,
            "divergence_enabled": False,
            "capital_governor": FakeGovernor(),
            "capital_governor_enabled": True,
            "capital_governor_block_on_status": True,
        }
    )

    candidate = {
        "name": "operator_blocked",
        "source": "strategy",
        "source_key": "strategy:trend_following:BTC",
        "strategy_type": "trend_following",
        "current_score": 0.84,
        "confidence": 0.8,
        "source_accuracy": 0.74,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 108.0,
        "parameters": {"coins": ["BTC"], "direction": "long"},
        "metadata": {"source": "strategy", "source_key": "strategy:trend_following:BTC"},
    }

    selected = engine.decide(
        [candidate],
        regime_data={"overall_regime": "trending_up", "overall_confidence": 0.8},
        open_positions=[],
        kelly_stats={},
    )

    assert selected == []
    assert "operator_risk_off" in engine._decision_blockers(candidate)
    assert engine.get_stats()["total_operator_blocks"] >= 1


def test_source_budget_allocator_scales_position_when_capital_governor_is_risk_off(monkeypatch):
    monkeypatch.setattr("src.signals.source_allocator.db.get_source_trade_outcome_summary", lambda **kwargs: [])
    monkeypatch.setattr("src.signals.source_allocator.db.get_source_attribution_summary", lambda **kwargs: [])

    class FakeGovernor:
        def evaluate(self):
            return {
                "status": "risk_off",
                "blocked": False,
                "multiplier": 0.4,
                "capital_score": 0.25,
                "reasons": ["live_drawdown_risk_off"],
            }

        def get_dashboard_payload(self):
            return {"runtime": {"status": "risk_off"}}

    allocator = SourceBudgetAllocator(
        {"enabled": True, "capital_governor_enabled": True},
        adaptive_learning=None,
        capital_governor=FakeGovernor(),
    )
    signal = TradeSignal(
        coin="ETH",
        side=SignalSide.SHORT,
        confidence=0.77,
        source=SignalSource.OPTIONS_FLOW,
        reason="capital risk off test",
        position_pct=0.10,
        leverage=2.0,
        risk=RiskParams(),
    )

    decision = allocator.evaluate(
        signal,
        signal={"source": "options_flow", "source_key": "options_flow:ETH", "coin": "ETH"},
        open_positions=[],
        account_balance=10_000.0,
    )

    assert decision.blocked is False
    assert decision.position_pct < decision.original_position_pct
    assert decision.capital_status == "risk_off"
    assert decision.capital_multiplier == 0.4


def test_source_budget_allocator_respects_promotion_ladder_caps(monkeypatch):
    monkeypatch.setattr("src.signals.source_allocator.db.get_source_trade_outcome_summary", lambda **kwargs: [])
    monkeypatch.setattr("src.signals.source_allocator.db.get_source_attribution_summary", lambda **kwargs: [])

    class FakeAdaptive:
        def get_source_profile(self, source_key="", source=""):
            return {
                "status": "active",
                "health_score": 0.66,
                "weight_multiplier": 0.98,
                "confidence_multiplier": 0.97,
                "promotion_stage": "incubating",
                "promotion_multiplier": 0.55,
                "promotion_cap_pct": 0.08,
                "promotion_reasons": ["insufficient_closed_trades"],
                "metadata": {
                    "promotion_stage": "incubating",
                    "promotion_multiplier": 0.55,
                    "promotion_cap_pct": 0.08,
                    "promotion_reasons": ["insufficient_closed_trades"],
                },
            }

    allocator = SourceBudgetAllocator(
        {"enabled": True, "promotion_ladder_enabled": True},
        adaptive_learning=FakeAdaptive(),
    )
    signal = TradeSignal(
        coin="BTC",
        side=SignalSide.LONG,
        confidence=0.79,
        source=SignalSource.OPTIONS_FLOW,
        reason="promotion ladder cap test",
        position_pct=0.20,
        leverage=2.0,
        risk=RiskParams(),
    )

    decision = allocator.evaluate(
        signal,
        signal={"source": "options_flow", "source_key": "options_flow:BTC", "coin": "BTC"},
        open_positions=[],
        account_balance=10_000.0,
    )

    assert decision.blocked is False
    assert decision.promotion_stage == "incubating"
    assert decision.promotion_cap_pct == 0.08
    assert decision.position_pct <= 0.08
    assert "promotion_ladder_scaled" in decision.reasons


def test_arena_capital_allocator_weights_by_status():
    allocator = CapitalAllocator(total_pool=90_000.0)
    champion = ArenaAgent(
        agent_id="champion",
        name="Champion",
        strategy_type="momentum_long",
        status=AgentStatus.CHAMPION,
        total_trades=20,
    )
    active = ArenaAgent(
        agent_id="active",
        name="Active",
        strategy_type="trend_following",
        status=AgentStatus.ACTIVE,
        total_trades=12,
    )
    incubating = ArenaAgent(
        agent_id="incubating",
        name="Incubating",
        strategy_type="mean_reversion",
        status=AgentStatus.ACTIVE,
        total_trades=1,
    )
    probation = ArenaAgent(
        agent_id="probation",
        name="Probation",
        strategy_type="contrarian",
        status=AgentStatus.PROBATION,
        total_trades=14,
    )

    allocations = allocator.reallocate([champion, active, incubating, probation])

    assert round(sum(allocations.values()), 2) == 90_000.0
    assert allocations["champion"] > allocations["active"] > allocations["probation"]
    assert allocations["active"] > allocations["incubating"]
