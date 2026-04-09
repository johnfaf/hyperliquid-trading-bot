from datetime import datetime, timedelta, timezone

from src.core.health_registry import SubsystemHealthRegistry, SubsystemState
from src.ui import dashboard


class _FakeFirewall:
    def get_stats(self):
        return {
            "total_signals": 10,
            "passed": 6,
            "top_rejection_reason": "rejected_confidence",
            "daily_losses": 12.5,
        }


class _FakeLiveTrader:
    def get_stats(self):
        return {
            "kill_switch_active": True,
            "kill_switch_reason": "env:LIVE_EXTERNAL_KILL_SWITCH",
            "canary_mode": True,
            "max_order_usd": 25.0,
            "daily_pnl": -7.5,
            "daily_pnl_limit": 100.0,
            "total_entry_signals_today": 3,
        }


def test_build_runtime_health_snapshot_includes_subsystems_and_safety(monkeypatch):
    registry = SubsystemHealthRegistry()
    registry.register("decision_firewall", affects_trading=True)
    registry.register("live_trader", affects_trading=True)
    registry.set_status(
        "decision_firewall",
        SubsystemState.HEALTHY,
        dependency_ready=True,
        startup_status="READY",
    )
    registry.set_status(
        "live_trader",
        SubsystemState.DEGRADED,
        reason="ws_lag",
        dependency_ready=True,
        startup_status="READY",
    )
    # Force one stale heartbeat.
    registry._subsystems["live_trader"].last_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=900)

    monkeypatch.setattr(dashboard, "_health_registry", registry)
    monkeypatch.setattr(dashboard, "_firewall", _FakeFirewall())
    monkeypatch.setattr(dashboard, "_live_trader", _FakeLiveTrader())
    monkeypatch.setenv("DASHBOARD_HEALTH_STALE_SECONDS", "600")

    snapshot = dashboard._build_runtime_health_snapshot()

    assert snapshot["overall"] == "at_risk"
    assert snapshot["all_trading_safe"] is True
    assert "live_trader" in snapshot["at_risk_subsystems"]
    assert "live_trader" in snapshot["stale_subsystems"]
    assert snapshot["firewall"]["top_rejection_reason"] == "rejected_confidence"
    assert snapshot["live_trader"]["kill_switch_active"] is True
