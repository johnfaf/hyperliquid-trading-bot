from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.core import readiness
from src.core.health_registry import SubsystemHealthRegistry, SubsystemState
from src.core.subsystem_registry import heartbeat_active


class _FakeLiveTrader:
    def __init__(self, stats):
        self._stats = dict(stats)

    def get_stats(self):
        return dict(self._stats)


class _FakeContainer:
    def __init__(self, live_stats=None):
        self.live_trader = _FakeLiveTrader(live_stats or {})


def _healthy_registry():
    registry = SubsystemHealthRegistry()
    registry.register("decision_firewall", affects_trading=True)
    registry.set_status(
        "decision_firewall",
        SubsystemState.HEALTHY,
        dependency_ready=True,
        startup_status="READY",
    )
    registry.heartbeat("decision_firewall")
    return registry


def test_evaluate_readiness_reports_ready_runtime(monkeypatch):
    monkeypatch.setattr(readiness, "_probe_db_readable", lambda: (True, ""))
    monkeypatch.setattr(readiness, "_probe_db_writable", lambda ttl_s=None: (True, ""))
    monkeypatch.setattr(readiness.db, "get_db_path", lambda: "test.db")

    snapshot = readiness.evaluate_readiness(
        container=_FakeContainer({"live_enabled": False}),
        health_registry=_healthy_registry(),
        stale_seconds=600,
    )

    assert snapshot["ready"] is True
    assert snapshot["live_ready"] is False
    assert snapshot["status"] == "ready"
    assert snapshot["reasons"] == []
    assert snapshot["checks"]["db_path"] == "test.db"


def test_evaluate_readiness_flags_live_deploy_blockers(monkeypatch):
    monkeypatch.setattr(readiness, "_probe_db_readable", lambda: (True, ""))
    monkeypatch.setattr(readiness, "_probe_db_writable", lambda ttl_s=None: (True, ""))
    monkeypatch.setattr(readiness.db, "get_db_path", lambda: "test.db")

    snapshot = readiness.evaluate_readiness(
        container=_FakeContainer(
            {
                "live_enabled": True,
                "deployable": False,
                "signer_available": False,
                "kill_switch_active": True,
                "kill_switch_reason": "manual_test",
                "status_reason": "missing signer",
            }
        ),
        health_registry=_healthy_registry(),
        stale_seconds=600,
    )

    assert snapshot["ready"] is True
    assert snapshot["live_ready"] is False
    assert "live_not_deployable:missing signer" in snapshot["reasons"]
    assert "missing_agent_wallet_signer" in snapshot["reasons"]
    assert "kill_switch_active:manual_test" in snapshot["reasons"]


def test_evaluate_readiness_flags_stale_trading_heartbeat(monkeypatch):
    monkeypatch.setattr(readiness, "_probe_db_readable", lambda: (True, ""))
    monkeypatch.setattr(readiness, "_probe_db_writable", lambda ttl_s=None: (True, ""))
    monkeypatch.setattr(readiness.db, "get_db_path", lambda: "test.db")

    registry = _healthy_registry()
    registry._subsystems["decision_firewall"].last_heartbeat = (
        datetime.now(timezone.utc) - timedelta(seconds=901)
    )

    snapshot = readiness.evaluate_readiness(
        container=_FakeContainer({"live_enabled": False}),
        health_registry=registry,
        stale_seconds=600,
    )

    assert snapshot["ready"] is False
    assert "stale_trading_heartbeats" in snapshot["reasons"]
    assert "decision_firewall" in snapshot["checks"]["stale_trading_subsystems"]


def test_runtime_incident_monitor_alerts_on_blocker_change(monkeypatch):
    snapshots = iter(
        [
            {
                "status": "not_ready",
                "ready": False,
                "live_ready": False,
                "reasons": ["stale_trading_heartbeats"],
                "checks": {"live_requested": False},
            },
            {
                "status": "not_ready",
                "ready": False,
                "live_ready": False,
                "reasons": ["db_write_failed:locked"],
                "checks": {"live_requested": False},
            },
        ]
    )
    monkeypatch.setattr(readiness, "evaluate_readiness", lambda **kwargs: next(snapshots))

    alerts = []
    import src.notifications.telegram_bot as telegram_bot

    monkeypatch.setattr(telegram_bot, "is_configured", lambda: True)
    monkeypatch.setattr(
        telegram_bot,
        "notify_runtime_incident",
        lambda snapshot, resolved=False: alerts.append((snapshot["reasons"], resolved)),
    )

    monitor = readiness.RuntimeIncidentMonitor(cooldown_s=0)
    monitor.evaluate_and_alert()
    monitor.evaluate_and_alert()

    assert alerts == [(["db_write_failed:locked"], False)]


def test_runtime_incident_monitor_alerts_on_resolution(monkeypatch):
    snapshots = iter(
        [
            {
                "status": "not_ready",
                "ready": False,
                "live_ready": False,
                "reasons": ["stale_trading_heartbeats"],
                "checks": {"live_requested": False},
            },
            {
                "status": "ready",
                "ready": True,
                "live_ready": False,
                "reasons": [],
                "checks": {"live_requested": False},
            },
        ]
    )
    monkeypatch.setattr(readiness, "evaluate_readiness", lambda **kwargs: next(snapshots))

    alerts = []
    import src.notifications.telegram_bot as telegram_bot

    monkeypatch.setattr(telegram_bot, "is_configured", lambda: True)
    monkeypatch.setattr(
        telegram_bot,
        "notify_runtime_incident",
        lambda snapshot, resolved=False: alerts.append((snapshot["status"], resolved)),
    )

    monitor = readiness.RuntimeIncidentMonitor(cooldown_s=0)
    monitor.evaluate_and_alert()
    monitor.evaluate_and_alert()

    assert alerts == [("ready", True)]


def test_heartbeat_active_refreshes_telegram_when_configured(monkeypatch):
    import src.notifications.telegram_bot as telegram_bot

    calls = []
    monkeypatch.setattr(telegram_bot, "is_configured", lambda: True)
    monkeypatch.setattr(telegram_bot, "heartbeat", lambda: calls.append("telegram"))

    registry = SubsystemHealthRegistry()
    heartbeat_active(SimpleNamespace(), registry)

    assert calls == ["telegram"]
