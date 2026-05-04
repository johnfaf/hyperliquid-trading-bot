"""Smoke tests for the v2 dashboard.

These exercise the FastAPI app via Starlette's TestClient. We avoid
spinning up uvicorn so tests stay fast and deterministic.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("starlette")

from fastapi.testclient import TestClient

from src.signals.calibration import CalibrationTracker
from src.ui.v2 import state as v2_state
from src.ui.v2.app import create_app


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("DASHBOARD_PUBLIC_READ", raising=False)
    v2_state.reset_components()
    cal = CalibrationTracker(db_path=str(tmp_path / "cal.db"))
    # Seed enough records that the global summary returns non-None.
    for _ in range(20):
        cal.record("strategy:m", 0.7, True, side="long", regime="trend")
        cal.record("strategy:m", 0.7, False, side="short", regime="trend")
    v2_state.set_components(calibration=cal)
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


def test_health_open_without_auth(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_serves_dashboard_when_no_token_configured(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Calibration breakdown" in r.text


def test_calibration_data_returns_summary(client):
    r = client.get("/api/calibration")
    assert r.status_code == 200
    payload = r.json()
    assert "global" in payload and "sources" in payload
    assert payload["global"]["n_sources"] >= 2
    assert any(row["source"] == "strategy:m" for row in payload["sources"])


def test_calibration_curve_returns_bins(client):
    r = client.get("/api/calibration/curve?key=global")
    assert r.status_code == 200
    payload = r.json()
    assert payload["key"] == "global"
    assert isinstance(payload["curve"], list)
    assert len(payload["curve"]) == 10  # N_BINS


def test_calibration_page_renders(client):
    r = client.get("/calibration")
    assert r.status_code == 200
    assert "Per-source calibration" in r.text


def test_login_redirect_when_token_required_and_no_cookie(monkeypatch, app):
    monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "secret123")
    client = TestClient(app)
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_login_round_trip_sets_session_cookie(monkeypatch, app):
    monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "secret123")
    client = TestClient(app)
    bad = client.post("/api/auth/login", data={"token": "wrong"}, follow_redirects=False)
    assert bad.status_code == 401

    ok = client.post("/api/auth/login", data={"token": "secret123"}, follow_redirects=False)
    assert ok.status_code == 303
    assert ok.headers["location"] == "/"
    assert "dashboard_v2_auth" in ok.headers.get("set-cookie", "")

    # The cookie should now grant access on the same client.
    follow = client.get("/", follow_redirects=False)
    assert follow.status_code == 200


def test_calibration_unavailable_when_component_not_set(monkeypatch, tmp_path):
    monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)
    v2_state.reset_components()
    app = create_app()
    client = TestClient(app)
    r = client.get("/api/calibration")
    assert r.status_code == 503
    assert r.json()["error"] == "calibration_unavailable"


def test_subsystem_registry_imports_cleanly():
    """Regression for the missing-``import os`` bug.

    The v2 dashboard wiring lives in src.core.subsystem_registry. Even
    when v2 is disabled, simply importing the module must succeed --
    last week a typo there crash-looped every boot.
    """
    import importlib
    mod = importlib.import_module("src.core.subsystem_registry")
    assert hasattr(mod, "boot_subsystems") or hasattr(mod, "BootContext") or mod is not None


def test_positions_endpoint_no_live_trader(client):
    r = client.get("/api/positions")
    assert r.status_code == 200
    payload = r.json()
    assert payload["live_available"] is False
    assert payload["positions"] == []
    assert payload["kill_switch"]["active"] is False


def test_positions_page_renders_without_live_trader(client):
    r = client.get("/positions")
    assert r.status_code == 200
    assert "Open positions" in r.text


def test_clear_kill_switch_requires_live_trader(client):
    r = client.post(
        "/api/operator/clear_kill_switch",
        data={"audit_reason": "investigating root cause"},
    )
    assert r.status_code == 503
    assert r.json()["error"] == "live_trader_unavailable"


def test_clear_kill_switch_rejects_short_reason(client):
    # Inject a stub live_trader so we exercise the audit-reason gate
    # without pulling in the full LiveTrader stack.
    class _Stub:
        def operator_clear_kill_switch(self, *, reason, operator):
            return {"cleared": True, "previous_reason": None,
                    "ts": "0", "operator": operator, "audit_reason": reason}
    v2_state.set_components(live_trader=_Stub())
    r = client.post(
        "/api/operator/clear_kill_switch",
        data={"audit_reason": "ok"},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "audit_reason_required"


def test_clear_kill_switch_calls_through_with_audit_reason(client):
    calls = []

    class _Stub:
        def operator_clear_kill_switch(self, *, reason, operator):
            calls.append({"reason": reason, "operator": operator})
            return {
                "cleared": True,
                "previous_active": True,
                "previous_reason": "daily_pnl_refresh_failed",
                "ts": "2026-05-03T00:00:00+00:00",
                "operator": operator,
                "audit_reason": reason,
            }

    v2_state.set_components(live_trader=_Stub())
    r = client.post(
        "/api/operator/clear_kill_switch",
        data={"audit_reason": "userFills API recovered, verified by hand"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["result"]["cleared"] is True
    assert calls and calls[0]["reason"].startswith("userFills")


def test_sources_endpoint_no_agent_scorer(client):
    r = client.get("/api/sources")
    assert r.status_code == 200
    payload = r.json()
    assert payload["available"] is False
    assert payload["rows"] == []


def test_sources_endpoint_aggregates_calibration(monkeypatch, tmp_path):
    """The scoreboard rolls per-(source|side|regime) calibration into a
    single source row. Verify the rollup matches the underlying
    calibrator state.
    """
    monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)
    v2_state.reset_components()
    cal = CalibrationTracker(db_path=str(tmp_path / "cal.db"))
    for _ in range(20):
        cal.record("strategy:m", 0.7, True, side="long", regime="trend")
        cal.record("strategy:m", 0.7, False, side="short", regime="trend")

    class _Scorer:
        def get_scorecard(self):
            return [{
                "source_key": "strategy:m",
                "rank": 1, "status": "active",
                "completed_trades": 40, "win_rate": 0.5,
                "weighted_accuracy": 0.5, "dynamic_weight": 0.5,
                "sharpe": 0.0, "avg_return": 0.0,
                "recent_pnl": 0.0, "total_pnl": 0.0,
                "last_trade_at": None,
            }]

    v2_state.set_components(agent_scorer=_Scorer(), calibration=cal)
    app = create_app()
    client = TestClient(app)
    r = client.get("/api/sources")
    assert r.status_code == 200
    payload = r.json()
    assert payload["available"] is True
    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    assert row["source"] == "strategy:m"
    assert row["calibration"]["subkeys"] == 2  # long+trend, short+trend
    assert row["calibration"]["samples"] >= 30


def test_sources_page_renders(client):
    r = client.get("/sources")
    assert r.status_code == 200
    # Renders the "agent scorer not initialised" banner when no scorer is wired.
    assert "Agent scorer is not initialised" in r.text or "Source scoreboard" in r.text


def test_traders_endpoint_smoke(client):
    r = client.get("/api/traders")
    assert r.status_code == 200
    payload = r.json()
    assert "rows" in payload
    assert "totals" in payload


def test_traders_page_renders(client):
    r = client.get("/traders")
    assert r.status_code == 200
    assert "Trader directory" in r.text or "Trader database is not initialised" in r.text


def test_audit_endpoint_smoke(client):
    r = client.get("/api/audit")
    assert r.status_code == 200
    payload = r.json()
    assert "kill_switch_log" in payload
    assert "calibration_quarantines" in payload
    assert "decisions" in payload
    assert "counts" in payload


def test_audit_endpoint_accepts_filters(client):
    r = client.get("/api/audit?status=executed&decision_id=abc123&days=3&limit=10")
    assert r.status_code == 200
    payload = r.json()
    assert payload["filters"]["status"] == "executed"
    assert payload["filters"]["decision_id"] == "abc123"
    assert payload["filters"]["days"] == 3
    assert payload["filters"]["limit"] == 10


def test_audit_page_renders(client):
    r = client.get("/audit")
    assert r.status_code == 200
    assert "Audit" in r.text or "Decision snapshots" in r.text


def test_backtest_status_endpoint(client):
    r = client.get("/api/backtest/status")
    assert r.status_code == 200
    payload = r.json()
    assert "running" in payload
    assert "recent_results" in payload


def test_backtest_run_requires_auth_when_token_set(monkeypatch, app):
    monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "secret")
    client = TestClient(app)
    r = client.post("/api/backtest/run", data={"max_wallets": 5})
    assert r.status_code == 401


def test_backtest_page_renders(client):
    r = client.get("/backtest")
    assert r.status_code == 200
    assert "Run a backtest" in r.text


def test_health_strip_returns_a_tone(client):
    r = client.get("/api/health/strip")
    assert r.status_code == 200
    payload = r.json()
    assert payload["tone"] in {"green", "amber", "rose"}
    assert "label" in payload


def test_clear_quarantine_requires_audit_reason(client):
    r = client.post(
        "/api/sources/clear_quarantine",
        data={"key": "strategy:m|long|trend", "audit_reason": "x"},
    )
    assert r.status_code == 400
    assert r.json()["error"] == "audit_reason_required"


def test_clear_quarantine_requires_calibration(client):
    v2_state.reset_components()
    r = client.post(
        "/api/sources/clear_quarantine",
        data={"key": "strategy:m|long|trend", "audit_reason": "post-incident review"},
    )
    assert r.status_code == 503
    assert r.json()["error"] == "calibration_unavailable"


def test_clear_quarantine_drops_records(monkeypatch, tmp_path):
    monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)
    v2_state.reset_components()
    cal = CalibrationTracker(
        db_path=str(tmp_path / "cal.db"),
        quarantine_min_samples=15,
        quarantine_ece=0.20,
    )
    for _ in range(30):
        cal.record("strategy:m", 0.9, False, side="long", regime="trend")
    v2_state.set_components(calibration=cal)
    app = create_app()
    client = TestClient(app)
    r = client.post(
        "/api/sources/clear_quarantine",
        data={"key": "strategy:m|long|trend", "audit_reason": "data quality fix"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["result"]["cleared"] is True
    # Source no longer tracked after clear.
    assert cal._source_total("strategy:m|long|trend") == 0


def test_recent_fills_endpoint_no_live_trader(client):
    r = client.get("/api/fills/recent")
    assert r.status_code == 200
    payload = r.json()
    assert payload["fills"] == []
    assert payload["count"] == 0
