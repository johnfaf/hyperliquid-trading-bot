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
