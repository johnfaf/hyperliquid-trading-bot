from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import sqlite3

from fastapi.testclient import TestClient

import config
from scripts import publish_dashboard_status as publisher
from src.ui.v2.app import create_app
from src.ui.v2.desk import host_status
from src.ui.v2.state import reset_components


def test_ops_publisher_exposes_status_only_and_uses_latest_entry():
    def event(state, mode, stamp):
        return json.dumps({
            "MESSAGE": json.dumps({"state": state, "mode": mode, "secret": "never publish"}),
            "__REALTIME_TIMESTAMP": stamp,
        })
    lines = "\n".join([event("no_order", "shadow", "123"), "broken", event("cycle_failed", "paper", "456")])
    assert publisher.collect_stock(lines) == {
        "state": "cycle_failed", "mode": "paper", "observed_at": "1970-01-01T00:00:00.000456+00:00",
    }


def test_publisher_collects_job_and_report_status(tmp_path, monkeypatch):
    def command(*args):
        if args[0] == "journalctl":
            return ""
        if args[2].endswith(".timer"):
            return "active\n"
        return "Result=success\nExecMainExitTimestamp=Mon 2026-11-16 00:10:00 UTC\n"

    monkeypatch.setattr(publisher, "host_command", command)
    reports = tmp_path / "exit-study/reports"
    reports.mkdir(parents=True)
    (reports / "scorecard-2026-11-15.md").write_text(
        "- Since capture began: 10,919/10,921 usable slots (99.98%); no missing samples.", encoding="utf-8",
    )
    (reports / "scorecard-latest.md").write_text("Not a dated report", encoding="utf-8")
    data = publisher.build_snapshot(tmp_path)
    assert data["exit_report"] == "scorecard-2026-11-15.md"
    assert data["coverage"]["percent"] == 99.98
    assert data["jobs"]["study_backup"]["timer_active"]
    destination = tmp_path / "dashboard_ops.json"
    publisher.publish(destination, data)
    assert host_status(destination, datetime.now(timezone.utc)) == data
    assert host_status(destination, datetime.now(timezone.utc) + timedelta(minutes=4)) == {}


def test_overview_uses_eligible_paper_costs_and_separates_decisions(tmp_path, monkeypatch):
    monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", False)
    path = tmp_path / "bot.db"
    monkeypatch.setattr(config, "DB_PATH", str(path))
    reset_components()
    now = datetime.now(timezone.utc).isoformat()
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE paper_account(id, balance, total_pnl);
            INSERT INTO paper_account VALUES (1, 9976.21, -23.79);
            CREATE TABLE paper_trades(id, coin, side, pnl, closed_at, metadata, status);
            CREATE TABLE decision_snapshots(decision_id, paper_trade_id, final_status, firewall_decision, created_at);
        """)
        db.executemany("INSERT INTO paper_trades VALUES (?, ?, ?, ?, ?, ?, ?)", [
            (1, "BTC", "short", -8.25, now, '{"close_reason":"trailing_stop","source":"copy_trade"}', "closed"),
            (2, "ETH", "long", 4.00, now, '{"close_reason":"time_limit"}', "closed"),
            (3, "XRP", "long", -900, now, '{"close_reason":"live_reconciled_closed"}', "closed"),
        ])
        db.executemany("INSERT INTO decision_snapshots VALUES (?, ?, ?, ?, ?)", [
            ("a", 1, "paper_closed", "approved", now), ("b", None, "copy_entry_rejected", "approved", now),
        ])
    client = TestClient(create_app())
    response = client.get("/")
    assert response.status_code == 200
    for value in ("$9,976.21", "-$4.25", "50.0%", "trailing stop", "Linked to a paper trade", "chart-loss"):
        assert value in response.text
    assert "XRP" not in response.text and "-$900.00" not in response.text
    assert "7-day window" in client.get("/?hours=168").text


def test_settings_are_private_even_when_overview_is_public(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "test-only-desk-token")
    monkeypatch.setenv("DASHBOARD_PUBLIC_READ", "true")
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "unavailable.db"))
    client = TestClient(create_app())
    assert client.get("/").status_code == 200
    assert client.get("/settings", follow_redirects=False).status_code == 303
    assert client.post("/api/auth/login", data={"token": "test-only-desk-token"}, follow_redirects=False).status_code == 303
    page = client.get("/settings")
    assert page.status_code == 200
    assert "Read only" in page.text and "0.25%" in page.text and "$75.00" in page.text
    assert "test-only-desk-token" not in page.text
    assert client.post("/settings").status_code == 405


def test_primary_dashboard_fails_closed_without_access_token(monkeypatch):
    monkeypatch.delenv("DASHBOARD_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("DASHBOARD_PUBLIC_READ", raising=False)
    monkeypatch.setenv("DASHBOARD_V2_PRIMARY", "true")
    client = TestClient(create_app())
    assert client.get("/", follow_redirects=False).status_code == 303
    assert client.get("/settings", follow_redirects=False).status_code == 303
    assert client.post("/api/auth/login", data={"token": "random"}, follow_redirects=False).status_code == 401
