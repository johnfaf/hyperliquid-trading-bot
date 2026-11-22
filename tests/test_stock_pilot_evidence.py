from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json

from fastapi.testclient import TestClient
import pytest

import config
from scripts import publish_dashboard_status as publisher
from scripts.stock_pilot_report import render
from src.trading.order_journal import Fill, OrderJournal
from src.trading.stock_evidence import StockEvidence, read_summary
from src.ui.v2.app import create_app
from src.core import clock_provider
from src.backtest.replay.clock import ReplayClock

START = datetime(2026, 11, 16, 15, tzinfo=timezone.utc)
SESSION = "2026-11-16"


def test_default_evidence_timestamps_follow_injected_clock(tmp_path):
    path = tmp_path / "replay.db"
    journal = OrderJournal(path, "paper-test")
    evidence = StockEvidence(path, journal.account)
    previous = clock_provider.install(ReplayClock(int(START.timestamp() * 1000)))
    try:
        evidence.start("cycle", "strategy")
        evidence.record("cycle", {"state": "no_order", "session": SESSION},
                        symbol="SPY", mode="shadow", purpose="strategy", finished=True)
        summary = read_summary(path)
        assert summary["decisions"][0]["observed_at"] == START.isoformat()
        assert summary["coverage"] == {"started": 1, "finished": 1, "overdue": 0}
    finally:
        clock_provider.restore(previous)


def mark(evidence, key, minutes, price, *, mode="shadow", state="no_order", purpose="strategy", **extras):
    at = START + timedelta(minutes=minutes)
    evidence.record(key, {
        "state": state, "session": SESSION, "quote_valid": True, "quote_age": 1.0,
        "quote_at": at.isoformat(), "mark": price, "quantity": "0", **extras,
    }, symbol="SPY", mode=mode, purpose=purpose, observed_at=at)


def order(journal, cid, venue_id, side, *, purpose):
    intent = {"client_order_id": cid, "symbol": "SPY", "side": side, "qty": "1"}
    assert journal.reserve(cid, intent, entry=side == "buy", purpose=purpose)
    journal.observe(cid, "filled", venue_id)
    return intent


def test_shadow_proposal_not_confused_with_later_paper_order(tmp_path):
    path = tmp_path / "orders.db"
    journal = OrderJournal(path, "demo-private-account")
    evidence = StockEvidence(path, journal.account)
    intent = order(journal, "same-session-key", "paper-id", "buy", purpose="strategy")
    mark(evidence, "a", 0, 100, state="shadow_intent", intent=intent)
    mark(evidence, "b", 5, 101)
    evidence.record(
        "closed", {"state": "outside_entry_or_exit_window", "session": SESSION}, symbol="SPY",
        mode="shadow", purpose="strategy", observed_at=START + timedelta(minutes=15),
    )
    data = read_summary(path)
    assert data["scorecard"]["strategy_move"] is None
    assert data["scorecard"]["hold_move"] == pytest.approx(10)
    assert data["scorecard"]["cash_move"] == 0
    assert data["scorecard"]["cycles"] == 3 and data["scorecard"]["quote_checks"] == 2
    assert data["decisions"][2]["status"] is None
    assert "demo-private-account" not in json.dumps(data)
    assert "Unmeasured" in render(data["scorecard"])
    assert evidence.ready_shadow(SESSION, START + timedelta(minutes=10))
    assert not evidence.ready_shadow(SESSION, START + timedelta(minutes=30))


def test_scorecard_uses_only_actual_strategy_fills_and_estimated_costs(tmp_path):
    path = tmp_path / "orders.db"
    journal = OrderJournal(path, "paper")
    evidence = StockEvidence(path, journal.account)
    mark(evidence, "start", 0, 100, mode="paper")
    trade = order(journal, "strat-one", "real-paper", "buy", purpose="strategy")
    order(journal, "test", "sim-drill", "buy", purpose="drill")
    journal.record_fills([
        Fill("a", "real-paper", "SPY", "buy", Decimal("1"), Decimal("100.20"),
             (START + timedelta(minutes=1)).isoformat()),
        Fill("b", "sim-drill", "SPY", "buy", Decimal("3"), Decimal("101"),
             (START + timedelta(minutes=2)).isoformat()),
    ])
    mark(evidence, "entry", 2, 101, state="intent_sent", mode="paper", intent=trade)
    mark(evidence, "end", 5, 110, mode="paper")
    result = read_summary(path)
    score = result["scorecard"]
    assert score["strategy_move"] == pytest.approx(9.77996)
    assert score["hold_move"] == pytest.approx(100)
    assert result["decisions"][1]["status"] == "filled"
    mark(evidence, "incident", 4, 109, state="broker_position_mismatch", mode="paper")
    assert read_summary(path)["scorecard"]["strategy_move"] is None


def test_supervised_drill_requires_cash_and_position_reconciliation(tmp_path):
    path = tmp_path / "orders.db"
    journal = OrderJournal(path, "paper")
    evidence = StockEvidence(path, journal.account)
    buy = order(journal, "buy", "one", "buy", purpose="drill")
    sell = order(journal, "sell", "two", "sell", purpose="drill")
    mark(evidence, "drill-buy", 0, 100, purpose="drill", state="intent_sent", intent=buy, account_cash="1000")
    mark(evidence, "drill-sell", 5, 100, purpose="drill", state="intent_sent", intent=sell, account_cash="990")
    journal.record_fills([
        Fill("a", "one", "SPY", "buy", Decimal(".1"), Decimal("100"), (START + timedelta(minutes=1)).isoformat()),
        Fill("b", "two", "SPY", "sell", Decimal(".1"), Decimal("101"), (START + timedelta(minutes=6)).isoformat()),
    ])
    assert read_summary(path)["drill"]["state"] == "Awaiting fills and balance check"
    mark(evidence, "follow-up", 10, 101, account_cash="1000.1")
    assert read_summary(path)["drill"]["state"] == "Reconciled"
    assert read_summary(path)["scorecard"]["strategy_move"] is None


def test_closed_session_quiet_and_quote_alert_repeat_suppressed(tmp_path, monkeypatch):
    snapshot = {"stock": {"state": "outside_entry_or_exit_window", "observed_at": START.isoformat()},
                "jobs": {"stock": {"timer_active": True, "result": "success"}}, "stock_pilot": {}}
    config_path, alert_state = tmp_path / "env", tmp_path / "alerts.json"
    calls = []
    monkeypatch.setattr(publisher, "send_telegram", lambda cfg, msg: calls.append(msg) or True)
    assert publisher.maybe_alert(snapshot, config_path, alert_state, now=START) == "quiet"
    snapshot["stock"]["state"] = "quote_not_usable"
    snapshot["stock_pilot"]["decisions"] = [{"state": "quote_not_usable"}] * 2
    assert publisher.maybe_alert(snapshot, config_path, alert_state, now=START) == "quiet"
    snapshot["stock_pilot"]["decisions"].append({"state": "quote_not_usable"})
    assert publisher.maybe_alert(snapshot, config_path, alert_state, now=START) == "sent"
    assert publisher.maybe_alert(snapshot, config_path, alert_state, now=START) == "suppressed"
    assert len(calls) == 1


def test_stock_pilot_page_requires_auth_and_renders_closed_session(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "test-login")
    monkeypatch.setenv("DASHBOARD_PUBLIC_READ", "true")
    path = tmp_path / "orders.db"
    journal = OrderJournal(path, "paper")
    evidence = StockEvidence(path, journal.account)
    mark(evidence, "a", 0, 100)
    mark(evidence, "b", 5, 101)
    snapshot = {"observed_at": datetime.now(timezone.utc).isoformat(), "stock_pilot": read_summary(path)}
    (tmp_path / "dashboard_ops.json").write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "crypto.db"))
    client = TestClient(create_app())
    assert client.get("/stock-pilot", follow_redirects=False).status_code == 303
    client.post("/api/auth/login", data={"token": "test-login"})
    response = client.get("/stock-pilot")
    assert response.status_code == 200
    assert "No confirmed stock paper fills" in response.text
    assert "Shadow or mixed session; strategy P&amp;L unmeasured" in response.text
    assert "test-login" not in response.text
