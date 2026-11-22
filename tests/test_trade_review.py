from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import sqlite3
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

import config
from src.data import database as db
from src.trading.trade_decision_memory import COPY_CONCENTRATION_STUDY, entry_memory
from src.ui.v2.app import create_app
from src.ui.v2.trade_review import detail, review


@pytest.fixture
def ledger(tmp_path):
    path = tmp_path / "paper.db"
    now = datetime.now(timezone.utc)
    def ts(minutes):
        return (now - timedelta(minutes=minutes)).isoformat()
    first = "0x" + "a" * 40
    second = "0x" + "b" * 40
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.executescript("""
            CREATE TABLE paper_trades(id, coin, side, status, opened_at, closed_at, entry_price,
                exit_price, size, leverage, pnl, stop_loss, take_profit, metadata);
            CREATE TABLE decision_snapshots(decision_id, created_at, signal_timestamp, firewall_decision,
                final_status, paper_trade_id, metadata);
            CREATE TABLE decision_stage_events(event_id, decision_id, created_at, stage, status, reason);
        """)
        for tid, side, opened, closed, fill, pnl, meta in [
            (1, "long", 130, 60, 98, -4, {"source": "copy_trade", "source_trader": first,
             "decision_id": "matching", "initial_stop_loss": 98.5, "stop_trigger_price": 98.5,
             "close_reason": "stop_loss", "total_fees_paid": .4}),
            (2, "long", 102, 40, 101, 3, {"source": "copy_trade", "source_trader": second,
             "decision_id": "disagree", "initial_stop_loss": 96, "stop_trigger_price": 101.5,
             "close_reason": "trailing_stop"}),
            (3, "short", 110, 55, 100, -1, {"source": "copy_trade", "source_trader": first,
             "close_reason": "stop_loss", "risk_event": "trailing"}),
            (4, "long", 20, 10, 92, -400, {"source": "copy_trade", "source_trader": first,
             "close_reason": "live_reconciled_closed"}),
            (5, "long", 20, None, None, None, {"source": "strategy", "strategy_type": "scalping"}),
            (6, "long", 40, 35, 102, 1, {"source": "copy_trade", "source_trader": second, "close_reason": "time_limit"}),
        ]:
            conn.execute("INSERT INTO paper_trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                tid, "BTC", side, "closed" if closed is not None else "open", ts(opened), ts(closed) if closed is not None else None,
                100, fill, .1, 2, pnl, 98, 110, json.dumps(meta),
            ))
        conn.executemany("INSERT INTO decision_snapshots VALUES (?, ?, ?, ?, ?, ?, ?)", [
            ("matching", ts(132), ts(132), "approved", "paper_closed", 1,
             '{"why_entered":{"signal_reason":"Followed wallet increased BTC long exposure"}}'),
            ("disagree", ts(130), ts(130), "approved", "paper_closed", 6,
             '{"why_entered":{"signal_reason":"Never attach this to trade two"}}'),
        ])
        conn.executemany("INSERT INTO decision_stage_events VALUES (?, ?, ?, ?, ?, ?)", [
            ("b", "matching", ts(60), "paper_close", "paper_closed", "stop_loss"),
            ("a", "matching", ts(130), "paper_execution", "paper_opened", "paper_trade_opened"),
        ])
    return path, now


def test_breakdown_excludes_reconciliation_and_counts_true_overlap(ledger):
    path, now = ledger
    out = review(path=path, group="asset", now=now)
    s = out["summary"]
    assert out["excluded"] == 1
    assert s["closed"] == 4 and s["pnl"] == -1 and s["fee_count"] == 1
    assert s["overlap"] == 2 and out["peak"] == 2
    assert s["original"] == s["tightened"] == s["unverified_stops"] == 1
    assert s["gap_count"] == 2
    assert s["max_gap"] == pytest.approx(50.7614213)
    assert {row["label"] for row in out["groups"]} == {"BTC Long", "BTC Short"}
    assert review(path=path, scope="all", now=now)["summary"]["open"] == 1
    assert {row["label"] for row in review(path=path, scope="all", group="route", now=now)["groups"]} == {"strategy", "copy trade"}


def test_detail_only_links_an_exact_record_and_never_guesses(ledger):
    path, _ = ledger
    trade = detail(1, path=path)
    assert trade["entry_reason"] == "Followed wallet increased BTC long exposure"
    assert trade["events"][0]["stage"] == "paper_execution"
    assert trade["events"][-1]["stage"] == "paper_close"
    assert trade["selected_at"] is None
    old = detail(2, path=path)
    assert old["entry_reason"] is None and not old["linked"] and not old["events"]
    assert detail(3, path=path)["stop_type"] == "Unverified stop"


def test_completed_week_overlap_includes_peer_closing_later_without_future_pnl(ledger):
    path, _ = ledger
    with closing(sqlite3.connect(path)) as conn, conn:
        for tid, closed, pnl in [(10, "2026-11-11T00:00:00Z", 1), (11, "2026-11-20T00:00:00Z", 999)]:
            conn.execute("INSERT INTO paper_trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                tid, "ETH", "long", "closed", "2026-11-10T00:00:00Z", closed,
                100, 101, .1, 2, pnl, 98, 110, '{"source":"copy_trade"}',
            ))
    now = datetime(2026, 11, 22, tzinfo=timezone.utc)
    report = review(path=path, now=now, period="week")
    assert report["summary"]["closed"] == 1 and report["summary"]["pnl"] == 1
    assert report["summary"]["overlap"] == 1 and report["peak"] == 2
    assert report["shadow"] is None


def test_copy_snapshot_uses_open_paper_book_and_does_not_block(monkeypatch):
    db.init_db()
    monkeypatch.setattr(db, "_same_side_open_trade_limit", lambda: 4)
    db.open_paper_trade(None, "BTC", "long", 100, 1, metadata={"source": "strategy"})
    first = db.open_paper_trade(None, "BTC", "long", 100, 1, metadata={"source": "copy_trade"})
    second = db.open_paper_trade(None, "BTC", "long", 100, 1, metadata={"source": "copy_trade"})
    assert detail(first)["study_peers"] == 0
    assert detail(second)["study_peers"] == 1
    assert detail(second)["status"] == "open"


def test_shadow_comparison_uses_only_measured_closes_and_counts_missed_winners(ledger):
    path, now = ledger
    with closing(sqlite3.connect(path)) as conn, conn:
        for tid, peers in [(1, 0), (2, 1)]:
            meta = json.loads(conn.execute("SELECT metadata FROM paper_trades WHERE id=?", (tid,)).fetchone()[0])
            meta["entry_decision"] = {"copy_concentration_study": {"policy": COPY_CONCENTRATION_STUDY, "peers": peers}}
            conn.execute("UPDATE paper_trades SET metadata=? WHERE id=?", (json.dumps(meta), tid))
    shadow = review(path=path, now=now)["shadow"]
    assert shadow["tracked"] == 2 and shadow["unmeasured"] == 2
    assert shadow["skipped"] == shadow["missed_winners"] == 1
    assert shadow["actual"] == {"closed": 2, "pnl": -1, "drawdown": 4}
    assert shadow["exclusion"] == {"closed": 1, "pnl": -4, "drawdown": 4}


def test_trade_memory_is_written_at_open_and_not_overwritten_by_duplicate_close():
    db.init_db()
    saved = db.get_paper_account()
    db.init_paper_account(10_000)
    memory = entry_memory(SimpleNamespace(
        timestamp="2026-11-16T10:00:00+00:00", reason="Generic strategy label", confidence=.74,
        regime="ranging", context={"decision_attribution": {"reason": "Spread and volume gates passed"},
                                   "ev_breakdown": {"ev_bps": 15}, "private_provider_payload": "ignore"},
    ), route="strategy", firewall="approved")
    try:
        trade_id = db.open_paper_trade(None, "BTC", "long", 100, 1, stop_loss=98, take_profit=103,
                                      metadata={"entry_decision": memory})
        assert db.close_paper_trade_and_credit_account(trade_id, 97, -3, metadata_updates={
            "close_reason": "stop_loss", "stop_trigger_price": 98, "intended_exit_price": 97.2,
        })
        assert not db.close_paper_trade_and_credit_account(trade_id, 120, 20, metadata_updates={
            "close_reason": "take_profit",
        })
        trade = detail(trade_id)
        assert trade["entry_reason"] == "Spread and volume gates passed"
        assert trade["entry"]["recorded_at"] == trade["opened_at"]
        assert trade["entry"]["edge_bps"] == 15 and "private_provider_payload" not in trade["entry"]
        assert trade["exit"]["recorded_at"] == trade["closed_at"]
        assert trade["reason"] == "stop_loss" and trade["trigger"] == 98
        assert db.get_paper_account()["balance"] == pytest.approx(9997)
    finally:
        if saved:
            db.update_paper_account(saved["balance"], saved["total_pnl"], saved["total_trades"], saved["winning_trades"])
        else:
            with db.get_connection() as conn:
                conn.execute("DELETE FROM paper_account WHERE id=1")


def test_review_requires_sign_in_and_exposes_trade_details(ledger, monkeypatch):
    path, _ = ledger
    monkeypatch.setattr(config, "DB_PATH", str(path))
    monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "test-review-token")
    monkeypatch.setenv("DASHBOARD_PUBLIC_READ", "true")
    client = TestClient(create_app())
    assert client.get("/trades", follow_redirects=False).status_code == 303
    assert client.get("/trades/1", follow_redirects=False).status_code == 303
    client.post("/api/auth/login", data={"token": "test-review-token"})
    response = client.get("/trades?days=1&group=asset")
    assert response.status_code == 200
    assert "BTC Long" in response.text and 'href="/trades/1"' in response.text
    assert "Copy concentration study" in response.text and "No closed copy trades with a recorded pre-entry check." in response.text
    response = client.get("/trades/1")
    assert response.status_code == 200
    assert "Followed wallet increased BTC long exposure" in response.text
    assert "paper trade opened" in response.text
    assert "test-review-token" not in response.text
    assert client.get("/trades/9999").status_code == 404
