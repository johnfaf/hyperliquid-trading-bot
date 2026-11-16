from contextlib import closing
from datetime import date, datetime, timedelta, timezone
import json
import sqlite3

import config
from scripts import publish_dashboard_status as publisher
from scripts import run_copy_walk_forward as replay
from src.data import decision_coverage
from src.trading import copy_shadow_report as daily
from src.trading.trade_decision_memory import COPY_CONCENTRATION_STUDY


def meta(peers=None, **fields):
    data = {"source": "copy_trade", **fields}
    if peers is not None:
        data["entry_decision"] = {"copy_concentration_study": {"peers": peers, "policy": COPY_CONCENTRATION_STUDY}}
    return json.dumps(data)


def ledger(path, closes):
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE paper_trades(id, closed_at, entry_price, size, pnl, metadata, status)")
        db.executemany("INSERT INTO paper_trades VALUES(?,?,?,?,?,?,'closed')", closes)


def test_daily_cutoff_compares_only_tagged_clean_closes(tmp_path):
    path = tmp_path / "bot.db"
    start, end = daily.utc_day(date(2026, 11, 15))
    ledger(path, [
        (1, (start - timedelta(seconds=1)).isoformat(), 100, 2, 100, meta(0)),
        (2, start.isoformat(), 100, 2, 5, meta(0)),
        (3, (start + timedelta(hours=1)).isoformat(), 100, 2, -9, meta(1)),
        (4, (start + timedelta(hours=2)).isoformat(), 100, 2, 2, meta(1)),
        (5, (start + timedelta(hours=3)).isoformat(), 100, 2, -500, meta(1, exit_decision={"reason": "live_reconciled_closed"})),
        # Record a tainted label independently of the entry tag.
        (6, (start + timedelta(hours=4)).isoformat(), 100, 2, 90, meta(0, tainted=True)),
        (7, (start + timedelta(hours=5)).isoformat(), 100, 2, 40, meta()),
        (8, end.isoformat(), 100, 2, 100, meta(0)),
    ])
    result = daily.build_daily(path, start.date())
    assert result["baseline"] == {"n": 3, "net": -2.0, "drawdown": 9.0, "wins": 2}
    assert result["all_copy"]["net"] == 38 and result["all_copy"]["n"] == 4
    assert result["shadow"]["net"] == 5
    assert abs(result["stress_baseline"]["net"] - (-2 - 3 * .32)) < 1e-9
    assert result["skipped"] == 2 and result["missed_winners"] == 1
    assert result["untagged"] == 1 and result["excluded"] == 2
    assert result["coverage"] is None


def test_daily_archive_rolls_after_ten_minutes_and_is_not_rewritten(tmp_path):
    path = tmp_path / "bot.db"
    ledger(path, [(1, "2026-11-15T01:00:00+00:00", None, 2, 4, meta(0))])
    reports = tmp_path / "reports"
    midnight = datetime(2026, 11, 16, tzinfo=timezone.utc)
    assert publisher.daily_copy_report(path, reports, midnight + timedelta(minutes=9))["date"] == "2026-11-14"
    result = publisher.daily_copy_report(path, reports, midnight + timedelta(minutes=10))
    assert result["baseline"]["net"] == 4 and result["stress_baseline"] is None
    assert (reports / "copy-shadow-2026-11-15.md").exists()
    assert result == publisher.daily_copy_report(tmp_path / "missing.db", reports, midnight + timedelta(hours=1))


def test_daily_digest_and_incident_rate_limit_follow_successful_delivery(tmp_path, monkeypatch):
    path = tmp_path / "bot.db"
    ledger(path, [])
    now = datetime(2026, 11, 16, 1, tzinfo=timezone.utc)
    report = daily.build_daily(path, date(2026, 11, 15))
    state = tmp_path / "notice.json"
    sent = []

    def send(_, message):
        sent.append(message)
        return len(sent) > 1

    monkeypatch.setattr(publisher, "send_telegram", send)
    args = (report, ["paper_intent"], tmp_path / "env", state)
    assert publisher.notify_copy(*args, now=now) == "delivery_unavailable"
    assert not state.exists()
    assert publisher.notify_copy(*args, now=now) == "sent"
    assert "No matched, priced" in sent[-1] and "after 20 minutes" in sent[-1]
    assert publisher.notify_copy(*args, now=now + timedelta(minutes=1)) == "quiet"
    assert publisher.notify_copy(*args, now=now + timedelta(hours=6)) == "sent"
    assert "Copy comparison" not in sent[-1]


def test_alert_ignores_unmeasured_venue_journal_but_flags_known_stalled_items():
    status = dict(available=True, late=False, failed_cycles=0, overdue_cycles=0, overdue_evaluations=0,
                  errors=0, overdue_intents=0, unresolved_orders=None)
    assert publisher.copy_incidents(status) == []
    assert publisher.copy_incidents({**status, "overdue_intents": 1, "unresolved_orders": 2}) == ["paper_intent", "venue_order"]


def test_overdue_maturity_and_future_snapshot_not_counted(tmp_path, monkeypatch):
    path = tmp_path / "bot.db"
    monkeypatch.setattr(config, "DB_PATH", str(path))
    monkeypatch.setenv("DECISION_COVERAGE_ENABLED", "true")
    now = datetime.now(timezone.utc)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.executescript("""
            CREATE TABLE paper_trades (id INT, opened_at TEXT, metadata TEXT);
            CREATE TABLE decision_snapshots (decision_id TEXT, created_at TEXT, updated_at TEXT, final_status TEXT, paper_trade_id INT);
        """)
        conn.execute("INSERT INTO decision_snapshots VALUES (?,?,?,?,NULL)",
                     ("later", now.isoformat(), (now + timedelta(hours=1)).isoformat(), "rejected"))
    with decision_coverage.track_cycle(1):
        ref, = decision_coverage.expect("copy", [{"coin": "BTC"}])
        decision_coverage.resolve(ref, "decision_linked", decision="later")
        decision_coverage.prepare_entry({}, symbol="BTC", side="long", price=100)
        early = decision_coverage.read_report(paper_path=path, now=now + timedelta(minutes=2))
        late = decision_coverage.read_report(paper_path=path, now=now + timedelta(minutes=21))
    assert early["terminal"] == 0 and early["unresolved_intents"] == 1
    assert early["overdue_cycles"] == early["overdue_intents"] == early["overdue_evaluations"] == 0
    assert late["overdue_cycles"] == late["overdue_intents"] == late["overdue_evaluations"] == 1


def test_future_close_cannot_change_observed_overlap_and_tainted_stays_in_book():
    def row(i, opened, closed, fields, pnl=-10):
        return {"id": i, "opened_at": f"2026-11-15T{opened}:00+00:00", "closed_at": f"2026-11-15T{closed}:00+00:00",
                "coin": "BTC", "side": "long", "status": "closed", "metadata": fields, "pnl": pnl, "entry_price": 100, "size": 2}

    rows = [row(1, "01:00", "03:00", meta(tainted=True)), row(2, "02:00", "04:00", meta()),
            row(3, "03:30", "04:30", meta(0)), row(4, "05:00", "07:00", meta())]
    entries, labels = replay.point_in_time(rows)
    assert [e.overlaps for e in entries] == [False, True, False, False]
    assert 1 not in labels and entries[2].tagged
    rows[0]["closed_at"] = "2026-11-15T06:00:00+00:00"
    assert replay.point_in_time(rows)[0][:2] == entries[:2]


def test_walk_forward_freezes_predictions_and_embargoes_crossing_labels():
    start = datetime(2026, 11, 14, tzinfo=timezone.utc)
    first = start + timedelta(days=1)
    second = first + timedelta(days=1)
    end = second + timedelta(days=1)
    entries, labels = [], {}
    for i in range(12):
        at = start + timedelta(minutes=i)
        overlap = i % 2 == 0
        entries.append(replay.Entry(i, at, overlap, True))
        net = -10 if overlap else 4
        labels[i] = replay.Label(at + timedelta(hours=1), net, net - 1)
    entries += [replay.Entry(90, first - timedelta(hours=2), False, True),
                replay.Entry(91, first, True, True), replay.Entry(92, first, False, True)]
    labels.update({90: replay.Label(first - timedelta(hours=1), -100, -100),
                   91: replay.Label(first + timedelta(hours=2), 3, 2),
                   92: replay.Label(second, 1000, 999)})
    before = replay.evaluate(entries, labels, start, [first, second], end)[0]
    changed = {**labels, 91: replay.Label(first + timedelta(hours=2), 99, 98)}
    after = replay.evaluate(entries, changed, start, [first, second], end)[0]
    assert before["train"] == 12 and before["omit"] and before["scored"] == 1
    assert after["predictions"] == before["predictions"] == {92}
    assert before["baseline"]["net"] != after["baseline"]["net"]


def test_private_trade_page_renders_daily_notices(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from src.ui.v2.app import create_app
    from src.ui.v2.cache import invalidate

    path = tmp_path / "bot.db"
    ledger(path, [])
    monkeypatch.setattr(config, "DB_PATH", str(path))
    monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "copy-fixture")
    report = daily.build_daily(path, date(2026, 11, 15))
    publisher.publish(tmp_path / "dashboard_ops.json", {"copy_daily": report, "copy_alerts": ["Paper alert."],
                      "observed_at": datetime.now(timezone.utc).isoformat()})
    invalidate(("trade-review",))
    try:
        client = TestClient(create_app())
        assert client.get("/trades", follow_redirects=False).status_code == 303
        assert client.post("/api/auth/login", data={"token": "copy-fixture"}, follow_redirects=False).status_code == 303
        page = client.get("/trades")
        assert page.status_code == 200
        assert "Daily copy comparison" in page.text and "Paper alert." in page.text
        assert "Comparison unmeasured" in page.text
    finally:
        invalidate(("trade-review",))
