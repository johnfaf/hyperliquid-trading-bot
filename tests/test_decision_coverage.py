import json
from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import os
import sqlite3
import subprocess
import sys

import config
from scripts import run_decision_evidence_backtest as replay
from src.data import decision_coverage as coverage
from src.trading.hyperliquid_order_recovery import recover_submissions
from src.trading.order_journal import OrderJournal
from src.trading.stock_evidence import StockEvidence, read_summary


def create_ledger(path):
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.executescript("""
            CREATE TABLE paper_trades (id INT, opened_at TEXT, metadata TEXT);
            CREATE TABLE decision_snapshots (decision_id TEXT, created_at TEXT, final_status TEXT, paper_trade_id INT);
        """)


def test_coverage_matches_exact_evidence_only(tmp_path, monkeypatch):
    ledger = tmp_path / "bot.db"
    create_ledger(ledger)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(config, "DB_PATH", str(ledger))
    monkeypatch.setenv("DECISION_COVERAGE_ENABLED", "true")
    btc = {"coin": "BTC"}
    with coverage.track_cycle(9):
        no_signal, done, ambiguous, ghost, rejected, errored = coverage.expect("copy", [btc] * 6)
        coverage.resolve(no_signal, "no_signal")
        coverage.resolve(done, "decision_linked", decision="done")
        coverage.resolve(ghost, "decision_linked", decision="ghost")
        coverage.resolve(rejected, "decision_linked", decision="rejected")
        coverage.resolve(errored, "error", reason="QuoteUnavailable")
        data = {"evaluation_ref": done}
        coverage.prepare_entry(data, symbol="BTC", side="long", price=100)
        coverage.prepare_entry({}, symbol="BTC", side="long", price=101)
        with closing(sqlite3.connect(ledger)) as conn, conn:
            conn.execute("INSERT INTO paper_trades VALUES (?,?,?)", (1, now.isoformat(), json.dumps(data)))
            conn.executemany("INSERT INTO decision_snapshots VALUES (?,?,?,?)", [
                ("ghost", now.isoformat(), "paper_opened", 1),
                ("rejected", now.isoformat(), "firewall_execution_rejected", None),
            ])
    with coverage.track_cycle(10):
        coverage.resolve(ambiguous, "blocked")  # A stale ticket cannot finalize an earlier cycle.
        coverage.mark_failed()
    report = coverage.read_report(paper_path=ledger, now=now + timedelta(seconds=1))
    assert (report["expected"], report["registered"], report["terminal"], report["errors"]) == (6, 5, 4, 1)
    assert {r["state"] for r in report["pending"]} == {"paper_opened", "evaluation_expected"}
    assert (report["cycles"], report["finished"], report["unresolved_cycles"]) == (2, 1, 1)
    assert (report["intents"], report["unresolved_intents"], report["unresolved_orders"]) == (2, 1, None)
    assert report["non_error_final"] == 3
    assert report["failures"][0]["reason"] == "QuoteUnavailable"
    assert report["failures"][0]["reference"] == errored


def test_missing_storage_blocks_new_entries_only_in_active_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "missing" / "bot.db"))
    monkeypatch.setenv("DECISION_COVERAGE_ENABLED", "true")
    with coverage.track_cycle(1):
        assert not coverage.entry_allowed()
    assert coverage.entry_allowed()


def test_real_process_death_keeps_paper_intent_and_order_receipt_distinct(tmp_path):
    ledger = tmp_path / "bot.db"
    journal_path = ledger.with_suffix(".orders.sqlite3")
    receipt = tmp_path / "venue_receipt.json"
    create_ledger(ledger)
    original = "0x" + "a" * 32
    env = {**os.environ, "DECISION_COVERAGE_ENABLED": "true", "HL_BOT_DB": str(ledger)}
    assert subprocess.run([sys.executable, "-c", """
import json, os, sys
from pathlib import Path
from src.data import decision_coverage as coverage
from src.trading.order_journal import OrderJournal
with coverage.track_cycle(4):
    [ref] = coverage.expect('strategy', [{'coin': 'BTC'}])
    coverage.prepare_entry({'evaluation_ref': ref}, symbol='BTC', side='long', price=100)
    journal = OrderJournal(Path(sys.argv[1]), 'test-only')
    assert journal.reserve(sys.argv[3], {'symbol':'BTC'}, entry=True)
    Path(sys.argv[2]).write_text(json.dumps({'status':'order','order': {'status':'filled','order': {
        'cloid':sys.argv[3], 'oid':12,'coin':'BTC','side':'B','origSz':'0.25','sz':'0'
    }}}))
    os._exit(91)
""", str(journal_path), str(receipt), original], env=env, timeout=20).returncode == 91
    before = coverage.read_report(paper_path=ledger)
    assert (before["unresolved_cycles"], before["unresolved_intents"], before["unresolved_orders"]) == (1, 1, 1)
    journal = OrderJournal(journal_path, "test-only")
    assert not journal.reserve(original, {"symbol": "BTC"}, entry=True)
    assert not journal.reserve("new", {"symbol": "BTC"}, entry=True)
    assert journal.reserve("protect", {"symbol": "BTC"}, entry=False)
    journal.observe("protect", "accepted", "13")
    fill = {"tid": 23, "oid": 12, "coin": "BTC", "side": "B", "sz": ".1", "px": "100"}
    recover_submissions(journal, lambda _: json.loads(receipt.read_text()),
                        lambda: [fill, fill, {**fill, "tid": 24, "sz": ".15"}])
    assert journal.inventory() == {"BTC": Decimal(".25")}
    assert journal.reserve("new", {"symbol": "BTC"}, entry=True)
    journal.observe("new", "accepted")
    after = coverage.read_report(paper_path=ledger)
    assert (after["unresolved_intents"], after["unresolved_orders"]) == (1, 0)


def test_stock_coverage_separates_start_from_submission_checkpoint(tmp_path):
    path = tmp_path / "stock.db"
    now = datetime.now(timezone.utc)
    initial = StockEvidence(path, "unbound")
    initial.start("dead", "strategy", observed_at=now - timedelta(minutes=20))
    initial.start("ok", "strategy", observed_at=now - timedelta(minutes=5))
    account = StockEvidence(path, "alpaca-paper:test")
    OrderJournal(path, "alpaca-paper:test")
    intent = {"client_order_id": "test", "side": "buy", "qty": 1}
    account.record("dead", {"state": "submitting", "intent": intent}, symbol="SPY", mode="paper", purpose="strategy")
    account.record("ok", {"state": "no_order"}, symbol="SPY", mode="shadow", purpose="strategy", finished=True)
    result = read_summary(path)
    assert result["coverage"] == {"started": 2, "finished": 1, "overdue": 1}


def test_future_bar_strict_boundary_and_horizon_cutoff():
    start = replay.dt("2026-11-14T00:00:00Z")
    t = int(start.timestamp() * 1000)
    bar = replay.BAR
    data = {"BTC": {t: (1, 2), **{t + i * bar: (100, 101) for i in range(1, 5)}}}
    item = {"created_at": start.isoformat(), "coin": "BTC", "side": "long"}
    end = start + timedelta(minutes=75)
    assert abs(replay.forward_price(item, data, end)[0] - 100) < 1e-8
    assert replay.forward_price(item, data, end - timedelta(seconds=1))[1] == "window_ends_before_horizon"
    del data["BTC"][t + 3 * bar]
    assert replay.forward_price(item, data, end)[1] == "missing_contiguous_bars"


def test_copy_exclusion_uses_prior_positions_even_if_not_in_cohort():
    start = replay.dt("2026-11-15T00:00:00Z")
    end = start + timedelta(days=1)
    def trade(i, opened, closed, pnl):
        return dict(id=i, coin="BTC", side="long", opened_at=(start + timedelta(hours=opened)).isoformat(),
                    closed_at=(start + timedelta(hours=closed)).isoformat(), copy=True, clean=True, pnl=pnl)
    base, filtered, omitted = replay.copy_replay([trade(1, -1, 4, -3), trade(2, 1, 2, -5), trade(3, 4, 5, 2)], start, end)
    assert (base["pnl"], filtered["pnl"], omitted["pnl"]) == (-3, 2, -5)


def test_prospective_candidate_never_approved_by_retrospective_results():
    start = replay.dt("2026-11-01T00:00:00Z")
    end = start + timedelta(days=14)
    ledger = []
    for i in range(30):
        overlap = i % 2
        opened = start + timedelta(hours=i)
        ledger.append(dict(id=i, opened_at=opened.isoformat(), closed_at=(opened + timedelta(minutes=60)).isoformat(),
                           copy=True, clean=True, entry_price=100, size=2, pnl=-10 if overlap else 5,
                           coverage_revision="build", shadow={"policy": "test", "peers": overlap}))
    coverage_summary = {"available": True, "schedule_contiguous": True, "revisions": ["build"], "protocols": ["test"],
                        "expected": 1000, "terminal": 970, "unresolved_cycles": 0, "unresolved_intents": 0,
                        "unresolved_orders": 0, "errors": 0}
    assert replay.prospective_review(ledger, coverage_summary, start, end, "test")["state"] == "Operator paper review required"
    result = replay.prospective_review(ledger, {**coverage_summary, "revisions": ["unknown"], "terminal": 949}, start, end, "test")
    assert result["state"] == "Not eligible"
    assert len(result["reasons"]) >= 2
