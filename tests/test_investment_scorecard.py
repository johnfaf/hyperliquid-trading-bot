from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import sqlite3
import json

import pytest

from scripts.check_stock_paper_readonly import compare
from scripts.publish_dashboard_status import investment_review
from scripts.screen_investment_volatility import signals_with_brake
from src.backtest.candle_exit_replay import Bar, Decision
from src.learning.investment_scorecard import FAIL, PASS, UNKNOWN, economics, paper_cash_check, paper_checkpoint, paper_protection_check
from src.trading.order_journal import Fill, OrderJournal

NOW = datetime(2026, 11, 23, 11, tzinfo=timezone.utc)


def paper(path):
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE paper_account(id,balance,total_pnl,total_trades,winning_trades);
            INSERT INTO paper_account VALUES (1,1000,0,0,0);
            CREATE TABLE paper_trades(status,pnl);
        """)


def test_daily_cash_roll_forward_and_mismatch(tmp_path):
    path = tmp_path/"bot.db"
    paper(path)
    before = paper_checkpoint(path, NOW)
    assert paper_cash_check(before, paper_checkpoint(path, NOW+timedelta(days=1)))["status"] == UNKNOWN
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("INSERT INTO paper_trades VALUES ('closed',-2.75)")
        db.execute("UPDATE paper_account SET balance=997.25,total_pnl=-2.75,total_trades=1")
    after = paper_checkpoint(path, NOW+timedelta(days=1))
    assert paper_cash_check(before, after)["status"] == PASS
    assert paper_cash_check(before, {**after, "balance": 1000})["status"] == FAIL


def test_protection_check_distinguishes_stale_and_gapped_quotes(tmp_path):
    path = tmp_path/"guard.db"
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE paper_trades(coin,side,stop_loss,status);
            INSERT INTO paper_trades VALUES ('BTC','long',95,'open');
            CREATE TABLE bot_state(key,value);
        """)
        marks = {"observed_at":NOW.isoformat(),"mids":{"BTC":100}}
        db.execute("INSERT INTO bot_state VALUES ('paper_mark_prices',?)",(json.dumps(marks),))
    assert paper_protection_check(path,NOW)["status"] == PASS
    assert paper_protection_check(path,NOW+timedelta(minutes=20))["status"] == FAIL
    with closing(sqlite3.connect(path)) as db, db:
        marks["mids"]["BTC"] = 90
        db.execute("UPDATE bot_state SET value=?",(json.dumps(marks),))
    assert "through" in paper_protection_check(path,NOW)["note"]


def test_investment_memo_is_immutable_and_missing_cost_is_not_zero(tmp_path):
    path = tmp_path/"bot.db"
    paper(path)
    output = tmp_path/"memos"
    first = investment_review(path, output, tmp_path/"absent.json", None, {}, NOW, None)
    files = list(output.glob("investment-*.json"))
    saved, = files
    original = saved.read_bytes()
    second = investment_review(path, output, tmp_path/"absent.json", None, {}, NOW+timedelta(hours=1), 10.)
    assert second["reviewed_at"] == first["reviewed_at"] and second["btc"]["monthly_ops"] is None
    assert saved.read_bytes() == original and first["live_approved"] is False


def test_economics_requires_ended_sample_and_both_cost_cases():
    compact = {"net": 4., "cash": 0., "passive": 1., "net_without_largest": 3., "drawdown_pct": .4, "closed": 35}
    report = {"state":"Window complete","start":(NOW-timedelta(days=28)).isoformat(),"end":NOW.isoformat(),
              "evaluated_at":NOW.isoformat(),"stale":False,"results":{"base":compact,"stress":compact}}
    assert economics(report,None)[0]["status"] == UNKNOWN
    assert economics(report,10.)[0]["status"] == FAIL
    assert economics(report,0.)[0]["status"] == PASS
    assert economics({**report, "state":"Observing"},0.)[0]["status"] == UNKNOWN
    assert economics({**report, "results":{"base":compact}},0.)[0]["status"] == UNKNOWN


def test_stock_read_only_account_scope_and_fill_mismatch(tmp_path):
    path = tmp_path/"orders.db"
    account = "alpaca-paper:fixture"
    journal = OrderJournal(path, account)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE stock_cycles(account)")
        db.execute("INSERT INTO stock_cycles VALUES (?)", (account,))
    calls, states = [], {"positions":[], "fills":[], "orders":[]}
    def request(route, **kwargs):
        assert "payload" not in kwargs
        calls.append(route)
        return {"/v2/account":{"id":"fixture","status":"ACTIVE","trading_blocked":False,"cash":"1000"},
                "/v2/positions":states["positions"],"/v2/orders":states["orders"]}[route]
    broker = SimpleNamespace(request=request, fills=lambda: states["fills"])
    assert compare(broker,path)["status"] == UNKNOWN
    states["fills"] = [Fill("x","o","SPY","buy",Decimal(1),Decimal(200))]
    assert compare(broker,path)["status"] == FAIL
    journal.reserve("c",{},entry=True)
    journal.observe("c","filled","o")
    journal.record_fills(states["fills"])
    states["positions"] = [{"symbol":"SPY","qty":"1"}]
    assert compare(broker,path)["status"] == PASS
    states["positions"][0]["qty"] = "2"
    assert compare(broker,path)["status"] == FAIL
    assert set(calls) == {"/v2/account","/v2/positions","/v2/orders"}


def test_volatility_brake_does_not_change_the_past_or_exits():
    normal = [Bar(i,100,101,99,100,50) for i in range(96)]
    shock = Bar(96,100,108,99,107,50)
    signal = Decision(1,.9,None,"fixture")
    inputs = [Decision(0,0.,None,"fixture")]*96 + [signal,signal]
    prefix, repeats = signals_with_brake(normal+[shock], inputs[:-1])
    full, removed = signals_with_brake(normal+[shock,Bar(97,105,106,104,105,50)], inputs)
    assert prefix == full[:-1] and repeats == removed == 1 and full[-1] == signal


def test_archive_empty_wal_opt_in_does_not_discard_sidecars(tmp_path, monkeypatch):
    from scripts import archive_sqlite_rollback as archiver
    monkeypatch.setattr(archiver,"_sync_dir",lambda _: None)
    name = "bot-pre-"+"a"*40+".db"
    original = tmp_path/name
    original.write_bytes(b"immutable")
    wal = Path(str(original)+"-wal")
    wal.write_bytes(b"pending")
    with pytest.raises(ValueError,match="sidecars"):
        archiver.archive(tmp_path,name,allow_empty_wal=True)
    wal.write_bytes(b"")
    archiver.archive(tmp_path,name,allow_empty_wal=True)
    assert wal.is_file() and wal.stat().st_size == 0
