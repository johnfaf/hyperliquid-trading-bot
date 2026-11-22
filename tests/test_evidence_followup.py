from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import sqlite3
from types import SimpleNamespace

import pytest

import config
from scripts.capture_hyperliquid_account_history import bounded
from scripts.capture_paper_funding import capture
from scripts.reconcile_hyperliquid_execution import _match_venue_order, actual_funding, venue_order_index
from scripts.run_forward_shadow import BAR, HOUR, LOOKBACK, cycle, decisions
from src.backtest.candle_exit_replay import Bar
from src.data import database as db
from src.learning.strategy_evidence import stamp_paper_entry
from src.trading.paper_funding import FundingLedger, MODEL, ledger_path, millis, verified_cost
from src.trading.paper_funding_health import funding_health
from src.trading.paper_trader import PaperTrader

NOW = datetime(2026, 11, 22, 9, 5, tzinfo=timezone.utc)


def settlement(ledger, hour, rate, price):
    ledger.record(observed_ms=hour-30_000, oracle_prices={"BTC": price})
    ledger.record(observed_ms=hour+15_000, rates=[{"coin": "BTC", "time": hour+10, "fundingRate": rate}])


def test_two_settlements_are_retried_without_double_debit(tmp_path):
    path = tmp_path/"funding.sqlite3"
    ledger = FundingLedger(path)
    h2 = millis(NOW)//HOUR*HOUR
    h1 = h2-HOUR
    settlement(ledger, h1, ".001", 100.)
    settlement(ledger, h2, "-.002", 200.)
    settlement(ledger, h2, "-.002", 200.)
    trade = {"coin": "BTC", "side": "long", "size": 2, "leverage": 1.5,
             "opened_at": (NOW-timedelta(hours=2)).isoformat(), "closed_at": NOW.isoformat()}
    summary = FundingLedger(path).position(trade, now=NOW)
    assert summary["state"] == "modeled_complete"
    assert [r["hour_ms"] for r in summary["receipts"]] == [h1, h2]
    assert verified_cost(trade, summary) == pytest.approx(-.9)
    summary["receipts"][1]["signed_units"] = 30
    assert verified_cost(trade, summary) is None


def test_missing_future_and_conflicting_records_are_not_verified_zero(tmp_path):
    ledger = FundingLedger(tmp_path/"funding.sqlite3")
    hour = millis(NOW)//HOUR*HOUR
    trade = {"coin": "BTC", "side": "short", "size": 1, "leverage": 1, "opened_at": (NOW-timedelta(minutes=30)).isoformat()}
    settlement(ledger, hour, 0, 100)
    assert ledger.position(trade, now=NOW)["known_cost"] == 0
    assert ledger.position(trade, now=NOW-timedelta(minutes=4, seconds=55))["state"] == "incomplete"
    ledger.record(observed_ms=hour+30_000, rates=[{"coin": "BTC", "time": hour+10, "fundingRate": .001}])
    missing = ledger.position(trade, now=NOW)
    assert missing["known_cost"] is None and missing["missing_hours"] == [hour]


def test_collector_captures_the_oracle_before_hour_then_fetches_published_rate(tmp_path):
    path = tmp_path/"bot.db"
    before = NOW.replace(hour=8, minute=59, second=20)
    hour = millis(NOW)//HOUR*HOUR
    opened = (before-timedelta(minutes=20)).isoformat()
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE paper_trades(coin,opened_at,metadata,status)")
        conn.execute("INSERT INTO paper_trades VALUES ('BTC',?,?,'open')", (opened, json.dumps({"paper_funding_model": MODEL})))
    requests = []
    def send(req):
        requests.append(req["type"])
        if req["type"] == "metaAndAssetCtxs":
            return [{"universe": [{"name": "BTC"}]}, [{"oraclePx": "100", "funding": ".005", "markPx": "90"}]]
        return [{"coin": "BTC", "time": hour+10, "fundingRate": ".001"}]
    capture(path, clock=lambda: before, send=send)
    capture(path, clock=lambda: NOW, send=send)
    capture(path, clock=lambda: NOW, send=send)
    assert requests == ["metaAndAssetCtxs", "fundingHistory"]
    trade = {"opened_at": opened, "coin": "BTC", "side": "short", "size": 2, "leverage": 1}
    result = FundingLedger(ledger_path(path)).position(trade, now=NOW)
    assert result["known_cost"] == pytest.approx(-.2)
    assert result["receipts"][0]["oracle_price"] == 100


def test_actual_paper_close_with_restart_proof_credits_once(tmp_path, monkeypatch):
    path = tmp_path/"bot.db"
    monkeypatch.setattr(config, "DB_PATH", str(path))
    monkeypatch.setattr(config, "DB_BACKEND", "sqlite")
    monkeypatch.setattr(config, "PAPER_FUNDING_LEDGER_ENABLED", True)
    monkeypatch.setattr(config, "PAPER_TRADING_FUNDING_ENABLED", True)
    monkeypatch.setattr(config, "PAPER_PORTFOLIO_GUARD_ENABLED", False)
    monkeypatch.setattr(config, "PAPER_TRADING_TAKER_FEE_BPS", 0)
    monkeypatch.setattr(PaperTrader, "_sync_fee_tier", lambda self: None)
    monkeypatch.setattr(PaperTrader, "_apply_slippage", staticmethod(lambda price, *a, **k: price))
    monkeypatch.setattr("src.core.clock_provider.utc_now", lambda: NOW-timedelta(hours=2))
    db.init_db()
    db.init_paper_account(1000.)
    trade_id = db.open_paper_trade(None, "BTC", "long", 100., 1., leverage=2., stop_loss=95.)
    ledger = FundingLedger(ledger_path(path))
    h2 = millis(NOW)//HOUR*HOUR
    settlement(ledger, h2-HOUR, .001, 100)
    settlement(ledger, h2, .002, 101)
    monkeypatch.setattr("src.core.clock_provider.utc_now", lambda: NOW)
    trader = PaperTrader()
    trade, = db.get_open_paper_trades()
    assert trader._close_trade(trade, 105., "take_profit")["pnl"] == 9.4
    assert PaperTrader()._close_trade(trade, 105., "take_profit") is None
    with db.get_connection(for_read=True) as conn:
        closed = dict(conn.execute("SELECT * FROM paper_trades WHERE id=?", (trade_id,)).fetchone())
    metadata = json.loads(closed["metadata"])
    assert verified_cost(closed, metadata["funding_coverage"]) == pytest.approx(.604)
    assert db.get_paper_account()["balance"] == pytest.approx(1009.4)
    report = funding_health(path, NOW+timedelta(seconds=1))
    assert report["cross_hour_closes"] == report["verified_closes"] == 1
    assert report["state"] == "Funded closes verified"


def test_funding_health_distinguishes_not_due_pending_and_overdue(tmp_path):
    path = tmp_path / "bot.db"
    hour = NOW.replace(minute=0)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE paper_trades(coin,side,size,leverage,opened_at,closed_at,metadata,status)")
        conn.execute("INSERT INTO paper_trades VALUES ('BTC','long',1,2,?,NULL,?,'open')",
                     ((hour-timedelta(minutes=15)).isoformat(), json.dumps({"paper_funding_model": MODEL})))
    assert funding_health(path, hour-timedelta(seconds=1))["open_due"] == 0
    assert funding_health(path, hour+timedelta(minutes=1))["pending_hours"] == 1
    report = funding_health(path, NOW)
    assert report["overdue_hours"] == 1 and report["verified_closes"] == 0
    settlement(FundingLedger(ledger_path(path)), millis(hour), 0, 100)
    report = funding_health(path, NOW)
    assert report["state"] == "Open settlements measured" and report["overdue_hours"] == 0


def test_rule_identity_ignores_deploy_commit_but_not_config_or_strategy_params(monkeypatch):
    monkeypatch.setenv("BOT_RUN_MODE", "paper")
    def identity(build, params):
        monkeypatch.setenv("BOT_BUILD_REVISION", build*40)
        meta = {"source_key": "strategy:scalping", "evidence_rule_inputs": params,
                "entry_decision": {"route": "strategy", "firewall": "approved"}}
        stamp_paper_entry(meta, 8)
        return meta["strategy_evidence"]
    first, second = identity("a", '{"a":2,"b":3}'), identity("b", {"b": 3, "a": 2})
    assert first["build_revision"] != second["build_revision"] and first["version"] == second["version"]
    assert second["version"] != identity("b", {"a": 5, "b": 3})["version"]
    monkeypatch.setattr(config, "PAPER_TRADING_TAKER_FEE_BPS", 99)
    assert second["version"] != identity("b", {"a": 2, "b": 3})["version"]


def test_shadow_freezes_before_outcomes_and_restarts_without_shifting_start(tmp_path):
    t = millis(NOW)
    calls = []
    def fetch(coin, first, end):
        calls.append((first, end))
        return {stamp: (100., 101., 99., 100., 30.) for stamp in range(first, end, BAR)}
    initial = cycle(tmp_path, now=t, fetch_bars=fetch)
    assert initial["results"] is None
    later = cycle(tmp_path, now=t+2*HOUR, fetch_bars=fetch)
    assert initial["start"] == later["start"]
    assert later["results"]["base"]["closed"] == 0 and later["results"]["base"]["passive"] < 0
    assert all(end <= t+2*HOUR for _, end in calls)
    warmup = [Bar(i*BAR, 100, 101, 99, 100, 30) for i in range(LOOKBACK)]
    signal_bar = Bar(LOOKBACK*BAR, 100, 110, 99, 102, 30)
    assert decisions(warmup + [signal_bar])[-1].side == 1
    assert decisions(warmup)[-1].side == 0
    with closing(sqlite3.connect(tmp_path/"market.sqlite3")) as conn, conn:
        conn.execute("UPDATE protocol SET fingerprint='different'")
    with pytest.raises(ValueError, match="frozen rule mismatch"):
        cycle(tmp_path, now=t+3*HOUR, fetch_bars=fetch)


def test_historical_order_lifecycle_deduplicates_only_identical_order_identities():
    cid = "0x"+"a"*32
    at = millis(NOW)
    order = {"oid": 42, "cloid": cid, "coin": "BTC", "side": "B", "origSz": ".2",
             "reduceOnly": False, "timestamp": at-5}
    indexed = venue_order_index([{"order": {**order, "sz": ".2"}, "status": "open"},
                                 {"order": {**order, "sz": "0"}, "status": "filled"}])
    batch = [{"coin": "BTC", "time_ms": at}]
    fills = [{"side": "B", "sz": ".2"}]
    decision = {"decision_id": "a", "coin": "BTC", "side": "long", "live_order_id": None,
                "created_at": (NOW-timedelta(seconds=1)).isoformat()}
    assert _match_venue_order("42", batch, fills, indexed["42"], {cid: [decision]}) == ("matched_full_entry", "a")
    assert _match_venue_order("42", batch, fills, indexed["42"], {}) == ("order_without_unique_decision", None)


def test_funding_receipts_deduplicate_and_do_not_accept_conflicts():
    receipt = {"time": 5, "hash": "one", "delta": {"coin": "BTC", "type": "funding", "usdc": ".25"}}
    assert actual_funding([receipt, receipt]) == {"receipt_rows": 1, "venue_net_cash_usdc": "0.25"}
    with pytest.raises(ValueError, match="conflicting"):
        actual_funding([receipt, {**receipt, "delta": {**receipt["delta"], "usdc": "-.25"}}])
    api = SimpleNamespace(call=lambda request: [{"time": request["startTime"]}])
    assert bounded(api.call, kind="userFunding", account="example", start=10, end=20) == [{"time": 10}]
