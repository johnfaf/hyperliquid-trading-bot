from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import sqlite3

import pytest

from scripts.import_hyperliquid_fills import import_fills
from scripts.reconcile_hyperliquid_execution import VENUE, match_execution, summarize_csv
from scripts.run_strategy_shadow import evaluate, window_ms
from src.backtest.candle_exit_replay import BAR_MS, Bar
from src.learning.strategy_evidence import scorecard, stamp_paper_entry
from src.trading.live_trader_helpers import make_cloid
from src.trading.order_journal import OrderJournal

NOW = datetime(2026, 11, 22, 10, tzinfo=timezone.utc)


def paper_meta(*, attempt="one", settings=None):
    return {"source_key": "strategy:scalping", "decision_id": attempt, "entry_decision": {
        "route": "strategy", "firewall": "approved",
    }, "evidence_rule_inputs": settings, "close_reason": "take_profit", "gross_pnl_before_fees": 1.,
        "total_fees_paid": .1, "total_slippage_cost": .02}


def create_paper(path):
    db = sqlite3.connect(path)
    db.executescript("""CREATE TABLE paper_trades(id INTEGER PRIMARY KEY, strategy_id, coin, side, status,
        opened_at,closed_at,pnl,entry_price,exit_price,size,leverage,metadata);
        CREATE TABLE decision_snapshots(decision_id PRIMARY KEY, paper_trade_id);
        CREATE TABLE agent_scores(total_signals, accuracy);
        INSERT INTO agent_scores VALUES (9999, 1.0);""")
    return db


def insert_outcome(db, entry, close, meta, *, pnl=.9, linked=True):
    pos = db.execute("""INSERT INTO paper_trades(strategy_id,coin,side,status,opened_at,closed_at,pnl,
        entry_price,exit_price,size,leverage,metadata) VALUES (1,'BTC','long','closed',?,?,?,100,101,1,1,?)""",
                     (entry.isoformat(), close.isoformat(), pnl, json.dumps(meta))).lastrowid
    if linked:
        db.execute("INSERT OR IGNORE INTO decision_snapshots VALUES (?,?)", (meta["decision_id"], pos))


@pytest.fixture
def paper_env(monkeypatch):
    monkeypatch.setenv("BOT_RUN_MODE", "paper")
    monkeypatch.setenv("BOT_BUILD_REVISION", "a"*40)


def test_distinct_rules_do_not_borrow_evidence_and_success_is_only_for_review(tmp_path, paper_env):
    path = tmp_path/"paper.db"
    with closing(create_paper(path)) as db, db:
        for i in range(32):
            meta = paper_meta(attempt=f"x{i}", settings={"period": 10 if i < 31 else 20})
            stamp_paper_entry(meta, 1)
            assert "evidence_rule_inputs" not in meta
            at = NOW-timedelta(days=1+i%8, minutes=30)
            insert_outcome(db, at, at+timedelta(minutes=5), meta)
    result = scorecard(path, now=NOW)
    assert [(r["verified"], r["state"]) for r in result["rows"]] == [
        (31, "Ready for human review"), (1, "Insufficient verified sample")]
    assert result["live_approved"] is False


def test_generated_signals_and_unverifiable_closes_cannot_pass(tmp_path, paper_env):
    path = tmp_path/"paper.db"
    at = NOW-timedelta(days=1, minutes=30)
    with closing(create_paper(path)) as db, db:
        legacy = paper_meta(attempt="legacy")
        insert_outcome(db, at, at+timedelta(minutes=2), legacy)
        for attempt, stop, patch, linked, pnl in [
            ("synthetic", 2, {"close_reason": "live_reconciled_closed"}, True, .9),
            ("missing", 2, {"total_slippage_cost": None}, True, .9),
            ("funding", 33, {}, True, .9),
            ("math", 2, {}, True, 1.01),
            ("unlink", 2, {}, False, .9),
            ("dup", 2, {}, True, .9),
            ("dup", 2, {}, False, .9),
            ("future", 1500, {}, True, .9),
        ]:
            meta = {**paper_meta(attempt=attempt), **patch}
            stamp_paper_entry(meta, 1)
            insert_outcome(db, at, at+timedelta(minutes=stop), meta, linked=linked, pnl=pnl)
    result = scorecard(path, now=NOW)
    cohort, unversioned = result["rows"]
    assert cohort["verified"] == 0 and cohort["closed"] == 7
    assert dict(cohort["excluded"]) == {"unverified_close": 1, "cost_evidence_missing": 1,
        "funding_coverage_unverified": 1, "cost_arithmetic_mismatch": 1, "missing_decision_link": 1,
        "missing_or_reused_decision": 2}
    assert unversioned["state"] == "Rule version missing"


def test_csv_fee_is_not_subtracted_again_and_overlaps_stay_separate(tmp_path):
    header = "time,coin,dir,px,sz,ntl,fee,closedPnl\n"
    paths = [tmp_path/n for n in ("first.csv", "second.csv")]
    for path in paths:
        path.write_text(header+"05/04/2026 - 22:30:58,ETH,Open Short,100,1,100,0.1,-0.1\n", encoding="utf-8")
    result = summarize_csv(paths)
    assert result["cross_exports"] == 1 and result["exact_links_possible"] is False
    assert [r["reported_closed_pnl"] for r in result["exports"]] == ["-0.1", "-0.1"]
    assert [r["opening_rows_with_pnl_equal_negative_fee"] for r in result["exports"]] == [1, 1]


def test_exact_reconciliation_keeps_partial_and_unmatched_distinct(tmp_path):
    address = "0x"+"a"*40
    at = int(NOW.timestamp()*1000)
    venue, bot, book = (tmp_path/n for n in ("fills.db", "orders.db", "paper.db"))
    fill = {"time": at, "oid": 8, "tid": 1, "coin": "BTC", "side": "B", "px": "100",
            "sz": "0.2", "fee": "0.02", "closedPnl": "-0.02", "startPosition": "0", "feeToken": "USDC"}
    import_fills(venue, address, [fill, fill, {**fill, "oid": 9, "tid": 2}])
    cid = make_cloid("entry", "decision-7")
    journal = OrderJournal(bot, f"hyperliquid:{VENUE}:{address}")
    journal.reserve(cid, {"orders": [{"c": cid, "s": ".5", "r": False, "b": True}]}, entry=True)
    journal.observe(cid, "canceled", "8")
    with closing(sqlite3.connect(book)) as db, db:
        db.execute("CREATE TABLE decision_snapshots(decision_id,coin,side,created_at,live_order_id)")
        db.execute("INSERT INTO decision_snapshots VALUES ('decision-7','BTC','long',?,'8')",
                   ((NOW-timedelta(seconds=1)).isoformat(),))
    result = match_execution(venue, address, order_path=bot, decision_path=book)
    assert result["states"] == {"matched_partial_entry": 1, "no_journal_order": 1}
    assert result["fees_by_token"] == {"USDC": "0.04"}
    assert result["coverage"] == "unknown"
    with closing(sqlite3.connect(book)) as db, db:
        db.execute("UPDATE decision_snapshots SET side='short'")
    result = match_execution(venue, address, order_path=bot, decision_path=book)
    assert result["states"] == {"order_decision_fill_conflict": 1, "no_journal_order": 1}


def test_frozen_shadow_requires_warmup_and_never_reads_bars_beyond_window():
    start, _ = window_ms("date_holdout")
    bars = [Bar(t, 100., 101., 99., 100., 100.) for t in range(start-100*BAR_MS, start+6*BAR_MS, BAR_MS)]
    end = start+5*BAR_MS
    first = evaluate(bars, {}, start, end, lambda row: None)
    second = evaluate([*bars[:-1], Bar(bars[-1].t, 1000., 1001., 999., 1000., 300.)], {}, start, end, lambda row: None)
    assert first == second
    assert first["base"]["cash_net"] == 0 and first["base"]["passive_net"] < 0
    with pytest.raises(ValueError, match="warmup"):
        evaluate(bars[1:], {}, start, end, lambda row: None)
