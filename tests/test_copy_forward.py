from datetime import UTC, datetime
from decimal import Decimal
import sqlite3

from scripts import run_copy_forward as capture
from scripts.publish_dashboard_status import copy_forward_status
from src.backtest.copy_forward import COSTS, SYMBOLS, Candidate, Quote, paired


def level(bid, ask, depth="8"):
    return Quote(Decimal(bid), Decimal(ask), Decimal(depth), Decimal(depth), 0, 0)


def test_copy_toggle_reallocates_the_same_slot_and_prices_each_side():
    pool = [Candidate("ETH","long","copy_trade",.85,"a","copy"),
            Candidate("ETH","short","strategy",.80,"b","ordinary")]
    bid, ask = {"ETH":level("999", "1000")}, {"ETH":level("1100", "1101")}
    base, receipts = paired([(0,pool,bid,ask)], COSTS["base"])
    stress, _ = paired([(0,pool,bid,ask)], COSTS["stress"])
    assert base["different_hours"] == base["scored_hours"] == 1
    assert base["copy_on"]["copy_closes"] == 1 and base["copy_off"]["copy_closes"] == 0
    assert base["copy_on"]["net"] > stress["copy_on"]["net"] > 0
    assert stress["copy_off"]["net"] < base["copy_off"]["net"] < 0
    assert [r["ref"] for r in receipts] == ["copy","ordinary"]
    missing, trace = paired([(0,pool,bid,None),(3600,pool,bid,{"ETH":level("1100","1101",".00001")})], COSTS["base"])
    assert missing["unscored_hours"] == 2 and not trace


def test_capture_is_forward_only_frozen_and_skips_late_input(tmp_path):
    db = sqlite3.connect(":memory:")
    released = int(datetime(2026,11,26,20,tzinfo=UTC).timestamp())
    now = released+100
    start = released+3600
    count = []

    def clock():
        return datetime.fromtimestamp(now, UTC)

    def book(coin):
        price = {"BTC":"80000","ETH":"1500","SOL":"100"}[coin]
        return {"coin":coin,"time":(now-1)*1000,"levels":[[{"px":price,"sz":"8"}],[{"px":price,"sz":"8"}]]}

    def candidates(path, hour, seen):
        assert hour == start and hour+600 <= seen.timestamp() < hour+660
        count.append(hour)
        return [Candidate("SOL","long","copy_trade",.85,clock().isoformat(),"one")]

    def step():
        return capture.cycle(db, tmp_path, clock=clock, book=book, candidate_feed=candidates)

    assert step()[0]["results"] is None and count == []
    now = start+610
    step()
    step()
    assert count == [start]
    now = start+910
    assert step()[0]["results"] is None
    now = start+2410
    report, trace = step()
    assert report["due_hours"] == report["capture_hours"] == 1
    assert report["results"]["base"]["copy_on"]["closes"] == 1
    assert report["results"]["base"]["copy_off"]["closes"] == 0
    assert trace["base"][0]["ref"] == "one"
    path = tmp_path/"latest.json"
    capture.atomic_json(path, report)
    assert copy_forward_status(path, clock())["verdict"] == "Insufficient evidence"
    now = start+3600+2500
    report, _ = step()
    assert report["due_hours"] == 2 and report["capture_hours"] == 1
    assert report["results"]["base"]["unscored_hours"] == 1
    assert len(count) == 1


def test_small_readonly_candidate_feed_checks_input_times_and_sources(tmp_path):
    path = tmp_path/"journal.db"
    hour = int(datetime(2026,11,26,20,tzinfo=UTC).timestamp())
    now = datetime.fromtimestamp(hour+610, UTC)
    stamps = [capture.iso(hour+x) for x in (20,30,40,600)]
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE decision_snapshots (
            decision_id TEXT,coin TEXT,side TEXT,source TEXT,raw_confidence REAL,created_at TEXT,signal_timestamp TEXT)""")
        db.executemany("INSERT INTO decision_snapshots VALUES(?,?,?,?,?,?,?)", [
            ("first","ETH","long","copy_trade",.85,stamps[0],stamps[0]),
            ("future","ETH","long","strategy",.8,stamps[1],stamps[2]),
            ("unrelated","ETH","long","unverified",.8,stamps[1],stamps[1]),
            ("cutoff","ETH","long","strategy",.8,stamps[3],stamps[3]),
        ])
    data = capture.read_candidates(path,hour,now)
    assert len(data) == 1 and data[0].coin in SYMBOLS and data[0].ref != "first"
