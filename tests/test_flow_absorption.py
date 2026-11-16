from contextlib import closing
from dataclasses import replace
import sqlite3

import pytest

from scripts.score_hyperliquid_absorption import build
from src.backtest.absorption_study import pairs_at_cutoff, round_trip, same_funding_hour
from src.data.absorption_feed import Capture, storage
from src.signals.flow_absorption import Frame, HISTORY, MINUTE_MS, SLOT_MS, absorption


def frame(i, *, mid=100, buy=10, sell=10, bid=70, ask=30):
    return Frame(i * SLOT_MS, "session", "ready", mid - .005, mid + .005, 1_000, 1_000,
                 bid, ask, buy, sell, 3)


@pytest.mark.parametrize("side", [1, -1])
def test_absorption_uses_observed_rejection_and_supporting_book(side):
    frames = [frame(i) for i in range(HISTORY)]
    for i in range(HISTORY - 12, HISTORY):
        frames[i] = frame(i, mid=100 - side * .035 if i < HISTORY - 1 else 100,
                          buy=10 if side == 1 else 100, sell=100 if side == 1 else 10,
                          bid=70 if side == 1 else 30, ask=30 if side == 1 else 70)
    outcome = absorption(frames)
    assert outcome["state"] == "absorption" and outcome["side"] == side and outcome["pressure"] == -side
    assert outcome["ratio"] > 2 and outcome["rebound_bps"] > 1
    assert absorption(frames[:-1]) == {"state": "warming_up", "side": 0, "pressure": 0}
    assert absorption([replace(frames[0], session="old"), *frames[1:]])["state"] == "coverage_gap"
    assert absorption([*frames[:-1], replace(frames[-1], bid_depth=50, ask_depth=50)])["state"] == "pressure_without_absorption"


def raw_book(coin, at):
    return {"channel": "l2Book", "data": {"coin": coin, "time": at - 5,
            "levels": [[{"px": str(99.99 - i * .01), "sz": "2"} for i in range(5)],
                       [{"px": str(100.01 + i * .01), "sz": "2"} for i in range(5)]]}}


def raw_trade(coin, at, tid, *, side="A", stamp=None):
    row = {"coin": coin, "time": at - 5 if stamp is None else stamp, "tid": tid, "side": side,
           "px": "99.99" if side == "A" else "100.01", "sz": "2", "users": ["do-not-store-wallet"]}
    return {"channel": "trades", "data": [row]}


def test_capture_uses_receipt_cutoffs_and_discards_duplicate_or_stale_events(tmp_path):
    with closing(storage(tmp_path / "flow.db")) as db:
        capture = Capture(db, 500)
        capture.advance(5000)
        capture.ingest(raw_book("BTC", 9_900), 9_900)
        before = raw_trade("BTC", 9_980, 1)
        capture.ingest(before, 9_980)
        capture.ingest(before, 9_981)
        boundary = raw_trade("BTC", 10_000, 2, side="B")
        capture.ingest(boundary, 10_000)  # Not readable until the next slot.
        capture.ingest(raw_book("BTC", 14_900), 14_900)
        capture.advance(15_000)
        capture.ingest(raw_book("BTC", 19_900), 19_900)
        capture.ingest(raw_trade("BTC", 19_999, 3, stamp=20_050), 19_999)
        capture.advance(20_000)
        rows = db.execute("SELECT at,state,buy,sell,trades FROM frames WHERE coin='BTC' ORDER BY at").fetchall()
        assert rows[1] == (10_000, "ready", 0., 199.98, 1)
        assert rows[2] == (15_000, "ready", 200.02, 0., 1)
        assert rows[3] == (20_000, "bad_message", 0., 0., 0)
        assert "users" not in {r[1] for r in db.execute("PRAGMA table_info(frames)")}


def event(n, state="pressure_without_absorption"):
    return {"at": n * MINUTE_MS, "coin": "BTC", "state": state, "pressure": -1, "side": 1,
            "session": "session", "features": {"ratio": 2.5, "flow": -.7}}


def test_control_selection_cannot_read_unmatured_or_later_candidates():
    target = event(15, "absorption")
    history = [event(7), event(12), target]
    data, _ = pairs_at_cutoff(history, 0, 30 * MINUTE_MS, 5)
    assert data == [(target, event(7))]
    assert pairs_at_cutoff([*history, event(16), event(20)], 0, 30 * MINUTE_MS, 5)[0] == data
    assert not same_funding_hour(55 * MINUTE_MS, 5)
    assert same_funding_hour(54 * MINUTE_MS, 5)


def test_prices_cross_the_spread_pay_fees_and_refuse_quote_gaps(tmp_path):
    with closing(storage(tmp_path / "flow.db")) as db, db:
        db.row_factory = sqlite3.Row
        db.executemany("INSERT INTO frames VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [
            ("BTC", i * SLOT_MS, "session", "ready", 99.99, 100.01, 1000, 1000, 2000, 2000,
             0, 0, 0, i * SLOT_MS - 50, i * SLOT_MS - 30, None, None, 0, 0)
            for i in range(62)
        ])
        signal = event(0, "absorption")
        value, issue = round_trip(db, signal, 1, 5)
        assert issue is None and value[0] == pytest.approx(-13, abs=.01) and value[1] == pytest.approx(-19, abs=.03)
        db.execute("DELETE FROM frames WHERE at=?", (10 * SLOT_MS,))
        assert round_trip(db, signal, 1, 5) == (None, "feed_gap")


def test_new_archive_reports_unmeasured_not_zero(tmp_path):
    path = tmp_path / "flow.db"
    storage(path).close()
    report = build(path)
    assert "Insufficient forward evidence" in report and "Unmeasured" in report and "no signals yet" in report
