from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
import json
import sqlite3

import pytest

from scripts import run_stock_forward_shadow as runner
from scripts.publish_dashboard_status import stock_forward_status
from src.backtest.stock_shadow import COSTS, Frame, simulate
from src.trading.stock_signals import NY, completed_sma20


class Feed:
    now = datetime(2026, 11, 23, 20, tzinfo=UTC)
    stale = False
    missing_bars = False

    def __init__(self):
        self.calls = []

    def clock(self):
        return self.now

    def get(self, endpoint, **params):
        self.calls.append(endpoint)
        today = self.now.astimezone(NY).date()
        if endpoint == "calendar":
            days = date.fromisoformat(params["start"])
            return [{"date": (days+timedelta(days=i)).isoformat(), "open": "09:30", "close": "16:00"}
                    for i in range((date.fromisoformat(params["end"])-days).days+1)
                    if (days+timedelta(days=i)).weekday() < 5]
        if endpoint == "clock":
            return {"timestamp": self.now.isoformat(), "is_open": True,
                    "next_close": datetime.combine(today, time(16), NY).isoformat()}
        if endpoint == "bars":
            days = date.fromisoformat(params["start"])
            values = [{"t": datetime.combine(days+timedelta(days=i), time(0), NY).isoformat(), "c": "100"}
                      for i in range((today-days).days) if (days+timedelta(days=i)).weekday() < 5]
            values[-1]["c"] = "110"
            if self.missing_bars:
                values.pop()
            values += [{"t": datetime.combine(today, time(0), NY).isoformat(), "c": "1"},
                       {"t": "2027-12-31T05:00:00Z", "c": "9999"}]
            return {"bars": values, "next_page_token": None}
        assert endpoint == "quote"
        return {"quote": {"t": (self.now-timedelta(seconds=45 if self.stale else 1)).isoformat(),
                          "bp": "110.00", "ap": "110.04"}}


def test_forward_capture_freezes_and_never_uses_future_or_same_quote():
    db, feed = sqlite3.connect(":memory:"), Feed()
    report, _ = runner.cycle(db, feed, clock=feed.clock, digest="v1")
    assert report["start"] == "2026-11-24" and report["end"] == "2027-01-23"
    assert report["next_slot"] == "2026-11-24T14:45:00+00:00"
    assert report["results"] is None and feed.calls == ["calendar"]
    feed.now = datetime(2026, 11, 24, 14, 45, 15, tzinfo=UTC)
    report, trace = runner.cycle(db, feed, clock=feed.clock, digest="v1")
    assert report["samples_captured"] == 1 and not trace
    data = db.execute("SELECT entry,exit,inputs FROM frames WHERE slot=0").fetchone()
    assert data[:2] == (1, 0)
    inputs = json.loads(data[2])
    assert len(inputs) == 21 and inputs[-1] == ["2026-11-23", "110"]
    assert "2026-11-24" not in data[2]
    feed.calls.clear()
    runner.cycle(db, feed, clock=feed.clock, digest="v1")
    assert feed.calls == []
    feed.now += timedelta(minutes=5)
    report, trace = runner.cycle(db, feed, clock=feed.clock, digest="v1")
    assert feed.calls == ["clock", "quote"] and report["paired"] == 1
    assert report["results"]["base"]["entries"] == 1 and report["results"]["base"]["closed"] == 0
    assert report["results"]["stress"]["net"] < report["results"]["base"]["net"] < 0
    assert trace["base"][0]["event"] == "buy"
    with pytest.raises(ValueError, match="protocol mismatch"):
        runner.cycle(db, feed, clock=feed.clock, digest="different-version")
    assert db.execute("SELECT COUNT(*) FROM frames").fetchone()[0] == 2


@pytest.mark.parametrize("invalid", ["stale", "late", "bars"])
def test_gaps_reported_and_are_never_repaired(invalid):
    db, feed = sqlite3.connect(":memory:"), Feed()
    runner.cycle(db, feed, clock=feed.clock)
    feed.now = datetime(2026, 11, 24, 14, 45, 15, tzinfo=UTC)
    if invalid == "stale":
        feed.stale = True
    elif invalid == "late":
        feed.now += timedelta(seconds=61)
    else:
        feed.missing_bars = True
    report, _ = runner.cycle(db, feed, clock=feed.clock)
    if invalid == "stale":
        assert report["market_status"] == "Market data not usable in this slot"
    if invalid == "late":
        assert feed.calls == ["calendar"]
    feed.stale = False
    feed.missing_bars = False
    feed.now = datetime(2026, 11, 24, 14, 50, 15, tzinfo=UTC)
    report, trace = runner.cycle(db, feed, clock=feed.clock)
    assert not trace and report["results"] is None and report["paired"] == 0
    assert report["samples_due"] == 2 and report["samples_captured"] == (2 if invalid == "bars" else 1)


def test_study_trace_costs_and_fresh_fill_constraint():
    def frame(t, bid, entry=None, exit=None):
        return Frame(t-1, t, Decimal(bid), Decimal(bid)+Decimal(".10"), entry, exit)

    first = frame(60, "90", True, False)
    days = [("missed", first, frame(61, "100")),
            ("cross", first, frame(360, "100")),
            ("below", frame(420, "110", False, True), frame(720, "120"))]
    base, trades = simulate(days, COSTS["base"])
    stress, _ = simulate(days, COSTS["stress"])
    assert [trade["session"] for trade in trades] == ["cross", "below"]
    assert base["missed_signal_slots"] == 1 and not base["open"]
    assert Decimal(trades[0]["price"]) > 100 and Decimal(trades[1]["price"]) < 120
    assert base["entries"] == base["closed"] == 1
    assert base["net"] == base["realized"] > stress["net"] > 0
    assert base["net_without_largest"] == base["cash"] == 0
    assert base["passive"] == base["net"]


def test_completed_closes_reject_duplicates_and_insufficient_history():
    day = date(2026, 11, 24)
    sessions = [(day-timedelta(days=i)).isoformat() for i in range(21, 0, -1)]
    bars = [{"t": session+"T05:00:00Z", "c": 100} for session in sessions]
    bars[-1]["c"] = 90
    assert completed_sma20(bars, sessions, str(day)) == (False, True)
    assert completed_sma20(bars+[bars[0]], sessions, str(day)) is None
    assert completed_sma20(bars[:-1], sessions, str(day)) is None


def test_allowlisted_dashboard_reports_dont_confuse_gaps_with_profit(tmp_path):
    db, feed = sqlite3.connect(":memory:"), Feed()
    path = tmp_path / "latest.json"
    report, _ = runner.cycle(db, feed, clock=feed.clock)
    report["account"] = "private"
    runner.atomic_json(path, report)
    dashboard = stock_forward_status(path, feed.now, 20)
    assert "account" not in dashboard and dashboard["monthly_ops"] == 20
    assert dashboard["verdict"] == "Insufficient evidence" and dashboard["operating_cost"] is None
    report["paired"] = 22
    runner.atomic_json(path, report)
    assert stock_forward_status(path, feed.now, 20) is None


def test_cost_screen_needs_final_coverage_sample_and_robust_net(tmp_path):
    path = tmp_path / "latest.json"
    now = datetime(2027, 1, 25, 19, tzinfo=UTC)
    row = {"closed": 10, "entries": 10, "net": 75, "cash": 0, "passive": 10,
           "net_without_largest": 45, "drawdown_pct": 1}
    report = {"rule": runner.RULE, "state": "Window complete", "published_at": now.isoformat(),
              "start": "2026-11-24", "end": "2027-01-23", "capital": 1000, "sleeve": 75,
              "live_approval": False, "sessions_planned": 40, "sessions_due": 40, "paired": 37,
              "samples_due": 80, "samples_captured": 76, "final_pair": True,
              "results": {"base": row, "stress": row}}
    runner.atomic_json(path, report)
    assert stock_forward_status(path, now, None)["verdict"] == "Insufficient evidence"
    status = stock_forward_status(path, now, 20)
    assert status["operating_cost"] == pytest.approx(39.43, abs=.01)
    assert status["verdict"] == "Did not clear the cost and benchmark screen"
    row["net_without_largest"] = 55
    runner.atomic_json(path, report)
    assert stock_forward_status(path, now, 20)["verdict"] == "Initial screen passed; research only"
    assert stock_forward_status(path, now+timedelta(hours=73), 20)["verdict"] == "Insufficient evidence"
    report["final_pair"] = False
    runner.atomic_json(path, report)
    assert stock_forward_status(path, now, 20)["verdict"] == "Insufficient evidence"


def test_forward_runner_has_no_order_endpoint_or_non_get_request(monkeypatch):
    feed = runner.QuoteFeed("dummy-key", "dummy-secret")

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self, size):
            assert size == runner.MAX_REPLY+1
            return b"{}"

    def open_get(request, *, timeout):
        assert request.get_method() == "GET" and timeout == 8
        assert request.full_url == "https://data.alpaca.markets/v2/stocks/SPY/quotes/latest?feed=iex"
        return Reply()

    monkeypatch.setattr(feed.opener, "open", open_get)
    assert feed.get("quote", feed="iex") == {}
    with pytest.raises(KeyError):
        feed.get("orders")
    assert runner.NoRedirects().redirect_request(None, None, 302, None, None, "https://example.com") is None
