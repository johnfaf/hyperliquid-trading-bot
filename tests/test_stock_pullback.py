from contextlib import closing
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
import sqlite3

import pytest

from scripts import run_stock_pullback_holdout as runner
from scripts.run_stock_orb_replay import NY, utc
from src.backtest.stock_pullback import (
    Day,
    Setup,
    proposals,
    quote_days,
    signal,
    simulate,
)


D = Decimal


def session(i, value):
    opened = 1700000000 + i * 86400
    label = str(date(2025, 9, 2) + timedelta(days=i))
    return Day(label, opened, opened + 390 * 60, D(value) - 1, D(value))


def quote(day, bid, ask, offset=24):
    return {"t": utc(day.slot + offset), "bp": bid, "ap": ask, "bs": 6, "as": 6}


def test_completed_reclaim_does_not_use_the_entry_session_bar():
    days = [session(i, D(100) + D(".1") * i) for i in range(78)]
    days[59] = replace(days[59], close=D("104.7"), low=D("102"))
    assert signal(days[:60]) is None
    setup = Setup(days[60].day, 1, D(106), D(102), D(116))
    assert signal(days[:61]) == setup
    assert proposals(days, days[60].day)[61] == setup
    changed = days[:61] + [replace(d, close=D(30), low=D(28)) for d in days[61:]]
    assert proposals(changed, days[60].day)[61] == setup
    assert 61 not in proposals(days, days[61].day)


def test_exit_uses_next_morning_bid_and_does_not_reuse_the_sleeve():
    days = [session(i, v) for i, v in enumerate((100, 98, 94, 93, 101))]
    days[1] = replace(days[1], low=D(90))
    setup = Setup(days[0].day, 1, D(100), D(95), D("112.5"))
    plans = dict.fromkeys((1, 2, 3), setup)
    quotes = {d.day: [quote(d, "99.99", "100")] for d in days}
    quotes[days[1].day].insert(0, quote(days[1], "97.99", "98", offset=1))
    quotes[days[3].day] = [quote(days[3], "92", "92.01")]
    assert quote_days(days, days[0].day, {1: setup}) == [0, 1, 3, 4]
    base, trace = simulate(days, days[0].day, plans, quotes, "base")
    stressed, _ = simulate(days, days[0].day, plans, quotes, "combined")
    assert [row["state"] for row in trace] == [
        "entered",
        "sleeve_unavailable",
        "closed",
        "sleeve_unavailable",
    ]
    assert trace[0]["quote_at"] == days[1].slot + 24
    assert D(trace[0]["price"]) * D(trace[0]["qty"]) * D("1.0002") <= 75
    assert trace[2]["day"] == days[3].day and trace[2]["reason"] == "stop"
    assert D(trace[2]["price"]) < 92
    assert base["sleeve_blocked"] == 2 and base["exposure_sessions"] == 2
    assert stressed["net"] < base["net"] < base["passive"] < 0


def test_tenth_completed_holding_session_schedules_a_next_session_exit():
    days = [session(i, 100) for i in range(13)]
    setup = Setup(days[0].day, 1, D(100), D(95), D("112.5"))
    plans = {1: setup}
    quotes = {d.day: [quote(d, "99.99", "100")] for d in days}
    assert quote_days(days, days[0].day, plans) == [0, 1, 11, 12]
    data, trace = simulate(days, days[0].day, plans, quotes, "base")
    assert trace[-1]["day"] == days[11].day and trace[-1]["reason"] == "time_limit"
    assert data["exposure_sessions"] == 10


@pytest.mark.parametrize("scenario", ["base", "exit_outage"])
def test_missing_exit_blocks_later_entries_and_withholds_net(scenario):
    days = [session(i, v) for i, v in enumerate((100, 94, 93, 100))]
    setup = Setup(days[0].day, 1, D(100), D(95), D("112.5"))
    plans = dict.fromkeys((1, 3), setup)
    quotes = {d.day: [quote(d, "99.99", "100")] for d in days}
    if scenario == "base":
        del quotes[days[2].day]
    data, trace = simulate(days, days[0].day, plans, quotes, scenario)
    assert data["net"] is None and data["without_largest"] is None
    assert data["unresolved"] == data["sleeve_blocked"] == 1
    assert [row["state"] for row in trace] == ["entered", "unresolved_exit"]


def test_preregistration_is_bound_to_dates_and_source(monkeypatch):
    warmup, start, end = date(2025, 5, 1), date(2025, 9, 2), date(2026, 6, 1)
    now = datetime(2026, 11, 27, 18, tzinfo=UTC)
    with closing(sqlite3.connect(":memory:")) as db:
        initial = runner.preregister(db, warmup, start, end, now)
        assert initial == runner.preregister(db, warmup, start, end, now)
        with pytest.raises(ValueError, match="already pinned"):
            runner.preregister(db, warmup, start, end - timedelta(days=1), now)
        monkeypatch.setattr(runner, "fingerprint", lambda: "unexpected")
        with pytest.raises(ValueError, match="already pinned"):
            runner.preregister(db, warmup, start, end, now)


def test_daily_data_requires_one_bar_per_session_and_the_same_price_basis():
    warmup, start, end = date(2025, 5, 1), date(2025, 9, 2), date(2026, 6, 1)
    daily = (warmup + timedelta(days=i) for i in range((end - warmup).days))
    dates = [d for d in daily if d.weekday() < 5]
    calendar = [{"date": str(d), "open": "09:30", "close": "16:00"} for d in dates]
    bars = [
        {
            "t": datetime.combine(d, time(0), NY).astimezone(UTC).isoformat(),
            "l": "98",
            "c": "100",
            "h": "101",
        }
        for d in dates
    ]
    data = {"calendar": calendar, "raw": bars, "split": [r.copy() for r in bars]}
    assert len(runner.read_market(data, warmup, start, end)) == len(dates)
    data["split"][-1]["c"] = "99"
    with pytest.raises(ValueError, match="prices differ"):
        runner.read_market(data, warmup, start, end)
    data["split"].pop()
    with pytest.raises(ValueError, match="missing daily bar"):
        runner.read_market(data, warmup, start, end)


def test_small_profitable_sample_is_not_eligible_for_promotion():
    days = [session(i, v) for i, v in enumerate((100, 115, 118, 100))]
    plan = Setup(days[0].day, 1, D(100), D(95), D("112.5"))
    quotes = {d.day: [quote(d, "99.99", "100")] for d in days}
    quotes[days[2].day] = [quote(days[2], "118.99", "119")]
    report, _ = runner.evaluate(
        {"start": days[0].day}, days, {1: plan}, quotes, range(4), datetime.now(UTC)
    )
    assert report["results"]["combined"]["net"] > 0
    assert report["results"]["combined"]["without_largest"] == 0
    assert not report["sufficient_sample"] and not report["passes_economics_screen"]
    assert not report["paper_approval"] and not report["live_approval"]
