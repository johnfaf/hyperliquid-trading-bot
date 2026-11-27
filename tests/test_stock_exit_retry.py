from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from http.client import responses
import sqlite3
from urllib.error import HTTPError

import pytest

from scripts import run_stock_exit_retry_holdout as runner
from scripts.run_stock_orb_replay import utc
from src.backtest.stock_exit_retry import (
    RETRY_OFFSETS,
    capture_slots,
    paired_episodes,
    retry_exit,
)
from src.backtest.stock_pullback import Day, Setup

D = Decimal


def session(i, value):
    opened = 1700000000 + i * 86400
    return Day(
        str(date(2025, 1, 2) + timedelta(days=i)),
        opened,
        opened + 23400,
        D(value) - 1,
        D(value),
    )


def quote(slot, bid, ask, *, at=24, size=6):
    return {"t": utc(slot + at), "bp": bid, "ap": ask, "bs": size, "as": size}


def episode():
    days = [session(i, p) for i, p in enumerate((100, 94, 93, 100))]
    setup = Setup(days[0].day, 1, D(100), D(95), D("112.5"))
    return days, {1: setup}


def test_original_policy_keeps_priority_and_identical_entry():
    days, plans = episode()
    entry, exit_slot = days[1].slot, days[2].slot
    windows = {
        entry: [quote(entry, "99.99", "100")],
        exit_slot: [quote(exit_slot, "92.99", "93")],
        exit_slot + 60: [quote(exit_slot + 60, "110", "110.01")],
    }
    summary, [trace] = paired_episodes(days, plans, windows, "base")
    assert trace["entry"]["quote_at"] == entry + 24
    assert trace["strict"]["quote_at"] == exit_slot + 24
    assert trace["exit"] == {"state": "primary_filled"}
    assert summary["paired_entries"] == summary["primary_fills"] == 1
    assert not summary["retry_fills"] and not summary["alert_events"]
    assert capture_slots(days, plans) == {
        entry: days[1].day,
        **{exit_slot + shift: days[2].day for shift in (0, *RETRY_OFFSETS)},
    }


def test_fallback_honors_original_reference_and_chooses_first_eligible_retry():
    days, plans = episode()
    entry, slot = days[1].slot, days[2].slot
    windows = {
        entry: [quote(entry, "99.99", "100")],
        slot: [quote(slot, "99.80", "100.20")],
        slot + 60: [
            quote(slot + 60, "99.80", "99.81", at=1),
            quote(slot + 60, "99.20", "99.22"),
        ],
        slot + 180: [
            quote(slot + 180, "99.70", "99.72"),
            quote(slot + 180, "100", "100.01", at=29),
        ],
        slot + 300: [quote(slot + 300, "115", "115.01")],
    }
    summary, [trace] = paired_episodes(days, plans, windows, "base")
    result = trace["exit"]
    assert trace["strict"]["state"] == "unresolved_exit"
    assert result["state"] == "retry_filled"
    assert (
        result["reference_source"] == "primary_midpoint"
        and D(result["reference"]) == 100
    )
    assert result["quote_at"] == slot + 180 + 24
    assert result["attempts"][0]["over_cost_cap"] == 1
    assert len(result["attempts"]) == 2
    assert D(result["cost_bps"]) == (100 - D(result["net_per_share"])) * 100
    assert summary["strict_unresolved"] == summary["retry_fills"] == 1
    assert summary["retry_unresolved"] == 0


def test_delay_spread_depth_and_extra_gap_cannot_be_bypassed():
    days, _ = episode()
    slot = days[2].slot
    windows = {
        slot: [quote(slot, "99.80", "100.20")],
        slot + 60: [
            quote(slot + 60, "99.99", "100", at=19),
            quote(slot + 60, "99.99", "100", size=4),
            quote(slot + 60, "99.99", "100.12"),
            quote(slot + 60, "99.80", "99.81"),
        ],
        slot + 180: [quote(slot + 180, "99.99", "100", at=31)],
        slot + 300: [quote(slot + 300, "99.99", "100", at=29)],
    }
    answer = retry_exit(days[1], days[2], windows, "combined")
    assert answer["quote_at"] == slot + 329
    assert (
        answer["attempts"][0]["eligible_quotes"]
        == answer["attempts"][0]["over_cost_cap"]
        == 1
    )
    assert answer["attempts"][1]["eligible_quotes"] == 0
    assert 53 < D(answer["cost_bps"]) <= 65


@pytest.mark.parametrize("scenario", ("base", "exit_outage"))
def test_missing_reference_never_reanchors_to_a_later_bid_and_emits_one_alert(scenario):
    days, plans = episode()
    entry, slot = days[1].slot, days[2].slot
    windows = {
        entry: [quote(entry, "99.99", "100")],
        slot + 60: [quote(slot + 60, "90.99", "91")],
        slot + 180: [quote(slot + 180, "90.98", "90.99")],
        slot + 300: [quote(slot + 300, "90.95", "90.96")],
    }
    summary, [trace] = paired_episodes(days, plans, windows, scenario)
    result = trace["exit"]
    assert (
        result["state"] == "still_open"
        and result["reference_source"] == "trigger_close"
    )
    assert D(result["reference"]) == 94
    assert result["alert"]["timestamp"] == slot + 330
    assert result["alert"]["kind"] == "exit_exposure_still_open"
    assert result["alert"]["qty"] == trace["entry"]["qty"]
    assert (
        summary["strict_unresolved"]
        == summary["retry_unresolved"]
        == summary["alert_events"]
        == 1
    )
    if scenario == "exit_outage":
        assert [a["capture"] for a in result["attempts"]] == ["forced_outage"] * 3
    else:
        assert [a["over_cost_cap"] for a in result["attempts"]] == [1, 1, 1]


def test_missing_primary_can_recover_using_known_trigger_close():
    days, _ = episode()
    slot = days[2].slot
    result = retry_exit(
        days[1], days[2], {slot + 180: [quote(slot + 180, "93.99", "94")]}, "base"
    )
    assert result["state"] == "retry_filled"
    assert result["reference_source"] == "trigger_close"
    assert result["attempts"][0]["capture"] == "missing"


def test_no_paired_exit_or_alert_is_fabricated_for_a_missed_entry():
    days, plans = episode()
    data, [trace] = paired_episodes(days, plans, {}, "base")
    assert trace["entry"]["state"] == "missed_entry" and "exit" not in trace
    assert not data["paired_entries"] and not data["alert_events"]


def test_protocol_freezes_configuration_before_collection(monkeypatch):
    with closing(sqlite3.connect(":memory:")) as db:
        now = datetime(2026, 11, 27, tzinfo=UTC)
        initial = runner.preregister(db, now)
        assert initial["quote_feeds"] == ["iex", "sip"]
        assert runner.preregister(db, now) == initial
        monkeypatch.setattr(runner, "fingerprint", lambda: "changed")
        with pytest.raises(ValueError, match="pinned"):
            runner.preregister(db, now)


def test_capture_is_feed_specific_paginated_and_does_not_accept_a_truncated_window():
    class Reader:
        def __init__(self, repeated):
            self.repeated = repeated
            self.calls = []

        def get(self, endpoint, **query):
            self.calls.append((endpoint, query))
            follow = "page_token" in query
            return {
                "quotes": [query.get("page_token")],
                "next_page_token": "p" if not follow or self.repeated else None,
            }

    reader = Reader(False)
    assert runner.capture_window(reader, "sip", 1000) == [None, "p"]
    assert all(
        endpoint == "quotes" and query["feed"] == "sip"
        for endpoint, query in reader.calls
    )
    assert reader.calls[-1][1]["start"] == utc(1002) and reader.calls[-1][1][
        "end"
    ] == utc(1030)
    with pytest.raises(ValueError, match="incomplete"):
        runner.capture_window(Reader(True), "sip", 1000)


def test_inaccessible_feed_does_not_report_a_zero_trade_result(monkeypatch):
    def capture(history, feed, slot):
        if feed == "sip":
            raise HTTPError(
                "https://data.alpaca.markets", 403, responses[403], {}, None
            )
        return []

    days, plans = episode()
    slots = capture_slots(days, plans)
    monkeypatch.setattr(runner, "capture_window", capture)
    with closing(sqlite3.connect(":memory:")) as db:
        now = datetime(2026, 11, 27, tzinfo=UTC)
        definition = runner.preregister(db, now)
        issues = dict(runner.collect(db, None, slots))
        result, _ = runner.evaluate(db, definition, days, plans, slots, issues, now)
    assert result["feeds"]["iex"]["captured"] == 5
    assert result["feeds"]["iex"]["results"]["base"]["paired_entries"] == 0
    assert result["feeds"]["sip"]["results"] == {}
    assert len(issues["sip"]) == 1 and len(result["feeds"]["sip"]["missing"]) == 5
    assert not result["paper_approval"] and not result["live_approval"]
