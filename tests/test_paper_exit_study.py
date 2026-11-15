import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts import paper_exit_experiment as cli
from src.backtest.paper_exit_study import (
    HOUR_MS, Entry, Frame, build_frames, evaluate_exit,
)
from src.data.paper_exit_feed import MarketSample, PaperExitFeed


E = Entry(1, 0, 100, 0.75, 98, 104)


def sample(seconds, mark=100, **overrides):
    return replace(MarketSample(
        seconds * 1000, "ready", mark=mark, bid=mark - 0.02, ask=mark + 0.02, bid_size=5,
        book_age_ms=0, context_age_ms=0, funding_rate=0.0001, oracle_price=100,
    ), **overrides)


def frames(*rows):
    return [Frame(row, 0.25, 1.0, 5.0) for row in rows]


def test_trail_tightens_after_fee_adjusted_gain_then_fills_at_bid_less_impact():
    rows = frames(
        sample(5, 102.07), sample(10, 102.07), sample(15, 102.07),
        sample(20, 103), sample(25, 102.8), sample(30, 102.3),
    )
    fixed = evaluate_exit(E, rows, 5_000, trailing=False)
    result = evaluate_exit(E, rows, 5_000, trailing=True)
    assert fixed.exit_ms is None
    assert result.trail_changes[0]["at_ms"] == 20_000  # Earlier move didn't clear 1R after costs.
    assert [r["new_stop"] for r in result.trail_changes] == [102.5]
    assert result.reason == "trailing_stop"
    assert result.exit_ms == 30_000
    assert result.exit_price == pytest.approx((102.3 - 0.02) * 0.9995)
    assert result.profit_giveback > 0


def test_fixed_sample_rate_can_miss_a_fast_move():
    rows = frames(sample(5, 97.5), *(sample(sec, 101) for sec in range(10, 61, 5)))
    faster = evaluate_exit(E, rows, 5_000, trailing=False)
    slower = evaluate_exit(E, rows, 60_000, trailing=False)
    assert faster.reason == "hard_stop"
    assert faster.exit_price < E.hard_stop
    assert slower.exit_ms is None


def test_wide_book_or_three_copies_of_same_book_do_not_tighten_stop():
    rows = frames(
        sample(5, 103), sample(10, 103, book_age_ms=5_000), sample(15, 103, book_age_ms=10_000),
        sample(20, 103, bid=102, ask=104), sample(25, 98),
    )
    result = evaluate_exit(E, rows, 5_000, trailing=True)
    assert result.trail_changes == []
    assert result.reason == "hard_stop"


def test_stale_price_cannot_trigger_and_thin_bid_keeps_exit_pending_until_recovery():
    rows = frames(
        sample(5, 97, status="book_stale"), sample(10, 97, bid_size=0.1),
        sample(15, 97, status="disconnected"), sample(30, 101),
    )
    result = evaluate_exit(E, rows, 5_000, trailing=True)
    assert result.trigger_ms == 10_000
    assert result.exit_ms == 30_000
    assert result.reason == "hard_stop"
    assert result.missing_samples == 2 and result.material_gap is True
    assert result.trail_changes == []


def test_pre_entry_quote_cannot_trigger_a_stop_even_if_its_age_is_within_limit():
    entry = replace(E, at_ms=3_000)
    rows = frames(sample(10, 97, book_age_ms=8_000), sample(15, 100))
    result = evaluate_exit(entry, rows, 5_000, trailing=False)
    assert result.exit_ms is None and result.missing_samples == 1


def test_funding_charged_once_and_missing_payment_stays_unknown():
    entry = replace(E, at_ms=HOUR_MS - 10_000)
    settles = frames(sample(3595), sample(3600), sample(3605), sample(3610, 105))
    result = evaluate_exit(entry, settles, 5_000, trailing=False)
    assert result.estimated_funding == pytest.approx(0.75 * 100 * 0.0001)
    assert result.unknown_funding_hours == 0
    bad = frames(sample(3595), sample(3600, status="disconnected", funding_rate=None), sample(3605, 105))
    unknown = evaluate_exit(entry, bad, 5_000, trailing=False)
    assert unknown.unknown_funding_hours == 1
    assert unknown.estimated_funding == 0


def test_atr_excludes_unclosed_buckets_and_resets_on_gaps():
    rows = [sample(sec, 100 + (sec % 60) / 60) for sec in range(0, 15 * 60 + 20, 5)]
    built = build_frames(rows)
    assert built[179].atr is None
    assert built[180].atr is not None
    assert built[180].one_minute_bps == pytest.approx(0)
    assert built[180].five_minute_bps == pytest.approx(0)
    assert built[181].atr == built[180].atr
    broken = build_frames(rows[:181] + [sample(15 * 60 + 15)])
    assert broken[-1].atr is None


def test_report_separates_latency_from_trailing_and_excludes_incomplete_cohorts():
    samples = [sample(sec, 105 if sec >= 60 else 100) for sec in range(5, 365, 5)]
    incomplete = replace(E, id=2, at_ms=400_000)
    report = cli.build_report([E, incomplete], samples, 0, 500_000)
    assert report["paired_exits_without_material_gaps_or_missing_funding"] == 1
    assert report["excluded_from_aggregate"] == 1
    assert report["paired_usd_deltas"] == {"faster_fixed_vs_minute": 0, "trailing_vs_faster_fixed": 0}
    no_evidence = cli.build_report([E], [], 0, 20_000)
    assert no_evidence["arms"]["trailing_5s"]["modeled_pnl_usd"] is None
    assert no_evidence["market_samples"] == {}


def test_loads_pilot_entries_read_only_and_refuses_unknown_market_database(tmp_path):
    path = tmp_path / "pilot.db"
    with closing(sqlite3.connect(path)) as conn:
        with conn:
            conn.execute("CREATE TABLE bot_state (key, value)")
            conn.execute("INSERT INTO bot_state VALUES (?, ?)", (cli.STATE_KEY, json.dumps(cli.SPEC)))
            conn.execute("CREATE TABLE paper_trades (id, opened_at, coin, side, leverage, entry_price, size, take_profit, metadata)")
            conn.execute(
                "INSERT INTO paper_trades VALUES (1, '1970-01-01T00:00:00.500000+00:00', 'BTC', 'long', 1, 100, 0.75, 104, ?)",
                (json.dumps({"source_key": cli.STRATEGY, "initial_stop_loss": 98}),),
            )
    original = path.read_bytes()
    assert cli.load_entries(path, 500, 501) == [replace(E, at_ms=500)]
    assert cli.load_entries(path, 0, 500) == []
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="dedicated"):
        cli._market_path(path)
    with closing(sqlite3.connect(path)) as conn, pytest.raises(ValueError, match="refusing"):
        cli._init_market(conn)
    assert path.read_bytes() == original


def test_feed_keeps_book_exchange_time_and_invalidates_on_reconnect():
    clock = [10_000]
    feed = PaperExitFeed("BTC", lambda: clock[0])
    feed.on_book({"data": {"coin": "BTC", "time": 9_000, "levels": [
        [{"px": "100", "sz": "2"}], [{"px": "101", "sz": "1"}],
    ]}})
    feed.on_context({"data": {"coin": "BTC", "ctx": {"markPx": 100.25, "funding": "-0.0001", "openInterest": "42"}}})
    ready = feed.snapshot(True)
    assert ready.ready and ready.book_age_ms == 1_000 and ready.depth_imbalance > 0
    assert ready.funding_rate == -0.0001 and ready.open_interest == 42
    clock[0] += 10_000
    assert not feed.snapshot(True).ready
    feed.reset()
    assert feed.snapshot(True).mark is None and feed.snapshot(True).bid is None
    assert feed.snapshot(False).status == "disconnected"


def test_fresh_top_quote_does_not_make_stale_depth_look_fresh():
    clock = [10_000]
    feed = PaperExitFeed("BTC", lambda: clock[0])
    feed.on_book({"data": {"coin": "BTC", "time": 9_000, "levels": [
        [{"px": "100", "sz": "2"}], [{"px": "101", "sz": "1"}],
    ]}})
    clock[0] = 20_000
    feed.on_context({"data": {"coin": "BTC", "ctx": {"markPx": 100.25}}})
    feed.on_bbo({"data": {"coin": "BTC", "time": 19_500, "bbo": [
        {"px": "100.2", "sz": "1.3"}, {"px": "100.3", "sz": "2"},
    ]}})
    ready = feed.snapshot(True)
    assert ready.ready and ready.quote_source == "bbo"
    assert ready.bid == 100.2 and ready.bid_size == 1.3
    assert ready.book_age_ms == 500
    assert ready.depth_age_ms == 11_000 and ready.depth_imbalance is None


def test_public_trade_notional_deduplicates_and_requires_warmup():
    clock = [0]
    feed = PaperExitFeed("BTC", lambda: clock[0])
    assert feed.snapshot(True).signed_flow_imbalance is None
    clock[0] = 60_000
    buy = {"time": 59_900, "tid": 8, "side": "B", "px": "100", "sz": "2", "coin": "BTC"}
    sell = {**buy, "tid": 9, "side": "A", "sz": "1"}
    feed.on_trades({"data": [buy, buy, sell]})
    assert feed.snapshot(True).signed_flow_imbalance == pytest.approx(1 / 3)


def test_market_database_round_trip_and_one_sample_per_bucket(tmp_path):
    path = tmp_path / cli.DB_NAME
    ready = sample(5, 102, open_interest=15.5, depth_imbalance=0.2)
    gap = MarketSample(10_000, "mark_stale")
    with closing(sqlite3.connect(path)) as conn:
        cli._init_market(conn)
        cli._record(conn, ready)
        cli._record(conn, gap)
        with pytest.raises(sqlite3.IntegrityError):
            cli._record(conn, replace(ready, at_ms=5_001))
    assert cli.load_samples(path, 0, 11_000) == [ready, gap]


def test_recorder_status_requires_recent_ready_quote_and_rejects_future_tail(tmp_path):
    path = tmp_path / cli.DB_NAME
    with closing(sqlite3.connect(path)) as conn:
        cli._init_market(conn)
        cli._record(conn, sample(5))
        cli._record(conn, MarketSample(10_000, "mark_stale"))
        result = cli.capture_status(path, 10_001)
        assert result["healthy"]
        assert result["recent_samples"] == {"ready": 1, "mark_stale": 1}
        assert not cli.capture_status(path, 38_000)["healthy"]
        assert not cli.capture_status(path, 8_000)["healthy"]


def test_research_subscriptions_and_state_reset_on_every_connection(monkeypatch):
    from src.core import api_manager

    monkeypatch.setattr(api_manager, "HAS_WEBSOCKET", True)
    feed = PaperExitFeed("BTC", lambda: 10_000)
    stream = feed.make_stream()
    sent = []
    transport = SimpleNamespace(send=lambda raw: sent.append(json.loads(raw)["subscription"]))
    stream._ws = transport
    stream._on_open(transport)
    assert {sub["type"] for sub in sent} == {"allMids", "trades", "l2Book", "bbo", "activeAssetCtx"}
    feed.on_context({"data": {"coin": "BTC", "ctx": {"markPx": 100}}})
    assert feed.snapshot(True).mark == 100
    stream._connected = False  # Simulate a reconnect with no new messages yet.
    stream._on_open(transport)
    assert sent.count({"type": "activeAssetCtx", "coin": "BTC"}) == 2
    assert feed.snapshot(True).mark is None
