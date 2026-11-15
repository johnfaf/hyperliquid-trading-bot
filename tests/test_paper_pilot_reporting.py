import json
import sqlite3
from contextlib import closing
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from scripts import daily_paper_report as report
from scripts import run_paper_pilot as pilot
from src.trading.paper_trader import PaperTrader


B = 100000 * pilot.HOUR


def candles():
    return [
        {"s": "BTC", "i": "1h", "t": t, "T": t + pilot.HOUR - 1, "c": "102" if t == B - pilot.HOUR else "100"}
        for t in range(B - 25 * pilot.HOUR, B, pilot.HOUR)
    ]


def test_only_complete_contiguous_unique_hours_may_signal():
    series = candles()
    assert pilot.upward_cross(series, B) is True
    assert pilot.upward_cross(series + [{"t": B, "c": "999"}], B) is True
    assert pilot.upward_cross(series[1:], B) is None
    assert pilot.upward_cross(series + [series[-1]], B) is None
    assert pilot.upward_cross(series[:-1] + [{**series[-1], "T": B}], B) is None
    assert pilot.upward_cross(series[:-1] + [{**series[-1], "c": "100"}], B) is False


def test_quotes_are_fresh_and_not_crossed():
    book = {"time": B, "coin": "BTC", "levels": [[{"px": "99"}], [{"px": "101"}]]}
    assert pilot.verified_mid(book, B + 1000) == 100
    assert pilot.verified_mid(book, B + 30_001) is None
    assert pilot.verified_mid(book, B - 2_001) is None
    assert pilot.verified_mid({**book, "levels": [[{"px": "103"}], [{"px": "101"}]]}, B) is None


def test_pilot_refuses_wrong_database_or_signing_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HL_BOT_DB", str(tmp_path / "bot.db"))
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    with pytest.raises(ValueError, match="own audit-paper"):
        pilot.validate_environment()
    monkeypatch.setenv("HL_BOT_DB", str(tmp_path / "audit-paper-1000.db"))
    monkeypatch.setenv("HL_AGENT_PRIVATE_KEY", "not-a-real-key")
    with pytest.raises(ValueError, match="must not inherit"):
        pilot.validate_environment()


def test_fresh_cross_opens_once_and_caps_notional(monkeypatch):
    fixed = datetime.fromtimestamp(B / 1000 + 10, timezone.utc)
    monkeypatch.setattr(pilot, "datetime", SimpleNamespace(now=lambda _: fixed))
    opened, events, state = [], [], {}
    db = SimpleNamespace(
        get_paper_account=lambda: {"balance": 1000},
        get_open_paper_trades=lambda: [],
        get_bot_state=state.get,
        set_bot_state=lambda key, value: state.update({key: value}) or True,
        open_paper_trade=lambda *args, **kwargs: opened.append((args, kwargs)) or 1,
        audit_log=lambda *args, **kwargs: events.append(args[0]),
    )
    market = SimpleNamespace(
        get_l2_book=lambda _: {"coin": "BTC", "time": B + 10_000, "levels": [[{"px": "99"}], [{"px": "101"}]]},
        get_candles=lambda *args: candles(),
    )
    trader = SimpleNamespace(check_open_positions=lambda **kw: None, drain_closed_events=lambda: [])
    monkeypatch.setattr(pilot, "_record_mark", lambda *args: None)
    assert pilot.tick(trader, db, market)
    assert pilot.tick(trader, db, market)
    assert len(opened) == 1
    args, kwargs = opened[0]
    assert args[2] == "long"
    assert args[3] * args[4] == pytest.approx(75)
    assert kwargs["leverage"] == 1
    assert events == ["paper_pilot_entry_simulated"]


@pytest.mark.parametrize(("side", "filled"), [("long", 98), ("short", 102)])
def test_exit_slippage_is_adverse_in_the_correct_direction(side, filled):
    assert PaperTrader._slippage_cost(side, 100, filled, 2, 1, is_entry=False) == 4


def test_report_costs_drawdown_and_tainted_and_other_day_filters(tmp_path):
    path = tmp_path / "restore #1.db"
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE paper_trades (id, coin, side, pnl, entry_price, exit_price, size, leverage,"
                   "metadata, status, opened_at, closed_at)")
        details = {
            "gross_pnl_before_fees": 6, "intended_entry_price": 100, "intended_exit_price": 105,
            "total_fees_paid": 0.4, "funding_accrued": 0.1, "close_reason": "stop_loss",
        }
        for i, when, meta in [(1, "2026-01-02", details), (2, "2026-01-02", {"synthetic_reconciliation": True}),
                             (3, "2026-01-03", details)]:
            db.execute(
                "INSERT INTO paper_trades VALUES (?, 'BTC', 'long', 5.5, 101, 104, 2, 1, ?, "
                "'closed', '2026-01-01T12:00:00', ?)", (i, json.dumps(meta), when + "T09:00:00"),
            )
        db.execute("CREATE TABLE paper_pilot_marks (observed_at, equity, passive_reference)")
        db.executemany("INSERT INTO paper_pilot_marks VALUES (?, ?, ?)", [
            ("2026-01-02T01:00:00", 1002, 1000), ("2026-01-02T02:00:00", 999, 1001)
        ])
        db.commit()
    original = path.read_bytes()
    body = report.build_report(path, date(2026, 1, 2), "fixture")
    assert "Closed paper trades: 1; synthetic or tainted closes excluded: 1." in body
    assert "Open paper trades at the end of the period or latest recorded state: 1." in body
    assert "| Gross before costs (un-slipped quotes) | $10.00 | 1/1 |" in body
    assert "| Modeled slippage (included in simulated fills) | $4.00 | 1/1 |" in body
    assert "| Net realized P&L | $5.50 | 1/1 |" in body
    assert "drawdown: $3.00" in body
    assert path.read_bytes() == original


def test_costs_never_infer_missing_slippage_or_funding():
    row = {"entry_price": 101, "exit_price": 104, "size": 2, "leverage": 1, "side": "long", "pnl": 5.5}
    costs = report.trade_costs(row, {"gross_pnl": 6})
    assert costs["Gross at simulated fills (slippage already included)"] == 6
    assert costs["Modeled slippage (included in simulated fills)"] is None
    assert costs["Estimated funding (positive = paid)"] is None
