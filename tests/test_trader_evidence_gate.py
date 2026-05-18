"""Minimum-evidence gate: hide thin/degenerate "100% winrate / 0% ROI"
traders from the dashboard + copy pool without branding them bots.

Covers:
  - db.trader_meets_evidence_bar (the predicate)
  - db.get_copyable_traders (active set minus thin rows)
  - db.get_known_bot_addresses excludes the low_evidence tag (so
    discovery keeps re-evaluating thin traders -- they are NOT bots)
  - the traders dashboard payload hides them and reports the count
"""
from __future__ import annotations

import contextlib
import json

import src.data.database as db
from src.ui.v2.routers import traders as traders_router


def _row(addr, *, trade_count, total_pnl=1.0, roi_pct=1.0):
    return {
        "address": addr,
        "trade_count": trade_count,
        "total_pnl": total_pnl,
        "roi_pct": roi_pct,
        "win_rate": 1.0,
    }


# ─── predicate ────────────────────────────────────────────────────

def test_evidence_bar_rejects_thin_history():
    assert db.trader_meets_evidence_bar(_row("0xa", trade_count=3), 10) is False


def test_evidence_bar_rejects_zero_pnl_zero_roi_even_with_trades():
    # The classic "100% winrate / 0% ROI" junk row.
    row = _row("0xb", trade_count=25, total_pnl=0.0, roi_pct=0.0)
    assert db.trader_meets_evidence_bar(row, 10) is False


def test_evidence_bar_accepts_real_track_record():
    row = _row("0xc", trade_count=30, total_pnl=412.5, roi_pct=18.0)
    assert db.trader_meets_evidence_bar(row, 10) is True


def test_evidence_bar_accepts_losing_trader_with_real_history():
    # A net-losing trader still has actionable evidence (not junk).
    row = _row("0xd", trade_count=40, total_pnl=-120.0, roi_pct=-9.0)
    assert db.trader_meets_evidence_bar(row, 10) is True


def test_evidence_bar_default_threshold_from_config():
    # No explicit threshold -> uses config.TRADER_MIN_CLOSED_TRADES (10).
    assert db.trader_meets_evidence_bar(_row("0xe", trade_count=9)) is False
    assert db.trader_meets_evidence_bar(_row("0xf", trade_count=10)) is True


# ─── get_copyable_traders ────────────────────────────────────────

def test_get_copyable_traders_filters_active(monkeypatch):
    active = [
        _row("0x1", trade_count=50, total_pnl=900.0, roi_pct=30.0),  # keep
        _row("0x2", trade_count=2, total_pnl=5.0, roi_pct=1.0),      # thin
        _row("0x3", trade_count=99, total_pnl=0.0, roi_pct=0.0),     # junk
    ]
    monkeypatch.setattr(db, "get_active_traders", lambda **kw: list(active))
    out = db.get_copyable_traders()
    addrs = {t["address"] for t in out}
    assert addrs == {"0x1"}


# ─── get_known_bot_addresses excludes low_evidence ───────────────

def test_trader_metadata_status_parses_all_shapes():
    assert db._trader_metadata_status(json.dumps({"status": "Low_Evidence"})) == "low_evidence"
    assert db._trader_metadata_status({"status": "bot_detected"}) == "bot_detected"
    assert db._trader_metadata_status(None) == ""
    assert db._trader_metadata_status("") == ""
    assert db._trader_metadata_status("not-json{") == ""
    assert db._trader_metadata_status(json.dumps({"bot_score": 5})) == ""


def test_known_bots_excludes_low_evidence_tag(monkeypatch):
    """A row deactivated for thin evidence must NOT be treated as a bot,
    so discovery keeps re-evaluating it. Real bots/quarantined rows stay
    in the set. (Connection monkeypatched -- conftest's lean test DB has
    no traders table; this exercises the real exclusion logic.)"""
    rows = [
        {"address": "0xBOT", "metadata": json.dumps({"status": "bot_detected"})},
        {"address": "0xQUAR", "metadata": json.dumps({"status": "quarantined"})},
        {"address": "0xNOMETA", "metadata": None},
        {"address": "0xLOWEV", "metadata": json.dumps({"status": "low_evidence"})},
    ]

    class _Cur:
        def fetchall(self):
            return rows

    class _Conn:
        def execute(self, *a, **k):
            return _Cur()

    @contextlib.contextmanager
    def _fake_conn(*a, **k):
        yield _Conn()

    monkeypatch.setattr(db, "get_connection", _fake_conn)
    known = db.get_known_bot_addresses()
    assert known == {"0xBOT", "0xQUAR", "0xNOMETA"}
    assert "0xLOWEV" not in known


# ─── dashboard payload hides junk + reports count ────────────────

def test_dashboard_payload_hides_low_evidence(monkeypatch):
    active = [
        _row("0xAAA", trade_count=60, total_pnl=1500.0, roi_pct=40.0),  # show
        _row("0xBBB", trade_count=1, total_pnl=2.0, roi_pct=1.0),       # hide
        _row("0xCCC", trade_count=80, total_pnl=0.0, roi_pct=0.0),      # hide
    ]
    monkeypatch.setattr(db, "get_active_traders", lambda **kw: list(active))
    monkeypatch.setattr(db, "get_known_bot_addresses", lambda: set())
    payload = traders_router._summary_payload()
    assert payload["available"] is True
    shown = {r["address"] for r in payload["rows"]}
    assert shown == {"0xAAA"}
    assert payload["totals"]["low_evidence_hidden"] == 2
    assert payload["totals"]["total_tracked"] == 3
