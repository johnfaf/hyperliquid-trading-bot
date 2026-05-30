"""Tests for the recent-loss guard time-window.

The live recent-loss guard reads the last N fills by COUNT only. When live
trading goes quiet (e.g. after the May freeze + bug fixes) that window keeps
treating weeks-old, bug-era fills as "recent" forever -- it blocked shorts on
108 pre-fix fills (last traded May 27).  The time-window drops fills older than
LIVE_RECENT_LOSS_LOOKBACK_HOURS so the guard reflects the fixed system.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

from src.trading.live_trader import LiveTrader


def _row(closed_at, pnl=-1.0, side="short", coin="BTC"):
    return {"closed_at": closed_at, "pnl": pnl, "side": side, "coin": coin}


def _iso(dt):
    return dt.isoformat()


def _fake_self(hours):
    # _drop_stale_fill_rows only touches these two members of self.
    return SimpleNamespace(
        _live_recent_loss_lookback_hours=hours,
        _fill_row_closed_dt=LiveTrader._fill_row_closed_dt,
    )


def _patch_now(monkeypatch, now_dt):
    import src.core.clock_provider as cp
    monkeypatch.setattr(cp, "utc_now", lambda: now_dt)


# ── _fill_row_closed_dt ──────────────────────────────────────────


def test_parse_closed_at_iso():
    dt = LiveTrader._fill_row_closed_dt(_row("2026-05-27T12:00:00+00:00"))
    assert dt is not None and (dt.year, dt.month, dt.day) == (2026, 5, 27)


def test_parse_naive_iso_assumes_utc():
    dt = LiveTrader._fill_row_closed_dt(_row("2026-05-27T12:00:00"))
    assert dt is not None and dt.tzinfo is not None


def test_parse_bad_returns_none():
    assert LiveTrader._fill_row_closed_dt(_row("not-a-date")) is None
    assert LiveTrader._fill_row_closed_dt({}) is None
    assert LiveTrader._fill_row_closed_dt("x") is None


# ── _drop_stale_fill_rows ────────────────────────────────────────


def test_drops_rows_older_than_window(monkeypatch):
    now = datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc)
    _patch_now(monkeypatch, now)
    rows = [
        _row(_iso(now - timedelta(hours=72)), coin="OLD"),    # bug-era -> drop
        _row(_iso(now - timedelta(hours=49)), coin="OLD2"),   # just past 48h -> drop
        _row(_iso(now - timedelta(hours=2)), coin="NEW"),     # recent -> keep
    ]
    kept = LiveTrader._drop_stale_fill_rows(_fake_self(48.0), rows)
    assert {r["coin"] for r in kept} == {"NEW"}


def test_all_stale_returns_empty_so_guard_has_no_data(monkeypatch):
    """The exact prod case: every fill is >48h old -> guard sees nothing ->
    _passes_live_recent_loss_guard returns True (no recent evidence)."""
    now = datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc)
    _patch_now(monkeypatch, now)
    rows = [_row(_iso(now - timedelta(hours=72 + i))) for i in range(108)]
    assert LiveTrader._drop_stale_fill_rows(_fake_self(48.0), rows) == []


def test_zero_hours_disables_time_filter(monkeypatch):
    now = datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc)
    _patch_now(monkeypatch, now)
    rows = [_row(_iso(now - timedelta(hours=999)))]
    assert len(LiveTrader._drop_stale_fill_rows(_fake_self(0), rows)) == 1


def test_unparseable_rows_kept_failsafe(monkeypatch):
    """A row with an unparseable close time is kept so we never silently
    discard genuine loss evidence."""
    now = datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc)
    _patch_now(monkeypatch, now)
    rows = [_row("garbage"), _row(_iso(now - timedelta(hours=72)))]
    kept = LiveTrader._drop_stale_fill_rows(_fake_self(48.0), rows)
    assert len(kept) == 1 and kept[0]["closed_at"] == "garbage"


def test_empty_input_is_safe(monkeypatch):
    now = datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc)
    _patch_now(monkeypatch, now)
    assert LiveTrader._drop_stale_fill_rows(_fake_self(48.0), []) == []
