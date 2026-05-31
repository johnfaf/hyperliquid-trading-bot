"""Paper-trade lifecycle timestamps must honour the injected clock.

`src/data/database.py` is intentionally *outside* the clock-injection ratchet
(`tests/test_clock_injection_ratchet.py` only audits signals/learning/trading/
analysis, since persistence is usually legitimately wall-clock). But the
paper-trade ``opened_at`` / ``closed_at`` columns are an exception: during a
historical replay they must record *market* time, otherwise every backtested
trade is stamped with the wall-clock instant the row was written (observed:
trades opened in March 2026 were persisted as "now"), which destroys hold-time
and any time-windowed analysis.

These functions route their timestamp through ``_trade_event_now_iso()``, which
delegates to ``clock_provider``. In production the default backend returns the
real wall clock, so live behaviour is unchanged; under the replay harness the
provider is swapped to the deterministic ReplayClock.
"""
from __future__ import annotations

import inspect
from datetime import datetime, timezone

from src.core import clock_provider
from src.backtest.replay.clock import ReplayClock
import src.data.database as db


_REPLAY_TS_MS = int(datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)


def test_trade_event_now_follows_replay_clock():
    """With a ReplayClock installed, the helper returns *market* time."""
    clk = ReplayClock(start_ts_ms=_REPLAY_TS_MS, label="ts-test")
    prev = clock_provider.install(clk)
    try:
        iso = db._trade_event_now_iso()
    finally:
        clock_provider.restore(prev)
    assert iso.startswith("2025-03-01T12:00"), f"expected replay time, got {iso}"


def test_trade_event_now_falls_back_to_wallclock():
    """With no replay clock installed, the helper returns ~wall-clock (not the
    replay window) so production behaviour is unchanged."""
    iso = db._trade_event_now_iso()
    parsed = datetime.fromisoformat(iso)
    # Must be a real, recent, tz-aware timestamp -- not the 2025-03 replay window.
    assert parsed.tzinfo is not None
    assert not iso.startswith("2025-03-01T12:00")
    assert parsed.year >= 2025


def test_persistence_fns_route_timestamp_through_clock_helper():
    """Wiring guard: the three paper-trade lifecycle writers must derive their
    timestamp from the clock helper, never a raw datetime.now(). Prevents a
    future edit from silently reintroducing the wall-clock leak."""
    for fn in (db.open_paper_trade, db.close_paper_trade,
               db.close_paper_trade_and_credit_account):
        src = inspect.getsource(fn)
        assert "_trade_event_now_iso()" in src, (
            f"{fn.__name__} no longer uses _trade_event_now_iso() for its "
            f"opened_at/closed_at timestamp"
        )
        assert "datetime.now(" not in src, (
            f"{fn.__name__} reintroduced a raw datetime.now() -- route trade "
            f"timestamps through _trade_event_now_iso() instead"
        )
