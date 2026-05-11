"""Integration test: build a SubsystemContainer inside the replay harness.

These tests prove that a real (not mocked) container with the REPLAY
profile can be constructed while the harness is engaged. They are the
boundary between the "data plane" tests (clock/oracle/shim) and the
"decision plane" -- once a container builds and stubs overlay cleanly,
the next step is invoking run_trading_cycle.
"""
import os
import sqlite3
from datetime import datetime, timezone

import pytest

from src.backtest.replay.harness import ReplayHarness
from src.backtest.replay.stub_subsystems import _StubBase
from src.backtest.replay.strategy_seed import (
    build_default_smoke_snapshot, seed_into,
)


CACHE_DB = "data/candle_cache.db"


def _cache_has_coverage(min_count: int = 2000) -> bool:
    if not os.path.exists(CACHE_DB):
        return False
    with sqlite3.connect(f"file:{CACHE_DB}?mode=ro", uri=True) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM candles WHERE coin='BTC' AND timeframe='1h'"
        ).fetchone()
        return bool(row) and row[0] >= min_count


pytestmark = pytest.mark.skipif(
    not _cache_has_coverage(),
    reason="BTC 1h cache not populated; integration tests need real candle data",
)

WINDOW_START_MS = int(datetime(2025, 8, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
WINDOW_END_MS = int(datetime(2025, 8, 8, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)


# --- Container boot --------------------------------------------------

def test_container_builds_with_replay_profile(monkeypatch):
    """The REPLAY profile should produce a usable container with real
    decision-pipeline subsystems and overlayed stubs."""
    monkeypatch.delenv("HL_BOT_DB", raising=False)
    with ReplayHarness(
        start_ts_ms=WINDOW_START_MS, end_ts_ms=WINDOW_END_MS,
        coins=["BTC"], engage_network_sandbox=True, build_container=True,
        keep_replay_db=False,
    ) as h:
        # Real subsystems for decision pipeline
        assert h.container is not None
        assert h.container.firewall is not None, "firewall must be real, not None"
        assert h.container.paper_trader is not None, "paper_trader must be real"
        assert h.container.regime_detector is not None
        assert h.container.scorer is not None
        # Container has the clock injected
        assert h.container.clock is h.clock


def test_container_stubs_replace_data_source_slots(monkeypatch):
    """Stub overlays must actually appear on the container, not the real
    subsystems that build_subsystems may have created."""
    monkeypatch.delenv("HL_BOT_DB", raising=False)
    with ReplayHarness(
        start_ts_ms=WINDOW_START_MS, end_ts_ms=WINDOW_END_MS,
        coins=["BTC"], engage_network_sandbox=True, build_container=True,
        keep_replay_db=False,
    ) as h:
        # Every overlay slot is a _StubBase instance, not whatever the
        # registry would have produced.
        for slot in ("polymarket", "options_scanner", "macro_regime",
                     "event_scanner", "exchange_agg", "multi_scanner",
                     "predictive_forecaster", "cross_venue_hedger"):
            sub = getattr(h.container, slot, None)
            assert isinstance(sub, _StubBase), (
                f"container.{slot} = {type(sub).__name__}, expected a stub"
            )


def test_container_firewall_uses_stub_event_scanner(monkeypatch):
    """After overlay, the firewall must consult the STUB event_scanner.

    The registry wires firewall.set_event_scanner(real_event_scanner)
    during build, but the harness then overlays a stub. _rewire_firewall_event_scanner
    re-attaches so the firewall actually sees the stub.
    """
    monkeypatch.delenv("HL_BOT_DB", raising=False)
    with ReplayHarness(
        start_ts_ms=WINDOW_START_MS, end_ts_ms=WINDOW_END_MS,
        coins=["BTC"], engage_network_sandbox=True, build_container=True,
        keep_replay_db=False,
    ) as h:
        fw = h.container.firewall
        es_on_fw = getattr(fw, "event_scanner", None)
        es_on_container = h.container.event_scanner
        assert es_on_fw is es_on_container, (
            "firewall.event_scanner must point at the stub, not the original"
        )
        assert isinstance(es_on_fw, _StubBase)


def test_replay_db_is_isolated_from_production(monkeypatch):
    """While the harness is engaged, the DB module's path must be the
    replay DB, never the production data/bot.db."""
    monkeypatch.delenv("HL_BOT_DB", raising=False)
    with ReplayHarness(
        start_ts_ms=WINDOW_START_MS, end_ts_ms=WINDOW_END_MS,
        coins=["BTC"], engage_network_sandbox=True, build_container=True,
        keep_replay_db=False,
    ) as h:
        from src.data import database as db
        assert "replay_" in db._DB_PATH, (
            f"Expected DB path to point at replay DB, got {db._DB_PATH}"
        )
        assert h.replay_db.db_path.exists()


def test_seeded_pool_visible_to_scorer(monkeypatch):
    """Seeding strategies into the replay DB makes them visible to the
    real scorer (proves DB plumbing works end-to-end)."""
    monkeypatch.delenv("HL_BOT_DB", raising=False)
    with ReplayHarness(
        start_ts_ms=WINDOW_START_MS, end_ts_ms=WINDOW_END_MS,
        coins=["BTC"], engage_network_sandbox=True, build_container=True,
        keep_replay_db=False,
    ) as h:
        seed_into(str(h.replay_db.db_path), build_default_smoke_snapshot())
        # The bot's database module is now pointing at our replay DB; the
        # scorer's view of the world should match what we just seeded.
        from src.data import database as db
        strategies = db.get_active_strategies(validated_only=False)
        assert len(strategies) == 10
        # Spot-check shape
        types = {s.get("strategy_type") for s in strategies}
        assert "momentum" in types
        assert "rsi" in types
