"""End-to-end smoke test: replay harness over the real cached BTC data.

These tests use the production `data/candle_cache.db` if it exists. If it
doesn't (CI without the cache), they skip. The point is to prove:

  1. The harness boots, swaps the api_manager singleton, ticks, tears down.
  2. The clock_provider is actually swapped -- code reading utc_now() during
     the with-block sees the replay clock, NOT real wall-clock.
  3. The api_manager.get_manager() singleton returns the shim during the
     with-block, and a regime_detector-style candle fetch returns only
     bars whose close is <= the replay tick.
  4. After teardown, the clock_provider is restored and get_manager()
     would build a fresh real APIManager on next call.
  5. With the network sandbox engaged, any direct requests.get is blocked.
"""
import os
import sqlite3
from datetime import datetime, timezone

import pytest

from src.backtest.replay.harness import ReplayHarness
from src.backtest.replay.network_sandbox import ReplayNetworkBlocked
from src.core import clock_provider
import src.core.api_manager as am


CACHE_DB = "data/candle_cache.db"


def _cache_has_coverage(min_count: int = 2000) -> bool:
    if not os.path.exists(CACHE_DB):
        return False
    conn = sqlite3.connect(f"file:{CACHE_DB}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM candles WHERE coin='BTC' AND timeframe='1h'"
        ).fetchone()
        return bool(row) and row[0] >= min_count
    finally:
        conn.close()


pytestmark = pytest.mark.skipif(
    not _cache_has_coverage(),
    reason="BTC 1h cache not populated; smoke test needs data/candle_cache.db",
)


# Use a window known to be inside the cached range.
# Cache covers ~2025-04-05 through 2026-05-09. Mid-window:
WINDOW_START_MS = int(datetime(2025, 8, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
WINDOW_END_MS = int(datetime(2025, 8, 8, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)


def test_harness_engages_and_tears_down():
    """Basic lifecycle: with-block sets globals, exits restores them."""
    assert am._manager is None or am._manager is not None  # whatever state
    before_clock = clock_provider.current()

    with ReplayHarness(
        start_ts_ms=WINDOW_START_MS, end_ts_ms=WINDOW_END_MS,
        coins=["BTC"], engage_network_sandbox=False,
    ) as h:
        # Inside: clock backend swapped
        assert clock_provider.current() is h.clock
        # Inside: api_manager singleton is the shim
        assert am.get_manager() is h.api

    # After: clock restored
    assert clock_provider.current() is before_clock
    # After: api_manager singleton cleared, would rebuild lazily
    assert am._manager is None


def test_harness_ticks_advance_clock_monotonically():
    step = 3_600_000  # 1h
    seen_timestamps = []
    with ReplayHarness(
        start_ts_ms=WINDOW_START_MS, end_ts_ms=WINDOW_END_MS,
        coins=["BTC"], engage_network_sandbox=False,
    ) as h:
        for tick in h.iter_ticks(step_ms=step):
            seen_timestamps.append(tick.ts_ms)
            assert h.clock.now_ms() == tick.ts_ms
    # Strictly monotonic increase
    assert all(b > a for a, b in zip(seen_timestamps, seen_timestamps[1:]))
    # No tick at or past end
    assert seen_timestamps[-1] < WINDOW_END_MS
    # Expected length = ceil((end - start) / step)
    assert len(seen_timestamps) == (WINDOW_END_MS - WINDOW_START_MS) // step


def test_harness_candle_fetch_is_causal():
    """At each tick, asking for BTC candles must return only bars whose close
    is <= the clock. This is the actual no-lookahead proof on real data."""
    step = 6 * 3_600_000  # 6h steps
    with ReplayHarness(
        start_ts_ms=WINDOW_START_MS, end_ts_ms=WINDOW_END_MS,
        coins=["BTC"], engage_network_sandbox=False,
    ) as h:
        for tick in h.iter_ticks(step_ms=step):
            # Mimic the live regime_detector candle request
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": "BTC", "interval": "1h",
                    "startTime": tick.ts_ms - 24 * 3_600_000,
                    "endTime": tick.ts_ms,
                },
            }
            candles = h.api.post(payload)
            # Every returned bar's close must be <= tick.ts_ms
            for c in candles:
                close_ms = int(c["T"])
                assert close_ms <= tick.ts_ms, (
                    f"LEAK: tick={tick.ts_ms} got candle closing at {close_ms} "
                    f"(future by {close_ms - tick.ts_ms} ms)"
                )


def test_harness_clock_provider_reroutes_through_replay():
    """clock_provider.utc_now() must return the replay clock value during the
    with-block. This proves the production datetime.now() sites we converted
    actually see the replay time."""
    with ReplayHarness(
        start_ts_ms=WINDOW_START_MS, end_ts_ms=WINDOW_END_MS,
        coins=["BTC"], engage_network_sandbox=False,
    ) as h:
        now1 = clock_provider.utc_now()
        assert int(now1.timestamp() * 1000) == WINDOW_START_MS

        h.clock.advance(3_600_000)
        now2 = clock_provider.utc_now()
        assert int(now2.timestamp() * 1000) == WINDOW_START_MS + 3_600_000
        assert now2 > now1


def test_harness_network_sandbox_blocks_outbound():
    """When the sandbox is engaged, direct requests.get raises."""
    import requests
    with ReplayHarness(
        start_ts_ms=WINDOW_START_MS, end_ts_ms=WINDOW_END_MS,
        coins=["BTC"], engage_network_sandbox=True,
    ):
        with pytest.raises(ReplayNetworkBlocked):
            requests.get("https://api.hyperliquid.xyz/info", timeout=1)


def test_harness_report_records_api_and_stub_activity():
    """Verify the run report aggregates telemetry from the shim and stubs."""
    step = 3_600_000
    with ReplayHarness(
        start_ts_ms=WINDOW_START_MS, end_ts_ms=WINDOW_END_MS,
        coins=["BTC"], engage_network_sandbox=False,
    ) as h:
        tick_count = 0
        for _ in h.iter_ticks(step_ms=step):
            h.api.post({"type": "allMids"})
            h.api.post({"type": "metaAndAssetCtxs"})
            h.stubs["polymarket"].get_market_sentiment()
            h.stubs["macro_regime"].get_risk_posture()
            tick_count += 1
        report = h.build_report(tick_count=tick_count, step_ms=step)
    assert report.tick_count == tick_count
    assert report.api_calls_by_type["allMids"] == tick_count
    assert report.api_calls_by_type["metaAndAssetCtxs"] == tick_count
    assert report.stub_calls["polymarket"]["get_market_sentiment"] == tick_count
    assert report.stub_calls["macro_regime"]["get_risk_posture"] == tick_count


def test_harness_strict_mode_raises_on_unknown_req_type():
    with ReplayHarness(
        start_ts_ms=WINDOW_START_MS, end_ts_ms=WINDOW_END_MS,
        coins=["BTC"], engage_network_sandbox=False, strict_api=True,
    ) as h:
        from src.backtest.replay.api_manager_shim import ReplayInterceptError
        with pytest.raises(ReplayInterceptError):
            h.api.post({"type": "openOrders"})
