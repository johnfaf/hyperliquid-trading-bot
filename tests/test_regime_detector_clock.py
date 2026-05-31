"""RegimeDetector's per-coin cache TTL must be measured in injected-clock time.

Bug this guards against: the cache keyed freshness on raw ``time.time()`` with a
120s TTL. During a historical replay the regime computed on the first tick then
stayed frozen for 2 *real* minutes -- i.e. for however many replay ticks ran in
that wall-time window -- so the regime classification (and which strategies the
firewall paused) depended on how fast the replay happened to run. The same
6-week window produced 1 vs 11 strategy trades depending on per-cycle workload.

The fix routes the TTL through ``clock_provider``. This test advances the
*replay* clock past the TTL while real wall-time barely moves: with the old
wall-clock cache the entry would still be "fresh" (wall-time didn't move) and we
would wrongly get a stale hit; with the fix the replay-time advance expires it.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.core import clock_provider
from src.backtest.replay.clock import ReplayClock
from src.analysis.regime_detector import RegimeDetector


def _synthetic_candles(n: int = 40, base: float = 100.0):
    out = []
    price = base
    for _ in range(n):
        nxt = price * 1.002
        out.append({"open": price, "high": price * 1.01,
                    "low": price * 0.99, "close": nxt, "volume": 1000.0})
        price = nxt
    return out


def test_regime_cache_ttl_keys_on_replay_clock():
    t0_ms = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    clk = ReplayClock(start_ts_ms=t0_ms, label="regime-clock-test")
    prev = clock_provider.install(clk)
    try:
        det = RegimeDetector()
        candles = _synthetic_candles()

        s1 = det.detect_regime("BTC", candles=candles)
        assert s1.timestamp.startswith("2026-01-01"), s1.timestamp
        ts1 = s1.timestamp

        # Advance replay clock by 60s (< CACHE_TTL=120). Real wall-time has
        # barely moved either way, so this only stays a hit if the TTL is
        # measured in replay time -> expected cache HIT (same timestamp).
        clk.set(t0_ms + 60_000)
        s2 = det.detect_regime("BTC", candles=candles)
        assert s2.timestamp == ts1, "within-TTL call should hit the cache"

        # Advance replay clock past the TTL (200s > 120s). With the OLD
        # wall-clock cache this would STILL be a hit (real time didn't move);
        # with the fix it expires and recomputes at the new replay instant.
        clk.set(t0_ms + 200_000)
        s3 = det.detect_regime("BTC", candles=candles)
        assert s3.timestamp != ts1, (
            "past-TTL replay advance must expire the cache (regression: cache "
            "still keyed on wall-clock time.time())"
        )
    finally:
        clock_provider.restore(prev)
