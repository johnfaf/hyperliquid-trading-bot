"""
Feature Cycle — Candle collection & feature computation.

Integrates with the existing 3-tier scheduling system:

  - **Tier 2 (trading, ~5 min)** — collect 5m candles, compute 5m + 1h features
  - **Tier 3 (discovery, ~24 h)** — collect 4h + 1d candles, recompute all features

The cycle is a no-op when Postgres is unavailable (``FEATURE_STORE_ENABLED``
is False).  This ensures the bot degrades gracefully.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

import config
from src.data import feature_store as fs

logger = logging.getLogger(__name__)

# Backfill depths per timeframe (first-time only)
_BACKFILL_DAYS = {
    "5m": int(getattr(config, "FEATURE_STORE_BACKFILL_5M_DAYS", 7)),
    "1h": int(getattr(config, "FEATURE_STORE_BACKFILL_1H_DAYS", 30)),
    "4h": int(getattr(config, "FEATURE_STORE_BACKFILL_4H_DAYS", 90)),
    "1d": int(getattr(config, "FEATURE_STORE_BACKFILL_1D_DAYS", 365)),
}

# Cap watched coins to avoid API flooding
_MAX_COINS = int(getattr(config, "FEATURE_STORE_MAX_COINS", 30))


# ─── Watched coins ─────────────────────────────────────────────

def _get_watched_coins(container=None) -> List[str]:
    """Return coins to track.  Always includes BTC + ETH.

    Pulls from:
      * Explicit ``FEATURE_STORE_COINS`` env var (comma-separated)
      * Open paper/live positions
      * Active strategies with recent scores

    Capped at ``FEATURE_STORE_MAX_COINS``.
    """
    override = getattr(config, "FEATURE_STORE_COINS", "").strip()
    if override:
        coins = set(c.strip().upper() for c in override.split(",") if c.strip())
    else:
        coins = set()

    # Always watch BTC + ETH (needed for cross-asset features)
    coins.update({"BTC", "ETH"})

    # Add coins from open positions
    try:
        from src.data import database as db
        with db.get_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT coin FROM paper_trades WHERE status='open'"
            ).fetchall()
            for r in rows:
                coins.add(r["coin"] if isinstance(r, dict) else r[0])
    except Exception:
        pass

    # Add coins from recent strategies
    try:
        from src.data import database as db
        with db.get_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT name FROM strategies WHERE active = ? LIMIT 20",
                (True,),
            ).fetchall()
            for r in rows:
                # Strategy names often contain the coin
                name = r["name"] if isinstance(r, dict) else r[0]
                parts = name.upper().split("_")
                for p in parts:
                    if len(p) >= 2 and len(p) <= 10 and p.isalpha():
                        coins.add(p)
    except Exception:
        pass

    # Add top coins by volume if we don't have enough
    if len(coins) < 10:
        try:
            from src.data import hyperliquid_client as hl
            all_coins = hl.get_all_coins()
            if all_coins:
                for c in all_coins[:20]:
                    coins.add(c.upper())
        except Exception:
            pass

    return sorted(coins)[:_MAX_COINS]


# ─── Asset context cache (one API call, shared across all coins) ─

_asset_ctx_cache: dict = {}
_asset_ctx_ts: float = 0


def _get_asset_contexts() -> dict:
    """Fetch and cache asset contexts (funding, OI) — refreshed every 60s."""
    global _asset_ctx_cache, _asset_ctx_ts
    if time.time() - _asset_ctx_ts < 60 and _asset_ctx_cache:
        return _asset_ctx_cache
    try:
        from src.data import hyperliquid_client as hl
        ctx = hl.get_asset_contexts()
        if ctx:
            _asset_ctx_cache = ctx
            _asset_ctx_ts = time.time()
    except Exception as exc:
        logger.debug("asset context fetch failed: %s", exc)
    return _asset_ctx_cache


# ─── Core cycle functions ──────────────────────────────────────

def _collect_and_compute(coins: List[str], timeframes: List[str]) -> dict:
    """Collect candles for the given timeframes and compute features.

    Returns stats dict: ``{candles_stored: N, features_stored: N, errors: N}``
    """
    stats = {"candles_stored": 0, "features_stored": 0, "errors": 0, "coins": len(coins)}
    ctx = _get_asset_contexts()

    # Phase 1: Collect all candles (BTC/ETH first so they're available for cross-asset)
    ordered = sorted(coins, key=lambda c: (c not in ("BTC", "ETH"), c))
    try:
        from src.data.historical_market_data import snapshot_live_derivatives_history

        derivative_stats = snapshot_live_derivatives_history(ordered)
        stats["funding_rows"] = int(derivative_stats.get("funding_rows", 0) or 0)
        stats["open_interest_rows"] = int(derivative_stats.get("open_interest_rows", 0) or 0)
    except Exception as exc:
        logger.debug("derivatives history snapshot failed: %s", exc)
        stats["errors"] += 1
    for coin in ordered:
        for tf in timeframes:
            try:
                n = fs.collect_candles_for_coin(
                    coin, tf,
                    backfill_days=_BACKFILL_DAYS.get(tf, 7),
                )
                stats["candles_stored"] += n
            except Exception as exc:
                logger.debug("candle collect error %s/%s: %s", coin, tf, exc)
                stats["errors"] += 1

    # Phase 2: Fetch BTC/ETH candles for cross-asset features (now populated)
    btc_candles_by_tf = {}
    eth_candles_by_tf = {}
    for tf in timeframes:
        btc_candles_by_tf[tf] = fs.get_candles("BTC", tf, limit=60)
        eth_candles_by_tf[tf] = fs.get_candles("ETH", tf, limit=60)

    # Phase 3: Compute features for all coins
    prev_oi: dict = {}

    for coin in coins:
        for tf in timeframes:
            try:
                # 1. Fetch stored candles for feature computation
                candles = fs.get_candles(coin, tf, limit=60)
                if len(candles) < 15:
                    continue

                # 2. Get funding + OI from context
                coin_ctx = ctx.get(coin, {})
                funding = float(coin_ctx.get("funding", 0))
                oi = float(coin_ctx.get("open_interest", 0))
                prev = prev_oi.get(coin, oi)  # Use current if no previous

                # 3. Compute features
                features = fs.compute_features(
                    coin=coin,
                    timeframe=tf,
                    candles=candles,
                    funding_rate=funding,
                    open_interest=oi,
                    prev_open_interest=prev,
                    btc_candles=btc_candles_by_tf.get(tf),
                    eth_candles=eth_candles_by_tf.get(tf),
                )

                # 5. Store features at the latest candle timestamp
                if features:
                    ts = int(candles[-1]["t"])
                    stored = fs.store_features(coin, tf, ts, features)
                    stats["features_stored"] += stored

                prev_oi[coin] = oi

            except Exception as exc:
                logger.debug("feature cycle error %s/%s: %s", coin, tf, exc)
                stats["errors"] += 1

    return stats


def run_feature_cycle(container=None, tier: str = "trading") -> Optional[dict]:
    """Entry point called from the main loop.

    Args:
        container: SubsystemContainer (used to discover watched coins)
        tier: ``"trading"`` (Tier 2) or ``"daily"`` (Tier 3)

    Returns stats dict or None if feature store is disabled.
    """
    # Guard: feature store requires Postgres
    if not getattr(config, "POSTGRES_DSN", ""):
        return None
    if not fs._pg_available():
        logger.debug("Feature cycle skipped -- Postgres unavailable")
        return None

    coins = _get_watched_coins(container)
    if not coins:
        return None

    t0 = time.time()

    if tier == "trading":
        # Tier 2: fast timeframes only
        timeframes = ["5m", "1h"]
    elif tier == "daily":
        # Tier 3: all timeframes
        timeframes = ["5m", "1h", "4h", "1d"]
    else:
        return None

    stats = _collect_and_compute(coins, timeframes)
    elapsed = time.time() - t0

    logger.info(
        "Feature cycle [%s]: %d coins x %d TFs -> %d candles, %d features "
        "(%d errors) in %.1fs",
        tier, stats["coins"], len(timeframes),
        stats["candles_stored"], stats["features_stored"],
        stats["errors"], elapsed,
    )
    return stats


def backfill_all(container=None) -> dict:
    """One-time historical backfill for all watched coins × all timeframes.

    Run on first startup when the feature store is empty, or manually
    via ``python -m src.core.cycles.feature_cycle``.
    """
    if not fs._pg_available():
        logger.warning("Cannot backfill -- Postgres unavailable")
        return {}

    coins = _get_watched_coins(container)
    logger.info("Backfilling %d coins across all timeframes...", len(coins))
    t0 = time.time()
    stats = _collect_and_compute(coins, list(fs.TIMEFRAMES))
    elapsed = time.time() - t0
    logger.info(
        "Backfill complete: %d candles, %d features (%d errors) in %.1fs",
        stats["candles_stored"], stats["features_stored"],
        stats["errors"], elapsed,
    )
    return stats


def feature_store_is_empty() -> bool:
    """Return True if the candles table has no data."""
    return fs.get_candle_count() == 0


# ─── CLI entrypoint ────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    if "--backfill" in sys.argv:
        stats = backfill_all()
    else:
        stats = run_feature_cycle(tier="daily")

    if stats:
        print(f"\nCandles stored:  {stats.get('candles_stored', 0)}")
        print(f"Features stored: {stats.get('features_stored', 0)}")
        print(f"Errors:          {stats.get('errors', 0)}")
    else:
        print("Feature store not available (check POSTGRES_DSN)")
