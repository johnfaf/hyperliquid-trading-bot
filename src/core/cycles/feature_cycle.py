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

# Cap watched coins to avoid API flooding.
# ★ AUDIT FIX (2026-05-26): cap raised 30 -> 80 because production
# showed signals firing on coins outside the watched set
# (e.g. BCH, VVV) -> data_readiness_missing rejection every cycle.
# 80 covers the typical long tail of copy-trader candidates without
# meaningfully increasing API load (1h candle fetch is ~80 calls
# per cycle = ~5 RPS, well under HL's rate limit).
_MAX_COINS = int(getattr(config, "FEATURE_STORE_MAX_COINS", 80))
_BOOTSTRAP_TOP_COINS = int(getattr(config, "FEATURE_STORE_BOOTSTRAP_TOP_COINS", 8))


# ─── Watched coins ─────────────────────────────────────────────

def _get_watched_coins(container=None) -> List[str]:
    """Return coins to track.  Always includes BTC + ETH.

    Pulls from:
      * Explicit ``FEATURE_STORE_COINS`` env var (comma-separated)
      * Open paper/live positions          (priority: ACTIVE)
      * Active strategies with recent scores   (priority: ACTIVE)
      * Recent copy-trade decision snapshots   (priority: CANDIDATE)
      * Bot's tracked-trader OPEN POSITIONS from position_snapshots
        (priority: CANDIDATE, new in this PR)
      * Top coins by volume (priority: BOOTSTRAP)

    Capped at ``FEATURE_STORE_MAX_COINS``.  Truncation is
    PRIORITY-ORDERED -- ACTIVE coins survive truncation first,
    then CANDIDATE, then BOOTSTRAP (instead of alphabetic).
    """
    # Three priority tiers; tuples (priority_rank, coin) get sorted
    # so lower rank = higher priority and we keep them when capping.
    active_coins: set = set()
    candidate_coins: set = set()
    bootstrap_coins: set = set()

    override = getattr(config, "FEATURE_STORE_COINS", "").strip()
    if override:
        # Explicit operator pin: highest priority.
        active_coins.update(
            c.strip().upper() for c in override.split(",") if c.strip()
        )

    # Always watch BTC + ETH (needed for cross-asset features).
    active_coins.update({"BTC", "ETH"})

    # ACTIVE: open paper trades
    try:
        from src.data import database as db
        with db.get_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT coin FROM paper_trades WHERE status='open'"
            ).fetchall()
            for r in rows:
                coin = r["coin"] if isinstance(r, dict) else r[0]
                if coin:
                    active_coins.add(str(coin).strip().upper())
    except Exception:
        pass

    # ACTIVE: execution source of truth (live exchange positions when
    # live mode is on; falls back to paper open positions).
    if container is not None:
        try:
            from src.core.live_execution import get_execution_open_positions

            for pos in get_execution_open_positions(container) or []:
                if isinstance(pos, dict):
                    coin = pos.get("coin") or pos.get("symbol") or ""
                else:
                    coin = getattr(pos, "coin", "") or getattr(pos, "symbol", "")
                coin = str(coin or "").upper().strip()
                if coin:
                    active_coins.add(coin)
        except Exception as exc:
            logger.debug("execution watched-coin lookup failed: %s", exc)

    # CANDIDATE: coins parsed out of active strategy names.
    try:
        from src.data import database as db
        with db.get_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT name FROM strategies WHERE active = ? LIMIT 20",
                (True,),
            ).fetchall()
            for r in rows:
                name = r["name"] if isinstance(r, dict) else r[0]
                parts = name.upper().split("_")
                for p in parts:
                    if len(p) >= 2 and len(p) <= 10 and p.isalpha():
                        candidate_coins.add(p)
    except Exception:
        pass

    # CANDIDATE: coins from recent copy-trade decision snapshots.
    # ★ AUDIT FIX (2026-05-26): default cap raised 25 -> 60 to match
    # the broader watched-coin universe.
    try:
        from src.data import database as db
        import config as _cfg
        _copy_cap = int(getattr(_cfg, "FEATURE_COPY_CANDIDATE_COINS_MAX", 60))
        if _copy_cap > 0:
            with db.get_connection() as conn:
                if db.table_exists("decision_snapshots"):
                    rows = conn.execute(
                        """
                        SELECT coin, MAX(created_at) AS m
                        FROM decision_snapshots
                        WHERE source LIKE 'copy_trade%'
                          AND coin IS NOT NULL AND coin != ''
                        GROUP BY coin
                        ORDER BY m DESC
                        LIMIT ?
                        """,
                        (_copy_cap,),
                    ).fetchall()
                    for r in rows:
                        c = r["coin"] if isinstance(r, dict) else r[0]
                        if c:
                            candidate_coins.add(str(c).strip().upper())
    except Exception:
        pass

    # CANDIDATE: coins from RECENT tracked-trader open positions.
    # ★ AUDIT FIX (2026-05-26): the copy-trade decision_snapshots path
    # only sees coins where the bot already EVALUATED a signal for
    # them, but a wallet that just opened a position is the very FIRST
    # event the bot processes -- so the signal arrives BEFORE any
    # decision_snapshot exists.  Read position_snapshots directly for
    # the coins our tracked traders currently hold.  Last 6 hours +
    # current top-up keeps the query cheap.  This is what was missing
    # for the BCH / VVV signals observed on 2026-05-26.
    try:
        from src.data import database as db
        import config as _cfg
        _pos_cap = int(getattr(_cfg, "FEATURE_POSITION_SNAPSHOT_COINS_MAX", 50))
        if _pos_cap > 0:
            with db.get_connection() as conn:
                if db.table_exists("position_snapshots"):
                    rows = conn.execute(
                        """
                        SELECT coin, MAX(timestamp) AS m
                        FROM position_snapshots
                        WHERE coin IS NOT NULL AND coin != ''
                          AND size > 0
                          AND timestamp >= datetime('now', '-12 hours')
                        GROUP BY coin
                        ORDER BY m DESC
                        LIMIT ?
                        """,
                        (_pos_cap,),
                    ).fetchall()
                    for r in rows:
                        c = r["coin"] if isinstance(r, dict) else r[0]
                        if c:
                            candidate_coins.add(str(c).strip().upper())
    except Exception:
        pass

    # BOOTSTRAP: top coins by volume if we don't have enough yet.
    total_so_far = len(active_coins) + len(candidate_coins)
    if total_so_far < 10:
        try:
            from src.data import hyperliquid_client as hl
            all_coins = hl.get_all_coins()
            if all_coins:
                target_total = min(_BOOTSTRAP_TOP_COINS, _MAX_COINS)
                for c in all_coins:
                    if total_so_far + len(bootstrap_coins) >= target_total:
                        break
                    bootstrap_coins.add(c.upper())
        except Exception:
            pass

    # Priority-ordered truncation: ACTIVE first, then CANDIDATE, then
    # BOOTSTRAP.  Within a tier, alphabetic for stability.  Bug
    # being fixed: ``sorted(coins)[:_MAX_COINS]`` was alphabetic across
    # all tiers -- so an ACTIVE position on ZRX could be dropped in
    # favour of a BOOTSTRAP coin starting with A.
    out: List[str] = []
    seen: set = set()
    for tier in (sorted(active_coins), sorted(candidate_coins), sorted(bootstrap_coins)):
        for c in tier:
            if c in seen:
                continue
            seen.add(c)
            out.append(c)
            if len(out) >= _MAX_COINS:
                return out
    return out


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
    # ★ M24 FIX: previously prev_oi was reset to {} at every invocation.
    # On the first cycle after a process restart, every coin's OI delta
    # was 0 because the in-memory dict had no prior observation -- even
    # if real OI had moved meaningfully overnight.  Bootstrap prev_oi
    # from the most recent stored sample in `open_interest_history`
    # before the loop begins so deltas span across restarts.
    prev_oi: dict = {}
    try:
        from src.data.historical_market_data import get_open_interest_history
        for _coin in coins:
            try:
                history = get_open_interest_history(_coin, limit=2) or []
                # history is ordered DESC: index 0 is the latest, 1 is prior.
                # We want the PRIOR observation (index 1) so the first cycle
                # post-restart computes a real delta against it.  If only
                # one row exists, fall back to that single value.
                pick = history[1] if len(history) >= 2 else (
                    history[0] if history else None
                )
                if pick:
                    prev_oi[_coin] = float(pick.get("open_interest", 0.0) or 0.0)
            except Exception:
                continue
    except Exception as exc:
        logger.debug("prev_oi bootstrap failed (will use first-cycle default): %s", exc)

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
