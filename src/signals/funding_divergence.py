"""Funding-rate vs price divergence safety brake.

Reads BTC + ETH funding (Hyperliquid native) against their 4h moving
average and flags cross-market divergence:

  - Funding > 0 AND price < 4h MA  ⇒ crowded longs paying premium into
    a selloff = capitulation fuel ⇒ block new longs.
  - Funding < 0 AND price > 4h MA  ⇒ crowded shorts paying premium into
    a rally = short-squeeze fuel ⇒ block new shorts.

We treat BTC + ETH as the market-wide read. Coin-level decisions
(altcoin trades) inherit the same gate because alts overwhelmingly
follow BTC/ETH macro direction. This isn't a precise edge call --
it's an asymmetric safety brake the same way the synthetic-regime
brake is asymmetric: it can block, never confirm.

Cache: 5-minute TTL is fine. Funding ticks hourly on Hyperliquid;
candle MA changes slowly.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import config

logger = logging.getLogger(__name__)


# Market-wide reference coins. Hardcoded because the divergence signal
# depends on these being the *price-discovery leaders* — substituting an
# alt coin here would defeat the point.
_REFERENCE_COINS = ("BTC", "ETH")

_CACHE_LOCK = threading.Lock()
_CACHE_PAYLOAD: Optional[Dict[str, Any]] = None
_CACHE_TS: float = 0.0


def _funding_threshold() -> float:
    """Per-interval funding rate above which a side is considered "crowded".

    Hyperliquid quotes funding per-interval (hourly). 0.0001 (1bp) is the
    default Hyperliquid funding rate when the market is balanced -- we
    only trigger the divergence brake when funding meaningfully exceeds
    that baseline. Configurable so operators can widen in quiet markets.
    """
    return float(
        getattr(config, "FUNDING_DIVERGENCE_FUNDING_THRESHOLD", 0.00015)
    )


def _price_deviation_threshold() -> float:
    """How far price must deviate from the 4h MA to count as "trending against funding".

    Default 0.5% deviation. Wider thresholds reduce false positives in
    chop; tighter thresholds make the brake more sensitive.
    """
    return float(
        getattr(config, "FUNDING_DIVERGENCE_PRICE_DEV_THRESHOLD", 0.005)
    )


def _cache_ttl_seconds() -> float:
    return float(getattr(config, "FUNDING_DIVERGENCE_CACHE_TTL_S", 300.0))


def _fetch_funding_rates() -> Dict[str, float]:
    """Pull current per-interval funding rates for the reference coins."""
    try:
        from src.data.hyperliquid_client import get_asset_contexts

        ctxs = get_asset_contexts() or {}
    except Exception as exc:
        logger.debug("funding_divergence: asset context fetch failed: %s", exc)
        return {}
    rates: Dict[str, float] = {}
    for coin in _REFERENCE_COINS:
        ctx = ctxs.get(coin) or ctxs.get(coin.upper()) or {}
        try:
            rate = float(ctx.get("funding", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        rates[coin] = rate
    return rates


def _recent_closes(coin: str, n: int = 4) -> List[float]:
    """Last *n* hourly closes for ``coin`` (oldest first)."""
    try:
        from src.data import feature_store

        candles = feature_store.get_candles(coin, "1h", limit=max(n, 4))
    except Exception as exc:
        logger.debug("funding_divergence: candle fetch failed for %s: %s", coin, exc)
        return []
    closes: List[float] = []
    for row in candles or []:
        if isinstance(row, dict):
            close = row.get("c") if "c" in row else row.get("close")
        else:
            try:
                close = row["c"] if "c" in row.keys() else row["close"]
            except Exception:
                close = None
        try:
            closes.append(float(close))
        except (TypeError, ValueError):
            continue
    return closes[-n:]


def _coin_divergence(coin: str) -> Tuple[Optional[str], Dict[str, float]]:
    """Compute divergence signal for a single coin.

    Returns ``(side_to_block, telemetry)`` where ``side_to_block`` is
    ``"long"`` / ``"short"`` / ``None`` and telemetry has the inputs
    used so logs and tests can reason about the call.
    """
    rates = _fetch_funding_rates()
    funding = rates.get(coin)
    closes = _recent_closes(coin, n=4)
    telemetry: Dict[str, float] = {"funding": float(funding or 0.0)}
    if funding is None or len(closes) < 4:
        return None, telemetry
    current_price = closes[-1]
    ma_4h = sum(closes) / len(closes)
    telemetry.update({
        "current_price": float(current_price),
        "ma_4h": float(ma_4h),
        "ratio": float(current_price / ma_4h) if ma_4h else 0.0,
    })
    if ma_4h <= 0 or current_price <= 0:
        return None, telemetry
    funding_thr = _funding_threshold()
    dev_thr = _price_deviation_threshold()
    # Crowded longs into selloff
    if funding > funding_thr and current_price < ma_4h * (1.0 - dev_thr):
        return "long", telemetry
    # Crowded shorts into rally
    if funding < -funding_thr and current_price > ma_4h * (1.0 + dev_thr):
        return "short", telemetry
    return None, telemetry


def _compute_market_divergence() -> Dict[str, Any]:
    """Build the cached market-wide divergence payload."""
    per_coin: Dict[str, Dict[str, Any]] = {}
    block_votes = {"long": 0, "short": 0}
    for coin in _REFERENCE_COINS:
        side, telemetry = _coin_divergence(coin)
        per_coin[coin] = {"side_to_block": side, **telemetry}
        if side in block_votes:
            block_votes[side] += 1

    if block_votes["long"] >= 1 and block_votes["short"] == 0:
        side_to_block = "long"
    elif block_votes["short"] >= 1 and block_votes["long"] == 0:
        side_to_block = "short"
    else:
        side_to_block = None

    confidence = 0.0
    if side_to_block:
        agree = block_votes[side_to_block]
        # 1-coin agreement = 0.55; 2-coin = 0.85 (asymmetric: don't
        # claim near-certainty from two correlated assets).
        confidence = 0.55 if agree == 1 else 0.85

    return {
        "side_to_block": side_to_block,
        "confidence": confidence,
        "per_coin": per_coin,
        "computed_at": time.time(),
    }


def get_market_divergence(*, force_refresh: bool = False) -> Dict[str, Any]:
    """Return cached market-wide divergence signal."""
    global _CACHE_PAYLOAD, _CACHE_TS
    now = time.time()
    with _CACHE_LOCK:
        if not force_refresh and _CACHE_PAYLOAD and (now - _CACHE_TS) < _cache_ttl_seconds():
            return dict(_CACHE_PAYLOAD)
        payload = _compute_market_divergence()
        _CACHE_PAYLOAD = payload
        _CACHE_TS = now
        return dict(payload)


def should_block_side(side: str) -> Tuple[bool, str]:
    """Return ``(block, reason)`` for a proposed trade side.

    Returns ``(False, "gate_disabled")`` when the brake is turned off via
    ``FUNDING_DIVERGENCE_ENABLED=false``.
    """
    if not bool(getattr(config, "FUNDING_DIVERGENCE_ENABLED", True)):
        return False, "gate_disabled"

    normalised = str(side or "").strip().lower()
    if normalised in {"buy"}:
        normalised = "long"
    if normalised in {"sell"}:
        normalised = "short"
    if normalised not in {"long", "short"}:
        return False, "non_directional"

    payload = get_market_divergence()
    block = payload.get("side_to_block")
    if block == normalised:
        conf = payload.get("confidence", 0.0)
        per_coin = payload.get("per_coin", {})
        details = ", ".join(
            f"{c}:funding={ctx.get('funding'):.4%} "
            f"price/ma={ctx.get('ratio', 0):.4f}"
            for c, ctx in per_coin.items()
            if ctx.get("side_to_block") == normalised
        )
        return True, f"funding_divergence_blocks_{normalised}:{conf:.2f}|{details}"
    return False, "no_divergence"


def reset_cache_for_tests() -> None:
    """Drop the cached payload — only for use in tests."""
    global _CACHE_PAYLOAD, _CACHE_TS
    with _CACHE_LOCK:
        _CACHE_PAYLOAD = None
        _CACHE_TS = 0.0
