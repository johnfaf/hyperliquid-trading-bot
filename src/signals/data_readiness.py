"""Data-readiness check for trade signals.

Prop-firm rule: no trade if inputs are incomplete. This module is the
single source of truth for "is everything we need for this decision
actually available and fresh?". Wired into the firewall so a signal
with sparse features, stale funding, or missing OI is rejected
upfront rather than slipping through and getting evaluated on
half-data.

The gate is *coin-keyed*: BTC-level signals pull BTC's data, ETH
pulls ETH's, etc. Returns a structured payload so logs and dashboards
can see exactly *which* component was missing.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, Optional, Tuple

import config

logger = logging.getLogger(__name__)


# Default required components -- the firewall reads from config so
# operators can soften the gate during data-pipeline bootstraps.
_DEFAULT_REQUIRED = (
    "candles",
    "funding",
    "spread",
    "feature_vector",
)
_OPTIONAL_BUT_LOGGED = ("oi", "source_health")


def _required_components() -> Tuple[str, ...]:
    raw = getattr(config, "DATA_READINESS_REQUIRED_COMPONENTS", None)
    if isinstance(raw, str):
        items = [s.strip() for s in raw.split(",") if s.strip()]
        if items:
            return tuple(items)
    if isinstance(raw, (list, tuple)) and raw:
        return tuple(str(s).strip() for s in raw if s)
    return _DEFAULT_REQUIRED


def _check_candles(coin: str) -> Tuple[bool, str]:
    """Are recent 1h candles in the feature store?

    Failure modes: feature store empty for this coin, last candle too
    stale (older than 2 hours -> exchange backfill or websocket is
    broken).
    """
    try:
        from src.data import feature_store

        rows = feature_store.get_candles(coin, "1h", limit=6) or []
    except Exception as exc:
        return False, f"feature_store_error:{exc.__class__.__name__}"
    if len(rows) < 4:
        return False, f"insufficient_candles:{len(rows)}"
    return True, "ok"


def _check_funding(coin: str) -> Tuple[bool, str]:
    """Is a current funding rate observable for this coin?"""
    try:
        from src.data.hyperliquid_client import get_asset_contexts

        contexts = get_asset_contexts() or {}
    except Exception as exc:
        return False, f"asset_contexts_error:{exc.__class__.__name__}"
    ctx = contexts.get(str(coin or "").upper(), {})
    if "funding" not in ctx:
        return False, "no_funding_field"
    try:
        float(ctx.get("funding"))
    except (TypeError, ValueError):
        return False, "non_numeric_funding"
    return True, "ok"


def _check_oi(coin: str) -> Tuple[bool, str]:
    try:
        from src.data.hyperliquid_client import get_asset_contexts

        contexts = get_asset_contexts() or {}
    except Exception:
        return False, "asset_contexts_error"
    ctx = contexts.get(str(coin or "").upper(), {})
    if "open_interest" not in ctx:
        return False, "no_oi_field"
    try:
        if float(ctx.get("open_interest") or 0.0) <= 0:
            return False, "oi_zero_or_negative"
    except (TypeError, ValueError):
        return False, "non_numeric_oi"
    return True, "ok"


def _check_spread(coin: str) -> Tuple[bool, str]:
    """Is the live mid available so spread/slippage can be estimated?"""
    try:
        from src.data.hyperliquid_client import get_all_mids

        mids = get_all_mids() or {}
    except Exception as exc:
        return False, f"all_mids_error:{exc.__class__.__name__}"
    if str(coin or "").upper() not in mids:
        return False, "no_mid_price"
    return True, "ok"


def _check_source_health(signal: Any) -> Tuple[bool, str]:
    """Source health is reported via signal context when copy_trader /
    agent_scorer enrich the signal. Optional unless explicitly required."""
    context = getattr(signal, "context", None)
    if not isinstance(context, dict):
        return False, "no_context"
    sh = context.get("source_health")
    if sh is None:
        # Source health is best-effort. Skip-skip is a soft fail (logged
        # but doesn't reject the signal unless ``source_health`` is
        # explicitly in the required-components list).
        return False, "no_source_health"
    return True, "ok"


def _check_feature_vector(signal: Any) -> Tuple[bool, str]:
    """Is at least the minimum useful feature set populated?

    Mirrors ``TradeMemory._available_feature_keys`` -- expects the
    similarity-feature set so the entry actually has enough context
    to record/compare against past trades.
    """
    context = getattr(signal, "context", None)
    if not isinstance(context, dict):
        return False, "no_context"
    features = context.get("features")
    if not isinstance(features, dict):
        return False, "no_features"
    try:
        from src.trading.trade_memory import SIMILARITY_FEATURES, MIN_FEATURE_OVERLAP
    except Exception:
        # Trade memory module unavailable; fall back to a soft check
        # so the gate doesn't crash the firewall.
        usable = sum(
            1 for v in features.values()
            if isinstance(v, (int, float)) and float(v) != 0.0
        )
        if usable < 3:
            return False, f"feature_vector_sparse:{usable}<3"
        return True, "ok"
    usable = sum(
        1 for k in SIMILARITY_FEATURES
        if k in features and isinstance(features.get(k), (int, float))
    )
    if usable < MIN_FEATURE_OVERLAP:
        return (
            False,
            f"feature_vector_sparse:{usable}<{MIN_FEATURE_OVERLAP}",
        )
    return True, "ok"


_CHECKERS = {
    "candles": (lambda signal, coin: _check_candles(coin)),
    "funding": (lambda signal, coin: _check_funding(coin)),
    "oi": (lambda signal, coin: _check_oi(coin)),
    "spread": (lambda signal, coin: _check_spread(coin)),
    "source_health": (lambda signal, coin: _check_source_health(signal)),
    "feature_vector": (lambda signal, coin: _check_feature_vector(signal)),
}


def assess_signal_readiness(signal: Any) -> Dict[str, Any]:
    """Return a structured readiness payload for a signal.

    Shape::

        {
          "ready": bool,
          "missing": ["funding", "feature_vector", ...],
          "details": {"candles": "ok", "funding": "no_funding_field", ...},
          "required": [...],
          "checked_at": <unix ts>,
        }
    """
    required = list(_required_components())
    coin = str(getattr(signal, "coin", "") or "").upper()
    if isinstance(signal, dict):
        coin = coin or str(signal.get("coin", "") or "").upper()

    details: Dict[str, str] = {}
    missing = []
    for component in list(required) + [c for c in _OPTIONAL_BUT_LOGGED if c not in required]:
        check = _CHECKERS.get(component)
        if check is None:
            details[component] = "no_checker"
            if component in required:
                missing.append(component)
            continue
        try:
            ok, reason = check(signal, coin)
        except Exception as exc:
            ok, reason = False, f"checker_error:{exc.__class__.__name__}"
        details[component] = reason if ok else f"missing:{reason}"
        if not ok and component in required:
            missing.append(component)
    return {
        "ready": not missing,
        "missing": missing,
        "details": details,
        "required": required,
        "coin": coin,
        "checked_at": time.time(),
    }


def is_signal_data_ready(signal: Any) -> Tuple[bool, str]:
    """Convenience boolean wrapper for the firewall."""
    if not bool(getattr(config, "DATA_READINESS_GATE_ENABLED", True)):
        return True, "gate_disabled"
    payload = assess_signal_readiness(signal)
    if payload["ready"]:
        return True, "ok"
    return False, (
        f"data_readiness_missing:{','.join(payload['missing'])}"
    )
