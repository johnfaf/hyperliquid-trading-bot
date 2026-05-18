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
from typing import Any, Dict, Tuple

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


def _min_feature_overlap() -> int:
    try:
        from src.trading.trade_memory import MIN_FEATURE_OVERLAP
        return int(MIN_FEATURE_OVERLAP)
    except Exception:
        return 3


def _similarity_features() -> Tuple[str, ...]:
    try:
        from src.trading.trade_memory import SIMILARITY_FEATURES
        return tuple(SIMILARITY_FEATURES)
    except Exception:
        return (
            "funding_rate", "oi_change", "price_change", "trend_strength",
            "volatility", "volume_ratio", "rsi", "momentum_score",
            "bollinger_position", "overall_score",
        )


def _usable_feature_count(features: Any) -> int:
    if not isinstance(features, dict) or not features:
        return 0
    sim = _similarity_features()
    aligned = sum(
        1 for k in sim
        if k in features and isinstance(features.get(k), (int, float))
    )
    if aligned:
        return aligned
    # Generic fallback: any numeric, non-zero values (handles feature
    # stores that use different naming, e.g. ``rsi_14``).
    return sum(
        1 for v in features.values()
        if isinstance(v, (int, float)) and float(v) != 0.0
    )


def _check_feature_vector(signal: Any) -> Tuple[bool, str]:
    """Is feature data *available* for this decision?

    A readiness gate asks "does the data exist?", not "did this specific
    code path attach it?". So: first check the signal's own context;
    if that's sparse, fall back to the persisted feature store for the
    coin. Only reject when no feature data exists anywhere -- which is
    the genuine "not ready" case (brand-new coin, feature cycle never
    ran). This is what lets options-flow and non-BTC/ETH/SOL signals
    through without weakening the gate: the candle features for those
    coins really do exist in the store, the signal object just didn't
    carry them.
    """
    min_overlap = _min_feature_overlap()
    context = getattr(signal, "context", None)
    if isinstance(context, dict):
        ctx_features = context.get("features")
        if _usable_feature_count(ctx_features) >= min_overlap:
            return True, "ok"

    # Fallback: persisted feature store for this coin.
    coin = str(getattr(signal, "coin", "") or "").upper()
    if isinstance(signal, dict):
        coin = coin or str(signal.get("coin", "") or "").upper()
    if coin:
        try:
            from src.data import feature_store
            stored = feature_store.get_feature_vector(coin, "1h")
        except Exception as exc:
            logger.debug("data_readiness: feature_store lookup failed: %s", exc)
            stored = None
        if isinstance(stored, dict):
            # ``timestamp_ms`` is bookkeeping, not a feature.
            usable = sum(
                1 for k, v in stored.items()
                if k != "timestamp_ms"
                and isinstance(v, (int, float))
                and float(v) != 0.0
            )
            if usable >= min_overlap:
                return True, "ok_from_feature_store"

    ctx_usable = (
        _usable_feature_count(context.get("features"))
        if isinstance(context, dict) else 0
    )
    return (
        False,
        f"feature_vector_sparse:ctx={ctx_usable},store=0<{min_overlap}",
    )


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
