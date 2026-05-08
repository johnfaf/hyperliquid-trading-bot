"""
Live Trader Helpers
====================
Pure functions and static helpers extracted from live_trader.py to keep that
file focused on the order-flow state machine.

Public surface:
  - _derive_missing_live_drawdown_cap
  - coerce_float
  - normalize_coin_key
  - signal_source_key
  - signal_side_value
  - resolve_signal_stop_roe_pct
  - normalize_order_side
  - make_cloid
  - parse_iso_timestamp

These are all stateless (no `self` access) and have no side effects beyond
returning a value. They are re-exported from live_trader so existing imports
(tests, downstream callers) keep working.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Optional

import config
from src.core.env_utils import safe_env_float as _safe_env_float

try:
    from eth_utils import keccak as _keccak  # type: ignore
    HAS_KECCAK = True
except ImportError:  # pragma: no cover
    _keccak = None  # type: ignore[assignment]
    HAS_KECCAK = False


def _derive_missing_live_drawdown_cap(
    *,
    max_daily_loss: float,
    max_position_size: float,
    max_order_usd: float,
    min_order_usd: float,
) -> float:
    """Derive a conservative rolling drawdown cap when the env var is absent.

    This does not replace an explicit operator cap. It only prevents a live
    deployment from running uncapped when the per-tier/per-order limits already
    define a tighter safety envelope.
    """
    candidates = []
    if max_daily_loss > 0:
        candidates.append(float(max_daily_loss) * 0.25)
    if max_position_size > 0:
        candidates.append(float(max_position_size) * 0.10)
    if max_order_usd > 0:
        candidates.append(float(max_order_usd) * 0.25)
    if not candidates:
        return 0.0
    default_min_cap = max(1.0, float(min_order_usd) * 0.10) if min_order_usd > 0 else 1.0
    min_cap = _safe_env_float("LIVE_MIN_DRAWDOWN_CAP_USD", default_min_cap, lo=0.0, hi=1e6)
    return max(float(min_cap), min(candidates))


def coerce_float(value: Any, default: float = 0.0) -> float:
    """Safely coerce a Hyperliquid API field to float.

    Hyperliquid returns many numeric fields as strings ("0.123"), sometimes
    as nested dicts ({"value": 5}), and occasionally as ``null``. Plain
    ``float(x)`` crashes on dicts and None.
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value) if value else default
        except ValueError:
            return default
    if isinstance(value, dict):
        # Hyperliquid encodes leverage as {"type": "cross", "value": 5} and
        # cum funding as {"allTime": "0.123", ...}.  Prefer "value" then
        # "allTime" then any numeric-looking entry.
        for key in ("value", "allTime", "sinceOpen", "sinceChange"):
            if key in value:
                return coerce_float(value[key], default)
        return default
    return default


def normalize_coin_key(coin: Any) -> str:
    return str(coin or "").strip().upper()


def signal_source_key(signal: Any) -> str:
    """Stable per-source key for throughput caps and policy lookup.

    Copy-trade caps apply per copied trader, not as one global "copy_trade"
    bucket — otherwise one early fill can starve all remaining copy signals
    for the day and create side skew.
    """
    source = getattr(signal, "source", None)
    if hasattr(source, "value"):
        source = source.value
    key = str(source or "unknown").strip().lower() or "unknown"

    if key == "copy_trade":
        trader_address = str(getattr(signal, "trader_address", "") or "").strip().lower()
        if trader_address:
            return f"{key}:{trader_address}"
        return key

    strategy_type = str(getattr(signal, "strategy_type", "") or "").strip().lower()
    if strategy_type:
        return f"{key}:{strategy_type}"

    return key


def signal_side_value(signal: Any) -> str:
    side = getattr(signal, "side", None)
    if hasattr(side, "value"):
        side = side.value
    return str(side or "").strip().lower()


def resolve_signal_stop_roe_pct(signal: Any) -> float:
    """Resolve the signal stop distance in margin/ROE space."""
    risk = getattr(signal, "risk", None)
    leverage = max(float(getattr(signal, "leverage", 1.0) or 1.0), 1.0)
    if risk is not None and hasattr(risk, "resolve_roe_stop_loss_pct"):
        try:
            stop_roe = float(risk.resolve_roe_stop_loss_pct(leverage))
            if stop_roe > 0:
                return stop_roe
        except Exception:
            pass

    context = getattr(signal, "context", {}) or {}
    if isinstance(context, dict):
        risk_policy = context.get("risk_policy", {}) or {}
        if isinstance(risk_policy, dict):
            try:
                stop_roe = float(risk_policy.get("stop_roe_pct", 0.0) or 0.0)
                if stop_roe > 0:
                    return stop_roe
            except (TypeError, ValueError):
                pass

    return max(float(getattr(config, "PAPER_TRADING_STOP_LOSS_PCT", 0.05) or 0.05), 0.001)


def normalize_order_side(side: str) -> str:
    """Normalize long/short or buy/sell inputs to buy/sell."""
    value = str(side or "").strip().lower()
    if value in {"buy", "long"}:
        return "buy"
    if value in {"sell", "short"}:
        return "sell"
    raise ValueError(f"Unsupported order side: {side}")


def make_cloid(*salt_parts: Any) -> str:
    """Produce an exchange-level client order ID.

    Hyperliquid supports a ``c`` field on each order, a 128-bit hex string.
    When present, the exchange rejects a *resubmission* of the same cloid as
    a duplicate — that is the authoritative idempotency guarantee. Our local
    dedup cache is client-side only; it cannot help when our process crashes
    and restarts between signing and fill. The cloid fills that gap.

    We derive the cloid from a keccak hash of the caller-supplied salt parts
    so retries of the *same logical order* produce the same cloid and benefit
    from exchange dedup, while genuinely-different orders get unique cloids.

    Returns a lowercase hex string of the form ``0x`` + 32 hex chars as
    required by the exchange.
    """
    payload = "|".join(str(p) for p in salt_parts)
    if HAS_KECCAK and _keccak is not None:
        digest = _keccak(payload.encode("utf-8"))[:16]
    else:
        digest = hashlib.sha256(payload.encode("utf-8")).digest()[:16]
    return "0x" + digest.hex()


def parse_iso_timestamp(value: Any) -> Optional[datetime]:
    """Parse a wide range of timestamp shapes into a tz-aware UTC datetime."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        # Hyperliquid timestamps in fills are ms since epoch.
        try:
            ms = float(value)
        except (TypeError, ValueError):
            return None
        try:
            return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None


def extract_free_margin_from_state(state: Any) -> Optional[float]:
    """Extract free/available margin from a clearinghouse-state payload.

    Returns None when no usable margin field is present so callers can
    distinguish "account exists with $0 free" from "couldn't read account."
    """
    if not isinstance(state, dict) or not state:
        return None
    for key in ("marginSummary", "crossMarginSummary"):
        margin_summary = state.get(key, {}) or {}
        try:
            acct = float(margin_summary.get("accountValue", 0) or 0)
            used = float(margin_summary.get("totalMarginUsed", 0) or 0)
            if acct > 0 or used > 0:
                return max(0.0, acct - used)
        except (TypeError, ValueError):
            continue
    withdrawable = state.get("withdrawable")
    try:
        if withdrawable is not None:
            return max(0.0, float(withdrawable))
    except (TypeError, ValueError):
        pass
    return None


def signed_position_size_for_coin(coin: str, positions: Any) -> Optional[float]:
    """Return signed szi for a coin from an exchange position snapshot.

    Returns None when ``positions`` is None (truly unknown) and 0.0 when
    the snapshot exists but doesn't list the coin (flat).
    """
    if positions is None:
        return None
    wanted = str(coin or "").upper()
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        if str(pos.get("coin", "") or "").upper() != wanted:
            continue
        return coerce_float(pos.get("szi", pos.get("size", 0)), 0.0)
    return 0.0


def shadow_trade_metadata(trade: Any) -> dict:
    """Decode a shadow-trade row's `metadata` field into a dict.

    Stored as JSON string in some DB rows; native dict in others.
    """
    import json
    if not isinstance(trade, dict):
        return {}
    metadata = trade.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata or "{}")
        except Exception:
            metadata = {}
    return dict(metadata or {})
