"""
Order Result Parsing
=====================
Pure helpers that interpret Hyperliquid order-response payloads. Extracted
from live_trader.py so that the order-flow file doesn't have to also own
the wire-format parsing.

Hyperliquid uses a two-level success model: the outer request can be
``status: ok`` while the inner per-order ``statuses`` list contains
``{"error": "..."}`` rejections — every helper below understands this.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.trading.live_trader_helpers import coerce_float

logger = logging.getLogger(__name__)


def extract_inner_order_statuses(result: Optional[Dict]) -> List[Dict[str, Any]]:
    """Pull the per-order ``statuses`` list out of a Hyperliquid response.

    Hyperliquid wraps successful requests in
    ``{"status": "ok", "response": {"type": "order", "data": {"statuses": [...]}}}``
    where each entry is one of:
      - ``{"resting": {"oid": int}}``     — posted, waiting to match
      - ``{"filled": {"oid": int, "totalSz": str, "avgPx": str}}``
      - ``{"error": "..."}``              — per-order rejection, outer
                                            status is STILL ``"ok"``

    Returns ``[]`` if the shape doesn't match.
    """
    if not isinstance(result, dict):
        return []
    response = result.get("response")
    if not isinstance(response, dict):
        return []
    data = response.get("data")
    if not isinstance(data, dict):
        return []
    statuses = data.get("statuses")
    if isinstance(statuses, list):
        return [s for s in statuses if isinstance(s, dict)]
    return []


def extract_reported_fill_size(result: Optional[Dict]) -> Optional[float]:
    """Return the exchange-reported filled size from an order response, if any."""
    total = 0.0
    found = False
    for entry in extract_inner_order_statuses(result):
        filled = entry.get("filled")
        if not isinstance(filled, dict):
            continue
        total_sz = coerce_float(filled.get("totalSz"), 0.0)
        if total_sz > 0:
            total += total_sz
            found = True
    return total if found else None


def extract_reported_fill_price(result: Optional[Dict]) -> Optional[float]:
    """Return size-weighted average fill price from an order response, if any."""
    weighted_notional = 0.0
    total_size = 0.0
    for entry in extract_inner_order_statuses(result):
        filled = entry.get("filled")
        if not isinstance(filled, dict):
            continue
        total_sz = coerce_float(filled.get("totalSz"), 0.0)
        avg_px = coerce_float(filled.get("avgPx"), 0.0)
        if total_sz <= 0 or avg_px <= 0:
            continue
        weighted_notional += total_sz * avg_px
        total_size += total_sz
    if total_size > 0:
        return weighted_notional / total_size
    return None


def extract_resting_order_ids(result: Optional[Dict]) -> List[int]:
    """Return order ids that were accepted but are still resting."""
    out: List[int] = []
    for entry in extract_inner_order_statuses(result):
        resting = entry.get("resting")
        if not isinstance(resting, dict):
            continue
        oid = resting.get("oid") or resting.get("order_id") or resting.get("id")
        try:
            out.append(int(oid))
        except (TypeError, ValueError):
            continue
    return out


def is_order_result_success(result: Optional[Dict]) -> bool:
    """Best-effort classification of exchange responses into success/failure.

    Outer status="ok" is necessary but not sufficient — wire-format rejections
    surface as inner-statuses errors with the outer wrapper still claiming
    success. Treating those as success has previously caused fill-verification
    to poll pointlessly for ten seconds before reporting a phantom failure.
    """
    if not result:
        return False
    if not isinstance(result, dict):
        return bool(result)
    status = str(result.get("status", "")).strip().lower()
    if not status:
        logger.warning("Order result has no status field: %s", result)
        return False
    if status not in {"success", "simulated", "verified", "filled", "accepted", "ok"}:
        return False

    inner = extract_inner_order_statuses(result)
    if inner and any("error" in entry for entry in inner):
        errors = [entry.get("error") for entry in inner if "error" in entry]
        logger.warning(
            "Order outer status was 'ok' but per-order statuses contain "
            "errors: %s",
            errors,
        )
        return False
    return True


def is_insufficient_margin_rejection(result: Optional[Dict]) -> bool:
    """Return True when an order rejection is caused by insufficient margin."""
    if not isinstance(result, dict):
        return False

    reason = str(result.get("reason", "")).strip().lower()
    if "insufficient_margin" in reason:
        return True

    messages: List[str] = []
    errors = result.get("errors")
    if isinstance(errors, list):
        messages.extend(str(err) for err in errors if err is not None)
    elif errors is not None:
        messages.append(str(errors))

    message = result.get("message")
    if message:
        messages.append(str(message))

    return any("insufficient margin" in msg.lower() for msg in messages)


def coerce_exchange_leverage(leverage: float) -> int:
    """Hyperliquid updateLeverage requires an integer leverage value.

    Mirror live_trader's previous floor-plus-half behaviour exactly so this
    extraction is byte-identical with the previous staticmethod.
    """
    import math
    try:
        leverage_value = float(leverage)
    except (TypeError, ValueError):
        leverage_value = 1.0
    if not math.isfinite(leverage_value) or leverage_value <= 0:
        leverage_value = 1.0
    return max(1, int(math.floor(leverage_value + 0.5)))
