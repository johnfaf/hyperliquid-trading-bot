"""
Protective Leg Classification
==============================
Pure helpers that classify open Hyperliquid orders as stop-loss or
take-profit legs and validate that they actually protect the position they
claim to. Extracted from live_trader.py.

The reduce-only / trigger / SL-vs-TP / size-coverage rules in this module
were the source of multiple "phantom protection" bugs in the past, so the
behaviour here is intentionally conservative — a leg has to clear every
check (correct kind, reduce-only, correct side, trigger on the right side
of entry, size >= 90% of position) before it counts as protecting the
position.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def is_protective_order(order: Dict[str, Any], coin: Optional[str] = None) -> bool:
    if not isinstance(order, dict):
        return False
    if coin and str(order.get("coin", "") or "").upper() != str(coin).upper():
        return False
    reduce_only = bool(order.get("reduceOnly") or order.get("reduce_only"))
    is_trigger = bool(order.get("isTrigger") or order.get("is_trigger"))
    is_position_tpsl = bool(order.get("isPositionTpsl") or order.get("is_position_tpsl"))
    order_type = str(order.get("orderType") or order.get("type") or "").lower()
    trigger_condition = str(order.get("triggerCondition") or "").lower()
    trigger_px = order.get("triggerPx") or order.get("trigger_px")
    has_trigger_px = trigger_px not in (None, "", "0", "0.0", 0, 0.0)
    return bool(
        reduce_only
        or is_trigger
        or is_position_tpsl
        or "stop" in order_type
        or "take" in order_type
        or "profit" in order_type
        or "trigger" in order_type
        or "price above" in trigger_condition
        or "price below" in trigger_condition
        or has_trigger_px
    )


def classify_protective_leg(order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Classify a single open-order entry as an SL or TP leg.

    Returns a dict with ``coin``, ``leg`` ("sl"/"tp"), ``side`` ("buy"/"sell"),
    ``trigger_px`` (float or None), ``size`` (float), ``reduce_only`` (bool),
    ``oid`` (int or None), and ``order_type`` (str). Returns None for orders
    that are NOT protective (entries, plain reduce-only limits, etc.).

    Hyperliquid represents trigger orders with:
      - ``t.trigger.tpsl`` = ``"sl"`` or ``"tp"`` (authoritative), or
      - ``orderType`` containing "stop"/"take" when the nested form is absent.
    """
    if not isinstance(order, dict):
        return None
    coin = str(order.get("coin", "") or "")
    if not coin:
        return None
    is_position_tpsl = bool(order.get("isPositionTpsl") or order.get("is_position_tpsl"))
    reduce_only = bool(order.get("reduceOnly") or order.get("reduce_only") or is_position_tpsl)
    order_type = str(order.get("orderType") or order.get("type") or "").lower()
    trigger_condition = str(
        order.get("triggerCondition") or order.get("trigger_condition") or ""
    ).lower()

    leg: Optional[str] = None
    trigger_px_raw: Any = None
    trigger_nested = order.get("t")
    if isinstance(trigger_nested, dict):
        trig = trigger_nested.get("trigger")
        if isinstance(trig, dict):
            tpsl = str(trig.get("tpsl", "") or "").lower()
            if tpsl in ("sl", "tp"):
                leg = tpsl
            trigger_px_raw = trig.get("triggerPx")
    if trigger_px_raw is None:
        trigger_px_raw = order.get("triggerPx") or order.get("trigger_px")

    if leg is None:
        if "stop" in order_type:
            leg = "sl"
        elif "take" in order_type or "profit" in order_type:
            leg = "tp"

    side = None
    if "b" in order and isinstance(order.get("b"), bool):
        side = "buy" if order["b"] else "sell"
    else:
        for side_key in ("side", "direction", "action"):
            if not isinstance(order.get(side_key), str):
                continue
            raw_side = order[side_key].strip().lower()
            if raw_side in ("buy", "b", "bid") or "close short" in raw_side:
                side = "buy"
                break
            if raw_side in ("sell", "s", "a", "ask") or "close long" in raw_side:
                side = "sell"
                break

    if leg is None and trigger_condition:
        # Hyperliquid sometimes omits ``t.trigger.tpsl`` and exposes only
        # "Close Long/Short" plus "Price above/below". Infer leg from
        # close side + trigger direction:
        #   close long  (sell): below=SL, above=TP
        #   close short (buy):  above=SL, below=TP
        if side == "sell":
            if "price below" in trigger_condition:
                leg = "sl"
            elif "price above" in trigger_condition:
                leg = "tp"
        elif side == "buy":
            if "price above" in trigger_condition:
                leg = "sl"
            elif "price below" in trigger_condition:
                leg = "tp"

    if leg is None:
        return None

    size_raw = order.get("sz")
    try:
        parsed_size_raw = float(size_raw) if size_raw not in (None, "") else 0.0
    except (TypeError, ValueError):
        parsed_size_raw = 0.0
    if parsed_size_raw <= 0:
        size_raw = order.get("origSz") or order.get("orig_sz") or order.get("s") or order.get("size")
    try:
        size_val = abs(float(size_raw)) if size_raw not in (None, "") else 0.0
    except (TypeError, ValueError):
        size_val = 0.0
    try:
        trigger_px_val = float(trigger_px_raw) if trigger_px_raw not in (None, "") else None
    except (TypeError, ValueError):
        trigger_px_val = None

    oid = order.get("oid") or order.get("order_id") or order.get("id")
    try:
        oid_val = int(oid) if oid is not None else None
    except (TypeError, ValueError):
        oid_val = None

    return {
        "coin": coin,
        "leg": leg,
        "side": side,
        "trigger_px": trigger_px_val,
        "size": size_val,
        "reduce_only": reduce_only,
        "oid": oid_val,
        "order_type": order_type,
    }


def split_valid_legs(
    legs: List[Dict[str, Any]],
    *,
    leg_kind: str,
    position_side: str,
    protect_side: str,
    entry_price: float,
    position_size: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition classified legs into (valid, invalid) for one position.

    A leg is only accepted as protecting the position if ALL of these hold:
      - ``leg`` matches ``leg_kind`` (sl vs tp)
      - ``reduce_only`` is True (otherwise it could add exposure)
      - ``side`` equals ``protect_side`` (long needs sell-side, short needs buy)
      - ``trigger_px`` is on the correct side of entry:
          SL long  -> trigger_px < entry
          SL short -> trigger_px > entry
          TP long  -> trigger_px > entry
          TP short -> trigger_px < entry
      - ``size`` covers >= 90% of position (rounding tolerance; below that
        the leg is treated as stale and listed in `invalid` for cleanup).

    Returns ``(valid_list, invalid_list)`` where each invalid entry has an
    ``invalid_reason`` string attached for logging.
    """
    valid: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []
    min_coverage = max(position_size * 0.9, 0.0)
    for leg in legs:
        reasons: List[str] = []
        if leg.get("leg") != leg_kind:
            reasons.append(f"leg_mismatch:{leg.get('leg')}")
        if not leg.get("reduce_only"):
            reasons.append("not_reduce_only")
        if leg.get("side") != protect_side:
            reasons.append(f"wrong_side:{leg.get('side')}")
        trigger_px = leg.get("trigger_px")
        if trigger_px is None:
            reasons.append("no_trigger_px")
        else:
            if leg_kind == "sl":
                if position_side == "long" and trigger_px >= entry_price:
                    reasons.append("sl_above_entry_on_long")
                elif position_side == "short" and trigger_px <= entry_price:
                    reasons.append("sl_below_entry_on_short")
            elif leg_kind == "tp":
                if position_side == "long" and trigger_px <= entry_price:
                    reasons.append("tp_below_entry_on_long")
                elif position_side == "short" and trigger_px >= entry_price:
                    reasons.append("tp_above_entry_on_short")
        if leg.get("size", 0.0) < min_coverage:
            reasons.append(
                f"undersized:{leg.get('size'):.6f}<{min_coverage:.6f}"
            )
        if reasons:
            leg["invalid_reason"] = ",".join(reasons)
            invalid.append(leg)
        else:
            valid.append(leg)
    return valid, invalid
