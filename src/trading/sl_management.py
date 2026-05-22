"""Shared primitives for stop-loss management policies.

The three layered SL policies — break-even, time-decay, trailing —
all need the same mechanical operations:

  * Identify the current position for a coin (entry price, side, size).
  * Identify the active SL trigger order for that position.
  * Compute the profit % the position is currently sitting on.
  * Replace the existing SL trigger with a new one at a different price.

This module hosts those primitives.  Each individual policy module
(break_even_stop / time_decay_sl / trailing_stop) implements the
*decision* (when to move SL and to where); this module owns the
*mechanics* (cancel-old, place-new, idempotency, never-loosen guard).

Safety properties:
  * ``replace_sl`` never loosens the SL — by construction, the helper
    refuses to move SL further from current price than it already is.
  * Every helper here is non-raising; errors are logged at debug and
    callers get a clean Optional / bool return.
  * No state is held at module level — these are pure functions
    parameterised by the trader + position data the caller supplies.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def position_side(pos: Dict[str, Any]) -> str:
    """Return ``"long"`` / ``"short"`` for a Hyperliquid-shaped position dict.

    Prefers the normalised ``side`` field; falls back to the sign of
    ``szi`` (raw HL signed position size).  Returns ``""`` for an empty
    position or unrecognised shape.
    """
    side_raw = str(pos.get("side", "") or "").strip().lower()
    if side_raw in ("long", "short"):
        return side_raw
    try:
        szi = float(pos.get("szi", 0) or 0)
    except (TypeError, ValueError):
        return ""
    if szi > 0:
        return "long"
    if szi < 0:
        return "short"
    return ""


def position_entry_price(pos: Dict[str, Any]) -> float:
    """Return the entry price of a position, normalising HL field names."""
    for key in ("entry_price", "entryPx", "avgEntryPx", "entry_px"):
        val = pos.get(key)
        if val is None:
            continue
        try:
            f = float(val)
            if f > 0:
                return f
        except (TypeError, ValueError):
            continue
    return 0.0


def position_size(pos: Dict[str, Any]) -> float:
    """Return the absolute size of a position."""
    for key in ("size", "szi"):
        val = pos.get(key)
        if val is None:
            continue
        try:
            return abs(float(val))
        except (TypeError, ValueError):
            continue
    return 0.0


def profit_pct(entry: float, current: float, side: str) -> float:
    """Signed profit % (positive = favourable, negative = adverse).

    For a long: (current - entry) / entry
    For a short: (entry - current) / entry
    """
    if entry <= 0:
        return 0.0
    if side == "long":
        return (current - entry) / entry
    if side == "short":
        return (entry - current) / entry
    return 0.0


def find_sl_order(
    orders: List[Dict[str, Any]],
    coin: str,
    position_side_str: str,
) -> Optional[Dict[str, Any]]:
    """Locate the active SL trigger order for ``coin`` in ``orders``.

    Hyperliquid frontendOpenOrders returns trigger orders with
    ``orderType`` like ``"Stop Market"`` / ``"Stop Limit"`` plus a
    ``triggerCondition`` flag.  For a long the SL is a SELL trigger
    below current price; for a short it's a BUY trigger above.

    We pick the order whose ``coin`` matches AND whose closing side
    matches the position's side AND whose triggerPx is on the correct
    side of the entry (defensive: avoids matching a TP by mistake when
    the order_type field is missing in fallback responses).

    Returns None if no matching SL order is visible.
    """
    if not orders or not coin:
        return None
    close_side = "B" if position_side_str == "short" else "A"  # Hyperliquid uses A=ask=sell, B=bid=buy
    coin_u = coin.upper()
    for o in orders:
        if str(o.get("coin", "")).upper() != coin_u:
            continue
        # Trigger orders have ``isTrigger`` or ``triggerPx`` set; market
        # entries don't.  Without these fields we can't tell SL from TP
        # without the orderType, which the fallback openOrders response
        # omits — so we conservatively skip those.
        trigger_px = o.get("triggerPx") or o.get("trigger_price") or o.get("trigger")
        if trigger_px in (None, 0, "0"):
            continue
        # Reduce-only is the hallmark of a protective leg.
        if not bool(o.get("reduceOnly", o.get("reduce_only", False))):
            continue
        # Side check: SL on long is SELL (side=A), SL on short is BUY (side=B).
        side_field = str(o.get("side", "") or "").strip().upper()
        if side_field and side_field != close_side and side_field not in ("SELL", "BUY"):
            continue
        # Distinguish SL from TP via orderType when available.
        order_type = str(
            o.get("orderType")
            or o.get("order_type")
            or ""
        ).lower()
        if "tp" in order_type or "take" in order_type or "take_profit" in order_type:
            continue
        return dict(o)
    return None


def sl_is_tighter(
    *,
    side: str,
    new_sl: float,
    current_sl: float,
) -> bool:
    """Return True iff ``new_sl`` is a *tighter* (closer-to-favourable) stop
    than ``current_sl`` for a position of ``side``.

    For a long, tighter SL is HIGHER (closer to current price → smaller
    loss).  For a short, tighter SL is LOWER.  Equal counts as "not
    tighter" so we never spam cancel/replace for the same price.
    """
    if new_sl <= 0 or current_sl <= 0:
        return False
    if side == "long":
        return new_sl > current_sl
    if side == "short":
        return new_sl < current_sl
    return False


def replace_sl(
    trader: Any,
    coin: str,
    *,
    position_side_str: str,
    position_size_qty: float,
    old_sl_oid: int,
    new_sl_price: float,
) -> bool:
    """Cancel the old SL by oid and place a new SL at ``new_sl_price``.

    Returns True only when BOTH the cancel and the replace succeed.
    On any failure the position keeps its original SL — never leaves
    it unprotected.

    NOTE: this function does NOT enforce the "tighter SL only" rule;
    callers are expected to have already verified via ``sl_is_tighter``.
    The split exists so the policy modules can log a clean
    "would-tighten" event in DRY_RUN mode before making this call.
    """
    if not coin or position_size_qty <= 0 or new_sl_price <= 0:
        return False
    try:
        cancelled = bool(trader.cancel_order(coin, int(old_sl_oid)))
    except Exception as exc:
        logger.error(
            "sl_management.replace_sl: cancel old SL failed for %s oid=%s: %s",
            coin, old_sl_oid, exc,
        )
        return False
    if not cancelled:
        logger.warning(
            "sl_management.replace_sl: exchange refused to cancel old SL "
            "for %s oid=%s -- keeping original SL", coin, old_sl_oid,
        )
        return False
    # Determine the close side for the new SL.  Long → sell; short → buy.
    close_side = "sell" if position_side_str == "long" else "buy"
    try:
        result = trader.place_trigger_order(
            coin, close_side, position_size_qty, new_sl_price, tp_or_sl="sl",
        )
    except Exception as exc:
        logger.error(
            "sl_management.replace_sl: place new SL FAILED for %s @ %.6f: %s. "
            "POSITION IS NOW UNPROTECTED -- manual review required.",
            coin, new_sl_price, exc,
        )
        return False
    status = str((result or {}).get("status", "")).lower()
    if status in ("success", "ok", "filled", "resting"):
        logger.warning(
            "sl_management.replace_sl: replaced %s SL -> %.6f (new oid=%s)",
            coin, new_sl_price,
            (result or {}).get("oid") or (result or {}).get("order_id") or "?",
        )
        return True
    logger.error(
        "sl_management.replace_sl: new SL placement returned non-success "
        "status=%s for %s @ %.6f. POSITION IS NOW UNPROTECTED -- "
        "manual review required. Full result: %s",
        status, coin, new_sl_price, result,
    )
    return False


def fetch_position_and_sl(
    trader: Any, coin: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Convenience: fetch fresh position + active SL order for a coin.

    Returns ``(position, sl_order)`` or ``(None, None)`` on any error.
    Both elements may individually be ``None`` (e.g. position exists
    but no SL attached -- orphaned).
    """
    try:
        positions = trader.get_positions(force_fresh=True) or []
    except Exception as exc:
        logger.debug("sl_management.fetch_position_and_sl positions failed: %s", exc)
        return None, None
    pos = next(
        (p for p in positions if str(p.get("coin", "")).upper() == coin.upper()),
        None,
    )
    if not pos:
        return None, None
    try:
        orders = trader.get_open_orders(force_fresh=True) or []
    except Exception as exc:
        logger.debug("sl_management.fetch_position_and_sl orders failed: %s", exc)
        return pos, None
    sl = find_sl_order(orders, coin, position_side(pos))
    return pos, sl
