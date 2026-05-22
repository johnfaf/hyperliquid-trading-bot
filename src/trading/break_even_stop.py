"""Break-even stop policy (default OFF).

Once a position is profitable by >= ``BREAK_EVEN_STOP_TRIGGER_PCT``,
move its stop-loss to the **entry price** (or slightly past, with a
configurable buffer to cover fees).  The position now has a guaranteed
non-loss floor — worst case is closed at entry +/- fees.

This is the SAFEST of the layered SL policies because it can ONLY
improve outcomes:
  * It never moves SL further from price (the ``sl_is_tighter`` guard
    in ``sl_management`` rejects loosening attempts).
  * It only ever activates when the position is ALREADY profitable.
  * It can't whipsaw — the trigger is one-shot per position (the new
    SL stays at entry until the position closes; subsequent cycles
    see ``new_sl == current_sl`` and short-circuit via the
    not-tighter check).

Layered safety gates (every gate must pass to move SL):

  1. ``BREAK_EVEN_STOP_ENABLED`` master switch is True.
  2. Live trading is active for the container.
  3. Position has a resolvable entry price + current mid price.
  4. Current profit % >= ``BREAK_EVEN_STOP_TRIGGER_PCT``.
  5. An active SL trigger order exists (no orphan → defer to ordinary
     orphan-protection).
  6. New SL (= entry + buffer) is strictly tighter than current SL.
  7. ``BREAK_EVEN_STOP_DRY_RUN`` is False (default True: log-only).
"""
from __future__ import annotations

import logging

import config
from src.core.live_execution import get_live_trader, is_live_trading_active
from src.data.hyperliquid_client import get_all_mids
from src.trading.sl_management import (
    fetch_position_and_sl,
    position_entry_price,
    position_side,
    position_size,
    profit_pct,
    replace_sl,
    sl_is_tighter,
)

__all__ = ["evaluate_break_even_stop"]

logger = logging.getLogger(__name__)


def _compute_break_even_sl(side: str, entry: float, buffer_pct: float) -> float:
    """Return the new SL price at entry +/- buffer.

    For a long, buffer raises SL slightly ABOVE entry so even after fees
    the close is non-negative.  For a short, buffer lowers SL slightly
    BELOW entry.  ``buffer_pct`` is expressed as a fraction (0.001 = 10
    bps).  Negative or zero buffer collapses to exact entry.
    """
    if entry <= 0:
        return 0.0
    b = max(0.0, float(buffer_pct))
    if side == "long":
        return entry * (1.0 + b)
    if side == "short":
        return entry * (1.0 - b)
    return entry


def evaluate_break_even_stop(container) -> None:
    """Walk all open live positions; promote SL to break-even when eligible.

    Never raises.  No-op when ``BREAK_EVEN_STOP_ENABLED`` is False.
    """
    if not bool(getattr(config, "BREAK_EVEN_STOP_ENABLED", False)):
        return
    trader = get_live_trader(container)
    if not trader or not is_live_trading_active(container):
        return

    trigger_pct = float(getattr(config, "BREAK_EVEN_STOP_TRIGGER_PCT", 0.01))
    buffer_pct = float(getattr(config, "BREAK_EVEN_STOP_BUFFER_PCT", 0.001))
    dry_run = bool(getattr(config, "BREAK_EVEN_STOP_DRY_RUN", True))

    try:
        positions = trader.get_positions(force_fresh=True) or []
    except Exception as exc:
        logger.debug("break_even_stop get_positions failed: %s", exc)
        return
    if not positions:
        return

    try:
        mids = get_all_mids() or {}
    except Exception as exc:
        logger.debug("break_even_stop get_all_mids failed: %s", exc)
        return

    for pos in positions:
        try:
            coin = str(pos.get("coin", "") or "").upper()
            if not coin:
                continue
            side = position_side(pos)
            if side not in ("long", "short"):
                continue
            entry = position_entry_price(pos)
            try:
                current = float(mids.get(coin, 0) or 0)
            except (TypeError, ValueError):
                continue
            if entry <= 0 or current <= 0:
                continue

            p_pct = profit_pct(entry, current, side)
            if p_pct < trigger_pct:
                # Not profitable enough yet -- nothing to do.
                continue

            # Fetch the active SL (re-uses fresh positions internally).
            _pos, sl_order = fetch_position_and_sl(trader, coin)
            if sl_order is None:
                logger.debug(
                    "break_even_stop: %s %s no active SL found; skipping "
                    "(orphan-protection should attach one shortly).",
                    side.upper(), coin,
                )
                continue
            try:
                current_sl = float(
                    sl_order.get("triggerPx")
                    or sl_order.get("trigger_price")
                    or sl_order.get("trigger")
                    or 0
                )
            except (TypeError, ValueError):
                current_sl = 0.0
            if current_sl <= 0:
                continue

            new_sl = _compute_break_even_sl(side, entry, buffer_pct)
            if not sl_is_tighter(side=side, new_sl=new_sl, current_sl=current_sl):
                # SL already at-or-above break-even (long) / at-or-below (short).
                # Nothing to do.  This is the steady-state after first trigger.
                continue

            size_qty = position_size(pos)
            if size_qty <= 0:
                continue

            if dry_run:
                logger.warning(
                    "[DRY-RUN] break_even_stop WOULD move %s %s SL: "
                    "current=%.6f -> new=%.6f (entry=%.6f, profit=%.2f%%, "
                    "buffer=%.3f%%). Set BREAK_EVEN_STOP_DRY_RUN=false to "
                    "enable real SL replacement.",
                    side.upper(), coin, current_sl, new_sl, entry,
                    p_pct * 100.0, buffer_pct * 100.0,
                )
                continue

            old_oid_raw = sl_order.get("oid") or sl_order.get("order_id")
            try:
                old_oid = int(old_oid_raw)
            except (TypeError, ValueError):
                logger.error(
                    "break_even_stop: %s SL has no valid oid (%s); skipping",
                    coin, old_oid_raw,
                )
                continue

            logger.warning(
                "break_even_stop: promoting %s %s SL %.6f -> %.6f "
                "(profit=%.2f%%, buffer=%.3f%%)",
                side.upper(), coin, current_sl, new_sl,
                p_pct * 100.0, buffer_pct * 100.0,
            )
            ok = replace_sl(
                trader, coin,
                position_side_str=side,
                position_size_qty=size_qty,
                old_sl_oid=old_oid,
                new_sl_price=new_sl,
            )
            if not ok:
                logger.error(
                    "break_even_stop: SL replacement FAILED for %s -- "
                    "position keeps original SL.  Manual review.",
                    coin,
                )
        except Exception as exc:
            logger.debug(
                "break_even_stop per-position eval failed for %s: %s",
                pos.get("coin", "?"), exc,
            )
            continue
