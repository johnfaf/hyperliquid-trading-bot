"""Trailing stop policy (default OFF).

Trails the SL behind the favourable-direction high-water mark by a
fixed offset.  Captures profit on runners that exceed the original TP
distance and protects against sudden reversals.

  * For a long: track the HIGHEST price seen since entry; SL trails
    ``TRAILING_STOP_OFFSET_PCT`` below that high-water mark.
  * For a short: track the LOWEST price seen since entry; SL trails
    ``TRAILING_STOP_OFFSET_PCT`` above the low-water mark.

Unlike time-decay (PR C), this policy can only ever tighten SL when
price moves FAVOURABLY.  Adverse moves leave SL alone -- the floor
holds, the trail just hasn't moved up yet.

To avoid spamming the exchange with tiny SL adjustments on noise, the
policy only moves SL when the proposed new SL is at least
``TRAILING_STOP_MIN_STEP_PCT`` higher (long) or lower (short) than the
current SL.

Layered safety gates (every gate must pass):

  1. ``TRAILING_STOP_ENABLED`` master switch is True.
  2. Live trading is active for the container.
  3. Position has a resolvable mid price.
  4. Position has been held >= TRAILING_STOP_ACTIVATION_PROFIT_PCT
     in profit (only start trailing once we've locked in *some*
     gain -- otherwise we'd trail the SL down on the FIRST move
     past entry, which conflicts with the break-even path).
  5. An active SL trigger order exists.
  6. Proposed new SL is at least MIN_STEP_PCT past the current SL.
  7. New SL is strictly tighter than current SL (sl_is_tighter).
  8. ``TRAILING_STOP_DRY_RUN`` is False (default True: log-only).

State: per (coin, side) high-water / low-water mark, in-memory.
On bot restart, the mark resets to the current price -- worst case
the trail is "too loose" for one cycle until the next favourable move
re-establishes the mark.  We accept that vs. persisting state to DB.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

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

logger = logging.getLogger(__name__)


__all__ = ["evaluate_trailing_stop"]


# Module-level high/low-water mark: (coin, side) -> best favourable price seen.
_water_marks: Dict[Tuple[str, str], float] = {}


def _update_water_mark(coin: str, side: str, current_price: float) -> float:
    """Update the per-(coin,side) HWM/LWM and return the new value."""
    key = (coin.upper(), side.lower())
    prev = _water_marks.get(key)
    if prev is None:
        _water_marks[key] = current_price
        return current_price
    if side == "long":
        new = max(prev, current_price)
    elif side == "short":
        new = min(prev, current_price)
    else:
        return prev
    _water_marks[key] = new
    return new


def _reset_water_mark(coin: str, side: str) -> None:
    _water_marks.pop((coin.upper(), side.lower()), None)


def _compute_trailing_sl(side: str, water_mark: float, offset_pct: float) -> float:
    """Compute the trailing SL price.

    For a long: SL = HWM * (1 - offset)
    For a short: SL = LWM * (1 + offset)
    """
    if water_mark <= 0 or offset_pct <= 0:
        return 0.0
    if side == "long":
        return water_mark * (1.0 - offset_pct)
    if side == "short":
        return water_mark * (1.0 + offset_pct)
    return 0.0


def evaluate_trailing_stop(container) -> None:
    """Walk live positions; trail SL behind favourable-direction water marks.

    Never raises.  No-op when ``TRAILING_STOP_ENABLED`` is False.
    """
    if not bool(getattr(config, "TRAILING_STOP_ENABLED", False)):
        return
    trader = get_live_trader(container)
    if not trader or not is_live_trading_active(container):
        return

    activation_pct = float(getattr(
        config, "TRAILING_STOP_ACTIVATION_PROFIT_PCT", 0.01,
    ))
    offset_pct = float(getattr(config, "TRAILING_STOP_OFFSET_PCT", 0.01))
    min_step_pct = float(getattr(config, "TRAILING_STOP_MIN_STEP_PCT", 0.002))
    dry_run = bool(getattr(config, "TRAILING_STOP_DRY_RUN", True))

    try:
        positions = trader.get_positions(force_fresh=True) or []
    except Exception as exc:
        logger.debug("trailing_stop get_positions failed: %s", exc)
        return
    if not positions:
        # Reset water-marks for coins we no longer hold (so a fresh
        # entry on the same coin doesn't inherit a stale mark).
        _water_marks.clear()
        return

    try:
        mids = get_all_mids() or {}
    except Exception as exc:
        logger.debug("trailing_stop get_all_mids failed: %s", exc)
        return

    # Clear marks for coins not in current positions (re-entries start fresh).
    held_keys = {
        (str(p.get("coin", "")).upper(), position_side(p))
        for p in positions
    }
    stale_keys = set(_water_marks.keys()) - held_keys
    for key in stale_keys:
        _water_marks.pop(key, None)

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

            # Update water mark FIRST so future cycles see the latest
            # high/low regardless of whether we act this cycle.
            water = _update_water_mark(coin, side, current)

            # Gate: minimum profit before trailing activates.  We want
            # the break-even path to handle the early-profit phase; the
            # trail should only kick in once we're meaningfully ahead.
            p_pct = profit_pct(entry, current, side)
            if p_pct < activation_pct:
                continue

            new_sl = _compute_trailing_sl(side, water, offset_pct)
            if new_sl <= 0:
                continue

            # Fetch the active SL.
            _pos2, sl_order = fetch_position_and_sl(trader, coin)
            if sl_order is None:
                logger.debug(
                    "trailing_stop: %s %s no active SL found; skipping",
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

            # Tighter-only guard (defence in depth -- we also enforce
            # MIN_STEP_PCT below).
            if not sl_is_tighter(side=side, new_sl=new_sl, current_sl=current_sl):
                continue

            # Min-step guard: don't spam the exchange with sub-MIN_STEP moves.
            step_size = abs(new_sl - current_sl) / max(current_sl, 1e-9)
            if step_size < min_step_pct:
                continue

            size_qty = position_size(pos)
            if size_qty <= 0:
                continue

            if dry_run:
                logger.warning(
                    "[DRY-RUN] trailing_stop WOULD move %s %s SL: "
                    "current=%.6f -> new=%.6f (hwm=%.6f, profit=%.2f%%, "
                    "step=%.2f%%, offset=%.2f%%). Set "
                    "TRAILING_STOP_DRY_RUN=false to enable.",
                    side.upper(), coin, current_sl, new_sl, water,
                    p_pct * 100.0, step_size * 100.0, offset_pct * 100.0,
                )
                continue

            old_oid_raw = sl_order.get("oid") or sl_order.get("order_id")
            try:
                old_oid = int(old_oid_raw)
            except (TypeError, ValueError):
                logger.error(
                    "trailing_stop: %s SL has no valid oid (%s); skipping",
                    coin, old_oid_raw,
                )
                continue

            logger.warning(
                "trailing_stop: trailing %s %s SL %.6f -> %.6f "
                "(hwm=%.6f, step=%.2f%%)",
                side.upper(), coin, current_sl, new_sl, water,
                step_size * 100.0,
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
                    "trailing_stop: SL replacement FAILED for %s -- "
                    "position keeps original SL.  Manual review.",
                    coin,
                )
        except Exception as exc:
            logger.debug(
                "trailing_stop per-position eval failed for %s: %s",
                pos.get("coin", "?"), exc,
            )
            continue
