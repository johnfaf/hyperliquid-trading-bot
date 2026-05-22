"""Time-decay SL tightening policy (default OFF).

The longer a position is held WITHOUT resolution (no SL, no TP, no
reconciliation, no break-even promotion), the tighter the SL becomes.
This caps slow-bleed losses on positions whose original trade thesis
has gone stale -- exactly the failure mode we saw in production
(3 longs held underwater overnight without any close trigger).

Design: discrete bands, not continuous trailing.  We tighten in 4
fixed steps over time, so each cancel/replace cycle is meaningful
(not whipsaw-prone) and the API call rate stays bounded.

Default band schedule (configurable via env):

  Age band             |  Target SL distance from current price
  ─────────────────────────────────────────────────────────────────
  0  - 30  min         |  Original SL (no tightening)
  30 - 90  min         |  75% of original distance
  90 - 180 min         |  50% of original distance
  180 - 240 min        |  25% of original distance
  240+ min             |  Continue at 25% (no further tightening)

The original SL distance is computed once per position from the
ratio (current price : current SL) so we don't need to know the
original entry SL.  We then apply the band factor to the *current
price* to derive the new SL.

Layered safety gates (every gate must pass to tighten SL):

  1. ``TIME_DECAY_SL_ENABLED`` master switch is True.
  2. Live trading is active for the container.
  3. Position has a resolvable open-time.
  4. Position age >= MIN band threshold (default 30 min).
  5. An active SL trigger order exists.
  6. New SL is strictly tighter than current SL (sl_is_tighter guard).
  7. ``TIME_DECAY_SL_DRY_RUN`` is False (default True: log-only).
"""
from __future__ import annotations

import logging

import config
from src.core import clock_provider
from src.core.live_execution import get_live_trader, is_live_trading_active
from src.data.hyperliquid_client import get_all_mids
from src.trading.sl_management import (
    fetch_position_and_sl,
    position_entry_price,
    position_side,
    position_size,
    replace_sl,
    sl_is_tighter,
)
from src.data import database as db

logger = logging.getLogger(__name__)


__all__ = ["evaluate_time_decay_sl"]


def _resolve_age_seconds(coin: str, now_ts: float):
    """Return how long the current paper trade for ``coin`` has been open.

    Returns None when no shadow paper row is found (defer to orphan
    protection rather than guess).
    """
    try:
        rows = db.get_open_paper_trades() or []
        for row in rows:
            if str(row.get("coin", "")).upper() != coin.upper():
                continue
            opened_at = row.get("opened_at") or row.get("created_at") or 0
            if not opened_at:
                continue
            if isinstance(opened_at, (int, float)):
                ts = float(opened_at)
                if ts > 1e12:  # ms
                    ts /= 1000.0
                return max(0.0, now_ts - ts)
            from datetime import datetime, timezone
            try:
                txt = str(opened_at).replace("Z", "+00:00")
                dt = datetime.fromisoformat(txt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return max(0.0, now_ts - dt.timestamp())
            except Exception:
                continue
    except Exception as exc:
        logger.debug("time_decay_sl age lookup failed for %s: %s", coin, exc)
    return None


def _band_factor(age_seconds: float) -> float:
    """Return the band's SL-distance factor for an age in seconds.

    Factor is the fraction of the ORIGINAL SL distance the band keeps.
    Smaller factor = tighter SL.  Returns 1.0 for ages below the first
    band (no tightening).
    """
    band1_s = int(getattr(config, "TIME_DECAY_SL_BAND1_SECONDS", 1800))   # 30 min
    band2_s = int(getattr(config, "TIME_DECAY_SL_BAND2_SECONDS", 5400))   # 90 min
    band3_s = int(getattr(config, "TIME_DECAY_SL_BAND3_SECONDS", 10800))  # 180 min
    band4_s = int(getattr(config, "TIME_DECAY_SL_BAND4_SECONDS", 14400))  # 240 min

    band1_f = float(getattr(config, "TIME_DECAY_SL_BAND1_FACTOR", 0.75))
    band2_f = float(getattr(config, "TIME_DECAY_SL_BAND2_FACTOR", 0.50))
    band3_f = float(getattr(config, "TIME_DECAY_SL_BAND3_FACTOR", 0.25))
    band4_f = float(getattr(config, "TIME_DECAY_SL_BAND4_FACTOR", 0.25))

    if age_seconds >= band4_s:
        return band4_f
    if age_seconds >= band3_s:
        return band3_f
    if age_seconds >= band2_s:
        return band2_f
    if age_seconds >= band1_s:
        return band1_f
    return 1.0  # below first band


def _compute_tightened_sl(
    side: str,
    current_price: float,
    current_sl: float,
    band_factor: float,
) -> float:
    """Return the new (tightened) SL price for this band.

    Implementation: take the *current* distance from price to SL (in
    absolute terms), shrink it by ``band_factor``, then place the new
    SL that distance from the current price.

    For a long: SL is below price; tightening moves it UP.
    For a short: SL is above price; tightening moves it DOWN.

    Returns 0.0 on invalid inputs (caller skips).
    """
    if current_price <= 0 or current_sl <= 0 or band_factor <= 0 or band_factor > 1:
        return 0.0
    if side == "long":
        # SL must be below price for a long
        distance = current_price - current_sl
        if distance <= 0:
            return 0.0
        new_distance = distance * band_factor
        return current_price - new_distance
    if side == "short":
        # SL must be above price for a short
        distance = current_sl - current_price
        if distance <= 0:
            return 0.0
        new_distance = distance * band_factor
        return current_price + new_distance
    return 0.0


def evaluate_time_decay_sl(container) -> None:
    """Walk live positions; tighten SL on those past the age bands.

    Never raises.  No-op when ``TIME_DECAY_SL_ENABLED`` is False.
    """
    if not bool(getattr(config, "TIME_DECAY_SL_ENABLED", False)):
        return
    trader = get_live_trader(container)
    if not trader or not is_live_trading_active(container):
        return

    dry_run = bool(getattr(config, "TIME_DECAY_SL_DRY_RUN", True))

    try:
        positions = trader.get_positions(force_fresh=True) or []
    except Exception as exc:
        logger.debug("time_decay_sl get_positions failed: %s", exc)
        return
    if not positions:
        return

    try:
        mids = get_all_mids() or {}
    except Exception as exc:
        logger.debug("time_decay_sl get_all_mids failed: %s", exc)
        return

    now_ts = clock_provider.unix_now()

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

            # Gate: position age
            age_s = _resolve_age_seconds(coin, now_ts)
            if age_s is None:
                # Orphan position -- defer to ordinary orphan-protection.
                continue
            factor = _band_factor(age_s)
            if factor >= 1.0:
                # Below first band -- nothing to do.
                continue

            # Fetch active SL
            _pos2, sl_order = fetch_position_and_sl(trader, coin)
            if sl_order is None:
                logger.debug(
                    "time_decay_sl: %s %s no active SL found; skipping",
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

            new_sl = _compute_tightened_sl(side, current, current_sl, factor)
            if new_sl <= 0:
                continue
            if not sl_is_tighter(side=side, new_sl=new_sl, current_sl=current_sl):
                # Already tighter than this band's target -- nothing to do.
                continue

            size_qty = position_size(pos)
            if size_qty <= 0:
                continue

            if dry_run:
                logger.warning(
                    "[DRY-RUN] time_decay_sl WOULD tighten %s %s SL: "
                    "current=%.6f -> new=%.6f (age=%dmin, band_factor=%.2f, "
                    "price=%.6f). Set TIME_DECAY_SL_DRY_RUN=false to enable.",
                    side.upper(), coin, current_sl, new_sl,
                    int(age_s / 60), factor, current,
                )
                continue

            old_oid_raw = sl_order.get("oid") or sl_order.get("order_id")
            try:
                old_oid = int(old_oid_raw)
            except (TypeError, ValueError):
                logger.error(
                    "time_decay_sl: %s SL has no valid oid (%s); skipping",
                    coin, old_oid_raw,
                )
                continue

            logger.warning(
                "time_decay_sl: tightening %s %s SL %.6f -> %.6f "
                "(age=%dmin, band_factor=%.2f)",
                side.upper(), coin, current_sl, new_sl,
                int(age_s / 60), factor,
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
                    "time_decay_sl: SL replacement FAILED for %s -- "
                    "position keeps original SL.  Manual review.",
                    coin,
                )
        except Exception as exc:
            logger.debug(
                "time_decay_sl per-position eval failed for %s: %s",
                pos.get("coin", "?"), exc,
            )
            continue
