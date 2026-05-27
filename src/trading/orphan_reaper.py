"""Orphan position reaper.

Orphan = a live position the bot found on the exchange but didn't open
itself. ``sync_shadow_book_to_live`` creates a synthetic paper trade to
track these so PnL accounting stays consistent, but it never closes
them -- the bot has no thesis for an unsolicited position, so it would
hold them forever by default.

This module provides an opt-in reaper that closes orphans past a
configurable age, with an optional break-even gate so the reaper
doesn't realize losses on positions the operator might want to manage
manually.

Opt-in by design: ``ORPHAN_REAPER_ENABLED=false`` by default. When you
turn it on, watch the next few cycles -- it can close positions the
operator opened by hand if the bot didn't observe the open.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import config
from src.data import database as db
from src.core.live_execution import get_live_trader

logger = logging.getLogger(__name__)


def _parse_meta(trade: Dict[str, Any]) -> Dict[str, Any]:
    raw = trade.get("metadata") or {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


def _is_orphan(trade: Dict[str, Any]) -> bool:
    meta = _parse_meta(trade)
    return bool(meta.get("orphan_found")) or meta.get("source") == "live_orphan"


def _orphan_age_hours(meta: Dict[str, Any], now: datetime) -> Optional[float]:
    raw = meta.get("orphan_found_at") or ""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (now - dt).total_seconds() / 3600.0)
    except (TypeError, ValueError):
        return None


def _approximate_pnl(trade: Dict[str, Any], current_price: float) -> float:
    """Return unrealised PnL in USD for an orphan trade.

    Best-effort: uses size * (price delta) * sign. Doesn't include
    funding or fees -- the break-even gate just wants a sign check.
    """
    try:
        side = str(trade.get("side") or "").lower()
        entry = float(trade.get("entry_price", 0) or 0)
        size = abs(float(trade.get("size", 0) or 0))
        if entry <= 0 or size <= 0 or current_price <= 0:
            return 0.0
        delta = current_price - entry
        if side == "short":
            delta = -delta
        return delta * size
    except (TypeError, ValueError):
        return 0.0


def reap_orphan_positions(container) -> List[Dict[str, Any]]:
    """Close orphan positions past max age, subject to break-even gate.

    Returns the list of reaped events (one per closed orphan). When
    disabled, returns ``[]``.
    """
    if not bool(getattr(config, "ORPHAN_REAPER_ENABLED", False)):
        return []

    trader = get_live_trader(container)
    if not trader:
        return []

    max_age_h = float(getattr(config, "ORPHAN_REAPER_MAX_AGE_HOURS", 24.0))
    require_breakeven = bool(
        getattr(config, "ORPHAN_REAPER_REQUIRE_BREAKEVEN", True)
    )

    try:
        open_trades = db.get_open_paper_trades() or []
    except Exception as exc:
        logger.debug("Orphan reaper: get_open_paper_trades failed: %s", exc)
        return []

    orphans = [t for t in open_trades if _is_orphan(t)]
    if not orphans:
        return []

    # Pull mid prices once for break-even checks.
    try:
        from src.data.hyperliquid_client import get_all_mids
        mids = get_all_mids() or {}
    except Exception as exc:
        logger.debug("Orphan reaper: get_all_mids failed: %s", exc)
        mids = {}

    now = datetime.now(timezone.utc)
    reaped: List[Dict[str, Any]] = []

    for orphan in orphans:
        meta = _parse_meta(orphan)
        coin = str(orphan.get("coin") or "").upper()
        side = str(orphan.get("side") or "").lower()
        trade_id = orphan.get("id")

        age_h = _orphan_age_hours(meta, now)
        if age_h is None:
            # Age unknown; treat as new (don't reap on first cycle after
            # introducing the reaper to an existing book).
            logger.debug(
                "Orphan reaper skip %s %s trade_id=%s: missing orphan_found_at",
                coin, side, trade_id,
            )
            continue
        if age_h < max_age_h:
            continue

        # Break-even gate
        pnl_check_failed = False
        current_price = float(mids.get(coin, 0) or 0)
        approx_pnl = _approximate_pnl(orphan, current_price)
        if require_breakeven:
            if current_price <= 0:
                logger.info(
                    "Orphan reaper holding %s %s trade_id=%s "
                    "(age=%.1fh): no mid price for break-even check",
                    coin, side, trade_id, age_h,
                )
                pnl_check_failed = True
            elif approx_pnl < 0:
                logger.info(
                    "Orphan reaper holding %s %s trade_id=%s "
                    "(age=%.1fh, approx_pnl=$%.2f): below break-even",
                    coin, side, trade_id, age_h, approx_pnl,
                )
                pnl_check_failed = True
        if pnl_check_failed:
            continue

        # Close via live trader
        logger.warning(
            "Orphan reaper closing %s %s trade_id=%s "
            "(age=%.1fh, approx_pnl=$%.2f, require_breakeven=%s)",
            coin, side, trade_id, age_h, approx_pnl, require_breakeven,
        )
        try:
            result = trader.close_position(coin)
        except Exception as exc:
            logger.error(
                "Orphan reaper close failed for %s %s: %s",
                coin, side, exc,
            )
            continue

        ok = bool(result) and str(result.get("status", "")).lower() not in {"error", "rejected"}
        try:
            db.audit_log(
                action="orphan_reaped",
                coin=coin,
                side=side,
                source="orphan_reaper",
                details={
                    "trade_id": trade_id,
                    "age_hours": round(age_h, 2),
                    "approx_pnl_usd": round(approx_pnl, 2),
                    "result_status": str(result.get("status", "")) if result else "no_result",
                    "ok": ok,
                },
            )
        except Exception:
            pass

        reaped.append({
            "coin": coin,
            "side": side,
            "trade_id": trade_id,
            "age_hours": age_h,
            "approx_pnl_usd": approx_pnl,
            "result": result,
            "ok": ok,
        })

    return reaped
