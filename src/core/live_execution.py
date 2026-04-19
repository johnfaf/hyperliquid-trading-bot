"""
Helpers for live-vs-paper execution state.

These utilities keep the trading cycles anchored to exchange truth whenever
live trading is actually enabled and deployable.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.data import database as db
from src.data.hyperliquid_client import get_all_mids
from src.signals.signal_schema import signal_from_execution_dict

logger = logging.getLogger(__name__)


def _trade_metadata(trade: Dict) -> Dict:
    try:
        existing_meta = trade.get("metadata", {})
        if isinstance(existing_meta, str):
            existing_meta = json.loads(existing_meta or "{}")
        return dict(existing_meta or {})
    except Exception:
        return {}


def _notify_manual_close_detected(trade: Dict, exit_price: float) -> None:
    try:
        from src.notifications import telegram_bot as tg
    except Exception:
        return

    try:
        tg.notify_manual_close_detected(trade, exit_price=exit_price)
    except Exception as exc:
        logger.debug("Manual close notification failed for %s: %s", trade.get("coin", "?"), exc)


def _is_insufficient_margin_rejection(result) -> bool:
    """Return True when a rejection payload indicates insufficient margin."""
    if not isinstance(result, dict):
        return False

    reason = str(result.get("reason", "") or "").strip().lower()
    if reason == "insufficient_margin":
        return True

    messages: List[str] = []
    errors = result.get("errors")
    if isinstance(errors, list):
        messages.extend(str(item) for item in errors)
    elif errors:
        messages.append(str(errors))

    message = result.get("message")
    if message:
        messages.append(str(message))

    return any("insufficient margin" in msg.lower() for msg in messages)


def get_live_trader(container):
    """Return the attached live trader, if any."""
    return getattr(container, "live_trader", None)


def is_live_trading_requested(container) -> bool:
    """True when the operator explicitly enabled live trading."""
    trader = get_live_trader(container)
    return bool(trader and trader.is_live_enabled())


def is_live_trading_active(container) -> bool:
    """True when the operator enabled live trading and the trader is deployable."""
    trader = get_live_trader(container)
    return bool(trader and trader.is_live_enabled() and trader.is_deployable())


def get_execution_open_positions(container) -> List[Dict]:
    """Use exchange positions as the source of truth when live trading is active."""
    trader = get_live_trader(container)
    if trader and is_live_trading_active(container):
        return trader.get_positions() or []
    return db.get_open_paper_trades()


def get_execution_account_balance(container) -> Optional[float]:
    """Use live account value when available, otherwise fall back to paper balance."""
    trader = get_live_trader(container)
    if trader and is_live_trading_active(container):
        value = trader.get_account_value()
        if value is not None:
            return float(value)

    account = db.get_paper_account()
    if not account:
        return None
    try:
        return float(account.get("balance", 0))
    except (TypeError, ValueError):
        return None


def sync_shadow_book_to_live(container) -> List[Dict]:
    """
    Close shadow paper trades that no longer exist on the exchange.

    In live mode, exchange state is the authority. This keeps the paper book as
    a reporting shadow instead of letting it drive runtime decisions.

    SAFETY: If the live account has $0 perps margin, skip reconciliation
    entirely.  Otherwise we'd close every paper trade because the exchange
    shows 0 positions (the money is in spot or hasn't been deposited).
    """
    if not is_live_trading_active(container) or not getattr(container, "paper_trader", None):
        return []

    # Guard: don't reconcile paper trades against an unfunded exchange account.
    # With $0 perps margin the exchange will always show 0 positions, which
    # would cause this function to close EVERY paper trade immediately.
    trader = get_live_trader(container)
    account_value = trader.get_account_value() if trader else None
    if account_value is None or account_value <= 0:
        return []

    fetched_positions = trader.get_positions() if trader else None
    if fetched_positions is None:
        logger.warning("Skipping shadow/live reconciliation: exchange positions unavailable")
        return []

    live_positions = {
        pos.get("coin", ""): pos
        for pos in fetched_positions
        if pos.get("coin") and abs(float(pos.get("szi", pos.get("size", 0)) or 0)) > 0
    }
    open_trades = db.get_open_paper_trades()
    if not open_trades:
        open_trades = []

    mids = get_all_mids() or {}
    closed = []
    matched_live_keys = set()
    for trade in open_trades:
        live_pos = live_positions.get(trade.get("coin", ""))
        if live_pos and live_pos.get("side") == trade.get("side"):
            matched_live_keys.add(
                (
                    str(trade.get("coin", "") or "").upper(),
                    str(trade.get("side", "") or "").lower(),
                )
            )
            continue

        current_price = float(
            mids.get(trade.get("coin", ""), trade.get("entry_price", 0))
            or trade.get("entry_price", 0)
            or 0
        )
        if current_price <= 0:
            continue

        trade_id = trade.get("id")
        if trade_id is None:
            continue

        existing_meta = _trade_metadata(trade)

        existing_meta.update({
            "synthetic_reconciliation": True,
            "reconciliation_reason": "live_reconciled_closed",
            "reconciliation_exit_price": current_price,
        })
        db.update_paper_trade_metadata(trade_id, existing_meta)
        # BUG-4 FIX: calculate actual PnL instead of hardcoding 0.0.
        # Without this, reconciled trades permanently lose their PnL
        # in the DB, making forensic analysis impossible.
        entry_price = float(trade.get("entry_price", 0) or 0)
        trade_size = float(trade.get("size", 0) or 0)
        trade_leverage = float(trade.get("leverage", 1) or 1)
        if trade.get("side") == "long":
            reconciled_pnl = (current_price - entry_price) * trade_size * trade_leverage
        else:
            reconciled_pnl = (entry_price - current_price) * trade_size * trade_leverage
        reconciled_pnl = round(reconciled_pnl, 2)
        if not db.close_paper_trade(trade_id, current_price, reconciled_pnl):
            continue
        _notify_manual_close_detected(
            {
                **trade,
                "metadata": existing_meta,
            },
            exit_price=current_price,
        )

        closed_trade = {
            "trade_id": trade_id,
            "entry_price": trade.get("entry_price", 0),
            "size": trade.get("size", 0),
            "leverage": trade.get("leverage", 1),
            "coin": trade.get("coin", ""),
            "side": trade.get("side", ""),
            "pnl": 0.0,
            "gross_pnl": 0.0,
            "fees_paid": 0.0,
            "slippage_cost": 0.0,
            "reason": "live_reconciled_closed",
            "strategy_type": existing_meta.get("strategy_type", "unknown"),
            "signal_id": existing_meta.get("signal_id", ""),
            "exit_price": current_price,
            "metadata": existing_meta,
            "opened_at": trade.get("opened_at", ""),
            "closed_at": trade.get("closed_at", ""),
        }
        closed.append(closed_trade)
        logger.info(
            "Shadow paper trade reconciled to exchange truth: %s %s",
            trade.get("side", "?").upper(),
            trade.get("coin", "?"),
        )

    for live_pos in live_positions.values():
        coin = str(live_pos.get("coin", "") or "").upper()
        side = str(live_pos.get("side", "") or "").lower()
        if not coin or side not in {"long", "short"}:
            continue
        if (coin, side) in matched_live_keys:
            continue

        entry_price = float(
            live_pos.get("entry_price", live_pos.get("entryPx", 0))
            or 0
        )
        size = abs(float(live_pos.get("size", live_pos.get("szi", 0)) or 0))
        leverage = float(live_pos.get("leverage", 1) or 1)
        if entry_price <= 0 or size <= 0:
            continue

        metadata = {
            "synthetic_reconciliation": True,
            "reconciliation_reason": "orphan_found",
            "orphan_found": True,
            "orphan_found_at": datetime.now(timezone.utc).isoformat(),
            "source": "live_orphan",
            "source_key": "live_orphan",
            "strategy_type": "orphan_found",
            "live_snapshot": {
                "coin": coin,
                "side": side,
                "entry_price": entry_price,
                "size": size,
                "leverage": leverage,
            },
        }
        trade_id = db.open_paper_trade(
            None,
            coin,
            side,
            entry_price,
            size,
            leverage=leverage,
            metadata=metadata,
        )
        db.audit_log(
            action="orphan_found",
            coin=coin,
            side=side,
            price=entry_price,
            size=size,
            source="live_execution",
            details={"trade_id": trade_id, "metadata": metadata},
        )
        logger.warning(
            "Created synthetic paper trade for orphan live position: %s %s size=%.6f entry=%.6f",
            side.upper(),
            coin,
            size,
            entry_price,
        )

    return closed


def _rescale_size_for_live(trade: Dict, trader) -> Optional[Dict]:
    """
    Rescale paper trade size proportionally to the live account balance.

    Paper sizes are computed from the paper account (default $10k).  When
    mirroring to live, the coin quantity must be adjusted so the same
    *percentage* of the live account is risked, not the same absolute size.

    After rescaling, the final notional is clamped to trader.max_order_usd
    (if set) so nothing above the bootstrap cap ever hits the exchange.
    """
    paper_account = db.get_paper_account()
    paper_balance = float((paper_account or {}).get("balance", 0) or 0)
    live_equity = trader.get_account_value()
    live_free_margin = None
    if hasattr(trader, "get_free_margin"):
        try:
            live_free_margin = trader.get_free_margin()
        except Exception as exc:
            logger.warning(
                "Cannot rescale %s: live free margin API call failed (%s). "
                "Blocking trade to prevent oversizing.",
                trade.get("coin", "?"),
                exc,
            )
            return None
    if not paper_balance or paper_balance <= 0:
        logger.error(
            "Cannot rescale %s: paper account balance unavailable (%s). "
            "Blocking trade to prevent wrong sizing.",
            trade.get("coin", "?"),
            paper_balance,
        )
        return None
    if live_equity is None:
        logger.error(
            "Cannot rescale %s: live account balance API call failed. "
            "Blocking trade to prevent wrong sizing.",
            trade.get("coin", "?"),
        )
        return None
    if live_free_margin is None:
        live_free_margin = live_equity
    live_free_margin = float(live_free_margin or 0.0)
    if live_free_margin <= 0:
        logger.warning(
            "Skipping live mirror for %s: free perps margin is $%.2f "
            "(equity=$%.2f). Transfer USDC from Spot to Perps or free margin "
            "before opening new positions.",
            trade.get("coin", "?"), live_free_margin, float(live_equity or 0.0),
        )
        return None

    scale = live_free_margin / paper_balance
    original_size = float(trade.get("size", 0) or 0)
    if original_size <= 0:
        return trade

    scaled_trade = dict(trade)  # shallow copy to avoid mutating caller's dict
    if abs(scale - 1.0) >= 0.01:
        scaled_trade["size"] = original_size * scale
        logger.info(
            "Rescaled %s size for live: %.6f → %.6f "
            "(paper=$%.0f, free_margin=$%.2f, equity=$%.2f, scale=%.4f)",
            trade.get("coin", "?"),
            original_size,
            scaled_trade["size"],
            paper_balance,
            live_free_margin,
            float(live_equity or 0.0),
            scale,
        )

    # Enforce per-order $ cap (bootstrap safety net).  Applied on top of the
    # proportional rescale so nothing above LIVE_MAX_ORDER_USD hits the exchange.
    max_order_usd = getattr(trader, "max_order_usd", None)
    min_order_usd = getattr(trader, "min_order_usd", None) or 0.0

    # IMPORTANT: prefer the *live mid price* over the signal's entry_price
    # for cap/floor calculations.  place_market_order/place_limit_order use
    # mid price when they execute, so we must size using the same reference
    # — otherwise a 2-5% price drift between signal generation and order
    # placement flips our "$11.55 target notional" into "$10.97 actual
    # notional" and the exchange rejects with below_exchange_minimum_notional.
    coin = scaled_trade.get("coin", "") or ""
    mids = get_all_mids() or {}
    mid_price = 0.0
    try:
        mid_price = float(mids.get(coin, 0) or 0)
    except (TypeError, ValueError):
        mid_price = 0.0
    entry_price = mid_price
    if entry_price <= 0:
        entry_price = float(
            scaled_trade.get("entry_price")
            or trade.get("entry_price")
            or 0
        )

    if max_order_usd and max_order_usd > 0 and entry_price > 0:
        current_size = float(scaled_trade.get("size", 0) or 0)
        notional = current_size * entry_price
        if notional > max_order_usd:
            capped_size = max_order_usd / entry_price
            logger.info(
                "Capping %s live mirror to $%.2f: %.6f → %.6f "
                "(notional $%.2f → $%.2f)",
                coin or "?",
                max_order_usd,
                current_size,
                capped_size,
                notional,
                capped_size * entry_price,
            )
            scaled_trade["size"] = capped_size

    # Floor-up to the exchange minimum when proportional rescaling would
    # otherwise produce an un-fillable order.  A small live wallet relative
    # to the paper book (e.g. $12 live vs $10k paper = 0.0012 scale factor)
    # makes every proportional rescale tiny, but Hyperliquid drops anything
    # under $10.  Rather than skipping every mirror, we floor up to the
    # exchange minimum — the LiveTrader's max_order_usd cap still bounds the
    # absolute max, so at most we send an $11 notional order.
    #
    # Safety: check that the *margin* required (notional / leverage) fits
    # in the live wallet with headroom for other positions.  Checking
    # notional would be wrong — a $11 notional at 5x leverage only ties up
    # $2.20 of margin, so a $12 wallet can comfortably hold 4+ of them.
    if min_order_usd > 0 and entry_price > 0:
        current_size = float(scaled_trade.get("size", 0) or 0)
        notional = current_size * entry_price
        if notional < min_order_usd:
            # Margin required = notional / leverage.  Fall back to 1x (the
            # most conservative assumption) if leverage is not specified.
            leverage = float(
                scaled_trade.get("leverage")
                or trade.get("leverage")
                or 1
            )
            if leverage <= 0:
                leverage = 1.0

            # Headroom budget: leave 5% of the wallet untouched for fees,
            # slippage, and funding.  Leverage multiplies the notional
            # each dollar of margin can support, so a $12 wallet at 5x
            # can carry up to $57 notional, while at 1x it caps out at
            # $11.40 notional (barely clearing the $11 minimum).  This
            # is the cap the check uses — NOT 80% of wallet, which
            # incorrectly blocked 1x trades on small wallets.
            wallet_notional_budget = max(0.0, live_free_margin) * 0.95 * leverage

            # Target 1.10x the minimum so normal price drift, slippage,
            # and size rounding (szDecimals) don't slip us back under
            # the floor.  Cap at max_order_usd AND at the wallet budget.
            headroom_target = min_order_usd * 1.10
            target_notional = headroom_target
            if max_order_usd and max_order_usd > 0:
                target_notional = min(target_notional, max_order_usd)
            target_notional = min(target_notional, wallet_notional_budget)

            # Only reject if even the bare minimum cannot fit in the
            # wallet at this leverage.  This is the true physical limit:
            # no amount of floor-up can make a $11 notional fit in a $10
            # wallet at 1x leverage.
            if target_notional < min_order_usd:
                logger.warning(
                    "Skipping %s live mirror: wallet $%.2f at %.1fx "
                    "leverage supports max $%.2f notional (95%% headroom), "
                    "which is below the $%.2f exchange minimum.  Fund "
                    "the wallet or raise leverage for this asset.",
                    coin or "?",
                    live_free_margin,
                    leverage,
                    wallet_notional_budget,
                    min_order_usd,
                )
                return None

            required_margin = target_notional / leverage
            floored_size = target_notional / entry_price
            logger.info(
                "Flooring %s live mirror UP to exchange minimum: "
                "%.6f → %.6f (notional $%.2f → $%.2f, margin @ %.1fx = "
                "$%.2f, ref_price=$%.4f %s).  Proportional rescale from "
                "paper was below $%.2f; this departs from strict "
                "paper-proportional sizing, unavoidable given live "
                "wallet $%.2f vs paper $%.0f.",
                coin or "?",
                current_size,
                floored_size,
                notional,
                floored_size * entry_price,
                leverage,
                required_margin,
                entry_price,
                "mid" if mid_price > 0 else "signal",
                min_order_usd,
                live_free_margin,
                paper_balance,
            )
            scaled_trade["size"] = floored_size

    return scaled_trade


def mirror_executed_trades_to_live(
    container,
    executed: List[Dict],
    success_label: str,
    skip_label: str,
) -> None:
    """Submit executed shadow trades to the live trader when live mode is active."""
    trader = get_live_trader(container)
    if not trader or not executed:
        return

    if is_live_trading_active(container):
        candidates = []
        for trade in executed:
            try:
                # Rescale size from paper balance to live balance
                scaled_trade = _rescale_size_for_live(trade, trader) if isinstance(trade, dict) else trade
                if scaled_trade is None:
                    continue  # blocked by rescale — already logged
                live_signal = signal_from_execution_dict(scaled_trade) if isinstance(scaled_trade, dict) else scaled_trade
                entry_price = float(
                    getattr(live_signal, "entry_price", 0)
                    or (scaled_trade.get("entry_price", scaled_trade.get("price", 0)) if isinstance(scaled_trade, dict) else 0)
                    or 0
                )
                size = abs(float(getattr(live_signal, "size", 0) or 0))
                leverage = max(1.0, float(getattr(live_signal, "leverage", 1.0) or 1.0))
                notional = max(0.0, size * entry_price)
                margin = notional / leverage if leverage > 0 else notional
                candidates.append({
                    "signal": live_signal,
                    "notional": notional,
                    "margin": margin,
                })
            except Exception as exc:
                logger.error("%s live execution prep error: %s", success_label, exc)

        if not candidates:
            return

        # Prefer free/available margin (accountValue - totalMarginUsed) so the
        # batch budget doesn't double-count margin already locked by open
        # positions.  Falls back to total account value only if the trader
        # doesn't expose a free-margin helper.
        free_margin: Optional[float] = None
        if hasattr(trader, "get_free_margin"):
            try:
                fm = trader.get_free_margin()
                free_margin = float(fm) if fm is not None else None
            except Exception as exc:
                logger.debug("%s get_free_margin failed: %s", success_label, exc)
                free_margin = None
        if free_margin is None:
            try:
                free_margin = float(trader.get_account_value() or 0.0)
            except Exception:
                free_margin = 0.0

        margin_budget = max(0.0, free_margin) * 0.95
        selected = []
        used_margin = 0.0

        # Zero/negative budget means "no room to mirror anything" — reject
        # all candidates rather than (accidentally) admitting them all.
        if margin_budget <= 0.0:
            logger.warning(
                "%s skipped %d candidate(s): no free margin available "
                "(free=$%.2f, budget=$%.2f)",
                success_label,
                len(candidates),
                free_margin or 0.0,
                margin_budget,
            )
            return

        # Keep the paper execution order to maximize live-vs-paper parity when
        # margin/canary caps force us to drop some mirrors.
        for item in candidates:
            projected = used_margin + item["margin"]
            if projected > margin_budget:
                logger.warning(
                    "%s skipped %s %s: batch margin budget exceeded "
                    "(need $%.2f, used $%.2f, budget $%.2f)",
                    success_label,
                    item["signal"].coin,
                    item["signal"].side.value,
                    item["margin"],
                    used_margin,
                    margin_budget,
                )
                continue
            selected.append(item)
            used_margin = projected

        for item in selected:
            live_signal = item["signal"]
            try:
                # Mirror path: the paper trade has already passed the firewall
                # (cooldown, risk checks, etc.), so bypass firewall validation
                # here.  Otherwise the firewall's cooldown check rejects every
                # mirrored trade as "COIN traded Ns ago" — the paper trade that
                # triggered the mirror.  Kill-switch and daily loss still apply.
                live_result = trader.execute_signal(live_signal, bypass_firewall=True)
                if live_result and live_result.get("status") not in ("error", "rejected"):
                    logger.info(
                        "%s: %s %s %s",
                        success_label,
                        live_result.get("status", "?"),
                        live_signal.coin,
                        live_signal.side.value,
                    )
                else:
                    if live_result is None:
                        logger.warning(
                            "%s skipped: %s %s blocked by live guardrails (no execution result)",
                            success_label,
                            live_signal.coin,
                            live_signal.side.value,
                        )
                    elif _is_insufficient_margin_rejection(live_result):
                        logger.warning(
                            "%s skipped due to insufficient margin: %s %s -> %s",
                            success_label,
                            live_signal.coin,
                            live_signal.side.value,
                            live_result,
                        )
                    else:
                        logger.error(
                            "%s FAILED: %s %s %s — result: %s",
                            success_label,
                            live_signal.coin,
                            live_signal.side.value,
                            live_signal.confidence,
                            live_result,
                        )
            except Exception as exc:
                logger.error("%s live execution error: %s", success_label, exc)
    elif trader.is_live_enabled():
        logger.warning("%s", skip_label)
