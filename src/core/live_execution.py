"""
Helpers for live-vs-paper execution state.

These utilities keep the trading cycles anchored to exchange truth whenever
live trading is actually enabled and deployable.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from src.data import database as db
from src.data.hyperliquid_client import get_all_mids
from src.signals.signal_schema import signal_from_execution_dict

logger = logging.getLogger(__name__)


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
        return trader.get_positions()
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
    """
    if not is_live_trading_active(container) or not getattr(container, "paper_trader", None):
        return []

    live_positions = {
        pos.get("coin", ""): pos
        for pos in (get_live_trader(container).get_positions() or [])
        if pos.get("coin") and abs(float(pos.get("szi", pos.get("size", 0)) or 0)) > 0
    }
    open_trades = db.get_open_paper_trades()
    if not open_trades:
        return []

    mids = get_all_mids() or {}
    closed = []
    for trade in open_trades:
        live_pos = live_positions.get(trade.get("coin", ""))
        if live_pos and live_pos.get("side") == trade.get("side"):
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

        try:
            existing_meta = trade.get("metadata", {})
            if isinstance(existing_meta, str):
                existing_meta = json.loads(existing_meta or "{}")
            existing_meta = dict(existing_meta or {})
        except Exception:
            existing_meta = {}

        existing_meta.update({
            "synthetic_reconciliation": True,
            "reconciliation_reason": "live_reconciled_closed",
            "reconciliation_exit_price": current_price,
        })
        db.update_paper_trade_metadata(trade_id, existing_meta)
        if not db.close_paper_trade(trade_id, current_price, 0.0):
            continue

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

    return closed


def _rescale_size_for_live(trade: Dict, trader) -> Optional[Dict]:
    """
    Rescale paper trade size proportionally to the live account balance.

    Paper sizes are computed from the paper account (default $10k).  When
    mirroring to live, the coin quantity must be adjusted so the same
    *percentage* of the live account is risked, not the same absolute size.
    """
    paper_account = db.get_paper_account()
    paper_balance = float((paper_account or {}).get("balance", 0) or 0)
    live_balance = trader.get_account_value()
    if not paper_balance or paper_balance <= 0:
        logger.error(
            "Cannot rescale %s: paper account balance unavailable (%s). "
            "Blocking trade to prevent wrong sizing.",
            trade.get("coin", "?"),
            paper_balance,
        )
        return None
    if live_balance is None:
        logger.error(
            "Cannot rescale %s: live account balance API call failed. "
            "Blocking trade to prevent wrong sizing.",
            trade.get("coin", "?"),
        )
        return None
    if live_balance <= 0:
        logger.warning(
            "Skipping live mirror for %s: live account balance is $%.2f. "
            "Deposit USDC to your Hyperliquid account to enable live trading.",
            trade.get("coin", "?"), live_balance,
        )
        return None

    scale = live_balance / paper_balance
    if abs(scale - 1.0) < 0.01:
        return trade  # balances are similar, no rescaling needed

    original_size = float(trade.get("size", 0) or 0)
    if original_size <= 0:
        return trade

    trade = dict(trade)  # shallow copy to avoid mutating caller's dict
    trade["size"] = original_size * scale
    logger.info(
        "Rescaled %s size for live: %.6f → %.6f (paper=$%.0f, live=$%.0f, scale=%.2f)",
        trade.get("coin", "?"),
        original_size,
        trade["size"],
        paper_balance,
        live_balance,
        scale,
    )
    return trade


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
        for trade in executed:
            try:
                # Rescale size from paper balance to live balance
                scaled_trade = _rescale_size_for_live(trade, trader) if isinstance(trade, dict) else trade
                if scaled_trade is None:
                    continue  # blocked by rescale — already logged
                live_signal = signal_from_execution_dict(scaled_trade) if isinstance(scaled_trade, dict) else scaled_trade
                live_result = trader.execute_signal(live_signal)
                if live_result and live_result.get("status") not in ("error", "rejected"):
                    logger.info(
                        "%s: %s %s %s",
                        success_label,
                        live_result.get("status", "?"),
                        live_signal.coin,
                        live_signal.side.value,
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
