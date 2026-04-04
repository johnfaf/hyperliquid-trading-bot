"""
Helpers for live-vs-paper execution state.

These utilities keep the trading cycles anchored to exchange truth whenever
live trading is actually enabled and deployable.
"""
from __future__ import annotations

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

        closed_trade = container.paper_trader._close_trade(
            trade,
            current_price=current_price,
            close_reason="live_reconciled_closed",
        )
        if closed_trade:
            closed.append(closed_trade)
            logger.info(
                "Shadow paper trade reconciled to exchange truth: %s %s",
                trade.get("side", "?").upper(),
                trade.get("coin", "?"),
            )

    return closed


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
                live_signal = signal_from_execution_dict(trade) if isinstance(trade, dict) else trade
                live_result = trader.execute_signal(live_signal)
                if live_result:
                    logger.info(
                        "%s: %s %s %s",
                        success_label,
                        live_result.get("status", "?"),
                        live_signal.coin,
                        live_signal.side.value,
                    )
            except Exception as exc:
                logger.warning("%s live execution error: %s", success_label, exc)
    elif trader.is_live_enabled():
        logger.warning("%s", skip_label)
