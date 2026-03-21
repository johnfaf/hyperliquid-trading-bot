"""
Copy Trading Engine
Monitors top traders' live positions on Hyperliquid and mirrors their new trades
as paper trades. Detects when top traders open/close positions and generates signals.
"""
import logging
import json
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src import database as db
from src import hyperliquid_client as hl

logger = logging.getLogger(__name__)


class CopyTrader:
    """Monitors top traders and mirrors their position changes."""

    def __init__(self):
        # Cache of last-known positions per trader: {address: {coin: position_dict}}
        self._position_cache: Dict[str, Dict[str, Dict]] = {}
        self._copy_count = 0

    def scan_top_traders(self, top_n: int = 10) -> List[Dict]:
        """
        Scan the top N traders by PnL, detect new/changed positions,
        and return copy-trade signals.
        """
        traders = db.get_active_traders()[:top_n]
        if not traders:
            return []

        signals = []
        mids = hl.get_all_mids() or {}

        for trader in traders:
            try:
                addr = trader["address"]
                state = hl.get_user_state(addr)
                if not state:
                    continue

                current_positions = {}
                for pos in state["positions"]:
                    if pos["size"] > 0:
                        current_positions[pos["coin"]] = pos

                # Compare with cached positions to find changes
                cached = self._position_cache.get(addr, {})
                new_signals = self._detect_position_changes(
                    addr, cached, current_positions, trader, mids
                )
                signals.extend(new_signals)

                # Update cache
                self._position_cache[addr] = current_positions

            except Exception as e:
                logger.debug(f"Error scanning trader {trader.get('address', '?')[:10]}: {e}")

        if signals:
            logger.info(f"Copy-trader detected {len(signals)} signals from top {top_n} traders")

        return signals

    def _detect_position_changes(
        self,
        address: str,
        old_positions: Dict[str, Dict],
        new_positions: Dict[str, Dict],
        trader: Dict,
        mids: Dict,
    ) -> List[Dict]:
        """Detect new, closed, or significantly changed positions."""
        signals = []
        old_coins = set(old_positions.keys())
        new_coins = set(new_positions.keys())

        # New positions opened by the trader
        for coin in new_coins - old_coins:
            pos = new_positions[coin]
            price = float(mids.get(coin, pos["entry_price"]))
            if price <= 0:
                continue

            signals.append({
                "type": "copy_open",
                "coin": coin,
                "side": pos["side"],
                "price": price,
                "leverage": min(pos["leverage"], config.PAPER_TRADING_MAX_LEVERAGE),
                "source_trader": address[:10],
                "source_pnl": trader.get("total_pnl", 0),
                "confidence": min(0.9, 0.5 + trader.get("win_rate", 0) * 0.5),
            })

        # Positions closed by the trader (they exited)
        for coin in old_coins - new_coins:
            signals.append({
                "type": "copy_close",
                "coin": coin,
                "source_trader": address[:10],
            })

        # Significantly increased positions (scaling in)
        for coin in old_coins & new_coins:
            old_size = old_positions[coin]["size"]
            new_size = new_positions[coin]["size"]
            if new_size > old_size * 1.5:  # 50%+ increase
                pos = new_positions[coin]
                price = float(mids.get(coin, pos["entry_price"]))
                if price <= 0:
                    continue
                signals.append({
                    "type": "copy_scale_in",
                    "coin": coin,
                    "side": pos["side"],
                    "price": price,
                    "leverage": min(pos["leverage"], config.PAPER_TRADING_MAX_LEVERAGE),
                    "source_trader": address[:10],
                    "source_pnl": trader.get("total_pnl", 0),
                    "confidence": min(0.85, 0.4 + trader.get("win_rate", 0) * 0.5),
                })

        # Side flips (trader reversed position)
        for coin in old_coins & new_coins:
            if old_positions[coin]["side"] != new_positions[coin]["side"]:
                pos = new_positions[coin]
                price = float(mids.get(coin, pos["entry_price"]))
                if price <= 0:
                    continue
                signals.append({
                    "type": "copy_flip",
                    "coin": coin,
                    "side": pos["side"],
                    "price": price,
                    "leverage": min(pos["leverage"], config.PAPER_TRADING_MAX_LEVERAGE),
                    "source_trader": address[:10],
                    "source_pnl": trader.get("total_pnl", 0),
                    "confidence": min(0.95, 0.6 + trader.get("win_rate", 0) * 0.5),
                })

        return signals

    def execute_copy_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Execute copy-trade signals as paper trades.
        Returns list of executed trades.
        """
        if not signals:
            return []

        account = db.get_paper_account()
        if not account:
            return []

        open_trades = db.get_open_paper_trades()
        mids = hl.get_all_mids() or {}
        executed = []

        for signal in signals:
            try:
                if signal["type"] == "copy_close":
                    # Close any matching open trades for this coin
                    closed = self._close_copy_trades(signal, open_trades, mids)
                    executed.extend(closed)
                    continue

                if signal["type"] in ("copy_open", "copy_scale_in", "copy_flip"):
                    trade = self._open_copy_trade(account, signal, open_trades)
                    if trade:
                        executed.append(trade)
                        open_trades.append(trade)

            except Exception as e:
                logger.error(f"Error executing copy signal: {e}")

        if executed:
            self._copy_count += len(executed)
            logger.info(f"Copy-trader executed {len(executed)} trades (total: {self._copy_count})")

        return executed

    def _open_copy_trade(self, account: Dict, signal: Dict, open_trades: List) -> Optional[Dict]:
        """Open a paper trade based on a copy signal."""
        # Position sizing: smaller for copy trades (5% of balance)
        size_usd = account["balance"] * 0.05 * signal.get("confidence", 0.5)
        price = signal["price"]
        if price <= 0:
            return None

        size = size_usd / price
        leverage = signal.get("leverage", 2)
        side = signal["side"]

        # Check basic risk: max 5 copy trades per coin
        coin_copies = sum(1 for t in open_trades if t.get("coin") == signal["coin"])
        if coin_copies >= 5:
            return None

        # SL/TP
        if side == "long":
            stop_loss = price * (1 - 0.04 / leverage)
            take_profit = price * (1 + 0.08 / leverage)
        else:
            stop_loss = price * (1 + 0.04 / leverage)
            take_profit = price * (1 - 0.08 / leverage)

        try:
            trade_id = db.open_paper_trade(
                strategy_id=None,
                coin=signal["coin"],
                side=side,
                entry_price=price,
                size=size,
                leverage=leverage,
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                metadata={
                    "type": signal["type"],
                    "source_trader": signal.get("source_trader", ""),
                    "confidence": signal.get("confidence", 0),
                    "is_copy_trade": True,
                },
            )
            logger.info(
                f"Copy trade: {side.upper()} {signal['coin']} @ ${price:,.2f} "
                f"(from trader {signal.get('source_trader', '?')})"
            )
            return {
                "id": trade_id,
                "coin": signal["coin"],
                "side": side,
                "entry_price": price,
                "size": size,
                "leverage": leverage,
                "strategy_id": None,
            }
        except Exception as e:
            logger.error(f"Error opening copy trade: {e}")
            return None

    def _close_copy_trades(self, signal: Dict, open_trades: List, mids: Dict) -> List[Dict]:
        """Close open copy trades for a coin when the source trader exits."""
        closed = []
        for trade in open_trades:
            meta = trade.get("metadata", "{}")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}

            if (trade["coin"] == signal["coin"] and
                meta.get("is_copy_trade") and
                meta.get("source_trader") == signal.get("source_trader")):

                current_price = float(mids.get(trade["coin"], 0))
                if current_price <= 0:
                    continue

                if trade["side"] == "long":
                    pnl = (current_price - trade["entry_price"]) * trade["size"] * trade["leverage"]
                else:
                    pnl = (trade["entry_price"] - current_price) * trade["size"] * trade["leverage"]
                pnl = round(pnl, 2)

                db.close_paper_trade(trade["id"], current_price, pnl)

                account = db.get_paper_account()
                db.update_paper_account(
                    account["balance"] + pnl,
                    account["total_pnl"] + pnl,
                    account["total_trades"] + 1,
                    account["winning_trades"] + (1 if pnl > 0 else 0),
                )

                logger.info(
                    f"Copy trade closed (source exited): {trade['coin']} "
                    f"PnL=${pnl:,.2f}"
                )
                closed.append({"trade_id": trade["id"], "coin": trade["coin"], "pnl": pnl})

        return closed

    @property
    def total_copy_trades(self) -> int:
        return self._copy_count
