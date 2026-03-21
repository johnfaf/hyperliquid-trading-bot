"""
Paper Trading Simulator
Simulates trades based on the top-scoring strategies without risking real funds.
Tracks performance to validate strategies before any real deployment.
"""
import logging
from datetime import datetime
from typing import List, Dict, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src import database as db
from src import hyperliquid_client as hl

logger = logging.getLogger(__name__)


class PaperTrader:
    """Simulates trading based on identified strategies."""

    def __init__(self):
        self._ensure_account()

    def _ensure_account(self):
        """Initialize paper trading account if it doesn't exist."""
        account = db.get_paper_account()
        if not account:
            db.init_paper_account(config.PAPER_TRADING_INITIAL_BALANCE)
            logger.info(f"Paper account initialized with ${config.PAPER_TRADING_INITIAL_BALANCE:,.2f}")

    def get_account_summary(self) -> Dict:
        """Get current paper trading account summary."""
        account = db.get_paper_account()
        open_trades = db.get_open_paper_trades()
        closed_trades = db.get_paper_trade_history(limit=100)

        # Calculate unrealized PnL from open trades
        unrealized_pnl = 0
        mids = hl.get_all_mids() or {}
        for trade in open_trades:
            current_price = float(mids.get(trade["coin"], trade["entry_price"]))
            if trade["side"] == "long":
                trade_pnl = (current_price - trade["entry_price"]) * trade["size"] * trade["leverage"]
            else:
                trade_pnl = (trade["entry_price"] - current_price) * trade["size"] * trade["leverage"]
            unrealized_pnl += trade_pnl

        total_equity = account["balance"] + unrealized_pnl if account else 0

        return {
            "balance": account["balance"] if account else 0,
            "total_pnl": account["total_pnl"] if account else 0,
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_equity": round(total_equity, 2),
            "total_trades": account["total_trades"] if account else 0,
            "winning_trades": account["winning_trades"] if account else 0,
            "win_rate": (account["winning_trades"] / account["total_trades"]
                        if account and account["total_trades"] > 0 else 0),
            "open_positions": len(open_trades),
            "roi_pct": round(((total_equity / config.PAPER_TRADING_INITIAL_BALANCE) - 1) * 100, 2),
        }

    def execute_strategy_signals(self, strategies: List[Dict]) -> List[Dict]:
        """
        Generate and execute paper trades based on top strategies.
        Only trades strategies with score above threshold.
        """
        account = db.get_paper_account()
        if not account:
            return []

        open_trades = db.get_open_paper_trades()
        executed = []

        # Get current prices
        mids = hl.get_all_mids() or {}

        for strategy in strategies:
            try:
                # Check if we already have a position from this strategy
                existing = [t for t in open_trades if t["strategy_id"] == strategy.get("id")]
                if existing:
                    continue

                # Generate signal from strategy
                signal = self._generate_signal(strategy, mids)
                if not signal:
                    continue

                # Risk management checks
                if not self._check_risk_limits(account, signal, open_trades):
                    logger.info(f"Risk limit hit, skipping signal for {signal['coin']}")
                    continue

                # Execute the paper trade
                trade = self._execute_paper_trade(account, strategy, signal)
                if trade:
                    executed.append(trade)
                    open_trades.append(trade)  # Update local list

            except Exception as e:
                logger.error(f"Error executing strategy {strategy.get('name', '?')}: {e}")

        if executed:
            logger.info(f"Executed {len(executed)} paper trades")

        return executed

    def _generate_signal(self, strategy: Dict, mids: Dict) -> Optional[Dict]:
        """
        Generate a trading signal from a strategy.
        Maps strategy types to concrete trade parameters.
        """
        import json
        params = strategy.get("parameters", "{}")
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except (json.JSONDecodeError, TypeError):
                params = {}

        strategy_type = strategy.get("strategy_type", strategy.get("type", ""))
        score = strategy.get("current_score", 0)

        # Only trade strategies with decent scores
        if score < 0.3:
            return None

        # Get target coins from strategy parameters
        coins = params.get("coins", [])
        if not coins:
            coins = params.get("coins_traded", ["BTC", "ETH"])
        if isinstance(coins, str):
            coins = [coins]

        # Pick the first available coin with a price
        target_coin = None
        target_price = 0
        for coin in coins:
            if coin in mids:
                target_coin = coin
                target_price = float(mids[coin])
                break

        if not target_coin or target_price == 0:
            # Fallback to BTC
            target_coin = "BTC"
            target_price = float(mids.get("BTC", 0))
            if target_price == 0:
                return None

        # Determine direction based on strategy type
        if strategy_type in ("momentum_long", "trend_following", "breakout"):
            side = "long"
        elif strategy_type in ("momentum_short", "contrarian"):
            side = "short"
        elif strategy_type == "mean_reversion":
            # Check if current price is above or below recent average
            side = params.get("direction", "long")
        elif strategy_type in ("scalping", "swing_trading"):
            # Use the bias from parameters or default to long
            direction = params.get("direction", params.get("bias", "long"))
            side = "long" if "long" in str(direction) else "short"
        elif strategy_type == "funding_arb":
            side = "short"  # Typically short to earn positive funding
        else:
            side = "long"  # Default

        # Determine leverage (capped by config)
        leverage = min(
            params.get("avg_leverage", 2),
            config.PAPER_TRADING_MAX_LEVERAGE
        )
        leverage = max(1, leverage)

        # Calculate position size
        account = db.get_paper_account()
        max_position = account["balance"] * config.PAPER_TRADING_MAX_POSITION_PCT
        size_usd = max_position * (score / 1.0)  # Scale by confidence
        size = size_usd / target_price

        # Stop loss and take profit
        if side == "long":
            stop_loss = target_price * (1 - config.PAPER_TRADING_STOP_LOSS_PCT / leverage)
            take_profit = target_price * (1 + config.PAPER_TRADING_TAKE_PROFIT_PCT / leverage)
        else:
            stop_loss = target_price * (1 + config.PAPER_TRADING_STOP_LOSS_PCT / leverage)
            take_profit = target_price * (1 - config.PAPER_TRADING_TAKE_PROFIT_PCT / leverage)

        return {
            "coin": target_coin,
            "side": side,
            "price": target_price,
            "size": size,
            "leverage": leverage,
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "strategy_type": strategy_type,
            "confidence": score,
        }

    def _check_risk_limits(self, account: Dict, signal: Dict, open_trades: List) -> bool:
        """Check if a new trade passes risk management rules."""
        # Max position count
        if len(open_trades) >= 10:
            return False

        # Max exposure per coin
        coin_exposure = sum(
            t["size"] * t["entry_price"] * t["leverage"]
            for t in open_trades if t["coin"] == signal["coin"]
        )
        max_coin_exposure = account["balance"] * 0.3  # 30% max per coin
        new_exposure = signal["size"] * signal["price"] * signal["leverage"]
        if coin_exposure + new_exposure > max_coin_exposure:
            return False

        # Max total exposure
        total_exposure = sum(
            t["size"] * t["entry_price"] * t["leverage"]
            for t in open_trades
        )
        max_total_exposure = account["balance"] * 2.0  # 200% max total
        if total_exposure + new_exposure > max_total_exposure:
            return False

        # Minimum balance remaining
        if account["balance"] < config.PAPER_TRADING_INITIAL_BALANCE * 0.1:
            return False

        return True

    def _execute_paper_trade(self, account: Dict, strategy: Dict, signal: Dict) -> Optional[Dict]:
        """Execute a paper trade and record it."""
        try:
            trade_id = db.open_paper_trade(
                strategy_id=strategy.get("id"),
                coin=signal["coin"],
                side=signal["side"],
                entry_price=signal["price"],
                size=signal["size"],
                leverage=signal["leverage"],
                stop_loss=signal["stop_loss"],
                take_profit=signal["take_profit"],
                metadata={
                    "strategy_type": signal["strategy_type"],
                    "confidence": signal["confidence"],
                },
            )

            logger.info(
                f"Paper trade opened: {signal['side'].upper()} {signal['coin']} "
                f"@ ${signal['price']:,.2f} | size={signal['size']:.4f} | "
                f"leverage={signal['leverage']}x | SL=${signal['stop_loss']:,.2f} | "
                f"TP=${signal['take_profit']:,.2f}"
            )

            return {
                "id": trade_id,
                "coin": signal["coin"],
                "side": signal["side"],
                "entry_price": signal["price"],
                "size": signal["size"],
                "leverage": signal["leverage"],
                "strategy_id": strategy.get("id"),
            }
        except Exception as e:
            logger.error(f"Error opening paper trade: {e}")
            return None

    def check_open_positions(self) -> List[Dict]:
        """
        Check all open paper positions against current prices.
        Close any that hit stop-loss or take-profit.
        """
        open_trades = db.get_open_paper_trades()
        if not open_trades:
            return []

        mids = hl.get_all_mids() or {}
        closed = []

        for trade in open_trades:
            current_price = float(mids.get(trade["coin"], 0))
            if current_price == 0:
                continue

            should_close = False
            close_reason = ""

            # Check stop loss
            if trade["stop_loss"]:
                if trade["side"] == "long" and current_price <= trade["stop_loss"]:
                    should_close = True
                    close_reason = "stop_loss"
                elif trade["side"] == "short" and current_price >= trade["stop_loss"]:
                    should_close = True
                    close_reason = "stop_loss"

            # Check take profit
            if trade["take_profit"] and not should_close:
                if trade["side"] == "long" and current_price >= trade["take_profit"]:
                    should_close = True
                    close_reason = "take_profit"
                elif trade["side"] == "short" and current_price <= trade["take_profit"]:
                    should_close = True
                    close_reason = "take_profit"

            if should_close:
                pnl = self._calculate_pnl(trade, current_price)
                db.close_paper_trade(trade["id"], current_price, pnl)

                # Update account
                account = db.get_paper_account()
                new_balance = account["balance"] + pnl
                new_total_pnl = account["total_pnl"] + pnl
                new_total_trades = account["total_trades"] + 1
                new_winning = account["winning_trades"] + (1 if pnl > 0 else 0)
                db.update_paper_account(new_balance, new_total_pnl, new_total_trades, new_winning)

                logger.info(
                    f"Paper trade closed ({close_reason}): {trade['side'].upper()} {trade['coin']} "
                    f"entry=${trade['entry_price']:,.2f} exit=${current_price:,.2f} "
                    f"PnL=${pnl:,.2f}"
                )

                closed.append({
                    "trade_id": trade["id"],
                    "coin": trade["coin"],
                    "side": trade["side"],
                    "pnl": pnl,
                    "reason": close_reason,
                })

        return closed

    def _calculate_pnl(self, trade: Dict, exit_price: float) -> float:
        """Calculate PnL for a trade."""
        if trade["side"] == "long":
            pnl = (exit_price - trade["entry_price"]) * trade["size"] * trade["leverage"]
        else:
            pnl = (trade["entry_price"] - exit_price) * trade["size"] * trade["leverage"]
        return round(pnl, 2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    db.init_db()
    trader = PaperTrader()
    summary = trader.get_account_summary()
    print(f"\nPaper Account Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
