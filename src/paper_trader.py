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

    def execute_strategy_signals(self, strategies: List[Dict], exchange_agg=None) -> List[Dict]:
        """
        Generate and execute paper trades based on top strategies.
        Only trades strategies with score above threshold.
        If exchange_agg is provided, checks multi-exchange volume confirmation.
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

                # Multi-exchange volume confirmation (if available)
                if exchange_agg:
                    try:
                        confirmed, vol_confidence = exchange_agg.get_volume_confirmation(
                            signal["coin"], signal["side"]
                        )
                        if not confirmed:
                            logger.info(f"Volume rejects {signal['side']} {signal['coin']} "
                                       f"(confidence={vol_confidence:.2f})")
                            continue
                        # Boost or reduce confidence based on volume
                        signal["confidence"] = signal.get("confidence", 0.5) * (0.5 + vol_confidence * 0.5)
                    except Exception:
                        pass  # Don't block trades if aggregator fails

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
            coins = params.get("coins_traded", [])
        if isinstance(coins, str):
            coins = [coins]

        # If strategy has no specific coins, pick from top liquid coins
        # to diversify across many assets instead of all piling into BTC
        if not coins:
            import random
            TOP_COINS = ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "ARB",
                         "OP", "SUI", "APT", "INJ", "SEI", "TIA", "JUP",
                         "WIF", "PEPE", "ONDO", "RENDER", "FET", "NEAR"]
            available = [c for c in TOP_COINS if c in mids]
            if available:
                # Use strategy ID as seed for deterministic but diverse selection
                strat_id = hash(str(strategy.get("id", random.random())))
                coins = [available[strat_id % len(available)]]
            else:
                coins = ["BTC", "ETH"]

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
        # CRITICAL: No conflicting sides on same asset (no long+short on same coin)
        for t in open_trades:
            if t["coin"] == signal["coin"] and t.get("side") != signal["side"]:
                logger.info(f"Risk: conflicting side for {signal['coin']} "
                           f"(have {t.get('side')}, want {signal['side']})")
                return False

        # Max position count — allow up to 20 simultaneous positions
        if len(open_trades) >= 20:
            logger.info(f"Risk: max positions ({len(open_trades)}/20)")
            return False

        # Max positions per coin — allow up to 3 per coin (same direction only)
        coin_positions = sum(1 for t in open_trades if t["coin"] == signal["coin"])
        if coin_positions >= 3:
            logger.info(f"Risk: max positions for {signal['coin']} ({coin_positions}/3)")
            return False

        # Max exposure per coin — 50% of balance
        coin_exposure = sum(
            t["size"] * t["entry_price"] * t["leverage"]
            for t in open_trades if t["coin"] == signal["coin"]
        )
        max_coin_exposure = account["balance"] * 0.50
        new_exposure = signal["size"] * signal["price"] * signal["leverage"]
        if coin_exposure + new_exposure > max_coin_exposure:
            logger.info(f"Risk: coin exposure for {signal['coin']} ${coin_exposure+new_exposure:,.0f} > ${max_coin_exposure:,.0f}")
            return False

        # Max total exposure — 500% of balance (leveraged)
        total_exposure = sum(
            t["size"] * t["entry_price"] * t["leverage"]
            for t in open_trades
        )
        max_total_exposure = account["balance"] * 5.0
        if total_exposure + new_exposure > max_total_exposure:
            logger.info(f"Risk: total exposure ${total_exposure+new_exposure:,.0f} > ${max_total_exposure:,.0f}")
            return False

        # Minimum balance remaining
        if account["balance"] < config.PAPER_TRADING_INITIAL_BALANCE * 0.05:
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
        Close any that hit stop-loss, take-profit, or trailing stop.
        Also implements trailing stop: if price moves 3%+ in our favor,
        ratchet the stop-loss to lock in at least breakeven.
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

            # --- Trailing stop logic ---
            # If price has moved 3%+ in our favor, tighten the stop to lock in profits
            entry = trade["entry_price"]
            leverage = trade.get("leverage", 1)
            sl = trade["stop_loss"]

            if trade["side"] == "long":
                move_pct = (current_price - entry) / entry
                if move_pct >= 0.03 and sl and sl < entry:
                    # Move SL to breakeven + 0.5%
                    new_sl = entry * 1.005
                    if new_sl > sl:
                        self._update_stop_loss(trade["id"], new_sl)
                        sl = new_sl
                elif move_pct >= 0.06 and sl:
                    # Trail SL at 2.5% below current price
                    trail_sl = current_price * (1 - 0.025 / max(leverage, 1))
                    if trail_sl > sl:
                        self._update_stop_loss(trade["id"], trail_sl)
                        sl = trail_sl
            else:
                move_pct = (entry - current_price) / entry
                if move_pct >= 0.03 and sl and sl > entry:
                    new_sl = entry * 0.995
                    if new_sl < sl:
                        self._update_stop_loss(trade["id"], new_sl)
                        sl = new_sl
                elif move_pct >= 0.06 and sl:
                    trail_sl = current_price * (1 + 0.025 / max(leverage, 1))
                    if trail_sl < sl:
                        self._update_stop_loss(trade["id"], trail_sl)
                        sl = trail_sl

            # Check stop loss (using potentially updated SL)
            if sl:
                if trade["side"] == "long" and current_price <= sl:
                    should_close = True
                    close_reason = "trailing_stop" if sl > trade["stop_loss"] else "stop_loss"
                elif trade["side"] == "short" and current_price >= sl:
                    should_close = True
                    close_reason = "trailing_stop" if sl < trade["stop_loss"] else "stop_loss"

            # Check take profit
            if trade["take_profit"] and not should_close:
                if trade["side"] == "long" and current_price >= trade["take_profit"]:
                    should_close = True
                    close_reason = "take_profit"
                elif trade["side"] == "short" and current_price <= trade["take_profit"]:
                    should_close = True
                    close_reason = "take_profit"

            # Time-based exit: close positions older than 24 hours
            if not should_close and trade.get("opened_at"):
                try:
                    opened = datetime.fromisoformat(trade["opened_at"])
                    age_hours = (datetime.utcnow() - opened).total_seconds() / 3600
                    if age_hours > 24:
                        should_close = True
                        close_reason = "time_exit_24h"
                except (ValueError, TypeError):
                    pass

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

    def _update_stop_loss(self, trade_id: int, new_sl: float):
        """Update stop loss for a trade (trailing stop)."""
        try:
            with db.get_connection() as conn:
                conn.execute(
                    "UPDATE paper_trades SET stop_loss = ? WHERE id = ? AND status = 'open'",
                    (round(new_sl, 2), trade_id)
                )
            logger.info(f"Trailing stop updated for trade {trade_id}: new SL=${new_sl:,.2f}")
        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")

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
