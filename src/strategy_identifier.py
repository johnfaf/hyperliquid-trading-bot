"""
Strategy Identifier & Classifier
Analyzes trader behavior patterns to identify and classify trading strategies.
Detects patterns like: momentum, mean-reversion, funding arbitrage, breakout, etc.
"""
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src import database as db
from src import hyperliquid_client as hl

logger = logging.getLogger(__name__)


# ─── Strategy Types ────────────────────────────────────────────

STRATEGY_TYPES = {
    "momentum_long": "Rides upward trends with leveraged long positions",
    "momentum_short": "Rides downward trends with leveraged short positions",
    "mean_reversion": "Buys dips and sells rips, expecting price to revert to mean",
    "breakout": "Enters positions on price breakouts from consolidation ranges",
    "scalping": "High-frequency small-profit trades with tight stops",
    "swing_trading": "Medium-term directional bets held for days",
    "funding_arb": "Captures funding rate differentials between long/short positions",
    "delta_neutral": "Hedged positions capturing yield while minimizing directional risk",
    "trend_following": "Follows established trends across multiple timeframes",
    "contrarian": "Takes positions against prevailing market sentiment",
    "concentrated_bet": "Large single-asset positions with high conviction",
    "diversified_portfolio": "Spread across many assets with managed risk",
}


class StrategyIdentifier:
    """Identifies trading strategies from trader behavior data."""

    def __init__(self):
        self.market_context = {}
        self._refresh_market_context()

    def _refresh_market_context(self):
        """Get current market conditions for context."""
        try:
            contexts = hl.get_asset_contexts()
            if contexts:
                self.market_context = contexts
        except Exception as e:
            logger.warning(f"Could not refresh market context: {e}")

    def identify_strategies(self, trader_profile: Dict) -> List[Dict]:
        """
        Analyze a trader profile and identify the strategies being used.
        Returns list of identified strategy dicts.
        """
        strategies = []

        positions = trader_profile.get("positions", [])
        pos_analysis = trader_profile.get("position_analysis", {})
        trade_analysis = trader_profile.get("trade_analysis", {})
        address = trader_profile.get("address", "unknown")

        active_positions = [p for p in positions if p["size"] > 0]

        if not active_positions and not trade_analysis.get("total_trades"):
            return strategies

        # Run each detection method
        detectors = [
            self._detect_momentum,
            self._detect_mean_reversion,
            self._detect_scalping,
            self._detect_swing_trading,
            self._detect_funding_arb,
            self._detect_delta_neutral,
            self._detect_concentrated_bet,
            self._detect_trend_following,
            self._detect_breakout,
        ]

        for detector in detectors:
            try:
                result = detector(active_positions, pos_analysis, trade_analysis, address)
                if result:
                    strategies.append(result)
            except Exception as e:
                logger.debug(f"Strategy detection error in {detector.__name__}: {e}")

        # Score and rank the identified strategies
        strategies.sort(key=lambda s: s.get("confidence", 0), reverse=True)

        if strategies:
            logger.debug(f"Identified {len(strategies)} strategies for trader {address[:10]}")
        return strategies

    def _detect_momentum(self, positions, pos_analysis, trade_analysis, address) -> Optional[Dict]:
        """Detect momentum trading (strong directional bias + leverage)."""
        bias = pos_analysis.get("bias", "neutral")
        avg_lev = pos_analysis.get("avg_leverage", 1)
        win_rate = trade_analysis.get("win_rate", 0)

        if bias in ("strongly_long", "strongly_short") and avg_lev > 2:
            direction = "long" if "long" in bias else "short"
            confidence = min(0.9, 0.4 + (avg_lev / 20) + (win_rate * 0.3))

            coins = pos_analysis.get("coins", [])
            return {
                "type": f"momentum_{direction}",
                "description": STRATEGY_TYPES[f"momentum_{direction}"],
                "confidence": confidence,
                "parameters": {
                    "direction": direction,
                    "avg_leverage": avg_lev,
                    "coins": coins,
                    "win_rate": win_rate,
                },
                "trader_address": address,
                "metrics": {
                    "pnl": trade_analysis.get("total_closed_pnl", 0),
                    "trade_count": trade_analysis.get("total_trades", 0),
                    "win_rate": win_rate,
                    "profit_factor": trade_analysis.get("profit_factor", 0),
                },
            }
        return None

    def _detect_mean_reversion(self, positions, pos_analysis, trade_analysis, address) -> Optional[Dict]:
        """Detect mean reversion strategy (buying dips / selling rips)."""
        if not positions:
            return None

        # Look for positions entered against recent price moves
        reversion_signals = 0
        total_checked = 0

        for pos in positions:
            coin = pos["coin"]
            if coin in self.market_context:
                ctx = self.market_context[coin]
                mark = ctx.get("mark_price", 0)
                oracle = ctx.get("oracle_price", 0)

                if mark and oracle and pos["entry_price"]:
                    # If long and entry is below current mark (bought the dip)
                    if pos["side"] == "long" and pos["entry_price"] < mark * 0.98:
                        reversion_signals += 1
                    # If short and entry is above current mark (sold the rip)
                    elif pos["side"] == "short" and pos["entry_price"] > mark * 1.02:
                        reversion_signals += 1
                    total_checked += 1

        if total_checked > 0 and reversion_signals / total_checked > 0.5:
            confidence = min(0.85, 0.3 + (reversion_signals / total_checked) * 0.4 +
                           trade_analysis.get("win_rate", 0) * 0.2)
            return {
                "type": "mean_reversion",
                "description": STRATEGY_TYPES["mean_reversion"],
                "confidence": confidence,
                "parameters": {
                    "reversion_pct": reversion_signals / total_checked,
                    "avg_leverage": pos_analysis.get("avg_leverage", 1),
                },
                "trader_address": address,
                "metrics": {
                    "pnl": trade_analysis.get("total_closed_pnl", 0),
                    "trade_count": trade_analysis.get("total_trades", 0),
                    "win_rate": trade_analysis.get("win_rate", 0),
                    "profit_factor": trade_analysis.get("profit_factor", 0),
                },
            }
        return None

    def _detect_scalping(self, positions, pos_analysis, trade_analysis, address) -> Optional[Dict]:
        """Detect scalping (high frequency, small profits)."""
        frequency = trade_analysis.get("trading_frequency", "")
        if frequency == "scalper" and trade_analysis.get("total_trades", 0) > 20:
            confidence = min(0.9, 0.5 + trade_analysis.get("win_rate", 0) * 0.3 +
                           min(trade_analysis.get("total_trades", 0) / 200, 0.2))
            return {
                "type": "scalping",
                "description": STRATEGY_TYPES["scalping"],
                "confidence": confidence,
                "parameters": {
                    "trades_per_day_est": trade_analysis.get("total_trades", 0),
                    "avg_trade_size": trade_analysis.get("avg_trade_size", 0),
                    "coins": trade_analysis.get("coins_traded", []),
                },
                "trader_address": address,
                "metrics": {
                    "pnl": trade_analysis.get("total_closed_pnl", 0),
                    "trade_count": trade_analysis.get("total_trades", 0),
                    "win_rate": trade_analysis.get("win_rate", 0),
                    "profit_factor": trade_analysis.get("profit_factor", 0),
                },
            }
        return None

    def _detect_swing_trading(self, positions, pos_analysis, trade_analysis, address) -> Optional[Dict]:
        """Detect swing trading (medium-term holds with moderate leverage)."""
        frequency = trade_analysis.get("trading_frequency", "")
        avg_lev = pos_analysis.get("avg_leverage", 1)

        if frequency in ("swing_trader", "position_trader") and 1 < avg_lev < 10:
            confidence = min(0.85, 0.4 + trade_analysis.get("win_rate", 0) * 0.3 +
                           (trade_analysis.get("profit_factor", 1) / 5) * 0.2)
            return {
                "type": "swing_trading",
                "description": STRATEGY_TYPES["swing_trading"],
                "confidence": confidence,
                "parameters": {
                    "avg_leverage": avg_lev,
                    "frequency": frequency,
                    "num_positions": pos_analysis.get("num_positions", 0),
                },
                "trader_address": address,
                "metrics": {
                    "pnl": trade_analysis.get("total_closed_pnl", 0),
                    "trade_count": trade_analysis.get("total_trades", 0),
                    "win_rate": trade_analysis.get("win_rate", 0),
                    "profit_factor": trade_analysis.get("profit_factor", 0),
                },
            }
        return None

    def _detect_funding_arb(self, positions, pos_analysis, trade_analysis, address) -> Optional[Dict]:
        """Detect funding rate arbitrage."""
        if not positions or not self.market_context:
            return None

        # Check if positions are in coins with extreme funding rates
        funding_aligned = 0
        for pos in positions:
            coin = pos["coin"]
            if coin in self.market_context:
                funding = self.market_context[coin].get("funding", 0)
                # Short when funding is very positive (earning funding)
                # Long when funding is very negative (earning funding)
                if (pos["side"] == "short" and funding > 0.0001) or \
                   (pos["side"] == "long" and funding < -0.0001):
                    funding_aligned += 1

        if len(positions) > 0 and funding_aligned / len(positions) > 0.6:
            confidence = min(0.8, 0.4 + (funding_aligned / len(positions)) * 0.4)
            return {
                "type": "funding_arb",
                "description": STRATEGY_TYPES["funding_arb"],
                "confidence": confidence,
                "parameters": {
                    "funding_aligned_pct": funding_aligned / len(positions),
                    "positions": [{"coin": p["coin"], "side": p["side"]} for p in positions],
                },
                "trader_address": address,
                "metrics": {
                    "pnl": trade_analysis.get("total_closed_pnl", 0),
                    "trade_count": trade_analysis.get("total_trades", 0),
                    "win_rate": trade_analysis.get("win_rate", 0),
                },
            }
        return None

    def _detect_delta_neutral(self, positions, pos_analysis, trade_analysis, address) -> Optional[Dict]:
        """Detect delta-neutral strategies (balanced long/short exposure)."""
        long_pct = pos_analysis.get("long_pct", 0.5)
        num_pos = pos_analysis.get("num_positions", 0)

        if num_pos >= 2 and 0.4 <= long_pct <= 0.6:
            confidence = min(0.8, 0.3 + (1 - abs(long_pct - 0.5) * 4) * 0.4 +
                           min(num_pos / 10, 0.2))
            return {
                "type": "delta_neutral",
                "description": STRATEGY_TYPES["delta_neutral"],
                "confidence": confidence,
                "parameters": {
                    "long_pct": long_pct,
                    "num_positions": num_pos,
                },
                "trader_address": address,
                "metrics": {
                    "pnl": trade_analysis.get("total_closed_pnl", 0),
                    "trade_count": trade_analysis.get("total_trades", 0),
                    "win_rate": trade_analysis.get("win_rate", 0),
                },
            }
        return None

    def _detect_concentrated_bet(self, positions, pos_analysis, trade_analysis, address) -> Optional[Dict]:
        """Detect concentrated high-conviction bets."""
        concentration = pos_analysis.get("concentration", "")
        num_pos = pos_analysis.get("num_positions", 0)
        avg_lev = pos_analysis.get("avg_leverage", 1)

        if concentration == "concentrated" and num_pos <= 2 and avg_lev > 3:
            confidence = min(0.85, 0.4 + (avg_lev / 20) * 0.3)
            coins = pos_analysis.get("coins", [])
            return {
                "type": "concentrated_bet",
                "description": STRATEGY_TYPES["concentrated_bet"],
                "confidence": confidence,
                "parameters": {
                    "coins": coins,
                    "avg_leverage": avg_lev,
                    "total_notional": pos_analysis.get("total_notional", 0),
                },
                "trader_address": address,
                "metrics": {
                    "pnl": trade_analysis.get("total_closed_pnl", 0),
                    "trade_count": trade_analysis.get("total_trades", 0),
                    "win_rate": trade_analysis.get("win_rate", 0),
                },
            }
        return None

    def _detect_trend_following(self, positions, pos_analysis, trade_analysis, address) -> Optional[Dict]:
        """Detect trend-following strategies."""
        if not positions:
            return None

        # Check if positions align with recent price trends
        trend_aligned = 0
        for pos in positions:
            coin = pos["coin"]
            if coin in self.market_context:
                mark = self.market_context[coin].get("mark_price", 0)
                entry = pos["entry_price"]
                if mark and entry:
                    price_change = (mark - entry) / entry
                    # Long and price went up, or short and price went down
                    if (pos["side"] == "long" and price_change > 0.01) or \
                       (pos["side"] == "short" and price_change < -0.01):
                        trend_aligned += 1

        if len(positions) > 0 and trend_aligned / len(positions) > 0.6:
            pf = trade_analysis.get("profit_factor", 1)
            confidence = min(0.8, 0.3 + (trend_aligned / len(positions)) * 0.3 +
                           min(pf / 5, 0.2))
            return {
                "type": "trend_following",
                "description": STRATEGY_TYPES["trend_following"],
                "confidence": confidence,
                "parameters": {
                    "trend_alignment": trend_aligned / len(positions),
                    "avg_leverage": pos_analysis.get("avg_leverage", 1),
                },
                "trader_address": address,
                "metrics": {
                    "pnl": trade_analysis.get("total_closed_pnl", 0),
                    "trade_count": trade_analysis.get("total_trades", 0),
                    "win_rate": trade_analysis.get("win_rate", 0),
                    "profit_factor": trade_analysis.get("profit_factor", 0),
                },
            }
        return None

    def _detect_breakout(self, positions, pos_analysis, trade_analysis, address) -> Optional[Dict]:
        """Detect breakout trading."""
        # Breakout traders typically have recent entries near current price with good PnL
        if not positions:
            return None

        recent_entries = 0
        for pos in positions:
            coin = pos["coin"]
            if coin in self.market_context:
                mark = self.market_context[coin].get("mark_price", 0)
                entry = pos["entry_price"]
                if mark and entry:
                    distance = abs(mark - entry) / mark
                    if distance < 0.03 and pos["unrealized_pnl"] > 0:
                        recent_entries += 1

        if len(positions) > 0 and recent_entries / len(positions) > 0.5:
            confidence = min(0.7, 0.3 + (recent_entries / len(positions)) * 0.3)
            return {
                "type": "breakout",
                "description": STRATEGY_TYPES["breakout"],
                "confidence": confidence,
                "parameters": {
                    "recent_entry_pct": recent_entries / len(positions),
                },
                "trader_address": address,
                "metrics": {
                    "pnl": trade_analysis.get("total_closed_pnl", 0),
                    "trade_count": trade_analysis.get("total_trades", 0),
                    "win_rate": trade_analysis.get("win_rate", 0),
                },
            }
        return None

    # Minimum confidence to save a strategy to DB (prevents strategy bloat)
    MIN_SAVE_CONFIDENCE = 0.60

    def save_identified_strategies(self, strategies: List[Dict]) -> List[int]:
        """
        Save identified strategies to the database with pre-save filtering
        and batch inserts for performance.

        Fixes applied:
        - Pre-save confidence filter: discard strategies below MIN_SAVE_CONFIDENCE
        - Trader address in logs: trace every strategy to its source wallet
        - Batch DB insert: single transaction instead of one-by-one
        """
        # Step 1: Pre-save filter — reject low-confidence garbage before DB
        qualified = []
        discarded = 0
        for strat in strategies:
            conf = strat.get("confidence", 0)
            addr = strat.get("trader_address", "unknown")[:10]
            stype = strat.get("type", "unknown")

            if conf < self.MIN_SAVE_CONFIDENCE:
                logger.debug(f"Discarded {stype} from {addr}... — "
                           f"confidence too low ({conf:.2f} < {self.MIN_SAVE_CONFIDENCE})")
                discarded += 1
                continue
            qualified.append(strat)

        if discarded > 0:
            logger.info(f"Pre-save filter: discarded {discarded}/{len(strategies)} "
                       f"low-confidence strategies (< {self.MIN_SAVE_CONFIDENCE})")

        if not qualified:
            return []

        # Step 2: Prepare batch data
        batch_data = []
        for strat in qualified:
            metrics = strat.get("metrics", {})
            addr = strat.get("trader_address", "unknown")[:8]
            batch_data.append({
                "name": f"{strat['type']}_{addr}",
                "description": strat.get("description", ""),
                "strategy_type": strat["type"],
                "parameters": strat.get("parameters", {}),
                "total_pnl": metrics.get("pnl", 0),
                "trade_count": metrics.get("trade_count", 0),
                "win_rate": metrics.get("win_rate", 0),
                "sharpe_ratio": 0,
            })

        # Step 3: Batch insert (single transaction)
        try:
            saved_ids = db.save_strategies_batch(batch_data)

            # Log with trader address for traceability
            for strat, sid in zip(qualified, saved_ids):
                addr = strat.get("trader_address", "unknown")[:10]
                logger.info(f"Saved strategy for {addr}...: {strat['type']} "
                           f"(confidence: {strat['confidence']:.2f})")

            logger.info(f"Batch saved {len(saved_ids)} strategies "
                       f"({discarded} discarded, {len(strategies)} total input)")
            return saved_ids

        except Exception as e:
            logger.error(f"Batch save error: {e} — falling back to individual inserts")
            # Fallback to one-by-one if batch fails
            saved_ids = []
            for strat in qualified:
                try:
                    metrics = strat.get("metrics", {})
                    addr = strat.get("trader_address", "unknown")[:8]
                    strategy_id = db.save_strategy(
                        name=f"{strat['type']}_{addr}",
                        description=strat.get("description", ""),
                        strategy_type=strat["type"],
                        parameters=strat.get("parameters", {}),
                        total_pnl=metrics.get("pnl", 0),
                        trade_count=metrics.get("trade_count", 0),
                        win_rate=metrics.get("win_rate", 0),
                        sharpe_ratio=0,
                    )
                    saved_ids.append(strategy_id)
                except Exception as inner_e:
                    logger.error(f"Error saving strategy {strat.get('type')}: {inner_e}")
            return saved_ids


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    identifier = StrategyIdentifier()
    print(f"Market context loaded for {len(identifier.market_context)} coins")
    print("Strategy identifier ready.")
