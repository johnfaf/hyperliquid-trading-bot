"""
Trader Discovery Engine
Finds and tracks the most profitable traders on Hyperliquid.
Analyzes their positions, trading patterns, and performance over time.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src import hyperliquid_client as hl
from src import database as db

logger = logging.getLogger(__name__)



# ─── Seed Addresses ────────────────────────────────────────────
# Known profitable / high-profile Hyperliquid traders and vaults.
# The bot will discover more over time, but these bootstrap the system.
SEED_TRADER_ADDRESSES = [
    # HLP (Hyperliquidity Provider) vault
    "0xdfc24b077bc1425ad1dea75bcb6f8158e3df2f0f",
    # Well-known active traders (sourced from public leaderboard/community)
    "0x7e48d9a5de906dfb691e4fccc6899877ed3e8c0a",
    "0x23b07a1c8845ce4ac18ba19fa8b0e93445f28967",
    "0x5e9ee194bb8b548df01e6de9b3f38b4e56afd809",
    "0x34c22e6e1e4a6fa675e1350be0eca1eb5150a0a1",
    "0x1fa3f34bf91f50beef3dd0cb181a6f1b89b02e8f",
    "0x4e5b2e1dc63f6b91cb6cd759936495434c7e972f",
    "0xe67154a95b71e9850e27dcec80caf98e70f5786c",
    "0xc6f3dde1a1fac291052f43008acb50e4e280c5e0",
    "0x20e831d416dcb47a6bd191c5bf46ce59a7f5e28a",
    "0x8ab3ef18bea98a44c5871a11d90b14e3a6fb9ac2",
    "0x071bda50d482bae598c2e7a0dbdff4a81be7fe83",
    "0x6b67a2e3a96427daba798f9a0e9c529b28eaeb56",
    "0xa01c6c3e0cf5a11cc92c7400e82753638c205a45",
    "0x14d460010b4d94e7c891d4e78c52c10a8f867e8f",
]


class TraderDiscovery:
    """Discovers and monitors top Hyperliquid traders."""

    def __init__(self):
        self.known_traders = {}
        self._load_known_traders()

    def _load_known_traders(self):
        """Load previously tracked traders from the database."""
        traders = db.get_active_traders()
        for t in traders:
            self.known_traders[t["address"]] = t
        logger.info(f"Loaded {len(self.known_traders)} known traders from database")

    def discover_top_traders(self) -> List[Dict]:
        """
        Discover top traders via multiple methods:
        1. Seed addresses (known profitable wallets)
        2. Hyperliquid leaderboard API
        3. Vault leader analysis
        4. Whale detection from recent large trades

        Known bots (stored in DB from previous scans) are excluded upfront
        so we don't waste API calls re-scanning them.
        """
        # Load known bots from DB — these persist across redeploys
        known_bots = db.get_known_bot_addresses()
        if known_bots:
            logger.info(f"Loaded {len(known_bots)} known bots from DB — will skip these")

        discovered = []

        # Method 0: Seed addresses — always check these (never skip seeds)
        for addr in SEED_TRADER_ADDRESSES:
            discovered.append({
                "address": addr,
                "total_pnl": 0,
                "roi_pct": 0,
                "source": "seed",
                "metadata": {},
            })
        logger.info(f"Added {len(SEED_TRADER_ADDRESSES)} seed trader addresses")

        # Method 1: Try the leaderboard endpoint
        leaderboard = hl.get_leaderboard()
        if leaderboard:
            traders = self._parse_leaderboard(leaderboard)
            discovered.extend(traders)
            logger.info(f"Found {len(traders)} traders from leaderboard")
        else:
            logger.warning("Leaderboard returned no data")

        # Method 2: Analyze top vaults for their leaders
        vault_traders = self._discover_from_vaults()
        discovered.extend(vault_traders)
        logger.info(f"Found {len(vault_traders)} traders from vault analysis")

        # Method 3: Look at large position holders in top coins
        whale_traders = self._discover_whales()
        discovered.extend(whale_traders)
        logger.info(f"Found {len(whale_traders)} whale traders")

        # Deduplicate by address
        seen = set()
        unique = []
        for t in discovered:
            if t["address"] not in seen:
                seen.add(t["address"])
                unique.append(t)

        # Filter out known bots BEFORE returning candidates
        # This is the key optimisation: we don't waste API calls on addresses
        # already confirmed as bots in previous scans
        before_filter = len(unique)
        unique = [t for t in unique if t["address"] not in known_bots]
        bots_skipped = before_filter - len(unique)
        if bots_skipped:
            logger.info(f"Skipped {bots_skipped} known bots from candidate list "
                       f"({len(unique)} candidates remaining)")

        logger.info(f"Total unique traders discovered: {len(unique)} "
                   f"(after filtering {bots_skipped} known bots)")
        return unique[:config.MAX_TRACKED_TRADERS]

    def _parse_leaderboard(self, data) -> List[Dict]:
        """Parse leaderboard API response into trader dicts.
        Handles multiple possible response formats flexibly."""
        traders = []

        # Log the raw structure to help debug
        if isinstance(data, dict):
            logger.info(f"Leaderboard response keys: {list(data.keys())}")
        elif isinstance(data, list):
            logger.info(f"Leaderboard response is list of {len(data)} items")
            if data and isinstance(data[0], dict):
                logger.info(f"First entry keys: {list(data[0].keys())}")

        # Flatten: find the actual list of entries regardless of nesting
        entries = []
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            # Try every common key pattern
            for key in ["leaderboardRows", "rows", "data", "traders", "leaderboard",
                        "result", "results", "entries", "positions"]:
                if key in data and isinstance(data[key], list):
                    entries = data[key]
                    logger.info(f"Found entries under key '{key}': {len(entries)} items")
                    break
            if not entries:
                # Maybe the dict itself is a single-entry or nested differently
                # Try to find any list value
                for key, val in data.items():
                    if isinstance(val, list) and len(val) > 0:
                        entries = val
                        logger.info(f"Found list under key '{key}': {len(entries)} items")
                        break

        for entry in entries:
            try:
                if not isinstance(entry, dict):
                    continue

                # Try multiple address field names
                address = ""
                for addr_key in ["ethAddress", "address", "trader", "user",
                                 "account", "wallet", "traderAddress"]:
                    if addr_key in entry and entry[addr_key]:
                        address = str(entry[addr_key])
                        break

                if not address:
                    continue

                # Try multiple PnL field names
                pnl = 0
                for pnl_key in ["accountValue", "pnl", "totalPnl", "total_pnl",
                                "profit", "totalProfit", "windowPerformance"]:
                    if pnl_key in entry:
                        try:
                            pnl = float(entry[pnl_key])
                            break
                        except (ValueError, TypeError):
                            continue

                # Try multiple ROI field names
                roi = 0
                for roi_key in ["roi", "roiPct", "roi_pct", "returnOnEquity", "pctReturn"]:
                    if roi_key in entry:
                        try:
                            roi = float(entry[roi_key])
                            break
                        except (ValueError, TypeError):
                            continue

                display_name = entry.get("displayName", entry.get("name", entry.get("label", "")))

                # Accept all traders from leaderboard (they're already filtered by the API)
                traders.append({
                    "address": address,
                    "total_pnl": pnl,
                    "roi_pct": roi * 100 if 0 < abs(roi) < 10 else roi,
                    "source": "leaderboard",
                    "metadata": {"display_name": display_name,
                               "raw_entry_keys": list(entry.keys())[:10]},
                })

            except Exception as e:
                logger.debug(f"Skipping leaderboard entry: {e}")
                continue

        logger.info(f"Parsed {len(traders)} traders from leaderboard data")
        return traders

    def _discover_from_vaults(self) -> List[Dict]:
        """Vault details API returns 422 — skip to avoid wasting 30s per cycle."""
        return []

    def _discover_whales(self) -> List[Dict]:
        """
        Discover whale traders by checking recent large trades on popular coins.
        This method finds traders not on the leaderboard but still very active.
        """
        traders = []
        top_coins = ["BTC", "ETH", "SOL", "DOGE", "ARB"]

        for coin in top_coins:
            try:
                recent = hl.get_recent_trades(coin)
                if not recent:
                    continue

                # Find addresses with large trades
                large_trade_addresses = set()
                for trade in (recent if isinstance(recent, list) else []):
                    size_usd = float(trade.get("px", 0)) * float(trade.get("sz", 0))
                    if size_usd > 50_000:  # $50k+ trades
                        for user_key in ["user", "users"]:
                            if user_key in trade:
                                addrs = trade[user_key] if isinstance(trade[user_key], list) else [trade[user_key]]
                                large_trade_addresses.update(addrs)

                # Check these whale addresses
                for addr in list(large_trade_addresses)[:5]:
                    if addr not in self.known_traders:
                        state = hl.get_user_state(addr)
                        if state and state["account_value"] > 100_000:
                            traders.append({
                                "address": addr,
                                "total_pnl": 0,  # Will be updated on analysis
                                "roi_pct": 0,
                                "account_value": state["account_value"],
                                "source": "whale_detection",
                                "metadata": {"detected_on": coin},
                            })
            except Exception as e:
                logger.debug(f"Error in whale detection for {coin}: {e}")
            time.sleep(0.3)
        return traders

    def _get_known_vault_addresses(self) -> List[str]:
        """Get list of known vault addresses to check."""
        return [
            "0xdfc24b077bc1425ad1dea75bcb6f8158e3df2f0f",  # HLP (Hyperliquidity Provider)
            "0x1fa3f34bf91f50beef3dd0cb181a6f1b89b02e8f",
            "0x4e5b2e1dc63f6b91cb6cd759936495434c7e972f",
            "0x20e831d416dcb47a6bd191c5bf46ce59a7f5e28a",
            "0xa01c6c3e0cf5a11cc92c7400e82753638c205a45",
        ]

    def analyze_trader(self, address: str) -> Optional[Dict]:
        """
        Deep analysis of a single trader.
        Returns comprehensive trading profile.
        """
        logger.info(f"Analyzing trader: {address[:10]}...")

        # Get current state
        state = hl.get_user_state(address)
        if not state:
            logger.warning(f"Could not fetch state for {address}")
            return None

        # Get recent fills
        one_week_ago = int((datetime.utcnow() - timedelta(days=7)).timestamp() * 1000)
        fills = hl.get_user_fills(address, start_time=one_week_ago)

        # Analyze positions
        positions = state["positions"]
        position_analysis = self._analyze_positions(positions)

        # Analyze fills/trades
        trade_analysis = self._analyze_fills(fills)

        # Save snapshots of current positions
        for pos in positions:
            if pos["size"] > 0:
                db.save_position_snapshot(
                    trader_address=address,
                    coin=pos["coin"],
                    side=pos["side"],
                    size=pos["size"],
                    entry_price=pos["entry_price"],
                    leverage=pos["leverage"],
                    unrealized_pnl=pos["unrealized_pnl"],
                    margin_used=pos["margin_used"],
                )

        # V6: Adaptive bot detection (continuous probability 0-1)
        try:
            from src.adaptive_bot_detector import AdaptiveBotDetector
            _detector = AdaptiveBotDetector()
            bot_result = _detector.detect(fills, positions, trade_analysis, address)
            bot_score = config.BOT_THRESHOLD if bot_result.is_bot else 0
            if bot_result.is_bot:
                logger.info(f"Bot DETECTED (prob={bot_result.bot_probability:.0%}, "
                           f"conf={bot_result.confidence:.0%}): {address[:10]}... "
                           f"reason={bot_result.reason}")
        except Exception:
            # Fallback to legacy detection
            bot_score = self._get_bot_score(fills, positions, trade_analysis)

        # Build trader profile (always, even for suspected bots — let caller decide)
        profile = {
            "address": address,
            "account_value": state["account_value"],
            "positions": positions,
            "position_analysis": position_analysis,
            "trade_analysis": trade_analysis,
            "total_margin_used": state["total_margin_used"],
            "num_open_positions": len([p for p in positions if p["size"] > 0]),
            "analyzed_at": datetime.utcnow().isoformat(),
            "bot_score": bot_score,
        }

        # Update trader in database
        db.upsert_trader(
            address=address,
            total_pnl=trade_analysis.get("total_closed_pnl", 0),
            roi_pct=trade_analysis.get("avg_roi", 0),
            account_value=state["account_value"],
            win_rate=trade_analysis.get("win_rate", 0),
            trade_count=trade_analysis.get("total_trades", 0),
            metadata={"last_profile": profile.get("position_analysis", {})},
        )

        return profile

    def _analyze_positions(self, positions: List[Dict]) -> Dict:
        """Analyze a trader's current open positions for patterns."""
        if not positions:
            return {"style": "inactive", "bias": "neutral"}

        active = [p for p in positions if p["size"] > 0]
        if not active:
            return {"style": "inactive", "bias": "neutral"}

        longs = [p for p in active if p["side"] == "long"]
        shorts = [p for p in active if p["side"] == "short"]

        total_long_value = sum(p["size"] * p["entry_price"] for p in longs)
        total_short_value = sum(p["size"] * p["entry_price"] for p in shorts)
        total_value = total_long_value + total_short_value

        avg_leverage = sum(p["leverage"] for p in active) / len(active) if active else 1
        max_leverage = max(p["leverage"] for p in active) if active else 1

        # Determine trading style
        if avg_leverage > 10:
            leverage_style = "high_leverage"
        elif avg_leverage > 3:
            leverage_style = "moderate_leverage"
        else:
            leverage_style = "low_leverage"

        # Determine directional bias
        if total_value > 0:
            long_pct = total_long_value / total_value
            if long_pct > 0.7:
                bias = "strongly_long"
            elif long_pct > 0.55:
                bias = "slightly_long"
            elif long_pct < 0.3:
                bias = "strongly_short"
            elif long_pct < 0.45:
                bias = "slightly_short"
            else:
                bias = "neutral"
        else:
            bias = "neutral"

        # Concentration analysis
        coins_traded = list(set(p["coin"] for p in active))
        concentration = "concentrated" if len(coins_traded) <= 2 else "diversified"

        return {
            "num_positions": len(active),
            "num_longs": len(longs),
            "num_shorts": len(shorts),
            "total_notional": total_value,
            "long_pct": total_long_value / total_value if total_value else 0,
            "avg_leverage": avg_leverage,
            "max_leverage": max_leverage,
            "leverage_style": leverage_style,
            "bias": bias,
            "coins": coins_traded,
            "concentration": concentration,
            "total_unrealized_pnl": sum(p["unrealized_pnl"] for p in active),
        }

    def _analyze_fills(self, fills: List[Dict]) -> Dict:
        """Analyze a trader's recent trade fills for performance metrics."""
        if not fills:
            return {
                "total_trades": 0, "win_rate": 0, "total_closed_pnl": 0,
                "avg_roi": 0, "avg_trade_size": 0, "trading_frequency": "inactive"
            }

        total_trades = len(fills)
        closed_pnls = [f["closed_pnl"] for f in fills if f["closed_pnl"] != 0]
        winning_trades = len([p for p in closed_pnls if p > 0])
        losing_trades = len([p for p in closed_pnls if p < 0])

        total_closed_pnl = sum(closed_pnls)
        avg_win = sum(p for p in closed_pnls if p > 0) / winning_trades if winning_trades else 0
        avg_loss = sum(p for p in closed_pnls if p < 0) / losing_trades if losing_trades else 0

        win_rate = winning_trades / len(closed_pnls) if closed_pnls else 0

        # Trading frequency
        if fills:
            times = sorted([f["time"] for f in fills if f["time"]])
            if len(times) >= 2:
                span_hours = (times[-1] - times[0]) / (1000 * 3600)
                trades_per_day = total_trades / max(span_hours / 24, 1)
                if trades_per_day > 50:
                    frequency = "scalper"
                elif trades_per_day > 10:
                    frequency = "day_trader"
                elif trades_per_day > 2:
                    frequency = "swing_trader"
                else:
                    frequency = "position_trader"
            else:
                frequency = "unknown"
        else:
            frequency = "inactive"

        # Profit factor
        gross_profit = sum(p for p in closed_pnls if p > 0)
        gross_loss = abs(sum(p for p in closed_pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Coins traded
        coins = list(set(f["coin"] for f in fills))

        # Average trade size
        avg_size = sum(f["size"] * f["price"] for f in fills) / total_trades if total_trades else 0

        # Liquidation count
        liquidations = len([f for f in fills if f.get("is_liquidation")])

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_closed_pnl": total_closed_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_trade_size": avg_size,
            "trading_frequency": frequency,
            "coins_traded": coins,
            "liquidations": liquidations,
            "avg_roi": total_closed_pnl / (avg_size * total_trades) if avg_size * total_trades > 0 else 0,
        }

    def _fast_prescreen(self, address: str) -> bool:
        """
        Fast pre-screen using only 1 API call (get_user_state).
        Returns True if the trader passes and should go to deep analysis.
        Returns False to skip (saves expensive fills API call).
        """
        try:
            state = hl.get_user_state(address)
            if not state:
                return False

            account_value = state.get("account_value", 0)
            positions = state.get("positions", [])
            active = [p for p in positions if p["size"] > 0]

            # Reject: tiny accounts (< $100 value)
            if account_value < 100:
                return False

            # Reject: too many simultaneous positions (likely portfolio bot)
            if len(active) > 15:
                return False

            # Reject: no activity at all
            if account_value < 500 and len(active) == 0:
                return False

            return True

        except Exception as e:
            logger.debug(f"Pre-screen error for {address[:10]}: {e}")
            return False

    def _compute_trades_per_day(self, fills: List[Dict]) -> float:
        """
        Compute actual trades-per-day from fill timestamps.
        The Hyperliquid API caps at 2000 fills, so we can't use len(fills)
        as the real trade count — we must look at the TIME SPAN instead.
        """
        if len(fills) < 2:
            return 0.0
        times = sorted([f["time"] for f in fills if f.get("time")])
        if len(times) < 2:
            return 0.0
        span_ms = times[-1] - times[0]
        span_days = max(span_ms / (1000 * 86400), 0.01)  # avoid div by zero
        return len(fills) / span_days

    def _get_bot_score(self, fills: List[Dict], positions: List[Dict],
                       trade_analysis: Dict) -> int:
        """
        Detect if an account is likely a bot / market-maker / automated system.
        We want to focus on real human traders whose strategies are reproducible.

        IMPORTANT: The Hyperliquid fills API caps at 2000 results, so raw
        fill count is NOT reliable. We use trades_per_day (time-based) instead.

        Returns an integer bot_score (0 = human, higher = more bot-like).
        Score of 3+ is considered a bot, but caller can use the score
        for ranking/fallback when too many get flagged.

        Signals (each adds 1-2 points):
        - Extremely high trade frequency (>300 trades/day by time span)
        - Very small, uniform trade sizes (market making)
        - Near-zero PnL per trade with huge volume (arb bots)
        - Many simultaneous positions across many coins (>15 active)
        - Uniform trade sizes (low coefficient of variation)
        - High liquidation rate
        - Spread bot: high frequency + near-zero median PnL per trade
        """
        if not fills:
            return 0  # No data = can't determine, assume human

        total = len(fills)
        avg_size = trade_analysis.get("avg_trade_size", 0)
        pnl = trade_analysis.get("total_closed_pnl", 0)
        liquidations = trade_analysis.get("liquidations", 0)
        active_positions = [p for p in positions if p["size"] > 0]

        # Compute REAL frequency from timestamps (not raw count)
        trades_per_day = self._compute_trades_per_day(fills)

        # ─── HARD CUTOFFS (instant bot, no signal counting needed) ────
        # These override the signal-based scoring entirely.
        # Evidence from logs: 115,043 trades/day classified "Human-like" is unacceptable.

        # Hard cutoff 1: >100 trades/day = bot, period
        # A very active human scalper might do 30-80, but >100 is automation
        if trades_per_day > config.BOT_HARD_CUTOFF_TRADES:
            logger.info(f"Bot INSTANT: {trades_per_day:.0f} trades/day (hard cutoff >{config.BOT_HARD_CUTOFF_TRADES})")
            return 10  # Guaranteed bot score

        # Hard cutoff 2: Spread bot — high frequency + micro PnL per trade
        # Median PnL < $0.50 with >50 trades/day = market maker / funding farmer
        if trades_per_day > 50:
            closed_pnls = sorted([f["closed_pnl"] for f in fills if f["closed_pnl"] != 0])
            if closed_pnls:
                median_pnl = closed_pnls[len(closed_pnls) // 2]
                if abs(median_pnl) < 0.50:
                    logger.info(f"Bot INSTANT: spread/MM bot (median PnL=${median_pnl:.2f}, "
                               f"{trades_per_day:.0f} trades/day)")
                    return 10  # Guaranteed bot score

        # ─── SIGNAL-BASED SCORING (for borderline cases) ─────────────
        #
        # Previous bug: signals were weighted too lightly. A wallet with an
        # arb pattern (1 signal) or high liquidation rate (1 signal) would
        # score only 1-2, below BOT_THRESHOLD=3, and pass as "Human-like".
        #
        # Fix: ANY single clear bot signal now scores >= 3 (instant reject).
        # Weaker signals still accumulate and 2+ weak signals = bot.
        #
        bot_signals = 0

        # Signal 1: Elevated frequency — 50-100 trades/day is suspicious
        # This alone is strong evidence of automation
        if trades_per_day > config.BOT_ELEVATED_FREQ:
            logger.info(f"Bot signal: {trades_per_day:.0f} trades/day (elevated frequency >{config.BOT_ELEVATED_FREQ})")
            bot_signals += 3  # Strong: 50+ trades/day alone = bot

        # Signal 2: Very small avg trade size with moderate frequency = market maker
        if trades_per_day > 30 and avg_size < 50:
            logger.info(f"Bot signal: tiny trades (${avg_size:.0f} avg) at {trades_per_day:.0f}/day")
            bot_signals += 3  # Strong: micro-size + frequency = MM bot

        # Signal 3: Near-zero PnL per trade with high volume = arb bot
        # This is DEFINITIVE: no human trades 50+ times making <$0.50 per trade
        if total > 50:
            pnl_per_trade = abs(pnl) / total if total else 0
            if pnl_per_trade < 0.5 and avg_size > 1000:
                logger.info(f"Bot signal: arb pattern (${pnl_per_trade:.2f}/trade, ${avg_size:.0f} avg)")
                bot_signals += 3  # Strong: arb pattern alone = bot

        # Signal 4: Too many simultaneous positions = portfolio bot
        if len(active_positions) > 15:
            logger.info(f"Bot signal: {len(active_positions)} simultaneous positions")
            bot_signals += 2  # Medium: could be an active human, but suspicious

        # Signal 5: Uniform trade sizes (low variance = bot)
        if len(fills) > 20:
            sizes = [f["size"] * f["price"] for f in fills[:50]]
            if sizes:
                import numpy as np
                mean_size = np.mean(sizes)
                std_size = np.std(sizes)
                cv = std_size / mean_size if mean_size > 0 else 0
                if cv < 0.05:  # coefficient of variation < 5% = robotic
                    logger.info(f"Bot signal: uniform trade sizes (CV={cv:.3f})")
                    bot_signals += 2  # Medium: very uniform = likely automated

        # Signal 6: High liquidation rate = reckless bot or badly coded algo
        if total > 10 and liquidations / total > 0.15:
            logger.info(f"Bot signal: high liquidation rate ({liquidations}/{total})")
            bot_signals += 3  # Strong: >15% liquidation rate = not a strategy worth copying

        # Signal 7: Near-zero median PnL (even at moderate frequency) = spread/funding bot
        if trades_per_day > 20:
            closed_pnls = sorted([f["closed_pnl"] for f in fills if f["closed_pnl"] != 0])
            if closed_pnls:
                median_pnl = closed_pnls[len(closed_pnls) // 2]
                if abs(median_pnl) < 1.0:
                    logger.info(f"Bot signal: micro PnL (median=${median_pnl:.2f}, {trades_per_day:.0f} trades/day)")
                    bot_signals += 3  # Strong: micro PnL + frequency = MM/funding bot

        addr_short = fills[0].get('user', 'unknown')[:10] if fills else 'unknown'
        if bot_signals >= 3:
            logger.info(f"Bot REJECTED ({bot_signals} signals, {trades_per_day:.0f} trades/day): "
                       f"{addr_short}...")
        elif bot_signals > 0:
            logger.info(f"Borderline ({bot_signals} signals, {trades_per_day:.0f} trades/day): "
                       f"{addr_short}... — passed but flagged")
        else:
            logger.info(f"Human-like (clean, {trades_per_day:.0f} trades/day)")
        return bot_signals

    def run_discovery_cycle(self) -> Dict:
        """
        Run a full discovery and analysis cycle with TWO-PHASE screening.

        Known bots (persisted in DB) are excluded before any API calls.
        Scanning uses batched processing with adaptive sleep to avoid 429 storms
        even with 2000 trader candidates.

        Phase 1 (Fast): Pre-screen candidates with 1 API call each.
                         Reject tiny accounts, bots with 15+ positions, inactive wallets.
        Phase 2 (Deep): Full analysis (fills + bot detection) only on candidates
                         that pass pre-screen. Mark detected bots as inactive in DB.
        """
        logger.info("=" * 60)
        logger.info("Starting TWO-PHASE trader discovery cycle...")
        logger.info("=" * 60)

        # Step 1: Discover candidate addresses (leaderboard + seeds + whales)
        # discover_top_traders() already filters out known bots from DB
        discovered = self.discover_top_traders()
        total_candidates = len(discovered)
        logger.info(f"Total candidate addresses (bots pre-filtered): {total_candidates}")

        # ─── Phase 1: Fast Pre-Screen ────────────────────────────────
        # Batched: process in groups of 50, with a longer pause between batches
        PRESCREEN_BATCH_SIZE = 50
        PRESCREEN_SLEEP = 0.8        # 800ms between individual calls
        PRESCREEN_BATCH_PAUSE = 5.0  # 5s pause between batches of 50

        logger.info(f"PHASE 1: Fast pre-screening {total_candidates} traders "
                   f"(batches of {PRESCREEN_BATCH_SIZE}, {PRESCREEN_SLEEP}s per call)...")
        passed_prescreen = []
        skipped_prescreen = 0
        already_known_active = 0
        rate_limit_hits = 0

        for i, trader in enumerate(discovered):
            addr = trader["address"]

            # Progress logging every 100 traders
            if (i + 1) % 100 == 0:
                logger.info(f"  Pre-screen progress: {i+1}/{total_candidates} "
                           f"(passed: {len(passed_prescreen)}, skipped: {skipped_prescreen}, "
                           f"429s: {rate_limit_hits})")

            # Batch pause every PRESCREEN_BATCH_SIZE calls
            if i > 0 and i % PRESCREEN_BATCH_SIZE == 0:
                logger.debug(f"  Batch pause ({PRESCREEN_BATCH_PAUSE}s) after {i} pre-screens")
                time.sleep(PRESCREEN_BATCH_PAUSE)

            # Skip traders already known and recently analyzed (within 6 hours)
            existing = self.known_traders.get(addr)
            if existing and existing.get("trade_count", 0) > 0:
                passed_prescreen.append(trader)
                already_known_active += 1
                continue

            # Fast pre-screen: 1 API call
            try:
                if self._fast_prescreen(addr):
                    passed_prescreen.append(trader)
                else:
                    skipped_prescreen += 1
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str:
                    rate_limit_hits += 1
                    # Back off harder on rate limit
                    backoff = min(30, 5 * (1 + rate_limit_hits))
                    logger.warning(f"  Rate limited during pre-screen ({rate_limit_hits}x), "
                                  f"backing off {backoff:.0f}s")
                    time.sleep(backoff)
                    # Retry once
                    try:
                        if self._fast_prescreen(addr):
                            passed_prescreen.append(trader)
                        else:
                            skipped_prescreen += 1
                    except:
                        skipped_prescreen += 1
                else:
                    skipped_prescreen += 1

            time.sleep(PRESCREEN_SLEEP)

        logger.info(f"PHASE 1 complete: {len(passed_prescreen)} passed, "
                    f"{skipped_prescreen} rejected, {already_known_active} already tracked, "
                    f"{rate_limit_hits} rate-limit hits")

        # ─── Phase 2: Deep Analysis ──────────────────────────────────
        # Wider spacing: deep analysis makes 2+ API calls per trader
        DEEP_BATCH_SIZE = 30
        DEEP_SLEEP = 1.2            # 1.2s between individual calls
        DEEP_BATCH_PAUSE = 10.0     # 10s pause between batches of 30

        logger.info(f"PHASE 2: Deep analyzing {len(passed_prescreen)} traders "
                   f"(batches of {DEEP_BATCH_SIZE}, {DEEP_SLEEP}s per call)...")
        all_profiles = []
        rate_limit_hits_deep = 0

        for i, trader in enumerate(passed_prescreen):
            addr = trader["address"]

            # Progress logging every 50 traders
            if (i + 1) % 50 == 0:
                human_count = len([p for p in all_profiles if p.get("bot_score", 0) < 3])
                logger.info(f"  Deep analysis progress: {i+1}/{len(passed_prescreen)} "
                           f"(human-like: {human_count}, total: {len(all_profiles)}, "
                           f"429s: {rate_limit_hits_deep})")

            # Batch pause
            if i > 0 and i % DEEP_BATCH_SIZE == 0:
                logger.debug(f"  Batch pause ({DEEP_BATCH_PAUSE}s) after {i} deep analyses")
                time.sleep(DEEP_BATCH_PAUSE)

            try:
                profile = self.analyze_trader(addr)
                if profile:
                    all_profiles.append(profile)
                    self.known_traders[addr] = trader
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str:
                    rate_limit_hits_deep += 1
                    backoff = min(60, 10 * (1 + rate_limit_hits_deep))
                    logger.warning(f"  Rate limited during deep analysis ({rate_limit_hits_deep}x), "
                                  f"backing off {backoff:.0f}s")
                    time.sleep(backoff)
                    # Retry once
                    try:
                        profile = self.analyze_trader(addr)
                        if profile:
                            all_profiles.append(profile)
                            self.known_traders[addr] = trader
                    except:
                        logger.error(f"Retry also failed for {addr[:10]}")
                else:
                    logger.error(f"Error analyzing trader {addr[:10]}: {e}")

            time.sleep(DEEP_SLEEP)

        # ─── Separate humans from bots, with guaranteed minimum ─────
        BOT_THRESHOLD = config.BOT_THRESHOLD
        MIN_TRADERS = 20

        humans = [p for p in all_profiles if p.get("bot_score", 0) < BOT_THRESHOLD]
        bots = [p for p in all_profiles if p.get("bot_score", 0) >= BOT_THRESHOLD]

        # If we don't have enough human traders, promote the least-bot-like bots
        profiles = humans
        promoted_bots = 0
        if len(profiles) < MIN_TRADERS and bots:
            bots_sorted = sorted(bots, key=lambda p: p.get("bot_score", 99))
            needed = MIN_TRADERS - len(profiles)
            promoted = bots_sorted[:needed]
            profiles.extend(promoted)
            promoted_bots = len(promoted)
            logger.info(f"Only {len(humans)} human traders found — promoted {promoted_bots} "
                       f"least-bot-like accounts (scores: {[p.get('bot_score',0) for p in promoted]})")

        # Mark high-confidence bots as inactive (score 3+, and not promoted)
        # Lowered from 4 to 3 so ALL detected bots get persisted and skipped next cycle
        promoted_addrs = {p["address"] for p in profiles}
        bots_marked = 0
        for p in bots:
            if p["address"] not in promoted_addrs and p.get("bot_score", 0) >= BOT_THRESHOLD:
                db.upsert_trader(
                    address=p["address"],
                    total_pnl=0, roi_pct=0,
                    account_value=p.get("account_value", 0),
                    win_rate=0, trade_count=0,
                    metadata={"status": "bot_detected",
                              "bot_score": p.get("bot_score", 0),
                              "detected_at": datetime.utcnow().isoformat()},
                    is_active=False,
                )
                bots_marked += 1

        logger.info(f"PHASE 2 complete: {len(humans)} human, {len(bots)} bot-like, "
                    f"{promoted_bots} promoted, {bots_marked} marked inactive in DB")
        logger.info(f"Final trader pool: {len(profiles)} traders for strategy analysis")
        logger.info(f"Rate limit hits: {rate_limit_hits} (pre-screen) + "
                   f"{rate_limit_hits_deep} (deep)")

        # ─── Summary ─────────────────────────────────────────────────
        known_bots_total = len(db.get_known_bot_addresses())
        summary = {
            "traders_discovered": total_candidates,
            "traders_prescreened": len(passed_prescreen),
            "traders_analyzed": len(all_profiles),
            "human_traders": len(humans),
            "bot_like_traders": len(bots),
            "promoted_bots": promoted_bots,
            "bots_marked_inactive": bots_marked,
            "bots_in_db_total": known_bots_total,
            "final_pool": len(profiles),
            "total_tracked": len(self.known_traders),
            "rate_limit_hits": rate_limit_hits + rate_limit_hits_deep,
            "timestamp": datetime.utcnow().isoformat(),
        }

        db.log_research_cycle(
            cycle_type="discovery",
            summary=(f"Scanned {total_candidates} → pre-screened {len(passed_prescreen)} "
                    f"→ analyzed {len(all_profiles)} → {len(profiles)} final pool "
                    f"({len(humans)} human + {promoted_bots} promoted) | "
                    f"{known_bots_total} bots in DB, {rate_limit_hits + rate_limit_hits_deep} 429s"),
            details=summary,
            traders_analyzed=len(profiles),
        )

        logger.info(f"Discovery cycle complete: {summary}")
        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    db.init_db()
    discovery = TraderDiscovery()
    result = discovery.run_discovery_cycle()
    print(f"\nDiscovery result: {result}")
