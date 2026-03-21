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
        """
        discovered = []

        # Method 0: Seed addresses — always check these
        for addr in SEED_TRADER_ADDRESSES:
            discovered.append({
                "address": addr,
                "total_pnl": 0,  # Will be populated during analysis
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

        logger.info(f"Total unique traders discovered: {len(unique)}")
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
        """Discover profitable traders who run vaults."""
        traders = []
        # Known popular vault addresses (these are discovered over time)
        # The bot will build this list as it runs
        known_vault_addresses = self._get_known_vault_addresses()

        for vault_addr in known_vault_addresses[:10]:  # limit API calls
            try:
                details = hl.get_vault_details(vault_addr)
                if details:
                    leader = details.get("leader", "")
                    if leader:
                        # Check leader's performance
                        pnl_history = details.get("portfolio", {}).get("pnlHistory", [])
                        total_pnl = sum(float(p[1]) for p in pnl_history) if pnl_history else 0

                        if total_pnl >= config.MIN_PNL_THRESHOLD:
                            traders.append({
                                "address": leader,
                                "total_pnl": total_pnl,
                                "roi_pct": 0,
                                "source": "vault",
                                "metadata": {"vault_address": vault_addr,
                                           "vault_name": details.get("name", "")},
                            })
            except Exception as e:
                logger.debug(f"Error fetching vault {vault_addr}: {e}")
                continue
            time.sleep(0.3)
        return traders

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

        # Build trader profile
        profile = {
            "address": address,
            "account_value": state["account_value"],
            "positions": positions,
            "position_analysis": position_analysis,
            "trade_analysis": trade_analysis,
            "total_margin_used": state["total_margin_used"],
            "num_open_positions": len([p for p in positions if p["size"] > 0]),
            "analyzed_at": datetime.utcnow().isoformat(),
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

    def run_discovery_cycle(self) -> Dict:
        """
        Run a full discovery and analysis cycle.
        Returns summary of what was found.
        """
        logger.info("Starting trader discovery cycle...")

        # Step 1: Discover traders
        discovered = self.discover_top_traders()

        # Step 2: Analyze each trader
        profiles = []
        for trader in discovered:
            try:
                profile = self.analyze_trader(trader["address"])
                if profile:
                    profiles.append(profile)
                    self.known_traders[trader["address"]] = trader
            except Exception as e:
                logger.error(f"Error analyzing trader {trader['address'][:10]}: {e}")
            time.sleep(0.5)  # Be gentle with the API

        # Step 3: Also re-analyze existing tracked traders
        for addr in list(self.known_traders.keys()):
            if addr not in {t["address"] for t in discovered}:
                try:
                    profile = self.analyze_trader(addr)
                    if profile:
                        profiles.append(profile)
                except Exception as e:
                    logger.debug(f"Error re-analyzing {addr[:10]}: {e}")
                time.sleep(0.5)

        summary = {
            "traders_discovered": len(discovered),
            "traders_analyzed": len(profiles),
            "total_tracked": len(self.known_traders),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Log the research cycle
        db.log_research_cycle(
            cycle_type="discovery",
            summary=f"Discovered {len(discovered)} traders, analyzed {len(profiles)}",
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
