"""
Trader Discovery Engine
Finds and tracks the most profitable traders on Hyperliquid.
Analyzes their positions, trading patterns, and performance over time.
"""
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data import hyperliquid_client as hl
from src.data import database as db

logger = logging.getLogger(__name__)

# Cached leaderboard response schema key so we only probe all candidates once
# and then use the known key on every subsequent call.
_leaderboard_schema_key: Optional[str] = None


def _detect_leaderboard_schema(data) -> Tuple[List, Optional[str]]:
    """
    Detect which key holds the list of trader entries in a leaderboard response.
    Caches the discovered key in _leaderboard_schema_key so future calls can
    skip the probe loop entirely.

    Returns (entries_list, detected_key_or_None).
    """
    global _leaderboard_schema_key

    if isinstance(data, list):
        return data, None  # Response IS the list

    if isinstance(data, dict):
        # Use cached key if we've seen this API before
        if _leaderboard_schema_key and _leaderboard_schema_key in data:
            val = data[_leaderboard_schema_key]
            if isinstance(val, list):
                return val, _leaderboard_schema_key

        # Probe known candidates
        for key in ["leaderboardRows", "rows", "data", "traders", "leaderboard",
                    "result", "results", "entries", "positions"]:
            if key in data and isinstance(data[key], list):
                _leaderboard_schema_key = key
                logger.info("Leaderboard schema detected: key='%s' (%d entries)", key, len(data[key]))
                return data[key], key

        # Fallback: first list-valued key with content
        for key, val in data.items():
            if isinstance(val, list) and len(val) > 0:
                _leaderboard_schema_key = key
                logger.info("Leaderboard schema fallback: key='%s' (%d entries)", key, len(val))
                return val, key

    return [], None



# ─── Seed Addresses ────────────────────────────────────────────
# Known profitable / high-profile Hyperliquid traders and vaults.
# The bot will discover more over time, but these bootstrap the system.
# Override via SEED_TRADER_ADDRESSES env var (comma-separated hex addresses).
_DEFAULT_SEED_ADDRESSES = [
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

def _load_seed_addresses() -> list:
    env_val = os.environ.get("SEED_TRADER_ADDRESSES", "").strip()
    if env_val:
        addrs = [a.strip() for a in env_val.split(",") if a.strip().startswith("0x")]
        if addrs:
            return addrs
    return _DEFAULT_SEED_ADDRESSES

SEED_TRADER_ADDRESSES = _load_seed_addresses()


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

        # Method 4: Active traders on high-OI coins (volume-weighted)
        oi_traders = self._discover_from_open_interest()
        discovered.extend(oi_traders)
        logger.info(f"Found {len(oi_traders)} traders from open-interest scan")

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
        Handles multiple possible response formats flexibly.

        Uses module-level schema detection/caching so the probe loop runs at
        most once per process lifetime.  Subsequent calls skip straight to the
        known key.
        """
        traders = []

        entries, detected_key = _detect_leaderboard_schema(data)

        if not entries:
            if isinstance(data, dict):
                logger.info("Leaderboard response keys: %s", list(data.keys()))
            logger.warning("Could not find entry list in leaderboard response")
            return traders

        logger.info("Leaderboard: %d entries (schema key=%s)", len(entries), detected_key or "root-list")

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
        Discover whale traders by checking recent large trades across many coins.
        Finds traders not on the leaderboard but moving serious size.
        """
        traders = []
        # Expanded to 25 coins for much wider coverage
        whale_coins = [
            "BTC", "ETH", "SOL", "DOGE", "ARB", "OP", "SUI", "APT",
            "AVAX", "LINK", "INJ", "SEI", "TIA", "JUP", "WIF",
            "PEPE", "ONDO", "RENDER", "FET", "NEAR", "AAVE",
            "MKR", "PENDLE", "STX", "WLD",
        ]

        seen_addrs = set()
        for coin in whale_coins:
            try:
                recent = hl.get_recent_trades(coin)
                if not recent:
                    continue

                # Find addresses with large trades ($25k+ to catch mid-size whales too)
                large_trade_addresses = set()
                for trade in (recent if isinstance(recent, list) else []):
                    size_usd = float(trade.get("px", 0)) * float(trade.get("sz", 0))
                    if size_usd > 25_000:
                        for user_key in ["user", "users"]:
                            if user_key in trade:
                                addrs = trade[user_key] if isinstance(trade[user_key], list) else [trade[user_key]]
                                large_trade_addresses.update(addrs)

                # Check up to 10 whale addresses per coin (was 5)
                for addr in list(large_trade_addresses)[:10]:
                    if addr in seen_addrs or addr in self.known_traders:
                        continue
                    seen_addrs.add(addr)
                    try:
                        state = hl.get_user_state(addr)
                        if state and state["account_value"] > 50_000:  # Lower threshold: $50k (was $100k)
                            traders.append({
                                "address": addr,
                                "total_pnl": 0,
                                "roi_pct": 0,
                                "account_value": state["account_value"],
                                "source": "whale_detection",
                                "metadata": {"detected_on": coin},
                            })
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Error in whale detection for {coin}: {e}")
            time.sleep(0.3)

        logger.info(f"Whale detection: found {len(traders)} whale traders across {len(whale_coins)} coins")
        return traders

    def _discover_from_open_interest(self) -> List[Dict]:
        """
        Discover active traders by scanning who holds the largest open positions.
        Uses metaAndAssetCtxs to find high-OI coins, then gets recent fills
        to identify who's trading them.
        """
        traders = []
        seen_addrs = set()

        try:
            from src.core.api_manager import get_manager, Priority
            # Get coins with highest open interest via API manager
            data = get_manager().post(
                payload={"type": "metaAndAssetCtxs"},
                priority=Priority.NORMAL, timeout=10,
            )
            if data is None:
                return []
            if not isinstance(data, list) or len(data) < 2:
                return []

            # data[0] = meta (asset names), data[1] = asset contexts (OI, funding, etc.)
            meta = data[0].get("universe", []) if isinstance(data[0], dict) else []
            ctxs = data[1] if isinstance(data[1], list) else []

            # Build sorted list of coins by open interest
            oi_ranked = []
            for i, ctx in enumerate(ctxs):
                if not isinstance(ctx, dict):
                    continue
                coin = meta[i]["name"] if i < len(meta) and isinstance(meta[i], dict) else f"UNKNOWN_{i}"
                oi = float(ctx.get("openInterest", 0))
                oi_ranked.append((coin, oi))

            oi_ranked.sort(key=lambda x: x[1], reverse=True)

            # Scan recent fills on top 15 OI coins for active traders
            for coin, oi in oi_ranked[:15]:
                try:
                    recent = hl.get_recent_trades(coin)
                    if not recent or not isinstance(recent, list):
                        continue

                    # Extract unique addresses from recent trades
                    addrs_with_size = {}
                    for trade in recent:
                        size_usd = float(trade.get("px", 0)) * float(trade.get("sz", 0))
                        for user_key in ["user", "users"]:
                            if user_key in trade:
                                raw = trade[user_key]
                                addr_list = raw if isinstance(raw, list) else [raw]
                                for addr in addr_list:
                                    if addr and addr not in self.known_traders and addr not in seen_addrs:
                                        addrs_with_size[addr] = addrs_with_size.get(addr, 0) + size_usd

                    # Take the top 5 by volume for this coin
                    top_addrs = sorted(addrs_with_size.items(), key=lambda x: x[1], reverse=True)[:5]
                    for addr, vol in top_addrs:
                        seen_addrs.add(addr)
                        traders.append({
                            "address": addr,
                            "total_pnl": 0,
                            "roi_pct": 0,
                            "source": "open_interest",
                            "metadata": {"detected_on": coin, "volume_usd": round(vol, 0)},
                        })
                except Exception:
                    pass
                time.sleep(0.3)

        except Exception as e:
            logger.debug(f"Error in OI discovery: {e}")

        logger.info(f"OI discovery: found {len(traders)} traders from high-OI coins")
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
        one_week_ago = int((datetime.now(timezone.utc) - timedelta(days=7)).timestamp() * 1000)
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
            from src.discovery.adaptive_bot_detector import AdaptiveBotDetector
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
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
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
        profit_factor = min(gross_profit / gross_loss, 999.0) if gross_loss > 0 else 999.0

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

    # ─── Bot detection helpers ────────────────────────────────────────────────

    @staticmethod
    def _detect_arb_pattern(fills: List[Dict]) -> bool:
        """
        Detect cross-exchange / funding arbitrage pattern.

        Signal: same coin, opposite sides, <5 s apart, tiny closed PnL each leg.
        This pattern is structurally impossible for a human trader to execute
        manually at scale.
        """
        if len(fills) < 10:
            return False

        by_coin: Dict[str, List[Dict]] = {}
        for f in fills:
            by_coin.setdefault(f.get("coin", ""), []).append(f)

        for coin, coin_fills in by_coin.items():
            if len(coin_fills) < 10:
                continue
            for i in range(len(coin_fills) - 1):
                f1 = coin_fills[i]
                f2 = coin_fills[i + 1]
                time_gap_ms = f2.get("time", 0) - f1.get("time", 0)
                if time_gap_ms < 5_000:  # within 5 seconds
                    if f1.get("side") != f2.get("side"):
                        # Tiny PnL on a round-trip = arb
                        if abs(f2.get("closed_pnl", 0)) < 10.0:
                            return True
        return False

    def _apply_hard_cutoffs(
        self,
        fills: List[Dict],
        trades_per_day: float,
    ) -> Optional[int]:
        """
        Apply hard-cutoff rules that immediately classify an account as a bot.
        Returns 10 (bot score) if any cutoff triggers, else None (keep going).
        """
        if trades_per_day > config.BOT_HARD_CUTOFF_TRADES:
            logger.info(
                "Bot INSTANT: %.0f trades/day (hard cutoff >%s)",
                trades_per_day, config.BOT_HARD_CUTOFF_TRADES,
            )
            return 10

        if trades_per_day > 50:
            closed_pnls = sorted([f["closed_pnl"] for f in fills if f["closed_pnl"] != 0])
            if closed_pnls:
                median_pnl = closed_pnls[len(closed_pnls) // 2]
                if abs(median_pnl) < 0.50:
                    logger.info(
                        "Bot INSTANT: spread/MM bot (median PnL=$%.2f, %.0f trades/day)",
                        median_pnl, trades_per_day,
                    )
                    return 10

        return None

    def _score_bot_signals(
        self,
        fills: List[Dict],
        positions: List[Dict],
        trade_analysis: Dict,
        trades_per_day: float,
    ) -> int:
        """
        Signal-based bot scoring for borderline accounts.
        Returns cumulative bot score (0 = clean human, ≥3 = bot).
        """
        total = len(fills)
        avg_size = trade_analysis.get("avg_trade_size", 0)
        pnl = trade_analysis.get("total_closed_pnl", 0)
        liquidations = trade_analysis.get("liquidations", 0)
        active_positions = [p for p in positions if p["size"] > 0]
        bot_signals = 0

        # Signal 1: Elevated frequency (50–100 trades/day)
        if trades_per_day > config.BOT_ELEVATED_FREQ:
            logger.info(
                "Bot signal: %.0f trades/day (elevated >%s)",
                trades_per_day, config.BOT_ELEVATED_FREQ,
            )
            bot_signals += 3

        # Signal 2: Micro trade size at moderate frequency = market maker
        if trades_per_day > 30 and avg_size < 50:
            logger.info(
                "Bot signal: tiny trades ($%.0f avg) at %.0f/day", avg_size, trades_per_day,
            )
            bot_signals += 3

        # Signal 3: Near-zero PnL per trade with high volume = arb bot
        if total > 50:
            pnl_per_trade = abs(pnl) / total
            if pnl_per_trade < 0.5 and avg_size > 1_000:
                logger.info(
                    "Bot signal: arb pattern ($%.2f/trade, $%.0f avg)", pnl_per_trade, avg_size,
                )
                bot_signals += 3

        # Signal 4: Too many simultaneous positions = portfolio bot
        if len(active_positions) > 15:
            logger.info("Bot signal: %d simultaneous positions", len(active_positions))
            bot_signals += 2

        # Signal 5: Uniform trade sizes (low coefficient of variation = robotic)
        if len(fills) > 20:
            sizes = [f["size"] * f["price"] for f in fills[:50]]
            if sizes:
                try:
                    import numpy as np
                    mean_size = np.mean(sizes)
                    std_size = np.std(sizes)
                    cv = std_size / mean_size if mean_size > 0 else 0
                    if cv < 0.05:
                        logger.info("Bot signal: uniform trade sizes (CV=%.3f)", cv)
                        bot_signals += 2
                except ImportError:
                    pass  # numpy optional for this signal

        # Signal 6: High liquidation rate = reckless / poorly coded algo
        if total > 10 and liquidations / total > 0.15:
            logger.info("Bot signal: high liquidation rate (%d/%d)", liquidations, total)
            bot_signals += 3

        # Signal 7: Near-zero median PnL at moderate frequency = spread/funding bot
        if trades_per_day > 20:
            closed_pnls = sorted([f["closed_pnl"] for f in fills if f["closed_pnl"] != 0])
            if closed_pnls:
                median_pnl = closed_pnls[len(closed_pnls) // 2]
                if abs(median_pnl) < 1.0:
                    logger.info(
                        "Bot signal: micro PnL (median=$%.2f, %.0f trades/day)",
                        median_pnl, trades_per_day,
                    )
                    bot_signals += 3

        # Signal 8: Cross-exchange / funding arb pattern
        if self._detect_arb_pattern(fills):
            logger.info("Bot signal: cross-exchange arb pattern detected")
            bot_signals += 3

        return bot_signals

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

        Delegates to:
          _apply_hard_cutoffs()  — instant bot classification
          _score_bot_signals()   — signal accumulation for borderline cases
          _detect_arb_pattern()  — cross-exchange arb detection
        """
        if not fills:
            return 0  # No data = can't determine, assume human

        trades_per_day = self._compute_trades_per_day(fills)

        # Fast path: hard cutoffs
        hard_score = self._apply_hard_cutoffs(fills, trades_per_day)
        if hard_score is not None:
            return hard_score

        # Signal accumulation
        bot_signals = self._score_bot_signals(fills, positions, trade_analysis, trades_per_day)

        addr_short = (fills[0].get("user", "unknown") or "unknown")[:10] if fills else "unknown"
        if bot_signals >= 3:
            logger.info(
                "Bot REJECTED (%d signals, %.0f trades/day): %s...",
                bot_signals, trades_per_day, addr_short,
            )
        elif bot_signals > 0:
            logger.info(
                "Borderline (%d signals, %.0f trades/day): %s... — passed but flagged",
                bot_signals, trades_per_day, addr_short,
            )
        else:
            logger.info("Human-like (clean, %.0f trades/day)", trades_per_day)

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
                    except Exception:
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
                    except Exception:
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
                              "detected_at": datetime.now(timezone.utc).isoformat()},
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
