"""
Lighter DEX Exchange Adapter
=============================
Adapter for Lighter's perpetual futures API.

Lighter (lighter.xyz) is an on-chain order book DEX with zero trading fees.
Their API provides trade data, orderbook snapshots, and account info.

API base: https://api.lighter.xyz/v1 (mainnet)
Docs: https://apidocs.lighter.xyz

Key endpoints used:
  - GET /orderBooks              - List all markets
  - GET /orderBookDetails/{id}   - Market details (OI, funding, etc.)
  - GET /trades/{market_id}      - Recent trades for a market
  - GET /account/{address}       - Account positions & balances
  - GET /orders/{address}        - Open orders for an account
  - GET /fills/{address}         - Trade history for an account

Note: Lighter doesn't have a public leaderboard like Hyperliquid.
Trader discovery is done by scanning recent large trades and identifying
recurring profitable addresses.
"""
import time
import logging
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from .base_adapter import (
    BaseExchangeAdapter,
    NormalizedTrader,
    NormalizedFill,
    NormalizedPosition,
    NormalizedMarketData,
)

logger = logging.getLogger(__name__)

# Lighter API base URLs
LIGHTER_API_BASE = "https://api.lighter.xyz"
LIGHTER_API_V1 = f"{LIGHTER_API_BASE}/v1"

# Rate limiting
_MIN_REQUEST_INTERVAL = 0.25  # 250ms between requests


class LighterAdapter(BaseExchangeAdapter):
    """
    Adapter for Lighter DEX perpetual futures.

    Lighter doesn't have a leaderboard, so trader discovery works differently:
    1. Scan recent large trades across all markets
    2. Identify addresses that appear frequently with profitable patterns
    3. Score these addresses by aggregated PnL and consistency

    This makes Lighter better suited as a CONFIRMATION venue rather than
    primary discovery — we find traders on Hyperliquid first, then check
    if they're also active on Lighter for cross-venue signal validation.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(
            name="lighter",
            base_url=LIGHTER_API_V1,
            config=config or {},
        )
        self._last_request_time = 0.0
        self._market_cache: Dict = {}
        self._market_cache_time: float = 0
        self._CACHE_TTL = 300  # 5 min

        # Symbol mapping: Lighter market_id -> canonical coin symbol
        # Built dynamically from orderBooks endpoint
        self._symbol_map: Dict[str, str] = {}
        self._reverse_symbol_map: Dict[str, str] = {}  # coin -> market_id

    def _rate_limit(self):
        """Enforce rate limit between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _get(self, path: str, params: Optional[Dict] = None, retries: int = 3) -> Optional[Dict]:
        """GET request to Lighter API with retries."""
        self._rate_limit()
        self._request_count += 1
        url = f"{self.base_url}{path}"

        for attempt in range(retries):
            try:
                resp = requests.get(url, params=params, timeout=20)
                if resp.status_code == 429:
                    wait = min(2 ** (attempt + 2), 30)
                    logger.warning(f"Lighter rate limited, backing off {wait}s")
                    time.sleep(wait)
                    continue
                if resp.status_code == 404:
                    logger.debug(f"Lighter 404: {path}")
                    return None
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Lighter API failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)

        self._error_count += 1
        return None

    def _ensure_markets_loaded(self):
        """Load market metadata if not cached."""
        now = time.time()
        if self._market_cache and (now - self._market_cache_time) < self._CACHE_TTL:
            return

        data = self._get("/orderBooks")
        if not data:
            logger.warning("Lighter: could not load markets")
            return

        markets = data if isinstance(data, list) else data.get("orderBooks", data.get("data", []))
        self._market_cache = {}
        self._symbol_map = {}
        self._reverse_symbol_map = {}

        for m in markets:
            market_id = str(m.get("id", m.get("market_id", m.get("orderBookId", ""))))
            # Try various field names Lighter uses
            symbol = (
                m.get("symbol", "") or
                m.get("baseToken", "") or
                m.get("name", "")
            )
            # Clean symbol to canonical form
            coin = self.normalize_coin_symbol(symbol)

            if market_id and coin:
                self._market_cache[market_id] = m
                self._symbol_map[market_id] = coin
                self._reverse_symbol_map[coin] = market_id

        self._market_cache_time = now
        logger.info(f"Lighter: loaded {len(self._market_cache)} markets: "
                    f"{list(self._symbol_map.values())[:10]}")

    # ─── Required: Trader Discovery ────────────────────────

    def get_top_traders(self, limit: int = 100) -> List[NormalizedTrader]:
        """
        Lighter doesn't have a leaderboard. Instead, we scan recent trades
        across all markets and identify high-volume, profitable addresses.

        This is a volume-based discovery method:
        1. Pull recent trades from each active market
        2. Aggregate by trader address
        3. Rank by total volume and trade frequency
        4. Return the most active addresses for deeper analysis
        """
        self._ensure_markets_loaded()

        if not self._market_cache:
            logger.warning("Lighter: no markets available for trader discovery")
            return []

        # Aggregate trades by address
        address_stats: Dict[str, Dict] = {}

        for market_id, coin in self._symbol_map.items():
            trades = self._get(f"/trades/{market_id}", params={"limit": 500})
            if not trades:
                continue

            trade_list = trades if isinstance(trades, list) else trades.get("trades", trades.get("data", []))

            for t in trade_list:
                # Lighter trades have both maker and taker addresses
                for addr_field in ["takerAddress", "taker", "makerAddress", "maker"]:
                    addr = t.get(addr_field, "")
                    if not addr or addr == "0x0000000000000000000000000000000000000000":
                        continue

                    if addr not in address_stats:
                        address_stats[addr] = {
                            "address": addr,
                            "total_volume": 0.0,
                            "trade_count": 0,
                            "coins_traded": set(),
                            "pnl_estimate": 0.0,
                        }

                    usd_amount = float(t.get("usd_amount", 0) or t.get("notional", 0) or 0)
                    size = float(t.get("size", 0) or t.get("amount", 0) or 0)
                    price = float(t.get("price", 0) or 0)

                    if not usd_amount and size and price:
                        usd_amount = size * price

                    address_stats[addr]["total_volume"] += usd_amount
                    address_stats[addr]["trade_count"] += 1
                    address_stats[addr]["coins_traded"].add(coin)

        # Rank by volume and convert to NormalizedTrader
        ranked = sorted(
            address_stats.values(),
            key=lambda x: x["total_volume"],
            reverse=True
        )[:limit]

        traders = []
        for stats in ranked:
            # Skip zero-volume or single-trade addresses
            if stats["trade_count"] < 3 or stats["total_volume"] < 1000:
                continue

            traders.append(NormalizedTrader(
                address=stats["address"],
                exchange="lighter",
                pnl_total=0.0,  # Can't determine from trade scan alone
                trade_count_30d=stats["trade_count"],
                avg_position_size=stats["total_volume"] / max(stats["trade_count"], 1),
                raw_data={
                    "total_volume": stats["total_volume"],
                    "coins_traded": list(stats["coins_traded"]),
                    "discovery_method": "volume_scan",
                },
            ))

        logger.info(f"Lighter: discovered {len(traders)} traders via volume scan "
                    f"(from {len(address_stats)} unique addresses)")
        return traders

    # ─── Required: Fills ───────────────────────────────────

    def get_trader_fills(self, address: str, limit: int = 200) -> List[NormalizedFill]:
        """Fetch trade history for a specific address."""
        data = self._get(f"/fills/{address}", params={"limit": limit})
        if not data:
            return []

        fill_list = data if isinstance(data, list) else data.get("fills", data.get("data", []))
        fills = []

        for f in fill_list[:limit]:
            market_id = str(f.get("market_id", f.get("marketId", f.get("orderBookId", ""))))
            coin = self._symbol_map.get(market_id, "UNKNOWN")

            size = float(f.get("size", 0) or f.get("base_amount", 0) or f.get("amount", 0) or 0)
            price = float(f.get("price", 0) or 0)
            usd_amount = float(f.get("usd_amount", 0) or 0)
            if not usd_amount and size and price:
                usd_amount = size * price

            # Determine side
            side_raw = str(f.get("side", f.get("type", ""))).lower()
            side = "buy" if "buy" in side_raw or "bid" in side_raw or side_raw == "0" else "sell"

            fills.append(NormalizedFill(
                exchange="lighter",
                trader_address=address,
                coin=coin,
                side=side,
                size=size,
                price=price,
                notional_usd=usd_amount,
                fee=float(f.get("fee", 0) or 0),
                realized_pnl=float(f.get("realized_pnl", 0) or f.get("pnl", 0) or 0),
                timestamp=str(f.get("timestamp", f.get("time", f.get("created_at", "")))),
                trade_id=str(f.get("trade_id", f.get("id", ""))),
                is_liquidation="liquidat" in str(f.get("type", "")).lower(),
            ))

        return fills

    # ─── Required: Positions ───────────────────────────────

    def get_trader_positions(self, address: str) -> List[NormalizedPosition]:
        """Fetch current open positions for an address."""
        data = self._get(f"/account/{address}")
        if not data:
            return []

        # Account response typically has positions array
        pos_list = data.get("positions", data.get("openPositions", []))
        if isinstance(data, list):
            pos_list = data

        positions = []
        for p in pos_list:
            market_id = str(p.get("market_id", p.get("marketId", p.get("orderBookId", ""))))
            coin = self._symbol_map.get(market_id, "UNKNOWN")

            size = float(p.get("size", 0) or p.get("position_size", 0) or 0)
            if size == 0:
                continue

            side = "long" if size > 0 else "short"
            entry = float(p.get("entry_price", 0) or p.get("avgEntryPrice", 0) or 0)
            mark = float(p.get("mark_price", 0) or p.get("markPrice", 0) or 0)

            positions.append(NormalizedPosition(
                exchange="lighter",
                trader_address=address,
                coin=coin,
                side=side,
                size=abs(size),
                entry_price=entry,
                mark_price=mark,
                unrealized_pnl=float(p.get("unrealized_pnl", 0) or p.get("unrealizedPnl", 0) or 0),
                leverage=float(p.get("leverage", 1) or 1),
                margin_used=float(p.get("margin", 0) or p.get("margin_used", 0) or 0),
                liquidation_price=float(p.get("liquidation_price", 0) or p.get("liqPrice", 0) or 0),
            ))

        return positions

    # ─── Required: Market Data ─────────────────────────────

    def get_market_data(self, coins: Optional[List[str]] = None) -> List[NormalizedMarketData]:
        """Fetch market data from orderBookDetails for each market."""
        self._ensure_markets_loaded()

        results = []
        for market_id, coin in self._symbol_map.items():
            if coins and coin not in coins:
                continue

            details = self._get(f"/orderBookDetails/{market_id}")
            if not details:
                continue

            # Parse orderbook details
            mid = float(details.get("midPrice", 0) or details.get("mid_price", 0) or 0)
            funding = float(details.get("fundingRate", 0) or details.get("funding_rate", 0) or 0)
            oi = float(details.get("openInterest", 0) or details.get("open_interest", 0) or 0)
            vol = float(details.get("volume24h", 0) or details.get("volume_24h", 0) or 0)
            best_bid = float(details.get("bestBid", 0) or details.get("best_bid", 0) or 0)
            best_ask = float(details.get("bestAsk", 0) or details.get("best_ask", 0) or 0)

            # Calculate spread in basis points
            spread_bps = 0.0
            if best_bid > 0 and best_ask > 0:
                spread_bps = ((best_ask - best_bid) / mid * 10000) if mid > 0 else 0

            results.append(NormalizedMarketData(
                exchange="lighter",
                coin=coin,
                mid_price=mid,
                funding_rate=funding,
                open_interest=oi,
                volume_24h=vol,
                mark_price=float(details.get("markPrice", 0) or details.get("mark_price", mid) or mid),
                index_price=float(details.get("indexPrice", 0) or details.get("index_price", 0) or 0),
                best_bid=best_bid,
                best_ask=best_ask,
                spread_bps=spread_bps,
            ))

        return results

    # ─── Required: Markets ─────────────────────────────────

    def get_available_markets(self) -> List[str]:
        """Return list of available trading symbols on Lighter."""
        self._ensure_markets_loaded()
        return list(self._reverse_symbol_map.keys())

    # ─── Required: Symbol Normalization ────────────────────

    def normalize_coin_symbol(self, raw_symbol: str) -> str:
        """
        Convert Lighter symbol to canonical form.
        Examples:
          "BTC-PERP" -> "BTC"
          "ETH-USD"  -> "ETH"
          "BTCUSDT"  -> "BTC"
          "SOL"      -> "SOL"
        """
        s = raw_symbol.strip().upper()
        # Remove common suffixes
        for suffix in ["-PERP", "-USD", "-USDT", "-USDC", "USDT", "USDC", "USD", "/USD", "/USDT"]:
            if s.endswith(suffix):
                s = s[: -len(suffix)]
                break
        return s

    # ─── Optional: Funding Rates ───────────────────────────

    def get_funding_rates(self) -> Dict[str, float]:
        """Get current funding rates from market details."""
        market_data = self.get_market_data()
        return {m.coin: m.funding_rate for m in market_data if m.funding_rate != 0}

    # ─── Lighter-Specific: Address Lookup ──────────────────

    def check_address_active(self, address: str) -> bool:
        """
        Quick check if an address has any activity on Lighter.
        Useful for cross-venue confirmation — check if a Hyperliquid
        trader also trades on Lighter.
        """
        data = self._get(f"/account/{address}")
        if not data:
            return False
        # Check for any positions or recent activity
        positions = data.get("positions", data.get("openPositions", []))
        return len(positions) > 0 if isinstance(positions, list) else bool(data)

    def get_address_summary(self, address: str) -> Optional[Dict]:
        """
        Get a quick summary of an address's activity on Lighter.
        Returns None if the address has no activity.
        """
        positions = self.get_trader_positions(address)
        fills = self.get_trader_fills(address, limit=50)

        if not positions and not fills:
            return None

        total_pnl = sum(f.realized_pnl for f in fills)
        coins_active = set()
        for p in positions:
            coins_active.add(p.coin)
        for f in fills:
            coins_active.add(f.coin)

        return {
            "address": address,
            "exchange": "lighter",
            "open_positions": len(positions),
            "recent_fills": len(fills),
            "realized_pnl_sample": total_pnl,
            "coins_active": list(coins_active),
            "has_activity": True,
        }
