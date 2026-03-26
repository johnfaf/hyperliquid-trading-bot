"""
Lighter DEX Exchange Adapter
=============================
Adapter for Lighter's perpetual futures API (zkLighter).

Lighter (lighter.xyz) is an on-chain order book DEX with zero trading fees.
Their API provides trade data, orderbook snapshots, and account info.

API base: https://mainnet.zklighter.elliot.ai/api/v1  (mainnet)
Docs: https://apidocs.lighter.xyz

Key endpoints used (all GET, under /api/v1):
  - /orderBooks                      - List all markets with metadata
  - /orderBookDetails?order_book_id= - Market details (fees, decimals)
  - /recentTrades?order_book_id=     - Recent trades for a market
  - /trades?account_index=           - Trades for a specific account
  - /account?by=l1_address&value=    - Account by ETH address
  - /accountsByL1Address?l1_address= - Account lookup by L1 address
  - /pnl?account_index=              - PnL data for an account

Note: Lighter doesn't have a public leaderboard like Hyperliquid.
Trader discovery is done by scanning recent large trades and identifying
recurring profitable addresses.

Note: Some endpoints (trades, pnl) use account_index (an integer) not
the raw ETH address. We must first resolve address → account_index
via the /account endpoint.
"""
import time
import logging
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum

from .base_adapter import (
    BaseExchangeAdapter,
    NormalizedTrader,
    NormalizedFill,
    NormalizedPosition,
    NormalizedMarketData,
)

logger = logging.getLogger(__name__)

# Lighter API base URLs — CORRECT as of 2026
LIGHTER_API_BASE = "https://mainnet.zklighter.elliot.ai"
LIGHTER_API_V1 = f"{LIGHTER_API_BASE}/api/v1"

# Rate limiting
_MIN_REQUEST_INTERVAL = 0.3  # 300ms between requests


class VenueState(Enum):
    """Three-state venue health tracking."""
    CONFIGURED = "configured"     # Code knows about it
    INITIALIZED = "initialized"   # Adapter created, no API contact yet
    HEALTHY = "healthy"           # API responding, markets loaded
    DEGRADED = "degraded"         # API partially working
    DOWN = "down"                 # API not responding


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

        # Symbol mapping: Lighter order_book_index -> canonical coin symbol
        self._symbol_map: Dict[str, str] = {}
        self._reverse_symbol_map: Dict[str, str] = {}  # coin -> order_book_index

        # Account index cache: ETH address -> Lighter account_index
        self._account_index_cache: Dict[str, Optional[int]] = {}

        # Venue state tracking
        self.state = VenueState.INITIALIZED
        self._last_health_check: float = 0
        self._health_check_interval = 300  # Re-check every 5 min

    def _rate_limit(self):
        """Enforce rate limit between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _get(self, path: str, params: Optional[Dict] = None,
             retries: int = 2, quiet: bool = False) -> Optional[Dict]:
        """
        GET request to Lighter API with retries.
        Returns None on failure. Does NOT retry on 4xx (client errors).
        """
        self._rate_limit()
        self._request_count += 1
        url = f"{self.base_url}{path}"

        for attempt in range(retries):
            try:
                resp = requests.get(url, params=params, timeout=15)

                if resp.status_code == 429:
                    wait = min(2 ** (attempt + 2), 30)
                    logger.warning(f"Lighter rate limited, backing off {wait}s")
                    time.sleep(wait)
                    continue

                # Don't retry client errors (400, 404, 422) — they won't change
                if 400 <= resp.status_code < 500:
                    if not quiet:
                        logger.debug(f"Lighter {resp.status_code}: {path} "
                                    f"params={params}")
                    self._error_count += 1
                    return None

                resp.raise_for_status()
                data = resp.json()

                # Update state on successful response
                if self.state in (VenueState.INITIALIZED, VenueState.DOWN):
                    self.state = VenueState.HEALTHY

                return data

            except requests.exceptions.ConnectionError as e:
                if not quiet:
                    logger.warning(f"Lighter connection failed: {e}")
                self.state = VenueState.DOWN
                self._error_count += 1
                return None
            except requests.exceptions.Timeout:
                if not quiet:
                    logger.warning(f"Lighter timeout: {path}")
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                self._error_count += 1
                return None
            except requests.exceptions.RequestException as e:
                if not quiet:
                    logger.warning(f"Lighter API failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)

        self._error_count += 1
        return None

    # ─── Health Check ──────────────────────────────────────

    def health_check(self) -> bool:
        """
        Check if Lighter API is reachable by fetching the orderBooks list.
        Fails loudly if market list cannot be loaded.
        Updates self.state accordingly.
        """
        now = time.time()
        self._last_health_check = now

        data = self._get("/orderBooks", quiet=True)
        if data is None:
            self.state = VenueState.DOWN
            logger.warning(f"Lighter health check FAILED — venue is DOWN "
                          f"(url={self.base_url}/orderBooks)")
            return False

        # Try to parse markets
        markets = data if isinstance(data, list) else data.get("order_books", data.get("orderBooks", data.get("data", [])))
        if not markets or len(markets) == 0:
            self.state = VenueState.DEGRADED
            logger.warning(f"Lighter health check: API responding but 0 markets found "
                          f"(response keys: {list(data.keys()) if isinstance(data, dict) else type(data).__name__})")
            return False

        self.state = VenueState.HEALTHY
        logger.info(f"Lighter health check PASSED: {len(markets)} markets, state={self.state.value}")
        return True

    def _ensure_markets_loaded(self):
        """Load market metadata if not cached."""
        now = time.time()
        if self._market_cache and (now - self._market_cache_time) < self._CACHE_TTL:
            return

        data = self._get("/orderBooks")
        if not data:
            logger.warning("Lighter: could not load markets — API may be down")
            self.state = VenueState.DOWN
            return

        # Parse the response — try multiple field name conventions
        markets = []
        if isinstance(data, list):
            markets = data
        elif isinstance(data, dict):
            markets = (
                data.get("order_books") or
                data.get("orderBooks") or
                data.get("data") or
                []
            )

        self._market_cache = {}
        self._symbol_map = {}
        self._reverse_symbol_map = {}

        # Debug: log sample market entry keys so we can adapt parsing
        if markets and len(markets) > 0:
            sample = markets[0]
            logger.debug(f"Lighter: sample market entry keys={list(sample.keys()) if isinstance(sample, dict) else type(sample).__name__}")

        for m in markets:
            if not isinstance(m, dict):
                continue

            # Lighter uses order_book_index as the market ID — try many field names
            market_id = str(
                m.get("order_book_index",
                       m.get("orderBookIndex",
                              m.get("id",
                                     m.get("order_book_id",
                                            m.get("orderBookId",
                                                   m.get("market_id",
                                                          m.get("marketId", "")))))))
            )

            # Symbol extraction — try various field names
            symbol = (
                m.get("symbol", "") or
                m.get("base_token", "") or
                m.get("baseToken", "") or
                m.get("base_symbol", "") or
                m.get("baseSymbol", "") or
                m.get("base_currency", "") or
                m.get("baseCurrency", "") or
                m.get("name", "") or
                m.get("ticker", "") or
                m.get("pair", "") or
                m.get("market", "")
            )

            # If symbol still empty, try to construct from quote/base fields
            if not symbol:
                base = m.get("base", m.get("baseAsset", ""))
                quote = m.get("quote", m.get("quoteAsset", ""))
                if base:
                    symbol = base

            # Clean symbol to canonical form
            coin = self.normalize_coin_symbol(symbol)

            # If market_id is empty, use index as fallback
            if not market_id and coin:
                market_id = str(markets.index(m))

            if market_id and coin and coin != "":
                self._market_cache[market_id] = m
                self._symbol_map[market_id] = coin
                self._reverse_symbol_map[coin] = market_id

        self._market_cache_time = now

        if self._market_cache:
            self.state = VenueState.HEALTHY
            logger.info(f"Lighter: loaded {len(self._market_cache)} markets: "
                       f"{list(self._symbol_map.values())[:10]}")
        else:
            self.state = VenueState.DEGRADED
            # Enhanced debug: log first market entry to diagnose field name mismatch
            sample_keys = []
            if markets and len(markets) > 0 and isinstance(markets[0], dict):
                sample_keys = list(markets[0].keys())
            logger.warning(f"Lighter: API responded with {len(markets)} markets but parsed 0. "
                          f"Response type={type(data).__name__}, "
                          f"keys={list(data.keys()) if isinstance(data, dict) else 'list'}, "
                          f"sample_market_keys={sample_keys}")

    def _resolve_account_index(self, address: str) -> Optional[int]:
        """
        Resolve an ETH address to a Lighter account_index.
        Some Lighter endpoints need account_index, not the raw address.
        """
        if address in self._account_index_cache:
            return self._account_index_cache[address]

        # Try /account endpoint with l1_address
        data = self._get("/account", params={"by": "l1_address", "value": address}, quiet=True)
        if data and isinstance(data, dict):
            idx = data.get("account_index", data.get("accountIndex"))
            if idx is not None:
                self._account_index_cache[address] = int(idx)
                return int(idx)

        # Try /accountsByL1Address
        data = self._get("/accountsByL1Address", params={"l1_address": address}, quiet=True)
        if data:
            accounts = data if isinstance(data, list) else data.get("accounts", [])
            if accounts and isinstance(accounts, list) and len(accounts) > 0:
                acct = accounts[0]
                idx = acct.get("account_index", acct.get("accountIndex"))
                if idx is not None:
                    self._account_index_cache[address] = int(idx)
                    return int(idx)

        self._account_index_cache[address] = None
        return None

    # ─── Required: Trader Discovery ────────────────────────

    def get_top_traders(self, limit: int = 100) -> List[NormalizedTrader]:
        """
        Lighter doesn't have a leaderboard. Instead, we scan recent trades
        across all markets and identify high-volume, profitable addresses.
        """
        if self.state == VenueState.DOWN:
            logger.debug("Lighter: skipping trader discovery — venue is DOWN")
            return []

        self._ensure_markets_loaded()

        if not self._market_cache:
            logger.warning("Lighter: no markets available for trader discovery")
            return []

        # Aggregate trades by address
        address_stats: Dict[str, Dict] = {}

        for market_id, coin in self._symbol_map.items():
            trades = self._get("/recentTrades",
                              params={"order_book_id": market_id, "limit": 200},
                              quiet=True)
            if not trades:
                continue

            trade_list = trades if isinstance(trades, list) else (
                trades.get("trades") or trades.get("recent_trades") or
                trades.get("data") or []
            )

            for t in trade_list:
                # Lighter trades may use account_index not raw address
                for addr_field in ["taker_account_index", "maker_account_index",
                                   "takerAddress", "taker", "makerAddress", "maker"]:
                    addr = str(t.get(addr_field, ""))
                    if not addr or addr == "0" or addr == "0x0000000000000000000000000000000000000000":
                        continue

                    if addr not in address_stats:
                        address_stats[addr] = {
                            "address": addr,
                            "total_volume": 0.0,
                            "trade_count": 0,
                            "coins_traded": set(),
                        }

                    # Try different field names for trade size/price
                    size = float(t.get("size", 0) or t.get("base_amount", 0) or
                                t.get("amount", 0) or 0)
                    price = float(t.get("price", 0) or 0)
                    usd_amount = float(t.get("usd_amount", 0) or t.get("notional", 0) or 0)

                    if not usd_amount and size and price:
                        usd_amount = size * price

                    address_stats[addr]["total_volume"] += usd_amount
                    address_stats[addr]["trade_count"] += 1
                    address_stats[addr]["coins_traded"].add(coin)

        # Rank by volume
        ranked = sorted(
            address_stats.values(),
            key=lambda x: x["total_volume"],
            reverse=True
        )[:limit]

        traders = []
        for stats in ranked:
            if stats["trade_count"] < 3 or stats["total_volume"] < 1000:
                continue

            traders.append(NormalizedTrader(
                address=stats["address"],
                exchange="lighter",
                pnl_total=0.0,
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
        if self.state == VenueState.DOWN:
            return []

        self._ensure_markets_loaded()

        # Lighter may need account_index for the trades endpoint
        account_idx = self._resolve_account_index(address)
        if account_idx is not None:
            data = self._get("/trades", params={"account_index": account_idx, "limit": limit})
        else:
            # Fallback: try with raw address
            data = self._get("/trades", params={"account": address, "limit": limit}, quiet=True)

        if not data:
            return []

        fill_list = data if isinstance(data, list) else (
            data.get("trades") or data.get("fills") or data.get("data") or []
        )
        fills = []

        for f in fill_list[:limit]:
            market_id = str(f.get("order_book_index",
                                   f.get("order_book_id",
                                          f.get("market_id",
                                                 f.get("orderBookId", "")))))
            coin = self._symbol_map.get(market_id, "UNKNOWN")

            size = float(f.get("size", 0) or f.get("base_amount", 0) or
                        f.get("amount", 0) or 0)
            price = float(f.get("price", 0) or 0)
            usd_amount = float(f.get("usd_amount", 0) or 0)
            if not usd_amount and size and price:
                usd_amount = size * price

            # Determine side
            side_raw = str(f.get("side", f.get("type", f.get("is_ask", "")))).lower()
            if side_raw in ("true", "1", "ask", "sell", "short"):
                side = "sell"
            else:
                side = "buy"

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
                trade_id=str(f.get("trade_id", f.get("id", f.get("trade_index", "")))),
                is_liquidation="liquidat" in str(f.get("type", "")).lower(),
            ))

        return fills

    # ─── Required: Positions ───────────────────────────────

    def get_trader_positions(self, address: str) -> List[NormalizedPosition]:
        """Fetch current open positions for an address."""
        if self.state == VenueState.DOWN:
            return []

        self._ensure_markets_loaded()

        # Try account lookup by L1 address
        data = self._get("/account", params={"by": "l1_address", "value": address}, quiet=True)
        if not data:
            return []

        # Account response typically has positions array
        pos_list = (
            data.get("positions") or
            data.get("open_positions") or
            data.get("openPositions") or
            []
        )

        positions = []
        for p in pos_list:
            market_id = str(p.get("order_book_index",
                                   p.get("order_book_id",
                                          p.get("market_id",
                                                 p.get("orderBookId", "")))))
            coin = self._symbol_map.get(market_id, "UNKNOWN")

            size = float(p.get("size", 0) or p.get("position_size", 0) or 0)
            if size == 0:
                continue

            side = "long" if size > 0 else "short"
            entry = float(p.get("entry_price", 0) or p.get("avg_entry_price", 0) or
                         p.get("avgEntryPrice", 0) or 0)
            mark = float(p.get("mark_price", 0) or p.get("markPrice", 0) or 0)

            positions.append(NormalizedPosition(
                exchange="lighter",
                trader_address=address,
                coin=coin,
                side=side,
                size=abs(size),
                entry_price=entry,
                mark_price=mark,
                unrealized_pnl=float(p.get("unrealized_pnl", 0) or
                                     p.get("unrealizedPnl", 0) or 0),
                leverage=float(p.get("leverage", 1) or 1),
                margin_used=float(p.get("margin", 0) or p.get("margin_used", 0) or 0),
                liquidation_price=float(p.get("liquidation_price", 0) or
                                       p.get("liqPrice", 0) or 0),
            ))

        return positions

    # ─── Required: Market Data ─────────────────────────────

    def get_market_data(self, coins: Optional[List[str]] = None) -> List[NormalizedMarketData]:
        """Fetch market data from orderBookDetails for each market."""
        if self.state == VenueState.DOWN:
            return []

        self._ensure_markets_loaded()

        results = []
        for market_id, coin in self._symbol_map.items():
            if coins and coin not in coins:
                continue

            details = self._get("/orderBookDetails",
                               params={"order_book_id": market_id},
                               quiet=True)
            if not details:
                continue

            # Parse orderbook details — try various field name conventions
            mid = float(details.get("mid_price", 0) or details.get("midPrice", 0) or 0)
            funding = float(details.get("funding_rate", 0) or
                           details.get("fundingRate", 0) or 0)
            oi = float(details.get("open_interest", 0) or
                      details.get("openInterest", 0) or 0)
            vol = float(details.get("volume_24h", 0) or
                       details.get("volume24h", 0) or 0)
            best_bid = float(details.get("best_bid", 0) or
                            details.get("bestBid", 0) or 0)
            best_ask = float(details.get("best_ask", 0) or
                            details.get("bestAsk", 0) or 0)

            # Calculate spread in basis points
            spread_bps = 0.0
            if best_bid > 0 and best_ask > 0 and mid > 0:
                spread_bps = (best_ask - best_bid) / mid * 10000

            results.append(NormalizedMarketData(
                exchange="lighter",
                coin=coin,
                mid_price=mid,
                funding_rate=funding,
                open_interest=oi,
                volume_24h=vol,
                mark_price=float(details.get("mark_price", 0) or
                                details.get("markPrice", mid) or mid),
                index_price=float(details.get("index_price", 0) or
                                 details.get("indexPrice", 0) or 0),
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
          "WBTC"     -> "BTC"
        """
        s = raw_symbol.strip().upper()
        # Remove common suffixes
        for suffix in ["-PERP", "-USD", "-USDT", "-USDC", "USDT", "USDC",
                       "USD", "/USD", "/USDT", "_PERP", "_USD"]:
            if s.endswith(suffix):
                s = s[: -len(suffix)]
                break
        # Common wrapped token mappings
        if s == "WBTC":
            s = "BTC"
        elif s == "WETH":
            s = "ETH"
        return s

    # ─── Optional: Funding Rates ───────────────────────────

    def get_funding_rates(self) -> Dict[str, float]:
        """Get current funding rates from market details."""
        market_data = self.get_market_data()
        return {m.coin: m.funding_rate for m in market_data if m.funding_rate != 0}

    # ─── Stats ─────────────────────────────────────────────

    def get_stats(self) -> Dict:
        base = super().get_stats()
        base["state"] = self.state.value
        base["markets_loaded"] = len(self._market_cache)
        base["symbols"] = list(self._symbol_map.values())[:15]
        base["accounts_cached"] = len(self._account_index_cache)
        return base
