"""
Crypto.com Exchange public REST API client for the trading bot.
Provides market data, orderbook, trades, and candlestick information.
No API key required — uses only public endpoints.

Supports spot markets (BTC_USD, ETH_USD, etc.) and perpetual futures
(BTC_USDPERP, ETH_USDPERP, etc.).
"""
import logging
import os
import time
import requests
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# Symbol mappings: standard coin names to crypto.com instrument names
CRYPTOCOM_SPOT_MAP = {
    "BTC": "BTC_USD",
    "ETH": "ETH_USD",
    "SOL": "SOL_USD",
    "DOGE": "DOGE_USD",
    "AVAX": "AVAX_USD",
    "LINK": "LINK_USD",
    "ARB": "ARB_USD",
    "OP": "OP_USD",
    "SUI": "SUI_USD",
    "APT": "APT_USD",
    "INJ": "INJ_USD",
    "NEAR": "NEAR_USD",
    "XRP": "XRP_USD",
    "SEI": "SEI_USD",
    "PEPE": "PEPE_USD",
    "FET": "FET_USD",
    "ONDO": "ONDO_USD",
    "TIA": "TIA_USD",
    "CRO": "CRO_USD",
    "ATOM": "ATOM_USD",
}

CRYPTOCOM_PERP_MAP = {
    "BTC": "BTC_USDPERP",
    "ETH": "ETH_USDPERP",
    "SOL": "SOL_USDPERP",
    "DOGE": "DOGE_USDPERP",
    "AVAX": "AVAX_USDPERP",
    "LINK": "LINK_USDPERP",
    "ARB": "ARB_USDPERP",
    "OP": "OP_USDPERP",
    "SUI": "SUI_USDPERP",
    "APT": "APT_USDPERP",
    "INJ": "INJ_USDPERP",
    "NEAR": "NEAR_USDPERP",
    "XRP": "XRP_USDPERP",
    "SEI": "SEI_USDPERP",
    "FET": "FET_USDPERP",
    "ONDO": "ONDO_USDPERP",
    "TIA": "TIA_USDPERP",
}


class CryptoComClient:
    """
    Client for crypto.com Exchange public REST API.
    Provides cached access to market data with configurable rate limiting.
    """

    BASE_URL = "https://api.crypto.com/exchange/v1/public"

    def __init__(self, cache_ttl: float = 30):
        """
        Initialize the crypto.com client.

        Args:
            cache_ttl: Cache TTL in seconds (default 30s)
        """
        self.session = requests.Session()
        self._cache: Dict[str, tuple] = {}  # key -> (data, timestamp)
        self._cache_ttl = cache_ttl
        self._last_request_time = 0
        self._request_interval = float(
            os.environ.get("CRYPTOCOM_MIN_REQUEST_INTERVAL_S", "0.2")
        )
        self._max_retries = int(os.environ.get("CRYPTOCOM_MAX_RETRIES", "3"))

    def _throttle(self):
        """Apply rate limiting between requests (150ms)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._request_interval:
            time.sleep(self._request_interval - elapsed)
        self._last_request_time = time.time()

    def _get_from_cache(self, cache_key: str) -> Optional[dict]:
        """Check cache and return data if valid."""
        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                logger.debug(f"Cache hit for {cache_key}")
                return data
            else:
                del self._cache[cache_key]  # Expire old entry
        return None

    def _set_cache(self, cache_key: str, data: dict):
        """Store data in cache with timestamp."""
        self._cache[cache_key] = (data, time.time())

    def _get(self, endpoint: str, params: dict = None, cache_ttl: Optional[float] = None) -> Optional[dict]:
        """
        Cached GET request to crypto.com public API.

        Args:
            endpoint: API endpoint (relative to BASE_URL)
            params: Query parameters
            cache_ttl: Override cache TTL for this request

        Returns:
            Parsed JSON response or None on error
        """
        # Build cache key
        cache_key = f"{endpoint}:{str(params) if params else ''}"

        # Check cache
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        # Make request
        url = f"{self.BASE_URL}/{endpoint}"
        for attempt in range(self._max_retries):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    raw = resp.json()
                    # Crypto.com v1 wraps responses in {"id": -1, "method": "...", "code": 0, "result": {...}}
                    # Unwrap the result if present
                    if isinstance(raw, dict) and "result" in raw:
                        data = raw["result"]
                    else:
                        data = raw
                    self._set_cache(cache_key, data)
                    return data

                if resp.status_code == 429 or resp.status_code >= 500:
                    backoff = min(8.0, 0.5 * (2 ** attempt))
                    logger.warning(
                        "crypto.com API %s returned %d (attempt %d/%d); retrying in %.1fs",
                        endpoint,
                        resp.status_code,
                        attempt + 1,
                        self._max_retries,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue

                logger.warning(
                    "crypto.com API error %d for %s: %s",
                    resp.status_code,
                    endpoint,
                    resp.text[:200],
                )
                return None
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                backoff = min(8.0, 0.5 * (2 ** attempt))
                logger.warning(
                    "crypto.com network error for %s (attempt %d/%d): %s; retrying in %.1fs",
                    endpoint,
                    attempt + 1,
                    self._max_retries,
                    e,
                    backoff,
                )
                time.sleep(backoff)
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error for {endpoint}: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error calling {endpoint}: {e}")
                return None
        return None

    def get_ticker(self, coin: str, is_perp: bool = False) -> Optional[Dict]:
        """
        Get normalized ticker data for exchange aggregator compatibility.

        Args:
            coin: Standard coin name (e.g., "BTC", "ETH")
            is_perp: If True, use perpetual symbol; else use spot

        Returns:
            Dict with normalized fields: exchange, price, volume_24h_usd,
            price_change_pct, high_24h, low_24h, open_interest (if perp)
        """
        # Map coin to instrument name
        symbol_map = CRYPTOCOM_PERP_MAP if is_perp else CRYPTOCOM_SPOT_MAP
        instrument_name = symbol_map.get(coin)
        if not instrument_name:
            logger.debug(f"No mapping for {coin} ({'perp' if is_perp else 'spot'})")
            return None

        # Fetch ticker (note: get-tickers is plural and returns array in "data")
        data = self._get("get-tickers", params={"instrument_name": instrument_name})
        if not data:
            return None

        try:
            # get-tickers returns {"data": [{...}]} or already unwrapped
            if isinstance(data, dict) and "data" in data:
                items = data["data"]
                if isinstance(items, list) and len(items) > 0:
                    ticker = items[0]
                else:
                    return None
            elif isinstance(data, list) and len(data) > 0:
                # Already an array
                ticker = data[0]
            else:
                return None

            # Parse fields. Handle both abbreviated (v1 API) and full field names
            # Abbreviated: "a"=last, "h"=high, "l"=low, "v"=volume, "vv"=volume_value,
            #             "c"=change, "b"=best_bid, "k"=best_ask, "oi"=open_interest
            # Full names: "last", "high", "low", "volume", "volume_value", "change", etc.
            price = float(ticker.get("a") or ticker.get("last", 0))
            change_pct = float(ticker.get("c") or ticker.get("change", 0))
            volume_usd = float(ticker.get("vv") or ticker.get("volume_value", 0))
            volume = float(ticker.get("v") or ticker.get("volume", 0))

            result = {
                "exchange": "crypto.com",
                "coin": coin,
                "instrument_name": instrument_name,
                "price": price,
                "volume_24h_usd": volume_usd if volume_usd > 0 else volume * price,
                "price_change_pct": change_pct,
                "high_24h": float(ticker.get("h") or ticker.get("high", 0)),
                "low_24h": float(ticker.get("l") or ticker.get("low", 0)),
                "open": float(ticker.get("o") or ticker.get("open", 0)),
                "timestamp": ticker.get("ut") or ticker.get("timestamp", 0),
            }

            # Add perp-specific fields
            oi = float(ticker.get("oi") or ticker.get("open_interest", 0))
            if is_perp and oi > 0:
                result["open_interest"] = oi

            return result

        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing ticker for {coin}: {e}")
            return None

    def get_orderbook(self, coin: str, depth: int = 10, is_perp: bool = False) -> Optional[Dict]:
        """
        Get order book with bid/ask imbalance calculation.

        Args:
            coin: Standard coin name (e.g., "BTC", "ETH")
            depth: Number of levels to fetch (default 10)
            is_perp: If True, use perpetual symbol; else use spot

        Returns:
            Dict with: bids (list), asks (list), spread (float),
            imbalance (float -1 to +1, positive = more bids = bullish)
        """
        # Map coin to instrument name
        symbol_map = CRYPTOCOM_PERP_MAP if is_perp else CRYPTOCOM_SPOT_MAP
        instrument_name = symbol_map.get(coin)
        if not instrument_name:
            logger.debug(f"No mapping for {coin} ({'perp' if is_perp else 'spot'})")
            return None

        # Fetch orderbook
        data = self._get("get-book", params={
            "instrument_name": instrument_name,
            "depth": depth
        })
        if not data:
            return None

        try:
            # Handle both direct object and nested structure
            book_data = data
            if isinstance(data, dict) and "data" in data:
                book_data = data["data"]

            bids = book_data.get("bids", [])
            asks = book_data.get("asks", [])

            if not bids or not asks:
                return None

            # Calculate bid/ask volume
            bid_volume = sum(float(b.get("qty", 0)) for b in bids)
            ask_volume = sum(float(a.get("qty", 0)) for a in asks)
            total_volume = bid_volume + ask_volume

            # Calculate imbalance (-1 to +1, positive = bullish)
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0

            # Calculate spread
            best_bid = float(bids[0].get("price", 0)) if bids else 0
            best_ask = float(asks[0].get("price", 0)) if asks else 0
            spread = ((best_ask - best_bid) / best_bid * 100) if best_bid > 0 else 0

            return {
                "exchange": "crypto.com",
                "coin": coin,
                "instrument_name": instrument_name,
                "bids": [{"price": float(b.get("price", 0)), "qty": float(b.get("qty", 0))} for b in bids],
                "asks": [{"price": float(a.get("price", 0)), "qty": float(a.get("qty", 0))} for a in asks],
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread_pct": round(spread, 4),
                "imbalance": round(imbalance, 4),  # -1 (all asks) to +1 (all bids)
                "bid_volume": round(bid_volume, 4),
                "ask_volume": round(ask_volume, 4),
                "timestamp": book_data.get("timestamp", 0),
            }

        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing orderbook for {coin}: {e}")
            return None

    def get_recent_trades(self, coin: str, count: int = 50, is_perp: bool = False) -> Optional[List[Dict]]:
        """
        Get recent trades with notional value calculation.

        Args:
            coin: Standard coin name (e.g., "BTC", "ETH")
            count: Number of trades to fetch (default 50, max ~150)
            is_perp: If True, use perpetual symbol; else use spot

        Returns:
            List of dicts with: price, qty, side, timestamp, notional, exchange
        """
        # Map coin to instrument name
        symbol_map = CRYPTOCOM_PERP_MAP if is_perp else CRYPTOCOM_SPOT_MAP
        instrument_name = symbol_map.get(coin)
        if not instrument_name:
            logger.debug(f"No mapping for {coin} ({'perp' if is_perp else 'spot'})")
            return None

        # Fetch trades
        data = self._get("get-trades", params={
            "instrument_name": instrument_name,
            "count": min(count, 150)  # Max 150 from API
        })
        if not data:
            return None

        try:
            # Extract trades array (may be nested or direct)
            trades_list = data.get("data", data) if isinstance(data, dict) else data
            if not isinstance(trades_list, list):
                return None

            trades = []
            for trade in trades_list:
                price = float(trade.get("price", 0))
                qty = float(trade.get("qty", 0))
                notional = price * qty

                trades.append({
                    "coin": coin,
                    "exchange": "crypto.com",
                    "instrument_name": instrument_name,
                    "price": price,
                    "qty": qty,
                    "notional": notional,
                    "side": trade.get("side", "").lower(),  # "buy" or "sell"
                    "timestamp": trade.get("timestamp", 0),
                })

            return trades

        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing trades for {coin}: {e}")
            return None

    def get_candlesticks(self, coin: str, timeframe: str = "5m", is_perp: bool = False) -> Optional[List[Dict]]:
        """
        Get candlestick data for technical analysis.

        Args:
            coin: Standard coin name (e.g., "BTC", "ETH")
            timeframe: Candlestick interval (e.g., "1m", "5m", "15m", "1h", "4h", "1d")
            is_perp: If True, use perpetual symbol; else use spot

        Returns:
            List of dicts with: open, high, low, close, volume, volume_usd, timestamp
            (most recent first)
        """
        # Map coin to instrument name
        symbol_map = CRYPTOCOM_PERP_MAP if is_perp else CRYPTOCOM_SPOT_MAP
        instrument_name = symbol_map.get(coin)
        if not instrument_name:
            logger.debug(f"No mapping for {coin} ({'perp' if is_perp else 'spot'})")
            return None

        # Fetch candlesticks
        data = self._get("get-candlestick", params={
            "instrument_name": instrument_name,
            "timeframe": timeframe
        })
        if not data:
            return None

        try:
            # Extract candles array (may be nested or direct)
            candles_list = data.get("data", data) if isinstance(data, dict) else data
            if not isinstance(candles_list, list):
                return None

            candles = []
            for candle in candles_list:
                # Handle both abbreviated field names (v1 API) and full names
                # Abbreviated: "o"=open, "h"=high, "l"=low, "c"=close
                #             "v"=volume, "vv"=volume_usd, "ut"=update_time/timestamp
                candles.append({
                    "coin": coin,
                    "exchange": "crypto.com",
                    "instrument_name": instrument_name,
                    "timeframe": timeframe,
                    "open": float(candle.get("o") or candle.get("open", 0)),
                    "high": float(candle.get("h") or candle.get("high", 0)),
                    "low": float(candle.get("l") or candle.get("low", 0)),
                    "close": float(candle.get("c") or candle.get("close", 0)),
                    "volume": float(candle.get("v") or candle.get("volume", 0)),
                    "volume_usd": float(candle.get("vv") or candle.get("volume_usd", 0)),
                    "timestamp": candle.get("ut") or candle.get("timestamp", 0),
                })

            return candles

        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing candlesticks for {coin}: {e}")
            return None

    def get_whale_trades(self, coin: str, min_notional: float = 50000, count: int = 100,
                         is_perp: bool = False) -> List[Dict]:
        """
        Filter recent trades for whale activity (large single trades).

        Args:
            coin: Standard coin name (e.g., "BTC", "ETH")
            min_notional: Minimum trade notional value (USD) to flag as whale trade
            count: Number of recent trades to scan
            is_perp: If True, use perpetual symbol; else use spot

        Returns:
            List of large trades with notional >= min_notional, tagged with direction
        """
        trades = self.get_recent_trades(coin, count=count, is_perp=is_perp)
        if not trades:
            return []

        whales = []
        for trade in trades:
            notional = trade.get("notional", 0)
            if notional >= min_notional:
                whales.append({
                    "coin": coin,
                    "exchange": "crypto.com",
                    "instrument_name": trade.get("instrument_name"),
                    "price": trade.get("price"),
                    "qty": trade.get("qty"),
                    "notional": notional,
                    "side": trade.get("side"),  # "buy" or "sell"
                    "timestamp": trade.get("timestamp"),
                    "is_perp": is_perp,
                })

        return whales

    def get_orderbook_imbalance(self, coin: str, depth: int = 10,
                                is_perp: bool = False) -> Optional[float]:
        """
        Get bid/ask volume imbalance.

        Args:
            coin: Standard coin name (e.g., "BTC", "ETH")
            depth: Number of orderbook levels
            is_perp: If True, use perpetual symbol; else use spot

        Returns:
            Imbalance value from -1 (all asks, bearish) to +1 (all bids, bullish)
        """
        book = self.get_orderbook(coin, depth=depth, is_perp=is_perp)
        if book:
            return book.get("imbalance", 0)
        return None

    def get_5m_volatility(self, coin: str, is_perp: bool = False) -> Optional[float]:
        """
        Compute 5-minute realized volatility from recent candles.

        Uses last 12 candles (1 hour of 5m data) to compute log-return std.

        Args:
            coin: Standard coin name (e.g., "BTC", "ETH")
            is_perp: If True, use perpetual symbol; else use spot

        Returns:
            Volatility as float 0-1 (normalized), or None on error
        """
        import math

        candles = self.get_candlesticks(coin, timeframe="5m", is_perp=is_perp)
        if not candles or len(candles) < 2:
            return None

        try:
            # Reverse to chronological order (oldest first)
            candles = list(reversed(candles))

            # Compute log returns
            log_returns = []
            for i in range(1, len(candles)):
                close_prev = float(candles[i - 1].get("close", 0))
                close_curr = float(candles[i].get("close", 0))
                if close_prev > 0:
                    log_ret = math.log(close_curr / close_prev)
                    log_returns.append(log_ret)

            if not log_returns:
                return None

            # Compute standard deviation
            mean = sum(log_returns) / len(log_returns)
            variance = sum((x - mean) ** 2 for x in log_returns) / len(log_returns)
            std_dev = math.sqrt(variance)

            # Normalize to 0-1 range (5% std = 1.0)
            volatility = min(1.0, std_dev / 0.05)

            return round(volatility, 4)

        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.error(f"Error computing volatility for {coin}: {e}")
            return None

    def get_instruments(self) -> Optional[List[str]]:
        """
        Get list of all available trading instruments.

        Returns:
            List of instrument names (e.g., ["BTC_USD", "ETH_USD", ...])
        """
        data = self._get("get-instruments")
        if not data:
            return None
        # Handle both nested and direct array formats
        instruments = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(instruments, list):
            return instruments
        return None


# ─── Module-level convenience functions (for compatibility) ───────

_client = None


def _get_client() -> CryptoComClient:
    """Get or create the default client instance."""
    global _client
    if _client is None:
        _client = CryptoComClient()
    return _client


def get_ticker(coin: str, is_perp: bool = False) -> Optional[Dict]:
    """Convenience function to get ticker from default client."""
    return _get_client().get_ticker(coin, is_perp=is_perp)


def get_orderbook(coin: str, depth: int = 10, is_perp: bool = False) -> Optional[Dict]:
    """Convenience function to get orderbook from default client."""
    return _get_client().get_orderbook(coin, depth=depth, is_perp=is_perp)


def get_recent_trades(coin: str, count: int = 50, is_perp: bool = False) -> Optional[List[Dict]]:
    """Convenience function to get recent trades from default client."""
    return _get_client().get_recent_trades(coin, count=count, is_perp=is_perp)


def get_candlesticks(coin: str, timeframe: str = "5m", is_perp: bool = False) -> Optional[List[Dict]]:
    """Convenience function to get candlesticks from default client."""
    return _get_client().get_candlesticks(coin, timeframe=timeframe, is_perp=is_perp)


def get_whale_trades(coin: str, min_notional: float = 50000, count: int = 100,
                     is_perp: bool = False) -> List[Dict]:
    """Convenience function to get whale trades from default client."""
    return _get_client().get_whale_trades(coin, min_notional=min_notional, count=count, is_perp=is_perp)


def get_orderbook_imbalance(coin: str, depth: int = 10, is_perp: bool = False) -> Optional[float]:
    """Convenience function to get orderbook imbalance from default client."""
    return _get_client().get_orderbook_imbalance(coin, depth=depth, is_perp=is_perp)


def get_5m_volatility(coin: str, is_perp: bool = False) -> Optional[float]:
    """Convenience function to get 5m volatility from default client."""
    return _get_client().get_5m_volatility(coin, is_perp=is_perp)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing crypto.com API client...")

    client = CryptoComClient()

    # Test ticker
    ticker = client.get_ticker("BTC")
    if ticker:
        logger.info(f"BTC Price: ${ticker['price']:,.2f}")
        logger.info(f"24h Volume: ${ticker.get('volume_24h_usd', 0):,.0f}")
        logger.info(f"Price Change: {ticker.get('price_change_pct', 0):+.2f}%")

    # Test orderbook
    book = client.get_orderbook("BTC", depth=5)
    if book:
        logger.info(f"BTC Spread: {book.get('spread_pct', 0):.4f}%")
        logger.info(f"Order Book Imbalance: {book.get('imbalance', 0):+.4f}")

    # Test trades
    trades = client.get_recent_trades("BTC", count=10)
    if trades:
        logger.info(f"Recent trades: {len(trades)} trades fetched")
        for trade in trades[:3]:
            logger.info(f"  {trade['side'].upper()} {trade['qty']:.4f} @ ${trade['price']:,.2f} (notional: ${trade['notional']:,.2f})")

    # Test candlesticks
    candles = client.get_candlesticks("BTC", timeframe="5m")
    if candles:
        logger.info(f"Candlesticks: {len(candles)} candles fetched")
        latest = candles[0]
        logger.info(f"  Latest 5m: O {latest['open']:.2f} H {latest['high']:.2f} L {latest['low']:.2f} C {latest['close']:.2f}")

    # Test whale detection
    whales = client.get_whale_trades("BTC", min_notional=100000, count=100)
    if whales:
        logger.info(f"Whale trades (>$100k): {len(whales)} found")

    # Test volatility
    vol = client.get_5m_volatility("BTC")
    if vol is not None:
        logger.info(f"5m Volatility: {vol:.4f} (normalized 0-1)")

    logger.info("crypto.com API client test complete.")
