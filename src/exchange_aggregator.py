"""
Multi-Exchange Data Aggregator
Pulls public market data from Binance, Bybit, Kraken, and Coinbase
to identify volume trends and directional bias across exchanges.
No API keys required — uses only public endpoints.
"""
import logging
import time
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Coin symbol mappings per exchange
SYMBOL_MAP = {
    "BTC": {"binance": "BTCUSDT", "bybit": "BTCUSDT", "kraken": "XBTUSD", "coinbase": "BTC-USD"},
    "ETH": {"binance": "ETHUSDT", "bybit": "ETHUSDT", "kraken": "ETHUSD", "coinbase": "ETH-USD"},
    "SOL": {"binance": "SOLUSDT", "bybit": "SOLUSDT", "kraken": "SOLUSD", "coinbase": "SOL-USD"},
    "DOGE": {"binance": "DOGEUSDT", "bybit": "DOGEUSDT", "kraken": "DOGEUSD", "coinbase": "DOGE-USD"},
    "AVAX": {"binance": "AVAXUSDT", "bybit": "AVAXUSDT", "kraken": "AVAXUSD", "coinbase": "AVAX-USD"},
    "LINK": {"binance": "LINKUSDT", "bybit": "LINKUSDT", "kraken": "LINKUSD", "coinbase": "LINK-USD"},
    "ARB": {"binance": "ARBUSDT", "bybit": "ARBUSDT", "kraken": None, "coinbase": "ARB-USD"},
    "OP": {"binance": "OPUSDT", "bybit": "OPUSDT", "kraken": None, "coinbase": "OP-USD"},
    "SUI": {"binance": "SUIUSDT", "bybit": "SUIUSDT", "kraken": None, "coinbase": "SUI-USD"},
    "APT": {"binance": "APTUSDT", "bybit": "APTUSDT", "kraken": None, "coinbase": "APT-USD"},
    "INJ": {"binance": "INJUSDT", "bybit": "INJUSDT", "kraken": None, "coinbase": "INJ-USD"},
    "NEAR": {"binance": "NEARUSDT", "bybit": "NEARUSDT", "kraken": None, "coinbase": "NEAR-USD"},
    "XRP": {"binance": "XRPUSDT", "bybit": "XRPUSDT", "kraken": "XRPUSD", "coinbase": "XRP-USD"},
    "HYPE": {"binance": None, "bybit": "HYPEUSDT", "kraken": None, "coinbase": None},
    "SEI": {"binance": "SEIUSDT", "bybit": "SEIUSDT", "kraken": None, "coinbase": "SEI-USD"},
    "PEPE": {"binance": "PEPEUSDT", "bybit": "PEPEUSDT", "kraken": None, "coinbase": "PEPE-USD"},
    "WIF": {"binance": "WIFUSDT", "bybit": "WIFUSDT", "kraken": None, "coinbase": None},
    "FET": {"binance": "FETUSDT", "bybit": "FETUSDT", "kraken": None, "coinbase": "FET-USD"},
    "ONDO": {"binance": "ONDOUSDT", "bybit": "ONDOUSDT", "kraken": None, "coinbase": "ONDO-USD"},
    "TIA": {"binance": "TIAUSDT", "bybit": "TIAUSDT", "kraken": None, "coinbase": "TIA-USD"},
}

# Rate limiting
_last_req = 0
_REQ_INTERVAL = 0.15  # 150ms between requests


def _throttle():
    global _last_req
    elapsed = time.time() - _last_req
    if elapsed < _REQ_INTERVAL:
        time.sleep(_REQ_INTERVAL - elapsed)
    _last_req = time.time()


def _safe_get(url: str, params: dict = None, timeout: int = 10) -> Optional[dict]:
    """Safe GET request with error handling."""
    _throttle()
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.debug(f"Request failed {url}: {e}")
    return None


# ─── Binance ──────────────────────────────────────────────────

def _binance_ticker(symbol: str) -> Optional[Dict]:
    """Get 24h ticker from Binance."""
    data = _safe_get("https://api.binance.com/api/v3/ticker/24hr", {"symbol": symbol})
    if data:
        return {
            "exchange": "binance",
            "price": float(data.get("lastPrice", 0)),
            "volume_24h_usd": float(data.get("quoteVolume", 0)),
            "price_change_pct": float(data.get("priceChangePercent", 0)),
            "high_24h": float(data.get("highPrice", 0)),
            "low_24h": float(data.get("lowPrice", 0)),
            "trades_24h": int(data.get("count", 0)),
            "taker_buy_volume": float(data.get("volume", 0)) * 0.5,  # estimate
        }
    return None


def _binance_futures_oi(symbol: str) -> Optional[Dict]:
    """Get open interest from Binance Futures."""
    data = _safe_get("https://fapi.binance.com/fapi/v1/openInterest", {"symbol": symbol})
    if data:
        return {
            "open_interest": float(data.get("openInterest", 0)),
            "open_interest_usd": float(data.get("openInterest", 0)) * float(data.get("price", 1)),
        }
    return None


def _binance_long_short_ratio(symbol: str) -> Optional[Dict]:
    """Get top trader long/short ratio from Binance Futures."""
    data = _safe_get(
        "https://fapi.binance.com/futures/data/topLongShortAccountRatio",
        {"symbol": symbol, "period": "1h", "limit": 1}
    )
    if data and len(data) > 0:
        return {
            "long_ratio": float(data[0].get("longAccount", 0.5)),
            "short_ratio": float(data[0].get("shortAccount", 0.5)),
            "long_short_ratio": float(data[0].get("longShortRatio", 1.0)),
        }
    return None


def _binance_funding_rate(symbol: str) -> Optional[float]:
    """Get current funding rate from Binance Futures."""
    data = _safe_get("https://fapi.binance.com/fapi/v1/premiumIndex", {"symbol": symbol})
    if data:
        return float(data.get("lastFundingRate", 0))
    return None


# ─── Bybit ────────────────────────────────────────────────────

def _bybit_ticker(symbol: str) -> Optional[Dict]:
    """Get 24h ticker from Bybit."""
    data = _safe_get("https://api.bybit.com/v5/market/tickers", {"category": "linear", "symbol": symbol})
    if data and data.get("result", {}).get("list"):
        t = data["result"]["list"][0]
        return {
            "exchange": "bybit",
            "price": float(t.get("lastPrice", 0)),
            "volume_24h_usd": float(t.get("turnover24h", 0)),
            "price_change_pct": float(t.get("price24hPcnt", 0)) * 100,
            "high_24h": float(t.get("highPrice24h", 0)),
            "low_24h": float(t.get("lowPrice24h", 0)),
            "open_interest": float(t.get("openInterest", 0)),
            "funding_rate": float(t.get("fundingRate", 0)),
        }
    return None


# ─── Kraken ───────────────────────────────────────────────────

def _kraken_ticker(pair: str) -> Optional[Dict]:
    """Get ticker from Kraken."""
    data = _safe_get("https://api.kraken.com/0/public/Ticker", {"pair": pair})
    if data and data.get("result") and len(data["result"]) > 0:
        key = list(data["result"].keys())[0]
        t = data["result"][key]
        price = float(t["c"][0])
        vol = float(t["v"][1])  # 24h volume
        return {
            "exchange": "kraken",
            "price": price,
            "volume_24h_usd": vol * price,
            "high_24h": float(t["h"][1]),
            "low_24h": float(t["l"][1]),
            "trades_24h": int(t["t"][1]),
        }
    return None


# ─── Coinbase ─────────────────────────────────────────────────

def _coinbase_ticker(product_id: str) -> Optional[Dict]:
    """Get ticker from Coinbase."""
    data = _safe_get(f"https://api.exchange.coinbase.com/products/{product_id}/stats")
    if data:
        price = float(data.get("last", 0))
        vol = float(data.get("volume", 0))
        return {
            "exchange": "coinbase",
            "price": price,
            "volume_24h_usd": vol * price,
            "high_24h": float(data.get("high", 0)),
            "low_24h": float(data.get("low", 0)),
            "open": float(data.get("open", 0)),
        }
    return None


# ─── Aggregation ──────────────────────────────────────────────

class ExchangeAggregator:
    """Aggregates market data across exchanges to find volume trends and directional bias."""

    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._cache_time: float = 0
        self._cache_ttl: float = 120  # 2-minute cache

    def get_multi_exchange_data(self, coin: str) -> Optional[Dict]:
        """
        Get aggregated data for a coin from all exchanges.
        Returns volume, price, directional bias, and confidence metrics.
        """
        symbols = SYMBOL_MAP.get(coin, {})
        if not symbols:
            return None

        results = {}

        # Binance
        if symbols.get("binance"):
            ticker = _binance_ticker(symbols["binance"])
            if ticker:
                results["binance"] = ticker
                # Add derivatives data
                ls = _binance_long_short_ratio(symbols["binance"])
                if ls:
                    results["binance"].update(ls)
                fr = _binance_funding_rate(symbols["binance"])
                if fr is not None:
                    results["binance"]["funding_rate"] = fr

        # Bybit
        if symbols.get("bybit"):
            ticker = _bybit_ticker(symbols["bybit"])
            if ticker:
                results["bybit"] = ticker

        # Kraken
        if symbols.get("kraken"):
            ticker = _kraken_ticker(symbols["kraken"])
            if ticker:
                results["kraken"] = ticker

        # Coinbase
        if symbols.get("coinbase"):
            ticker = _coinbase_ticker(symbols["coinbase"])
            if ticker:
                results["coinbase"] = ticker

        if not results:
            return None

        return self._compute_aggregate(coin, results)

    def _compute_aggregate(self, coin: str, exchange_data: Dict) -> Dict:
        """Compute aggregate metrics from multi-exchange data."""
        total_volume = 0
        weighted_price = 0
        price_changes = []
        funding_rates = []
        long_ratios = []
        prices = []
        exchanges_reporting = []

        for exchange, data in exchange_data.items():
            vol = data.get("volume_24h_usd", 0)
            price = data.get("price", 0)
            total_volume += vol
            if price > 0 and vol > 0:
                weighted_price += price * vol
                prices.append(price)
            if data.get("price_change_pct") is not None:
                price_changes.append(data["price_change_pct"])
            if data.get("funding_rate") is not None:
                funding_rates.append(data["funding_rate"])
            if data.get("long_ratio") is not None:
                long_ratios.append(data["long_ratio"])
            exchanges_reporting.append(exchange)

        vwap = weighted_price / total_volume if total_volume > 0 else (prices[0] if prices else 0)
        avg_change = sum(price_changes) / len(price_changes) if price_changes else 0
        avg_funding = sum(funding_rates) / len(funding_rates) if funding_rates else 0
        avg_long_ratio = sum(long_ratios) / len(long_ratios) if long_ratios else 0.5

        # Directional bias scoring (-1.0 to +1.0)
        # Positive = bullish, Negative = bearish
        bias_score = 0
        bias_signals = 0

        # Price momentum signal
        if avg_change != 0:
            momentum = max(-1, min(1, avg_change / 5))  # normalize: 5% = max signal
            bias_score += momentum * 0.3
            bias_signals += 1

        # Funding rate signal (positive funding = too many longs = slightly bearish contrarian)
        if avg_funding != 0:
            funding_signal = -max(-1, min(1, avg_funding / 0.001))  # 0.1% = max signal
            bias_score += funding_signal * 0.2
            bias_signals += 1

        # Long/short ratio signal
        if avg_long_ratio != 0.5:
            ls_signal = (avg_long_ratio - 0.5) * 2  # 0.5 = neutral, 0.7 = bullish
            bias_score += ls_signal * 0.3
            bias_signals += 1

        # Volume surge detection
        # (we'd need historical volume for this - use a simple heuristic)
        volume_score = min(1.0, total_volume / 1_000_000_000)  # normalize to $1B
        bias_score += (volume_score * 0.1 if avg_change > 0 else -volume_score * 0.1)
        bias_signals += 1

        # Normalize
        if bias_signals > 0:
            bias_score = bias_score / (0.3 + 0.2 + 0.3 + 0.1)  # normalize by max possible

        # Confidence based on data availability
        confidence = min(1.0, len(exchanges_reporting) / 3)

        # Determine bias label
        if bias_score > 0.15:
            bias = "bullish"
        elif bias_score < -0.15:
            bias = "bearish"
        else:
            bias = "neutral"

        return {
            "coin": coin,
            "vwap": round(vwap, 4),
            "total_volume_24h": round(total_volume, 2),
            "avg_price_change_pct": round(avg_change, 2),
            "avg_funding_rate": round(avg_funding, 6),
            "long_short_ratio": round(avg_long_ratio, 3),
            "directional_bias": bias,
            "bias_score": round(bias_score, 4),
            "confidence": round(confidence, 2),
            "exchanges_reporting": exchanges_reporting,
            "exchange_data": exchange_data,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_market_overview(self, coins: List[str] = None) -> Dict:
        """
        Get market overview across all tracked coins.
        Returns aggregate bias and volume data.
        """
        if coins is None:
            coins = ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "XRP"]

        # Check cache
        now = time.time()
        if now - self._cache_time < self._cache_ttl and self._cache:
            return self._cache

        overview = {
            "coins": {},
            "overall_bias": "neutral",
            "overall_bias_score": 0,
            "total_market_volume": 0,
            "bullish_coins": [],
            "bearish_coins": [],
            "neutral_coins": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

        bias_scores = []

        for coin in coins:
            try:
                data = self.get_multi_exchange_data(coin)
                if data:
                    overview["coins"][coin] = data
                    overview["total_market_volume"] += data.get("total_volume_24h", 0)
                    bias_scores.append(data["bias_score"])

                    if data["directional_bias"] == "bullish":
                        overview["bullish_coins"].append(coin)
                    elif data["directional_bias"] == "bearish":
                        overview["bearish_coins"].append(coin)
                    else:
                        overview["neutral_coins"].append(coin)
            except Exception as e:
                logger.debug(f"Error getting data for {coin}: {e}")

        if bias_scores:
            avg_bias = sum(bias_scores) / len(bias_scores)
            overview["overall_bias_score"] = round(avg_bias, 4)
            if avg_bias > 0.1:
                overview["overall_bias"] = "bullish"
            elif avg_bias < -0.1:
                overview["overall_bias"] = "bearish"

        # Cache the result
        self._cache = overview
        self._cache_time = now

        return overview

    def get_volume_confirmation(self, coin: str, side: str) -> Tuple[bool, float]:
        """
        Check if multi-exchange volume confirms a trade direction.
        Returns (confirmed: bool, confidence: float 0-1).

        Used by the paper trader to filter signals:
        - Only take longs when volume bias is bullish or neutral
        - Only take shorts when volume bias is bearish or neutral
        """
        data = self.get_multi_exchange_data(coin)
        if not data:
            return True, 0.5  # No data = don't block, but low confidence

        bias = data["directional_bias"]
        score = data["bias_score"]

        if side == "long":
            if bias == "bearish" and score < -0.25:
                return False, 0.2  # Strong bearish = don't go long
            elif bias == "bullish":
                return True, min(1.0, 0.6 + abs(score))
            else:
                return True, 0.5
        else:  # short
            if bias == "bullish" and score > 0.25:
                return False, 0.2  # Strong bullish = don't go short
            elif bias == "bearish":
                return True, min(1.0, 0.6 + abs(score))
            else:
                return True, 0.5


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agg = ExchangeAggregator()

    print("Testing multi-exchange aggregator...")
    for coin in ["BTC", "ETH", "SOL"]:
        data = agg.get_multi_exchange_data(coin)
        if data:
            print(f"\n{coin}:")
            print(f"  VWAP: ${data['vwap']:,.2f}")
            print(f"  24h Volume: ${data['total_volume_24h']:,.0f}")
            print(f"  Price Change: {data['avg_price_change_pct']:+.2f}%")
            print(f"  Directional Bias: {data['directional_bias']} (score: {data['bias_score']:+.4f})")
            print(f"  Exchanges: {', '.join(data['exchanges_reporting'])}")
