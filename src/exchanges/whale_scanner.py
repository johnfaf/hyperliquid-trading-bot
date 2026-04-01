"""
Whale Trade Detection Scanner
==============================
Monitors crypto.com for large whale trades (>$100K notional) and converts
them into trading signals. Runs as part of the fast cycle (60s intervals).

Integration into pipeline:
  fast_cycle → scan_cryptocom_whales() → whale trades detected
       ↓
  signal_from_whale_trade() → TradeSignal objects
       ↓
  signal_processor.process() → deduplicated signals
       ↓
  firewall validation → execution
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class WhaleScanner:
    """Detect and track whale trades across monitored coins."""

    # Track recent whales to avoid duplicate signals within 30 seconds
    _whale_cache: Dict[str, float] = {}  # key: f"{coin}:{side}:{notional}" -> timestamp

    def __init__(self, cryptocom_client=None, coins: Optional[List[str]] = None,
                 min_notional: float = 100000.0, cache_ttl: float = 30.0):
        """
        Initialize whale scanner.

        Args:
            cryptocom_client: Instance of CryptoComClient (lazy-loaded if None)
            coins: List of coins to monitor (default: ["BTC", "ETH", "SOL"])
            min_notional: Minimum notional USD value to flag as whale (default $100k)
            cache_ttl: Cache TTL in seconds to avoid duplicate signals
        """
        self.cryptocom_client = cryptocom_client
        self._coins = coins or ["BTC", "ETH", "SOL"]
        self._min_notional = min_notional
        self._cache_ttl = cache_ttl
        self._last_scan_time = 0.0
        self._scan_count = 0
        self._whales_found = 0

        logger.info(
            f"WhaleScanner initialized: monitoring {self._coins} "
            f"with min_notional=${min_notional:,.0f}"
        )

    def _get_cryptocom_client(self):
        """Lazy-load crypto.com client if not provided."""
        if self.cryptocom_client is None:
            try:
                from src.data.cryptocom_client import CryptoComClient
                self.cryptocom_client = CryptoComClient()
            except ImportError:
                logger.error("Failed to import CryptoComClient")
                return None
        return self.cryptocom_client

    def scan_whales(self) -> List[Dict]:
        """
        Scan all monitored coins for whale trades on crypto.com.

        Returns:
            List of whale trade dicts with: coin, side, price, qty, notional, timestamp
        """
        self._scan_count += 1
        client = self._get_cryptocom_client()
        if not client:
            return []

        whales = []
        current_time = datetime.utcnow().timestamp()

        for coin in self._coins:
            try:
                # Get recent large trades from crypto.com
                whale_trades = client.get_whale_trades(
                    coin=coin,
                    min_notional=self._min_notional,
                    count=100,  # Scan last 100 trades
                    is_perp=False  # Monitor spot market
                )

                if not whale_trades:
                    continue

                for whale in whale_trades:
                    # Deduplicate within cache TTL
                    cache_key = f"{whale['coin']}:{whale['side']}:{whale['notional']}"
                    if cache_key in self._whale_cache:
                        cache_age = current_time - self._whale_cache[cache_key]
                        if cache_age < self._cache_ttl:
                            # Skip duplicate within TTL
                            continue

                    # Record in cache
                    self._whale_cache[cache_key] = current_time

                    whales.append({
                        "coin": whale["coin"],
                        "side": whale["side"],  # "buy" or "sell"
                        "price": whale["price"],
                        "qty": whale["qty"],
                        "notional": whale["notional"],
                        "timestamp": whale["timestamp"],
                        "exchange": whale["exchange"],
                    })
                    self._whales_found += 1

            except Exception as e:
                logger.debug(f"Error scanning whales for {coin}: {e}")

        if whales:
            logger.info(f"[whale] Detected {len(whales)} whale trades across {self._coins}")

        return whales

    def get_stats(self) -> Dict:
        """Return scanner statistics."""
        return {
            "scan_count": self._scan_count,
            "whales_found": self._whales_found,
            "coins_monitored": len(self._coins),
        }
