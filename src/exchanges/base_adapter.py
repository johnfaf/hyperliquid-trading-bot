"""
Base Exchange Adapter
=====================
Abstract interface that all exchange adapters must implement.

The goal is uniform trader discovery across venues:
  - Discover top traders (leaderboard or volume-based)
  - Fetch recent fills/trades for a trader
  - Fetch current positions for a trader
  - Get market data (prices, funding, OI)
  - Classify whether an address is a bot or human

Each adapter normalizes its exchange's API responses into a common schema
so the upstream pipeline (trader_discovery, strategy_identifier, etc.)
can consume data from any venue without knowing the source.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ─── Common Data Models ────────────────────────────────────────

@dataclass
class NormalizedTrader:
    """Uniform trader representation across all exchanges."""
    address: str
    exchange: str                    # "hyperliquid", "lighter", etc.
    display_name: Optional[str] = None
    pnl_total: float = 0.0
    pnl_7d: float = 0.0
    pnl_30d: float = 0.0
    win_rate: float = 0.0
    trade_count_30d: int = 0
    avg_position_size: float = 0.0
    max_leverage: float = 0.0
    account_age_days: int = 0
    is_bot: bool = False
    bot_score: int = 0              # 0-10 (0=human, 10=definite bot)
    raw_data: Dict = field(default_factory=dict)
    discovered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class NormalizedFill:
    """Uniform trade/fill representation."""
    exchange: str
    trader_address: str
    coin: str                        # Normalized symbol (e.g., "BTC", "ETH")
    side: str                        # "buy" or "sell"
    size: float                      # In base asset units
    price: float
    notional_usd: float
    fee: float = 0.0
    realized_pnl: float = 0.0
    timestamp: str = ""
    trade_id: str = ""
    is_liquidation: bool = False


@dataclass
class NormalizedPosition:
    """Uniform position representation."""
    exchange: str
    trader_address: str
    coin: str
    side: str                        # "long" or "short"
    size: float                      # Unsigned, in base units
    entry_price: float
    mark_price: float = 0.0
    unrealized_pnl: float = 0.0
    leverage: float = 1.0
    margin_used: float = 0.0
    liquidation_price: float = 0.0


@dataclass
class NormalizedMarketData:
    """Uniform market snapshot for a single asset."""
    exchange: str
    coin: str
    mid_price: float
    funding_rate: float = 0.0        # Current hourly funding rate
    open_interest: float = 0.0       # In USD
    volume_24h: float = 0.0          # In USD
    mark_price: float = 0.0
    index_price: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread_bps: float = 0.0          # Spread in basis points


# ─── Abstract Base Adapter ──────────────────────────────────────

class BaseExchangeAdapter(ABC):
    """
    Interface contract for exchange adapters.

    Every adapter must implement these methods to participate in
    the multi-venue trader discovery pipeline.
    """

    # ★ H34 FIX: cross_venue.scan_funding_arb normalizes funding rates to a
    # per-hour basis before comparing across venues.  The most common perp
    # funding cadence (Binance, Bybit, OKX) is 8 hours -- so 8 is the safer
    # default for adapters that don't know otherwise.  Hyperliquid and
    # Lighter both override to 1.  When adding a new adapter, set this on
    # the class to match the venue's funding cadence.
    funding_period_hours: float = 8.0

    def __init__(self, name: str, base_url: str, config: Optional[Dict] = None):
        self.name = name
        self.base_url = base_url
        self.config = config or {}
        self._request_count = 0
        self._error_count = 0
        self._last_request_time = 0.0
        self.logger = logging.getLogger(f"exchange.{name}")

    @property
    def exchange_name(self) -> str:
        return self.name

    # ─── Required Methods ──────────────────────────────

    @abstractmethod
    def get_top_traders(self, limit: int = 100) -> List[NormalizedTrader]:
        """
        Discover top traders on this exchange.

        Implementation varies by exchange:
          - Hyperliquid: leaderboard endpoint
          - Lighter: volume leaders / top PnL
          - Others: whatever discovery mechanism is available

        Returns normalized trader objects sorted by relevance/PnL.
        """
        ...

    @abstractmethod
    def get_trader_fills(self, address: str, limit: int = 200) -> List[NormalizedFill]:
        """
        Fetch recent fills/trades for a specific trader.
        Used for strategy identification and bot detection.
        """
        ...

    @abstractmethod
    def get_trader_positions(self, address: str) -> List[NormalizedPosition]:
        """
        Fetch current open positions for a trader.
        Used for copy trading and position analysis.
        """
        ...

    @abstractmethod
    def get_market_data(self, coins: Optional[List[str]] = None) -> List[NormalizedMarketData]:
        """
        Get market data snapshots.
        If coins is None, return data for all available markets.
        """
        ...

    @abstractmethod
    def get_available_markets(self) -> List[str]:
        """Return list of available trading pairs/coins on this exchange."""
        ...

    @abstractmethod
    def normalize_coin_symbol(self, raw_symbol: str) -> str:
        """
        Convert exchange-specific symbol to canonical form.
        e.g., "BTC-PERP" -> "BTC", "ETHUSDT" -> "ETH"
        """
        ...

    # ─── Optional Methods (with defaults) ──────────────

    def get_trader_pnl_history(self, address: str, days: int = 30) -> List[Dict]:
        """
        Fetch PnL history for a trader. Not all exchanges support this.
        Returns list of {date, pnl, cumulative_pnl}.
        """
        self.logger.debug(f"PnL history not implemented for {self.name}")
        return []

    def get_funding_rates(self) -> Dict[str, float]:
        """
        Get current funding rates for all markets.
        Returns {coin: rate}.
        """
        self.logger.debug(f"Funding rates not implemented for {self.name}")
        return {}

    def health_check(self) -> bool:
        """
        Check if the exchange API is reachable and functioning.
        Default implementation tries to fetch market data.

        ★ L14 FIX: previously this called ``get_available_markets()`` which
        many adapters serve from a 5-minute cache.  A venue that went down
        within the cache window kept reporting healthy until the cache
        expired -- silent fail-open at exactly the wrong time.  Adapters
        can now override ``ping()`` to make a fresh, uncached round-trip
        (e.g. /info ping, /v1/exchange info).  When ``ping()`` is not
        overridden, fall back to the cached path but log a one-shot
        DEBUG so it's clear why.
        """
        try:
            ping = self.ping()
            if ping is not None:
                return bool(ping)
        except Exception as exc:
            self.logger.warning(f"Health-check ping failed for {self.name}: {exc}")
            return False
        # Adapter hasn't overridden ping(); fall back to cached probe.
        try:
            markets = self.get_available_markets()
            return len(markets) > 0
        except Exception as e:
            self.logger.warning(f"Health check failed for {self.name}: {e}")
            return False

    def ping(self) -> Optional[bool]:
        """★ L14: optional uncached liveness probe.

        Adapters should override this to perform a minimal-cost call to
        the exchange API that bypasses any local caches (e.g. fetching
        a status endpoint, requesting a single market's mid price with
        cache disabled).  Return True/False; return None to signal that
        no fresh probe is implemented and ``health_check`` should fall
        back to the cached path.
        """
        return None

    # ─── Stats ─────────────────────────────────────────

    def get_stats(self) -> Dict:
        return {
            "exchange": self.name,
            "requests": self._request_count,
            "errors": self._error_count,
            "error_rate": self._error_count / max(self._request_count, 1),
        }
