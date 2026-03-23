"""
Multi-Exchange Scanner
======================
Top-level orchestrator that coordinates trader discovery and signal
confirmation across all connected venues.

Architecture:
  1. Primary discovery on Hyperliquid (richest leaderboard data)
  2. Secondary discovery on Lighter (volume-based)
  3. Cross-venue confirmation for all signals
  4. Funding arbitrage scanning across venues
  5. Unified candidate ranking with cross-venue boost

This module is designed to slot into the existing pipeline:
  trader_discovery.py → strategy_identifier.py → signal_processor.py
                                    ↑
                         multi_exchange_scanner (this)
                              ↑
                    [hyperliquid + lighter adapters]

The scanner runs as part of the main cycle, providing enriched
trader/strategy data to the existing pipeline.
"""
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .base_adapter import BaseExchangeAdapter, NormalizedTrader, NormalizedMarketData
from .hyperliquid_adapter import HyperliquidAdapter
from .lighter_adapter import LighterAdapter
from .cross_venue import CrossVenueConfirmation, CrossVenueSignal, FundingArbitrageOpportunity

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Result of a multi-exchange scan cycle."""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    cycle_duration_s: float = 0.0

    # Trader discovery
    traders_discovered: Dict[str, int] = field(default_factory=dict)  # {venue: count}
    total_unique_traders: int = 0
    multi_venue_traders: int = 0

    # Signal confirmation
    signals_checked: int = 0
    signals_confirmed: int = 0
    avg_confirmation_score: float = 0.0

    # Funding arb
    funding_arb_opportunities: int = 0
    best_arb_spread: float = 0.0

    # Venue health
    venue_health: Dict[str, bool] = field(default_factory=dict)

    # Errors
    errors: List[str] = field(default_factory=list)


class MultiExchangeScanner:
    """
    Coordinates multi-exchange trader discovery and signal confirmation.

    Usage in main.py:
        scanner = MultiExchangeScanner()
        # During each cycle:
        traders = scanner.discover_traders(limit=100)
        confirmed_signals = scanner.confirm_signals(raw_signals)
        arb_opps = scanner.scan_funding_arb()
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Initialize adapters
        self.adapters: Dict[str, BaseExchangeAdapter] = {}
        self._init_adapters()

        # Cross-venue confirmation engine
        adapter_list = list(self.adapters.values())
        self.cross_venue = CrossVenueConfirmation(adapter_list) if len(adapter_list) > 1 else None

        # State
        self._last_scan: Optional[ScanResult] = None
        self._trader_cache: Dict[str, NormalizedTrader] = {}
        self._scan_count = 0

        logger.info(f"MultiExchangeScanner initialized with {len(self.adapters)} venues: "
                    f"{list(self.adapters.keys())}")

    def _init_adapters(self):
        """Initialize exchange adapters based on config."""
        # Hyperliquid — always enabled (primary venue)
        try:
            hl = HyperliquidAdapter(self.config.get("hyperliquid", {}))
            self.adapters["hyperliquid"] = hl
            logger.info("Initialized Hyperliquid adapter (primary)")
        except Exception as e:
            logger.error(f"Failed to init Hyperliquid adapter: {e}")

        # Lighter — enabled by default, can be disabled via config
        if self.config.get("lighter_enabled", True):
            try:
                lighter = LighterAdapter(self.config.get("lighter", {}))
                self.adapters["lighter"] = lighter
                logger.info("Initialized Lighter adapter (secondary)")
            except Exception as e:
                logger.warning(f"Failed to init Lighter adapter: {e}")

        # Future adapters go here:
        # if self.config.get("paradex_enabled", False):
        #     self.adapters["paradex"] = ParadexAdapter(...)
        # if self.config.get("grvt_enabled", False):
        #     self.adapters["grvt"] = GRVTAdapter(...)

    # ─── Health Check ──────────────────────────────────────

    def check_health(self) -> Dict[str, bool]:
        """Check all venue APIs are reachable."""
        health = {}
        for name, adapter in self.adapters.items():
            try:
                health[name] = adapter.health_check()
            except Exception:
                health[name] = False
        return health

    # ─── Trader Discovery ──────────────────────────────────

    def discover_traders(self, limit: int = 100) -> List[NormalizedTrader]:
        """
        Discover traders across all venues and merge into unified list.

        Priority:
          1. Hyperliquid leaderboard (highest quality data)
          2. Lighter volume scan (supplementary)

        Traders found on multiple venues get a boost in ranking.
        """
        start = time.time()
        all_traders: Dict[str, NormalizedTrader] = {}
        per_venue_counts: Dict[str, int] = {}

        for venue_name, adapter in self.adapters.items():
            try:
                traders = adapter.get_top_traders(limit=limit)
                per_venue_counts[venue_name] = len(traders)

                for t in traders:
                    key = t.address.lower()
                    if key in all_traders:
                        # Trader exists from another venue — merge
                        existing = all_traders[key]
                        existing.raw_data[f"also_on_{venue_name}"] = True
                        # Boost PnL if secondary venue has data
                        if t.pnl_total and not existing.pnl_total:
                            existing.pnl_total = t.pnl_total
                    else:
                        all_traders[key] = t

            except Exception as e:
                logger.error(f"Trader discovery failed on {venue_name}: {e}")

        # Sort by PnL (primary) then trade count (secondary)
        sorted_traders = sorted(
            all_traders.values(),
            key=lambda t: (t.pnl_total, t.trade_count_30d),
            reverse=True,
        )[:limit]

        # Count multi-venue traders
        multi_venue = sum(
            1 for t in sorted_traders
            if any(k.startswith("also_on_") for k in t.raw_data)
        )

        duration = time.time() - start
        logger.info(
            f"MultiExchange discovery: {len(sorted_traders)} traders "
            f"({per_venue_counts}) in {duration:.1f}s, "
            f"{multi_venue} multi-venue"
        )

        # Cache for later use
        self._trader_cache = {t.address.lower(): t for t in sorted_traders}

        return sorted_traders

    # ─── Signal Confirmation ───────────────────────────────

    def confirm_signals(
        self,
        signals: List[Dict],
        primary_exchange: str = "hyperliquid",
    ) -> List[CrossVenueSignal]:
        """
        Run cross-venue confirmation on a batch of signals.

        Integrates with the existing pipeline — call this after
        SignalProcessor produces filtered strategies, before
        DecisionEngine does final ranking.

        signals: list of dicts with keys {coin, direction, score}
        Returns: list of CrossVenueSignal with confirmation scores
        """
        if not self.cross_venue:
            logger.debug("No cross-venue engine (single venue mode)")
            return [
                CrossVenueSignal(
                    coin=s.get("coin", ""),
                    direction=s.get("direction", "long"),
                    primary_exchange=primary_exchange,
                    primary_score=s.get("score", 0.5),
                )
                for s in signals
            ]

        return self.cross_venue.confirm_batch(signals, primary_exchange)

    # ─── Funding Arb ───────────────────────────────────────

    def scan_funding_arb(self) -> List[FundingArbitrageOpportunity]:
        """Scan for funding rate arbitrage across venues."""
        if not self.cross_venue:
            return []
        return self.cross_venue.scan_funding_arb()

    # ─── Market Data Aggregation ───────────────────────────

    def get_aggregated_market_data(
        self,
        coins: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, NormalizedMarketData]]:
        """
        Get market data from all venues, organized by coin.

        Returns: {coin: {venue_name: NormalizedMarketData}}
        """
        result: Dict[str, Dict[str, NormalizedMarketData]] = {}

        for venue_name, adapter in self.adapters.items():
            try:
                markets = adapter.get_market_data(coins)
                for m in markets:
                    if m.coin not in result:
                        result[m.coin] = {}
                    result[m.coin][venue_name] = m
            except Exception as e:
                logger.warning(f"Market data fetch failed for {venue_name}: {e}")

        return result

    def get_common_markets(self) -> List[str]:
        """Get coins listed on ALL connected venues."""
        if not self.adapters:
            return []

        market_sets = []
        for adapter in self.adapters.values():
            try:
                markets = set(adapter.get_available_markets())
                if markets:
                    market_sets.append(markets)
            except Exception:
                continue

        if not market_sets:
            return []

        common = market_sets[0]
        for s in market_sets[1:]:
            common = common.intersection(s)

        return sorted(common)

    # ─── Full Scan Cycle ───────────────────────────────────

    def run_scan_cycle(self, trader_limit: int = 100) -> ScanResult:
        """
        Run a complete multi-exchange scan cycle.
        Returns a ScanResult with all findings.
        """
        start = time.time()
        self._scan_count += 1
        result = ScanResult()

        # 1. Health check
        result.venue_health = self.check_health()
        unhealthy = [v for v, ok in result.venue_health.items() if not ok]
        if unhealthy:
            logger.warning(f"Unhealthy venues: {unhealthy}")
            result.errors.append(f"Unhealthy venues: {unhealthy}")

        # 2. Trader discovery
        try:
            traders = self.discover_traders(limit=trader_limit)
            result.total_unique_traders = len(traders)
            result.traders_discovered = {
                name: adapter.get_stats().get("requests", 0)
                for name, adapter in self.adapters.items()
            }
            result.multi_venue_traders = sum(
                1 for t in traders
                if any(k.startswith("also_on_") for k in t.raw_data)
            )
        except Exception as e:
            result.errors.append(f"Discovery error: {e}")
            logger.error(f"Discovery cycle failed: {e}")

        # 3. Funding arb scan
        try:
            arbs = self.scan_funding_arb()
            result.funding_arb_opportunities = len(arbs)
            if arbs:
                result.best_arb_spread = arbs[0].funding_spread_annualized
        except Exception as e:
            result.errors.append(f"Funding arb error: {e}")

        result.cycle_duration_s = time.time() - start
        self._last_scan = result

        logger.info(
            f"Scan cycle #{self._scan_count} complete in {result.cycle_duration_s:.1f}s: "
            f"{result.total_unique_traders} traders, "
            f"{result.funding_arb_opportunities} arb opps"
        )

        return result

    # ─── Stats ─────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Get scanner stats for dashboard."""
        stats = {
            "venues": list(self.adapters.keys()),
            "venue_count": len(self.adapters),
            "scan_count": self._scan_count,
            "cached_traders": len(self._trader_cache),
        }

        # Per-venue stats
        for name, adapter in self.adapters.items():
            stats[f"venue_{name}"] = adapter.get_stats()

        # Cross-venue stats
        if self.cross_venue:
            stats["cross_venue"] = self.cross_venue.get_stats()

        # Last scan result
        if self._last_scan:
            stats["last_scan"] = {
                "timestamp": self._last_scan.timestamp,
                "duration": self._last_scan.cycle_duration_s,
                "traders": self._last_scan.total_unique_traders,
                "multi_venue": self._last_scan.multi_venue_traders,
                "arbs": self._last_scan.funding_arb_opportunities,
                "errors": len(self._last_scan.errors),
            }

        return stats
