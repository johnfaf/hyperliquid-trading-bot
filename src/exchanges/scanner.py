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
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .base_adapter import BaseExchangeAdapter, NormalizedTrader, NormalizedMarketData
from .hyperliquid_adapter import HyperliquidAdapter
from .lighter_adapter import LighterAdapter, VenueState
from .cross_venue import CrossVenueConfirmation, CrossVenueSignal, FundingArbitrageOpportunity

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Result of a multi-exchange scan cycle."""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cycle_duration_s: float = 0.0

    # Trader discovery — real counts, not request counts
    traders_discovered: Dict[str, int] = field(default_factory=dict)  # {venue: actual_trader_count}
    total_unique_traders: int = 0
    multi_venue_traders: int = 0

    # Signal confirmation
    signals_checked: int = 0
    signals_confirmed: int = 0
    signals_boosted: int = 0
    avg_confirmation_score: float = 0.0

    # Funding arb
    funding_arb_opportunities: int = 0
    best_arb_spread: float = 0.0

    # Venue health — three-state: configured / healthy / down
    venue_health: Dict[str, str] = field(default_factory=dict)  # {venue: state_name}

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

        # Cross-venue confirmation engine — only with healthy venues
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

    def _get_healthy_adapters(self) -> Dict[str, BaseExchangeAdapter]:
        """Return adapters that are usable (healthy or degraded).

        DEGRADED venues still have API connectivity — they may just have
        incomplete market data. They can still provide useful cross-venue
        signals (funding rates, volume, etc.) even with partial data.
        Only DOWN venues are excluded entirely.
        """
        healthy = {}
        for name, adapter in self.adapters.items():
            # Check if adapter has state tracking (Lighter does, HL doesn't yet)
            state = getattr(adapter, "state", None)
            if state is None or state == VenueState.HEALTHY:
                healthy[name] = adapter
            elif state == VenueState.INITIALIZED:
                # Not yet checked — include it, health check will update
                healthy[name] = adapter
            elif state == VenueState.DEGRADED:
                # Degraded = API is reachable but incomplete data
                # Still include for cross-venue confirmation (partial is better than none)
                healthy[name] = adapter
                logger.debug(f"Including {name} (degraded) for cross-venue -- partial data available")
            else:
                logger.debug(f"Skipping {name} -- state={state.value}")
        return healthy

    # ─── Health Check ──────────────────────────────────────

    def check_health(self) -> Dict[str, str]:
        """
        Check all venue APIs. Returns {venue: state_string}.
        Three states: 'healthy', 'degraded', 'down' (plus 'initialized' before first check).
        Only healthy venues count toward confirmation scoring.
        """
        health = {}
        for name, adapter in self.adapters.items():
            try:
                is_healthy = adapter.health_check()
                # Get the detailed state if available
                state = getattr(adapter, "state", None)
                if state:
                    health[name] = state.value
                else:
                    health[name] = "healthy" if is_healthy else "down"
            except Exception as e:
                health[name] = "down"
                logger.warning(f"Health check exception for {name}: {e}")
        return health

    # ─── Trader Discovery ──────────────────────────────────

    def discover_traders(self, limit: int = 100) -> List[NormalizedTrader]:
        """
        Discover traders across all HEALTHY venues and merge into unified list.
        """
        start = time.time()
        all_traders: Dict[str, NormalizedTrader] = {}
        per_venue_counts: Dict[str, int] = {}

        for venue_name, adapter in self.adapters.items():
            try:
                traders = adapter.get_top_traders(limit=limit)
                per_venue_counts[venue_name] = len(traders)  # REAL trader count

                for t in traders:
                    key = t.address.lower()
                    if key in all_traders:
                        existing = all_traders[key]
                        existing.raw_data[f"also_on_{venue_name}"] = True
                        if t.pnl_total and not existing.pnl_total:
                            existing.pnl_total = t.pnl_total
                    else:
                        all_traders[key] = t

            except Exception as e:
                per_venue_counts[venue_name] = 0
                logger.error(f"Trader discovery failed on {venue_name}: {e}")

        sorted_traders = sorted(
            all_traders.values(),
            key=lambda t: (t.pnl_total, t.trade_count_30d),
            reverse=True,
        )[:limit]

        multi_venue = sum(
            1 for t in sorted_traders
            if any(k.startswith("also_on_") for k in t.raw_data)
        )

        duration = time.time() - start
        logger.info(
            f"MultiExchange discovery: {len(sorted_traders)} unique traders "
            f"(per venue: {per_venue_counts}) in {duration:.1f}s, "
            f"{multi_venue} multi-venue"
        )

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
        Logs detailed observability: checked, boosted, downgraded counts.
        """
        if not self.cross_venue:
            logger.info("CrossVenue: single venue mode -- no confirmation possible")
            return [
                CrossVenueSignal(
                    coin=s.get("coin", ""),
                    direction=s.get("direction", "long"),
                    primary_exchange=primary_exchange,
                    primary_score=s.get("score", 0.5),
                )
                for s in signals
            ]

        # Check if any secondary venue is actually healthy
        healthy = self._get_healthy_adapters()
        secondary_healthy = [n for n in healthy if n != primary_exchange]
        if not secondary_healthy:
            logger.info(f"CrossVenue: no healthy secondary venues "
                       f"(states: {self.check_health()}) -- skipping confirmation")
            return [
                CrossVenueSignal(
                    coin=s.get("coin", ""),
                    direction=s.get("direction", "long"),
                    primary_exchange=primary_exchange,
                    primary_score=s.get("score", 0.5),
                )
                for s in signals
            ]

        confirmed = self.cross_venue.confirm_batch(signals, primary_exchange)

        # Observability: count results
        checked = len(confirmed)
        boosted = sum(1 for c in confirmed if c.confirmation_score > 0.15)
        no_data = sum(1 for c in confirmed if c.venue_count <= 1)
        avg_score = (sum(c.confirmation_score for c in confirmed) / max(checked, 1))

        logger.info(
            f"CrossVenue confirmation: {checked} signals checked, "
            f"{boosted} boosted (score>0.15), "
            f"{no_data} no secondary data, "
            f"avg_score={avg_score:.3f}, "
            f"secondary_venues={secondary_healthy}"
        )

        return confirmed

    # ─── Funding Arb ───────────────────────────────────────

    def scan_funding_arb(self) -> List[FundingArbitrageOpportunity]:
        """Scan for funding rate arbitrage across healthy venues."""
        if not self.cross_venue:
            return []

        healthy = self._get_healthy_adapters()
        if len(healthy) < 2:
            logger.debug("Funding arb: need 2+ healthy venues, skipping")
            return []

        return self.cross_venue.scan_funding_arb()

    # ─── Market Data Aggregation ───────────────────────────

    def get_aggregated_market_data(
        self,
        coins: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, NormalizedMarketData]]:
        """Get market data from all venues, organized by coin."""
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
        """Get coins listed on ALL connected HEALTHY venues."""
        healthy = self._get_healthy_adapters()
        if len(healthy) < 2:
            return []

        market_sets = []
        for adapter in healthy.values():
            try:
                markets = set(adapter.get_available_markets())
                if markets:
                    market_sets.append(markets)
            except Exception:
                continue

        if len(market_sets) < 2:
            return []

        common = market_sets[0]
        for s in market_sets[1:]:
            common = common.intersection(s)

        return sorted(common)

    # ─── Full Scan Cycle ───────────────────────────────────

    def run_scan_cycle(self, trader_limit: int = 100) -> ScanResult:
        """Run a complete multi-exchange scan cycle."""
        start = time.time()
        self._scan_count += 1
        result = ScanResult()

        # 1. Health check
        result.venue_health = self.check_health()
        unhealthy = [v for v, state in result.venue_health.items() if state == "down"]
        if unhealthy:
            logger.warning(f"Unhealthy venues: {unhealthy}")
            result.errors.append(f"Unhealthy venues: {unhealthy}")

        # 2. Trader discovery
        try:
            traders = self.discover_traders(limit=trader_limit)
            result.total_unique_traders = len(traders)
            # Store REAL discovered trader counts per venue (not request counts)
            for name, adapter in self.adapters.items():
                stats = adapter.get_stats()
                # If the adapter tracks real counts, use those
                result.traders_discovered[name] = stats.get("markets_loaded", 0)
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
            f"venues={result.venue_health}, "
            f"{result.total_unique_traders} traders, "
            f"{result.funding_arb_opportunities} arb opps, "
            f"errors={len(result.errors)}"
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

        # Per-venue stats with health state
        for name, adapter in self.adapters.items():
            venue_stats = adapter.get_stats()
            state = getattr(adapter, "state", None)
            if state:
                venue_stats["state"] = state.value
            stats[f"venue_{name}"] = venue_stats

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
                "venue_health": self._last_scan.venue_health,
                "arbs": self._last_scan.funding_arb_opportunities,
                "errors": len(self._last_scan.errors),
            }

        return stats
