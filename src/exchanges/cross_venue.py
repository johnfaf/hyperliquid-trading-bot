"""
Cross-Venue Confirmation System
================================
Validates trading signals by checking if the same thesis is supported
across multiple exchanges. A signal that appears on both Hyperliquid
and Lighter is stronger than one on a single venue.

Confirmation dimensions:
  1. Trader overlap    — Same address active on multiple venues
  2. Direction consensus — Multiple venues showing same directional bias
  3. Funding divergence — Funding rate differences create arb opportunities
  4. Volume confirmation — High volume on multiple venues = stronger signal
  5. Spread/liquidity   — Tighter spreads on a venue = better execution

Scoring: Each confirmation adds to the signal's composite score.
         No confirmation doesn't kill a signal — it just doesn't boost it.
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .base_adapter import (
    BaseExchangeAdapter,
    NormalizedMarketData,
    NormalizedPosition,
    NormalizedFill,
)

logger = logging.getLogger(__name__)


@dataclass
class CrossVenueSignal:
    """A trading signal enriched with cross-venue confirmation data."""
    coin: str
    direction: str                     # "long" or "short"
    primary_exchange: str              # Where the signal originated
    primary_score: float               # Original signal score (0-1)

    # Cross-venue confirmation scores
    venue_count: int = 1               # How many venues confirm this direction
    direction_agreement: float = 0.0   # 0-1, how aligned venues are directionally
    funding_spread: float = 0.0        # Funding rate difference across venues
    volume_confirmation: float = 0.0   # Cross-venue volume strength
    trader_overlap: int = 0            # # of top traders active on multiple venues for this coin
    liquidity_score: float = 0.0       # Best available liquidity across venues

    # Composite
    confirmation_score: float = 0.0    # Overall cross-venue confirmation (0-1)
    confirmed_venues: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


@dataclass
class FundingArbitrageOpportunity:
    """Identified funding rate arbitrage between venues."""
    coin: str
    long_venue: str                    # Go long here (low/negative funding)
    short_venue: str                   # Go short here (high/positive funding)
    funding_spread_annualized: float   # Annualized spread in %
    long_funding_rate: float
    short_funding_rate: float
    confidence: float = 0.0            # 0-1


class CrossVenueConfirmation:
    """
    Cross-venue signal confirmation engine.

    Takes signals from the primary discovery pipeline (Hyperliquid)
    and checks them against secondary venues (Lighter, etc.) to
    boost confidence in genuine signals.
    """

    # Weights for confirmation scoring
    WEIGHTS = {
        "direction": 0.35,       # Same direction on multiple venues
        "volume": 0.25,          # Volume confirms the move
        "funding": 0.20,         # Funding rate alignment
        "trader_overlap": 0.10,  # Top traders active on multiple venues
        "liquidity": 0.10,       # Execution quality check
    }

    # Thresholds
    MIN_VOLUME_FOR_CONFIRMATION = 50_000   # $50k 24h volume to count
    FUNDING_ARB_THRESHOLD = 0.0005         # 0.05% hourly = ~43.8% annualized
    MIN_CONFIRMATION_SCORE = 0.15          # Below this, no meaningful confirmation

    def __init__(self, adapters: List[BaseExchangeAdapter]):
        """
        Initialize with a list of exchange adapters.
        The first adapter is treated as the primary venue.
        """
        self.adapters = {a.exchange_name: a for a in adapters}
        self.primary = adapters[0].exchange_name if adapters else "hyperliquid"

        # Cache for market data across venues (refreshed each cycle)
        self._market_data_cache: Dict[str, List[NormalizedMarketData]] = {}
        self._cache_age: float = 0
        self._CACHE_TTL = 120  # 2 minutes

        # Stats
        self.stats = {
            "confirmations_checked": 0,
            "confirmations_found": 0,
            "funding_arbs_found": 0,
            "avg_confirmation_score": 0.0,
        }

        logger.info(f"CrossVenueConfirmation initialized with {len(self.adapters)} venues: "
                    f"{list(self.adapters.keys())}")

    def _refresh_market_data(self, force: bool = False):
        """Refresh market data cache from all venues."""
        import time
        now = time.time()
        if not force and self._market_data_cache and (now - self._cache_age) < self._CACHE_TTL:
            return

        for name, adapter in self.adapters.items():
            try:
                self._market_data_cache[name] = adapter.get_market_data()
            except Exception as e:
                logger.warning(f"Failed to refresh market data from {name}: {e}")
                self._market_data_cache[name] = []

        self._cache_age = now

    def _get_market_for_coin(self, exchange: str, coin: str) -> Optional[NormalizedMarketData]:
        """Find market data for a specific coin on a specific exchange."""
        markets = self._market_data_cache.get(exchange, [])
        for m in markets:
            if m.coin == coin:
                return m
        return None

    # ─── Main Confirmation Method ──────────────────────────

    def confirm_signal(
        self,
        coin: str,
        direction: str,
        primary_score: float,
        primary_exchange: str = "hyperliquid",
    ) -> CrossVenueSignal:
        """
        Check a trading signal against all available venues.

        Args:
            coin: The asset (e.g., "BTC")
            direction: "long" or "short"
            primary_score: The signal's original score from primary venue
            primary_exchange: Which exchange generated the signal

        Returns:
            CrossVenueSignal with confirmation data filled in
        """
        self.stats["confirmations_checked"] += 1
        self._refresh_market_data()

        signal = CrossVenueSignal(
            coin=coin,
            direction=direction,
            primary_exchange=primary_exchange,
            primary_score=primary_score,
            confirmed_venues=[primary_exchange],
        )

        secondary_venues = [
            name for name in self.adapters.keys()
            if name != primary_exchange
        ]

        if not secondary_venues:
            # Single venue — no cross-confirmation possible
            signal.confirmation_score = 0.0
            return signal

        # Collect confirmation evidence from each secondary venue
        direction_scores = []
        volume_scores = []
        funding_rates = {}
        liquidity_scores = []

        # Get primary venue market data
        primary_market = self._get_market_for_coin(primary_exchange, coin)
        if primary_market:
            funding_rates[primary_exchange] = primary_market.funding_rate

        for venue_name in secondary_venues:
            market = self._get_market_for_coin(venue_name, coin)
            if not market:
                continue  # Coin not available on this venue

            # 1. Direction consensus via funding rate
            # Negative funding when long = market paying longs = bullish pressure
            # Positive funding when short = market paying shorts = bearish pressure
            if market.funding_rate != 0:
                funding_rates[venue_name] = market.funding_rate
                # Does the funding support our direction?
                if direction == "long" and market.funding_rate < 0:
                    direction_scores.append(1.0)
                elif direction == "short" and market.funding_rate > 0:
                    direction_scores.append(1.0)
                elif direction == "long" and market.funding_rate > 0.001:
                    direction_scores.append(-0.5)  # Funding opposes our direction
                elif direction == "short" and market.funding_rate < -0.001:
                    direction_scores.append(-0.5)
                else:
                    direction_scores.append(0.2)  # Neutral funding

            # 2. Volume confirmation
            if market.volume_24h >= self.MIN_VOLUME_FOR_CONFIRMATION:
                # Normalize volume score: log scale, cap at 1.0
                import math
                vol_score = min(1.0, math.log10(max(market.volume_24h, 1)) / 8)
                volume_scores.append(vol_score)
                signal.confirmed_venues.append(venue_name)

            # 3. Liquidity / spread quality
            if market.spread_bps > 0:
                # Tighter spread = better. <5bps is excellent, >50bps is poor
                liq_score = max(0, 1.0 - (market.spread_bps / 50))
                liquidity_scores.append(liq_score)

        # Compute composite confirmation
        signal.venue_count = len(signal.confirmed_venues)

        if direction_scores:
            signal.direction_agreement = max(0, sum(direction_scores) / len(direction_scores))

        if volume_scores:
            signal.volume_confirmation = sum(volume_scores) / len(volume_scores)

        if liquidity_scores:
            signal.liquidity_score = sum(liquidity_scores) / len(liquidity_scores)

        # Funding spread (for arb detection)
        if len(funding_rates) >= 2:
            rates = list(funding_rates.values())
            signal.funding_spread = max(rates) - min(rates)

        # Weighted composite score
        signal.confirmation_score = (
            self.WEIGHTS["direction"] * signal.direction_agreement +
            self.WEIGHTS["volume"] * signal.volume_confirmation +
            self.WEIGHTS["funding"] * min(1.0, signal.funding_spread * 1000) +
            self.WEIGHTS["trader_overlap"] * (min(1.0, signal.trader_overlap / 5) if signal.trader_overlap else 0) +
            self.WEIGHTS["liquidity"] * signal.liquidity_score
        )

        signal.details = {
            "venues_checked": len(secondary_venues),
            "venues_with_coin": signal.venue_count - 1,
            "funding_rates": funding_rates,
            "direction_scores": direction_scores,
        }

        if signal.confirmation_score >= self.MIN_CONFIRMATION_SCORE:
            self.stats["confirmations_found"] += 1

        # Update running average
        total = self.stats["confirmations_checked"]
        prev_avg = self.stats["avg_confirmation_score"]
        self.stats["avg_confirmation_score"] = (
            prev_avg * (total - 1) + signal.confirmation_score
        ) / total

        logger.info(
            f"CrossVenue [{coin} {direction.upper()}]: "
            f"confirmation={signal.confirmation_score:.3f} "
            f"(venues={signal.venue_count}, dir={signal.direction_agreement:.2f}, "
            f"vol={signal.volume_confirmation:.2f})"
        )

        return signal

    # ─── Batch Confirmation ────────────────────────────────

    def confirm_batch(
        self,
        signals: List[Dict],
        primary_exchange: str = "hyperliquid",
    ) -> List[CrossVenueSignal]:
        """
        Confirm a batch of signals efficiently.

        signals: list of {coin, direction, score} dicts
        Returns: list of CrossVenueSignal with confirmation data
        """
        # Pre-fetch all market data once
        self._refresh_market_data(force=True)

        results = []
        for sig in signals:
            confirmed = self.confirm_signal(
                coin=sig.get("coin", ""),
                direction=sig.get("direction", "long"),
                primary_score=sig.get("score", 0.5),
                primary_exchange=primary_exchange,
            )
            results.append(confirmed)

        # Sort by confirmation_score descending
        results.sort(key=lambda x: x.confirmation_score, reverse=True)
        return results

    # ─── Funding Arbitrage Scanner ─────────────────────────

    def scan_funding_arb(self) -> List[FundingArbitrageOpportunity]:
        """
        Scan for funding rate arbitrage opportunities across venues.

        Strategy: If BTC funding is +0.05% on Hyperliquid and -0.02% on Lighter,
        short on HL (collect funding) and long on Lighter (collect funding).
        The spread is 0.07% per hour ≈ 61.3% annualized.
        """
        self._refresh_market_data(force=True)

        opportunities = []

        # Collect all coins available on at least 2 venues
        coin_venues: Dict[str, Dict[str, float]] = {}
        for venue_name, markets in self._market_data_cache.items():
            for m in markets:
                if m.coin not in coin_venues:
                    coin_venues[m.coin] = {}
                if m.funding_rate != 0:
                    coin_venues[m.coin][venue_name] = m.funding_rate

        for coin, venues in coin_venues.items():
            if len(venues) < 2:
                continue

            # Find max and min funding venues
            sorted_venues = sorted(venues.items(), key=lambda x: x[1])
            lowest_venue, lowest_rate = sorted_venues[0]
            highest_venue, highest_rate = sorted_venues[-1]

            spread = highest_rate - lowest_rate
            if spread < self.FUNDING_ARB_THRESHOLD:
                continue

            # Annualize: hourly rate × 8760 hours/year
            annualized = spread * 8760 * 100  # As percentage

            # Confidence based on spread magnitude and venue reliability
            confidence = min(1.0, spread / 0.005)  # Full confidence at 0.5%/hr spread

            opportunities.append(FundingArbitrageOpportunity(
                coin=coin,
                long_venue=lowest_venue,    # Go long where funding is lowest
                short_venue=highest_venue,  # Go short where funding is highest
                funding_spread_annualized=annualized,
                long_funding_rate=lowest_rate,
                short_funding_rate=highest_rate,
                confidence=confidence,
            ))

        opportunities.sort(key=lambda x: x.funding_spread_annualized, reverse=True)
        self.stats["funding_arbs_found"] = len(opportunities)

        if opportunities:
            top = opportunities[0]
            logger.info(
                f"CrossVenue funding arb: {len(opportunities)} opportunities, "
                f"best: {top.coin} ({top.funding_spread_annualized:.1f}% ann.) "
                f"long@{top.long_venue} / short@{top.short_venue}"
            )

        return opportunities

    # ─── Trader Overlap Detection ──────────────────────────

    def check_trader_overlap(
        self,
        addresses: List[str],
        coin: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Check which addresses are active on multiple venues.

        Returns: {address: [list of venues where active]}
        """
        overlap: Dict[str, List[str]] = {}

        for addr in addresses:
            active_venues = []
            for venue_name, adapter in self.adapters.items():
                try:
                    positions = adapter.get_trader_positions(addr)
                    if positions:
                        if coin:
                            # Check if they have a position in the specific coin
                            if any(p.coin == coin for p in positions):
                                active_venues.append(venue_name)
                        else:
                            active_venues.append(venue_name)
                except Exception:
                    continue

            if len(active_venues) > 1:
                overlap[addr] = active_venues

        if overlap:
            logger.info(f"CrossVenue: {len(overlap)}/{len(addresses)} traders active on multiple venues")

        return overlap

    # ─── Stats ─────────────────────────────────────────────

    def get_stats(self) -> Dict:
        return {
            "venues": list(self.adapters.keys()),
            "venue_count": len(self.adapters),
            **self.stats,
            "market_data_cached": {
                name: len(markets)
                for name, markets in self._market_data_cache.items()
            },
        }
