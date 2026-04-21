"""
Polymarket Scanner (V6)
========================
Scans Polymarket prediction markets for:
1. High-volume markets with crypto relevance
2. Sharp odds movements (smart money detection)
3. Cross-correlation with Hyperliquid signals
4. Event-driven trading opportunities

Feeds signals into the existing strategy pipeline.

Architecture:
  Markets discovery → Odds movement detection → Hyperliquid correlation
                    ↓
              Signal generation (compatible with DecisionEngine)

Polymarket API:
  REST: https://clob.polymarket.com
  - Markets: GET /markets (returns list of markets with tokens, outcomes, prices)
  - Order book: GET /book?token_id={token_id}
  - Trades: GET /trades?market={condition_id}
  - Prices: GET /prices?token_ids={comma_separated_ids}
  - Gamma API: https://gamma-api.polymarket.com (enriched data)
"""

import json
import logging
import time
import requests
from typing import Dict, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class PolymarketMarket:
    """Normalized Polymarket market data."""
    token_id: str
    market_id: str  # condition_id
    title: str
    description: str
    outcomes: List[str]
    current_prices: List[float]  # Probabilities for each outcome
    volume_24h: float
    liquidity: float
    last_traded: str  # ISO timestamp
    category: str


@dataclass
class OddsMovement:
    """Detected odds movement in a market."""
    market_id: str
    title: str
    direction: str  # "up" or "down"
    magnitude: float  # Percentage change
    timeframe: str  # "1h" or "24h"
    current_probability: float
    volume_move: float  # Volume during the move
    smart_money_score: float  # 0-1, higher = more likely smart money


@dataclass
class PolymarketSignal:
    """Signal for the bot's decision engine."""
    source: str = "polymarket"
    coin: str = ""
    side: str = ""  # "long" or "short"
    confidence: float = 0.0
    reason: str = ""
    polymarket_market: str = ""
    polymarket_probability: float = 0.0
    polymarket_volume_24h: float = 0.0
    correlation_with_hl: str = ""  # Description of correlation
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict:
        """Convert to dict for pipeline compatibility."""
        return {
            "source": self.source,
            "coin": self.coin,
            "side": self.side,
            "confidence": self.confidence,
            "reason": self.reason,
            "polymarket_market": self.polymarket_market,
            "polymarket_probability": self.polymarket_probability,
            "polymarket_volume_24h": self.polymarket_volume_24h,
            "correlation_with_hl": self.correlation_with_hl,
            "timestamp": self.timestamp,
        }


class PolymarketScanner:
    """
    Scans Polymarket for crypto-relevant prediction markets and generates
    trading signals based on odds movements and cross-correlation with Hyperliquid data.

    Usage:
        scanner = PolymarketScanner()
        signals = scanner.generate_signals()
        sentiment = scanner.get_market_sentiment()
        stats = scanner.get_stats()
    """

    # Crypto keyword mapping
    CRYPTO_KEYWORDS = {
        "btc": ["bitcoin", "btc"],
        "eth": ["ethereum", "eth"],
        "sol": ["solana", "sol"],
        "doge": ["dogecoin", "doge"],
        "xrp": ["ripple", "xrp"],
        "arb": ["arbitrum", "arb"],
        "op": ["optimism", "op"],
        "avax": ["avalanche", "avax"],
        "matic": ["polygon", "matic"],
        "link": ["chainlink", "link"],
        "sui": ["sui"],
        "inj": ["injective", "inj"],
        "near": ["near"],
        "sei": ["sei"],
        "tia": ["celestia", "tia"],
    }

    # Event-to-crypto sentiment mapping
    EVENT_SENTIMENT_MAP = {
        # Bullish events (risk-on)
        "fed": ("risk_on", 0.3),
        "rate cut": ("risk_on", 0.5),
        "inflation down": ("risk_on", 0.4),
        "stimulus": ("risk_on", 0.5),
        "bitcoin etf": ("risk_on", 0.6),
        "approval": ("risk_on", 0.4),
        "breakout": ("risk_on", 0.3),
        "bull": ("risk_on", 0.4),

        # Bearish events (risk-off)
        "rate hike": ("risk_off", 0.5),
        "inflation": ("risk_off", 0.4),
        "recession": ("risk_off", 0.6),
        "crash": ("risk_off", 0.5),
        "liquidation": ("risk_off", 0.4),
        "regulation": ("risk_off", 0.3),
        "ban": ("risk_off", 0.5),
        "bear": ("risk_off", 0.4),

        # Neutral/uncertain
        "election": ("uncertain", 0.2),
        "sec": ("uncertain", 0.2),
    }

    # API endpoints
    CLOB_API = "https://clob.polymarket.com"
    GAMMA_API = "https://gamma-api.polymarket.com"

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.source_registry = self.config.get("source_registry")
        self.market_provider = self.config.get("market_provider")
        self.replay_as_of_ms = self.config.get("replay_as_of_ms")

        # Cache configuration
        self.cache_ttl_minutes = self.config.get("cache_ttl_minutes", 5)
        self.min_volume_threshold = self.config.get("min_volume_threshold", 10000)  # $10k
        self.min_liquidity_threshold = self.config.get("min_liquidity_threshold", 1000)  # $1k

        # Detection thresholds
        self.odds_movement_threshold_1h = self.config.get("odds_movement_1h", 0.005)  # 0.5% (for 3-min scans)
        self.odds_movement_threshold_24h = self.config.get("odds_movement_24h", 0.10)  # 10%
        self.smart_money_volume_threshold = self.config.get("smart_money_volume_threshold", 50000)  # $50k

        # Scan configuration
        self.scan_interval_seconds = self.config.get("scan_interval_seconds", 60)
        self.max_markets_per_scan = max(1, int(self.config.get("max_markets_per_scan", 100)))
        self.max_retries = int(self.config.get("max_retries", 3))

        # Per-host rate limiting for direct external API calls.
        self.rate_limit_delay = float(self.config.get("rate_limit_delay", 0.35))
        self._rate_limit_by_host = {
            "clob.polymarket.com": float(
                self.config.get("clob_rate_limit_delay", self.rate_limit_delay)
            ),
            "gamma-api.polymarket.com": float(
                self.config.get("gamma_rate_limit_delay", self.rate_limit_delay)
            ),
        }
        self._last_request_time_by_host: Dict[str, float] = {}

        # Cache
        self._market_cache: Dict[str, PolymarketMarket] = {}
        self._price_cache: Dict[str, List[float]] = {}  # token_id -> prices
        self._price_cache_time: Dict[str, float] = {}  # token_id -> timestamp
        self._movement_cache: Dict[str, List[OddsMovement]] = {}  # market_id -> movements

        # Statistics
        self._scan_count = 0
        self._last_scan_time: Optional[datetime] = None
        self._last_scan_attempt_ts: float = 0.0
        self._last_raw_market_count = 0
        self._last_filtered_market_count = 0
        self._signals_generated = 0
        self._markets_tracked = 0
        self._crypto_markets_found = 0
        self._movements_detected = 0
        self._scan_cache_window_s = max(
            30.0,
            min(float(self.cache_ttl_minutes) * 60.0, float(self.scan_interval_seconds)),
        )

        logger.info(f"PolymarketScanner initialized with cache TTL={self.cache_ttl_minutes}min, "
                   f"rate_limit={self.rate_limit_delay}s")

    def set_replay_time(self, as_of_ms: Optional[int]) -> None:
        """Switch provider-backed scans to a point-in-time replay timestamp."""
        self.replay_as_of_ms = int(as_of_ms) if as_of_ms is not None else None

    def _rate_limit(self, url: str):
        """Enforce per-host rate limiting for Polymarket endpoints."""
        host = urlparse(url).netloc.lower()
        delay = self._rate_limit_by_host.get(host, self.rate_limit_delay)
        last = self._last_request_time_by_host.get(host, 0.0)
        elapsed = time.time() - last
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_time_by_host[host] = time.time()

    def _fetch_json(self, url: str, timeout: int = 10) -> Optional[Dict]:
        """Fetch JSON with per-host throttling and retry/backoff on transient failures."""
        for attempt in range(self.max_retries):
            self._rate_limit(url)
            try:
                response = requests.get(url, timeout=timeout)
                if response.status_code == 429 or response.status_code >= 500:
                    backoff = min(8.0, 0.5 * (2 ** attempt))
                    logger.warning(
                        "Polymarket fetch %s returned %d (attempt %d/%d); retrying in %.1fs",
                        url,
                        response.status_code,
                        attempt + 1,
                        self.max_retries,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout:
                backoff = min(8.0, 0.5 * (2 ** attempt))
                logger.warning(
                    "Timeout fetching %s (attempt %d/%d); retrying in %.1fs",
                    url,
                    attempt + 1,
                    self.max_retries,
                    backoff,
                )
                time.sleep(backoff)
            except requests.exceptions.ConnectionError:
                backoff = min(8.0, 0.5 * (2 ** attempt))
                logger.warning(
                    "Connection error fetching %s (attempt %d/%d); retrying in %.1fs",
                    url,
                    attempt + 1,
                    self.max_retries,
                    backoff,
                )
                time.sleep(backoff)
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                logger.warning(f"HTTP error {status} fetching {url}")
                return None
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                return None
        return None

    def _fetch_raw_markets(self) -> List[Dict]:
        """
        Fetch raw market dicts from Polymarket, handling pagination.

        Gamma is the preferred source for market discovery because it exposes
        indexed volume/liquidity and uses `limit` + `offset` pagination.
        Falls back to the CLOB listing endpoint if Gamma returns nothing.
        """
        if self.market_provider is not None:
            try:
                try:
                    markets = self.market_provider.fetch_markets(
                        limit=self.max_markets_per_scan,
                        active_only=True,
                        as_of_ms=self.replay_as_of_ms,
                    )
                except TypeError:
                    markets = self.market_provider.fetch_markets(
                        limit=self.max_markets_per_scan,
                        active_only=True,
                    )
                if markets:
                    if self.source_registry:
                        self.source_registry.mark_up(
                            "polymarket",
                            reason="provider fetch ok",
                            metadata={"provider": type(self.market_provider).__name__},
                        )
                    return list(markets)
                if self.source_registry:
                    self.source_registry.mark_degraded(
                        "polymarket",
                        reason="provider returned no markets",
                        metadata={"provider": type(self.market_provider).__name__},
                    )
            except Exception as exc:
                if self.source_registry:
                    self.source_registry.mark_down(
                        "polymarket",
                        reason=f"provider fetch failed: {exc}",
                        metadata={"provider": type(self.market_provider).__name__},
                    )
                logger.warning("Polymarket provider fetch failed: %s", exc)
                return []

        raw_markets = []

        # --- Primary: Gamma paginated endpoint ---
        try:
            offset = 0
            pages_fetched = 0
            max_pages = max(1, min(10, (self.max_markets_per_scan + 99) // 100 + 1))

            while pages_fetched < max_pages:
                url = (
                    f"{self.GAMMA_API}/markets?limit=100&offset={offset}"
                    "&active=true&closed=false&order=volume&ascending=false"
                )
                resp = self._fetch_json(url)
                if not resp:
                    break

                if isinstance(resp, list):
                    page_data = resp
                elif isinstance(resp, dict):
                    page_data = resp.get("data", resp.get("markets", []))
                else:
                    break

                if not page_data:
                    break

                raw_markets.extend(page_data)
                pages_fetched += 1
                if len(page_data) < 100:
                    break
                offset += 100

        except Exception as e:
            logger.debug(f"Gamma markets fetch error: {e}")

        # --- Fallback: CLOB paginated endpoint ---
        if not raw_markets:
            try:
                next_cursor = ""
                pages_fetched = 0
                max_pages = max(1, min(10, (self.max_markets_per_scan + 99) // 100 + 1))

                while pages_fetched < max_pages:
                    url = f"{self.CLOB_API}/markets?limit=100&active=true"
                    if next_cursor:
                        url += f"&next_cursor={next_cursor}"

                    resp = self._fetch_json(url)
                    if not resp:
                        break

                    if isinstance(resp, dict):
                        page_data = resp.get("data", resp.get("markets", []))
                        raw_markets.extend(page_data)
                        next_cursor = resp.get("next_cursor", "")
                        pages_fetched += 1
                        if not next_cursor or next_cursor == "LTE=":
                            break
                    elif isinstance(resp, list):
                        raw_markets.extend(resp)
                        break
                    else:
                        break
            except Exception as e:
                logger.debug(f"CLOB markets fetch error: {e}")

        return raw_markets
    def scan_markets(self, force_refresh: bool = False) -> List[PolymarketMarket]:
        """
        Fetch active markets from Polymarket.
        Filter for crypto-related markets and major macro markets.

        Returns:
            List of enriched market data (title, volume, liquidity, current price, 24h change)
        """
        logger.debug("Scanning Polymarket markets...")

        now = time.time()
        if (
            not force_refresh
            and self._last_scan_attempt_ts
            and (now - self._last_scan_attempt_ts) < self._scan_cache_window_s
        ):
            return list(self._market_cache.values())

        self._last_scan_attempt_ts = now
        self._last_scan_time = datetime.now(timezone.utc)
        raw_markets = self._fetch_raw_markets()
        self._last_raw_market_count = len(raw_markets)

        if not raw_markets:
            logger.warning("Failed to fetch markets from Polymarket")
            self._last_filtered_market_count = 0
            self._markets_tracked = 0
            self._crypto_markets_found = 0
            self._market_cache = {}
            if self.source_registry:
                self.source_registry.mark_down(
                    "polymarket",
                    reason="raw market fetch failed",
                    metadata={"raw_markets": 0},
                )
            return []

        enriched = []
        crypto_count = 0

        for market_data in raw_markets:
            try:
                # Parse market structure — handle both CLOB and Gamma API field names
                market_id = (
                    market_data.get("condition_id")
                    or market_data.get("conditionId")
                    or market_data.get("id")
                    or ""
                )
                title = (
                    market_data.get("question")
                    or market_data.get("title")
                    or market_data.get("name")
                    or ""
                )
                description = market_data.get("description", "")

                if not market_id or not title:
                    continue

                # Get outcomes and current prices
                outcomes = market_data.get("outcomes", [])
                if isinstance(outcomes, str):
                    try:
                        outcomes = json.loads(outcomes)
                    except json.JSONDecodeError:
                        outcomes = [outcomes] if outcomes else []

                tokens = market_data.get("tokens", [])
                if isinstance(tokens, str):
                    try:
                        tokens = json.loads(tokens)
                    except json.JSONDecodeError:
                        tokens = []

                # Extract current prices from tokens
                prices = []
                token_ids = []
                for token in tokens:
                    token_ids.append(token.get("token_id", ""))
                    # Try to get price from token, fallback to 0
                    price = token.get("price")
                    if price is not None:
                        prices.append(float(price))
                    else:
                        prices.append(0.0)

                if not prices:
                    outcome_prices = (
                        market_data.get("outcomePrices")
                        or market_data.get("outcome_prices")
                        or []
                    )
                    if isinstance(outcome_prices, str):
                        try:
                            outcome_prices = json.loads(outcome_prices)
                        except json.JSONDecodeError:
                            outcome_prices = []
                    if isinstance(outcome_prices, list):
                        for price in outcome_prices:
                            try:
                                prices.append(float(price))
                            except (ValueError, TypeError):
                                prices.append(0.0)

                if not token_ids:
                    clob_token_ids = (
                        market_data.get("clobTokenIds")
                        or market_data.get("clob_token_ids")
                        or []
                    )
                    if isinstance(clob_token_ids, str):
                        try:
                            clob_token_ids = json.loads(clob_token_ids)
                        except json.JSONDecodeError:
                            clob_token_ids = []
                    if isinstance(clob_token_ids, list):
                        token_ids = [str(token_id) for token_id in clob_token_ids if token_id]

                # Get volume and liquidity ? field names differ between CLOB and Gamma API
                volume_24h = (
                    market_data.get("volume_24hr")
                    or market_data.get("volume_24h")
                    or market_data.get("volume24hr")
                    or market_data.get("volume24hrClob")
                    or market_data.get("volumeClob")
                    or market_data.get("volumeNum")
                    or market_data.get("volume")
                    or 0
                )
                liquidity = (
                    market_data.get("liquidity")
                    or market_data.get("liquidityNum")
                    or market_data.get("liquidityClob")
                    or market_data.get("total_liquidity")
                    or 0
                )
                try:
                    volume_24h = float(volume_24h)
                    liquidity = float(liquidity)
                except (ValueError, TypeError):
                    volume_24h = 0.0
                    liquidity = 0.0

                last_traded = market_data.get("last_traded", datetime.now(timezone.utc).isoformat())

                # Determine category
                category = market_data.get("category", "general")

                # Check if crypto-related
                is_crypto = self._is_crypto_market(title, description, category)

                market = PolymarketMarket(
                    token_id=token_ids[0] if token_ids else market_id,
                    market_id=market_id,
                    title=title,
                    description=description,
                    outcomes=outcomes,
                    current_prices=prices,
                    volume_24h=volume_24h,
                    liquidity=liquidity,
                    last_traded=last_traded,
                    category=category,
                )

                enriched.append(market)
                if is_crypto:
                    crypto_count += 1

            except Exception as e:
                logger.debug(f"Error parsing market: {e}")
                continue

        # Prefer crypto-relevant markets and only fall back to relaxed filtering
        # when strict thresholds would otherwise leave the forecaster blind.
        crypto_markets = [
            market
            for market in enriched
            if self._is_crypto_market(market.title, market.description, market.category)
        ]
        filtered = [
            market
            for market in crypto_markets
            if market.volume_24h >= self.min_volume_threshold
            and market.liquidity >= self.min_liquidity_threshold
        ]
        filtered.sort(
            key=lambda market: (market.volume_24h, market.liquidity),
            reverse=True,
        )
        capped = filtered[: self.max_markets_per_scan]
        fallback_used = False
        if not capped and crypto_markets:
            fallback_used = True
            relaxed = [
                market
                for market in crypto_markets
                if market.volume_24h > 0 or market.liquidity > 0
            ]
            relaxed.sort(
                key=lambda market: (market.volume_24h, market.liquidity),
                reverse=True,
            )
            capped = relaxed[: self.max_markets_per_scan]
            if capped:
                logger.info(
                    "Polymarket strict filters produced 0 markets; falling back to %d active crypto markets "
                    "(strict min_volume=$%.0f, min_liquidity=$%.0f)",
                    len(capped),
                    self.min_volume_threshold,
                    self.min_liquidity_threshold,
                )
        self._last_filtered_market_count = len(filtered)

        self._markets_tracked = len(capped)
        self._crypto_markets_found = len(capped)

        logger.info(
            "Scanned %d raw markets, parsed %d, liquidity-filtered %d, capped to %d "
            "(crypto parsed=%d, crypto in cap=%d, min_volume=$%.0f, min_liquidity=$%.0f, fallback=%s)",
            len(raw_markets),
            len(enriched),
            len(filtered),
            len(capped),
            len(crypto_markets),
            self._crypto_markets_found,
            self.min_volume_threshold,
            self.min_liquidity_threshold,
            fallback_used,
        )

        # Cache only markets that survive liquidity and top-N filters.
        self._market_cache = {market.market_id: market for market in capped}
        if self.source_registry:
            if capped:
                state_method = self.source_registry.mark_degraded if fallback_used else self.source_registry.mark_up
                state_reason = (
                    f"fallback active with {len(capped)} markets"
                    if fallback_used
                    else f"{len(capped)} markets active"
                )
                state_method(
                    "polymarket",
                    reason=state_reason,
                    metadata={
                        "raw_markets": len(raw_markets),
                        "filtered_markets": len(filtered),
                        "tracked_markets": len(capped),
                        "fallback_used": fallback_used,
                    },
                )
            else:
                self.source_registry.mark_degraded(
                    "polymarket",
                    reason="no usable crypto markets after filtering",
                    metadata={
                        "raw_markets": len(raw_markets),
                        "filtered_markets": len(filtered),
                        "tracked_markets": 0,
                    },
                )

        return capped

    def _is_crypto_market(self, title: str, description: str, category: str) -> bool:
        """Check if market is crypto-related."""
        text = f"{title} {description} {category}".lower()

        # Check for crypto keywords
        for coin, keywords in self.CRYPTO_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return True

        # Check for major macro events that affect crypto
        macro_keywords = ["fed", "rate", "inflation", "election", "stimulus",
                         "recession", "gdp", "jobs"]
        for keyword in macro_keywords:
            if keyword in text:
                return True

        return False

    def detect_odds_movements(self, markets: List[PolymarketMarket]) -> List[OddsMovement]:
        """
        Compare current prices to cached prices.
        Detect sharp movements (>5% in 1h, >10% in 24h).
        Flag "smart money" patterns: large volume + price movement.

        Returns:
            List of movements with direction and magnitude
        """
        movements = []

        for market in markets:
            try:
                current_prices = market.current_prices
                if not current_prices:
                    continue

                # Get cached prices if available
                if market.token_id in self._price_cache:
                    cache_age = time.time() - self._price_cache_time.get(market.token_id, 0)

                    if cache_age < 600:  # Compare against prices cached within last 10 minutes
                        old_prices = self._price_cache[market.token_id]

                        # Calculate changes for each outcome
                        for outcome_idx, (old_price, new_price) in enumerate(zip(old_prices, current_prices)):
                            if old_price == 0:
                                continue

                            change = (new_price - old_price) / old_price

                            # Check 1h movement threshold
                            if abs(change) > self.odds_movement_threshold_1h:
                                direction = "up" if change > 0 else "down"

                                # Calculate smart money score
                                smart_money_score = self._calculate_smart_money_score(
                                    market, change, outcome_idx
                                )

                                movement = OddsMovement(
                                    market_id=market.market_id,
                                    title=market.title,
                                    direction=direction,
                                    magnitude=abs(change),
                                    timeframe="1h",
                                    current_probability=new_price,
                                    volume_move=market.volume_24h,
                                    smart_money_score=smart_money_score,
                                )
                                movements.append(movement)

                # Update cache
                self._price_cache[market.token_id] = current_prices
                self._price_cache_time[market.token_id] = time.time()

            except Exception as e:
                logger.debug(f"Error detecting movement in {market.title}: {e}")
                continue

        self._movements_detected = len(movements)
        cached_count = len(self._price_cache)
        compared_count = sum(1 for m in markets if m.token_id in self._price_cache)
        logger.info(
            f"Detected {len(movements)} odds movements "
            f"(cached={cached_count}, compared={compared_count}/{len(markets)}, "
            f"threshold={self.odds_movement_threshold_1h:.3f})"
        )

        return movements

    def _calculate_smart_money_score(self, market: PolymarketMarket,
                                     price_change: float, outcome_idx: int) -> float:
        """
        Calculate score for smart money pattern detection.
        0-1, where 1 = high confidence smart money (large volume + movement).
        """
        # Base score from magnitude of move
        magnitude_score = min(abs(price_change) / 0.20, 1.0)  # Normalize to 20% move

        # Volume boost: more volume = more likely smart money
        volume_score = min(market.volume_24h / self.smart_money_volume_threshold, 1.0)

        # Liquidity check: need adequate liquidity for large moves
        liquidity_score = min(market.liquidity / (self.smart_money_volume_threshold * 2), 1.0)

        # Composite (weighted average)
        composite = (
            magnitude_score * 0.5 +
            volume_score * 0.3 +
            liquidity_score * 0.2
        )

        return round(composite, 3)

    def correlate_with_hyperliquid(self, polymarket_signals: List[Dict],
                                   hl_regime: Optional[Dict] = None) -> List[Dict]:
        """
        Cross-reference Polymarket signals with Hyperliquid data.

        Examples:
          - "BTC above $100k by March" market moving up → bullish BTC signal
          - "Fed rate cut" market spiking → risk-on → bullish crypto

        Returns:
            Correlated signals with confidence boost
        """
        if hl_regime is None:
            hl_regime = {}

        enhanced = []

        for signal in polymarket_signals:
            # Map market sentiment to crypto asset
            coin = self._map_market_to_coin(signal.get("polymarket_market", ""))

            if not coin:
                continue

            # Check event-to-crypto mapping
            sentiment_boost = self._get_event_sentiment_boost(
                signal.get("polymarket_market", ""),
                hl_regime.get("overall_regime", "neutral")
            )

            # Boost confidence if correlated with regime
            confidence = signal.get("confidence", 0.5)
            if sentiment_boost != 0:
                confidence = min(confidence + 0.15, 1.0) if sentiment_boost > 0 else max(confidence - 0.1, 0.0)

            # Enhance signal
            enhanced_signal = {
                **signal,
                "coin": coin,
                "confidence": confidence,
                "correlation_with_hl": f"Regime: {hl_regime.get('overall_regime', 'neutral')}, "
                                       f"Boost: {'+' if sentiment_boost > 0 else ''}{sentiment_boost:.2f}",
            }

            enhanced.append(enhanced_signal)

        return enhanced

    def _map_market_to_coin(self, market_title: str) -> str:
        """Map Polymarket title to crypto coin symbol."""
        text = market_title.lower()

        for coin, keywords in self.CRYPTO_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return coin.upper()

        return ""

    @staticmethod
    def _market_bias(title: str) -> str:
        """Classify whether a market title is bullish or bearish for the asset."""
        text = str(title or "").lower()
        bullish_keywords = ["above", "bull", "rally", "surge", "pump"]
        bearish_keywords = ["below", "bear", "crash", "down", "dump"]
        is_bullish = any(keyword in text for keyword in bullish_keywords)
        is_bearish = any(keyword in text for keyword in bearish_keywords)
        if is_bearish and not is_bullish:
            return "bearish"
        if is_bullish and not is_bearish:
            return "bullish"
        return "neutral"

    def _get_event_sentiment_boost(self, market_title: str, hl_regime: str) -> float:
        """
        Determine sentiment boost from event-to-crypto mapping.
        Returns float: positive for bullish, negative for bearish.
        """
        text = market_title.lower()

        for event, (sentiment, weight) in self.EVENT_SENTIMENT_MAP.items():
            if event in text:
                if sentiment == "risk_on":
                    return weight  # Positive boost for risk-on
                elif sentiment == "risk_off":
                    return -weight  # Negative boost for risk-off
                else:
                    return 0.0  # Neutral

        return 0.0

    def generate_signals(self, hl_regime: Optional[Dict] = None) -> List[Dict]:
        """
        Main entry point called by the bot loop.
        Calls scan_markets → detect_odds_movements → correlate_with_hyperliquid.

        Returns:
            Signals in format compatible with the bot's signal pipeline:
            {
                "source": "polymarket",
                "coin": str,
                "side": "long" | "short",
                "confidence": float,
                "reason": str,
                "polymarket_market": str,
                "polymarket_probability": float,
                "polymarket_volume_24h": float,
                "correlation_with_hl": str,
            }
        """
        self._scan_count += 1
        logger.info(f"Generating Polymarket signals (scan #{self._scan_count})...")

        # Step 1: Scan markets
        markets = self.scan_markets()
        if not markets:
            if self._last_raw_market_count > 0:
                logger.info(
                    "No Polymarket markets passed filters "
                    "(raw=%d, min_volume=$%.0f, min_liquidity=$%.0f)",
                    self._last_raw_market_count,
                    self.min_volume_threshold,
                    self.min_liquidity_threshold,
                )
            else:
                logger.info("No Polymarket markets available from upstream scan")
            return []

        # Step 2: Detect odds movements
        movements = self.detect_odds_movements(markets)
        if not movements:
            logger.info("No significant odds movements detected")
            return []

        # Step 3: Generate signals from movements
        raw_signals = []
        for movement in movements:
            # Find the market
            market = self._market_cache.get(movement.market_id)
            if not market:
                continue

            # Skip low confidence smart money signals
            if movement.smart_money_score < 0.15:
                continue

            # Movement direction is the odds move, not the underlying asset move.
            market_bias = self._market_bias(movement.title)
            is_bullish_move = movement.direction == "up"
            if market_bias == "bearish":
                is_bullish_move = not is_bullish_move
            signal_side = "long" if is_bullish_move else "short"

            # Map to crypto coin
            coin = self._map_market_to_coin(movement.title)
            if not coin:
                continue

            # Create raw signal
            reason = (f"Polymarket odds movement: {movement.title} "
                     f"{movement.direction} {movement.magnitude*100:.1f}% ({movement.timeframe}), "
                     f"smart money score: {movement.smart_money_score:.2f}")

            signal = {
                "source": "polymarket",
                "coin": coin,
                "side": signal_side,
                "confidence": min(movement.smart_money_score + 0.2, 1.0),
                "reason": reason,
                "polymarket_market": movement.title,
                "polymarket_probability": movement.current_probability,
                "polymarket_volume_24h": movement.volume_move,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            raw_signals.append(signal)

        # Step 4: Correlate with Hyperliquid regime
        final_signals = self.correlate_with_hyperliquid(raw_signals, hl_regime)

        self._signals_generated = len(final_signals)
        logger.info(f"Generated {len(final_signals)} Polymarket signals")

        return final_signals

    def get_market_sentiment(self) -> Dict:
        """
        Aggregate crypto-related market probabilities into overall sentiment.

        Returns:
            {
                "sentiment": "bullish" | "bearish" | "neutral",
                "confidence": float (0-1),
                "markets_analyzed": int,
                "bullish_probability": float,
                "bearish_probability": float,
            }
        """
        if (
            not self._market_cache
            and (
                not self._last_scan_attempt_ts
                or (time.time() - self._last_scan_attempt_ts) >= self._scan_cache_window_s
            )
        ):
            self.scan_markets()

        # Find crypto-relevant markets with price data
        crypto_markets = [m for m in self._market_cache.values()
                         if self._is_crypto_market(m.title, m.description, m.category)]

        if not crypto_markets:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "markets_analyzed": 0,
                "bullish_probability": 0.5,
                "bearish_probability": 0.5,
            }

        # Aggregate probabilities
        # For BTC/ETH markets: higher price = bullish
        bullish_probs = []
        bearish_probs = []

        for market in crypto_markets:
            if not market.current_prices:
                continue

            # Heuristic: if most outcomes are bullish (e.g., "BTC above X"),
            # the highest price outcome is bullish
            market_bias = self._market_bias(market.title)

            # Take probability of highest-priced outcome
            if market.current_prices:
                max_idx = market.current_prices.index(max(market.current_prices))
                prob = market.current_prices[max_idx]

                if market_bias == "bullish":
                    bullish_probs.append(prob)
                elif market_bias == "bearish":
                    bearish_probs.append(prob)
                else:
                    # Neutral market — use 50/50
                    bullish_probs.append(0.5)
                    bearish_probs.append(0.5)

        # Aggregate
        avg_bullish = sum(bullish_probs) / len(bullish_probs) if bullish_probs else 0.5
        avg_bearish = sum(bearish_probs) / len(bearish_probs) if bearish_probs else 0.5

        # Determine overall sentiment
        if avg_bullish > avg_bearish + 0.1:
            sentiment = "bullish"
            confidence = avg_bullish - 0.5  # 0-0.5 range
        elif avg_bearish > avg_bullish + 0.1:
            sentiment = "bearish"
            confidence = avg_bearish - 0.5  # 0-0.5 range
        else:
            sentiment = "neutral"
            confidence = 0.0

        return {
            "sentiment": sentiment,
            "confidence": round(min(confidence, 0.5), 3),
            "markets_analyzed": len(crypto_markets),
            "bullish_probability": round(avg_bullish, 3),
            "bearish_probability": round(avg_bearish, 3),
        }

    def get_stats(self) -> Dict:
        """Return statistics for dashboard and monitoring."""
        return {
            "scan_count": self._scan_count,
            "last_scan_time": self._last_scan_time.isoformat() if self._last_scan_time else None,
            "signals_generated": self._signals_generated,
            "markets_tracked": self._markets_tracked,
            "crypto_markets_found": self._crypto_markets_found,
            "movements_detected": self._movements_detected,
            "cache_size": len(self._market_cache),
            "rate_limit": f"{self.rate_limit_delay}s",
            "cache_ttl": f"{self.cache_ttl_minutes}min",
        }


# ─── Standalone CLI for testing ───────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    print("\n=== Polymarket Scanner Test ===\n")

    scanner = PolymarketScanner()

    # Test 1: Scan markets
    print("1. Scanning markets...")
    markets = scanner.scan_markets()
    print(f"   Found {len(markets)} markets, {scanner._crypto_markets_found} crypto-related")

    # Test 2: Detect movements
    print("\n2. Detecting odds movements...")
    movements = scanner.detect_odds_movements(markets)
    print(f"   Detected {len(movements)} movements")
    for m in movements[:3]:
        print(f"   - {m.title}: {m.direction} {m.magnitude*100:.1f}% ({m.smart_money_score:.2f})")

    # Test 3: Generate signals
    print("\n3. Generating signals...")
    signals = scanner.generate_signals()
    print(f"   Generated {len(signals)} signals")
    for s in signals[:3]:
        print(f"   - {s['coin']}: {s['side']} (conf: {s['confidence']:.2f})")

    # Test 4: Market sentiment
    print("\n4. Computing market sentiment...")
    sentiment = scanner.get_market_sentiment()
    print(f"   Sentiment: {sentiment['sentiment']} "
          f"(bullish: {sentiment['bullish_probability']:.2f}, "
          f"bearish: {sentiment['bearish_probability']:.2f})")

    # Test 5: Stats
    print("\n5. Scanner stats:")
    stats = scanner.get_stats()
    for key, val in stats.items():
        print(f"   {key}: {val}")

    print("\n=== Test Complete ===\n")
