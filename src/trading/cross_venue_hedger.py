"""
Cross-Venue Hedging Module
===========================
Automatically places reduce-only hedging orders on Binance/Bybit when
crash regime is detected by the XGBoost regime forecaster.

This module monitors market regime predictions and dynamically hedges
open positions on alternative venues to reduce portfolio exposure during
crash scenarios (regime="crash" with confidence > threshold).

Features:
  - Crash-triggered hedging: Places reduce-only orders on hedging venues
  - Regime-aware closure: Closes hedges when regime returns to neutral/bullish
  - Dry-run by default: Logs actions without executing (controlled via dry_run flag)
  - Venue abstraction: Supports Binance Futures and Bybit Perpetuals (v5 API)
  - Rate limiting: Built-in delays to avoid exchange rate limits
  - Environment-based auth: API keys loaded from env vars (BINANCE_API_KEY, etc.)

Environment Variables (all optional when dry_run=True):
  - BINANCE_API_KEY: Binance Futures API key
  - BINANCE_API_SECRET: Binance Futures API secret
  - BYBIT_API_KEY: Bybit v5 API key
  - BYBIT_API_SECRET: Bybit v5 API secret

Configuration:
  config = {
      "dry_run": True,              # Default: no live execution
      "hedge_ratio": 0.5,           # Default: hedge 50% of open position
      "crash_confidence": 0.5,      # Confidence threshold for crash detection
      "binance_enabled": True,      # Use Binance for hedging
      "bybit_enabled": False,       # Use Bybit for hedging
      "rate_limit_ms": 100,         # Delay between API calls (ms)
  }
"""

import logging
import os
import time
from typing import Dict, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class HedgeVenue(Enum):
    """Supported hedging venues."""
    BINANCE = "binance"
    BYBIT = "bybit"


class CrossVenueHedger:
    """
    Manages reduce-only hedging across multiple venues in response to
    regime predictions from XGBoostRegimeForecaster.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize cross-venue hedger.

        Args:
            config: Configuration dict with optional keys:
                - dry_run (bool): If True, log actions without executing (default: True)
                - hedge_ratio (float): Fraction of position to hedge (default: 0.5)
                - crash_confidence (float): Confidence threshold for crash (default: 0.5)
                - binance_enabled (bool): Enable Binance hedging (default: True)
                - bybit_enabled (bool): Enable Bybit hedging (default: False)
                - rate_limit_ms (int): Delay between API calls in ms (default: 100)
        """
        cfg = config or {}

        self.dry_run = cfg.get("dry_run", True)
        self.hedge_ratio = cfg.get("hedge_ratio", 0.5)
        self.crash_confidence = cfg.get("crash_confidence", 0.5)
        self.binance_enabled = cfg.get("binance_enabled", True)
        self.bybit_enabled = cfg.get("bybit_enabled", False)
        self.rate_limit_ms = cfg.get("rate_limit_ms", 100)
        self.allow_unimplemented_live = bool(cfg.get("allow_unimplemented_live", False))

        # API credentials (loaded from environment)
        self.binance_api_key = os.environ.get("BINANCE_API_KEY", "")
        self.binance_api_secret = os.environ.get("BINANCE_API_SECRET", "")
        self.bybit_api_key = os.environ.get("BYBIT_API_KEY", "")
        self.bybit_api_secret = os.environ.get("BYBIT_API_SECRET", "")

        # Active hedges tracking: {venue: {coin: {"side": str, "size": float, "ts": float}}}
        self._active_hedges: Dict[str, Dict] = {
            HedgeVenue.BINANCE.value: {},
            HedgeVenue.BYBIT.value: {},
        }

        # Statistics
        self._stats = {
            "total_hedges_placed": 0,
            "total_hedges_closed": 0,
            "total_hedge_value": 0.0,
            "last_hedge_ts": 0.0,
        }

        # Rate limiting state
        self._last_api_call_ts = 0.0

        mode_str = "DRY_RUN" if self.dry_run else "LIVE"

        if not self.dry_run and not self.allow_unimplemented_live and (self.binance_enabled or self.bybit_enabled):
            logger.error(
                "Cross-venue hedger live execution is disabled: API signing/execution "
                "is not fully implemented. Forcing hedger to no-op live mode."
            )
            self.binance_enabled = False
            self.bybit_enabled = False

        venues = []
        if self.binance_enabled:
            venues.append("Binance")
        if self.bybit_enabled:
            venues.append("Bybit")
        venues_str = ", ".join(venues) if venues else "none"

        logger.info(
            f"CrossVenueHedger initialized ({mode_str} mode, venues: {venues_str}, "
            f"hedge_ratio={self.hedge_ratio}, crash_confidence_threshold={self.crash_confidence})"
        )

    def check_and_hedge(self, regime_data: Dict, open_positions: Dict) -> Dict:
        """
        Check regime prediction and automatically hedge or close hedges.

        Main entry point for the trading bot loop.

        Args:
            regime_data: Regime prediction dict with keys:
                - regime (str): "crash", "neutral", or "bullish"
                - confidence (float): 0-1 confidence score
            open_positions: Dict of open positions by coin: {coin: {"side": str, "size": float}}

        Returns:
            Dict with hedge actions taken:
                {
                    "regime": str,
                    "action": "hedged"|"closed"|"idle",
                    "hedges_placed": int,
                    "hedges_closed": int,
                    "coins_affected": [str, ...],
                }
        """
        regime = regime_data.get("regime", "neutral")
        confidence = regime_data.get("confidence", 0.0)

        result = {
            "regime": regime,
            "action": "idle",
            "hedges_placed": 0,
            "hedges_closed": 0,
            "coins_affected": [],
        }

        # Check if we should activate crash hedges
        if regime == "crash" and confidence > self.crash_confidence:
            logger.info(
                f"Crash regime detected (confidence={confidence:.3f}). "
                f"Placing hedges on {len(open_positions)} position(s)."
            )
            hedges_placed = 0
            # Accept both dict {coin: pos} and list [{coin, side, size, ...}]
            if isinstance(open_positions, list):
                pos_iter = [(p.get("coin", ""), p) for p in open_positions]
            else:
                pos_iter = open_positions.items()
            for coin, position in pos_iter:
                if self._place_hedges(coin, position):
                    hedges_placed += 1
                    result["coins_affected"].append(coin)

            result["action"] = "hedged"
            result["hedges_placed"] = hedges_placed
            self._stats["total_hedges_placed"] += hedges_placed
            self._stats["last_hedge_ts"] = time.time()

        # Check if we should close existing hedges
        elif regime in ["neutral", "bullish"] and self._has_active_hedges():
            logger.info(
                f"Regime returned to {regime} (confidence={confidence:.3f}). "
                f"Closing {len(self._count_active_hedges())} active hedge(s)."
            )
            hedges_closed = 0
            for venue in [HedgeVenue.BINANCE.value, HedgeVenue.BYBIT.value]:
                for coin in list(self._active_hedges[venue].keys()):
                    if self._close_hedge(coin, venue):
                        hedges_closed += 1
                        result["coins_affected"].append(coin)

            result["action"] = "closed"
            result["hedges_closed"] = hedges_closed
            self._stats["total_hedges_closed"] += hedges_closed

        return result

    def _place_hedges(self, coin: str, position: Dict) -> bool:
        """
        Place reduce-only hedge orders for a single position across enabled venues.

        Args:
            coin: Coin/asset symbol (e.g., "BTC", "ETH")
            position: Position dict with keys:
                - side (str): "long" or "short"
                - size (float): position size in contracts/coins

        Returns:
            True if at least one hedge was placed, False otherwise.
        """
        side = position.get("side", "long").lower()
        size = position.get("size", 0.0)

        if size <= 0:
            return False

        # Hedge side is opposite to position
        hedge_side = "SELL" if side == "long" else "BUY"
        hedge_size = size * self.hedge_ratio

        success = False

        if self.binance_enabled:
            if self._place_binance_hedge(coin, hedge_side, hedge_size):
                success = True
                self._active_hedges[HedgeVenue.BINANCE.value][coin] = {
                    "side": hedge_side,
                    "size": hedge_size,
                    "ts": time.time(),
                }
                logger.debug(f"Binance hedge placed for {coin}: {hedge_side} {hedge_size}")

        self._rate_limit()

        if self.bybit_enabled:
            if self._place_bybit_hedge(coin, hedge_side, hedge_size):
                success = True
                self._active_hedges[HedgeVenue.BYBIT.value][coin] = {
                    "side": hedge_side,
                    "size": hedge_size,
                    "ts": time.time(),
                }
                logger.debug(f"Bybit hedge placed for {coin}: {hedge_side} {hedge_size}")

        return success

    def _place_binance_hedge(self, coin: str, side: str, size: float) -> bool:
        """
        Place a reduce-only order on Binance Futures.

        Endpoint: POST https://fapi.binance.com/fapi/v1/order
        Required params: symbol, side, type, quantity, reduceOnly

        Args:
            coin: Coin symbol (e.g., "BTC")
            side: "BUY" or "SELL"
            size: Order quantity

        Returns:
            True if order placed (or logged in dry-run), False on error.
        """
        try:
            symbol = f"{coin}USDT"

            order_params = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": size,
                "reduceOnly": True,
                "timestamp": int(time.time() * 1000),
            }

            if self.dry_run:
                logger.info(
                    f"[DRY-RUN] Binance hedge order: {side} {size} {symbol} "
                    f"(reduce-only)"
                )
                return True

            if not self.allow_unimplemented_live:
                logger.error(
                    "Binance live hedge blocked for %s: order execution path is not implemented.",
                    coin,
                )
                return False
            logger.warning(
                "allow_unimplemented_live=True but Binance execution still unimplemented. "
                "Returning failure to avoid false hedge accounting."
            )
            return False

        except Exception as e:
            logger.error(f"Failed to place Binance hedge for {coin}: {e}")
            return False

    def _place_bybit_hedge(self, coin: str, side: str, size: float) -> bool:
        """
        Place a reduce-only order on Bybit Perpetuals (v5 API).

        Endpoint: POST https://api.bybit.com/v5/order/create
        Required JSON body: category, symbol, side, orderType, qty, reduceOnly

        Args:
            coin: Coin symbol (e.g., "BTC")
            side: "Buy" or "Sell"
            size: Order quantity

        Returns:
            True if order placed (or logged in dry-run), False on error.
        """
        try:
            symbol = f"{coin}USDT"
            bybit_side = "Buy" if side == "BUY" else "Sell"

            order_body = {
                "category": "linear",
                "symbol": symbol,
                "side": bybit_side,
                "orderType": "Market",
                "qty": str(size),
                "reduceOnly": True,
            }

            if self.dry_run:
                logger.info(
                    f"[DRY-RUN] Bybit hedge order: {bybit_side} {size} {symbol} "
                    f"(reduce-only)"
                )
                return True

            if not self.allow_unimplemented_live:
                logger.error(
                    "Bybit live hedge blocked for %s: order execution path is not implemented.",
                    coin,
                )
                return False
            logger.warning(
                "allow_unimplemented_live=True but Bybit execution still unimplemented. "
                "Returning failure to avoid false hedge accounting."
            )
            return False

        except Exception as e:
            logger.error(f"Failed to place Bybit hedge for {coin}: {e}")
            return False

    def _close_hedge(self, coin: str, venue: str) -> bool:
        """
        Close an existing hedge by placing an opposite-side reduce-only order.

        Args:
            coin: Coin symbol
            venue: "binance" or "bybit"

        Returns:
            True if close order was placed, False otherwise.
        """
        hedges = self._active_hedges.get(venue, {})
        if coin not in hedges:
            return False

        hedge_data = hedges[coin]
        original_side = hedge_data.get("side", "SELL")
        original_size = hedge_data.get("size", 0.0)

        # Close by placing opposite side
        close_side = "BUY" if original_side == "SELL" else "SELL"

        success = False
        if venue == HedgeVenue.BINANCE.value:
            success = self._place_binance_hedge(coin, close_side, original_size)
        elif venue == HedgeVenue.BYBIT.value:
            success = self._place_bybit_hedge(coin, close_side, original_size)

        if success:
            del self._active_hedges[venue][coin]
            logger.debug(f"Hedge closed for {coin} on {venue}")

        return success

    def _has_active_hedges(self) -> bool:
        """Check if there are any active hedges."""
        for venue_hedges in self._active_hedges.values():
            if venue_hedges:
                return True
        return False

    def _count_active_hedges(self) -> Dict[str, int]:
        """Count active hedges per venue."""
        return {venue: len(hedges) for venue, hedges in self._active_hedges.items()}

    def _rate_limit(self) -> None:
        """Apply rate limiting between API calls."""
        elapsed_ms = (time.time() - self._last_api_call_ts) * 1000
        if elapsed_ms < self.rate_limit_ms:
            sleep_ms = self.rate_limit_ms - elapsed_ms
            time.sleep(sleep_ms / 1000.0)
        self._last_api_call_ts = time.time()

    def get_active_hedges(self) -> Dict[str, List[Dict]]:
        """
        Get list of currently active hedges.

        Returns:
            Dict with venue keys, each mapping to list of active hedges:
            {
                "binance": [
                    {"coin": "BTC", "side": "SELL", "size": 0.5, "opened_ts": 1234567.0},
                    ...
                ],
                "bybit": [...],
            }
        """
        result = {}
        for venue, hedges in self._active_hedges.items():
            hedges_list = []
            for coin, data in hedges.items():
                hedges_list.append({
                    "coin": coin,
                    "side": data.get("side", ""),
                    "size": data.get("size", 0.0),
                    "opened_ts": data.get("ts", 0.0),
                })
            result[venue] = hedges_list
        return result

    def get_stats(self) -> Dict:
        """
        Get hedger statistics and performance metrics.

        Returns:
            Dict with keys:
            - total_hedges_placed: int
            - total_hedges_closed: int
            - active_hedges_count: int
            - last_hedge_ts: float (Unix timestamp)
            - dry_run: bool
            - venues_enabled: [str, ...]
        """
        active_count = sum(len(h) for h in self._active_hedges.values())
        venues = []
        if self.binance_enabled:
            venues.append("binance")
        if self.bybit_enabled:
            venues.append("bybit")

        return {
            **self._stats,
            "active_hedges_count": active_count,
            "dry_run": self.dry_run,
            "venues_enabled": venues,
        }
