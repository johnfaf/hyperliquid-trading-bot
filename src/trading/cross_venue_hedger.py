"""
Cross-Venue Hedging Module
===========================
Automatically places reduce-only hedging orders on Kraken Futures (default),
Binance, or Bybit when crash regime is detected by the regime forecaster.

This module monitors market regime predictions and dynamically hedges
open positions on alternative venues to reduce portfolio exposure during
crash scenarios (regime="crash" with confidence > threshold).

Features:
  - Crash-triggered hedging: Places reduce-only orders on hedging venues
  - Regime-aware closure: Closes hedges when regime returns to neutral/bullish
  - Dry-run by default: Logs actions without executing (controlled via dry_run flag)
  - Venue abstraction: Kraken Futures (default), Binance Futures, Bybit Perp v5
  - Rate limiting: Built-in delays to avoid exchange rate limits
  - Environment-based auth: API keys loaded from env vars

Environment Variables (all optional when dry_run=True):
  - KRAKEN_FUTURES_API_KEY:    Kraken Futures (futures.kraken.com) API key
  - KRAKEN_FUTURES_API_SECRET: Kraken Futures API secret (base64-encoded)
  - BINANCE_API_KEY / BINANCE_API_SECRET: Binance Futures
  - BYBIT_API_KEY / BYBIT_API_SECRET: Bybit v5

Configuration:
  config = {
      "dry_run": True,             # Default: no live execution
      "hedge_ratio": 0.5,          # Default: hedge 50% of open position
      "crash_confidence": 0.5,     # Confidence threshold for crash detection
      "kraken_enabled": True,      # Use Kraken Futures (default)
      "binance_enabled": False,    # Use Binance for hedging
      "bybit_enabled": False,      # Use Bybit for hedging
      "rate_limit_ms": 100,        # Delay between API calls (ms)
  }
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from typing import Dict, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class HedgeVenue(Enum):
    """Supported hedging venues."""
    KRAKEN = "kraken"
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
                - kraken_enabled (bool): Enable Kraken Futures hedging (default: True)
                - binance_enabled (bool): Enable Binance hedging (default: False)
                - bybit_enabled (bool): Enable Bybit hedging (default: False)
                - kraken_symbol_template (str): Symbol format, default "PF_{COIN}USD"
                - rate_limit_ms (int): Delay between API calls in ms (default: 100)
        """
        cfg = config or {}

        self.dry_run = cfg.get("dry_run", True)
        self.hedge_ratio = cfg.get("hedge_ratio", 0.5)
        self.crash_confidence = cfg.get("crash_confidence", 0.5)
        self.kraken_enabled = cfg.get("kraken_enabled", True)
        self.binance_enabled = cfg.get("binance_enabled", False)
        self.bybit_enabled = cfg.get("bybit_enabled", False)
        self.rate_limit_ms = cfg.get("rate_limit_ms", 100)
        self.allow_unimplemented_live = bool(cfg.get("allow_unimplemented_live", False))
        # PF_<COIN>USD = USD-collateralized perp on Kraken Futures (post-2022).
        # Override to "PI_{COIN}USD" if you specifically want inverse contracts.
        self.kraken_symbol_template = str(
            cfg.get("kraken_symbol_template", "PF_{COIN}USD")
        )
        self.kraken_order_type = str(cfg.get("kraken_order_type", "mkt")).lower()
        self.kraken_api_url = str(
            cfg.get("kraken_api_url", "https://futures.kraken.com")
        ).rstrip("/")
        self.kraken_request_timeout_s = float(cfg.get("kraken_request_timeout_s", 5.0))

        # API credentials (loaded from environment)
        self.kraken_api_key = os.environ.get("KRAKEN_FUTURES_API_KEY", "")
        self.kraken_api_secret = os.environ.get("KRAKEN_FUTURES_API_SECRET", "")
        self.binance_api_key = os.environ.get("BINANCE_API_KEY", "")
        self.binance_api_secret = os.environ.get("BINANCE_API_SECRET", "")
        self.bybit_api_key = os.environ.get("BYBIT_API_KEY", "")
        self.bybit_api_secret = os.environ.get("BYBIT_API_SECRET", "")

        # Active hedges tracking: {venue: {coin: {"side": str, "size": float, "ts": float}}}
        self._active_hedges: Dict[str, Dict] = {
            HedgeVenue.KRAKEN.value: {},
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

        # Binance and Bybit live execution are still NOT implemented; refuse to
        # silently no-op. Kraken Futures live execution IS implemented (signed
        # v3 sendorder), but require credentials before going live.
        if not self.dry_run and not self.allow_unimplemented_live and (self.binance_enabled or self.bybit_enabled):
            logger.error(
                "Binance/Bybit live hedge execution is not implemented; disabling those "
                "venues. Use kraken_enabled=True for real hedges."
            )
            self.binance_enabled = False
            self.bybit_enabled = False
        if not self.dry_run and self.kraken_enabled and not (
            self.kraken_api_key and self.kraken_api_secret
        ):
            logger.error(
                "Kraken Futures hedger requested LIVE but KRAKEN_FUTURES_API_KEY / "
                "KRAKEN_FUTURES_API_SECRET are not set. Disabling kraken hedger."
            )
            self.kraken_enabled = False

        venues = []
        if self.kraken_enabled:
            venues.append("Kraken-Futures")
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
                pos_iter = [
                    (str(p.get("coin", "") or "").strip().upper(), p)
                    for p in open_positions
                    if str(p.get("coin", "") or "").strip()
                ]
            else:
                pos_iter = [
                    (str(coin or "").strip().upper(), position)
                    for coin, position in open_positions.items()
                    if str(coin or "").strip()
                ]
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
            for venue in [
                HedgeVenue.KRAKEN.value,
                HedgeVenue.BINANCE.value,
                HedgeVenue.BYBIT.value,
            ]:
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

        if self.kraken_enabled:
            if self._place_kraken_hedge(coin, hedge_side, hedge_size):
                success = True
                self._active_hedges[HedgeVenue.KRAKEN.value][coin] = {
                    "side": hedge_side,
                    "size": hedge_size,
                    "ts": time.time(),
                }
                logger.debug(f"Kraken hedge placed for {coin}: {hedge_side} {hedge_size}")

        self._rate_limit()

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

    def _kraken_symbol(self, coin: str) -> str:
        """Map an internal coin symbol to Kraken Futures symbol.

        Kraken uses XBT for Bitcoin; everything else maps directly.
        Template comes from cfg.kraken_symbol_template (default PF_{COIN}USD).
        """
        c = coin.upper().strip()
        if c == "BTC":
            c = "XBT"
        return self.kraken_symbol_template.format(COIN=c)

    def _kraken_sign(self, post_data: str, nonce: str, endpoint_path: str) -> str:
        """Compute Kraken Futures Authent header.

        spec: Authent = base64(HMAC_SHA512(
            base64_decode(api_secret),
            SHA256(postData + nonce + endpointPath)
        ))
        """
        message = (post_data + nonce + endpoint_path).encode("utf-8")
        sha = hashlib.sha256(message).digest()
        try:
            secret_decoded = base64.b64decode(self.kraken_api_secret)
        except Exception as exc:
            raise ValueError(f"KRAKEN_FUTURES_API_SECRET is not valid base64: {exc}") from exc
        mac = hmac.new(secret_decoded, sha, hashlib.sha512).digest()
        return base64.b64encode(mac).decode("ascii")

    def _place_kraken_hedge(self, coin: str, side: str, size: float) -> bool:
        """Place a reduce-only order on Kraken Futures.

        Endpoint: POST https://futures.kraken.com/derivatives/api/v3/sendorder
        Auth: APIKey + Authent (HMAC-SHA512 over SHA256(post + nonce + path)).

        Args:
            coin: Internal coin symbol (e.g. "BTC", "ETH").
            side: "BUY" or "SELL" (we pass through as lowercase).
            size: Order quantity in contracts.

        Returns:
            True on accepted order or on dry-run; False on any error path.
            Raises NotImplementedError ONLY if explicitly disabled later.
        """
        try:
            symbol = self._kraken_symbol(coin)
            kraken_side = "buy" if side.upper() == "BUY" else "sell"

            if self.dry_run:
                logger.info(
                    f"[DRY-RUN] Kraken hedge order: {kraken_side} {size} {symbol} "
                    f"(reduce-only, type={self.kraken_order_type})"
                )
                return True

            if not (self.kraken_api_key and self.kraken_api_secret):
                logger.error("Kraken hedge requested without API credentials")
                return False

            endpoint_path = "/derivatives/api/v3/sendorder"
            params = {
                "orderType": self.kraken_order_type,
                "symbol": symbol,
                "side": kraken_side,
                "size": str(size),
                "reduceOnly": "true",
            }
            post_data = urllib.parse.urlencode(params)
            # Nonce: monotonically-increasing string. Microsecond ts is fine.
            nonce = str(int(time.time() * 1000_000))
            authent = self._kraken_sign(post_data, nonce, endpoint_path)

            url = f"{self.kraken_api_url}{endpoint_path}"
            req = urllib.request.Request(
                url,
                data=post_data.encode("utf-8"),
                method="POST",
                headers={
                    "APIKey": self.kraken_api_key,
                    "Authent": authent,
                    "Nonce": nonce,
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=self.kraken_request_timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                logger.error("Kraken hedge: non-JSON response: %s", body[:200])
                return False
            result = str(payload.get("result", "")).lower()
            if result != "success":
                logger.error(
                    "Kraken hedge rejected for %s: result=%s, error=%s",
                    coin, result, payload.get("error") or payload.get("errors"),
                )
                return False
            order_status = (payload.get("sendStatus") or {}).get("status", "")
            order_id = (payload.get("sendStatus") or {}).get("order_id", "")
            logger.info(
                "Kraken hedge placed %s %s %s (status=%s, oid=%s)",
                kraken_side, size, symbol, order_status, order_id,
            )
            # Per Kraken docs, status="placed" or "fullyExecuted" both indicate
            # the order was accepted; anything else is a soft failure.
            return order_status in ("placed", "fullyExecuted", "partiallyFilled")
        except Exception as e:
            logger.error(f"Failed to place Kraken hedge for {coin}: {e}")
            return False

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

            if self.dry_run:
                logger.info(
                    f"[DRY-RUN] Binance hedge order: {side} {size} {symbol} "
                    f"(reduce-only)"
                )
                return True

            # Live execution is NOT implemented.  Always raise so callers
            # can never silently believe a hedge was placed.
            raise NotImplementedError(
                f"Binance live hedge execution is not implemented for {coin}. "
                "Remove this venue from config or use dry_run=True."
            )

        except NotImplementedError:
            raise  # Never swallow this
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

            if self.dry_run:
                logger.info(
                    f"[DRY-RUN] Bybit hedge order: {bybit_side} {size} {symbol} "
                    f"(reduce-only)"
                )
                return True

            # Live execution is NOT implemented.  Always raise so callers
            # can never silently believe a hedge was placed.
            raise NotImplementedError(
                f"Bybit live hedge execution is not implemented for {coin}. "
                "Remove this venue from config or use dry_run=True."
            )

        except NotImplementedError:
            raise  # Never swallow this
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
        try:
            if venue == HedgeVenue.KRAKEN.value:
                success = self._place_kraken_hedge(coin, close_side, original_size)
            elif venue == HedgeVenue.BINANCE.value:
                success = self._place_binance_hedge(coin, close_side, original_size)
            elif venue == HedgeVenue.BYBIT.value:
                success = self._place_bybit_hedge(coin, close_side, original_size)
        except NotImplementedError:
            logger.warning(
                "_close_hedge(%s, %s): live execution not implemented -- removing stale tracking entry",
                coin, venue,
            )
            # Remove tracking regardless: if execution isn't implemented,
            # keeping the entry would block future hedge attempts.
            self._active_hedges.get(venue, {}).pop(coin, None)
            return False
        except Exception as exc:
            logger.error("_close_hedge(%s, %s) failed: %s", coin, venue, exc)
            return False

        if success:
            del self._active_hedges[venue][coin]
            logger.debug(f"Hedge closed for {coin} on {venue}")
        else:
            # Position may have been liquidated/closed by exchange.
            # Remove from tracking to avoid stale entries.
            logger.warning(
                "_close_hedge(%s, %s): close order failed -- position may have been "
                "liquidated. Removing from tracking to prevent inconsistency.",
                coin, venue,
            )
            self._active_hedges.get(venue, {}).pop(coin, None)

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
        if self.kraken_enabled:
            venues.append("kraken")
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
