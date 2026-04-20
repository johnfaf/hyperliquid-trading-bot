"""
Centralized API Manager
=======================
Solves the "thundering herd" problem: every module was independently
hitting the Hyperliquid REST API, causing 429 storms.

This module provides:
1. Token Bucket Rate Limiter — proactive, not reactive
2. TTL Cache — deduplicates identical requests within a time window
3. Jittered Exponential Backoff — prevents synchronized retry bursts
4. WebSocket feed — real-time price/trade/L2 data without REST polling
5. Request priority — critical requests (execution) skip the queue

All modules MUST route requests through this manager instead of
calling requests.post() directly.
"""
import threading
import time
import json
import random
import logging
import hashlib
import requests
from typing import Optional, Dict, Any, Callable
from collections import OrderedDict, deque
from enum import IntEnum

try:
    import websocket as _ws_lib
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.core.env_utils import safe_env_float

logger = logging.getLogger("api_manager")
logger.addHandler(logging.NullHandler())

_WS_GAP_WARN_THRESHOLD_MS_MIN = 5000.0
_WS_GAP_WARN_COOLDOWN_S_MIN = 5.0
_WS_GAP_WARN_COOLDOWN_S_MAX = 900.0


# ─── Priority levels ─────────────────────────────────────────────
class Priority(IntEnum):
    CRITICAL = 0    # Trade execution, position checks
    HIGH = 1        # Copy trader scans, stop loss checks
    NORMAL = 2      # Discovery, scoring, regime detection
    LOW = 3         # Analytics, logging, non-urgent data


# ─── Token Bucket Rate Limiter ───────────────────────────────────
class TokenBucket:
    """
    Proactive rate limiter using the token bucket algorithm.
    Instead of reacting to 429s, we prevent them.

    Hyperliquid limit: ~1200 requests/minute = 20/second.
    We target 15/second to leave headroom.
    """

    def __init__(self, rate: float = 15.0, capacity: int = 30):
        """
        rate: tokens added per second (sustained throughput)
        capacity: max burst size (bucket depth)
        """
        self.rate = rate
        self.capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

        # Adaptive: slow down when we see 429s
        self._penalty_until = 0.0
        self._consecutive_429s = 0

    def acquire(self, priority: Priority = Priority.NORMAL, timeout: float = 60.0) -> bool:
        """
        Block until a token is available, or timeout.
        Higher priority requests get served faster.
        Returns True if acquired, False on timeout.
        """
        deadline = time.monotonic() + timeout

        # During penalty period, only CRITICAL requests go through at full speed
        if priority > Priority.CRITICAL and time.monotonic() < self._penalty_until:
            penalty_wait = min(self._penalty_until - time.monotonic(), 10.0)
            # Add priority-based delay: LOW waits more than NORMAL
            penalty_wait *= (1 + priority * 0.3)
            time.sleep(min(penalty_wait, timeout))

        while time.monotonic() < deadline:
            with self._lock:
                self._refill()
                if self._tokens >= 1:
                    self._tokens -= 1
                    return True

            # Wait a bit before retrying, with small jitter
            time.sleep(0.05 + random.uniform(0, 0.02))

        logger.warning(f"Token bucket timeout after {timeout}s (priority={priority.name})")
        return False

    def report_429(self):
        """Called when we receive a 429 — adaptively reduce throughput."""
        with self._lock:
            self._consecutive_429s += 1
            # Escalating penalty: 5s base + 3s per consecutive 429
            penalty = min(5 + self._consecutive_429s * 3, 60)
            self._penalty_until = time.monotonic() + penalty
            # Drain tokens to force waiting
            self._tokens = 0
            logger.warning(
                f"429 received -- penalty {penalty}s "
                f"(consecutive={self._consecutive_429s})"
            )

    def report_success(self):
        """Called on successful request — slowly recover from penalty."""
        with self._lock:
            if self._consecutive_429s > 0:
                self._consecutive_429s = max(0, self._consecutive_429s - 1)

    def _refill(self):
        """Add tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        # During penalty, refill at half rate
        rate = self.rate
        if self._consecutive_429s > 0:
            rate = max(2.0, self.rate / (1 + self._consecutive_429s * 0.5))
        self._tokens = min(self.capacity, self._tokens + elapsed * rate)

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "tokens_available": round(self._tokens, 1),
                "rate": self.rate,
                "capacity": self.capacity,
                "consecutive_429s": self._consecutive_429s,
                "in_penalty": time.monotonic() < self._penalty_until,
                "penalty_remaining_s": max(0, round(self._penalty_until - time.monotonic(), 1)),
            }


# ─── TTL Cache ───────────────────────────────────────────────────
class TTLCache:
    """
    Thread-safe time-to-live cache for API responses.
    Prevents duplicate requests when multiple modules need the same data.

    Example: RegimeDetector and OptionsFlowScanner both need allMids —
    only one API call happens, the other reads from cache.
    """

    def __init__(self, default_ttl: float = 2.0, max_size: int = 500):
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if time.monotonic() < expiry:
                    self._hits += 1
                    self._cache.move_to_end(key)
                    return value
                else:
                    del self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        with self._lock:
            ttl = ttl if ttl is not None else self.default_ttl
            self._cache[key] = (value, time.monotonic() + ttl)
            self._cache.move_to_end(key)
            # Evict oldest if over capacity
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def invalidate(self, key: str):
        with self._lock:
            self._cache.pop(key, None)

    def clear(self):
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total * 100, 1) if total > 0 else 0,
                "ttl": self.default_ttl,
            }


# ─── Cache TTLs by request type ─────────────────────────────────
# How long each type of data stays fresh in cache (seconds)
CACHE_TTLS = {
    "allMids": 5.0,            # Prices: 5s (mid prices read by multiple modules in same cycle; 5s still fresh)
    "metaAndAssetCtxs": 30.0,  # Funding/OI: 30s (regime+options+features all need this)
    "meta": 600.0,             # Exchange metadata: 10 min (almost never changes)
    "l2Book": 10.0,            # Order book: 10s (doesn't need sub-second freshness for paper trading)
    "recentTrades": 15.0,      # Recent trades: 15s (trades don't change that fast)
    "clearinghouseState": 10.0,# Trader positions: 10s (scanning 258 traders, same trader
                                #   may be queried by discovery + copy trader within 10s)
    "userFills": 30.0,         # Fill history: 30s (doesn't change between cycle phases)
    "userFunding": 120.0,      # Funding payments: 2 min
    "leaderboard": 300.0,      # Leaderboard: 5 min (only refreshed hourly anyway)
    "candleSnapshot": 60.0,    # Candles: 60s (already minute-level data, 1m freshness is adequate)
    "fundingHistory": 300.0,   # Funding history: 5 min
}

# Priority by request type
REQUEST_PRIORITIES = {
    "exchange": Priority.CRITICAL,
    "clearinghouseState": Priority.HIGH,    # Position checks for stop losses
    "allMids": Priority.HIGH,              # Price data for execution
    "userFills": Priority.NORMAL,
    "metaAndAssetCtxs": Priority.NORMAL,
    "l2Book": Priority.NORMAL,
    "leaderboard": Priority.LOW,
    "meta": Priority.LOW,
    "candleSnapshot": Priority.LOW,
    "fundingHistory": Priority.LOW,
    "recentTrades": Priority.NORMAL,
}

# Request-type cooldowns for endpoints that can fail globally but are safe to
# skip temporarily because the bot can continue from cached or stored data.
REQUEST_TYPE_FAILURE_THRESHOLDS = {
    "candleSnapshot": 3,
}
REQUEST_TYPE_COOLDOWNS = {
    "candleSnapshot": 120.0,
}
REQUEST_TYPE_FAIL_FAST_ON_SERVER_ERROR = {"candleSnapshot"}


# ─── Jittered Backoff ────────────────────────────────────────────
def jittered_backoff(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    """
    Exponential backoff with full jitter.
    Prevents synchronized retry bursts (the "thundering herd" on retry).
    """
    exp = min(base * (2 ** attempt), cap)
    return random.uniform(0, exp)


def ranged_backoff(min_wait: float, max_wait: float, attempt: int, growth: float) -> float:
    """Backoff with a hard minimum and bounded widening as attempts increase."""
    upper = min(max_wait, min_wait + growth * max(attempt + 1, 1))
    if upper <= min_wait:
        return min_wait
    return min_wait + random.uniform(0, upper - min_wait)


# ─── WebSocket Feed ──────────────────────────────────────────────
class HyperliquidWebSocket:
    """
    Real-time WebSocket feed for market data.
    Modules read from the local state dict instead of polling REST.

    Subscribes to:
    - allMids: real-time mid prices for all assets
    - l2Book: L2 order book updates for subscribed coins
    - trades: recent trades for subscribed coins
    - userEvents: (optional) position/fill updates for tracked wallets
    """

    WS_URL = "wss://api.hyperliquid.xyz/ws"
    _TRANSIENT_DISCONNECT_MARKERS = (
        "expired",
        "connection to remote host was lost",
        "goodbye",
        "closed connection",
        "connection reset",
        "broken pipe",
        "timed out",
        "ping/pong timed out",
        "no close frame",
    )

    def __init__(self):
        self._ws = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()

        # Live state — modules read from these dicts
        self.mids: Dict[str, float] = {}       # {coin: mid_price}
        self.l2_books: Dict[str, Dict] = {}    # {coin: {bids: [...], asks: [...]}}
        self.recent_trades: Dict[str, list] = {} # {coin: [trade, ...]}
        self.last_update: float = 0

        # Subscriptions
        self._subscribed_coins: set = set()
        self._callbacks: Dict[str, list] = {}  # {channel: [callback, ...]}

        # Stats
        self._messages_received = 0
        self._reconnect_count = 0
        self._connected = False

        # Sequence tracking for gap detection
        # Hyperliquid doesn't use explicit sequence numbers, so we track
        # timestamps to detect if we missed data during disconnects
        self._last_msg_time: float = 0.0
        self._last_ws_activity_time: float = 0.0
        self._gap_count: int = 0       # Number of detected gaps
        self._max_gap_ms: float = 0.0  # Largest gap seen
        self._gap_warn_threshold_ms: float = max(
            _WS_GAP_WARN_THRESHOLD_MS_MIN,
            float(getattr(config, "WS_FEED_GAP_WARN_MS", 30000.0)),
        )
        self._gap_warn_cooldown_s: float = min(
            _WS_GAP_WARN_COOLDOWN_S_MAX,
            max(
                _WS_GAP_WARN_COOLDOWN_S_MIN,
                float(getattr(config, "WS_FEED_GAP_WARN_COOLDOWN_S", 60.0)),
            ),
        )
        self._gap_warn_startup_grace_s: float = max(
            0.0,
            float(getattr(config, "WS_FEED_GAP_WARN_STARTUP_GRACE_S", 45.0)),
        )
        self._gap_warn_grace_until: float = 0.0
        self._last_gap_warn_time: float = 0.0
        self._reconnect_delay_override_s: Optional[float] = None
        self._reconnect_reason: Optional[str] = None

    def start(self):
        """Start the WebSocket connection in a background thread."""
        if not HAS_WEBSOCKET:
            logger.info("WebSocket feed unavailable (websocket-client not installed). "
                        "Install with: pip install websocket-client")
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_forever, daemon=True)
        self._thread.start()
        logger.info("WebSocket feed starting...")

    def stop(self):
        """Stop the WebSocket connection."""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def subscribe_coin(self, coin: str):
        """Subscribe to L2 book + trades for a specific coin."""
        self._subscribed_coins.add(coin)
        if self._ws and self._connected:
            self._send_subscribe(coin)

    def on_message(self, channel: str, callback: Callable):
        """Register a callback for a specific channel."""
        self._callbacks.setdefault(channel, []).append(callback)

    def get_mid(self, coin: str) -> Optional[float]:
        """Get latest mid price from WS feed (or None if not available)."""
        with self._lock:
            return self.mids.get(coin)

    def get_all_mids(self) -> Dict[str, float]:
        """Get all mid prices from WS feed."""
        with self._lock:
            return dict(self.mids)

    def is_connected(self) -> bool:
        return self._connected

    def mids_are_fresh(self, max_age_s: float = 2.0) -> bool:
        """
        Return True only when the WS feed is connected AND has received a
        message within ``max_age_s`` seconds.

        A silent TCP stall (no FIN, no failed pong) leaves ``_connected=True``
        while ``last_update`` grows stale.  Callers that use WS mids for
        order sizing/slippage MUST gate on this helper, not on
        ``is_connected`` alone — otherwise they'll size trades off prices
        that are minutes old during a silent disconnect.
        """
        if not self._connected:
            return False
        if self.last_update <= 0:
            return False
        return (time.time() - self.last_update) <= max_age_s

    def _run_forever(self):
        """Main loop: connect, subscribe, handle messages, reconnect."""
        while self._running:
            try:
                self._ws = _ws_lib.WebSocketApp(
                    self.WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_pong=self._on_pong,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                if self._running:
                    if self._is_transient_disconnect_error(e):
                        logger.info("WebSocket transport dropped: %s", e)
                    else:
                        logger.error("WebSocket loop error: %s", e)

            if self._running:
                wait, reason = self._consume_reconnect_wait()
                if reason:
                    logger.info("WebSocket refreshing %s in %.1fs", reason, wait)
                else:
                    logger.info(
                        f"WebSocket reconnecting in {wait:.1f}s "
                        f"(reconnect #{self._reconnect_count})"
                    )
                time.sleep(wait)

    def _on_open(self, ws):
        """
        Connected — subscribe to all channels and reconcile state.

        After a reconnect, local state (mids, books, trades) may be stale.
        We clear local caches and re-subscribe so fresh data overwrites them.
        If a reconciliation callback is registered, fire it so modules can
        verify their positions match the exchange.
        """
        was_reconnect = self._connected is False and self._reconnect_count > 0
        self._connected = True
        self._reconnect_count = 0
        now = time.time()
        self._last_msg_time = now
        self._last_ws_activity_time = now
        self._gap_warn_grace_until = now + self._gap_warn_startup_grace_s

        if was_reconnect:
            logger.info("WebSocket RECONNECTED -- clearing stale local state")
            # Invalidate stale data so modules don't act on old prices
            with self._lock:
                self.mids.clear()
                self.l2_books.clear()
                self.recent_trades.clear()
            # Fire reconciliation callbacks so modules can re-check positions
            for cb in self._callbacks.get("_reconnect", []):
                try:
                    cb()
                except Exception as e:
                    logger.error(f"Reconciliation callback error: {e}")
        else:
            logger.info("WebSocket connected")

        # Subscribe to allMids (real-time prices for all assets)
        ws.send(json.dumps({
            "method": "subscribe",
            "subscription": {"type": "allMids"}
        }))

        # Subscribe to L2 books and trades for specific coins
        for coin in self._subscribed_coins:
            self._send_subscribe(coin)

    def _send_subscribe(self, coin: str):
        """Subscribe to L2 book + trades for a coin."""
        if not self._ws or not self._connected:
            return
        try:
            self._ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "l2Book", "coin": coin}
            }))
            self._ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "trades", "coin": coin}
            }))
        except Exception as e:
            logger.debug(f"Subscribe error for {coin}: {e}")

    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages."""
        self._messages_received += 1
        now = time.time()

        # Gap detection with cooldown + startup grace to avoid noisy false alarms.
        if self._last_msg_time > 0:
            gap_ms = (now - self._last_msg_time) * 1000
            if gap_ms > self._gap_warn_threshold_ms:
                self._gap_count += 1
                prev_max = self._max_gap_ms
                self._max_gap_ms = max(self._max_gap_ms, gap_ms)
                in_grace = now < self._gap_warn_grace_until
                should_warn = (
                    not in_grace
                    and (
                        (now - self._last_gap_warn_time) >= self._gap_warn_cooldown_s
                        or gap_ms > (prev_max + 1000.0)
                    )
                )
                if should_warn:
                    self._last_gap_warn_time = now
                    logger.warning(
                        "WebSocket gap detected: %.0fms since last message (gap #%d, warn_threshold=%.0fms)",
                        gap_ms,
                        self._gap_count,
                        self._gap_warn_threshold_ms,
                    )
                else:
                    logger.debug(
                        "WebSocket gap observed: %.0fms (gap #%d, warn_threshold=%.0fms, in_grace=%s)",
                        gap_ms,
                        self._gap_count,
                        self._gap_warn_threshold_ms,
                        in_grace,
                    )
        self._last_msg_time = now
        self._last_ws_activity_time = now
        self.last_update = now

        try:
            data = json.loads(message)
            channel = data.get("channel")

            if channel == "allMids":
                mids_data = data.get("data", {}).get("mids", {})
                with self._lock:
                    for coin, price_str in mids_data.items():
                        try:
                            self.mids[coin] = float(price_str)
                        except (ValueError, TypeError):
                            pass

            elif channel == "l2Book":
                book_data = data.get("data", {})
                coin = book_data.get("coin")
                if coin:
                    book = {
                        "bids": book_data.get("levels", [[]])[0] if book_data.get("levels") else [],
                        "asks": book_data.get("levels", [[], []])[1] if len(book_data.get("levels", [])) > 1 else [],
                        "time": time.time(),
                    }
                    with self._lock:
                        self.l2_books[coin] = book

            elif channel == "trades":
                trades_data = data.get("data", [])
                if trades_data:
                    coin = trades_data[0].get("coin") if isinstance(trades_data[0], dict) else None
                    if coin:
                        with self._lock:
                            existing = self.recent_trades.get(coin, [])
                            existing.extend(trades_data)
                            self.recent_trades[coin] = existing[-100:]  # Keep last 100

            # Fire callbacks
            if channel in self._callbacks:
                for cb in self._callbacks[channel]:
                    try:
                        cb(data)
                    except Exception as e:
                        logger.debug(f"Callback error: {e}")

        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.debug(f"WS message parse error: {e}")

    def _on_pong(self, ws, message):
        """Treat pong frames as healthy transport activity."""
        self._last_ws_activity_time = time.time()

    @classmethod
    def _is_transient_disconnect_error(cls, error: Any) -> bool:
        """Classify common transport disconnects as transient/recoverable."""
        if error is None:
            return False
        text = str(error).lower()
        return any(marker in text for marker in cls._TRANSIENT_DISCONNECT_MARKERS)

    def _request_fast_reconnect(self, reason: str, delay_s: float = 1.0) -> None:
        """Override the next reconnect delay for session-expiry refreshes."""
        self._reconnect_delay_override_s = max(0.0, float(delay_s))
        self._reconnect_reason = reason
        self._reconnect_count = 0

    def _consume_reconnect_wait(self) -> tuple[float, Optional[str]]:
        """Return the next reconnect wait and clear any one-shot override."""
        if self._reconnect_delay_override_s is not None:
            wait = self._reconnect_delay_override_s
            reason = self._reconnect_reason
            self._reconnect_delay_override_s = None
            self._reconnect_reason = None
            return wait, reason
        self._reconnect_count += 1
        wait = min(5 * (2 ** min(self._reconnect_count, 4)), 60)
        wait += random.uniform(0, 3)
        return wait, None

    def _on_error(self, ws, error):
        error_text = str(error or "")
        expired = "expired" in error_text.lower()
        if self._running:
            if expired:
                logger.info("WebSocket session expired; refreshing subscription")
            elif self._is_transient_disconnect_error(error):
                logger.info("WebSocket transient disconnect: %s", error)
            else:
                logger.warning("WebSocket error: %s", error)
        self._connected = False
        if expired:
            self._request_fast_reconnect("expired WebSocket session")

    def _on_close(self, ws, close_status_code=None, close_msg=None):
        self._connected = False
        close_text = " ".join(
            part for part in (str(close_status_code or ""), str(close_msg or "")) if part
        )
        if "expired" in close_text.lower():
            self._request_fast_reconnect("expired WebSocket session")
        if self._running:
            logger.info(f"WebSocket closed (code={close_status_code})")

    def get_stats(self) -> Dict:
        return {
            "connected": self._connected,
            "messages_received": self._messages_received,
            "reconnect_count": self._reconnect_count,
            "coins_subscribed": len(self._subscribed_coins),
            "mids_tracked": len(self.mids),
            "last_update": self.last_update,
            "age_seconds": round(time.time() - self.last_update, 1) if self.last_update else None,
            "gaps_detected": self._gap_count,
            "max_gap_ms": round(self._max_gap_ms, 0),
            "gap_warn_threshold_ms": round(self._gap_warn_threshold_ms, 0),
        }


# ─── Unified API Manager ─────────────────────────────────────────
class APIManager:
    """
    Singleton that ALL modules use to make API requests.
    Routes through token bucket + cache, falls back to WebSocket data.
    """

    def __init__(self):
        self.bucket = TokenBucket(rate=15.0, capacity=30)
        self.cache = TTLCache(default_ttl=2.0, max_size=500)
        self.ws = HyperliquidWebSocket()

        # Stats
        self._total_requests = 0
        self._cache_served = 0
        self._ws_served = 0
        self._rest_requests = 0
        self._lock = threading.Lock()

        # Failure circuit breaker: if we see too many transient errors in a
        # rolling time window, pause non-critical requests to let the exchange
        # recover.  Client errors (4xx) are NOT counted — they indicate a bad
        # request, not a struggling exchange.
        self._failure_timestamps: deque = deque()   # monotonic timestamps of recent transient failures
        self._circuit_open_until = 0.0
        self._CIRCUIT_THRESHOLD = 10    # failures in window to trip
        self._FAILURE_WINDOW_S = 60.0   # rolling window duration
        self._CIRCUIT_COOLDOWN = 30.0   # seconds to pause non-critical traffic after trip
        # Max age of WS mids before we stop serving them and fall back to REST.
        # A silent TCP stall can leave is_connected=True with minutes-old data.
        # C9: clamp to [0.1, 300]; lower = more fallback to REST, higher = stale data risk.
        self._WS_MIDS_MAX_AGE_S = safe_env_float(
            "WS_MIDS_MAX_AGE_S", 2.0, lo=0.1, hi=300.0,
        )
        self._req_type_failures: Dict[str, int] = {}
        self._req_type_cooldown_until: Dict[str, float] = {}
        # Track when we last logged a SERVER_ERROR warning per req_type so we
        # can coalesce repeated failures instead of spamming the log.
        self._req_type_last_warn_at: Dict[str, float] = {}
        self._req_type_suppressed_warns: Dict[str, int] = {}

    # Failure kinds that genuinely indicate the exchange is struggling.
    # client_error (4xx) and auth_error do NOT count — they're caller bugs.
    _TRANSIENT_FAILURE_KINDS = frozenset({
        "server_error", "timeout", "connection_error", "request_error",
    })

    def _record_transient_failure(self, now: float) -> int:
        """
        Append *now* to the sliding failure window and return the current
        window count.  Automatically evicts timestamps outside the window.
        Must be called with ``self._lock`` held.
        """
        self._failure_timestamps.append(now)
        cutoff = now - self._FAILURE_WINDOW_S
        while self._failure_timestamps and self._failure_timestamps[0] < cutoff:
            self._failure_timestamps.popleft()
        return len(self._failure_timestamps)

    def _maybe_trip_circuit(self, n: int, now: float) -> None:
        """
        If the sliding-window failure count *n* has reached the threshold,
        open the circuit breaker and clear the window.
        Must be called with ``self._lock`` held.
        """
        if n >= self._CIRCUIT_THRESHOLD:
            self._circuit_open_until = now + self._CIRCUIT_COOLDOWN
            self._failure_timestamps.clear()
            logger.warning(
                "Circuit breaker TRIPPED after %d failures in %.0fs window -- "
                "pausing non-critical requests for %.0fs",
                n,
                self._FAILURE_WINDOW_S,
                self._CIRCUIT_COOLDOWN,
            )

    def _is_req_type_cooling_down(self, req_type: str, now: float) -> bool:
        """Return True while a request type is in a temporary cooldown."""
        if req_type not in self._req_type_cooldown_until:
            return False
        cooldown_until = self._req_type_cooldown_until.get(req_type, 0.0)
        if cooldown_until <= now:
            self._req_type_cooldown_until.pop(req_type, None)
            self._req_type_failures.pop(req_type, None)
            return False
        return True

    def _record_req_type_result(self, req_type: str, failure_kind: Optional[str]) -> None:
        """Track request-type failures so noisy upstream outages cool down quickly."""
        threshold = REQUEST_TYPE_FAILURE_THRESHOLDS.get(req_type)
        if not threshold:
            return
        if failure_kind == "server_error":
            failures = self._req_type_failures.get(req_type, 0) + 1
            self._req_type_failures[req_type] = failures
            if failures >= threshold:
                cooldown = REQUEST_TYPE_COOLDOWNS.get(req_type, self._CIRCUIT_COOLDOWN)
                self._req_type_cooldown_until[req_type] = time.monotonic() + cooldown
                self._req_type_failures[req_type] = 0
                suppressed = self._req_type_suppressed_warns.pop(req_type, 0)
                self._req_type_last_warn_at.pop(req_type, None)
                logger.warning(
                    "Request-type circuit OPEN for %s after %d server errors "
                    "(%d warnings suppressed) - cooling down for %.0fs",
                    req_type,
                    threshold,
                    suppressed,
                    cooldown,
                )
        elif failure_kind is None:
            self._req_type_failures[req_type] = 0
            # Clear suppression state on successful call so the next outage
            # gets its own fresh warning at the head of the window.
            self._req_type_last_warn_at.pop(req_type, None)
            self._req_type_suppressed_warns.pop(req_type, None)

    def start_websocket(self, coins: list = None):
        """Start the WebSocket feed for real-time data."""
        if coins:
            for coin in coins:
                self.ws.subscribe_coin(coin)
        self.ws.start()

    def stop_websocket(self):
        """Stop the shared WebSocket feed."""
        self.ws.stop()

    def post(
        self,
        payload: dict,
        priority: Priority = None,
        cache_ttl: Optional[float] = None,
        retries: int = 3,
        endpoint_url: Optional[str] = None,
        cache_response: bool = True,
        req_type: Optional[str] = None,
        timeout: int = 30,
        raise_on_timeout: bool = False,
        force_fresh: bool = False,
    ) -> Optional[Any]:
        """
        Central API call with rate limiting + caching.
        All modules should use this instead of direct requests.post().

        Parameters
        ----------
        force_fresh : bool
            P0-3 (audit).  Bypasses the TTL cache AND the WebSocket fast path
            so a *freshly fetched* REST response is returned to the caller.
            The fetched value still overwrites the cache so normal callers
            benefit on subsequent reads.  Use this on safety-critical reads:
            fill verification, emergency close decisions, realised-PnL
            computations off ``userFills``, and any cap/risk check that
            must not be satisfied by a stale snapshot older than the
            request that made the trade.
        """
        req_type = req_type or payload.get("type", "unknown")
        endpoint_url = endpoint_url or config.HYPERLIQUID_INFO_URL

        # Determine priority
        if priority is None:
            priority = REQUEST_PRIORITIES.get(req_type, Priority.NORMAL)

        # Build cache key from payload
        cache_key = self._cache_key(endpoint_url, payload)

        # P0-3: when force_fresh is set, pre-emptively evict the cache entry
        # so other threads cannot read the about-to-be-replaced stale value
        # while this request is in flight.  Short window, but the whole point
        # of this path is that staleness is unsafe.
        if force_fresh:
            try:
                self.cache.invalidate(cache_key)
            except Exception:
                pass

        # Check cache first (unless force_fresh)
        if cache_response and not force_fresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                with self._lock:
                    self._cache_served += 1
                return cached

        # For allMids, try WebSocket first — but only if the feed is FRESH.
        # ``is_connected()`` alone returns True during silent TCP stalls, which
        # would serve price data that's minutes old.  ``mids_are_fresh`` also
        # confirms we received a WS message recently.  On stale feeds we fall
        # through to the REST path instead.
        #
        # force_fresh also bypasses the WS shortcut — caller is explicitly
        # asking for a round-trip to the exchange, not a cached or streamed
        # value from a separate pipeline.
        if (
            cache_response
            and not force_fresh
            and endpoint_url == config.HYPERLIQUID_INFO_URL
            and req_type == "allMids"
            and self.ws.mids_are_fresh(max_age_s=self._WS_MIDS_MAX_AGE_S)
            and self.ws.mids
        ):
            # Convert back to string format like the REST API returns
            mids = {k: str(v) for k, v in self.ws.get_all_mids().items()}
            if mids:
                self.cache.put(cache_key, mids, ttl=0.5)
                with self._lock:
                    self._ws_served += 1
                return mids

        # Circuit breaker: if too many recent failures, block non-critical requests
        now = time.monotonic()
        if now < self._circuit_open_until and priority > Priority.HIGH:
            remaining = round(self._circuit_open_until - now, 1)
            logger.debug(f"Circuit breaker OPEN -- skipping {req_type} ({remaining}s remaining)")
            return None
        if self._is_req_type_cooling_down(req_type, now) and priority > Priority.HIGH:
            return None

        # Acquire token from bucket
        if not self.bucket.acquire(priority=priority, timeout=30):
            logger.warning(f"Rate limiter timeout for {req_type} (priority={priority.name})")
            return None

        # Make the actual REST request
        with self._lock:
            self._total_requests += 1
            self._rest_requests += 1

        try:
            result, failure_kind = self._do_request(
                payload,
                endpoint_url=endpoint_url,
                req_type=req_type,
                retries=retries,
                timeout=timeout,
                raise_on_timeout=raise_on_timeout,
            )
        except requests.exceptions.Timeout:
            with self._lock:
                n = self._record_transient_failure(time.monotonic())
                self._maybe_trip_circuit(n, time.monotonic())
            raise
        self._record_req_type_result(req_type, failure_kind)

        # Track transient failures for circuit breaker (sliding window).
        # Only server-side / network errors count — client errors (4xx) are
        # the caller's fault and should not be charged against the exchange.
        with self._lock:
            if result is None and failure_kind in self._TRANSIENT_FAILURE_KINDS:
                n = self._record_transient_failure(time.monotonic())
                self._maybe_trip_circuit(n, time.monotonic())
            elif result is not None:
                # Success: evict stale timestamps from window (passive decay).
                now_s = time.monotonic()
                cutoff = now_s - self._FAILURE_WINDOW_S
                while self._failure_timestamps and self._failure_timestamps[0] < cutoff:
                    self._failure_timestamps.popleft()

        # Cache the result
        if cache_response and result is not None:
            ttl = cache_ttl or CACHE_TTLS.get(req_type, self.cache.default_ttl)
            self.cache.put(cache_key, result, ttl=ttl)

        return result

    def _do_request(
        self,
        payload: dict,
        endpoint_url: str,
        req_type: Optional[str] = None,
        retries: int = 3,
        timeout: int = 30,
        raise_on_timeout: bool = False,
    ) -> tuple[Optional[Any], Optional[str]]:
        """
        Execute the HTTP request with classified error handling.

        Error categories:
        - 429 (rate limit): backoff hard, report to bucket, always retry
        - 401/403 (auth): NEVER retry — credentials are wrong
        - 422 (bad payload): NEVER retry — request is malformed
        - 400 (other client): don't retry — request won't work
        - 5xx (server): retry with backoff — transient
        - Network errors: retry with backoff — transient
        """
        req_type = req_type or payload.get("type", "unknown")
        failure_kind: Optional[str] = None

        for attempt in range(retries):
            try:
                resp = requests.post(
                    endpoint_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=timeout,
                )

                # ── 429: Rate limited — always retry with hard backoff ──
                if resp.status_code == 429:
                    self.bucket.report_429()
                    wait = ranged_backoff(20.0, 60.0, attempt, growth=10.0)
                    logger.warning(
                        f"429 RATE_LIMITED type='{req_type}' -- "
                        f"wait {wait:.1f}s (attempt {attempt+1}/{retries})"
                    )
                    time.sleep(wait)
                    continue

                # ── 401/403: Auth error — NEVER retry ──
                if resp.status_code in (401, 403):
                    logger.error(
                        f"AUTH_ERROR {resp.status_code} type='{req_type}': "
                        f"check API credentials (not retrying)"
                    )
                    return None, "auth_error"

                # ── 422: Bad payload — NEVER retry ──
                if resp.status_code == 422:
                    body_preview = resp.text[:200] if resp.text else "(empty)"
                    logger.warning(
                        f"BAD_PAYLOAD 422 type='{req_type}': {body_preview} (not retrying)"
                    )
                    return None, "client_error"

                # ── Other 4xx: Client error — don't retry ──
                if 400 <= resp.status_code < 500:
                    body_preview = resp.text[:200] if resp.text else "(empty)"
                    logger.warning(
                        f"CLIENT_ERROR {resp.status_code} type='{req_type}': "
                        f"{body_preview} (not retrying)"
                    )
                    return None, "client_error"

                # ── 500 + null text: Hyperliquid specific client error — don't retry ──
                # Hyperliquid returns 500 with a body of literally "null" for some
                # invalid payloads (e.g. candleSnapshot on an invalid/delisted coin).
                # We categorize this as a client error to avoid tripping the fail-fast
                # server-error circuit breaker.
                if resp.status_code == 500 and resp.text and resp.text.strip() == "null":
                    logger.debug(
                        f"CLIENT_ERROR {resp.status_code} (null payload) type='{req_type}': "
                        f"treated as invalid parameters/coin (not retrying)"
                    )
                    return None, "client_error"

                # ── 5xx: Server error — retry with backoff ──
                if resp.status_code >= 500:
                    failure_kind = "server_error"
                    if req_type in REQUEST_TYPE_FAIL_FAST_ON_SERVER_ERROR:
                        # Coalesce noisy repeats: only warn once per ~60s window
                        # per req_type. Intermediate failures are counted and
                        # rolled into the cooldown-open summary.
                        now_m = time.monotonic()
                        last_warn = self._req_type_last_warn_at.get(req_type, 0.0)
                        if now_m - last_warn >= 60.0:
                            logger.warning(
                                "SERVER_ERROR %s type='%s' - fail-fast cooldown trigger",
                                resp.status_code,
                                req_type,
                            )
                            self._req_type_last_warn_at[req_type] = now_m
                            self._req_type_suppressed_warns[req_type] = 0
                        else:
                            self._req_type_suppressed_warns[req_type] = (
                                self._req_type_suppressed_warns.get(req_type, 0) + 1
                            )
                            logger.debug(
                                "SERVER_ERROR %s type='%s' (suppressed, %d within window)",
                                resp.status_code,
                                req_type,
                                self._req_type_suppressed_warns[req_type],
                            )
                        return None, failure_kind
                    wait = ranged_backoff(2.0, 8.0, attempt, growth=2.0)
                    logger.warning(
                        f"SERVER_ERROR {resp.status_code} type='{req_type}' -- "
                        f"retry in {wait:.1f}s (attempt {attempt+1}/{retries})"
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                self.bucket.report_success()
                return resp.json(), None

            except requests.exceptions.Timeout:
                if raise_on_timeout:
                    raise
                failure_kind = "timeout"
                wait = jittered_backoff(attempt, base=2.0, cap=20.0)
                logger.warning(
                    f"TIMEOUT type='{req_type}' -- retry in {wait:.1f}s "
                    f"(attempt {attempt+1}/{retries})"
                )
                time.sleep(wait)

            except requests.exceptions.ConnectionError:
                failure_kind = "connection_error"
                wait = jittered_backoff(attempt, base=3.0, cap=30.0)
                logger.warning(
                    f"CONNECTION_ERROR type='{req_type}' -- retry in {wait:.1f}s "
                    f"(attempt {attempt+1}/{retries})"
                )
                time.sleep(wait)

            except requests.exceptions.RequestException as e:
                failure_kind = "request_error"
                logger.warning(
                    f"REQUEST_ERROR type='{req_type}' "
                    f"(attempt {attempt+1}/{retries}): {e}"
                )
                if attempt < retries - 1:
                    wait = jittered_backoff(attempt)
                    time.sleep(wait)

        return None, failure_kind

    def _cache_key(self, endpoint_url: str, payload: dict) -> str:
        """Generate a deterministic cache key from a payload."""
        # Sort keys for deterministic ordering
        raw = json.dumps({"endpoint": endpoint_url, "payload": payload}, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    def get_stats(self) -> Dict:
        with self._lock:
            total = self._total_requests or 1
            now = time.monotonic()
            return {
                "total_requests": self._total_requests,
                "cache_served": self._cache_served,
                "ws_served": self._ws_served,
                "rest_requests": self._rest_requests,
                "cache_hit_pct": round(self._cache_served / total * 100, 1),
                "ws_hit_pct": round(self._ws_served / total * 100, 1),
                "circuit_breaker": {
                    "open": now < self._circuit_open_until,
                    "failures_in_window": len(self._failure_timestamps),
                    "threshold": self._CIRCUIT_THRESHOLD,
                    "window_s": self._FAILURE_WINDOW_S,
                    "cooldown_remaining_s": max(0, round(self._circuit_open_until - now, 1)),
                },
                "request_type_cooldowns": {
                    req_type: max(0, round(until - now, 1))
                    for req_type, until in self._req_type_cooldown_until.items()
                    if until > now
                },
                "bucket": self.bucket.get_stats(),
                "cache": self.cache.get_stats(),
                "websocket": self.ws.get_stats(),
            }


# ─── Global singleton ────────────────────────────────────────────
_manager: Optional[APIManager] = None
_manager_lock = threading.Lock()


def get_manager() -> APIManager:
    """Get or create the global API manager singleton."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = APIManager()
    return _manager
