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
from collections import OrderedDict
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

logger = logging.getLogger("api_manager")
logger.addHandler(logging.NullHandler())


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
                f"429 received — penalty {penalty}s "
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


# ─── Jittered Backoff ────────────────────────────────────────────
def jittered_backoff(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    """
    Exponential backoff with full jitter.
    Prevents synchronized retry bursts (the "thundering herd" on retry).
    """
    exp = min(base * (2 ** attempt), cap)
    return random.uniform(0, exp)


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
        self._gap_count: int = 0       # Number of detected gaps
        self._max_gap_ms: float = 0.0  # Largest gap seen

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

    def _run_forever(self):
        """Main loop: connect, subscribe, handle messages, reconnect."""
        while self._running:
            try:
                self._ws = _ws_lib.WebSocketApp(
                    self.WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

            if self._running:
                self._reconnect_count += 1
                wait = min(5 * (2 ** min(self._reconnect_count, 4)), 60)
                wait += random.uniform(0, 3)  # jitter
                logger.info(f"WebSocket reconnecting in {wait:.1f}s "
                           f"(reconnect #{self._reconnect_count})")
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

        if was_reconnect:
            logger.info("WebSocket RECONNECTED — clearing stale local state")
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

        # Gap detection: if time since last message > 5s, we likely missed data
        if self._last_msg_time > 0:
            gap_ms = (now - self._last_msg_time) * 1000
            if gap_ms > 5000:  # 5 second gap = suspicious
                self._gap_count += 1
                self._max_gap_ms = max(self._max_gap_ms, gap_ms)
                logger.warning(f"WebSocket gap detected: {gap_ms:.0f}ms since last message "
                             f"(gap #{self._gap_count})")
        self._last_msg_time = now
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

    def _on_error(self, ws, error):
        if self._running:
            logger.warning(f"WebSocket error: {error}")
        self._connected = False

    def _on_close(self, ws, close_status_code=None, close_msg=None):
        self._connected = False
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

        # Failure circuit breaker: if we see too many errors in a window,
        # pause non-critical requests to let the exchange recover.
        # This prevents the bot from hammering a struggling API.
        self._recent_failures = 0
        self._circuit_open_until = 0.0
        self._CIRCUIT_THRESHOLD = 10    # failures in window to trip
        self._CIRCUIT_COOLDOWN = 60.0   # seconds to wait when tripped

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
    ) -> Optional[Any]:
        """
        Central API call with rate limiting + caching.
        All modules should use this instead of direct requests.post().
        """
        req_type = req_type or payload.get("type", "unknown")
        endpoint_url = endpoint_url or config.HYPERLIQUID_INFO_URL

        # Determine priority
        if priority is None:
            priority = REQUEST_PRIORITIES.get(req_type, Priority.NORMAL)

        # Build cache key from payload
        cache_key = self._cache_key(endpoint_url, payload)

        # Check cache first
        if cache_response:
            cached = self.cache.get(cache_key)
            if cached is not None:
                with self._lock:
                    self._cache_served += 1
                return cached

        # For allMids, try WebSocket first
        if (
            cache_response
            and endpoint_url == config.HYPERLIQUID_INFO_URL
            and req_type == "allMids"
            and self.ws.is_connected()
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
            logger.debug(f"Circuit breaker OPEN — skipping {req_type} ({remaining}s remaining)")
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
            result = self._do_request(
                payload,
                endpoint_url=endpoint_url,
                req_type=req_type,
                retries=retries,
                timeout=timeout,
                raise_on_timeout=raise_on_timeout,
            )
        except requests.exceptions.Timeout:
            with self._lock:
                self._recent_failures += 1
            raise

        # Track failures for circuit breaker
        with self._lock:
            if result is None:
                self._recent_failures += 1
                if self._recent_failures >= self._CIRCUIT_THRESHOLD:
                    self._circuit_open_until = time.monotonic() + self._CIRCUIT_COOLDOWN
                    logger.warning(
                        f"Circuit breaker TRIPPED after {self._recent_failures} failures — "
                        f"pausing non-critical requests for {self._CIRCUIT_COOLDOWN}s"
                    )
                    self._recent_failures = 0  # Reset counter
            else:
                # Successful request: decay failure counter
                self._recent_failures = max(0, self._recent_failures - 1)

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
    ) -> Optional[Any]:
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
                    wait = jittered_backoff(attempt, base=5.0, cap=60.0)
                    logger.warning(
                        f"429 RATE_LIMITED type='{req_type}' — "
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
                    return None

                # ── 422: Bad payload — NEVER retry ──
                if resp.status_code == 422:
                    body_preview = resp.text[:200] if resp.text else "(empty)"
                    logger.warning(
                        f"BAD_PAYLOAD 422 type='{req_type}': {body_preview} (not retrying)"
                    )
                    return None

                # ── Other 4xx: Client error — don't retry ──
                if 400 <= resp.status_code < 500:
                    body_preview = resp.text[:200] if resp.text else "(empty)"
                    logger.warning(
                        f"CLIENT_ERROR {resp.status_code} type='{req_type}': "
                        f"{body_preview} (not retrying)"
                    )
                    return None

                # ── 5xx: Server error — retry with backoff ──
                if resp.status_code >= 500:
                    wait = jittered_backoff(attempt, base=3.0, cap=30.0)
                    logger.warning(
                        f"SERVER_ERROR {resp.status_code} type='{req_type}' — "
                        f"retry in {wait:.1f}s (attempt {attempt+1}/{retries})"
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                self.bucket.report_success()
                return resp.json()

            except requests.exceptions.Timeout:
                if raise_on_timeout:
                    raise
                wait = jittered_backoff(attempt, base=2.0, cap=20.0)
                logger.warning(
                    f"TIMEOUT type='{req_type}' — retry in {wait:.1f}s "
                    f"(attempt {attempt+1}/{retries})"
                )
                time.sleep(wait)

            except requests.exceptions.ConnectionError:
                wait = jittered_backoff(attempt, base=3.0, cap=30.0)
                logger.warning(
                    f"CONNECTION_ERROR type='{req_type}' — retry in {wait:.1f}s "
                    f"(attempt {attempt+1}/{retries})"
                )
                time.sleep(wait)

            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"REQUEST_ERROR type='{req_type}' "
                    f"(attempt {attempt+1}/{retries}): {e}"
                )
                if attempt < retries - 1:
                    wait = jittered_backoff(attempt)
                    time.sleep(wait)

        return None

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
                    "recent_failures": self._recent_failures,
                    "threshold": self._CIRCUIT_THRESHOLD,
                    "cooldown_remaining_s": max(0, round(self._circuit_open_until - now, 1)),
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
