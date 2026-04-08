"""
Real-Time Position Monitor
===========================
Subscribes to Hyperliquid WebSocket userEvents for top traders,
detecting position changes in real-time instead of 5-minute polling.

Feeds detected changes into the CopyTrader signal pipeline.

Design:
- Maintains a position cache per trader (in-memory)
- Subscribes to userEvents for each tracked address
- Detects: new positions, closed positions, scale-ins (>50% increase), side flips
- Emits signals in the SAME format as copy_trader.py's _detect_position_changes()
- Thread-safe signal queue for main loop to drain
- Auto-reconnects on WebSocket disconnect
- Dynamically add/remove traders without restart
"""
import threading
import time
import json
import logging
from typing import List, Dict, Optional, Set
from collections import deque

try:
    import websocket as _ws_lib
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data import hyperliquid_client as hl

logger = logging.getLogger(__name__)


class PositionMonitor:
    """
    Monitors top traders' positions via Hyperliquid WebSocket userEvents.

    Real-time detection beats the 5-minute polling cycle in copy_trader.py
    by streaming position changes as they happen.
    """

    WS_URL = "wss://api.hyperliquid.xyz/ws"

    def __init__(self, max_signal_queue: int = 1000):
        """
        Initialize the position monitor.

        Args:
            max_signal_queue: Maximum pending signals before dropping oldest
        """
        self._ws = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()

        # Position cache: {address: {coin: {size, side, entry_price, leverage, ...}}}
        self._position_cache: Dict[str, Dict[str, Dict]] = {}

        # Signal queue (thread-safe deque)
        self._signal_queue: deque = deque(maxlen=max_signal_queue)

        # Tracked addresses and subscriptions
        self._tracked_addresses: Set[str] = set()
        self._subscribed_addresses: Set[str] = set()

        # Stats
        self._messages_received = 0
        self._reconnect_count = 0
        self._signals_emitted = 0
        self._connected = False
        self._last_msg_time = 0.0
        self._last_ws_activity_time = 0.0
        self._gap_count = 0
        self._max_gap_ms = 0.0
        self._last_gap_warn_time = 0.0

        # Mid-price cache is refreshed by a dedicated REST thread so the
        # WebSocket message handler never blocks on external HTTP calls.
        self._mids_cache: Dict[str, float] = {}
        self._mids_last_refresh = 0.0
        self._mids_refresh_interval_s = max(
            0.5,
            float(getattr(config, "POSITION_MONITOR_MIDS_REFRESH_S", 1.5)),
        )
        self._mids_thread = None

        # Watchdog: if WS stays "connected" but no messages arrive for too
        # long, force-close so run_forever reconnects and re-subscribes.
        self._watchdog_timeout_s = max(
            10.0,
            float(getattr(config, "POSITION_MONITOR_WATCHDOG_TIMEOUT_S", 30.0)),
        )
        self._watchdog_reconnect_cooldown_s = max(
            15.0,
            float(
                getattr(
                    config,
                    "POSITION_MONITOR_WATCHDOG_RECONNECT_COOLDOWN_S",
                    120.0,
                )
            ),
        )
        self._last_watchdog_reconnect_time = 0.0
        self._watchdog_thread = None

        # Gap warning controls. A quiet userEvents stream can naturally have
        # multi-second idle windows; keep warnings for materially long silence.
        default_gap_warn_s = max(15.0, self._watchdog_timeout_s * 0.5)
        self._gap_warn_threshold_s = max(
            5.0,
            float(getattr(config, "POSITION_MONITOR_GAP_WARN_S", default_gap_warn_s)),
        )
        self._gap_warn_cooldown_s = max(
            5.0,
            float(getattr(config, "POSITION_MONITOR_GAP_WARN_COOLDOWN_S", 60.0)),
        )

    def start(self, addresses: List[str]) -> None:
        """
        Connect to WebSocket and subscribe to userEvents for the given addresses.

        Args:
            addresses: List of trader addresses to monitor (0x-prefixed hex strings)
        """
        if not HAS_WEBSOCKET:
            logger.error("WebSocket client not installed. Install with: pip install websocket-client")
            return

        if self._running:
            logger.warning("PositionMonitor already running")
            return

        self._tracked_addresses = set(addresses)
        self._running = True
        self._thread = threading.Thread(target=self._run_forever, daemon=True)
        self._thread.start()
        self._mids_thread = threading.Thread(target=self._refresh_mids_loop, daemon=True)
        self._mids_thread.start()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()
        logger.info(f"PositionMonitor starting with {len(addresses)} traders")

    def stop(self) -> None:
        """Stop the WebSocket connection and monitoring."""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        logger.info("PositionMonitor stopped")

    def add_trader(self, address: str) -> None:
        """
        Subscribe to a new trader's positions dynamically.

        Args:
            address: Trader address to monitor
        """
        with self._lock:
            self._tracked_addresses.add(address)
            # If connected, subscribe immediately
            if self._ws and self._connected:
                self._subscribe_to_address(address)
                logger.info(f"Added trader {address[:10]} to live monitoring")

    def remove_trader(self, address: str) -> None:
        """
        Stop monitoring a trader's positions.

        Args:
            address: Trader address to stop monitoring
        """
        with self._lock:
            self._tracked_addresses.discard(address)
            self._position_cache.pop(address, None)
            logger.info(f"Removed trader {address[:10]} from monitoring")

    def drain_signals(self) -> List[Dict]:
        """
        Get all pending signals and clear the queue.
        Thread-safe — call from main loop to get new signals.

        Returns:
            List of signal dicts (empty if none pending)
        """
        signals = []
        try:
            while True:
                signals.append(self._signal_queue.popleft())
        except IndexError:
            pass
        return signals

    def get_stats(self) -> Dict:
        """
        Get monitoring statistics.

        Returns:
            Dict with connection status, message counts, etc.
        """
        with self._lock:
            return {
                "connected": self._connected,
                "running": self._running,
                "tracked_addresses": len(self._tracked_addresses),
                "subscribed_addresses": len(self._subscribed_addresses),
                "cached_positions": sum(len(p) for p in self._position_cache.values()),
                "pending_signals": len(self._signal_queue),
                "messages_received": self._messages_received,
                "signals_emitted": self._signals_emitted,
                "reconnect_count": self._reconnect_count,
                "gaps_detected": self._gap_count,
                "max_gap_ms": round(self._max_gap_ms, 0),
                "gap_warn_threshold_s": round(self._gap_warn_threshold_s, 2),
                "last_update": self._last_msg_time,
            }

    # ─── Private: WebSocket lifecycle ──────────────────────────────

    def _run_forever(self) -> None:
        """Main loop: connect, subscribe, handle messages, reconnect on error."""
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
                logger.error(f"WebSocket error: {e}")

            if self._running:
                self._reconnect_count += 1
                wait = min(5 * (2 ** min(self._reconnect_count, 4)), 60)
                import random
                wait += random.uniform(0, 3)
                logger.info(f"PositionMonitor reconnecting in {wait:.1f}s "
                           f"(reconnect #{self._reconnect_count})")
                time.sleep(wait)

    def _on_open(self, ws) -> None:
        """
        Connected — subscribe to all tracked traders' userEvents.
        """
        was_reconnect = self._connected is False and self._reconnect_count > 0

        with self._lock:
            self._connected = True
            now = time.time()
            self._last_msg_time = now
            self._last_ws_activity_time = now
            tracked = list(self._tracked_addresses)

        if was_reconnect:
            logger.info("PositionMonitor RECONNECTED — clearing stale cache")
            with self._lock:
                self._position_cache.clear()
                self._subscribed_addresses.clear()
            # Bootstrap fresh position state before consuming new deltas.
            for address in tracked:
                self._bootstrap_positions(address)
        else:
            logger.info("PositionMonitor connected")
            for address in tracked:
                self._bootstrap_positions(address)

        # Subscribe to all tracked addresses
        for address in tracked:
            self._subscribe_to_address(address)

    def _subscribe_to_address(self, address: str) -> None:
        """
        Send subscription request for a single trader's userEvents.
        Must be called with lock held or from WebSocket thread.
        """
        if not self._ws or not self._connected:
            return
        try:
            self._ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {
                    "type": "userEvents",
                    "user": address
                }
            }))
            with self._lock:
                self._subscribed_addresses.add(address)
            logger.debug(f"Subscribed to userEvents for {address[:10]}")
        except Exception as e:
            logger.warning(f"Failed to subscribe to {address[:10]}: {e}")

    def _on_message(self, ws, message: str) -> None:
        """
        Handle incoming WebSocket messages from userEvents subscriptions.
        """
        gap_log = None
        with self._lock:
            self._messages_received += 1
            now = time.time()

            # Gap detection
            if self._last_msg_time > 0:
                gap_ms = (now - self._last_msg_time) * 1000
                if gap_ms > self._gap_warn_threshold_s * 1000:
                    self._gap_count += 1
                    prev_max = self._max_gap_ms
                    self._max_gap_ms = max(self._max_gap_ms, gap_ms)
                    should_warn = (
                        (now - self._last_gap_warn_time) >= self._gap_warn_cooldown_s
                        or gap_ms > (prev_max + 1000.0)
                    )
                    if should_warn:
                        self._last_gap_warn_time = now
                        gap_log = ("warn", gap_ms, self._gap_count)
                    else:
                        gap_log = ("debug", gap_ms, self._gap_count)
            self._last_msg_time = now
            self._last_ws_activity_time = now

        if gap_log:
            level, gap_ms, gap_count = gap_log
            msg = (
                "PositionMonitor gap: %.0fms since last message (gap #%d, warn_threshold=%.1fs)"
                % (gap_ms, gap_count, self._gap_warn_threshold_s)
            )
            if level == "warn":
                logger.warning(msg)
            else:
                logger.debug(msg)

        try:
            data = json.loads(message)
            channel = data.get("channel")

            if channel == "userEvents":
                self._process_user_events(data.get("data", {}))
        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.debug(f"Message processing error: {e}")

    def _on_pong(self, ws, message) -> None:
        """Treat pong frames as transport activity for watchdog liveness."""
        with self._lock:
            self._last_ws_activity_time = time.time()

    def _consume_watchdog_trigger_locked(self, now: float) -> Optional[float]:
        """Return idle seconds when watchdog should force reconnect, else None.

        Must be called with self._lock held.
        """
        if not self._connected:
            return None
        last_activity = self._last_ws_activity_time or self._last_msg_time
        if last_activity <= 0:
            return None
        idle_s = now - last_activity
        if idle_s <= self._watchdog_timeout_s:
            return None
        if (now - self._last_watchdog_reconnect_time) < self._watchdog_reconnect_cooldown_s:
            return None
        self._last_watchdog_reconnect_time = now
        return idle_s

    def _on_error(self, ws, error) -> None:
        """Handle WebSocket errors."""
        if self._running:
            logger.warning(f"PositionMonitor WebSocket error: {error}")
        with self._lock:
            self._connected = False

    def _on_close(self, ws, close_status_code=None, close_msg=None) -> None:
        """Handle WebSocket closure."""
        with self._lock:
            self._connected = False
        if self._running:
            logger.info(f"PositionMonitor WebSocket closed (code={close_status_code})")

    # ─── Private: Position processing ─────────────────────────────

    def _process_user_events(self, event_data: Dict) -> None:
        """
        Process a userEvents message from the WebSocket.

        Format:
        {
            "user": "0x...",
            "fills": [...],
            "positions": [
                {
                    "coin": "BTC",
                    "szi": "1.5",
                    "entryPx": "67000",
                    "leverage": {"value": "5"},
                    ...
                }
            ]
        }
        """
        user = event_data.get("user", "")
        if not user:
            return

        # Parse positions from the event
        positions_data = event_data.get("positions", [])
        if not positions_data:
            return

        # Build current positions dict
        current_positions = {}
        for pos in positions_data:
            try:
                coin = pos.get("coin", "")
                if not coin:
                    continue

                size = float(pos.get("szi", 0))
                if size == 0:
                    continue  # Skip zero positions

                side = "long" if size > 0 else "short"
                size = abs(size)
                entry_px = float(pos.get("entryPx", 0))

                # Parse leverage (can be dict or float)
                leverage_data = pos.get("leverage", {})
                if isinstance(leverage_data, dict):
                    leverage = float(leverage_data.get("value", 1))
                else:
                    leverage = float(leverage_data)

                current_positions[coin] = {
                    "size": size,
                    "side": side,
                    "entry_price": entry_px,
                    "leverage": leverage,
                    "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                }
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"Error parsing position for {coin}: {e}")
                continue

        # Get cached positions for this user
        with self._lock:
            cached = self._position_cache.get(user, {})

        # Detect changes and emit signals
        signals = self._detect_position_changes(user, cached, current_positions)

        # Update cache
        with self._lock:
            self._position_cache[user] = current_positions
            for signal in signals:
                self._signal_queue.append(signal)
                self._signals_emitted += 1

        if signals:
            logger.debug(f"Detected {len(signals)} position changes for {user[:10]}")

    def _detect_position_changes(
        self,
        address: str,
        old_positions: Dict[str, Dict],
        new_positions: Dict[str, Dict],
    ) -> List[Dict]:
        """
        Detect new, closed, scaled, or flipped positions.

        Returns signals in the SAME format as copy_trader.py's
        _detect_position_changes() output.
        """
        signals = []
        old_coins = set(old_positions.keys())
        new_coins = set(new_positions.keys())

        # Read latest prices from dedicated mids-refresh cache.
        with self._lock:
            mids = dict(self._mids_cache)

        # 1. New positions opened
        for coin in new_coins - old_coins:
            pos = new_positions[coin]
            price = float(mids.get(coin, pos["entry_price"]))
            if price <= 0:
                continue

            signals.append({
                "type": "copy_open",
                "coin": coin,
                "side": pos["side"],
                "price": price,
                "leverage": min(pos["leverage"], config.PAPER_TRADING_MAX_LEVERAGE),
                "source_trader": address[:10],
                "source_pnl": 0,  # userEvents doesn't include account PnL, would need separate call
                "confidence": 0.85,  # High confidence for real-time detection
            })

        # 2. Positions closed (disappeared from positions array)
        for coin in old_coins - new_coins:
            signals.append({
                "type": "copy_close",
                "coin": coin,
                "source_trader": address[:10],
            })

        # 3. Scaled-in positions (>50% size increase)
        for coin in old_coins & new_coins:
            old_size = old_positions[coin]["size"]
            new_size = new_positions[coin]["size"]

            if new_size > old_size * 1.5:  # 50%+ increase
                pos = new_positions[coin]
                price = float(mids.get(coin, pos["entry_price"]))
                if price <= 0:
                    continue

                signals.append({
                    "type": "copy_scale_in",
                    "coin": coin,
                    "side": pos["side"],
                    "price": price,
                    "leverage": min(pos["leverage"], config.PAPER_TRADING_MAX_LEVERAGE),
                    "source_trader": address[:10],
                    "source_pnl": 0,
                    "confidence": 0.80,
                })

        # 4. Side flips (trader reversed position)
        for coin in old_coins & new_coins:
            if old_positions[coin]["side"] != new_positions[coin]["side"]:
                pos = new_positions[coin]
                price = float(mids.get(coin, pos["entry_price"]))
                if price <= 0:
                    continue

                signals.append({
                    "type": "copy_flip",
                    "coin": coin,
                    "side": pos["side"],
                    "price": price,
                    "leverage": min(pos["leverage"], config.PAPER_TRADING_MAX_LEVERAGE),
                    "source_trader": address[:10],
                    "source_pnl": 0,
                    "confidence": 0.90,
                })

        return signals

    def _refresh_mids_loop(self) -> None:
        """Refresh all-mids cache out-of-band from the WebSocket handler thread."""
        while self._running:
            try:
                mids_raw = hl.get_all_mids() or {}
                if isinstance(mids_raw, dict):
                    mids_clean = {}
                    for coin, px in mids_raw.items():
                        try:
                            price = float(px)
                            if price > 0:
                                mids_clean[coin] = price
                        except (TypeError, ValueError):
                            continue
                    if mids_clean:
                        with self._lock:
                            self._mids_cache = mids_clean
                            self._mids_last_refresh = time.time()
            except Exception as exc:
                logger.debug("PositionMonitor mids refresh error: %s", exc)
            time.sleep(self._mids_refresh_interval_s)

    def _watchdog_loop(self) -> None:
        """Force reconnect if WS is silent for too long."""
        check_interval = max(5.0, self._watchdog_timeout_s / 3.0)
        while self._running:
            time.sleep(check_interval)
            with self._lock:
                ws = self._ws
                now = time.time()
                idle_s = self._consume_watchdog_trigger_locked(now)
            if idle_s is None:
                continue
            logger.warning(
                "PositionMonitor watchdog: no WS activity for %.1fs (> %.1fs), forcing reconnect",
                idle_s,
                self._watchdog_timeout_s,
            )
            try:
                if ws:
                    ws.close()
            except Exception as exc:
                logger.debug("PositionMonitor watchdog close error: %s", exc)

    # ─── Bootstrap: Initial position fetch ──────────────────────────

    def _bootstrap_positions(self, address: str) -> Dict[str, Dict]:
        """
        Fetch initial positions via REST API before WebSocket takes over.
        One-time call per trader to populate cache at startup.

        Args:
            address: Trader address

        Returns:
            Current positions dict: {coin: {size, side, entry_price, leverage, ...}}
        """
        try:
            state = hl.get_user_state(address)
            if not state:
                return {}

            positions = {}
            for pos in state["positions"]:
                if pos["size"] > 0:  # Only non-zero positions
                    positions[pos["coin"]] = {
                        "size": pos["size"],
                        "side": pos["side"],
                        "entry_price": pos["entry_price"],
                        "leverage": pos["leverage"],
                        "unrealized_pnl": pos.get("unrealized_pnl", 0),
                    }

            with self._lock:
                self._position_cache[address] = positions

            logger.info(f"Bootstrapped {len(positions)} positions for {address[:10]}")
            return positions
        except Exception as e:
            logger.warning(f"Bootstrap failed for {address[:10]}: {e}")
            return {}
