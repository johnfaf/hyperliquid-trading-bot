"""Pub-sub event bus for the v2 dashboard.

The bot's cycles call :func:`publish` with a dict; every connected
WebSocket client gets the event. The bus is intentionally tiny —
no persistence, no replay, no backpressure beyond a per-client
bounded queue. If a client falls behind, we drop oldest events for
that client rather than block the producer.

This module is asyncio-aware. The bot's cycles run in plain threads,
so they call :func:`publish_threadsafe`, which schedules onto the
running event loop without requiring the producer to know about
asyncio.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

_PER_CLIENT_QUEUE_MAX = 100


class EventBus:
    def __init__(self) -> None:
        self._subscribers: Set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the bus to the FastAPI app's event loop.

        Producer threads (the trading cycles) need the loop to schedule
        ``publish``. We capture it once at app startup; tests can
        re-bind by calling this again.
        """
        self._loop = loop

    async def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=_PER_CLIENT_QUEUE_MAX)
        async with self._lock:
            self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.discard(queue)

    async def publish(self, event: Dict[str, Any]) -> None:
        """Fan out an event to every subscriber.

        On a full queue we drop the oldest entry rather than blocking
        the publisher -- a slow client must not stall the trading loop.
        """
        # Snapshot under the lock so a concurrent unsubscribe doesn't
        # race with the iteration.
        async with self._lock:
            targets = list(self._subscribers)
        for queue in targets:
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(event)

    def publish_threadsafe(self, event: Dict[str, Any]) -> None:
        """Schedule a ``publish`` from a non-asyncio thread.

        Idempotent if the loop hasn't been bound yet (calls are
        silently dropped). The bot's cycles run independently of the
        dashboard, so a missing dashboard mustn't break trading.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self.publish(event), loop)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("publish_threadsafe failed: %s", exc)


_bus = EventBus()


def get_bus() -> EventBus:
    return _bus


def publish_event(kind: str, **payload: Any) -> None:
    """Convenience wrapper used by the bot's cycles.

    The dashboard treats ``kind`` as a content-type tag (``cycle``,
    ``kill_switch``, ``calibration``, ...). Payload is opaque JSON.
    """
    event = {"kind": kind, "ts": time.time(), **payload}
    _bus.publish_threadsafe(event)
