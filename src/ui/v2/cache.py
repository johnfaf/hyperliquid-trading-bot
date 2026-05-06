"""Small TTL cache for dashboard read paths.

The v2 dashboard polls several expensive endpoints from multiple widgets.
This cache is deliberately tiny and in-process: it removes duplicate work
inside a short operator-refresh window without becoming a second source of
truth for live trading state.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Hashable, Optional


@dataclass
class _CacheEntry:
    expires_at: float
    value: Any


_LOCK = threading.RLock()
_CACHE: dict[Hashable, _CacheEntry] = {}


def get_ttl(key: Hashable, ttl_s: float, builder: Callable[[], Any]) -> Any:
    """Return a deepcopy of a cached value, rebuilding after ``ttl_s``.

    Values are copied on read/write so route handlers can safely mutate a
    response payload without contaminating the next request.
    """
    now = time.monotonic()
    ttl = max(0.0, float(ttl_s or 0.0))
    if ttl > 0:
        with _LOCK:
            entry = _CACHE.get(key)
            if entry is not None and entry.expires_at > now:
                return deepcopy(entry.value)

    value = builder()
    if ttl > 0:
        with _LOCK:
            _CACHE[key] = _CacheEntry(now + ttl, deepcopy(value))
    return deepcopy(value)


def invalidate(prefix: Optional[Hashable] = None) -> None:
    """Clear all cache entries, or entries whose tuple key starts with prefix."""
    with _LOCK:
        if prefix is None:
            _CACHE.clear()
            return
        if isinstance(prefix, tuple):
            doomed = [k for k in _CACHE if isinstance(k, tuple) and k[: len(prefix)] == prefix]
        else:
            doomed = [k for k in _CACHE if k == prefix or (isinstance(k, tuple) and k[:1] == (prefix,))]
        for key in doomed:
            _CACHE.pop(key, None)
