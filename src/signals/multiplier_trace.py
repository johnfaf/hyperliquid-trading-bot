"""A3: Runtime multiplier-cascade trace.

A thin, dependency-free trace recorder for confidence multipliers applied
during the firewall+filter+scorer pipeline. Each call records:

    (signal_id, source, stage, multiplier_name, multiplier_value)

A small in-process ring buffer holds the last N records so the dashboard
can render a "live multiplier-product histogram per source" — the panel
that would have made the 0.43 cascade visible in hours instead of weeks.

This is *not* a metrics system (A7 will add Prometheus). It is a focused
postmortem-friendly buffer: the dashboard reads `recent_traces()` and
groups by source to surface "is any source's effective confidence after
multipliers collapsing into a tight band?"

Disable in production by setting MULTIPLIER_TRACE_ENABLED=false in env.
Default ON because it is bounded-memory (deque maxlen) and read-only on
the hot path.
"""
from __future__ import annotations

import os
import threading
from collections import deque
from dataclasses import dataclass, field
from time import time
from typing import Deque, Dict, List, Optional


_MAX_TRACE_RECORDS = int(os.environ.get("MULTIPLIER_TRACE_MAX", 5000))
_TRACE_ENABLED = os.environ.get("MULTIPLIER_TRACE_ENABLED", "true").lower() not in {
    "false", "0", "no", "off",
}


@dataclass(frozen=True)
class MultiplierEvent:
    """One confidence multiplier application."""
    ts: float
    signal_id: str
    source: str
    stage: str             # e.g. "firewall.long_hardening", "llm_filter.exhaustion"
    multiplier: float
    confidence_before: float
    confidence_after: float


@dataclass
class CascadeStats:
    """Per-source aggregate over the trace buffer."""
    source: str
    n: int = 0
    product_mean: float = 1.0
    product_std: float = 0.0
    confidence_after_min: float = 1.0
    confidence_after_max: float = 0.0
    confidence_after_mean: float = 0.0
    distinct_stages: List[str] = field(default_factory=list)


class _TraceBuffer:
    """Thread-safe bounded ring buffer for multiplier events."""

    def __init__(self, maxlen: int = _MAX_TRACE_RECORDS) -> None:
        self._buf: Deque[MultiplierEvent] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, event: MultiplierEvent) -> None:
        with self._lock:
            self._buf.append(event)

    def recent(self, n: Optional[int] = None) -> List[MultiplierEvent]:
        with self._lock:
            if n is None:
                return list(self._buf)
            if n <= 0:
                return []
            return list(self._buf)[-n:]

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def __len__(self) -> int:  # pragma: no cover — trivial
        return len(self._buf)


_BUFFER = _TraceBuffer()


def record_multiplier(
    signal_id: str,
    source: str,
    stage: str,
    multiplier: float,
    confidence_before: float,
    confidence_after: float,
) -> None:
    """Record a multiplier application. No-op when disabled.

    Safe to call from any thread. Constant-time (deque append + lock).
    Failures are silently swallowed — this is a trace, never the cause
    of a trading-path crash.
    """
    if not _TRACE_ENABLED:
        return
    try:
        _BUFFER.record(MultiplierEvent(
            ts=time(),
            signal_id=str(signal_id or ""),
            source=str(source or ""),
            stage=str(stage or ""),
            multiplier=float(multiplier),
            confidence_before=float(confidence_before),
            confidence_after=float(confidence_after),
        ))
    except Exception:
        pass


def recent_traces(n: Optional[int] = None) -> List[MultiplierEvent]:
    """Return the latest `n` (or all) recorded multiplier events."""
    return _BUFFER.recent(n)


def clear_traces() -> None:
    """Empty the trace buffer (intended for tests)."""
    _BUFFER.clear()


def cascade_stats_by_source(events: Optional[List[MultiplierEvent]] = None) -> Dict[str, CascadeStats]:
    """Aggregate trace events by source so the dashboard can render a
    "is any source's effective confidence collapsing?" view.

    The signal of a cascade is a *narrow* `confidence_after` band across
    many signals from the same source: e.g. all of source X land between
    [0.42, 0.44] after multipliers. That's what the 0.43 deadlock looked
    like in retrospect — the panel would have made it immediately obvious.
    """
    if events is None:
        events = recent_traces()
    by_source: Dict[str, List[MultiplierEvent]] = {}
    for ev in events:
        by_source.setdefault(ev.source, []).append(ev)

    out: Dict[str, CascadeStats] = {}
    for source, evs in by_source.items():
        if not evs:
            continue
        afters = [e.confidence_after for e in evs]
        products = [
            (e.confidence_after / e.confidence_before)
            if e.confidence_before > 0 else 0.0
            for e in evs
        ]
        n = len(evs)
        mean_product = sum(products) / n
        var_product = sum((p - mean_product) ** 2 for p in products) / n
        out[source] = CascadeStats(
            source=source,
            n=n,
            product_mean=mean_product,
            product_std=var_product ** 0.5,
            confidence_after_min=min(afters),
            confidence_after_max=max(afters),
            confidence_after_mean=sum(afters) / n,
            distinct_stages=sorted({e.stage for e in evs}),
        )
    return out


def detect_cascade(
    stats: Dict[str, CascadeStats],
    *,
    min_samples: int = 10,
    dead_band_width: float = 0.01,
    floor: float = 0.45,
) -> List[str]:
    """Return source keys whose `confidence_after` distribution is
    collapsed into a tight band near the source floor — the cascade
    signature.

    A source with >= `min_samples` events whose
    (max - min) `confidence_after` < `dead_band_width` AND mean is within
    `dead_band_width` of `floor` is flagged.
    """
    flagged: List[str] = []
    for source, st in stats.items():
        if st.n < min_samples:
            continue
        span = st.confidence_after_max - st.confidence_after_min
        if span < dead_band_width and abs(st.confidence_after_mean - floor) < dead_band_width:
            flagged.append(source)
    return flagged
