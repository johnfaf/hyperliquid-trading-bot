"""A7: Prometheus telemetry for trading decisions and signal rejection.

Both major bugs we shipped in the last week went *silent* because there
was no SLO on the right metrics:

- The 0.43 cascade: every copy_trade signal was getting rejected, but
  rejection-rate-by-source was not measured. Adding
  ``signal_rejection_total{stage,source,reason}`` would have surfaced
  it in hours, not weeks.

- The reconciliation metadata bug: 22/27 copy_trade closes had
  ``metadata.net_pnl_after_fees IS NULL``. Adding a gauge for the
  null-ratio would have flagged it in 15 minutes.

This module exposes a small, *tightly-scoped* set of trading-domain
counters / histograms / gauges. It gracefully no-ops if
``prometheus_client`` is not installed, so the bot starts fine on bare
requirements.

The ``/metrics`` endpoint is registered as a v2 dashboard router
(``src/ui/v2/routers/metrics.py``) and renders the standard Prometheus
text format on GET.

Disable everything by setting ``METRICS_ENABLED=false`` in the
environment. Default is ON (so we *get* the telemetry; the only cost
is a few atomic counter increments per signal).
"""
from __future__ import annotations

import os
import threading
from typing import Optional


_ENABLED = os.environ.get("METRICS_ENABLED", "true").lower() not in {"false", "0", "no", "off"}


# ── prometheus_client (optional) ───────────────────────────────────────
try:  # pragma: no cover - import guard
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    _PROM_AVAILABLE = True
except ImportError:  # pragma: no cover - degraded path
    CollectorRegistry = None  # type: ignore[assignment,misc]
    Counter = None  # type: ignore[assignment,misc]
    Gauge = None  # type: ignore[assignment,misc]
    Histogram = None  # type: ignore[assignment,misc]
    generate_latest = None  # type: ignore[assignment]
    CONTENT_TYPE_LATEST = "text/plain"
    _PROM_AVAILABLE = False


def metrics_available() -> bool:
    """True when prometheus_client is installed and metrics are enabled."""
    return _PROM_AVAILABLE and _ENABLED


# ── No-op fallback so call sites never branch on availability ──────────
class _NoOpMetric:
    """Drop-in replacement when prometheus_client is missing.

    All chainable methods return self so calls like
    ``metric.labels(...).inc()`` stay safe.
    """

    def labels(self, *_args, **_kwargs):
        return self

    def inc(self, _amount: float = 1.0) -> None:
        return None

    def dec(self, _amount: float = 1.0) -> None:
        return None

    def set(self, _value: float) -> None:
        return None

    def observe(self, _value: float) -> None:
        return None

    def time(self):
        class _Ctx:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()


# ── Registry + metric handles ──────────────────────────────────────────
_REGISTRY: Optional["CollectorRegistry"] = None
_INIT_LOCK = threading.Lock()
_INITIALIZED = False


# Module-global metric handles. Set by _build_metrics(); read by callers.
signal_emit_total = _NoOpMetric()              # Counter: signals emitted per source/symbol/regime
signal_rejection_total = _NoOpMetric()         # Counter: rejections per stage/reason/source
signal_accept_total = _NoOpMetric()            # Counter: signals that cleared all gates
confidence_distribution = _NoOpMetric()        # Histogram: raw confidence by source
decision_latency_seconds = _NoOpMetric()       # Histogram: cycle latency
pnl_metadata_null_ratio = _NoOpMetric()        # Gauge: % of closed trades with NULL net_pnl_after_fees
bandit_arm_pull_total = _NoOpMetric()          # Counter: Thompson allocator arm pulls (A2)
cascade_detector_flag_total = _NoOpMetric()    # Counter: A3 cascade detector flags raised
position_open_count = _NoOpMetric()            # Gauge: open positions snapshot


# Standard histogram buckets tuned to crypto-bot decision latencies.
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
_CONFIDENCE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95)


def _build_metrics() -> None:
    """Construct the real Prometheus metrics (idempotent)."""
    global _REGISTRY, _INITIALIZED
    global signal_emit_total, signal_rejection_total, signal_accept_total
    global confidence_distribution, decision_latency_seconds
    global pnl_metadata_null_ratio, bandit_arm_pull_total
    global cascade_detector_flag_total, position_open_count

    if _INITIALIZED:
        return
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        if not metrics_available():
            _INITIALIZED = True
            return

        _REGISTRY = CollectorRegistry(auto_describe=True)

        signal_emit_total = Counter(
            "signal_emit_total",
            "Trade signals produced before firewall filtering.",
            ["source", "symbol", "regime"],
            registry=_REGISTRY,
        )
        signal_rejection_total = Counter(
            "signal_rejection_total",
            "Signals rejected, labeled by firewall stage and reason.",
            ["stage", "reason", "source"],
            registry=_REGISTRY,
        )
        signal_accept_total = Counter(
            "signal_accept_total",
            "Signals that cleared every firewall stage.",
            ["source", "symbol"],
            registry=_REGISTRY,
        )
        confidence_distribution = Histogram(
            "signal_confidence",
            "Raw confidence distribution per signal source.",
            ["source"],
            buckets=_CONFIDENCE_BUCKETS,
            registry=_REGISTRY,
        )
        decision_latency_seconds = Histogram(
            "decision_latency_seconds",
            "Decision pipeline latency per stage.",
            ["stage"],
            buckets=_LATENCY_BUCKETS,
            registry=_REGISTRY,
        )
        pnl_metadata_null_ratio = Gauge(
            "pnl_metadata_null_ratio",
            "Fraction of recent closed trades with NULL "
            "metadata.net_pnl_after_fees (should be ~0).",
            registry=_REGISTRY,
        )
        bandit_arm_pull_total = Counter(
            "bandit_arm_pull_total",
            "Thompson-sampling allocator arm pulls per source.",
            ["source"],
            registry=_REGISTRY,
        )
        cascade_detector_flag_total = Counter(
            "cascade_detector_flag_total",
            "Cascade detector flag raised — confidence dead-band collapse.",
            ["source"],
            registry=_REGISTRY,
        )
        position_open_count = Gauge(
            "position_open_count",
            "Snapshot of currently open positions.",
            ["venue"],
            registry=_REGISTRY,
        )

        _INITIALIZED = True


def get_registry() -> Optional["CollectorRegistry"]:
    """Return the metrics registry (None if disabled or unavailable)."""
    _build_metrics()
    return _REGISTRY


def render_metrics() -> bytes:
    """Render the registry in Prometheus text format.

    Returns an empty body when metrics are disabled/unavailable so the
    ``/metrics`` endpoint can still respond 200 without scaring scrapers.
    """
    _build_metrics()
    if not metrics_available() or _REGISTRY is None:
        return b""
    return generate_latest(_REGISTRY)


def content_type() -> str:
    """Return the correct Content-Type for the ``/metrics`` response."""
    return CONTENT_TYPE_LATEST


# Initialize lazily on import so callers using the module-level handles
# get real metrics if prometheus_client is importable.
_build_metrics()
