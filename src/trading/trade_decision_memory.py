"""Capture the decision inputs that were actually available at execution."""
from datetime import datetime, timezone
import math

from src.core import clock_provider


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError, OverflowError):
        return None


def utc_stamp(value):
    if value is None:
        return None
    try:
        stamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return stamp.astimezone(timezone.utc).isoformat() if stamp.tzinfo is not None else None


COPY_REASONS = {
    "copy_open": "Followed source emitted a new-position signal.",
    "copy_scale_in": "Followed source emitted a scale-in signal.",
    "copy_flip": "Followed source emitted a direction-change signal.",
    "golden_copy": "A tracked golden-wallet signal was selected.",
}


def entry_memory(signal, *, route, reason=None, firewall=None, rotation=None, copy=None):
    """Build a small, versioned record. Do not persist arbitrary provider context."""
    context = getattr(signal, "context", None)
    context = context if isinstance(context, dict) else {}
    attribution = context.get("decision_attribution")
    attribution = attribution if isinstance(attribution, dict) else {}
    edge = context.get("ev_breakdown")
    edge = edge if isinstance(edge, dict) else {}
    data = {
        "version": 1,
        "route": route,
        "selected_at": clock_provider.utc_now().isoformat(),
        "signal_at": utc_stamp(getattr(signal, "timestamp", None)),
        "reason": str(reason or attribution.get("reason") or getattr(signal, "reason", "") or "")[:450] or None,
        "confidence": number(getattr(signal, "confidence", None)),
        "regime": str(getattr(signal, "regime", "") or "")[:60] or None,
        "firewall": firewall,
        "rotation": str(rotation or "")[:220] or None,
        "edge_bps": number(edge.get("ev_bps")),
        "estimated_cost_bps": number(edge.get("cost_bps")),
    }
    for key in ("volume_confirmed", "options_flow_aligned"):
        value = attribution.get(key, getattr(signal, key, None))
        if isinstance(value, bool):
            data[key] = value
    if copy is not None:
        inputs = copy.get("confidence_inputs")
        inputs = inputs if isinstance(inputs, dict) else {}
        data["copy"] = {
            "event": str(copy.get("type") or "")[:35],
            "source_entry_price": number(copy.get("source_entry_price")),
            "lag_bps": number(copy.get("copy_lag_bps")),
            "source_win_rate": number(inputs.get("win_rate")),
            "source_trade_count": number(inputs.get("trade_count")),
            "size_multiplier": number(copy.get("copy_source_size_multiplier")),
            "regime_size_modifier": number(copy.get("regime_size_modifier")),
        }
    return data
