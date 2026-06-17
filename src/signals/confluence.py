"""Confluence gate: require N independent confirmations for non-copy entries
(signal #3).

A lone signal is lower quality; a trade backed by *independent* evidence
(options flow, multi-exchange volume, regime alignment, cross-venue confirmation)
is higher quality. This counts those confirmations and gates entries below a
threshold. copy_trade is exempt -- it's the proven edge and is sourced
differently. Pure + flag-gated (FIREWALL_CONFLUENCE_MIN_CONFIRMATIONS, 0 = OFF).
"""
from __future__ import annotations

from typing import Tuple

# Context keys (besides the explicit TradeSignal booleans) that count as an
# independent confirmation when truthy.
_CONTEXT_CONFIRMATION_KEYS = ("regime_aligned", "cross_venue_confirmed",
                              "liquidation_confirmed", "macro_aligned")


def _regime_aligned(signal, regime_data) -> bool:
    """Whether a signal agrees with the current regime read. Derived from
    ``regime_data`` (the firewall already has it) because the signal's own
    ``context['regime_aligned']`` is almost never pre-populated upstream --
    without this, regime alignment can never count as a confirmation and the
    gate is unsatisfiable. Defensive: any malformed input -> False."""
    if not isinstance(regime_data, dict):
        return False
    try:
        stype = str(getattr(signal, "strategy_type", "") or "").strip().lower()
        guidance = regime_data.get("strategy_guidance") or {}
        activate = {str(s).strip().lower() for s in guidance.get("activate", []) or []}
        pause = {str(s).strip().lower() for s in guidance.get("pause", []) or []}
        if stype and stype in pause:
            return False
        if stype and stype in activate:
            return True
        side = getattr(signal, "side", None)
        side = str(getattr(side, "value", side) or "").strip().lower()
        regime = str(regime_data.get("overall_regime")
                     or regime_data.get("regime") or "").strip().lower()
        if side in ("long", "buy") and "up" in regime:
            return True
        if side in ("short", "sell") and "down" in regime:
            return True
    except Exception:
        return False
    return False


def count_confirmations(signal, regime_data=None) -> int:
    """Number of truthy independent confirmations on a signal.

    ``regime_data`` (optional) lets us derive the regime-alignment confirmation
    when the signal context doesn't carry it explicitly -- the common case, so
    passing it is what makes the gate able to pass at all for real signals."""
    n = 0
    if bool(getattr(signal, "options_flow_aligned", False)):
        n += 1
    if bool(getattr(signal, "volume_confirmed", False)):
        n += 1
    ctx = getattr(signal, "context", None)
    ctx = ctx if isinstance(ctx, dict) else {}
    counted_regime = False
    for key in _CONTEXT_CONFIRMATION_KEYS:
        if bool(ctx.get(key)):
            n += 1
            if key == "regime_aligned":
                counted_regime = True
    if not counted_regime and _regime_aligned(signal, regime_data):
        n += 1
    return n


def _is_copy(signal) -> bool:
    src = getattr(signal, "source", None)
    src = str(getattr(src, "value", src) or "").strip().lower()
    stype = str(getattr(signal, "strategy_type", "") or "").strip().lower()
    return src.startswith("copy") or stype.startswith("copy")


def confluence_ok(signal, min_confirmations: int, regime_data=None) -> Tuple[bool, str]:
    """``(allow, reason)``. Copy trades are exempt; ``min_confirmations <= 0``
    disables the gate (always allow). A non-copy entry below the threshold is
    rejected with a reason. ``regime_data`` is forwarded so regime alignment can
    count as a confirmation."""
    if min_confirmations <= 0 or _is_copy(signal):
        return True, ""
    c = count_confirmations(signal, regime_data)
    if c < min_confirmations:
        return False, f"confluence {c} < {min_confirmations} independent confirmations"
    return True, ""
