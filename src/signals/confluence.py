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


def count_confirmations(signal) -> int:
    """Number of truthy independent confirmations on a signal."""
    n = 0
    if bool(getattr(signal, "options_flow_aligned", False)):
        n += 1
    if bool(getattr(signal, "volume_confirmed", False)):
        n += 1
    ctx = getattr(signal, "context", None)
    if isinstance(ctx, dict):
        for key in _CONTEXT_CONFIRMATION_KEYS:
            if bool(ctx.get(key)):
                n += 1
    return n


def _is_copy(signal) -> bool:
    src = getattr(signal, "source", None)
    src = str(getattr(src, "value", src) or "").strip().lower()
    stype = str(getattr(signal, "strategy_type", "") or "").strip().lower()
    return src.startswith("copy") or stype.startswith("copy")


def confluence_ok(signal, min_confirmations: int) -> Tuple[bool, str]:
    """``(allow, reason)``. Copy trades are exempt; ``min_confirmations <= 0``
    disables the gate (always allow). A non-copy entry below the threshold is
    rejected with a reason."""
    if min_confirmations <= 0 or _is_copy(signal):
        return True, ""
    c = count_confirmations(signal)
    if c < min_confirmations:
        return False, f"confluence {c} < {min_confirmations} independent confirmations"
    return True, ""
