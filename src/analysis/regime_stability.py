"""Regime hysteresis (#7).

The overall regime label is consumed as hard truth by ~12 gates
(market-side guard, strategy guidance, copy confidence capping). In the
production logs it flipped ``bullish`` -> ``crash`` -> ``neutral`` cycle
to cycle; each flip re-poisoned the gates. Full probabilistic regime
across every gate is a large, separately-reviewed refactor; this is the
high-value, low-risk core of it: a debounce so a *changed* label must
prove itself before the gates act on it, while a genuine high-confidence
regime change (e.g. a real crash) still passes through immediately.

Pure and dependency-free so it is exhaustively unit-testable; the caller
(RegimeDetector) owns the persisted state and the enable flag. With the
flag off this module is never invoked, so behavior is byte-identical.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# State shape carried across cycles by the caller:
#   {"effective": str|None, "streak": int, "pending": str|None,
#    "pending_count": int}
_EMPTY: Dict[str, Any] = {
    "effective": None,
    "streak": 0,
    "pending": None,
    "pending_count": 0,
}


def empty_state() -> Dict[str, Any]:
    return dict(_EMPTY)


def apply_regime_hysteresis(
    state: Optional[Dict[str, Any]],
    *,
    new_regime: str,
    new_confidence: float,
    min_streak: int,
    override_confidence: float,
) -> Tuple[str, Dict[str, Any]]:
    """Return ``(effective_regime, new_state)``.

    Rules:
      - First ever read: adopt it (no history to debounce against).
      - Same as current effective: reaffirm, clear any pending challenger.
      - Different from effective:
          * ``new_confidence >= override_confidence`` -> flip immediately
            (a real, high-confidence regime change must not be debounced;
            this is the crash-gets-through escape hatch).
          * else the challenger must appear ``min_streak`` consecutive
            cycles before it replaces the effective label; until then the
            previous effective label is held.
    """
    st = dict(state or _EMPTY)
    prev_effective = st.get("effective")
    new_regime = str(new_regime or "").strip()
    try:
        conf = float(new_confidence)
        if conf != conf:  # NaN
            conf = 0.0
    except (TypeError, ValueError):
        conf = 0.0
    min_streak = max(1, int(min_streak or 1))

    # First observation — nothing to debounce against.
    if not prev_effective:
        return new_regime, {
            "effective": new_regime,
            "streak": 1,
            "pending": None,
            "pending_count": 0,
        }

    # Reaffirming the current effective label.
    if new_regime == prev_effective:
        return prev_effective, {
            "effective": prev_effective,
            "streak": int(st.get("streak", 0) or 0) + 1,
            "pending": None,
            "pending_count": 0,
        }

    # A challenger. High-confidence change bypasses the debounce.
    if conf >= float(override_confidence):
        return new_regime, {
            "effective": new_regime,
            "streak": 1,
            "pending": None,
            "pending_count": 0,
        }

    # Otherwise the challenger must persist min_streak cycles.
    if st.get("pending") == new_regime:
        pending_count = int(st.get("pending_count", 0) or 0) + 1
    else:
        pending_count = 1
    if pending_count >= min_streak:
        return new_regime, {
            "effective": new_regime,
            "streak": 1,
            "pending": None,
            "pending_count": 0,
        }
    # Hold the previous effective label while the challenger proves itself.
    return prev_effective, {
        "effective": prev_effective,
        "streak": int(st.get("streak", 0) or 0),
        "pending": new_regime,
        "pending_count": pending_count,
    }
