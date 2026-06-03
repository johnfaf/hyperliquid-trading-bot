"""Meta-labeling: size a proposed trade by its calibrated win-probability
(signal #8, PR-A).

A primary signal proposes a trade; a secondary model estimates P(this specific
trade wins) and we size by it. v1 uses the calibration tracker as the secondary
model -- it IS a learned, evidence-weighted win-probability estimator
(``proven_evidence(source, side, regime) -> (edge, n)`` returns the
empirical-Bayes edge + sample size). Size then scales with edge above breakeven,
mirroring the edge-proportional-leverage logic so the two compose sanely.

Flag-gated (META_LABEL_ENABLED, default OFF). Observe-first: with no evidence
the win-prob falls back to (capped) signal confidence and the multiplier sits at
its neutral floor -- an unproven signal is never sized UP.
"""
from __future__ import annotations


def meta_win_probability(signal_confidence, calibrated_edge=None, n=None,
                         *, min_n: float = 30.0) -> float:
    """P(win) for a proposed trade.

    Uses the calibrated edge when there is enough source evidence
    (``n >= min_n``); otherwise falls back to the signal's own confidence,
    capped at 0.5 so an unproven source can't assert a positive edge. Always in
    [0.05, 0.95]."""
    try:
        conf = max(0.0, min(float(signal_confidence), 1.0))
    except (TypeError, ValueError):
        conf = 0.5
    if calibrated_edge is not None and n is not None:
        try:
            if float(n) >= float(min_n):
                return max(0.05, min(float(calibrated_edge), 0.95))
        except (TypeError, ValueError):
            pass
    return min(conf, 0.5)


def meta_size_multiplier(p_win, *, neutral: float = 0.5, full: float = 0.65,
                         min_mult: float = 0.25, max_mult: float = 1.5) -> float:
    """Position-size multiplier scaling with edge above breakeven: at or below
    ``neutral`` -> ``min_mult``; ramps linearly to ``max_mult`` as ``p_win``
    reaches ``full``. Monotonic non-decreasing, clamped to [min_mult, max_mult].
    """
    try:
        p = float(p_win)
    except (TypeError, ValueError):
        return float(min_mult)
    if full <= neutral or p <= neutral:
        return float(min_mult)
    frac = min(1.0, (p - neutral) / (full - neutral))
    return float(min_mult + frac * (max_mult - min_mult))


def meta_label_size_factor(signal_confidence, calibrated_edge=None, n=None,
                           *, min_n: float = 30.0, neutral: float = 0.5,
                           full: float = 0.65, min_mult: float = 0.25,
                           max_mult: float = 1.5) -> float:
    """Convenience: win-probability -> size multiplier in one call."""
    p = meta_win_probability(signal_confidence, calibrated_edge, n, min_n=min_n)
    return meta_size_multiplier(p, neutral=neutral, full=full,
                                min_mult=min_mult, max_mult=max_mult)
