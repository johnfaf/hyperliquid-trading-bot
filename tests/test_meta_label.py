"""Meta-labeling: win-probability -> size multiplier (signal #8, PR-A)."""
from __future__ import annotations

from src.signals.meta_label import (
    meta_label_size_factor,
    meta_size_multiplier,
    meta_win_probability,
)


def test_win_prob_uses_calibrated_edge_when_enough_evidence():
    # n >= min_n -> trust the calibrated edge over raw confidence
    assert meta_win_probability(0.50, calibrated_edge=0.62, n=50, min_n=30) == 0.62


def test_win_prob_falls_back_capped_when_thin():
    # n < min_n -> fall back to signal confidence, capped at 0.5 (no edge claim)
    assert meta_win_probability(0.90, calibrated_edge=0.62, n=5, min_n=30) == 0.5
    assert meta_win_probability(0.30, calibrated_edge=None, n=None) == 0.30


def test_size_multiplier_floor_and_ramp():
    assert meta_size_multiplier(0.50) == 0.25           # at neutral -> floor
    assert meta_size_multiplier(0.40) == 0.25           # below neutral -> floor
    assert meta_size_multiplier(0.65) == 1.5            # at full -> cap
    assert meta_size_multiplier(0.80) == 1.5            # above full -> clamped
    mid = meta_size_multiplier(0.575)                   # halfway -> 0.25 + 0.5*(1.25)=0.875
    assert abs(mid - 0.875) < 1e-9


def test_size_multiplier_monotonic():
    xs = [0.5, 0.55, 0.6, 0.65, 0.7]
    mults = [meta_size_multiplier(x) for x in xs]
    assert mults == sorted(mults)


def test_size_factor_composition():
    # proven high-edge source -> sizes up; thin low-conf -> floored down
    assert meta_label_size_factor(0.5, calibrated_edge=0.65, n=100, min_n=30) == 1.5
    assert meta_label_size_factor(0.45, calibrated_edge=None, n=None) == 0.25
