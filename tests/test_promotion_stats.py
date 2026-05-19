"""A5: Deflated Sharpe + paired SPRT promotion-gate primitives.

The motivating worry: any continuous-learning loop will repeatedly
sample max-Sharpe candidates, and without a multiple-testing
correction the gate promotes overfit noise. DSR is the
canonical fix (Bailey & López de Prado 2014). SPRT lets the gate
stop as soon as evidence is conclusive without pre-committing to a
sample size.
"""
from __future__ import annotations

import math
import random

import pytest

from src.learning.promotion_stats import (
    DSRResult,
    SPRTResult,
    deflated_sharpe,
    sharpe_ratio,
    sprt_pair,
)


# ── Sharpe ratio sanity ────────────────────────────────────────────


def test_sharpe_zero_for_empty():
    assert sharpe_ratio([]) == 0.0
    assert sharpe_ratio([0.5]) == 0.0  # n=1 -> no variance


def test_sharpe_zero_for_zero_variance():
    assert sharpe_ratio([0.1, 0.1, 0.1, 0.1]) == 0.0


def test_sharpe_positive_for_positive_drift():
    rng = random.Random(2026)
    rets = [0.005 + 0.01 * rng.gauss(0, 1) for _ in range(200)]
    assert sharpe_ratio(rets) > 0


# ── Deflated Sharpe ────────────────────────────────────────────────


def test_dsr_correctly_penalises_high_trial_count():
    """The same return series with more trials must have a lower
    deflated Sharpe than with fewer — that's the whole point."""
    rng = random.Random(7)
    rets = [0.01 + 0.02 * rng.gauss(0, 1) for _ in range(100)]
    low = deflated_sharpe(rets, num_trials=1)
    high = deflated_sharpe(rets, num_trials=1000)
    assert isinstance(low, DSRResult)
    assert low.sharpe == high.sharpe
    assert low.deflated_sharpe > high.deflated_sharpe


def test_dsr_returns_zero_object_on_empty():
    result = deflated_sharpe([], num_trials=10)
    assert result.deflated_sharpe == 0.0
    assert not result.significant_at_95
    assert result.num_observations == 0


def test_dsr_flags_strong_signal_as_significant():
    """A clearly profitable strategy with low trial count must be
    flagged significant. Use mu/sigma = 0.5 on 200 obs — that's a
    Sharpe of 0.5 with sqrt(200-1) ≈ 14.1 standard errors over zero."""
    rng = random.Random(11)
    rets = [0.005 + 0.01 * rng.gauss(0, 1) for _ in range(200)]
    result = deflated_sharpe(rets, num_trials=3)
    assert result.significant_at_95


def test_dsr_does_not_flag_noise():
    """White noise must not be significant under realistic search
    space (num_trials=100). The DSR is *supposed* to penalize for
    multiple-comparison bias — that's the whole point — so we test
    against the realistic scenario, not the degenerate single-trial
    case where any non-zero sample mean is significant."""
    rng = random.Random(11)
    rets = [0.001 * rng.gauss(0, 1) for _ in range(200)]
    result = deflated_sharpe(rets, num_trials=100)
    assert not result.significant_at_95


def test_dsr_skew_kurt_override():
    """Passing extreme skew/kurt overrides what would otherwise be
    accepted. Captures the "fat-tail penalty" path."""
    rng = random.Random(13)
    rets = [0.01 + 0.02 * rng.gauss(0, 1) for _ in range(100)]
    plain = deflated_sharpe(rets, num_trials=5)
    fat_tail = deflated_sharpe(rets, num_trials=5, skew=-3.0, kurt=20.0)
    # Same returns, but heavier-tailed prior on the distribution should
    # lower DSR (or at minimum, not raise it).
    assert fat_tail.deflated_sharpe <= plain.deflated_sharpe


def test_dsr_p_value_in_unit_interval():
    rng = random.Random(17)
    rets = [0.003 + 0.015 * rng.gauss(0, 1) for _ in range(150)]
    for n_trials in (1, 5, 100, 10_000):
        result = deflated_sharpe(rets, num_trials=n_trials)
        assert 0.0 <= result.p_value <= 1.0


# ── SPRT ───────────────────────────────────────────────────────────


def test_sprt_continue_with_insufficient_samples():
    result = sprt_pair([0.01], [0.01])
    assert result.decision == "CONTINUE"
    assert result.num_observations == 1


def test_sprt_accept_when_challenger_clearly_better():
    """Challenger strictly above champion by a consistent edge —
    SPRT must ACCEPT after enough samples."""
    rng = random.Random(2)
    champ = [0.001 * rng.gauss(0, 1) for _ in range(120)]
    chal = [c + 0.02 for c in champ]   # +2% per period — way over mde=0.5%
    result = sprt_pair(chal, champ, alpha=0.05, beta=0.05, mde=0.005)
    assert result.decision == "ACCEPT"
    assert result.log_likelihood_ratio >= result.upper_threshold


def test_sprt_reject_when_challenger_clearly_worse():
    rng = random.Random(3)
    champ = [0.001 * rng.gauss(0, 1) for _ in range(120)]
    chal = [c - 0.02 for c in champ]  # -2% per period
    result = sprt_pair(chal, champ, alpha=0.05, beta=0.05, mde=0.005)
    assert result.decision == "REJECT"
    assert result.log_likelihood_ratio <= result.lower_threshold


def test_sprt_handles_identical_returns():
    """No information case: returns are identical. Decision: CONTINUE."""
    rets = [0.005] * 50
    result = sprt_pair(rets, rets)
    assert result.decision == "CONTINUE"


def test_sprt_thresholds_monotone_in_alpha_beta():
    """Smaller alpha (stricter) -> higher accept threshold.
    Smaller beta -> lower reject threshold (closer to 0 from below)."""
    a_strict = sprt_pair([0.01] * 10, [0.005] * 10, alpha=0.01, beta=0.01, mde=0.001)
    a_lax = sprt_pair([0.01] * 10, [0.005] * 10, alpha=0.10, beta=0.10, mde=0.001)
    assert a_strict.upper_threshold > a_lax.upper_threshold
    assert a_strict.lower_threshold < a_lax.lower_threshold


# ── Integration: typical promotion workflow ────────────────────────


def test_promotion_workflow_blocks_overfit_candidate():
    """End-to-end: a strategy that *looks* good but is in fact noise
    chosen from a large search space must not pass DSR significance."""
    rng = random.Random(999)
    # Generate 100 fake candidates and pick the one with the highest
    # raw Sharpe. This simulates exactly the failure mode DSR exists
    # to catch.
    candidates = [
        [0.001 * rng.gauss(0, 1) for _ in range(150)]
        for _ in range(100)
    ]
    best = max(candidates, key=sharpe_ratio)
    naive = sharpe_ratio(best)
    # Naive Sharpe looks high because we cherry-picked
    assert naive > 0.05  # nominal threshold; varies but positive
    # DSR with num_trials=100 must NOT promote
    dsr = deflated_sharpe(best, num_trials=100)
    assert not dsr.significant_at_95, (
        f"DSR with num_trials=100 should reject a cherry-picked noise "
        f"strategy. Got DSR p={dsr.p_value:.4f}"
    )


def test_promotion_workflow_promotes_genuine_alpha():
    """End-to-end: a strategy with a real, *detectable* edge passes at
    least one of DSR/SPRT. With per-period mean=0.002 and std=0.005
    (Sharpe ≈ 0.4 per period) over 200 obs at num_trials=3 — that's
    well above the multi-testing noise floor."""
    rng = random.Random(2024)
    real_alpha = [0.002 + 0.005 * rng.gauss(0, 1) for _ in range(200)]
    benchmark = [0.005 * rng.gauss(0, 1) for _ in range(200)]
    dsr = deflated_sharpe(real_alpha, num_trials=3)
    sprt = sprt_pair(real_alpha, benchmark, alpha=0.05, beta=0.05, mde=0.001)
    # A clearly profitable strategy should pass at least one gate
    assert dsr.significant_at_95 or sprt.decision == "ACCEPT", (
        f"At least one of DSR/SPRT must promote a real alpha; got "
        f"DSR p={dsr.p_value:.4f}, SPRT={sprt.decision}"
    )
