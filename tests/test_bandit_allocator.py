"""A2: Thompson-sampling allocator tests.

Properties under test:
- Beta(1,1) prior on fresh arms (no opinion).
- Outcome updates move the posterior in the right direction.
- Exponential decay reverts an idle arm toward the prior.
- Wilson lower-bound is more conservative than the posterior mean,
  especially for small N.
- sample_weights() sums to 1 and caps exploratory share.
- Reproducible with a fixed-seed RNG.
- Snapshot/load round-trips state losslessly.
"""
from __future__ import annotations

import math
import random

import pytest

from src.signals.bandit_allocator import (
    DEFAULT_MIN_EVIDENCE_TRADES,
    DEFAULT_PRIOR_ALPHA,
    DEFAULT_PRIOR_BETA,
    ArmState,
    ThompsonAllocator,
)


def _fixed_rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


# ── ArmState ─────────────────────────────────────────────────────────


def test_fresh_arm_uses_uniform_prior():
    arm = ArmState(source="copy_trade:0xabc")
    assert arm.alpha == DEFAULT_PRIOR_ALPHA == 1.0
    assert arm.beta == DEFAULT_PRIOR_BETA == 1.0
    assert arm.posterior_mean() == pytest.approx(0.5)
    assert arm.n_observed == 0


def test_win_loss_updates_move_posterior_correctly():
    # Freeze time so wall-clock decay doesn't perturb the float math.
    arm = ArmState(source="src")
    arm.update(won=True, now_ts=0.0)
    assert arm.alpha == pytest.approx(2.0)
    assert arm.beta == pytest.approx(1.0)
    arm.update(won=False, now_ts=0.0)
    arm.update(won=False, now_ts=0.0)
    assert arm.alpha == pytest.approx(2.0)
    assert arm.beta == pytest.approx(3.0)
    assert arm.posterior_mean() == pytest.approx(2 / 5)


def test_wilson_lower_bound_more_conservative_than_mean():
    """For small N, Wilson lower bound must be strictly less than the
    posterior mean — that's the whole point: it punishes uncertainty."""
    arm = ArmState(source="src")
    for _ in range(3):
        arm.update(won=True, now_ts=0.0)
    # 3 wins, 0 losses: mean = 4/5 = 0.8; Wilson lower must be << 0.8
    assert arm.posterior_mean() > arm.wilson_lower_95()
    assert arm.wilson_lower_95() < 0.7


def test_decay_reverts_idle_arm_toward_prior():
    """After many half-lives of idleness, an arm's posterior must shrink
    back to ~the prior."""
    arm = ArmState(source="src", half_life_days=1.0, last_update_ts=0.0)
    # Bank 50 wins at t=0
    for _ in range(50):
        arm.update(won=True, now_ts=0.0)
    assert arm.alpha == pytest.approx(51.0)
    # Jump 100 half-lives forward in time
    arm._decay_to(now_ts=100 * 86400.0)
    # Informative mass decays to ~0; prior remains
    assert arm.alpha == pytest.approx(DEFAULT_PRIOR_ALPHA, abs=1e-6)
    assert arm.beta == pytest.approx(DEFAULT_PRIOR_BETA, abs=1e-6)


def test_decay_preserves_recent_updates():
    """Half-life of 30 days — an update from one day ago should retain
    most of its weight."""
    arm = ArmState(source="src", half_life_days=30.0, last_update_ts=0.0)
    for _ in range(10):
        arm.update(won=True, now_ts=0.0)
    initial_alpha = arm.alpha
    arm._decay_to(now_ts=86400.0)  # +1 day
    # alpha decays by factor 0.5^(1/30) ≈ 0.977 on the info mass only
    expected_info = (initial_alpha - DEFAULT_PRIOR_ALPHA) * (0.5 ** (1 / 30))
    expected_alpha = DEFAULT_PRIOR_ALPHA + expected_info
    assert arm.alpha == pytest.approx(expected_alpha, rel=1e-4)


# ── ThompsonAllocator ───────────────────────────────────────────────


def test_allocator_creates_arm_on_first_sample():
    alloc = ThompsonAllocator(rng=_fixed_rng())
    s = alloc.sample("copy_trade:0xnew")
    assert 0.0 <= s <= 1.0
    snap = alloc.arm_snapshot("copy_trade:0xnew")
    assert snap is not None
    assert snap["alpha"] == DEFAULT_PRIOR_ALPHA
    assert snap["beta"] == DEFAULT_PRIOR_BETA


def test_sample_weights_sum_to_one():
    alloc = ThompsonAllocator(rng=_fixed_rng())
    for s in ["a", "b", "c"]:
        for _ in range(DEFAULT_MIN_EVIDENCE_TRADES + 1):
            alloc.update(s, won=True)
    weights = alloc.sample_weights(["a", "b", "c"])
    assert math.isclose(sum(weights.values()), 1.0, rel_tol=1e-6)
    assert all(w > 0 for w in weights.values())


def test_sample_weights_caps_exploratory_share():
    """A brand-new arm (no evidence) cannot exceed the exploratory cap
    in allocation, even when a well-evidenced arm exists."""
    alloc = ThompsonAllocator(
        rng=_fixed_rng(),
        min_evidence_share_cap=0.10,
        min_evidence_trades=5,
    )
    # Well-evidenced arm with many wins
    for _ in range(20):
        alloc.update("veteran", won=True)
    # Brand-new arm with zero evidence
    # (allocator creates it lazily on sample)
    weights = alloc.sample_weights(["veteran", "newbie"])
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert weights["newbie"] <= 0.10 + 1e-9
    assert weights["veteran"] >= 0.90 - 1e-9


def test_pnl_convenience_treats_positive_as_win():
    alloc = ThompsonAllocator(rng=_fixed_rng())
    alloc.update_pnl("src", pnl=5.0, now_ts=0.0)
    alloc.update_pnl("src", pnl=-3.0, now_ts=0.0)
    alloc.update_pnl("src", pnl=0.0, now_ts=0.0)         # neither win nor loss strictly
    snap = alloc.arm_snapshot("src")
    # 1 win (5.0), 2 losses (-3, 0 with fee_floor=0 means 0 is NOT > 0)
    assert snap["alpha"] == pytest.approx(2.0)
    assert snap["beta"] == pytest.approx(3.0)


def test_pnl_fee_floor_filters_marginal_wins():
    alloc = ThompsonAllocator(rng=_fixed_rng())
    # Net wins above fee floor count, marginal wins don't
    alloc.update_pnl("src", pnl=10.0, fee_floor=2.0, now_ts=0.0)   # win (>2)
    alloc.update_pnl("src", pnl=1.0, fee_floor=2.0, now_ts=0.0)    # loss (<=2)
    snap = alloc.arm_snapshot("src")
    assert snap["alpha"] == pytest.approx(2.0)
    assert snap["beta"] == pytest.approx(2.0)


def test_sample_is_deterministic_with_fixed_seed():
    """Same seed + same updates -> identical samples."""
    rng1 = random.Random(123)
    rng2 = random.Random(123)
    a1 = ThompsonAllocator(rng=rng1)
    a2 = ThompsonAllocator(rng=rng2)
    for w in [True, True, False, True]:
        a1.update("s", won=w, now_ts=1.0)
        a2.update("s", won=w, now_ts=1.0)
    assert a1.sample("s", now_ts=2.0) == a2.sample("s", now_ts=2.0)


def test_snapshot_load_roundtrip():
    a1 = ThompsonAllocator(rng=_fixed_rng())
    for s, w in [("src_a", True), ("src_a", True), ("src_b", False)]:
        a1.update(s, won=w, now_ts=1.0)
    snap = a1.arms_snapshot()

    a2 = ThompsonAllocator(rng=_fixed_rng())
    a2.load_state(snap)

    snap2 = a2.arms_snapshot()
    assert set(snap.keys()) == set(snap2.keys())
    for s in snap:
        assert snap[s]["alpha"] == snap2[s]["alpha"]
        assert snap[s]["beta"] == snap2[s]["beta"]
        assert snap[s]["n_observed"] == snap2[s]["n_observed"]


def test_load_state_tolerates_garbage():
    """A malformed snapshot row must not crash boot."""
    alloc = ThompsonAllocator(rng=_fixed_rng())
    alloc.load_state({
        "good": {"alpha": 3.0, "beta": 2.0, "n_observed": 4},
        "bad_alpha": {"alpha": "garbage", "beta": 1.0},
        "missing_fields": {},
    })
    snap = alloc.arms_snapshot()
    assert snap.get("good", {}).get("alpha") == 3.0
    # bad_alpha should NOT be in snap (skipped on parse failure)
    assert "bad_alpha" not in snap
    # missing_fields populates with prior defaults
    assert snap.get("missing_fields", {}).get("alpha") == DEFAULT_PRIOR_ALPHA


def test_no_evidence_arms_fall_back_to_uniform_when_all_unevidenced():
    alloc = ThompsonAllocator(rng=_fixed_rng())
    weights = alloc.sample_weights(["a", "b", "c"])
    assert math.isclose(sum(weights.values()), 1.0, rel_tol=1e-6)
    # No arm should be exactly 0 since prior is non-degenerate
    assert all(w > 0 for w in weights.values())


def test_strong_winner_dominates_after_evidence():
    """Run a many-round simulation where source A wins 90% and source B
    wins 30%. After enough updates, the allocator should give A the
    majority share."""
    rng = random.Random(2026)
    alloc = ThompsonAllocator(rng=rng)
    # Bank 100 trades each with the stated win rates
    for i in range(100):
        alloc.update("A", won=rng.random() < 0.9)
        alloc.update("B", won=rng.random() < 0.3)
    # Now sample weights many times and average
    runs = 200
    sums = {"A": 0.0, "B": 0.0}
    for _ in range(runs):
        w = alloc.sample_weights(["A", "B"])
        sums["A"] += w["A"]
        sums["B"] += w["B"]
    avg_a = sums["A"] / runs
    avg_b = sums["B"] / runs
    assert avg_a > 0.7
    assert avg_b < 0.3


def test_empty_sources_returns_empty_dict():
    alloc = ThompsonAllocator(rng=_fixed_rng())
    assert alloc.sample_weights([]) == {}


def test_arm_snapshot_unknown_source_returns_none():
    alloc = ThompsonAllocator(rng=_fixed_rng())
    assert alloc.arm_snapshot("never_seen") is None
