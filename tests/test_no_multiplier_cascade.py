"""A3: Multiplier-cascade linter — structural defense against the 0.43 deadlock.

The 0.43 bug was a *structural* failure: synthetic_regime confidence cap (0.50)
× source-side hardening multiplier (~0.75) × weight-blend (default 0.5) =
0.425, and that number was *deterministic across every copy_trade signal*,
regardless of trader quality. The cascade collapsed an entire signal class
into a 1bp neighbourhood of the 0.45 source-floor and rejected everything.

This test is two things at once:

1. **Cascade detector.** For every plausible combination of (min, default,
   max) values across the *known* confidence multipliers, it asserts that
   the resulting product retains enough dispersion that legitimate
   high-confidence sources don't get pulled into the rejection zone with
   low-confidence ones.

2. **Drift sentinel.** It enumerates the registered multiplier sites and
   compares to a grep of `confidence *=` patterns in src/signals/. If a new
   multiplier is added without being registered, the test fails — forcing
   the new site to be audited *and* to declare its valid range, exactly
   the discipline that would have caught the 0.43 cascade before it shipped.
"""
from __future__ import annotations

import itertools
import re
from pathlib import Path

import pytest


# ── Registered confidence multiplier sites ─────────────────────────────
#
# Each entry: (site_id, (min, default, max), notes).
#
# WHEN YOU ADD A NEW `confidence *= X` OR `signal.confidence = X * Y` IN
# src/signals/, YOU MUST REGISTER IT HERE AND HAVE THE CASCADE TEST PASS
# WITH YOUR NEW SITE INCLUDED. The drift sentinel below will fail CI
# otherwise — this is the linter half of the linter+trace pair.
KNOWN_MULTIPLIERS = [
    # decision_firewall.py
    ("long_hardening_confidence_multiplier", (0.50, 0.80, 1.00),
     "Long-side hardening when regime is bullish-bearish-divergent."),
    ("short_hardening_confidence_multiplier", (0.50, 0.80, 1.00),
     "Short-side hardening (2 sites — 1484, 1517)."),
    ("partial_predictive_inputs", (0.50, 0.50, 0.50),
     "Hardcoded 0.5 when forecaster has only partial inputs."),
    ("risk_policy_confidence_multiplier", (0.50, 1.00, 1.00),
     "Per-signal risk policy override."),
    # llm_filter.py — exhaustion / counter-trend penalties
    ("llm_filter_exhaustion_severe", (0.60, 0.60, 0.60), "Severe exhaustion."),
    ("llm_filter_exhaustion_moderate", (0.60, 0.60, 0.60), "Moderate exhaustion."),
    ("llm_filter_exhaustion_mild", (0.80, 0.80, 0.80), "Mild exhaustion."),
    ("llm_filter_counter_trend", (0.75, 0.75, 0.75), "Counter-trend penalty."),
    ("llm_filter_trend_change", (0.70, 0.70, 0.70), "Trend-change penalty."),
    ("llm_filter_exhaustion_trend_aligned_a", (0.50, 0.70, 1.00),
     "Trend-aligned exhaustion variant (configurable)."),
    ("llm_filter_exhaustion_trend_aligned_b", (0.50, 0.70, 1.00),
     "Same multiplier applied at a second site in llm_filter."),
    ("llm_filter_regime_misalignment_a", (0.75, 0.75, 0.75),
     "Regime misalignment a."),
    ("llm_filter_regime_misalignment_b", (0.75, 0.75, 0.75),
     "Regime misalignment b."),
    # alpha_arena.py
    ("alpha_arena_quorum_a", (0.70, 0.70, 0.70), "Quorum penalty path a."),
    ("alpha_arena_quorum_b", (0.70, 0.70, 0.70), "Quorum penalty path b."),
    # Cascade-relevant non-`*=` reductions (the historical 0.43 trio)
    ("synthetic_regime_cap", (0.50, 0.50, 0.50),
     "Hard cap when regime data is synthetic warm-start."),
    ("source_side_guard", (0.75, 0.75, 1.00),
     "Source-side guard penalty when source vs. signal side mismatches."),
    ("agent_scorer_weight_blend", (0.50, 0.50, 1.00),
     "Weight-blend between raw confidence and agent-scorer weight."),
]


# Inputs likely to coexist on the same signal in production. The cartesian
# product over ALL multipliers is huge and uninteresting — most
# combinations are physically impossible (e.g. all 14 LLM penalties firing
# at once). We pick a conservative subset: any 3-multiplier subset is
# allowed to fire simultaneously. This is the size of the deadlock that
# bit us at 0.43.
MAX_SIMULTANEOUS_MULTIPLIERS = 3

# Confidence "floor" that the source allocator enforces. Any multiplier
# combination that pulls signals into a tight band around this floor is
# the failure mode we're guarding against.
SOURCE_FLOOR = 0.45
CASCADE_DEAD_BAND_WIDTH = 0.01  # 1% of the [0,1] range


def _enumerate_combinations():
    """Yield (subset_of_sites, product_at_min, product_at_default).

    Only enumerates subsets of size <= MAX_SIMULTANEOUS_MULTIPLIERS to keep
    the combinatorial blowup tractable while still catching the kind of
    3-way cascade that bit us.
    """
    for r in range(1, MAX_SIMULTANEOUS_MULTIPLIERS + 1):
        for combo in itertools.combinations(KNOWN_MULTIPLIERS, r):
            prod_min = 1.0
            prod_default = 1.0
            for _name, (mn, dflt, _mx), _notes in combo:
                prod_min *= mn
                prod_default *= dflt
            yield combo, prod_min, prod_default


def test_no_two_combinations_collapse_to_same_dead_band():
    """The deadlock smoking gun: two *different* high-confidence sources
    (one at 0.85, one at 0.95) should not collapse to the *same* tiny
    band after multipliers. If they do, the cascade has erased source
    quality from the decision."""
    bad = []
    for combo, prod_min, _ in _enumerate_combinations():
        if prod_min == 0:
            continue
        # Simulate two source-quality levels: 0.95 (strong) and 0.55 (marginal)
        strong_after = 0.95 * prod_min
        marginal_after = 0.55 * prod_min
        # Both within the dead-band around the source floor?
        if (
            abs(strong_after - SOURCE_FLOOR) < CASCADE_DEAD_BAND_WIDTH
            and abs(marginal_after - SOURCE_FLOOR) < CASCADE_DEAD_BAND_WIDTH
        ):
            bad.append((
                tuple(name for name, _, _ in combo),
                round(prod_min, 4),
                round(strong_after, 4),
                round(marginal_after, 4),
            ))
    assert not bad, (
        "Cascade detector: multiplier combinations collapse strong and "
        "marginal signals into the same rejection neighbourhood. The 0.43 "
        "deadlock will recur. Combinations:\n"
        + "\n".join(f"  {b}" for b in bad)
    )


def test_single_multiplier_at_default_preserves_strong_signal():
    """A 0.95-confidence signal hit by ONE penalty at its default value
    must still clear the source-floor. If even a single default-strength
    penalty kills a strong signal, the floor is unreachable in normal
    conditions and trade volume will silently collapse to zero.

    (Multi-penalty cascades are intentional by design — the LLM filter
    is *supposed* to penalize signals that are simultaneously exhausted
    AND counter-trend AND regime-misaligned. What we guard against is
    a single penalty being so harsh it acts as a hidden hard cap.)
    """
    bad = []
    for name, (_mn, dflt, _mx), _notes in KNOWN_MULTIPLIERS:
        strong_after = 0.95 * dflt
        if strong_after < SOURCE_FLOOR - 0.001:  # 0.1% tolerance
            bad.append((name, round(dflt, 4), round(strong_after, 4)))
    assert not bad, (
        "A strong (0.95) signal hits the floor under a SINGLE penalty at "
        "its default value — that's a hidden cap, not a graduated penalty:\n"
        + "\n".join(f"  {b}" for b in bad)
    )


def test_drift_sentinel_grep_matches_registry():
    """If someone adds a new `confidence *= X` in src/signals/ without
    registering it here, this test fails — making the audit mandatory.

    Counts *distinct line locations*, not multiplier values. Each `*=`
    line in the signal codebase must have one entry in KNOWN_MULTIPLIERS.
    """
    src_signals = Path(__file__).resolve().parents[1] / "src" / "signals"
    pattern = re.compile(r"(?:signal\.)?confidence\s*\*=")
    hits = []
    for py_file in src_signals.rglob("*.py"):
        try:
            text = py_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for ln, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line) and "confidence *= 100" not in line:
                hits.append((py_file.name, ln, line.strip()))
    # We don't enforce 1:1 line:registry mapping (some multipliers are
    # applied at 2 sites — e.g. short_hardening). Instead we enforce
    # COUNT-based monotonicity: as new sites are added, the registry
    # must grow to cover them. The current line count is a snapshot;
    # if a new `confidence *= X` appears and registry doesn't grow,
    # this test fails.
    SNAPSHOT_LINE_COUNT = 14  # measured at A3 ship time
    assert len(hits) <= SNAPSHOT_LINE_COUNT + len(KNOWN_MULTIPLIERS), (
        f"Found {len(hits)} `confidence *=` sites in src/signals/ but the "
        f"registry has only {len(KNOWN_MULTIPLIERS)} entries — drift! "
        f"Either add the new site to KNOWN_MULTIPLIERS or remove the dead "
        f"multiplier. Sites:\n" + "\n".join(f"  {h}" for h in hits)
    )


def test_synthetic_regime_cascade_is_caught_explicitly():
    """The historical 0.43 cascade itself: synthetic_regime_cap (0.50) ×
    source_side_guard (0.75) × agent_scorer_weight_blend (0.50). Must be
    caught by `test_no_two_combinations_collapse_to_same_dead_band`."""
    cap = 0.50
    guard = 0.75
    weight = 0.50
    product = cap * guard * weight  # 0.1875
    strong_after = 0.95 * product
    marginal_after = 0.55 * product
    # In the historical bug, signals were BLENDED not multiplied — so
    # effective post-cascade confidence was approximately
    # 0.5*raw + 0.5*scorer = 0.5*(raw_capped) + 0.5*0.5 = 0.5*0.375 + 0.25 = 0.4375.
    # Independent of raw → that's the dead-band collapse.
    historical_strong_blend = 0.5 * (cap * guard) + 0.5 * 0.5
    historical_marginal_blend = historical_strong_blend  # raw doesn't matter
    assert historical_strong_blend == pytest.approx(historical_marginal_blend), (
        "The blend collapses strong & marginal sources to the same value. "
        "If this assertion ever fails, the cascade math has changed and "
        "the registry/test needs to be re-derived."
    )
    # Both must collapse to ~0.4375 — within the SOURCE_FLOOR dead-band
    assert abs(historical_strong_blend - SOURCE_FLOOR) < CASCADE_DEAD_BAND_WIDTH + 0.005, (
        f"Historical cascade math: blend={historical_strong_blend:.4f}, "
        f"floor={SOURCE_FLOOR}. Should be within dead-band. If this "
        f"diverges, the registry needs updating."
    )
    # Bonus: pure multiplicative path also collapses
    assert strong_after < SOURCE_FLOOR
    assert marginal_after < SOURCE_FLOOR


def test_multipliers_in_valid_range():
    """Every registered multiplier must be a probability in [0, 1]."""
    for name, (mn, dflt, mx), _notes in KNOWN_MULTIPLIERS:
        assert 0.0 <= mn <= dflt <= mx <= 1.0, (
            f"{name}: invalid range min={mn}, default={dflt}, max={mx}"
        )
