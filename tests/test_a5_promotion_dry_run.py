"""Tests for the A5 promotion-gate retro dry-run.

DB-free; uses hand-built Returns lists so the DSR/SPRT math is
deterministic.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from a5_promotion_dry_run import (  # noqa: E402
    Returns,
    build_report,
    equal_weight_champion,
    render,
)


def _strong_returns(n: int = 30, mean: float = 0.005, sd: float = 0.005,
                    seed: int = 1) -> list[float]:
    """High-Sharpe candidate: mean=50bps, sd=50bps -> Sharpe ~1.0."""
    rng = random.Random(seed)
    return [rng.gauss(mean, sd) for _ in range(n)]


def _weak_returns(n: int = 30, mean: float = 0.0, sd: float = 0.01,
                  seed: int = 2) -> list[float]:
    """Coin-flip candidate, ~zero Sharpe."""
    rng = random.Random(seed)
    return [rng.gauss(mean, sd) for _ in range(n)]


# ── Champion construction ──────────────────────────────────────────


def test_equal_weight_champion_aligns_chronologically():
    per_source = [
        Returns(source="A", series=[1.0, 2.0, 3.0]),
        Returns(source="B", series=[5.0, 6.0]),
    ]
    champ = equal_weight_champion(per_source)
    # i=0 → (1+5)/2=3.0; i=1 → (2+6)/2=4.0; i=2 → just 3.0
    assert champ == [3.0, 4.0, 3.0]


def test_equal_weight_champion_empty_input():
    assert equal_weight_champion([]) == []


# ── Report construction ────────────────────────────────────────────


def test_build_report_skips_low_n_candidates():
    per_source = [Returns(source="A", series=[0.1, 0.2])]
    reports = build_report(per_source, num_trials=1, mde=0.001, min_n=8)
    assert reports == []


def test_strong_candidate_has_higher_dsr_than_weak():
    """Sanity: a candidate with structural alpha should have a higher
    deflated_sharpe than a coin-flip candidate."""
    strong = Returns(source="strong", series=_strong_returns(n=30))
    weak = Returns(source="weak", series=_weak_returns(n=30))
    reports = build_report([strong, weak], num_trials=2, mde=0.001, min_n=8)
    by_src = {r.source: r for r in reports}
    assert by_src["strong"].deflated_sharpe > by_src["weak"].deflated_sharpe
    assert by_src["strong"].sharpe > by_src["weak"].sharpe


def test_high_num_trials_kills_marginal_dsr_significance():
    """Multi-testing penalty: increasing num_trials by 10x should
    drag DSR p_value HIGHER (less significant) for the same series."""
    series = _strong_returns(n=30)
    r_low = build_report(
        [Returns(source="x", series=series)],
        num_trials=1, mde=0.001, min_n=8,
    )[0]
    r_high = build_report(
        [Returns(source="x", series=series)],
        num_trials=200, mde=0.001, min_n=8,
    )[0]
    # More trials → larger p_value (less significant)
    assert r_high.dsr_p_value > r_low.dsr_p_value


def test_promote_flag_requires_both_dsr_and_sprt():
    """A would_promote=True row must have BOTH dsr_significant_95
    and sprt_decision==ACCEPT. If either is off, would_promote is False."""
    series = _strong_returns(n=30)
    reports = build_report(
        [Returns(source="strong", series=series)],
        num_trials=1, mde=0.001, min_n=8,
    )
    for r in reports:
        if r.would_promote:
            assert r.dsr_significant_95 is True
            assert r.sprt_decision == "ACCEPT"
        else:
            assert (
                not r.dsr_significant_95
                or r.sprt_decision != "ACCEPT"
            )


def test_render_emits_table_with_header():
    series = _strong_returns(n=30)
    reports = build_report(
        [Returns(source="X", series=series)],
        num_trials=1, mde=0.001, min_n=8,
    )
    md = render(reports)
    assert "A5 — retro promotion-gate dry-run" in md
    assert "DSR" in md
    assert "SPRT" in md


def test_render_empty():
    md = render([])
    assert "0 eligible candidates" in md
    assert "0 would PASS" in md
