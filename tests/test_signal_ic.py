"""Information Coefficient measurement (signal #1, keystone).

IC = rank correlation between a source's pre-trade confidence and its realized
outcome. This is the "does this source predict?" metric the bot was missing.
"""
from __future__ import annotations

from src.analysis.signal_ic import spearman_ic, compute_source_ic


def test_spearman_perfect_monotone():
    assert abs(spearman_ic([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) - 1.0) < 1e-9


def test_spearman_perfect_inverse():
    assert abs(spearman_ic([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) + 1.0) < 1e-9


def test_spearman_flat_input_is_none():
    # No variance in x (confidence pinned flat) -> undefined IC.
    assert spearman_ic([0.5] * 5, [1, 2, 3, 4, 5]) is None


def test_spearman_too_few_points():
    assert spearman_ic([1, 2], [3, 4]) is None


def test_compute_source_ic_classifies_sources():
    rows = []
    rows += [("good", i / 12.0, float(i)) for i in range(12)]        # monotone -> predictive
    rows += [("bad", i / 12.0, float(-i)) for i in range(12)]        # inverse  -> negative
    rows += [("flat", 0.5, float(i)) for i in range(12)]             # no conf variance -> flat
    rows += [("thin", i / 5.0, float(i)) for i in range(5)]          # n<min_n -> insufficient

    out = compute_source_ic(rows, min_n=10, band=0.05)

    assert out["good"]["verdict"] == "predictive" and out["good"]["ic"] > 0.9
    assert out["bad"]["verdict"] == "negative" and out["bad"]["ic"] < -0.9
    assert out["flat"]["verdict"] == "flat" and out["flat"]["ic"] is None
    assert out["thin"]["verdict"] == "insufficient" and out["thin"]["n"] == 5


def test_compute_source_ic_ignores_unparseable():
    rows = [("good", None, 1.0), ("good", "x", 2.0)] + \
           [("good", i / 12.0, float(i)) for i in range(12)]
    out = compute_source_ic(rows, min_n=10)
    assert out["good"]["n"] == 12  # the two junk rows skipped
