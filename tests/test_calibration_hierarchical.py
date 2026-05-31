"""Hierarchical (empirical-Bayes) calibration.

A thin ``source|side|regime`` cell should borrow strength from its pooled
parents (``source|side|*`` -> ``source|*`` -> ``global``) instead of collapsing
to the cold-start prior. The behaviour is flag-gated: OFF reproduces the legacy
cap exactly; ON lets a source with enough *total* evidence (spread across
regimes) emit a real calibrated confidence, while a source that is thin *even
pooled* stays conservatively capped.
"""
from __future__ import annotations

from src.signals.calibration import CalibrationTracker


def _tracker(tmp_path, hierarchical: bool) -> CalibrationTracker:
    t = CalibrationTracker(
        db_path=str(tmp_path / "calib.db"),
        min_outcomes=30, isotonic_min_outcomes=100, coldstart_prior=0.50,
    )
    t._hierarchical_enabled = hierarchical
    return t


def _seed(t, src, n_per_regime=12, win_rate=0.75, conf=0.70):
    """Spread outcomes across 3 regimes so each fine cell is thin (<30) but the
    source|side pool clears min_outcomes."""
    for regime in ("trend", "range", "crash"):
        wins = round(n_per_regime * win_rate)
        for i in range(n_per_regime):
            t.record(src, conf, actual_win=(i < wins), side="long", regime=regime)


def test_legacy_caps_thin_cell_at_prior(tmp_path):
    t = _tracker(tmp_path, hierarchical=False)
    _seed(t, "trader_good")  # 12 per regime -> each fine cell < 30
    adj = t.get_adjustment_factor("trader_good", 0.70, side="long", regime="trend")
    assert abs(adj - 0.50) < 1e-6, f"legacy should cap thin cell at prior, got {adj}"


def test_hierarchical_borrows_strength_from_parent(tmp_path):
    t = _tracker(tmp_path, hierarchical=True)
    _seed(t, "trader_good")  # 36 total at source|long (>=30), but per-regime only 12
    adj = t.get_adjustment_factor("trader_good", 0.70, side="long", regime="trend")
    # The pooled ~75% win rate now drives confidence above the 0.50 cap.
    assert adj > 0.60, f"hierarchical should escape the prior cap, got {adj}"
    assert adj <= 0.95


def test_hierarchical_stays_conservative_when_truly_thin(tmp_path):
    t = _tracker(tmp_path, hierarchical=True)
    # Only 6 outcomes total -> even pooled (n_ss=6) is below min_outcomes.
    for i in range(6):
        t.record("trader_new", 0.70, actual_win=(i < 5), side="long", regime="trend")
    adj = t.get_adjustment_factor("trader_new", 0.70, side="long", regime="trend")
    assert abs(adj - 0.50) < 1e-6, (
        f"an evidence-free source must stay capped at the prior, got {adj}"
    )


def test_hierarchical_respects_low_parent_rate(tmp_path):
    """If the pooled parent is a proven *loser*, hierarchical must pull
    confidence DOWN, not up -- it borrows the real rate, not just a boost."""
    t = _tracker(tmp_path, hierarchical=True)
    _seed(t, "trader_bad", win_rate=0.20)  # 36 at source|long, ~20% win rate
    adj = t.get_adjustment_factor("trader_bad", 0.70, side="long", regime="trend")
    assert adj < 0.50, f"low pooled rate should drag confidence below prior, got {adj}"
