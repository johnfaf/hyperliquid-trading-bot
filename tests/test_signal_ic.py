"""Information Coefficient measurement (signal #1, keystone).

IC = rank correlation between a source's pre-trade confidence and its realized
outcome. This is the "does this source predict?" metric the bot was missing.
"""
from __future__ import annotations

import sqlite3

from src.analysis.signal_ic import (
    spearman_ic, compute_source_ic, ic_weight, source_ic_weight,
)


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


# ── IC -> confidence weight (signal #6) ──

def test_ic_weight_neutral_when_insufficient():
    # Too few outcomes or no IC -> observe-first neutral 1.0 (no-op).
    assert ic_weight(0.9, 5, min_n=20) == 1.0
    assert ic_weight(None, 100, min_n=20) == 1.0
    assert ic_weight(-0.9, None, min_n=20) == 1.0


def test_ic_weight_positive_scales_up_negative_down():
    # IC=+0.2, gain 2.5 -> 1 + 0.5 = 1.5 ; IC=-0.2 -> 1 - 0.5 = 0.5
    assert abs(ic_weight(0.2, 50, min_n=20, gain=2.5) - 1.5) < 1e-9
    assert abs(ic_weight(-0.2, 50, min_n=20, gain=2.5) - 0.5) < 1e-9
    # A predictive source weighs more than an anti-predictive one.
    assert ic_weight(0.1, 50) > 1.0 > ic_weight(-0.1, 50)


def test_ic_weight_clamps_to_bounds():
    # Strong IC saturates at the clamps, never runs away.
    assert ic_weight(0.9, 50, gain=2.5, max_weight=1.5) == 1.5
    assert ic_weight(-0.9, 50, gain=2.5, min_weight=0.25) == 0.25


def _mk_calib(db, rows):
    """rows: (source_key, predicted_confidence, pnl)."""
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE calibration_records (source_key TEXT, "
              "predicted_confidence REAL, pnl REAL)")
    c.executemany("INSERT INTO calibration_records VALUES (?,?,?)", rows)
    c.commit()
    c.close()


def test_source_ic_weight_predictive_scales_up(tmp_path):
    db = str(tmp_path / "c.db")
    # Monotone: higher predicted confidence -> higher realized pnl (IC ~ +1).
    _mk_calib(db, [("funding_div", i / 12.0, float(i)) for i in range(12)])
    w = source_ic_weight("funding_div", db, min_n=10)
    assert w > 1.0


def test_source_ic_weight_anti_predictive_scales_down(tmp_path):
    db = str(tmp_path / "c.db")
    # Inverse: higher confidence -> worse pnl (IC ~ -1) -> clamp to min_weight.
    _mk_calib(db, [("liq_cascade", i / 12.0, float(-i)) for i in range(12)])
    w = source_ic_weight("liq_cascade", db, min_n=10, min_weight=0.25)
    assert w == 0.25


def test_source_ic_weight_unmeasured_is_neutral(tmp_path):
    db = str(tmp_path / "c.db")
    _mk_calib(db, [("thin", i / 4.0, float(i)) for i in range(4)])  # n < min_n
    assert source_ic_weight("thin", db, min_n=20) == 1.0
    # An unknown source with no rows is also neutral.
    assert source_ic_weight("never_seen", db, min_n=20) == 1.0


def test_source_ic_weight_missing_db_is_neutral(tmp_path):
    # Fail-open: a missing/unreadable DB must never break the decision path.
    assert source_ic_weight("x", str(tmp_path / "nope.db")) == 1.0


# ── DecisionEngine wiring (observe-first, default OFF) ──

def _ev_proxy_for(cfg, db):
    from src.signals.decision_engine import DecisionEngine
    eng = DecisionEngine(cfg)
    strat = {
        "strategy_type": "momentum_long",   # -> direction "long", no regime needed
        "current_score": 0.6,
        "confidence": 0.6,
        "source": "funding_divergence",
        "source_key": "funding_divergence",
        "parameters": "{}",
    }
    return eng._compute_composite_score(strat, None, set(), None)["ev_proxy"]


def test_decision_engine_ic_weight_off_is_noop(tmp_path):
    db = str(tmp_path / "c.db")
    # Even with an anti-predictive source on disk, flag OFF => unchanged EV.
    _mk_calib(db, [("funding_divergence", i / 12.0, float(-i)) for i in range(12)])
    off = _ev_proxy_for({"ev_first_enabled": True, "ic_db_path": db,
                         "ic_min_n": 10}, db)
    baseline = _ev_proxy_for({"ev_first_enabled": True}, db)  # no IC, no db
    assert abs(off - baseline) < 1e-9


def test_decision_engine_ic_weight_fades_negative_source(tmp_path):
    db = str(tmp_path / "c.db")
    _mk_calib(db, [("funding_divergence", i / 12.0, float(-i)) for i in range(12)])
    on = _ev_proxy_for({"ev_first_enabled": True, "ic_weight_enabled": True,
                        "ic_db_path": db, "ic_min_n": 10}, db)
    off = _ev_proxy_for({"ev_first_enabled": True, "ic_weight_enabled": False,
                         "ic_db_path": db, "ic_min_n": 10}, db)
    # An anti-predictive source's EV is pushed DOWN when grading is enabled.
    assert on < off


def test_decision_engine_ic_weight_exempts_copy(tmp_path):
    db = str(tmp_path / "c.db")
    _mk_calib(db, [("copy_open", i / 12.0, float(-i)) for i in range(12)])
    from src.signals.decision_engine import DecisionEngine
    eng = DecisionEngine({"ev_first_enabled": True, "ic_weight_enabled": True,
                          "ic_db_path": db, "ic_min_n": 10})
    strat = {
        "strategy_type": "momentum_long",
        "current_score": 0.6, "confidence": 0.6,
        "source": "copy_open", "source_key": "copy_open",
        "parameters": "{}",
    }
    ev_copy = eng._compute_composite_score(strat, None, set(), None)["ev_proxy"]
    # copy is exempt -> EV equals the un-weighted baseline (p_win 0.6 untouched).
    eng_off = DecisionEngine({"ev_first_enabled": True})
    ev_base = eng_off._compute_composite_score(strat, None, set(), None)["ev_proxy"]
    assert abs(ev_copy - ev_base) < 1e-9
