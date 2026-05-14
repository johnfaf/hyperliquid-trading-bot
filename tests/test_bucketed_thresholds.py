"""Per-bucket calibration threshold tests.

Covers ``CalibrationTracker.get_bucketed_min_confidence`` directly and the
firewall's bucketed-threshold gate.
"""
from __future__ import annotations

import tempfile

import pytest

import config
from src.signals.calibration import CalibrationTracker, compose_calibration_key


def _fresh_tracker(monkeypatch) -> CalibrationTracker:
    """Build a CalibrationTracker with an isolated SQLite path."""
    tmp = tempfile.mkdtemp(prefix="cal_test_")
    db_path = f"{tmp}/cal_test.db"
    return CalibrationTracker(db_path=db_path)


def _seed_outcomes(
    tracker: CalibrationTracker,
    *,
    source_key: str,
    side: str,
    regime: str,
    n_trades: int,
    confidence: float,
    win_rate: float,
) -> None:
    """Push N outcomes at given confidence with target win rate."""
    n_wins = int(round(n_trades * win_rate))
    for i in range(n_trades):
        win = i < n_wins
        tracker.record(
            source_key,
            predicted_confidence=confidence,
            actual_win=win,
            pnl=1.0 if win else -1.0,
            side=side,
            regime=regime,
        )


def test_no_data_returns_coldstart_prior(monkeypatch):
    tracker = _fresh_tracker(monkeypatch)
    threshold, reason = tracker.get_bucketed_min_confidence(
        "strategy:fresh", side="long", regime="trend", global_min=0.40
    )
    # Coldstart prior defaults to 0.50 — should land at max(global_min, prior).
    assert threshold == pytest.approx(0.50, abs=1e-6)
    assert reason.startswith("coldstart_no_data")


def test_thin_sample_returns_coldstart_cap(monkeypatch):
    tracker = _fresh_tracker(monkeypatch)
    _seed_outcomes(
        tracker,
        source_key="strategy:thin",
        side="long",
        regime="trend",
        n_trades=5,
        confidence=0.65,
        win_rate=0.60,
    )
    threshold, reason = tracker.get_bucketed_min_confidence(
        "strategy:thin", side="long", regime="trend", global_min=0.40
    )
    assert threshold == pytest.approx(0.50, abs=1e-6)
    assert reason.startswith("coldstart")


def test_healthy_bucket_uses_global_floor(monkeypatch):
    tracker = _fresh_tracker(monkeypatch)
    # 100 outcomes, well-calibrated at 0.7 confidence with 70% win rate.
    _seed_outcomes(
        tracker,
        source_key="strategy:healthy",
        side="long",
        regime="trend",
        n_trades=100,
        confidence=0.70,
        win_rate=0.70,
    )
    threshold, reason = tracker.get_bucketed_min_confidence(
        "strategy:healthy", side="long", regime="trend", global_min=0.40
    )
    assert threshold == pytest.approx(0.40, abs=1e-6)
    assert reason.startswith("healthy")


def test_miscalibrated_bucket_raises_floor(monkeypatch):
    """High ECE bucket should get the high (effectively block) threshold."""
    tracker = _fresh_tracker(monkeypatch)
    # 100 outcomes at confidence 0.80 but only 30% wins => ECE around 0.50.
    _seed_outcomes(
        tracker,
        source_key="strategy:miscal",
        side="short",
        regime="range",
        n_trades=100,
        confidence=0.80,
        win_rate=0.30,
    )
    threshold, reason = tracker.get_bucketed_min_confidence(
        "strategy:miscal", side="short", regime="range", global_min=0.40
    )
    assert threshold >= 0.95
    assert "quarantine" in reason


def test_threshold_never_below_global_min(monkeypatch):
    """Even a healthy bucket can't return below the operator-set floor."""
    tracker = _fresh_tracker(monkeypatch)
    _seed_outcomes(
        tracker,
        source_key="strategy:healthy",
        side="long",
        regime="trend",
        n_trades=100,
        confidence=0.70,
        win_rate=0.70,
    )
    threshold, _ = tracker.get_bucketed_min_confidence(
        "strategy:healthy", side="long", regime="trend", global_min=0.65
    )
    # Healthy returns global_min, which is 0.65 here.
    assert threshold == pytest.approx(0.65, abs=1e-6)


def test_hierarchical_fallback_walks_chain(monkeypatch):
    """A leaf with no data should fall back to (source|side|any) then global."""
    tracker = _fresh_tracker(monkeypatch)
    # Populate the side-level bucket but NOT the specific regime.
    _seed_outcomes(
        tracker,
        source_key="strategy:hier",
        side="long",
        regime="any",
        n_trades=80,
        confidence=0.65,
        win_rate=0.60,
    )
    # Ask for trend-regime that has no rows: should walk up to (source|long|any).
    threshold, reason = tracker.get_bucketed_min_confidence(
        "strategy:hier", side="long", regime="trend", global_min=0.40
    )
    # Resolved should land on the populated bucket -> healthy (assuming low ECE).
    assert threshold == pytest.approx(0.40, abs=1e-6)
    # The reason should reference the fallback key (which contains "any").
    assert "any" in reason


def test_global_fallback_does_not_quarantine(monkeypatch):
    """High ECE on the *global* bucket must NOT trigger the 0.95 quarantine.

    This is the production bug observed on 2026-05-14: with no source-
    specific data, every signal's bucket resolution fell back to
    ``global``; the global bucket had ECE=0.368 (above the 0.25
    quarantine threshold); my gate then raised every threshold to
    0.95, blocking every long copy-trade while leaving an existing
    SHORT ETH position in place. The carve-out: quarantine only fires
    on *specific* buckets, never on the global fallback. Global with
    bad ECE should yield cold-start caution at most.
    """
    tracker = _fresh_tracker(monkeypatch)
    # Push 100 outcomes into the global bucket directly, miscalibrated.
    # ``record`` propagates to both the composed key and the global
    # aggregate, so we use a synthesised non-fallback source then drop
    # its composed bucket to leave only the global aggregate populated.
    _seed_outcomes(
        tracker,
        source_key="strategy:transient",
        side="long",
        regime="trend",
        n_trades=100,
        confidence=0.80,
        win_rate=0.30,  # miscalibrated => global ECE high
    )
    # Drop the specific bucket so only ``global`` carries the bad ECE.
    composed = compose_calibration_key("strategy:transient", "long", "trend")
    tracker._bins.pop(composed, None)
    tracker._brier.pop(composed, None)

    # Now query a brand-new source -- resolution should walk up to "global".
    threshold, reason = tracker.get_bucketed_min_confidence(
        "strategy:new_source",
        side="long",
        regime="trend",
        global_min=0.40,
    )
    # Cold-start prior is 0.50; threshold must be that, NOT 0.95.
    assert threshold < 0.95, (
        f"Global fallback must not trigger quarantine; got {threshold} ({reason})"
    )
    assert threshold == pytest.approx(0.50, abs=1e-6)


def test_compose_key_roundtrip():
    """Sanity check the key composition used by the threshold method."""
    key = compose_calibration_key("strategy:momentum", "long", "trend")
    assert key.startswith("strategy:momentum")
    assert "long" in key
    assert "trend" in key
