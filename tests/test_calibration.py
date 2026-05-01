from src.signals.calibration import CalibrationTracker


def test_reliability_multiplier_derisks_poor_global_ece_without_source_bins(tmp_path):
    cal = CalibrationTracker(db_path=str(tmp_path / "calibration.db"))
    # 0.8-confidence bin that actually loses every time gives a high global ECE.
    for _ in range(12):
        cal.record("global", 0.8, False)

    assert cal.get_ece("global") >= 0.35
    assert cal.get_reliability_multiplier("strategy:thin_source") == 0.65


def test_reliability_multiplier_leaves_good_calibration_unchanged(tmp_path):
    cal = CalibrationTracker(db_path=str(tmp_path / "calibration.db"))
    for _ in range(10):
        cal.record("global", 0.8, True)
    for _ in range(2):
        cal.record("global", 0.8, False)

    assert cal.get_reliability_multiplier("global") == 1.0
