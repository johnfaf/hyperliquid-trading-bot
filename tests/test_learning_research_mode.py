"""Tests for the env-tunable learning quality gates.

Covers:
- Strict defaults unchanged (back-compat)
- Per-knob env overrides (6 thresholds)
- LEARNING_QUALITY_RESEARCH_MODE=1 preset applies to all unset knobs
- Explicit kwargs still win (constructor parameters preserved)
- The auditor blocks/passes correctly for prod-shape data under each mode
"""
from __future__ import annotations

from typing import Dict, List, Optional

from src.learning.data_quality import (
    DatasetQualityAuditor,
    _RESEARCH_THRESHOLDS,
    _STRICT_THRESHOLDS,
)
from src.learning.dataset_builder import DatasetBuildResult, LearningExample


# ── Strict defaults (back-compat) ──────────────────────────────


def test_strict_defaults_match_legacy(monkeypatch):
    """No env vars set -> auditor uses the strict legacy thresholds."""
    for key in (
        "LEARNING_QUALITY_RESEARCH_MODE",
        "LEARNING_QUALITY_MIN_ROWS",
        "LEARNING_QUALITY_MIN_LABELLED",
        "LEARNING_QUALITY_MAX_MISSING_FEATURE_RATIO",
        "LEARNING_QUALITY_MIN_POSITIVE_RATIO",
        "LEARNING_QUALITY_MAX_POSITIVE_RATIO",
        "LEARNING_QUALITY_MAX_DATA_GAP_RATIO",
    ):
        monkeypatch.delenv(key, raising=False)
    a = DatasetQualityAuditor()
    assert a.min_rows == _STRICT_THRESHOLDS["min_rows"] == 50
    assert a.min_labelled == _STRICT_THRESHOLDS["min_labelled"] == 30
    assert a.max_missing_feature_ratio == _STRICT_THRESHOLDS["max_missing_feature_ratio"] == 0.15
    assert a.min_positive_ratio == _STRICT_THRESHOLDS["min_positive_ratio"] == 0.20
    assert a.max_positive_ratio == _STRICT_THRESHOLDS["max_positive_ratio"] == 0.80
    assert a.max_data_gap_ratio == _STRICT_THRESHOLDS["max_data_gap_ratio"] == 0.05


# ── Research-mode preset ───────────────────────────────────────


def test_research_mode_preset_applied(monkeypatch):
    """LEARNING_QUALITY_RESEARCH_MODE=1 -> looser thresholds."""
    monkeypatch.setenv("LEARNING_QUALITY_RESEARCH_MODE", "1")
    for k in (
        "LEARNING_QUALITY_MIN_ROWS",
        "LEARNING_QUALITY_MIN_LABELLED",
        "LEARNING_QUALITY_MAX_MISSING_FEATURE_RATIO",
        "LEARNING_QUALITY_MIN_POSITIVE_RATIO",
        "LEARNING_QUALITY_MAX_POSITIVE_RATIO",
        "LEARNING_QUALITY_MAX_DATA_GAP_RATIO",
    ):
        monkeypatch.delenv(k, raising=False)
    a = DatasetQualityAuditor()
    assert a.min_labelled == _RESEARCH_THRESHOLDS["min_labelled"] == 20
    assert a.max_missing_feature_ratio == _RESEARCH_THRESHOLDS["max_missing_feature_ratio"] == 0.60
    assert a.min_positive_ratio == _RESEARCH_THRESHOLDS["min_positive_ratio"] == 0.05
    assert a.max_positive_ratio == _RESEARCH_THRESHOLDS["max_positive_ratio"] == 0.95
    assert a.max_data_gap_ratio == _RESEARCH_THRESHOLDS["max_data_gap_ratio"] == 0.80


def test_research_mode_truthy_values(monkeypatch):
    for val in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("LEARNING_QUALITY_RESEARCH_MODE", val)
        a = DatasetQualityAuditor()
        assert a.min_positive_ratio == _RESEARCH_THRESHOLDS["min_positive_ratio"]


def test_research_mode_falsy_values(monkeypatch):
    for val in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("LEARNING_QUALITY_RESEARCH_MODE", val)
        a = DatasetQualityAuditor()
        assert a.min_positive_ratio == _STRICT_THRESHOLDS["min_positive_ratio"]


# ── Per-knob env overrides ─────────────────────────────────────


def test_per_knob_env_override(monkeypatch):
    """Individual env vars override their preset value."""
    monkeypatch.setenv("LEARNING_QUALITY_MAX_DATA_GAP_RATIO", "0.50")
    monkeypatch.setenv("LEARNING_QUALITY_MIN_LABELLED", "15")
    a = DatasetQualityAuditor()
    assert a.max_data_gap_ratio == 0.50
    assert a.min_labelled == 15
    # The other knobs are still at strict defaults
    assert a.min_positive_ratio == _STRICT_THRESHOLDS["min_positive_ratio"]


def test_per_knob_env_clamped_to_range(monkeypatch):
    """Out-of-range env values get clamped to [lo, hi]."""
    monkeypatch.setenv("LEARNING_QUALITY_MAX_MISSING_FEATURE_RATIO", "2.5")
    monkeypatch.setenv("LEARNING_QUALITY_MIN_POSITIVE_RATIO", "-1.0")
    a = DatasetQualityAuditor()
    assert a.max_missing_feature_ratio == 1.0
    assert a.min_positive_ratio == 0.0


def test_per_knob_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("LEARNING_QUALITY_MIN_LABELLED", "not-a-number")
    a = DatasetQualityAuditor()
    assert a.min_labelled == _STRICT_THRESHOLDS["min_labelled"]


def test_per_knob_env_overrides_research_preset(monkeypatch):
    """A per-knob env value wins over the research preset for that knob."""
    monkeypatch.setenv("LEARNING_QUALITY_RESEARCH_MODE", "1")
    monkeypatch.setenv("LEARNING_QUALITY_MAX_DATA_GAP_RATIO", "0.10")
    a = DatasetQualityAuditor()
    assert a.max_data_gap_ratio == 0.10  # env wins
    # other knobs still come from research preset
    assert a.max_missing_feature_ratio == _RESEARCH_THRESHOLDS["max_missing_feature_ratio"]


# ── Explicit kwargs still win ──────────────────────────────────


def test_constructor_kwarg_wins_over_env_and_preset(monkeypatch):
    monkeypatch.setenv("LEARNING_QUALITY_RESEARCH_MODE", "1")
    monkeypatch.setenv("LEARNING_QUALITY_MIN_LABELLED", "10")
    a = DatasetQualityAuditor(min_labelled=999)
    assert a.min_labelled == 999  # explicit caller wins
    # Knobs not passed explicitly still come from env / preset
    assert a.max_data_gap_ratio == _RESEARCH_THRESHOLDS["max_data_gap_ratio"]


# ── End-to-end: prod-shape data ────────────────────────────────


def _example(features: Dict[str, float], label_win: Optional[int]) -> LearningExample:
    """Build the minimal LearningExample shape the auditor reads."""
    return LearningExample(
        decision_id="d_test",
        coin="BTC",
        side="long",
        source="strategy",
        created_at="2026-05-28T00:00:00Z",
        features=features,
        confidence=0.5,
        executed=label_win is not None,
        label_win=label_win,
        outcome_pnl=0.0 if label_win is None else (1.0 if label_win == 1 else -1.0),
        paper_trade_id=None,
    )


def _prod_shape_dataset() -> DatasetBuildResult:
    """Synthetic dataset that approximates the real prod failure mode:
    ~50% missing-feature ratio, ~6% positive labels, plenty of rows.

    Each example carries 5 of the 11 expected features (so missing
    ratio = 6/11 = 54.5%; STRICT 0.15 fails, RESEARCH 0.60 passes).
    """
    examples: List[LearningExample] = []
    base_features = {
        "rsi": 50.0, "rsi_signal": 1.0, "volume_trend": 0.5,
        "overall_score": 0.5, "funding_signal": 0.5,
    }
    # 48 labelled (3 wins, 45 losses) -- matches the 6.3% positive ratio
    for i in range(3):
        examples.append(_example(dict(base_features), label_win=1))
    for i in range(45):
        examples.append(_example(dict(base_features), label_win=0))
    # 200 more unlabelled but with features (to keep missing_ratio honest)
    for i in range(200):
        examples.append(_example(dict(base_features), label_win=None))
    return DatasetBuildResult(
        dataset_id="lds_prod_shape",
        examples=examples,
        feature_names=["rsi", "rsi_signal", "volume_trend", "overall_score",
                       "funding_signal", "missing_a", "missing_b",
                       "missing_c", "missing_d", "missing_e", "missing_f"],
        quality_report={},
    )


def test_prod_shape_blocked_under_strict(monkeypatch):
    """Strict defaults -> prod data fails (the current blocked state)."""
    monkeypatch.delenv("LEARNING_QUALITY_RESEARCH_MODE", raising=False)
    auditor = DatasetQualityAuditor()
    report = auditor.audit(_prod_shape_dataset(), persist=False)
    assert report.blocks_training is True
    assert report.status == "fail"
    failed = set(report.summary["failed_checks"])
    # All three real prod failures should be present.
    assert "missing_feature_ratio" in failed
    assert "positive_label_balance" in failed


def test_prod_shape_passes_under_research_mode(monkeypatch):
    """Research mode -> the same prod data is good enough to learn from."""
    monkeypatch.setenv("LEARNING_QUALITY_RESEARCH_MODE", "1")
    auditor = DatasetQualityAuditor()
    report = auditor.audit(_prod_shape_dataset(), persist=False)
    # missing_feature_ratio: looser bar 0.6 catches the synthetic prod 0.49
    # positive_label_balance: looser min 0.05 catches the synthetic 0.063
    # max_data_gap_ratio: looser bar 0.8 catches the synthetic 0.61
    failed = set(report.summary["failed_checks"])
    assert "missing_feature_ratio" not in failed
    assert "positive_label_balance" not in failed
