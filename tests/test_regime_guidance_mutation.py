from unittest.mock import MagicMock


def test_crash_reconciliation_does_not_mutate_trending_up_guidance():
    from src.analysis.regime_detector import REGIME_STRATEGY_MAP, Regime
    from src.core.cycles.trading_cycle import _reconcile_regimes

    original_activate = list(REGIME_STRATEGY_MAP[Regime.TRENDING_UP]["activate"])
    original_pause = list(REGIME_STRATEGY_MAP[Regime.TRENDING_UP]["pause"])
    original_size = REGIME_STRATEGY_MAP[Regime.TRENDING_UP]["size_modifier"]

    container = MagicMock()
    container.predictive_forecaster.predict_regime.return_value = {
        "regime": "crash",
        "confidence": 0.90,
    }
    regime_data = {
        "overall_regime": "trending_up",
        "overall_confidence": 0.95,
        "strategy_guidance": REGIME_STRATEGY_MAP[Regime.TRENDING_UP],
    }

    result = _reconcile_regimes(regime_data, container)

    assert result["regime_override"] == "forecaster_crash"
    assert result["strategy_guidance"]["activate"] == []
    assert result["countertrend_block_side"] == "short"
    assert REGIME_STRATEGY_MAP[Regime.TRENDING_UP]["activate"] == original_activate
    assert REGIME_STRATEGY_MAP[Regime.TRENDING_UP]["pause"] == original_pause
    assert REGIME_STRATEGY_MAP[Regime.TRENDING_UP]["size_modifier"] == original_size


def test_macro_overlay_does_not_compound_global_trending_up_size_modifier():
    from src.analysis.regime_detector import REGIME_STRATEGY_MAP, Regime
    from src.core.cycles.trading_cycle import _apply_macro_regime_overlay

    original_size = REGIME_STRATEGY_MAP[Regime.TRENDING_UP]["size_modifier"]
    container = MagicMock()
    container.macro_regime.get_risk_posture.return_value = {
        "macro_risk_level": "elevated",
        "macro_score": -0.1,
        "size_modifier": 0.70,
        "confidence_drag": -0.07,
        "block_new_entries": False,
        "reasons": ["test"],
    }
    regime_data = {
        "overall_regime": "trending_up",
        "strategy_guidance": REGIME_STRATEGY_MAP[Regime.TRENDING_UP],
    }

    result = _apply_macro_regime_overlay(container, regime_data)

    assert result["strategy_guidance"]["size_modifier"] == 0.70
    assert REGIME_STRATEGY_MAP[Regime.TRENDING_UP]["size_modifier"] == original_size


def test_detector_returns_independent_strategy_guidance(monkeypatch):
    from src.analysis.regime_detector import (
        REGIME_STRATEGY_MAP,
        Regime,
        RegimeDetector,
        RegimeState,
    )

    detector = RegimeDetector()
    monkeypatch.setattr(
        detector,
        "detect_regime",
        lambda coin: RegimeState(
            regime=Regime.TRENDING_UP,
            confidence=0.95,
            adx=35.0,
            atr_pct=0.02,
            volume_ratio=1.2,
            trend_direction=1.0,
            momentum=0.03,
            timestamp="2026-04-22T00:00:00Z",
        ),
    )
    monkeypatch.setattr("src.analysis.regime_detector.time.sleep", lambda _: None)

    result = detector.get_market_regime(coins=["BTC"])
    result["strategy_guidance"]["activate"].clear()
    result["strategy_guidance"]["size_modifier"] = 0.0

    assert REGIME_STRATEGY_MAP[Regime.TRENDING_UP]["activate"]
    assert REGIME_STRATEGY_MAP[Regime.TRENDING_UP]["size_modifier"] == 1.0
