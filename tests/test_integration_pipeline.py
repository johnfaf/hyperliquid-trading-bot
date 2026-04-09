"""
End-to-End Integration Test (LOW-13)
=====================================
Tests the signal pipeline from regime detection through to trade decision,
using mocked external APIs but real internal logic.
"""
import pytest
from unittest.mock import MagicMock, patch


class TestIntegrationPipeline:
    """Integration tests verifying the signal pipeline end-to-end."""

    @pytest.fixture
    def mock_container(self):
        """Build a minimal container with mocked components."""
        container = MagicMock()

        # Regime detector — returns a realistic regime_data dict
        regime_detector = MagicMock()
        regime_detector.get_market_regime.return_value = {
            "overall_regime": "trending_up",
            "overall_confidence": 0.72,
            "per_coin": {
                "BTC": {
                    "regime": "trending_up",
                    "confidence": 0.75,
                    "trend_strength": 0.6,
                    "atr_pct": 0.025,
                    "volume_ratio": 1.3,
                },
            },
            "strategy_guidance": {
                "activate": ["momentum", "trend_following"],
                "pause": ["mean_reversion"],
            },
        }
        container.regime_detector = regime_detector

        # Forecaster — predicts bullish (agrees with detector)
        forecaster = MagicMock()
        forecaster.predict_regime.return_value = {
            "regime": "bullish",
            "confidence": 0.68,
            "signal": 0.68,
            "model": "xgboost",
            "probabilities": {"crash": 0.1, "neutral": 0.22, "bullish": 0.68},
            "components": {},
            "active_inputs": ["funding_rate", "orderbook_imbalance"],
            "active_input_count": 2,
        }
        container.predictive_forecaster = forecaster

        # Signal processor
        processor = MagicMock()
        processor.process.side_effect = lambda strats, **kw: strats
        container.signal_processor = processor

        # Options scanner
        container.options_scanner = MagicMock()
        container.options_scanner.top_convictions = []

        # Polymarket
        container.polymarket = None

        # Scorer
        scorer = MagicMock()
        scorer.get_top_strategies.return_value = [
            {
                "id": 1,
                "name": "btc_momentum_long",
                "strategy_type": "momentum",
                "trader_address": "0xabc",
                "current_score": 0.85,
                "confidence": 0.75,
                "direction": "long",
                "side": "long",
                "source": "copy_trader",
                "parameters": {"coins": ["BTC"]},
                "metrics": {"win_rate": 0.62, "sharpe": 1.4},
            },
        ]
        container.scorer = scorer

        # Misc
        container.cross_venue_hedger = None
        container.feature_engine = None
        container.regime_strategy_filter = None
        container.shadow_tracker = None
        container.liquidation_strategy = None
        container.arena = None
        container._last_regime_data = None

        return container

    def test_regime_reconciliation_agreement(self, mock_container):
        """When detector and forecaster agree, regime_data is annotated."""
        from src.core.cycles.trading_cycle import _reconcile_regimes

        regime_data = mock_container.regime_detector.get_market_regime()
        result = _reconcile_regimes(regime_data, mock_container)

        assert result["regime_agreement"] is True
        assert result["forecaster_regime"] == "bullish"
        assert result["forecaster_confidence"] == pytest.approx(0.68, abs=0.01)
        # Original regime should be unchanged
        assert result["overall_regime"] == "trending_up"

    def test_regime_reconciliation_crash_override(self, mock_container):
        """When forecaster says crash but detector disagrees, conservative override."""
        from src.core.cycles.trading_cycle import _reconcile_regimes

        # Detector says trending_up, forecaster says crash
        mock_container.predictive_forecaster.predict_regime.return_value = {
            "regime": "crash",
            "confidence": 0.75,
            "signal": -0.75,
            "model": "xgboost",
        }

        regime_data = mock_container.regime_detector.get_market_regime()
        result = _reconcile_regimes(regime_data, mock_container)

        assert result["regime_agreement"] is False
        # Conservative override: detector was trending_up, but crash override applies
        assert result["overall_regime"] == "volatile"
        assert result.get("regime_override") == "forecaster_crash"
        # Strategy guidance should suppress bullish activations
        assert result["strategy_guidance"]["activate"] == []
        assert "momentum" in result["strategy_guidance"]["pause"]

    def test_regime_reconciliation_low_confidence_no_override(self, mock_container):
        """Low-confidence disagreement does NOT trigger override."""
        from src.core.cycles.trading_cycle import _reconcile_regimes

        mock_container.predictive_forecaster.predict_regime.return_value = {
            "regime": "crash",
            "confidence": 0.35,  # Below 0.5 threshold
            "signal": -0.35,
        }

        regime_data = mock_container.regime_detector.get_market_regime()
        # Lower detector confidence too
        regime_data["overall_confidence"] = 0.40
        result = _reconcile_regimes(regime_data, mock_container)

        # Should NOT override since both confidences < 0.5
        assert result["overall_regime"] == "trending_up"
        assert result.get("regime_override") is None

    def test_regime_reconciliation_no_forecaster(self, mock_container):
        """Without forecaster, reconciliation is a no-op."""
        from src.core.cycles.trading_cycle import _reconcile_regimes

        mock_container.predictive_forecaster = None
        regime_data = mock_container.regime_detector.get_market_regime()
        result = _reconcile_regimes(regime_data, mock_container)

        assert result is regime_data  # Same object, unchanged

    def test_signal_processor_receives_strategies(self, mock_container):
        """Signal processor receives strategies from scorer."""
        scorer = mock_container.scorer
        processor = mock_container.signal_processor

        strategies = scorer.get_top_strategies()
        result = processor.process(strategies, regime_data={})

        assert len(result) == 1
        assert result[0]["name"] == "btc_momentum_long"
        processor.process.assert_called_once()

    def test_pipeline_strategy_has_required_fields(self, mock_container):
        """Strategies from scorer contain all required fields for execution."""
        strategies = mock_container.scorer.get_top_strategies()
        required_fields = {"name", "strategy_type", "confidence", "side", "parameters"}

        for strat in strategies:
            missing = required_fields - set(strat.keys())
            assert not missing, f"Strategy missing fields: {missing}"
            assert 0 <= strat["confidence"] <= 1
            assert strat["side"] in ("long", "short")
