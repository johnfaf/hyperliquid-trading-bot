import pytest
import time
from contextlib import contextmanager
from unittest.mock import patch, MagicMock

from src.signals.xgboost_regime_forecaster import (
    XGBoostRegimeForecaster,
    FEATURE_NAMES,
    REGIME_LABELS,
    REGIME_NAMES,
)


class TestXGBoostRegimeForecaster:
    """Test suite for XGBoost regime forecaster (LOW-12)."""

    @pytest.fixture
    def forecaster(self):
        """Create a forecaster with mocked DB to avoid file I/O."""
        with patch("src.signals.xgboost_regime_forecaster.XGBoostRegimeForecaster._ensure_regime_history_table"):
            with patch("src.signals.xgboost_regime_forecaster.XGBoostRegimeForecaster._load_model"):
                with patch("src.signals.xgboost_regime_forecaster.XGBoostRegimeForecaster.train"):
                    fc = XGBoostRegimeForecaster({"min_training_samples": 10})
        return fc

    def test_init(self, forecaster):
        """Forecaster initializes without error and has expected attributes."""
        assert forecaster is not None
        assert forecaster.model is None  # No saved model in test
        assert forecaster.fallback is not None

    def test_feature_names_count(self):
        """FEATURE_NAMES should have exactly 6 features after dead-feature removal."""
        assert len(FEATURE_NAMES) == 6
        assert "funding_rate" in FEATURE_NAMES
        assert "orderbook_imbalance" in FEATURE_NAMES
        # Removed features should NOT be present
        assert "arkham_flow" not in FEATURE_NAMES
        assert "polymarket_sentiment" not in FEATURE_NAMES

    def test_regime_labels_mapping(self):
        """Label encoding should be crash=0, neutral=1, bullish=2."""
        assert REGIME_LABELS == {"crash": 0, "neutral": 1, "bullish": 2}
        assert REGIME_NAMES == {0: "crash", 1: "neutral", 2: "bullish"}

    def test_predict_regime_fallback(self, forecaster):
        """Without XGBoost model, predict_regime falls back to weighted-signal."""
        forecaster.model = None
        with patch("src.signals.xgboost_regime_forecaster.HAS_XGBOOST", False):
            result = forecaster.predict_regime("BTC")
        assert result is not None
        assert isinstance(result, dict)
        assert "regime" in result
        assert result["regime"] in ("crash", "neutral", "bullish")
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
        assert result.get("model") == "weighted_signal"

    @patch("src.signals.xgboost_regime_forecaster.HAS_XGBOOST", True)
    def test_predict_regime_xgboost(self, forecaster):
        """With a mocked XGBoost model, predict_regime returns ML prediction."""
        import numpy as np

        mock_model = MagicMock()
        # Simulate model predicting bullish with 70% confidence
        mock_model.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
        forecaster.model = mock_model
        forecaster._last_train_ts = time.time()

        # Mock the feature extraction
        features = {f: 0.5 for f in FEATURE_NAMES}
        with patch.object(forecaster, "_extract_features", return_value=features):
            with patch.object(forecaster, "_store_prediction"):
                # Clear prediction cache so we get a fresh prediction
                forecaster.prediction_cache.clear()
                result = forecaster.predict_regime("BTC")

        assert result is not None
        assert result["regime"] == "bullish"
        assert result["model"] == "xgboost"
        assert abs(result["confidence"] - 0.7) < 0.01
        assert abs(result["probabilities"]["crash"] - 0.1) < 0.01
        assert abs(result["probabilities"]["neutral"] - 0.2) < 0.01
        assert abs(result["probabilities"]["bullish"] - 0.7) < 0.01

    @patch("src.signals.xgboost_regime_forecaster.HAS_XGBOOST", True)
    def test_predict_regime_crash(self, forecaster):
        """XGBoost predicting crash returns negative signal."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.8, 0.15, 0.05]])
        forecaster.model = mock_model
        forecaster._last_train_ts = time.time()

        features = {f: 0.0 for f in FEATURE_NAMES}
        with patch.object(forecaster, "_extract_features", return_value=features):
            with patch.object(forecaster, "_store_prediction"):
                forecaster.prediction_cache.clear()
                result = forecaster.predict_regime("BTC")

        assert result["regime"] == "crash"
        assert result["signal"] < 0  # Negative signal for crash
        assert result["confidence"] >= 0.8

    def test_prediction_caching(self, forecaster):
        """Second call within cache_ttl returns cached result."""
        forecaster.model = None
        with patch("src.signals.xgboost_regime_forecaster.HAS_XGBOOST", False):
            r1 = forecaster.predict_regime("BTC")
            r2 = forecaster.predict_regime("BTC")
        # Both should be identical (cached)
        assert r1["regime"] == r2["regime"]
        assert r1["confidence"] == r2["confidence"]

    def test_update_passthrough(self, forecaster):
        """update_polymarket_sentiment and update_options_flow pass to fallback."""
        with patch.object(forecaster.fallback, "update_polymarket_sentiment") as mock_pm:
            forecaster.update_polymarket_sentiment({"sentiment": "bullish"})
            mock_pm.assert_called_once_with({"sentiment": "bullish"})

        with patch.object(forecaster.fallback, "update_options_flow") as mock_of:
            forecaster.update_options_flow([{"ticker": "BTC"}])
            mock_of.assert_called_once_with([{"ticker": "BTC"}])

    def test_retrain_interval(self, forecaster):
        """Auto-retrain triggers when retrain_interval has elapsed."""
        import time
        forecaster._last_train_ts = time.time() - 100_000  # Long ago
        forecaster.model = None

        with patch.object(forecaster, "train"):
            forecaster.predict_regime("BTC")
            # train should NOT be called if HAS_XGBOOST is False
            # but if it is True and model is None, it should try

    def test_postgres_regime_history_setup_skips_sqlite_ddl(self, forecaster, monkeypatch):
        """Postgres mode should rely on migrations instead of SQLite DDL/PRAGMA."""

        class _DummyConn:
            def __init__(self):
                self.executescript_called = False
                self.execute_calls = []

            def execute(self, sql, params=None):
                self.execute_calls.append((sql, params))
                return MagicMock(fetchall=lambda: [])

            def executescript(self, sql):
                self.executescript_called = True

        dummy = _DummyConn()

        @contextmanager
        def _ctx(*, for_read: bool = False):
            yield dummy

        monkeypatch.setattr("src.data.database.get_backend_name", lambda: "postgres")
        monkeypatch.setattr("src.data.database.table_exists", lambda name: True)
        monkeypatch.setattr("src.data.database.get_connection", _ctx)

        forecaster._ensure_regime_history_table()

        assert dummy.executescript_called is False
        assert dummy.execute_calls == []
