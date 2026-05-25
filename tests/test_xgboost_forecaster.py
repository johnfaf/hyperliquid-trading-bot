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


def test_labeler_classifies_forward_returns(monkeypatch):
    """label_predictions_with_forward_returns maps fwd-returns to crash/neutral/bullish."""
    from src.data import database as db_mod
    from src.data import hyperliquid_client as hl_mod

    with patch.object(XGBoostRegimeForecaster, "_ensure_regime_history_table"), \
         patch.object(XGBoostRegimeForecaster, "_load_model"), \
         patch.object(XGBoostRegimeForecaster, "train"):
        fc = XGBoostRegimeForecaster({"min_training_samples": 10})

    base_ms = 1700000000000
    fake_rows = [
        {"id": 11, "timestamp": "2023-11-14T22:13:20+00:00", "coin": "BTC"},
        {"id": 22, "timestamp": "2023-11-14T22:13:20+00:00", "coin": "ETH"},
        {"id": 33, "timestamp": "2023-11-14T22:13:20+00:00", "coin": "SOL"},
    ]

    class _FakeConn:
        def __init__(self):
            self.executed = []

        def execute(self, sql, params=()):
            class _Cur:
                def fetchall(_self):
                    return fake_rows
            return _Cur()

        def executemany(self, sql, seq):
            self.executed.append((sql, list(seq)))

    fake_conn = _FakeConn()

    @contextmanager
    def fake_get_connection(*args, **kwargs):
        yield fake_conn

    candle_map = {
        "BTC": [{"t": base_ms - 60_000, "c": 100.0}, {"t": base_ms + 60 * 60_000, "c": 105.0}],
        "ETH": [{"t": base_ms - 60_000, "c": 100.0}, {"t": base_ms + 60 * 60_000, "c": 97.0}],
        "SOL": [{"t": base_ms - 60_000, "c": 100.0}, {"t": base_ms + 60 * 60_000, "c": 100.5}],
    }

    monkeypatch.setattr(db_mod, "get_backend_name", lambda: "sqlite")
    monkeypatch.setattr(db_mod, "get_connection", fake_get_connection)
    monkeypatch.setattr(
        hl_mod,
        "get_candles",
        lambda coin, **kw: candle_map.get(coin.upper(), []),
    )

    stats = fc.label_predictions_with_forward_returns(
        forward_minutes=60, crash_pct=-0.015, bullish_pct=0.015,
        batch_size=10, min_age_minutes=0,
    )

    assert stats["scanned"] == 3
    assert stats["labeled"] == 3
    flat = []
    for _sql, seq in fake_conn.executed:
        flat.extend(seq)
    by_id = {rid: label for label, rid in flat}
    assert by_id[11] == REGIME_LABELS["bullish"]  # BTC +5%
    assert by_id[22] == REGIME_LABELS["crash"]    # ETH -3%
    assert by_id[33] == REGIME_LABELS["neutral"]  # SOL +0.5%


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

        # Mock the feature extraction.  ★ AUDIT FIX: _extract_features now
        # returns ``(features, missing_features)`` so callers can flag
        # predictions degraded when an upstream API failed.  Tests that
        # mock this method must return the tuple shape.
        features = {f: 0.5 for f in FEATURE_NAMES}
        with patch.object(forecaster, "_extract_features", return_value=(features, [])):
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
        with patch.object(forecaster, "_extract_features", return_value=(features, [])):
            with patch.object(forecaster, "_store_prediction"):
                forecaster.prediction_cache.clear()
                result = forecaster.predict_regime("BTC")

        assert result["regime"] == "crash"
        assert result["signal"] < 0  # Negative signal for crash
        assert result["confidence"] >= 0.8

    @patch("src.signals.xgboost_regime_forecaster.HAS_XGBOOST", True)
    def test_synthetic_warm_start_caps_prediction_confidence(self, forecaster):
        """Synthetic warm-start models are non-authoritative live inputs."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.9, 0.05, 0.05]])
        forecaster.model = mock_model
        forecaster._last_train_ts = time.time()
        forecaster._model_training_source = "synthetic"
        forecaster._model_observed_rows = 0
        forecaster._model_uses_synthetic_warm_start = True
        forecaster._synthetic_max_confidence = 0.60

        features = {f: 0.0 for f in FEATURE_NAMES}
        with patch.object(forecaster, "_extract_features", return_value=(features, [])):
            with patch.object(forecaster, "_store_prediction"):
                forecaster.prediction_cache.clear()
                result = forecaster.predict_regime("BTC")

        assert result["regime"] == "crash"
        assert result["raw_confidence"] == pytest.approx(0.9)
        assert result["confidence"] == pytest.approx(0.6)
        assert result["signal"] == pytest.approx(-0.6)
        assert result["synthetic_warm_start"] is True
        assert result["training_source"] == "synthetic"

    def test_prediction_caching(self, forecaster):
        """Second call within cache_ttl returns cached result."""
        forecaster.model = None
        with patch("src.signals.xgboost_regime_forecaster.HAS_XGBOOST", False):
            r1 = forecaster.predict_regime("BTC")
            r2 = forecaster.predict_regime("BTC")
        # Both should be identical (cached)
        assert r1["regime"] == r2["regime"]
        assert r1["confidence"] == r2["confidence"]

    def test_synthetic_warm_start_model_is_not_persisted(self, forecaster, tmp_path):
        mock_model = MagicMock()
        forecaster.model = mock_model
        forecaster.model_path = str(tmp_path / "regime_xgboost.json")
        forecaster._model_training_source = "synthetic"
        forecaster._model_observed_rows = 0
        forecaster._model_uses_synthetic_warm_start = True

        forecaster._save_model()

        mock_model.save_model.assert_not_called()
        assert not (tmp_path / "regime_xgboost.json").exists()
        assert not (tmp_path / "regime_xgboost.json.meta.json").exists()

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

    def test_postgres_regime_history_setup_bootstraps_postgres_table(self, forecaster, monkeypatch):
        """Postgres mode should create native regime_history schema without SQLite PRAGMA."""

        class _DummyConn:
            def __init__(self):
                self.executescript_sql = None
                self.execute_calls = []

            def execute(self, sql, params=None):
                self.execute_calls.append((sql, params))
                return MagicMock(fetchall=lambda: [])

            def executescript(self, sql):
                self.executescript_sql = sql

        dummy = _DummyConn()

        @contextmanager
        def _ctx(*, for_read: bool = False):
            yield dummy

        monkeypatch.setattr("src.data.database.get_backend_name", lambda: "postgres")
        monkeypatch.setattr("src.data.database.get_connection", _ctx)

        forecaster._ensure_regime_history_table()

        assert dummy.executescript_sql is not None
        assert "CREATE TABLE IF NOT EXISTS regime_history" in dummy.executescript_sql
        assert "PRAGMA table_info" not in dummy.executescript_sql
        assert "datetime('now')" not in dummy.executescript_sql
        assert dummy.execute_calls == []
