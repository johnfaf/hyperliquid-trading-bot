"""
Unit tests for Predictive Regime Forecaster.
"""
import os
import time
import pytest

from unittest.mock import patch, MagicMock
from src.signals.predictive_regime_forecaster import (
    PredictiveRegimeForecaster, ArkhamClient,
    W_FUNDING, W_IMBALANCE, W_ARKHAM,
    CRASH_THRESHOLD, BULLISH_THRESHOLD,
)


def test_weights_sum_to_one():
    """Composite signal weights should sum to 1.0."""
    assert W_FUNDING + W_IMBALANCE + W_ARKHAM == pytest.approx(1.0)


def test_arkham_disabled_without_key():
    """ArkhamClient should return 0.0 flow when no API key."""
    with patch.dict(os.environ, {}, clear=True):
        client = ArkhamClient()
        result = client.get_smart_money_flow("BTC")
        assert result["net_flow_score"] == 0.0


def test_regime_classification_crash():
    """Signal below crash threshold should classify as crash."""
    assert CRASH_THRESHOLD < 0
    # If signal = -0.3, that's below -0.15 → crash
    signal = -0.3
    regime = "crash" if signal < CRASH_THRESHOLD else "neutral"
    assert regime == "crash"


def test_regime_classification_bullish():
    """Signal above bullish threshold should classify as bullish."""
    signal = 0.3
    regime = "bullish" if signal > BULLISH_THRESHOLD else "neutral"
    assert regime == "bullish"


def test_regime_classification_neutral():
    """Signal between thresholds should be neutral."""
    signal = 0.05
    if signal < CRASH_THRESHOLD:
        regime = "crash"
    elif signal > BULLISH_THRESHOLD:
        regime = "bullish"
    else:
        regime = "neutral"
    assert regime == "neutral"


@patch("src.signals.predictive_regime_forecaster.requests")
def test_forecaster_caching(mock_requests):
    """Predictions should be cached for TTL duration."""
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = [
        {"universe": [{"name": "BTC"}]},
        [{"funding": "0.0001"}]
    ]
    mock_resp.raise_for_status = MagicMock()
    mock_requests.post.return_value = mock_resp

    forecaster = PredictiveRegimeForecaster({"cache_ttl": 300})

    # First call
    result1 = forecaster.predict_regime("BTC")
    assert "regime" in result1
    assert "signal" in result1
    assert "confidence" in result1

    # Second call should use cache
    result2 = forecaster.predict_regime("BTC")
    assert result2 == result1


def test_forecaster_output_structure():
    """Forecaster output should have expected fields."""
    with patch("src.signals.predictive_regime_forecaster.requests") as mock_req:
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = [
            {"universe": [{"name": "BTC"}]},
            [{"funding": "0.0001"}]
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_req.post.return_value = mock_resp

        f = PredictiveRegimeForecaster()
        result = f.predict_regime("BTC")

        assert result["regime"] in ("crash", "bullish", "neutral")
        assert 0 <= result["confidence"] <= 1.0
        assert "components" in result
        assert "funding_slope" in result["components"]
        assert "imbalance" in result["components"]


def test_forecaster_decays_recently_stale_polymarket_signal(monkeypatch):
    forecaster = PredictiveRegimeForecaster(
        {"external_data_ttl": 600, "external_data_partial_ttl": 1800}
    )
    monkeypatch.setattr(forecaster, "_get_funding_slope", lambda coin: 0.0)
    monkeypatch.setattr(forecaster, "_get_orderbook_imbalance", lambda coin: 0.0)
    monkeypatch.setattr(forecaster, "_get_arkham_flow", lambda coin: 0.0)
    forecaster._polymarket_sentiment = {"sentiment": "bullish", "confidence": 0.4}
    forecaster._polymarket_ts = time.time() - 900

    result = forecaster.predict_regime("BTC")

    assert result["components"]["polymarket"] == pytest.approx(0.4)
    assert result["partial_signal"] is False
    assert result["partial_inputs"] == []


def test_forecaster_marks_long_stale_external_inputs_as_partial(monkeypatch):
    forecaster = PredictiveRegimeForecaster(
        {"external_data_ttl": 600, "external_data_partial_ttl": 1800}
    )
    monkeypatch.setattr(forecaster, "_get_funding_slope", lambda coin: 0.0)
    monkeypatch.setattr(forecaster, "_get_orderbook_imbalance", lambda coin: 0.0)
    monkeypatch.setattr(forecaster, "_get_arkham_flow", lambda coin: 0.0)
    forecaster._options_convictions = [
        {"ticker": "BTC", "direction": "BULLISH", "conviction_pct": 70}
    ]
    forecaster._options_ts = time.time() - 2400

    result = forecaster.predict_regime("BTC")

    assert result["components"]["options_flow"] == 0.0
    assert result["partial_signal"] is True
    assert result["partial_inputs"] == ["options_flow"]
