from unittest.mock import patch

from src.signals.feature_store_alpha import AlphaPrediction, FeatureStoreAlphaPipeline


class _FakeMember:
    def __init__(self, prob_up):
        self.prob_up = prob_up

    def predict_proba(self, X):
        return [[1.0 - self.prob_up, self.prob_up] for _ in range(len(X))]


def _pipeline(tmp_path):
    with patch.object(FeatureStoreAlphaPipeline, "_load_models", lambda self: None):
        with patch.object(FeatureStoreAlphaPipeline, "train_if_due", lambda self, force=False: {}):
            return FeatureStoreAlphaPipeline({"model_dir": str(tmp_path)})


def test_build_training_frame_creates_forward_labels():
    feature_rows = [
        {"coin": "BTC", "timestamp_ms": 1, "feature_name": "return_1", "value": 0.1},
        {"coin": "BTC", "timestamp_ms": 2, "feature_name": "return_1", "value": 0.2},
        {"coin": "BTC", "timestamp_ms": 3, "feature_name": "return_1", "value": 0.3},
        {"coin": "BTC", "timestamp_ms": 4, "feature_name": "return_1", "value": 0.4},
        {"coin": "BTC", "timestamp_ms": 5, "feature_name": "return_1", "value": 0.5},
    ]
    candle_rows = [
        {"coin": "BTC", "timestamp_ms": 1, "close": 100.0},
        {"coin": "BTC", "timestamp_ms": 2, "close": 102.0},
        {"coin": "BTC", "timestamp_ms": 3, "close": 101.0},
        {"coin": "BTC", "timestamp_ms": 4, "close": 103.0},
        {"coin": "BTC", "timestamp_ms": 5, "close": 104.0},
    ]

    frame = FeatureStoreAlphaPipeline.build_training_frame(
        feature_rows=feature_rows,
        candle_rows=candle_rows,
        feature_names=["return_1"],
    )

    assert not frame.empty
    assert "forward_return_1h" in frame.columns
    assert "label_4h" in frame.columns
    first = frame.iloc[0]
    assert round(float(first["forward_return_1h"]), 4) == 0.02
    assert float(first["label_1h"]) == 1.0


def test_prediction_eligible_respects_model_significance(tmp_path):
    pipeline = _pipeline(tmp_path)
    pipeline.signal_min_confidence = 0.58
    pipeline.model_metadata["1h"] = {"eligible": True}
    pipeline.model_metadata["4h"] = {"eligible": False}

    assert pipeline._prediction_eligible("1h", 0.60) is True
    assert pipeline._prediction_eligible("1h", 0.55) is False
    assert pipeline._prediction_eligible("4h", 0.90) is False


def test_generate_signals_combines_aligned_horizons(monkeypatch, tmp_path):
    pipeline = _pipeline(tmp_path)
    monkeypatch.setattr("src.signals.feature_store_alpha.HAS_ALPHA_ML", True)
    monkeypatch.setattr(pipeline, "_pg_available", lambda: True)
    pipeline.models = {"1h": [_FakeMember(0.7)], "4h": [_FakeMember(0.68)]}
    pipeline.model_metadata = {
        "1h": {"eligible": True, "model_version": "v1", "significance_pvalue": 0.04},
        "4h": {"eligible": True, "model_version": "v2", "significance_pvalue": 0.03},
    }
    monkeypatch.setattr(pipeline, "_prediction_coins", lambda: ["BTC"])
    monkeypatch.setattr(
        pipeline,
        "predict_coin",
        lambda coin, horizon: {
            "1h": AlphaPrediction(
                coin="BTC",
                timeframe="1h",
                horizon="1h",
                feature_timestamp_ms=123,
                raw_probability_up=0.70,
                calibrated_probability_up=0.71,
                confidence=0.71,
                predicted_side="long",
                expected_return_bps=18.0,
                significance_pvalue=0.04,
                eligible=True,
                model_version="v1",
            ),
            "4h": AlphaPrediction(
                coin="BTC",
                timeframe="1h",
                horizon="4h",
                feature_timestamp_ms=123,
                raw_probability_up=0.68,
                calibrated_probability_up=0.69,
                confidence=0.69,
                predicted_side="long",
                expected_return_bps=24.0,
                significance_pvalue=0.03,
                eligible=True,
                model_version="v2",
            ),
        }[horizon],
    )

    signals = pipeline.generate_signals()

    assert len(signals) == 1
    signal = signals[0]
    assert signal["source"] == "ml_alpha"
    assert signal["side"] == "long"
    assert signal["parameters"]["horizons"] == ["1h", "4h"]
    assert signal["confidence"] > 0.68


def test_generate_signals_skips_conflicting_horizons_without_clear_winner(monkeypatch, tmp_path):
    pipeline = _pipeline(tmp_path)
    monkeypatch.setattr("src.signals.feature_store_alpha.HAS_ALPHA_ML", True)
    monkeypatch.setattr(pipeline, "_pg_available", lambda: True)
    pipeline.models = {"1h": [_FakeMember(0.7)], "4h": [_FakeMember(0.3)]}
    pipeline.model_metadata = {
        "1h": {"eligible": True, "model_version": "v1", "significance_pvalue": 0.04},
        "4h": {"eligible": True, "model_version": "v2", "significance_pvalue": 0.03},
    }
    monkeypatch.setattr(pipeline, "_prediction_coins", lambda: ["BTC"])
    monkeypatch.setattr(
        pipeline,
        "predict_coin",
        lambda coin, horizon: {
            "1h": AlphaPrediction("BTC", "1h", "1h", 123, 0.70, 0.70, 0.70, "long", 10.0, 0.04, True, "v1"),
            "4h": AlphaPrediction("BTC", "1h", "4h", 123, 0.46, 0.45, 0.59, "short", 8.0, 0.03, True, "v2"),
        }[horizon],
    )

    assert pipeline.generate_signals() == []
