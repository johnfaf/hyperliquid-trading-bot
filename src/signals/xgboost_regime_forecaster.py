"""
XGBoost Regime Forecaster
==========================
ML-based upgrade to PredictiveRegimeForecaster.

Uses an XGBoost gradient-boosted classifier trained on features:
  - Funding rate (current + slope)
  - Orderbook imbalance
  - Arkham smart-money net flow
  - Volatility (rolling std of returns)
  - Volume profile (current vs 20-period SMA)

The model auto-trains on first run from accumulated feature snapshots,
and saves to models/regime_xgboost.json for persistence.

Requires: scikit-learn>=1.3.0, xgboost>=2.0.0 (optional deps).
If not installed, falls back to the weighted-signal PredictiveRegimeForecaster.
"""

import logging
import os
import time
import json
from typing import Dict, Optional, List
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Try importing ML libraries
try:
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.info("XGBoost not installed — ML forecaster disabled. "
                "pip install xgboost scikit-learn to enable.")

from src.signals.predictive_regime_forecaster import PredictiveRegimeForecaster


# Feature order: [funding, funding_slope, imbalance, arkham_flow, volatility, volume_ratio]
FEATURE_NAMES = [
    "funding_rate", "funding_slope", "orderbook_imbalance",
    "arkham_flow", "volatility_20", "volume_ratio"
]

# Label encoding: crash=0, neutral=1, bullish=2
REGIME_LABELS = {"crash": 0, "neutral": 1, "bullish": 2}
REGIME_NAMES = {0: "crash", 1: "neutral", 2: "bullish"}


class XGBoostRegimeForecaster:
    """
    XGBoost-powered regime forecaster with auto-training and persistence.

    Falls back to PredictiveRegimeForecaster if XGBoost is not installed
    or if insufficient training data exists.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}

        self.model_path = cfg.get("model_path", "models/regime_xgboost.json")
        self.min_samples = cfg.get("min_training_samples", 100)
        self.retrain_interval = cfg.get("retrain_interval", 86400)  # 24h

        # Fallback predictor
        self.fallback = PredictiveRegimeForecaster(cfg)

        # Training data accumulator
        self._feature_log: List[Dict] = []
        self._max_log_size = cfg.get("max_log_size", 10000)

        # Model state
        self.model = None
        self._last_train_ts = 0

        # Try loading saved model
        if HAS_XGBOOST:
            self._load_model()

    def predict_regime(self, coin: str = "BTC") -> Dict:
        """
        Predict regime using XGBoost if available, else fallback.

        Always logs features for future training.
        """
        # Get base prediction + features from fallback
        base = self.fallback.predict_regime(coin)

        # Extract features
        features = self._extract_features(coin, base)

        # Log for training
        self._log_features(features, base["regime"])

        # Use ML model if available and trained
        if HAS_XGBOOST and self.model is not None:
            try:
                X = np.array([[features.get(f, 0.0) for f in FEATURE_NAMES]])
                proba = self.model.predict_proba(X)[0]
                pred_class = int(np.argmax(proba))
                regime = REGIME_NAMES[pred_class]
                confidence = float(proba[pred_class])

                return {
                    "signal": base["signal"],
                    "regime": regime,
                    "confidence": round(confidence, 4),
                    "model": "xgboost",
                    "probabilities": {
                        "crash": round(float(proba[0]), 4),
                        "neutral": round(float(proba[1]), 4),
                        "bullish": round(float(proba[2]), 4),
                    },
                    "components": base.get("components", {}),
                }
            except Exception as e:
                logger.debug(f"XGBoost prediction failed, using fallback: {e}")

        # Fallback
        base["model"] = "weighted_signal"
        return base

    def _extract_features(self, coin: str, base_prediction: Dict) -> Dict:
        """Extract feature vector from current market state."""
        components = base_prediction.get("components", {})

        # Funding rate (current, not slope)
        funding_rate = 0.0
        try:
            import requests
            resp = requests.post(
                self.fallback.hl_info_url,
                json={"type": "metaAndAssetCtxs"},
                timeout=5
            )
            if resp.ok:
                data = resp.json()
                if len(data) >= 2:
                    meta = data[0]
                    asset_ctxs = data[1]
                    for i, asset in enumerate(meta.get("universe", [])):
                        if asset.get("name", "").upper() == coin.upper() and i < len(asset_ctxs):
                            funding_rate = float(asset_ctxs[i].get("funding", 0))
                            break
        except Exception:
            pass

        return {
            "funding_rate": funding_rate,
            "funding_slope": components.get("funding_slope", 0.0),
            "orderbook_imbalance": components.get("imbalance", 0.0),
            "arkham_flow": components.get("arkham_flow", 0.0),
            "volatility_20": 0.0,  # Populated from candle data when available
            "volume_ratio": 0.0,   # Populated from candle data when available
        }

    def _log_features(self, features: Dict, regime: str):
        """Log feature snapshot for model training."""
        entry = {
            "ts": time.time(),
            "features": features,
            "label": regime,
        }
        self._feature_log.append(entry)
        if len(self._feature_log) > self._max_log_size:
            self._feature_log = self._feature_log[-self._max_log_size:]

        # Auto-train if enough data and interval elapsed
        now = time.time()
        if (len(self._feature_log) >= self.min_samples and
                now - self._last_train_ts > self.retrain_interval and
                HAS_XGBOOST):
            self.train()

    def train(self) -> Optional[Dict]:
        """
        Train XGBoost model on accumulated feature logs.

        Returns training metrics or None if insufficient data.
        """
        if not HAS_XGBOOST:
            logger.warning("XGBoost not installed — cannot train")
            return None

        if len(self._feature_log) < self.min_samples:
            logger.info(f"Insufficient data for training: "
                       f"{len(self._feature_log)}/{self.min_samples}")
            return None

        logger.info(f"Training XGBoost regime forecaster on {len(self._feature_log)} samples...")

        # Build training matrices
        X = np.array([
            [entry["features"].get(f, 0.0) for f in FEATURE_NAMES]
            for entry in self._feature_log
        ])
        y = np.array([
            REGIME_LABELS.get(entry["label"], 1)
            for entry in self._feature_log
        ])

        # XGBoost classifier
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=3,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )

        # Cross-validation
        try:
            cv_scores = cross_val_score(self.model, X, y, cv=min(5, len(X) // 10 + 1),
                                         scoring="accuracy")
            cv_mean = float(np.mean(cv_scores))
            logger.info(f"XGBoost CV accuracy: {cv_mean:.3f} (+/- {np.std(cv_scores):.3f})")
        except Exception:
            cv_mean = 0.0

        # Train on full data
        self.model.fit(X, y)
        self._last_train_ts = time.time()

        # Save model
        self._save_model()

        metrics = {
            "samples": len(self._feature_log),
            "cv_accuracy": round(cv_mean, 4),
            "feature_importance": dict(zip(FEATURE_NAMES,
                                            self.model.feature_importances_.tolist())),
        }
        logger.info(f"XGBoost trained: {metrics}")
        return metrics

    def _save_model(self):
        """Save trained model to disk."""
        if self.model is None:
            return
        try:
            path = Path(self.model_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save_model(str(path))
            logger.info(f"XGBoost model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save XGBoost model: {e}")

    def _load_model(self):
        """Load saved model from disk."""
        if not os.path.exists(self.model_path):
            return
        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
            logger.info(f"XGBoost model loaded from {self.model_path}")
        except Exception as e:
            logger.debug(f"Could not load XGBoost model: {e}")
            self.model = None

    def get_stats(self) -> Dict:
        """Return forecaster statistics."""
        return {
            "model_loaded": self.model is not None,
            "feature_log_size": len(self._feature_log),
            "min_samples_for_training": self.min_samples,
            "last_train_ts": self._last_train_ts,
            "has_xgboost": HAS_XGBOOST,
        }
