"""
XGBoost Regime Forecaster (V2 — DB-backed walk-forward)
========================================================
ML-based upgrade to PredictiveRegimeForecaster.

Uses an XGBoost gradient-boosted classifier trained on the bot's own
SQLite history (``regime_history`` table) with walk-forward retraining
every 24 hours.

Features (8-input model):
  1. Funding rate              (Hyperliquid public API)
  2. Funding rate slope         (linear regression over last N observations)
  3. Orderbook imbalance        (bid/ask depth ratio)
  4. Arkham smart-money flow    (optional, key-gated)
  5. 5-minute volatility        (from HL candle snapshots)
  6. CEX-DEX basis spread       (HL vs Binance funding delta)
  7. Polymarket sentiment       (injected each cycle)
  8. Options flow conviction    (injected each cycle)

Regime output: "crash" / "neutral" / "bullish" with confidence score.
Same interface as PredictiveRegimeForecaster so it's a drop-in replacement.

Requires: scikit-learn>=1.3.0, xgboost>=2.0.0 (optional deps).
If not installed, falls back to the weighted-signal PredictiveRegimeForecaster.
"""

import logging
import os
import time
from typing import Dict, Optional, List
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Try importing ML libraries (optional)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.info("XGBoost not installed — ML forecaster disabled. "
                "pip install xgboost scikit-learn to enable.")

from src.signals.predictive_regime_forecaster import PredictiveRegimeForecaster

# Try importing crypto.com client for enhanced signals
try:
    from src.data.cryptocom_client import CryptoComClient
    _cryptocom = CryptoComClient()
    HAS_CRYPTOCOM = True
except ImportError:
    _cryptocom = None
    HAS_CRYPTOCOM = False


# Feature order must match training and prediction
FEATURE_NAMES = [
    "funding_rate", "funding_slope", "orderbook_imbalance",
    "arkham_flow", "volatility_5m", "basis_spread",
    "polymarket_sentiment", "options_flow_conviction",
]

# Label encoding: crash=0, neutral=1, bullish=2
REGIME_LABELS = {"crash": 0, "neutral": 1, "bullish": 2}
REGIME_NAMES = {0: "crash", 1: "neutral", 2: "bullish"}

# ─── DB schema for training data ──────────────────────────────────
_REGIME_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS regime_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT (datetime('now')),
    coin TEXT DEFAULT 'BTC',
    funding_rate REAL DEFAULT 0,
    funding_slope REAL DEFAULT 0,
    orderbook_imbalance REAL DEFAULT 0,
    arkham_flow REAL DEFAULT 0,
    volatility_5m REAL DEFAULT 0,
    basis_spread REAL DEFAULT 0,
    polymarket_sentiment REAL DEFAULT 0,
    options_flow_conviction REAL DEFAULT 0,
    regime_label INTEGER DEFAULT 1,
    confidence REAL DEFAULT 0,
    predicted_regime TEXT DEFAULT 'neutral'
)
"""


class XGBoostRegimeForecaster:
    """
    XGBoost-powered regime forecaster with DB-backed walk-forward training.

    Falls back to PredictiveRegimeForecaster if XGBoost is not installed
    or if insufficient training data exists.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}

        self.model_path = cfg.get("model_path", "models/regime_xgboost.json")
        self.min_samples = cfg.get("min_training_samples", 100)
        self.retrain_interval = cfg.get("retrain_interval", 86400)  # 24h
        self.cache_ttl = cfg.get("cache_ttl", 120)  # 2 min prediction cache
        self.prediction_cache: Dict[str, Dict] = {}

        # Fallback predictor (uses the hand-tuned 5-input model)
        self.fallback = PredictiveRegimeForecaster(cfg)

        # Model state
        self.model = None
        self._last_train_ts = 0

        # Ensure models dir exists
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)

        # Ensure DB table exists
        self._ensure_regime_history_table()

        # Try loading a saved model
        if HAS_XGBOOST:
            self._load_model()

        # If no saved model, try initial train from DB or synthetic data
        if HAS_XGBOOST and self.model is None:
            try:
                self.train()
            except Exception as e:
                logger.warning("Initial XGBoost training failed (will use fallback): %s", e)

        logger.info(
            "XGBoostRegimeForecaster V2 initialized (model=%s, has_xgb=%s)",
            "loaded" if self.model else "fallback", HAS_XGBOOST,
        )

    # ─── Pass-through: external data injection ──────────────────────

    def update_polymarket_sentiment(self, sentiment: Dict) -> None:
        """Pass through to fallback forecaster."""
        self.fallback.update_polymarket_sentiment(sentiment)

    def update_options_flow(self, convictions: list) -> None:
        """Pass through to fallback forecaster."""
        self.fallback.update_options_flow(convictions)

    # ─── Prediction ─────────────────────────────────────────────────

    def predict_regime(self, coin: str = "BTC") -> Dict:
        """
        Predict regime using XGBoost if available, else fallback.
        Each prediction is stored in DB for future walk-forward training.

        Returns same schema as PredictiveRegimeForecaster.
        """
        now = time.time()

        # Cache check
        if coin in self.prediction_cache:
            cached = self.prediction_cache[coin]
            if now - cached.get("ts", 0) < self.cache_ttl:
                return cached["data"]

        # Get base prediction + features from the fallback forecaster
        base = self.fallback.predict_regime(coin)

        # Extract the full 8-feature vector
        features = self._extract_features(coin, base)

        # Auto-retrain check
        if (HAS_XGBOOST and
                now - self._last_train_ts > self.retrain_interval):
            try:
                self.train()
            except Exception as exc:
                logger.debug("Auto-retrain skipped: %s", exc)

        # Use ML model if available
        if HAS_XGBOOST and self.model is not None:
            try:
                X = np.array([[features.get(f, 0.0) for f in FEATURE_NAMES]])
                proba = self.model.predict_proba(X)[0]
                pred_class = int(np.argmax(proba))
                regime = REGIME_NAMES[pred_class]
                confidence = float(proba[pred_class])

                result = {
                    "signal": confidence if regime == "bullish"
                              else -confidence if regime == "crash"
                              else 0.0,
                    "regime": regime,
                    "confidence": round(confidence, 4),
                    "model": "xgboost",
                    "probabilities": {
                        "crash": round(float(proba[0]), 4),
                        "neutral": round(float(proba[1]), 4),
                        "bullish": round(float(proba[2]), 4),
                    },
                    "components": base.get("components", {}),
                    "active_inputs": base.get("active_inputs", 0),
                }

                self.prediction_cache[coin] = {"data": result, "ts": now}

                logger.info(
                    "XGBoost Forecaster %s → %s (conf=%.1f%%, signal=%.3f)",
                    coin, regime, confidence * 100, result["signal"],
                )

                # Store prediction for future training
                self._store_prediction(coin, features, regime, confidence)

                return result

            except Exception as e:
                logger.debug("XGBoost prediction failed, using fallback: %s", e)

        # Fallback
        base["model"] = "weighted_signal"
        self.prediction_cache[coin] = {"data": base, "ts": now}

        # Still store for training (using the fallback's regime label)
        self._store_prediction(coin, features, base["regime"], base.get("confidence", 0))

        return base

    # ─── Feature Extraction ─────────────────────────────────────────

    def _extract_features(self, coin: str, base_prediction: Dict) -> Dict:
        """
        Extract the full 8-feature vector from current market state.
        Reuses the base forecaster's components + adds volatility & basis.
        Enhanced with cross-exchange validation from Crypto.com.
        """
        components = base_prediction.get("components", {})

        # Start with features the base forecaster already computed
        features = {
            "funding_slope": components.get("funding_slope", 0.0),
            "orderbook_imbalance": components.get("imbalance", 0.0),
            "arkham_flow": components.get("arkham_flow", 0.0),
            "polymarket_sentiment": components.get("polymarket", 0.0),
            "options_flow_conviction": components.get("options_flow", 0.0),
        }

        # Funding rate (raw, not slope)
        features["funding_rate"] = self._get_funding_rate(coin)

        # 5-minute volatility (from HL candle API)
        features["volatility_5m"] = self._get_5m_volatility(coin)

        # CEX-DEX basis spread (HL vs Binance)
        features["basis_spread"] = self._get_basis_spread(coin)

        # Cross-exchange volatility from Crypto.com (validation signal)
        if HAS_CRYPTOCOM:
            try:
                cdc_vol = _cryptocom.get_5m_volatility(coin)
                if cdc_vol > 0:
                    # Blend with existing volatility: average of both sources
                    existing_vol = features.get("volatility_5m", 0)
                    if existing_vol > 0:
                        features["volatility_5m"] = (existing_vol + cdc_vol) / 2
                    else:
                        features["volatility_5m"] = cdc_vol
            except Exception:
                pass

        return features

    def _get_funding_rate(self, coin: str) -> float:
        """Get current funding rate from Hyperliquid."""
        try:
            import requests
            resp = requests.post(
                self.fallback.hl_info_url,
                json={"type": "metaAndAssetCtxs"},
                timeout=5,
            )
            if resp.ok:
                data = resp.json()
                if len(data) >= 2:
                    meta, asset_ctxs = data[0], data[1]
                    for i, asset in enumerate(meta.get("universe", [])):
                        if asset.get("name", "").upper() == coin.upper() and i < len(asset_ctxs):
                            return float(asset_ctxs[i].get("funding", 0))
        except Exception:
            pass
        return 0.0

    def _get_5m_volatility(self, coin: str) -> float:
        """
        Compute recent 5-min return volatility from Hyperliquid candle data.
        Returns normalized value in [0, 1] range.
        """
        try:
            import requests
            now_ms = int(time.time() * 1000)
            resp = requests.post(
                self.fallback.hl_info_url,
                json={
                    "type": "candleSnapshot",
                    "req": {
                        "coin": coin,
                        "interval": "5m",
                        "startTime": now_ms - 3_600_000,  # last hour
                        "endTime": now_ms,
                    },
                },
                timeout=5,
            )
            if resp.ok:
                candles = resp.json()
                if len(candles) >= 5:
                    closes = [float(c["c"]) for c in candles[-12:]]
                    if len(closes) >= 2:
                        returns = np.diff(np.log(closes))
                        vol = float(np.std(returns))
                        # Normalize: typical 5-min vol ~0.001-0.01
                        return min(vol * 100, 1.0)
        except Exception:
            pass
        return 0.0

    def _get_basis_spread(self, coin: str) -> float:
        """
        CEX-DEX basis: Hyperliquid funding minus Binance funding, enhanced with Crypto.com data.
        Positive = HL funding higher (shorts pay more on HL vs Binance).
        Multi-exchange basis uses weighted average if crypto.com data is available.
        """
        try:
            import requests as req
            # Hyperliquid funding (reuse from fallback's internal history)
            hl_funding = 0.0
            hist = getattr(self.fallback, '_funding_history', {})
            if coin in hist and hist[coin]:
                hl_funding = hist[coin][-1]
            else:
                hl_funding = self._get_funding_rate(coin)

            # Binance funding (public endpoint, no API key needed)
            resp = req.get(
                "https://fapi.binance.com/fapi/v1/premiumIndex",
                params={"symbol": f"{coin}USDT"},
                timeout=5,
            )
            if resp.ok:
                binance_funding = float(resp.json().get("lastFundingRate", 0))
                basis = hl_funding - binance_funding

                # Crypto.com price comparison (spot-perp basis proxy)
                if HAS_CRYPTOCOM:
                    try:
                        ticker = _cryptocom.get_ticker(coin)
                        if ticker:
                            cdc_price = ticker.get("price", 0)
                            if cdc_price > 0:
                                # Try to get orderbook imbalance from Crypto.com as additional signal
                                cdc_imbalance = 0.0
                                try:
                                    cdc_imbalance = _cryptocom.get_orderbook_imbalance(coin)
                                except Exception:
                                    pass

                                # Multi-exchange basis: average HL-Binance funding spread with CDC orderbook imbalance
                                # Blend: 70% funding basis + 30% orderbook imbalance signal
                                if cdc_imbalance != 0:
                                    basis = basis * 0.7 + cdc_imbalance * 0.0003 * 0.3
                    except Exception:
                        pass

                return max(min(basis * 10_000, 1.0), -1.0)
        except Exception:
            pass
        return 0.0

    # ─── Model Training (DB-backed walk-forward) ────────────────────

    def train(self) -> Optional[Dict]:
        """
        Train XGBoost on DB regime_history + any in-memory accumulation.
        Uses walk-forward: always trains on latest 90 days of data.
        """
        if not HAS_XGBOOST:
            logger.warning("XGBoost not installed — cannot train")
            return None

        X, y = self._get_training_data()

        if len(y) < self.min_samples:
            logger.info(
                "Insufficient data for training: %d/%d samples",
                len(y), self.min_samples,
            )
            return None

        logger.info("Training XGBoost regime forecaster on %d samples...", len(y))

        self.model = xgb.XGBClassifier(
            n_estimators=180,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=3,
            use_label_encoder=False,
            eval_metric="mlogloss",
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42,
            verbosity=0,
        )

        # Cross-validation (if enough data)
        cv_mean = 0.0
        try:
            from sklearn.model_selection import cross_val_score
            n_splits = min(5, max(2, len(X) // 20))
            cv_scores = cross_val_score(self.model, X, y, cv=n_splits, scoring="accuracy")
            cv_mean = float(np.mean(cv_scores))
            logger.info(
                "XGBoost CV accuracy: %.3f (+/- %.3f)",
                cv_mean, float(np.std(cv_scores)),
            )
        except Exception:
            pass

        # Train on full data
        self.model.fit(X, y)
        self._last_train_ts = time.time()

        # Save model
        self._save_model()

        # Clear prediction cache since model changed
        self.prediction_cache.clear()

        metrics = {
            "samples": len(y),
            "cv_accuracy": round(cv_mean, 4),
            "feature_importance": dict(zip(
                FEATURE_NAMES,
                [round(float(v), 4) for v in self.model.feature_importances_],
            )),
        }
        logger.info("XGBoost trained: %s", metrics)
        return metrics

    def _get_training_data(self):
        """
        Pull training data from regime_history table (last 90 days).
        Falls back to synthetic warm-start if insufficient history.
        """
        X_rows = []
        y_rows = []

        try:
            from src.data.database import get_connection
            with get_connection() as conn:
                rows = conn.execute("""
                    SELECT funding_rate, funding_slope, orderbook_imbalance,
                           arkham_flow, volatility_5m, basis_spread,
                           polymarket_sentiment, options_flow_conviction,
                           regime_label
                    FROM regime_history
                    WHERE timestamp > datetime('now', '-90 days')
                    ORDER BY timestamp DESC
                """).fetchall()

            if rows:
                X_rows = np.array(
                    [[float(r[i]) for i in range(8)] for r in rows],
                    dtype=np.float32,
                )
                y_rows = np.array(
                    [int(r[8]) for r in rows],
                    dtype=np.int32,
                )
                logger.info("Loaded %d rows from regime_history for training", len(y_rows))

                if len(y_rows) >= self.min_samples:
                    return X_rows, y_rows

        except Exception as exc:
            logger.debug("Could not load training data from DB: %s", exc)

        # Fallback: synthetic warm-start with realistic distributions
        logger.warning(
            "Using synthetic warm-start (%d DB rows, need %d)",
            len(y_rows) if isinstance(y_rows, np.ndarray) else 0,
            self.min_samples,
        )
        n = 2000
        rng = np.random.RandomState(42)
        X_synth = np.column_stack([
            rng.normal(0, 0.0005, n),   # funding_rate
            rng.normal(0, 0.3, n),      # funding_slope
            rng.normal(0, 0.2, n),      # orderbook_imbalance
            rng.normal(0, 0.15, n),     # arkham_flow
            rng.exponential(0.03, n),   # volatility_5m (always positive)
            rng.normal(0, 0.1, n),      # basis_spread
            rng.normal(0, 0.25, n),     # polymarket_sentiment
            rng.normal(0, 0.2, n),      # options_flow_conviction
        ]).astype(np.float32)

        # Labels: derived from weighted composite to be internally consistent
        composite = (X_synth[:, 1] * 0.30 + X_synth[:, 2] * 0.25 +
                     X_synth[:, 6] * 0.20 + rng.normal(0, 0.15, n))
        y_synth = np.where(composite < -0.15, 0, np.where(composite > 0.15, 2, 1)).astype(np.int32)

        # If we have some DB rows, prepend them
        if isinstance(X_rows, np.ndarray) and len(X_rows) > 0:
            X_synth = np.vstack([X_rows, X_synth])
            y_synth = np.concatenate([y_rows, y_synth])

        return X_synth, y_synth

    # ─── Prediction Storage ──────────────────────────────────────────

    def _store_prediction(self, coin: str, features: Dict, regime: str, confidence: float):
        """Store each prediction in regime_history for future training."""
        try:
            from src.data.database import get_connection
            with get_connection() as conn:
                conn.execute(
                    """INSERT INTO regime_history
                       (coin, funding_rate, funding_slope, orderbook_imbalance,
                        arkham_flow, volatility_5m, basis_spread,
                        polymarket_sentiment, options_flow_conviction,
                        regime_label, confidence, predicted_regime)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        coin,
                        features.get("funding_rate", 0),
                        features.get("funding_slope", 0),
                        features.get("orderbook_imbalance", 0),
                        features.get("arkham_flow", 0),
                        features.get("volatility_5m", 0),
                        features.get("basis_spread", 0),
                        features.get("polymarket_sentiment", 0),
                        features.get("options_flow_conviction", 0),
                        REGIME_LABELS.get(regime, 1),
                        confidence,
                        regime,
                    ),
                )
        except Exception as exc:
            logger.debug("Failed to store prediction: %s", exc)

    # ─── DB Schema ──────────────────────────────────────────────────

    def _ensure_regime_history_table(self):
        """Create regime_history table if it doesn't exist."""
        try:
            from src.data.database import get_connection
            with get_connection() as conn:
                conn.execute(_REGIME_HISTORY_DDL)
        except Exception as exc:
            logger.debug("Could not create regime_history table: %s", exc)

    # ─── Model Persistence ──────────────────────────────────────────

    def _save_model(self):
        """Save trained model to disk."""
        if self.model is None:
            return
        try:
            self.model.save_model(self.model_path)
            logger.info("XGBoost model saved to %s", self.model_path)
        except Exception as e:
            logger.error("Failed to save XGBoost model: %s", e)

    def _load_model(self):
        """Load saved model from disk."""
        if not os.path.exists(self.model_path):
            return
        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
            self._last_train_ts = os.path.getmtime(self.model_path)
            logger.info("XGBoost model loaded from %s", self.model_path)
        except Exception as e:
            logger.debug("Could not load XGBoost model: %s", e)
            self.model = None

    # ─── Stats ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return forecaster statistics."""
        return {
            "model_loaded": self.model is not None,
            "model_type": "xgboost" if self.model else "fallback",
            "last_train_ts": self._last_train_ts,
            "has_xgboost": HAS_XGBOOST,
            "cache_size": len(self.prediction_cache),
        }
