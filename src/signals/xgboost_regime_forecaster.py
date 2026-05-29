"""
XGBoost Regime Forecaster (V2 — DB-backed walk-forward)
========================================================
ML-based upgrade to PredictiveRegimeForecaster.

Uses an XGBoost gradient-boosted classifier trained on the bot's own
regime history (``regime_history`` table) with walk-forward retraining
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
# ruff: noqa: E402

import logging
import json
import os
import time
from typing import Dict, List, Optional, Tuple
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
    logger.info("XGBoost not installed -- ML forecaster disabled. "
                "pip install xgboost scikit-learn to enable.")

from src.signals.predictive_regime_forecaster import PredictiveRegimeForecaster
from src.core.api_manager import get_manager, Priority

# Try importing crypto.com client for enhanced signals
try:
    from src.data.cryptocom_client import CryptoComClient
    _cryptocom = CryptoComClient()
    HAS_CRYPTOCOM = True
except ImportError:
    _cryptocom = None
    HAS_CRYPTOCOM = False


# Feature order must match training and prediction.
# NOTE: arkham_flow and polymarket_sentiment were removed — they are always
# zero (no live data source) and just added noise to the model.
FEATURE_NAMES = [
    "funding_rate", "funding_slope", "orderbook_imbalance",
    "volatility_5m", "basis_spread", "options_flow_conviction",
]

# Label encoding: crash=0, neutral=1, bullish=2
REGIME_LABELS = {"crash": 0, "neutral": 1, "bullish": 2}
REGIME_NAMES = {0: "crash", 1: "neutral", 2: "bullish"}

# ─── DB schema for training data ──────────────────────────────────
_SQLITE_REGIME_HISTORY_DDL = """
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
    predicted_regime TEXT DEFAULT 'neutral',
    label_source TEXT DEFAULT 'predicted'
)
"""

_POSTGRES_REGIME_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS regime_history (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    coin TEXT DEFAULT 'BTC',
    funding_rate DOUBLE PRECISION DEFAULT 0,
    funding_slope DOUBLE PRECISION DEFAULT 0,
    orderbook_imbalance DOUBLE PRECISION DEFAULT 0,
    arkham_flow DOUBLE PRECISION DEFAULT 0,
    volatility_5m DOUBLE PRECISION DEFAULT 0,
    basis_spread DOUBLE PRECISION DEFAULT 0,
    polymarket_sentiment DOUBLE PRECISION DEFAULT 0,
    options_flow_conviction DOUBLE PRECISION DEFAULT 0,
    regime_label INTEGER,
    confidence DOUBLE PRECISION DEFAULT 0,
    predicted_regime TEXT DEFAULT 'neutral',
    label_source TEXT DEFAULT 'predicted'
);
CREATE INDEX IF NOT EXISTS idx_regime_history_timestamp
    ON regime_history (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_regime_history_coin_timestamp
    ON regime_history (coin, timestamp DESC);
ALTER TABLE regime_history
    ADD COLUMN IF NOT EXISTS label_source TEXT DEFAULT 'predicted';
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
        # ★ Phase 2 fix: the previous default (100) was too high given the
        # labeler's ~3-min cadence and the bot's typically narrow regime
        # distribution.  Production saw 0 of 15,484 prediction rows reach
        # the threshold, so the forecaster was permanently stuck in
        # synthetic warm-start mode.  Operators can still tune via env.
        try:
            import os as _os
            _env_min = _os.environ.get("XGBOOST_MIN_TRAINING_SAMPLES")
            _env_min_int = int(_env_min) if _env_min else None
        except (TypeError, ValueError):
            _env_min_int = None
        self.min_samples = (
            cfg.get("min_training_samples")
            or _env_min_int
            or 30
        )
        self.retrain_interval = cfg.get("retrain_interval", 86400)  # 24h
        self.cache_ttl = cfg.get("cache_ttl", 120)  # 2 min prediction cache
        self.prediction_cache: Dict[str, Dict] = {}

        # Fallback predictor (uses the hand-tuned 5-input model)
        self.fallback = PredictiveRegimeForecaster(cfg)

        # Model state
        self.model = None
        self._last_train_ts = 0
        self._model_training_source = "untrained"
        self._model_observed_rows = 0
        self._model_uses_synthetic_warm_start = False
        try:
            self._synthetic_max_confidence = float(
                cfg.get(
                    "synthetic_max_confidence",
                    os.environ.get("XGB_SYNTHETIC_MAX_CONFIDENCE", 0.45),
                )
            )
        except (TypeError, ValueError):
            self._synthetic_max_confidence = 0.45
        self._synthetic_max_confidence = max(0.0, min(0.60, self._synthetic_max_confidence))

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

        # Extract the full 8-feature vector + list of features whose
        # underlying API/parse failed.  ★ AUDIT FIX: see _extract_features
        # docstring -- previously this returned only ``features`` and any
        # 0.0 substitution from a failed fetch was invisible to the
        # consumer.
        features, missing_features = self._extract_features(coin, base)
        feature_fetch_degraded = bool(missing_features)
        if feature_fetch_degraded:
            logger.warning(
                "XGBoost forecaster %s: %d/%d external features unavailable "
                "(%s) -- prediction will be flagged degraded and confidence "
                "capped so downstream consumers can de-rate.",
                coin,
                len(missing_features),
                3,  # the three external fetchers we explicitly track
                ", ".join(missing_features),
            )

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
                raw_confidence = float(proba[pred_class])
                confidence = raw_confidence
                if self._model_uses_synthetic_warm_start:
                    confidence = min(confidence, self._synthetic_max_confidence)
                # ★ AUDIT FIX: cap confidence by the fraction of external
                # features that actually arrived.  Each missing feature
                # (funding_rate / volatility_5m / basis_spread) costs 1/3
                # of the maximum.  All three missing => confidence floored
                # at 1/3 of raw, so downstream EV/risk-sizing code can
                # de-rate even though XGBoost is happy to spit out a
                # confident-looking probability on a zero-padded vector.
                if feature_fetch_degraded:
                    coverage = max(0.0, 1.0 - len(missing_features) / 3.0)
                    # Floor at 1/3 so a single API blip doesn't zero the
                    # signal; a complete external-feature outage takes
                    # the model down to roughly the base rate.
                    confidence = min(confidence, max(coverage, 1.0 / 3.0) * confidence)

                degraded_reasons = []
                if self._model_uses_synthetic_warm_start:
                    degraded_reasons.append("synthetic_warm_start")
                if feature_fetch_degraded:
                    degraded_reasons.append(
                        "missing_features:" + ",".join(missing_features)
                    )

                result = {
                    "signal": confidence if regime == "bullish"
                              else -confidence if regime == "crash"
                              else 0.0,
                    "regime": regime,
                    "confidence": round(confidence, 4),
                    "raw_confidence": round(raw_confidence, 4),
                    "model": "xgboost",
                    "training_source": self._model_training_source,
                    "observed_training_rows": int(self._model_observed_rows),
                    "synthetic_warm_start": bool(self._model_uses_synthetic_warm_start),
                    "authoritative": (
                        not bool(self._model_uses_synthetic_warm_start)
                        and not feature_fetch_degraded
                    ),
                    "degraded": bool(degraded_reasons),
                    "degraded_reason": ";".join(degraded_reasons),
                    "missing_features": list(missing_features),
                    "probabilities": {
                        "crash": round(float(proba[0]), 4),
                        "neutral": round(float(proba[1]), 4),
                        "bullish": round(float(proba[2]), 4),
                    },
                    "components": base.get("components", {}),
                    "active_inputs": base.get("active_inputs", []),
                    "active_input_count": base.get(
                        "active_input_count",
                        len(base.get("active_inputs", []) or []),
                    ),
                }

                self.prediction_cache[coin] = {"data": result, "ts": now}

                logger.info(
                    "XGBoost Forecaster %s -> %s (conf=%.1f%%, signal=%.3f, source=%s)",
                    coin, regime, confidence * 100, result["signal"], self._model_training_source,
                )

                # Store prediction for future training
                self._store_prediction(coin, features, regime, confidence)

                return result

            except Exception as e:
                logger.debug("XGBoost prediction failed, using fallback: %s", e)

        # Fallback
        base["model"] = "weighted_signal"
        base.setdefault("training_source", "fallback")
        base.setdefault("observed_training_rows", int(self._model_observed_rows))
        # ★ AUDIT FIX: propagate feature-fetch degradation to the
        # fallback path too.  The weighted-signal forecaster uses some
        # of the same components and the consumer should see a single,
        # consistent degraded flag regardless of which model branch ran.
        if feature_fetch_degraded:
            base["degraded"] = True
            existing_reason = base.get("degraded_reason", "") or ""
            extra = "missing_features:" + ",".join(missing_features)
            base["degraded_reason"] = (
                ";".join([existing_reason, extra]) if existing_reason else extra
            )
            base["missing_features"] = list(missing_features)
            # Cap confidence proportionally (same 1/3 floor as the XGB path).
            try:
                conf = float(base.get("confidence", 0.0) or 0.0)
                coverage = max(0.0, 1.0 - len(missing_features) / 3.0)
                base["confidence"] = round(
                    conf * max(coverage, 1.0 / 3.0), 4,
                )
                # Recompute signal magnitude with the capped confidence
                # so it stays consistent with the published confidence.
                regime = base.get("regime", "neutral")
                base["signal"] = (
                    base["confidence"] if regime == "bullish"
                    else -base["confidence"] if regime == "crash"
                    else 0.0
                )
            except (TypeError, ValueError):
                pass
        base.setdefault("synthetic_warm_start", False)
        self.prediction_cache[coin] = {"data": base, "ts": now}

        # Still store for training (using the fallback's regime label)
        self._store_prediction(coin, features, base["regime"], base.get("confidence", 0))

        return base

    # ─── Feature Extraction ─────────────────────────────────────────

    def _extract_features(self, coin: str, base_prediction: Dict) -> Tuple[Dict, List[str]]:
        """
        Extract the full 8-feature vector from current market state.
        Reuses the base forecaster's components + adds volatility & basis.
        Enhanced with cross-exchange validation from Crypto.com.

        ★ AUDIT FIX: previously the three fetchers ``_get_funding_rate``,
        ``_get_5m_volatility``, and ``_get_basis_spread`` silently
        returned 0.0 on any exception (network timeout, HTTP 429, parse
        error).  The XGBoost model trained on real values, so at inference
        a silent 0.0 substitution produced a confident-looking prediction
        on a corrupted feature vector.  ~1-5% of predictions were
        affected during transient API issues and the consumer had no way
        to know.

        Now the fetchers return ``Optional[float]`` (``None`` on failure)
        and this method tracks which features were unavailable.  The
        caller in ``predict_regime`` uses the missing-feature list to:
          1. substitute 0.0 only for the model input (preserve model
             contract) so XGBoost still produces *some* prediction;
          2. flag the prediction as ``degraded=True`` so downstream
             consumers can choose to skip it; and
          3. cap confidence so a single API outage doesn't crown a
             corrupt prediction the regime-of-the-cycle.

        Returns a tuple ``(features, missing_features)``.
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

        missing_features: List[str] = []

        # Funding rate (raw, not slope)
        funding_rate = self._get_funding_rate(coin)
        if funding_rate is None:
            missing_features.append("funding_rate")
            features["funding_rate"] = 0.0
        else:
            features["funding_rate"] = funding_rate

        # 5-minute volatility (from HL candle API)
        volatility_5m = self._get_5m_volatility(coin)
        if volatility_5m is None:
            missing_features.append("volatility_5m")
            features["volatility_5m"] = 0.0
        else:
            features["volatility_5m"] = volatility_5m

        # CEX-DEX basis spread (HL vs Binance)
        basis_spread = self._get_basis_spread(coin)
        if basis_spread is None:
            missing_features.append("basis_spread")
            features["basis_spread"] = 0.0
        else:
            features["basis_spread"] = basis_spread

        # Cross-exchange volatility from Crypto.com (validation signal).
        # If HL volatility failed but CDC provides a real value, count
        # the feature as recovered (remove from missing list).
        if HAS_CRYPTOCOM:
            try:
                cdc_vol = _cryptocom.get_5m_volatility(coin)
                if cdc_vol > 0:
                    existing_vol = features.get("volatility_5m", 0)
                    if existing_vol > 0:
                        features["volatility_5m"] = (existing_vol + cdc_vol) / 2
                    else:
                        features["volatility_5m"] = cdc_vol
                    # Recovered via CDC — drop from missing if listed.
                    if "volatility_5m" in missing_features:
                        missing_features.remove("volatility_5m")
            except Exception:
                pass

        return features, missing_features

    def _get_funding_rate(self, coin: str) -> Optional[float]:
        """Get current funding rate from Hyperliquid.

        Returns ``None`` when the API call or parse fails, so the caller
        can mark the prediction degraded rather than substitute a silent
        0.0 that looks like a real measurement.
        """
        try:
            resp = get_manager().post(
                {"type": "metaAndAssetCtxs"},
                priority=Priority.NORMAL,
                timeout=5,
            )
            if resp:
                data = resp
                if len(data) >= 2:
                    meta, asset_ctxs = data[0], data[1]
                    for i, asset in enumerate(meta.get("universe", [])):
                        if asset.get("name", "").upper() == coin.upper() and i < len(asset_ctxs):
                            return float(asset_ctxs[i].get("funding", 0))
        except Exception as exc:
            logger.debug("_get_funding_rate(%s) failed: %s", coin, exc)
            return None
        # Asset not found in universe -- treat as missing, not 0.0.
        return None

    def _get_5m_volatility(self, coin: str) -> Optional[float]:
        """
        Compute recent 5-min return volatility from Hyperliquid candle data.
        Returns normalized value in [0, 1] range, or ``None`` when the
        underlying API or parse fails.
        """
        try:
            now_ms = int(time.time() * 1000)
            candles = get_manager().post(
                {
                    "type": "candleSnapshot",
                    "req": {
                        "coin": coin,
                        "interval": "5m",
                        "startTime": now_ms - 3_600_000,  # last hour
                        "endTime": now_ms,
                    },
                },
                priority=Priority.NORMAL,
                timeout=5,
            )
            if isinstance(candles, list):
                if len(candles) >= 5:
                    closes = [float(c["c"]) for c in candles[-12:]]
                    if len(closes) >= 2:
                        returns = np.diff(np.log(closes))
                        vol = float(np.std(returns))
                        # Normalize: typical 5-min vol ~0.001-0.01
                        return min(vol * 100, 1.0)
        except Exception as exc:
            logger.debug("_get_5m_volatility(%s) failed: %s", coin, exc)
            return None
        return None

    def _get_basis_spread(self, coin: str) -> Optional[float]:
        """
        CEX-DEX basis: Hyperliquid funding minus Binance funding, enhanced with Crypto.com data.
        Positive = HL funding higher (shorts pay more on HL vs Binance).
        Multi-exchange basis uses weighted average if crypto.com data is available.

        Returns ``None`` when the dependent calls fail so the caller can
        flag the prediction degraded.
        """
        try:
            import requests as req
            # Hyperliquid funding (reuse from fallback's internal history)
            hl_funding = 0.0
            hist = getattr(self.fallback, '_funding_history', {})
            if coin in hist and hist[coin]:
                hl_funding = hist[coin][-1]
            else:
                hl_funding_fetched = self._get_funding_rate(coin)
                if hl_funding_fetched is None:
                    # If HL funding itself is unavailable we can't compute
                    # the basis at all -- mark missing.
                    return None
                hl_funding = hl_funding_fetched

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
                                cdc_imbalance = 0.0
                                try:
                                    cdc_imbalance = _cryptocom.get_orderbook_imbalance(coin)
                                except Exception:
                                    pass

                                if cdc_imbalance != 0:
                                    basis = basis * 0.7 + cdc_imbalance * 0.0003 * 0.3
                    except Exception:
                        pass

                return max(min(basis * 10_000, 1.0), -1.0)
        except Exception as exc:
            logger.debug("_get_basis_spread(%s) failed: %s", coin, exc)
            return None
        # Binance response not OK -- treat as missing.
        return None

    # ─── Model Training (DB-backed walk-forward) ────────────────────

    def train(self) -> Optional[Dict]:
        """
        Train XGBoost on DB regime_history + any in-memory accumulation.
        Uses walk-forward: always trains on latest 90 days of data.
        """
        if not HAS_XGBOOST:
            logger.warning("XGBoost not installed -- cannot train")
            return None

        # Promote any matured predictions to observed labels first so the
        # training set actually grows over time (otherwise the dataset is
        # stuck at 0 observed rows and training falls back to synthetic).
        try:
            import config as _cfg
            if getattr(_cfg, "XGBOOST_LABELER_ENABLED", True):
                self.label_predictions_with_forward_returns()
        except Exception as exc:
            logger.debug("Pre-train labeling skipped: %s", exc)

        X, y = self._get_training_data()

        if len(y) < self.min_samples:
            logger.info(
                "Insufficient data for training: %d/%d samples",
                len(y), self.min_samples,
            )
            return None

        logger.info("Training XGBoost regime forecaster on %d samples...", len(y))

        model_kwargs = dict(
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
        self.model = xgb.XGBClassifier(**model_kwargs)

        # Time-series walk-forward validation (if enough chronologically-labeled data)
        cv_mean = 0.0
        cv_failed = False
        try:
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import TimeSeriesSplit

            n_splits = min(5, max(2, len(X) // 50))
            tscv = TimeSeriesSplit(n_splits=n_splits)
            fold_scores = []

            for train_idx, test_idx in tscv.split(X):
                if len(train_idx) < 20 or len(test_idx) < 5:
                    continue
                fold_model = xgb.XGBClassifier(**model_kwargs)
                fold_model.fit(X[train_idx], y[train_idx])
                preds = fold_model.predict(X[test_idx])
                fold_scores.append(float(accuracy_score(y[test_idx], preds)))

            if fold_scores:
                cv_mean = float(np.mean(fold_scores))
                logger.info(
                    "XGBoost walk-forward accuracy: %.3f (+/- %.3f, folds=%d)",
                    cv_mean,
                    float(np.std(fold_scores)),
                    len(fold_scores),
                )
            else:
                cv_failed = True
                logger.warning("XGBoost walk-forward: no folds completed")
        except Exception as exc:
            # ★ M9 FIX: was logger.debug — CV failure should be loud, not silent
            cv_failed = True
            logger.warning("XGBoost walk-forward validation FAILED: %s", exc)

        # ★ M9 FIX: refuse to save a model with no validation evidence
        if cv_failed:
            logger.error(
                "XGBoost training complete but CV validation failed -- "
                "NOT saving model to avoid deploying unvalidated predictions"
            )
            self.model = None
            return None

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
            "training_source": self._model_training_source,
            "observed_training_rows": int(self._model_observed_rows),
            "synthetic_warm_start": bool(self._model_uses_synthetic_warm_start),
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
        observed_count = 0

        try:
            from src.data import database as db
            from src.data.database import get_connection
            with get_connection(for_read=True) as conn:
                cutoff_sql = (
                    "now() - INTERVAL '90 days'"
                    if db.get_backend_name() == "postgres"
                    else "datetime('now', '-90 days')"
                )
                rows = conn.execute(f"""
                    SELECT funding_rate, funding_slope, orderbook_imbalance,
                           volatility_5m, basis_spread, options_flow_conviction,
                           regime_label
                    FROM regime_history
                    WHERE timestamp > {cutoff_sql}
                      AND regime_label IN (0, 1, 2)
                      AND label_source = 'observed'
                    ORDER BY timestamp ASC
                """).fetchall()

            if rows:
                # ★ H9 FIX: defensive row access — works whether connection
                # returns sqlite3.Row, dict, or tuple cursor
                def _row_get(r, name: str, idx: int):
                    try:
                        return r[name]
                    except (TypeError, IndexError, KeyError):
                        try:
                            return r[idx]
                        except Exception:
                            return 0

                X_rows = np.array(
                    [
                        [float(_row_get(r, name, i)) for i, name in enumerate(FEATURE_NAMES)]
                        for r in rows
                    ],
                    dtype=np.float32,
                )
                y_rows = np.array(
                    [int(_row_get(r, "regime_label", len(FEATURE_NAMES))) for r in rows],
                    dtype=np.int32,
                )
                observed_count = len(y_rows)
                self._model_observed_rows = int(observed_count)
                logger.info(
                    "Loaded %d observed-label rows from regime_history for training",
                    len(y_rows),
                )

                if len(y_rows) >= self.min_samples:
                    self._model_training_source = "observed"
                    self._model_uses_synthetic_warm_start = False
                    return X_rows, y_rows

        except Exception as exc:
            logger.debug("Could not load training data from DB: %s", exc)

        # Fallback: synthetic warm-start with realistic distributions
        observed_count = len(y_rows) if isinstance(y_rows, np.ndarray) else observed_count
        self._model_observed_rows = int(observed_count)
        self._model_training_source = "mixed_synthetic" if observed_count > 0 else "synthetic"
        self._model_uses_synthetic_warm_start = True
        logger.info(
            "Using synthetic warm-start (%d DB rows, need %d)",
            observed_count,
            self.min_samples,
        )
        n = 2000
        rng = np.random.RandomState(42)
        X_synth = np.column_stack([
            rng.normal(0, 0.0005, n),   # funding_rate
            rng.normal(0, 0.3, n),      # funding_slope
            rng.normal(0, 0.2, n),      # orderbook_imbalance
            rng.exponential(0.03, n),   # volatility_5m (always positive)
            rng.normal(0, 0.1, n),      # basis_spread
            rng.normal(0, 0.2, n),      # options_flow_conviction
        ]).astype(np.float32)

        # Labels: derived from weighted composite to be internally consistent
        # Indices: 1=funding_slope, 2=orderbook_imbalance, 5=options_flow_conviction
        composite = (X_synth[:, 1] * 0.30 + X_synth[:, 2] * 0.25 +
                     X_synth[:, 5] * 0.20 + rng.normal(0, 0.15, n))
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
                        regime_label, confidence, predicted_regime, label_source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                        None,  # Unknown at prediction time; do not self-label training data.
                        confidence,
                        regime,
                        "predicted",
                    ),
                )
        except Exception as exc:
            logger.debug("Failed to store prediction: %s", exc)

    # ─── Forward-return labeler ─────────────────────────────────────

    def label_predictions_with_forward_returns(
        self,
        forward_minutes: Optional[int] = None,
        crash_pct: Optional[float] = None,
        bullish_pct: Optional[float] = None,
        batch_size: Optional[int] = None,
        min_age_minutes: Optional[int] = None,
    ) -> Dict[str, int]:
        """Promote past predictions to observed training labels.

        For each row in regime_history older than min_age_minutes that still
        has label_source='predicted' and regime_label IS NULL, fetch the
        forward 1m candles for the row's coin from the prediction timestamp
        out to forward_minutes, compute (last_close - entry_close)/entry_close,
        and write back regime_label + label_source='observed'.

        Returns:
            {"scanned": int, "labeled": int, "no_data": int, "errors": int}
        """
        import config as _cfg
        from datetime import datetime, timezone, timedelta

        forward_minutes = int(
            forward_minutes
            if forward_minutes is not None
            else getattr(_cfg, "XGBOOST_LABELER_FORWARD_MINUTES", 60)
        )
        crash_pct = float(
            crash_pct
            if crash_pct is not None
            else getattr(_cfg, "XGBOOST_LABELER_CRASH_PCT", -0.015)
        )
        bullish_pct = float(
            bullish_pct
            if bullish_pct is not None
            else getattr(_cfg, "XGBOOST_LABELER_BULLISH_PCT", 0.015)
        )
        batch_size = int(
            batch_size
            if batch_size is not None
            else getattr(_cfg, "XGBOOST_LABELER_BATCH_SIZE", 200)
        )
        min_age_minutes = int(
            min_age_minutes
            if min_age_minutes is not None
            else getattr(_cfg, "XGBOOST_LABELER_MIN_AGE_MINUTES", forward_minutes + 5)
        )

        stats = {"scanned": 0, "labeled": 0, "no_data": 0, "errors": 0}
        try:
            from src.data import database as db
            from src.data.database import get_connection
            from src.data import hyperliquid_client as hl
        except Exception as exc:
            logger.warning("Labeler aborted (imports): %s", exc)
            stats["errors"] += 1
            return stats

        backend = db.get_backend_name()
        cutoff_sql = (
            f"now() - INTERVAL '{min_age_minutes} minutes'"
            if backend == "postgres"
            else f"datetime('now', '-{min_age_minutes} minutes')"
        )
        try:
            with get_connection(for_read=True) as conn:
                rows = conn.execute(
                    f"""
                    SELECT id, timestamp, coin
                    FROM regime_history
                    WHERE label_source = 'predicted'
                      AND regime_label IS NULL
                      AND timestamp <= {cutoff_sql}
                    ORDER BY timestamp ASC
                    LIMIT {int(batch_size)}
                    """
                ).fetchall()
        except Exception as exc:
            logger.warning("Labeler query failed: %s", exc)
            stats["errors"] += 1
            return stats

        if not rows:
            return stats

        # Group by coin so we make at most one candle fetch per coin.
        by_coin: Dict[str, list] = {}
        for r in rows:
            try:
                row_id = int(r["id"]) if hasattr(r, "keys") else int(r[0])
                ts_raw = r["timestamp"] if hasattr(r, "keys") else r[1]
                coin = (r["coin"] if hasattr(r, "keys") else r[2]) or "BTC"
            except Exception:
                stats["errors"] += 1
                continue
            stats["scanned"] += 1
            ts_dt: Optional[datetime] = None
            try:
                if isinstance(ts_raw, datetime):
                    ts_dt = ts_raw if ts_raw.tzinfo else ts_raw.replace(tzinfo=timezone.utc)
                else:
                    ts_dt = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                    if ts_dt.tzinfo is None:
                        ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            except Exception:
                stats["errors"] += 1
                continue
            by_coin.setdefault(coin.upper(), []).append((row_id, ts_dt))

        # Fetch one window of candles per coin covering the earliest pred
        # through (latest pred + forward window).
        updates: list[tuple[int, int]] = []  # (row_id, regime_label)
        for coin, items in by_coin.items():
            try:
                earliest_ts = min(t for _, t in items)
                latest_ts = max(t for _, t in items)
                start_ms = int(earliest_ts.timestamp() * 1000) - 60_000
                end_ms = int(
                    (latest_ts + timedelta(minutes=forward_minutes)).timestamp() * 1000
                ) + 60_000
                candles = hl.get_candles(
                    coin, interval="1m", start_time=start_ms, end_time=end_ms
                ) or []
            except Exception as exc:
                logger.debug("Labeler candle fetch failed for %s: %s", coin, exc)
                stats["errors"] += len(items)
                continue
            if not candles:
                stats["no_data"] += len(items)
                continue

            # Hyperliquid candles use keys: t (open ms), c (close), o, h, l ...
            try:
                candle_list = sorted(
                    [(int(c.get("t", 0)), float(c.get("c", 0))) for c in candles],
                    key=lambda x: x[0],
                )
            except Exception:
                stats["errors"] += len(items)
                continue

            for row_id, ts_dt in items:
                pred_ms = int(ts_dt.timestamp() * 1000)
                target_ms = pred_ms + forward_minutes * 60_000
                entry = self._closest_close(candle_list, pred_ms)
                exit_ = self._closest_close(candle_list, target_ms)
                if entry is None or exit_ is None or entry <= 0:
                    stats["no_data"] += 1
                    continue
                ret = (exit_ - entry) / entry
                if ret <= crash_pct:
                    label = REGIME_LABELS["crash"]
                elif ret >= bullish_pct:
                    label = REGIME_LABELS["bullish"]
                else:
                    label = REGIME_LABELS["neutral"]
                updates.append((row_id, label))

        if updates:
            try:
                with get_connection() as conn:
                    if backend == "postgres":
                        conn.executemany(
                            "UPDATE regime_history SET regime_label = %s, "
                            "label_source = 'observed' WHERE id = %s",
                            [(label, rid) for rid, label in updates],
                        )
                    else:
                        conn.executemany(
                            "UPDATE regime_history SET regime_label = ?, "
                            "label_source = 'observed' WHERE id = ?",
                            [(label, rid) for rid, label in updates],
                        )
                stats["labeled"] = len(updates)
            except Exception as exc:
                logger.warning("Labeler write-back failed: %s", exc)
                stats["errors"] += len(updates)
        if stats["scanned"]:
            logger.info(
                "XGBoost labeler: scanned=%d labeled=%d no_data=%d errors=%d "
                "(forward=%dm, crash<=%.2f%%, bullish>=%.2f%%)",
                stats["scanned"],
                stats["labeled"],
                stats["no_data"],
                stats["errors"],
                forward_minutes,
                crash_pct * 100,
                bullish_pct * 100,
            )
        return stats

    @staticmethod
    def _closest_close(
        sorted_candles: list, target_ms: int
    ) -> Optional[float]:
        """Return the close price of the candle whose open is closest to and
        not after target_ms. Returns None if there's no candle at or before."""
        if not sorted_candles:
            return None
        best = None
        for ts, close in sorted_candles:
            if ts > target_ms:
                break
            best = close
        return best

    # ─── DB Schema ──────────────────────────────────────────────────

    def _ensure_regime_history_table(self):
        """Create regime_history table if it doesn't exist."""
        try:
            from src.data import database as db
            from src.data.database import get_connection
            backend = db.get_backend_name()
            with get_connection() as conn:
                if backend == "postgres":
                    conn.executescript(_POSTGRES_REGIME_HISTORY_DDL)
                    return

                conn.executescript(_SQLITE_REGIME_HISTORY_DDL)
                # Backward-compatible migrations for existing SQLite deployments.
                cols = conn.execute("PRAGMA table_info(regime_history)").fetchall()
                col_names = {str(row["name"]) for row in cols} if cols else set()
                if "label_source" not in col_names:
                    conn.execute(
                        "ALTER TABLE regime_history ADD COLUMN label_source TEXT DEFAULT 'predicted'"
                    )
                    conn.execute(
                        "UPDATE regime_history SET label_source = 'predicted' "
                        "WHERE label_source IS NULL OR label_source = ''"
                    )
        except Exception as exc:
            logger.debug("Could not create regime_history table: %s", exc)

    # ─── Model Persistence ──────────────────────────────────────────

    def _model_metadata_path(self) -> str:
        return f"{self.model_path}.meta.json"

    def _save_model(self):
        """Save trained model to disk."""
        if self.model is None:
            return
        if self._model_uses_synthetic_warm_start:
            logger.info(
                "XGBoost warm-start model not saved (source=%s, observed_rows=%d); "
                "waiting for observed regime_history labels before persisting.",
                self._model_training_source,
                self._model_observed_rows,
            )
            return
        try:
            self.model.save_model(self.model_path)
            metadata = {
                "training_source": self._model_training_source,
                "observed_training_rows": int(self._model_observed_rows),
                "synthetic_warm_start": bool(self._model_uses_synthetic_warm_start),
                "saved_at_epoch_s": time.time(),
            }
            with open(self._model_metadata_path(), "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, sort_keys=True)
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
            metadata = {}
            meta_path = self._model_metadata_path()
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as fh:
                    metadata = json.load(fh)
            else:
                metadata = {
                    "training_source": "unknown_legacy",
                    "observed_training_rows": 0,
                    "synthetic_warm_start": True,
                }
                logger.warning(
                    "XGBoost model %s has no metadata sidecar; treating as "
                    "non-authoritative until retrained on observed labels.",
                    self.model_path,
                )
            self._model_training_source = str(metadata.get("training_source") or "unknown")
            self._model_observed_rows = int(metadata.get("observed_training_rows") or 0)
            raw_synthetic = metadata.get(
                "synthetic_warm_start",
                self._model_training_source != "observed",
            )
            if isinstance(raw_synthetic, str):
                self._model_uses_synthetic_warm_start = raw_synthetic.strip().lower() in {
                    "1", "true", "yes", "on",
                }
            else:
                self._model_uses_synthetic_warm_start = bool(raw_synthetic)
            logger.info("XGBoost model loaded from %s", self.model_path)
        except Exception as e:
            logger.debug("Could not load XGBoost model: %s", e)
            self.model = None
            self._model_training_source = "untrained"
            self._model_observed_rows = 0
            self._model_uses_synthetic_warm_start = False

    # ─── Stats ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return forecaster statistics."""
        return {
            "model_loaded": self.model is not None,
            "model_type": "xgboost" if self.model else "fallback",
            "last_train_ts": self._last_train_ts,
            "has_xgboost": HAS_XGBOOST,
            "cache_size": len(self.prediction_cache),
            "training_source": self._model_training_source,
            "observed_training_rows": int(self._model_observed_rows),
            "synthetic_warm_start": bool(self._model_uses_synthetic_warm_start),
        }
