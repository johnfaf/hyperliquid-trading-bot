"""
Feature-store-backed alpha pipeline.

Phase B adds:
  - Walk-forward train/test on the Postgres feature store
  - Direction classifiers for 1h and 4h forward returns
  - Learned confidence calibration from out-of-fold predictions
  - Statistical significance checks before the model is allowed to trade

The pipeline is Postgres-only because it depends on the feature store.
When Postgres or the feature tables are unavailable it degrades gracefully.
"""
from __future__ import annotations

import json
import logging
import math
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss
    from sklearn.model_selection import TimeSeriesSplit

    HAS_ALPHA_ML = True
except ImportError:
    HAS_ALPHA_ML = False
    xgb = None
    IsotonicRegression = None
    accuracy_score = None
    balanced_accuracy_score = None
    brier_score_loss = None
    log_loss = None
    TimeSeriesSplit = None

from src.data import feature_store as fs

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

HORIZON_STEPS = {"1h": 1, "4h": 4}
DEFAULT_FEATURE_NAMES = list(fs.FEATURE_NAMES)


class _ConstantProbModel:
    """Fallback model used when a split contains only one class."""

    def __init__(self, prob_up: float):
        self.prob_up = float(np.clip(prob_up, 1e-6, 1.0 - 1e-6))

    def predict_proba(self, X) -> np.ndarray:
        rows = len(X)
        probs = np.zeros((rows, 2), dtype=float)
        probs[:, 1] = self.prob_up
        probs[:, 0] = 1.0 - self.prob_up
        return probs


@dataclass
class AlphaPrediction:
    coin: str
    timeframe: str
    horizon: str
    feature_timestamp_ms: int
    raw_probability_up: float
    calibrated_probability_up: float
    confidence: float
    predicted_side: str
    expected_return_bps: float
    significance_pvalue: float
    eligible: bool
    model_version: str

    def to_signal_dict(self) -> Dict:
        source = "ml_alpha"
        confidence = max(0.0, min(self.confidence, 0.99))
        return {
            "id": None,
            "name": f"{source}_{self.coin}_{self.horizon}_{self.predicted_side}",
            "strategy_type": f"{source}_{self.horizon}",
            "trader_address": source,
            "current_score": confidence,
            "confidence": confidence,
            "direction": self.predicted_side,
            "side": self.predicted_side,
            "source": source,
            "parameters": {
                "coins": [self.coin],
                "timeframe": self.timeframe,
                "horizon": self.horizon,
                "feature_timestamp_ms": self.feature_timestamp_ms,
                "raw_probability_up": round(self.raw_probability_up, 6),
                "calibrated_probability_up": round(self.calibrated_probability_up, 6),
            },
            "metrics": {
                "expected_return_bps": round(self.expected_return_bps, 4),
                "significance_pvalue": round(self.significance_pvalue, 6),
            },
            "metadata": {
                "model_version": self.model_version,
                "expected_return_bps": round(self.expected_return_bps, 4),
                "significance_pvalue": round(self.significance_pvalue, 6),
                "alpha_horizon": self.horizon,
                "confidence_source": "learned_calibration",
                "eligible": self.eligible,
            },
        }


class FeatureStoreAlphaPipeline:
    """Walk-forward XGBoost alpha model backed by the Postgres feature store."""

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}

        self.timeframe = cfg.get("timeframe", "1h")
        self.feature_names = list(cfg.get("feature_names", DEFAULT_FEATURE_NAMES))
        self.lookback_days = int(cfg.get("lookback_days", 120))
        self.min_samples = int(cfg.get("min_training_samples", 250))
        self.retrain_interval = int(cfg.get("retrain_interval", 21600))
        self.walk_forward_splits = int(cfg.get("walk_forward_splits", 5))
        self.label_min_abs_return = float(cfg.get("label_min_abs_return", 0.0005))
        self.signal_min_confidence = float(cfg.get("signal_min_confidence", 0.58))
        self.min_significant_trades = int(cfg.get("min_significant_trades", 60))
        self.min_significance_pvalue = float(cfg.get("min_significance_pvalue", 0.10))
        self.max_prediction_coins = int(cfg.get("max_prediction_coins", 12))
        self.cache_ttl = int(cfg.get("cache_ttl", 180))
        self.model_dir = Path(cfg.get("model_dir", "models/alpha_direction"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._ensemble_specs = cfg.get(
            "ensemble_specs",
            [
                {"n_estimators": 90, "max_depth": 3, "learning_rate": 0.05, "random_state": 41},
                {"n_estimators": 75, "max_depth": 4, "learning_rate": 0.06, "random_state": 43},
                {"n_estimators": 60, "max_depth": 2, "learning_rate": 0.08, "random_state": 47},
            ],
        )

        self.models: Dict[str, List] = {}
        self.calibrators: Dict[str, object] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.prediction_cache: Dict[Tuple[str, str], Dict] = {}
        self._last_train_ts = 0.0

        self._load_models()
        if HAS_ALPHA_ML and self._pg_available() and not self.models:
            try:
                self.train_if_due(force=True)
            except Exception as exc:
                logger.warning("Alpha pipeline initial training skipped: %s", exc)

    def _pg_available(self) -> bool:
        return bool(getattr(fs, "_pg_available", lambda: False)())

    def _pg_conn(self):
        return fs._pg_conn()

    def _artifact_path(self, horizon: str) -> Path:
        return self.model_dir / f"{self.timeframe}_{horizon}.pkl"

    def _latest_run_rows(self) -> List[Dict]:
        conn, ret = self._pg_conn()
        if conn is None:
            return []
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT horizon, trained_at, model_version, sample_count, oof_sample_count,
                       active_trade_count, accuracy, balanced_accuracy, brier_score,
                       log_loss, ece, win_rate, avg_signed_return_bps,
                       significance_pvalue, bootstrap_ci_low_bps, bootstrap_ci_high_bps,
                       eligible, metrics
                FROM alpha_model_runs
                WHERE timeframe=%s
                ORDER BY trained_at DESC
                LIMIT 8
                """,
                (self.timeframe,),
            )
            return [dict(row) for row in cur.fetchall()]
        except Exception:
            return []
        finally:
            ret(conn)

    def _latest_prediction_rows(self, limit: int = 12) -> List[Dict]:
        conn, ret = self._pg_conn()
        if conn is None:
            return []
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT coin, horizon, feature_timestamp_ms, predicted_side,
                       confidence, expected_return_bps, significance_pvalue,
                       eligible, created_at
                FROM alpha_predictions
                WHERE timeframe=%s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (self.timeframe, limit),
            )
            return [dict(row) for row in cur.fetchall()]
        except Exception:
            return []
        finally:
            ret(conn)

    def _record_training_run(self, horizon: str, model_version: str, metrics: Dict) -> None:
        conn, ret = self._pg_conn()
        if conn is None:
            return
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO alpha_model_runs (
                    timeframe, horizon, model_version, sample_count, oof_sample_count,
                    active_trade_count, accuracy, balanced_accuracy, brier_score,
                    log_loss, ece, win_rate, avg_signed_return_bps,
                    significance_pvalue, bootstrap_ci_low_bps, bootstrap_ci_high_bps,
                    eligible, feature_names, metrics
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                )
                """,
                (
                    self.timeframe,
                    horizon,
                    model_version,
                    int(metrics.get("sample_count", 0)),
                    int(metrics.get("oof_sample_count", 0)),
                    int(metrics.get("active_trade_count", 0)),
                    float(metrics.get("accuracy", 0.0)),
                    float(metrics.get("balanced_accuracy", 0.0)),
                    float(metrics.get("brier_score", 0.0)),
                    float(metrics.get("log_loss", 0.0)),
                    float(metrics.get("ece", 0.0)),
                    float(metrics.get("win_rate", 0.0)),
                    float(metrics.get("avg_signed_return_bps", 0.0)),
                    float(metrics.get("significance_pvalue", 1.0)),
                    float(metrics.get("bootstrap_ci_low_bps", 0.0)),
                    float(metrics.get("bootstrap_ci_high_bps", 0.0)),
                    bool(metrics.get("eligible", False)),
                    json.dumps(self.feature_names),
                    json.dumps(metrics),
                ),
            )
            conn.commit()
        except Exception as exc:
            logger.warning("Alpha training run record failed (%s): %s", horizon, exc)
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            ret(conn)
    def _record_prediction(self, prediction: AlphaPrediction) -> None:
        conn, ret = self._pg_conn()
        if conn is None:
            return
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO alpha_predictions (
                    coin, timeframe, horizon, feature_timestamp_ms, model_version,
                    raw_probability_up, calibrated_probability_up, predicted_side,
                    confidence, expected_return_bps, significance_pvalue, eligible, metadata
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                """,
                (
                    prediction.coin,
                    prediction.timeframe,
                    prediction.horizon,
                    int(prediction.feature_timestamp_ms),
                    prediction.model_version,
                    float(prediction.raw_probability_up),
                    float(prediction.calibrated_probability_up),
                    prediction.predicted_side,
                    float(prediction.confidence),
                    float(prediction.expected_return_bps),
                    float(prediction.significance_pvalue),
                    bool(prediction.eligible),
                    json.dumps(
                        {
                            "confidence_source": "learned_calibration",
                            "eligible": prediction.eligible,
                        }
                    ),
                ),
            )
            conn.commit()
        except Exception as exc:
            logger.debug("Alpha prediction record failed (%s/%s): %s", prediction.coin, prediction.horizon, exc)
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            ret(conn)

    @staticmethod
    def build_training_frame(
        feature_rows: Sequence[Dict],
        candle_rows: Sequence[Dict],
        feature_names: Sequence[str],
        label_min_abs_return: float = 0.0,
    ) -> pd.DataFrame:
        """Build a supervised training frame from EAV features plus candles."""
        if not feature_rows or not candle_rows:
            return pd.DataFrame()

        feat_df = pd.DataFrame(feature_rows)
        candle_df = pd.DataFrame(candle_rows)
        if feat_df.empty or candle_df.empty:
            return pd.DataFrame()

        pivot = (
            feat_df.pivot_table(
                index=["coin", "timestamp_ms"],
                columns="feature_name",
                values="value",
                aggfunc="first",
            )
            .reset_index()
        )
        for name in feature_names:
            if name not in pivot.columns:
                pivot[name] = np.nan
        pivot = pivot[["coin", "timestamp_ms", *feature_names]]

        candles = candle_df[["coin", "timestamp_ms", "close"]].copy()
        candles = candles.sort_values(["coin", "timestamp_ms"]).reset_index(drop=True)

        for horizon, steps in HORIZON_STEPS.items():
            future_close = candles.groupby("coin")["close"].shift(-steps)
            forward_return = (future_close / candles["close"]) - 1.0
            label = (forward_return > 0).astype(float)
            if label_min_abs_return > 0:
                mask = forward_return.abs() < label_min_abs_return
                label = label.mask(mask)
            candles[f"forward_return_{horizon}"] = forward_return
            candles[f"label_{horizon}"] = label

        frame = pivot.merge(candles, on=["coin", "timestamp_ms"], how="inner")
        return frame.sort_values(["timestamp_ms", "coin"]).reset_index(drop=True)

    def _load_training_frame(self) -> pd.DataFrame:
        if not self._pg_available():
            return pd.DataFrame()

        since_ms = int((time.time() - (self.lookback_days * 86400)) * 1000)
        conn, ret = self._pg_conn()
        if conn is None:
            return pd.DataFrame()

        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT coin, timestamp_ms, feature_name, value
                FROM features
                WHERE timeframe=%s AND timestamp_ms >= %s
                ORDER BY timestamp_ms ASC, coin ASC
                """,
                (self.timeframe, since_ms),
            )
            feature_rows = cur.fetchall()

            cur.execute(
                """
                SELECT coin, timestamp_ms, close
                FROM candles
                WHERE timeframe=%s AND timestamp_ms >= %s
                ORDER BY timestamp_ms ASC, coin ASC
                """,
                (self.timeframe, since_ms),
            )
            candle_rows = cur.fetchall()
            return self.build_training_frame(
                feature_rows=feature_rows,
                candle_rows=candle_rows,
                feature_names=self.feature_names,
                label_min_abs_return=self.label_min_abs_return,
            )
        except Exception as exc:
            logger.warning("Alpha training frame load failed: %s", exc)
            return pd.DataFrame()
        finally:
            ret(conn)

    def _fit_member(self, X_train: np.ndarray, y_train: np.ndarray, spec: Dict):
        classes = np.unique(y_train)
        if len(classes) < 2:
            return _ConstantProbModel(float(np.mean(y_train)) if len(y_train) else 0.5)

        model = xgb.XGBClassifier(
            n_estimators=int(spec.get("n_estimators", 80)),
            max_depth=int(spec.get("max_depth", 3)),
            learning_rate=float(spec.get("learning_rate", 0.05)),
            subsample=float(spec.get("subsample", 0.85)),
            colsample_bytree=float(spec.get("colsample_bytree", 0.85)),
            min_child_weight=float(spec.get("min_child_weight", 2.0)),
            reg_lambda=float(spec.get("reg_lambda", 1.0)),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=int(spec.get("random_state", 42)),
            n_jobs=1,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        return model

    def _ensemble_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, List]:
        members = []
        probs = []
        for spec in self._ensemble_specs:
            member = self._fit_member(X_train, y_train, spec)
            members.append(member)
            probs.append(np.clip(member.predict_proba(X_test)[:, 1], 1e-6, 1.0 - 1e-6))
        matrix = np.column_stack(probs)
        return matrix.mean(axis=1), members

    @staticmethod
    def _ece(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
        if len(y_true) == 0:
            return 0.0
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        for idx in range(bins):
            left, right = edges[idx], edges[idx + 1]
            if idx == bins - 1:
                mask = (probabilities >= left) & (probabilities <= right)
            else:
                mask = (probabilities >= left) & (probabilities < right)
            if not np.any(mask):
                continue
            bin_probs = probabilities[mask]
            bin_true = y_true[mask]
            ece += (len(bin_true) / len(y_true)) * abs(bin_probs.mean() - bin_true.mean())
        return float(ece)

    @staticmethod
    def _exact_binomial_pvalue(successes: int, total: int, p0: float = 0.5) -> float:
        if total <= 0:
            return 1.0
        if total > 500:
            mean = total * p0
            variance = total * p0 * (1.0 - p0)
            if variance <= 0:
                return 1.0
            z = abs((successes - mean) / math.sqrt(variance))
            return float(math.erfc(z / math.sqrt(2.0)))

        probs = []
        for k in range(total + 1):
            probs.append(math.comb(total, k) * (p0 ** k) * ((1.0 - p0) ** (total - k)))
        observed = probs[successes]
        pvalue = sum(p for p in probs if p <= observed + 1e-12)
        return float(min(1.0, pvalue))

    @staticmethod
    def _bootstrap_mean_ci(values: np.ndarray, draws: int = 1000) -> Tuple[float, float]:
        if len(values) == 0:
            return 0.0, 0.0
        rng = np.random.default_rng(42)
        samples = rng.choice(values, size=(draws, len(values)), replace=True)
        means = samples.mean(axis=1)
        lower = float(np.percentile(means, 2.5))
        upper = float(np.percentile(means, 97.5))
        return lower, upper

    @staticmethod
    def _expectancy_curve(confidences: np.ndarray, signed_returns_bps: np.ndarray) -> List[Dict]:
        buckets = np.linspace(0.5, 1.0, 11)
        curve = []
        for idx in range(len(buckets) - 1):
            left = buckets[idx]
            right = buckets[idx + 1]
            if idx == len(buckets) - 2:
                mask = (confidences >= left) & (confidences <= right)
            else:
                mask = (confidences >= left) & (confidences < right)
            if not np.any(mask):
                curve.append({"min": round(left, 2), "max": round(right, 2), "mean_return_bps": 0.0, "count": 0})
                continue
            curve.append(
                {
                    "min": round(left, 2),
                    "max": round(right, 2),
                    "mean_return_bps": round(float(np.mean(signed_returns_bps[mask])), 4),
                    "count": int(mask.sum()),
                }
            )
        return curve
    def _average_feature_importance(self, members: Sequence[object]) -> Dict[str, float]:
        importances = []
        for member in members:
            raw = getattr(member, "feature_importances_", None)
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=float)
            if arr.size != len(self.feature_names):
                continue
            importances.append(arr)
        if not importances:
            return {}
        avg = np.mean(importances, axis=0)
        return {
            name: round(float(val), 6)
            for name, val in zip(self.feature_names, avg)
        }

    def _evaluate_horizon(self, frame: pd.DataFrame, horizon: str) -> Optional[Dict]:
        label_col = f"label_{horizon}"
        return_col = f"forward_return_{horizon}"
        required = ["coin", "timestamp_ms", return_col, label_col, *self.feature_names]
        if frame.empty or any(col not in frame.columns for col in required):
            return None

        data = frame[required].copy()
        data = data.dropna(subset=[label_col, return_col])
        if len(data) < self.min_samples:
            return None

        data = data.sort_values(["timestamp_ms", "coin"]).reset_index(drop=True)
        unique_ts = np.array(sorted(data["timestamp_ms"].unique()))
        if len(unique_ts) < max(12, self.walk_forward_splits + 2):
            return None

        n_splits = min(self.walk_forward_splits, max(2, len(unique_ts) // 8))
        splitter = TimeSeriesSplit(n_splits=n_splits)

        oof_chunks = []
        for train_idx, test_idx in splitter.split(unique_ts):
            train_ts = unique_ts[train_idx]
            test_ts = unique_ts[test_idx]

            train_mask = data["timestamp_ms"].isin(train_ts)
            test_mask = data["timestamp_ms"].isin(test_ts)
            train_df = data.loc[train_mask]
            test_df = data.loc[test_mask]
            if len(train_df) < max(50, self.min_samples // 2) or len(test_df) < 20:
                continue

            X_train = train_df[self.feature_names].fillna(0.0).astype(np.float32).to_numpy()
            y_train = train_df[label_col].astype(np.int32).to_numpy()
            X_test = test_df[self.feature_names].fillna(0.0).astype(np.float32).to_numpy()

            raw_probs, _ = self._ensemble_predict(X_train, y_train, X_test)
            chunk = test_df[["coin", "timestamp_ms", return_col, label_col]].copy()
            chunk["raw_probability_up"] = raw_probs
            oof_chunks.append(chunk)

        if not oof_chunks:
            return None

        oof = pd.concat(oof_chunks, ignore_index=True)
        y_true = oof[label_col].astype(np.int32).to_numpy()
        raw_probs = oof["raw_probability_up"].clip(1e-6, 1.0 - 1e-6).to_numpy()

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_probs, y_true)
        calibrated_probs = np.clip(calibrator.transform(raw_probs), 1e-6, 1.0 - 1e-6)

        predicted_up = calibrated_probs >= 0.5
        direction_confidence = np.maximum(calibrated_probs, 1.0 - calibrated_probs)
        forward_returns = oof[return_col].astype(float).to_numpy()
        signed_returns_bps = np.where(predicted_up, forward_returns, -forward_returns) * 10_000.0

        accuracy = float(accuracy_score(y_true, predicted_up))
        balanced_accuracy = float(balanced_accuracy_score(y_true, predicted_up))
        brier = float(brier_score_loss(y_true, calibrated_probs))
        loss = float(log_loss(y_true, calibrated_probs))
        ece = self._ece(y_true, calibrated_probs)

        active_mask = direction_confidence >= self.signal_min_confidence
        active_returns = signed_returns_bps[active_mask]
        active_count = int(active_mask.sum())
        successes = int(np.sum(active_returns > 0)) if active_count else 0
        pvalue = self._exact_binomial_pvalue(successes, active_count) if active_count else 1.0
        ci_low, ci_high = self._bootstrap_mean_ci(active_returns) if active_count else (0.0, 0.0)
        avg_active_return = float(np.mean(active_returns)) if active_count else 0.0
        win_rate = float(np.mean(active_returns > 0)) if active_count else 0.0
        eligible = (
            active_count >= self.min_significant_trades
            and avg_active_return > 0.0
            and pvalue <= self.min_significance_pvalue
        )

        final_X = data[self.feature_names].fillna(0.0).astype(np.float32).to_numpy()
        final_y = data[label_col].astype(np.int32).to_numpy()
        final_members = [self._fit_member(final_X, final_y, spec) for spec in self._ensemble_specs]

        metrics = {
            "timeframe": self.timeframe,
            "horizon": horizon,
            "sample_count": int(len(data)),
            "oof_sample_count": int(len(oof)),
            "active_trade_count": active_count,
            "accuracy": round(accuracy, 6),
            "balanced_accuracy": round(balanced_accuracy, 6),
            "brier_score": round(brier, 6),
            "log_loss": round(loss, 6),
            "ece": round(ece, 6),
            "win_rate": round(win_rate, 6),
            "avg_signed_return_bps": round(avg_active_return, 4),
            "significance_pvalue": round(float(pvalue), 8),
            "bootstrap_ci_low_bps": round(float(ci_low), 4),
            "bootstrap_ci_high_bps": round(float(ci_high), 4),
            "eligible": bool(eligible),
            "feature_importance": self._average_feature_importance(final_members),
            "expectancy_curve": self._expectancy_curve(direction_confidence, signed_returns_bps),
            "latest_feature_timestamp_ms": int(data["timestamp_ms"].max()),
            "signal_min_confidence": float(self.signal_min_confidence),
            "label_min_abs_return": float(self.label_min_abs_return),
        }
        return {
            "members": final_members,
            "calibrator": calibrator,
            "metrics": metrics,
        }

    def _save_artifacts(self, horizon: str) -> None:
        path = self._artifact_path(horizon)
        payload = {
            "members": self.models.get(horizon, []),
            "calibrator": self.calibrators.get(horizon),
            "metadata": self.model_metadata.get(horizon, {}),
        }
        # Atomic write: temp file + os.replace so a crash mid-write cannot
        # corrupt the model artifact that _load_models() reads at startup.
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("wb") as fh:
            pickle.dump(payload, fh)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(str(tmp_path), str(path))

    def _load_models(self) -> None:
        if not HAS_ALPHA_ML:
            return

        latest_ts = 0.0
        for horizon in HORIZON_STEPS:
            path = self._artifact_path(horizon)
            if not path.exists():
                continue
            try:
                with path.open("rb") as fh:
                    payload = pickle.load(fh)
                members = payload.get("members") or []
                calibrator = payload.get("calibrator")
                metadata = dict(payload.get("metadata") or {})
                if not members or calibrator is None:
                    continue
                self.models[horizon] = members
                self.calibrators[horizon] = calibrator
                self.model_metadata[horizon] = metadata
                latest_ts = max(latest_ts, float(metadata.get("trained_at_epoch_s", 0.0)))
            except Exception as exc:
                logger.warning("Alpha artifact load failed for %s: %s", horizon, exc)
        self._last_train_ts = latest_ts

    def train(self) -> Dict[str, Dict]:
        if not HAS_ALPHA_ML:
            logger.info("Alpha pipeline disabled - xgboost/scikit-learn not installed")
            return {}
        if not self._pg_available():
            logger.info("Alpha pipeline skipped - Postgres feature store unavailable")
            return {}

        frame = self._load_training_frame()
        if frame.empty:
            logger.info("Alpha pipeline skipped - no feature-store training frame available")
            return {}

        trained_at_epoch = time.time()
        trained_at = datetime.now(timezone.utc).isoformat()
        results: Dict[str, Dict] = {}

        for horizon in HORIZON_STEPS:
            evaluated = self._evaluate_horizon(frame, horizon)
            if not evaluated:
                continue

            model_version = f"alpha_{self.timeframe}_{horizon}_{int(trained_at_epoch)}"
            metrics = dict(evaluated["metrics"])
            metrics["model_version"] = model_version
            metrics["trained_at"] = trained_at
            metrics["trained_at_epoch_s"] = trained_at_epoch

            self.models[horizon] = list(evaluated["members"])
            self.calibrators[horizon] = evaluated["calibrator"]
            self.model_metadata[horizon] = metrics
            self._save_artifacts(horizon)
            self._record_training_run(horizon, model_version, metrics)
            results[horizon] = metrics

        if results:
            self._last_train_ts = trained_at_epoch
            logger.info(
                "Alpha pipeline trained horizons=%s timeframe=%s lookback_days=%d",
                ",".join(sorted(results.keys())),
                self.timeframe,
                self.lookback_days,
            )
        return results

    def train_if_due(self, force: bool = False) -> Dict[str, Dict]:
        if not HAS_ALPHA_ML or not self._pg_available():
            return {}
        now = time.time()
        if not force and self.models and now - self._last_train_ts < self.retrain_interval:
            return {}
        return self.train()
    def _calibrate_probability(self, horizon: str, raw_probability_up: float) -> float:
        calibrator = self.calibrators.get(horizon)
        clipped = float(np.clip(raw_probability_up, 1e-6, 1.0 - 1e-6))
        if calibrator is None:
            return clipped
        calibrated = float(calibrator.transform(np.array([clipped]))[0])
        return float(np.clip(calibrated, 1e-6, 1.0 - 1e-6))

    def _expected_return_bps(self, horizon: str, confidence: float) -> float:
        metadata = self.model_metadata.get(horizon, {})
        curve = metadata.get("expectancy_curve") or []
        for bucket in curve:
            left = float(bucket.get("min", 0.5))
            right = float(bucket.get("max", 1.0))
            if confidence < left:
                continue
            if confidence < right or math.isclose(confidence, right):
                return float(bucket.get("mean_return_bps", 0.0))
        base = float(metadata.get("avg_signed_return_bps", 0.0))
        return float(base * max(0.0, (confidence - 0.5) * 2.0))

    def _prediction_eligible(self, horizon: str, confidence: float) -> bool:
        metadata = self.model_metadata.get(horizon, {})
        return bool(metadata.get("eligible", False)) and confidence >= self.signal_min_confidence

    def predict_coin(self, coin: str, horizon: str = "1h") -> Optional[AlphaPrediction]:
        if horizon not in HORIZON_STEPS:
            return None
        if not HAS_ALPHA_ML or not self._pg_available():
            return None

        self.train_if_due()
        members = self.models.get(horizon)
        if not members:
            return None

        coin = str(coin or "").upper().strip()
        if not coin:
            return None

        key = (coin, horizon)
        now = time.time()
        cached = self.prediction_cache.get(key)
        if cached and now - cached.get("ts", 0.0) < self.cache_ttl:
            return cached["prediction"]

        vector = fs.get_feature_vector(coin, self.timeframe)
        if not vector:
            return None

        X = np.array(
            [[float(vector.get(name, 0.0) or 0.0) for name in self.feature_names]],
            dtype=np.float32,
        )
        raw_probability_up = float(np.mean([model.predict_proba(X)[0][1] for model in members]))
        calibrated_probability_up = self._calibrate_probability(horizon, raw_probability_up)
        confidence = float(max(calibrated_probability_up, 1.0 - calibrated_probability_up))
        predicted_side = "long" if calibrated_probability_up >= 0.5 else "short"

        metadata = self.model_metadata.get(horizon, {})
        prediction = AlphaPrediction(
            coin=coin,
            timeframe=self.timeframe,
            horizon=horizon,
            feature_timestamp_ms=int(vector.get("timestamp_ms", 0) or 0),
            raw_probability_up=float(np.clip(raw_probability_up, 1e-6, 1.0 - 1e-6)),
            calibrated_probability_up=calibrated_probability_up,
            confidence=confidence,
            predicted_side=predicted_side,
            expected_return_bps=self._expected_return_bps(horizon, confidence),
            significance_pvalue=float(metadata.get("significance_pvalue", 1.0)),
            eligible=self._prediction_eligible(horizon, confidence),
            model_version=str(metadata.get("model_version", "untrained")),
        )
        self._record_prediction(prediction)
        self.prediction_cache[key] = {"ts": now, "prediction": prediction}
        return prediction

    def _prediction_coins(self) -> List[str]:
        candidates: List[str] = []
        try:
            from src.core.cycles.feature_cycle import _get_watched_coins

            candidates = [c.upper() for c in _get_watched_coins()]
        except Exception:
            candidates = []

        if not candidates and self._pg_available():
            conn, ret = self._pg_conn()
            if conn is not None:
                try:
                    cur = conn.cursor()
                    cur.execute(
                        """
                        SELECT DISTINCT coin
                        FROM features
                        WHERE timeframe=%s
                        ORDER BY coin ASC
                        LIMIT %s
                        """,
                        (self.timeframe, self.max_prediction_coins * 3),
                    )
                    candidates = [str(row["coin"]).upper() for row in cur.fetchall()]
                except Exception:
                    candidates = []
                finally:
                    ret(conn)

        filtered = []
        for coin in candidates:
            if coin in filtered:
                continue
            if fs.get_feature_vector(coin, self.timeframe):
                filtered.append(coin)
            if len(filtered) >= self.max_prediction_coins:
                break
        return filtered

    def generate_signals(self) -> List[Dict]:
        if not HAS_ALPHA_ML or not self._pg_available():
            return []

        self.train_if_due()
        if not self.models:
            return []

        signals: List[Dict] = []
        for coin in self._prediction_coins():
            predictions = [
                pred
                for pred in (self.predict_coin(coin, horizon) for horizon in HORIZON_STEPS)
                if pred is not None
            ]
            if not predictions:
                continue

            predictions.sort(key=lambda item: item.confidence, reverse=True)
            unique_sides = {pred.predicted_side for pred in predictions}
            if len(unique_sides) > 1:
                if len(predictions) < 2 or predictions[0].confidence - predictions[1].confidence < 0.12:
                    continue
                active_predictions = [predictions[0]]
            else:
                active_predictions = predictions

            if not active_predictions or not all(pred.eligible for pred in active_predictions):
                continue

            combined_confidence = float(np.mean([pred.confidence for pred in active_predictions]))
            if combined_confidence < self.signal_min_confidence:
                continue

            combined_probability_up = float(np.mean([pred.calibrated_probability_up for pred in active_predictions]))
            expected_return_bps = float(np.mean([pred.expected_return_bps for pred in active_predictions]))
            significance_pvalue = float(max(pred.significance_pvalue for pred in active_predictions))
            feature_timestamp_ms = int(max(pred.feature_timestamp_ms for pred in active_predictions))
            predicted_side = active_predictions[0].predicted_side
            horizons = [pred.horizon for pred in active_predictions]
            model_version = "|".join(sorted({pred.model_version for pred in active_predictions}))

            signal = {
                "id": None,
                "name": f"ml_alpha_{coin}_{predicted_side}",
                "strategy_type": "ml_alpha_direction",
                "trader_address": "ml_alpha",
                "current_score": combined_confidence,
                "confidence": combined_confidence,
                "direction": predicted_side,
                "side": predicted_side,
                "source": "ml_alpha",
                "parameters": {
                    "coins": [coin],
                    "timeframe": self.timeframe,
                    "horizons": horizons,
                    "feature_timestamp_ms": feature_timestamp_ms,
                    "calibrated_probability_up": round(combined_probability_up, 6),
                },
                "metrics": {
                    "expected_return_bps": round(expected_return_bps, 4),
                    "significance_pvalue": round(significance_pvalue, 6),
                },
                "metadata": {
                    "model_version": model_version,
                    "confidence_source": "learned_calibration",
                    "prediction_count": len(active_predictions),
                    "horizons": horizons,
                    "alpha_expected_return_bps": round(expected_return_bps, 4),
                    "alpha_significance_pvalue": round(significance_pvalue, 6),
                    "alpha_eligible": True,
                },
            }
            signals.append(signal)

        signals.sort(key=lambda item: item.get("confidence", 0.0), reverse=True)
        return signals

    def get_dashboard_data(self) -> Dict:
        recent_models = {
            horizon: {
                "eligible": bool(meta.get("eligible", False)),
                "model_version": meta.get("model_version"),
                "accuracy": meta.get("accuracy"),
                "balanced_accuracy": meta.get("balanced_accuracy"),
                "active_trade_count": meta.get("active_trade_count"),
                "avg_signed_return_bps": meta.get("avg_signed_return_bps"),
                "significance_pvalue": meta.get("significance_pvalue"),
            }
            for horizon, meta in self.model_metadata.items()
        }
        return {
            "enabled": bool(HAS_ALPHA_ML and self._pg_available()),
            "timeframe": self.timeframe,
            "tracked_horizons": sorted(self.model_metadata.keys()),
            "signal_min_confidence": self.signal_min_confidence,
            "last_trained_ts": self._last_train_ts,
            "recent_models": recent_models,
            "recent_runs": self._latest_run_rows(),
            "recent_predictions": self._latest_prediction_rows(),
            "cache_size": len(self.prediction_cache),
        }

    def get_stats(self) -> Dict:
        return self.get_dashboard_data()
