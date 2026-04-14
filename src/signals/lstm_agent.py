"""
LSTM Alpha Signal Agent
========================
Time-series forecasting agent that predicts 1h/4h price direction using
an LSTM neural network trained on the feature store candle data.

Integrates with the Alpha Arena as a new agent type:
  - Trains on historical OHLCV + feature data (walk-forward, no leakage)
  - Produces directional signals (long/short) with confidence scores
  - Participates in Arena tournament alongside rule-based agents
  - Auto-retrains periodically on new data

Architecture:
  Input:  (sequence_length, n_features) per coin
  Model:  LSTM(64) -> Dropout(0.3) -> Dense(32) -> Dense(3) [up/down/flat]
  Output: Softmax probabilities for direction classification

Requirements:
  pip install torch  (PyTorch — CPU is fine for this small model)
"""
import logging
import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy torch import — only loaded when LSTM is actually used
_torch = None
_nn = None


def _ensure_torch():
    """Lazy-load PyTorch to avoid import errors when not installed."""
    global _torch, _nn
    if _torch is not None:
        return True
    try:
        import torch
        import torch.nn as nn
        _torch = torch
        _nn = nn
        return True
    except ImportError:
        logger.warning("LSTMAgent: PyTorch not installed. pip install torch")
        return False


# ─── LSTM Model Definition ───────────────────────────────────

# Deferred nn.Module subclass — created lazily after torch is imported
_LSTMBlockClass = None


def _get_lstm_block_class():
    """Create the _LSTMBlock nn.Module subclass (deferred to avoid import-time torch dependency)."""
    global _LSTMBlockClass
    if _LSTMBlockClass is not None:
        return _LSTMBlockClass

    class LSTMBlock(_nn.Module):
        """LSTM layer that returns only the last hidden state."""

        def __init__(self, input_size: int, hidden_size: int, dropout: float):
            super().__init__()
            self.lstm = _nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
            )

        def forward(self, x):
            output, (h_n, _) = self.lstm(x)
            return h_n[-1]

    _LSTMBlockClass = LSTMBlock
    return _LSTMBlockClass


class DirectionLSTM:
    """Lightweight LSTM for 3-class direction prediction."""

    def __init__(self, n_features: int, hidden_size: int = 64,
                 sequence_length: int = 30, dropout: float = 0.3):
        if not _ensure_torch():
            raise ImportError("PyTorch required for LSTM agent")

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.device = _torch.device("cpu")

        LSTMBlock = _get_lstm_block_class()

        # Build model
        self.model = _nn.Sequential(
            LSTMBlock(n_features, hidden_size, dropout),
            _nn.Linear(hidden_size, 32),
            _nn.ReLU(),
            _nn.Dropout(dropout),
            _nn.Linear(32, 3),  # [down, flat, up]
        ).to(self.device)

        self.optimizer = _torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = _nn.CrossEntropyLoss()
        self._trained = False

    def train_model(self, X: np.ndarray, y: np.ndarray,
                    epochs: int = 50, batch_size: int = 32,
                    val_split: float = 0.15) -> Dict:
        """
        Train the LSTM on labeled sequences.

        Args:
            X: (n_samples, sequence_length, n_features) input sequences
            y: (n_samples,) labels — 0=down, 1=flat, 2=up
            epochs: Training epochs
            batch_size: Mini-batch size
            val_split: Fraction held out for validation

        Returns:
            Training metrics dict
        """
        n = len(X)
        val_n = max(1, int(n * val_split))
        # Walk-forward: validation is the most recent data
        X_train, y_train = X[:-val_n], y[:-val_n]
        X_val, y_val = X[-val_n:], y[-val_n:]

        X_t = _torch.FloatTensor(X_train).to(self.device)
        y_t = _torch.LongTensor(y_train).to(self.device)
        X_v = _torch.FloatTensor(X_val).to(self.device)
        y_v = _torch.LongTensor(y_val).to(self.device)

        best_val_acc = 0
        best_state = None
        patience = 8
        no_improve = 0

        self.model.train()
        for epoch in range(epochs):
            # Shuffle training data
            perm = _torch.randperm(len(X_t))
            X_t, y_t = X_t[perm], y_t[perm]

            total_loss = 0
            n_batches = 0
            for i in range(0, len(X_t), batch_size):
                batch_X = X_t[i:i + batch_size]
                batch_y = y_t[i:i + batch_size]

                self.optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                _torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            # Validation
            self.model.eval()
            with _torch.no_grad():
                val_preds = self.model(X_v)
                val_acc = (val_preds.argmax(dim=1) == y_v).float().mean().item()
            self.model.train()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)
        self._trained = True

        self.model.eval()
        with _torch.no_grad():
            val_preds = self.model(X_v)
            val_labels = val_preds.argmax(dim=1)
            acc = (val_labels == y_v).float().mean().item()

        logger.info("LSTM trained: val_acc=%.1f%%, samples=%d, epochs=%d",
                    acc * 100, n, epoch + 1)
        return {"val_accuracy": acc, "samples": n, "epochs": epoch + 1}

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict direction probabilities.

        Args:
            X: (n_samples, sequence_length, n_features)

        Returns:
            (classes, probabilities) where classes are 0=down, 1=flat, 2=up
        """
        if not self._trained:
            return np.array([1]), np.array([[0.33, 0.34, 0.33]])

        self.model.eval()
        with _torch.no_grad():
            X_t = _torch.FloatTensor(X).to(self.device)
            logits = self.model(X_t)
            probs = _torch.softmax(logits, dim=1).cpu().numpy()
            classes = probs.argmax(axis=1)
        return classes, probs

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _torch.save({
            "state_dict": self.model.state_dict(),
            "n_features": self.n_features,
            "hidden_size": self.hidden_size,
            "sequence_length": self.sequence_length,
            "trained": self._trained,
        }, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            ckpt = _torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["state_dict"])
            self._trained = ckpt.get("trained", True)
            self.model.eval()
            logger.info("LSTM model loaded from %s", path)
            return True
        except Exception as e:
            logger.warning("LSTM model load failed: %s", e)
            return False


# ─── Feature Engineering ─────────────────────────────────────

# Features extracted from OHLCV candles
CANDLE_FEATURES = [
    "returns",           # Close-to-close return
    "log_volume",        # Log volume (normalized)
    "high_low_range",    # (High - Low) / Close
    "close_position",    # (Close - Low) / (High - Low)
    "sma_ratio_5",       # Close / SMA(5)
    "sma_ratio_20",      # Close / SMA(20)
    "rsi_norm",          # RSI / 100 (normalized)
    "volatility",        # Rolling std of returns
]
N_FEATURES = len(CANDLE_FEATURES)


def extract_features_from_candles(candles: List[Dict]) -> np.ndarray:
    """
    Convert OHLCV candle list to feature matrix.

    Args:
        candles: List of dicts with open, high, low, close, volume keys.

    Returns:
        (n_candles, N_FEATURES) numpy array. First ~20 rows may have NaN.
    """
    n = len(candles)
    features = np.zeros((n, N_FEATURES), dtype=np.float32)

    closes = np.array([c["close"] for c in candles], dtype=np.float64)
    highs = np.array([c["high"] for c in candles], dtype=np.float64)
    lows = np.array([c["low"] for c in candles], dtype=np.float64)
    volumes = np.array([c.get("volume", 1.0) for c in candles], dtype=np.float64)

    # Returns
    returns = np.zeros(n)
    returns[1:] = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-10)
    features[:, 0] = returns

    # Log volume (normalized by rolling mean)
    log_vol = np.log1p(volumes)
    for i in range(n):
        window = log_vol[max(0, i - 20):i + 1]
        mean_v = window.mean() if len(window) > 0 else 1.0
        features[i, 1] = (log_vol[i] / (mean_v + 1e-10)) - 1.0

    # High-Low range
    features[:, 2] = (highs - lows) / (closes + 1e-10)

    # Close position within bar
    hl_range = highs - lows
    hl_range[hl_range < 1e-10] = 1e-10
    features[:, 3] = (closes - lows) / hl_range

    # SMA ratios
    for i in range(n):
        if i >= 4:
            sma5 = closes[max(0, i - 4):i + 1].mean()
            features[i, 4] = closes[i] / (sma5 + 1e-10) - 1.0
        if i >= 19:
            sma20 = closes[max(0, i - 19):i + 1].mean()
            features[i, 5] = closes[i] / (sma20 + 1e-10) - 1.0

    # RSI (14-period)
    for i in range(14, n):
        window_returns = returns[i - 13:i + 1]
        gains = np.maximum(window_returns, 0).mean()
        losses = -np.minimum(window_returns, 0).mean()
        rs = gains / (losses + 1e-10)
        features[i, 6] = (100 - 100 / (1 + rs)) / 100.0  # Normalized 0-1

    # Rolling volatility (14-period std of returns)
    for i in range(14, n):
        features[i, 7] = returns[max(0, i - 13):i + 1].std()

    return features


def create_sequences(features: np.ndarray, labels: np.ndarray,
                     seq_len: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create (X, y) sequences for LSTM training.

    Args:
        features: (n, n_features) feature matrix
        labels: (n,) direction labels (0=down, 1=flat, 2=up)
        seq_len: Number of past bars per sequence

    Returns:
        X: (n_sequences, seq_len, n_features)
        y: (n_sequences,) labels
    """
    X, y = [], []
    for i in range(seq_len, len(features)):
        seq = features[i - seq_len:i]
        if not np.isnan(seq).any():
            X.append(seq)
            y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def label_direction(candles: List[Dict], lookahead: int = 6,
                    threshold: float = 0.003) -> np.ndarray:
    """
    Label each candle with future direction.

    Args:
        candles: OHLCV candle list
        lookahead: Bars to look ahead for direction
        threshold: Minimum return to count as up/down (0.3% default)

    Returns:
        (n,) array — 0=down, 1=flat, 2=up. Last `lookahead` bars are 1 (flat).
    """
    closes = np.array([c["close"] for c in candles], dtype=np.float64)
    n = len(closes)
    labels = np.ones(n, dtype=np.int64)  # default flat

    for i in range(n - lookahead):
        future_return = (closes[i + lookahead] - closes[i]) / (closes[i] + 1e-10)
        if future_return > threshold:
            labels[i] = 2  # up
        elif future_return < -threshold:
            labels[i] = 0  # down
        # else stays 1 (flat)

    return labels


# ─── LSTM Arena Agent ─────────────────────────────────────────

class LSTMAgent:
    """
    LSTM-based trading agent for the Alpha Arena.

    Wraps the DirectionLSTM model and provides the same signal generation
    interface that the Arena backtester expects.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.sequence_length = cfg.get("sequence_length", 30)
        self.hidden_size = cfg.get("hidden_size", 64)
        self.lookahead = cfg.get("lookahead", 6)  # 6 bars = 6h on 1h candles
        self.direction_threshold = cfg.get("direction_threshold", 0.003)
        self.retrain_interval = cfg.get("retrain_interval", 21600)  # 6 hours
        self.min_training_samples = cfg.get("min_training_samples", 200)
        self.model_dir = cfg.get("model_dir", "models/lstm_direction")

        self._model: Optional[DirectionLSTM] = None
        self._last_train_time: float = 0
        self._train_metrics: Dict = {}
        self._initialized = False

        # Try to load existing model
        if _ensure_torch():
            self._model = DirectionLSTM(
                n_features=N_FEATURES,
                hidden_size=self.hidden_size,
                sequence_length=self.sequence_length,
            )
            model_path = os.path.join(self.model_dir, "lstm_direction.pt")
            if self._model.load(model_path):
                self._initialized = True
                logger.info("LSTMAgent: loaded pre-trained model")
            else:
                logger.info("LSTMAgent: no pre-trained model, will train on first data")
        else:
            logger.warning("LSTMAgent: PyTorch not available, agent disabled")

    def train(self, candles: List[Dict]) -> Optional[Dict]:
        """
        Train or retrain the LSTM on candle data.

        Args:
            candles: Historical OHLCV candles (chronological order)

        Returns:
            Training metrics or None if training skipped/failed
        """
        if self._model is None:
            return None

        if len(candles) < self.min_training_samples + self.sequence_length:
            logger.info("LSTMAgent: not enough candles (%d < %d), skipping training",
                       len(candles), self.min_training_samples + self.sequence_length)
            return None

        # Check retrain interval
        now = time.time()
        if self._initialized and (now - self._last_train_time) < self.retrain_interval:
            return None

        logger.info("LSTMAgent: training on %d candles...", len(candles))

        features = extract_features_from_candles(candles)
        labels = label_direction(candles, self.lookahead, self.direction_threshold)
        X, y = create_sequences(features, labels, self.sequence_length)

        if len(X) < self.min_training_samples:
            logger.info("LSTMAgent: not enough valid sequences (%d)", len(X))
            return None

        metrics = self._model.train_model(X, y)
        self._train_metrics = metrics
        self._last_train_time = now
        self._initialized = True

        # Save model
        model_path = os.path.join(self.model_dir, "lstm_direction.pt")
        self._model.save(model_path)
        logger.info("LSTMAgent: model saved to %s", model_path)

        return metrics

    def generate_signal(self, candles: List[Dict]) -> Optional[Dict]:
        """
        Generate a trading signal from recent candles.

        Compatible with Arena's signal format:
        {"side": "long"/"short", "confidence": 0.0-1.0, "price": float, "atr_pct": float}

        Args:
            candles: Recent OHLCV candles (at least sequence_length + 20)

        Returns:
            Signal dict or None
        """
        if not self._initialized or self._model is None:
            return None

        min_required = self.sequence_length + 20
        if len(candles) < min_required:
            return None

        features = extract_features_from_candles(candles)
        # Take the last sequence
        seq = features[-self.sequence_length:]
        if np.isnan(seq).any():
            return None

        X = seq.reshape(1, self.sequence_length, N_FEATURES)
        classes, probs = self._model.predict(X)

        # probs[0] = [P(down), P(flat), P(up)]
        p_down, p_flat, p_up = probs[0]
        current_price = candles[-1]["close"]

        # Compute ATR% for context
        recent_returns = []
        for i in range(max(1, len(candles) - 14), len(candles)):
            r = (candles[i]["close"] - candles[i - 1]["close"]) / (candles[i - 1]["close"] + 1e-10)
            recent_returns.append(r)
        atr_pct = float(np.std(recent_returns)) if recent_returns else 0.02

        # Decision: signal if dominant class has >45% probability
        # and is clearly above the opposite direction
        min_prob = 0.45
        min_edge = 0.15  # Must beat opposite by this margin

        if p_up > min_prob and (p_up - p_down) > min_edge:
            confidence = float(min(0.4 + (p_up - 0.45) * 4, 0.90))
            return {
                "side": "long",
                "confidence": confidence,
                "price": current_price,
                "atr_pct": atr_pct,
                "lstm_probs": {"down": round(p_down, 3), "flat": round(p_flat, 3), "up": round(p_up, 3)},
            }
        elif p_down > min_prob and (p_down - p_up) > min_edge:
            confidence = float(min(0.4 + (p_down - 0.45) * 4, 0.90))
            return {
                "side": "short",
                "confidence": confidence,
                "price": current_price,
                "atr_pct": atr_pct,
                "lstm_probs": {"down": round(p_down, 3), "flat": round(p_flat, 3), "up": round(p_up, 3)},
            }

        return None  # No clear signal

    def get_stats(self) -> Dict:
        """Return agent status."""
        return {
            "initialized": self._initialized,
            "model_available": self._model is not None,
            "last_train_time": self._last_train_time,
            "train_metrics": self._train_metrics,
        }
