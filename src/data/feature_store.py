"""
Feature Store — OHLCV candle storage + computed features for ML/quant signals.

Three layers:
  1. **Candle storage** — upsert/query OHLCV data in Postgres
  2. **Feature computation** — pure numpy functions: candles → feature dict
  3. **Consumer API** — ``get_feature_vector(coin)`` for ML models

The feature store is **Postgres-only** (not dual-write) because it is new,
high-volume, and never existed in SQLite.  If Postgres is unavailable the
feature store gracefully returns ``None`` everywhere.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────

TIMEFRAMES = ("5m", "1h", "4h", "1d")
TIMEFRAME_MS = {
    "5m": 300_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}
# Annualisation factors (periods per year) for realised vol
_PERIODS_PER_YEAR = {
    "5m": 365.25 * 24 * 12,      # 105_192
    "1h": 365.25 * 24,            #   8_766
    "4h": 365.25 * 6,             #   2_191.5
    "1d": 365.25,
}

# All features this module produces
FEATURE_NAMES = [
    # Volatility (5)
    "realized_vol_20", "realized_vol_50", "atr_14", "atr_ratio", "vol_regime",
    # Momentum (7)
    "return_1", "return_5", "return_20", "rsi_14",
    "macd_histogram", "roc_10", "price_vs_sma50",
    # Volume (3)
    "volume_ratio_20", "volume_sma_ratio", "vwap_distance",
    # Funding & OI (3)
    "funding_rate", "funding_annualised", "oi_change_pct",
    # Cross-asset (4)
    "btc_corr_20", "eth_corr_20", "btc_rel_strength", "eth_rel_strength",
]

# ─── Postgres helpers (reuses the existing pool) ──────────────────

def _pg_conn():
    """Get a raw psycopg connection from the existing pool.

    Returns (conn, return_fn) or (None, None) if Postgres unavailable.
    """
    try:
        from src.data.db.postgres import get_connection, return_connection
        conn = get_connection()
        return conn, return_connection
    except Exception as exc:
        logger.debug("Feature store: Postgres unavailable — %s", exc)
        return None, None


def _pg_available() -> bool:
    """Quick check that the Postgres pool is alive."""
    try:
        from src.data.db.postgres import check_health
        return check_health()
    except Exception:
        return False


# =====================================================================
#  Layer 1 — Candle Storage
# =====================================================================

def store_candles(coin: str, timeframe: str, candles: List[dict]) -> int:
    """Upsert OHLCV candles into Postgres.  Returns count written."""
    if not candles:
        return 0
    conn, ret = _pg_conn()
    if conn is None:
        return 0
    try:
        cur = conn.cursor()
        sql = (
            "INSERT INTO candles (coin, timeframe, timestamp_ms, open, high, low, close, volume) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (coin, timeframe, timestamp_ms) DO UPDATE SET "
            "open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, "
            "close=EXCLUDED.close, volume=EXCLUDED.volume"
        )
        rows = [
            (coin, timeframe, int(c["t"]), float(c["o"]), float(c["h"]),
             float(c["l"]), float(c["c"]), float(c.get("v", 0)))
            for c in candles
        ]
        cur.executemany(sql, rows)
        conn.commit()
        return len(rows)
    except Exception as exc:
        logger.warning("store_candles(%s/%s) failed: %s", coin, timeframe, exc)
        try:
            conn.rollback()
        except Exception:
            pass
        return 0
    finally:
        ret(conn)


def get_candles(coin: str, timeframe: str, limit: int = 200) -> List[dict]:
    """Fetch most recent candles, returned oldest-first."""
    conn, ret = _pg_conn()
    if conn is None:
        return []
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT timestamp_ms AS t, open AS o, high AS h, low AS l, "
            "close AS c, volume AS v FROM candles "
            "WHERE coin=%s AND timeframe=%s "
            "ORDER BY timestamp_ms DESC LIMIT %s",
            (coin, timeframe, limit),
        )
        rows = cur.fetchall()
        # Return oldest-first for numpy arrays
        return list(reversed(rows))
    except Exception as exc:
        logger.warning("get_candles(%s/%s) failed: %s", coin, timeframe, exc)
        return []
    finally:
        ret(conn)


def get_sync_watermark(coin: str, timeframe: str) -> int:
    """Return the last candle timestamp_ms we have, or 0."""
    conn, ret = _pg_conn()
    if conn is None:
        return 0
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT last_timestamp_ms FROM candle_sync_state "
            "WHERE coin=%s AND timeframe=%s",
            (coin, timeframe),
        )
        row = cur.fetchone()
        return int(row["last_timestamp_ms"]) if row else 0
    except Exception:
        return 0
    finally:
        ret(conn)


def set_sync_watermark(coin: str, timeframe: str, ts: int) -> None:
    conn, ret = _pg_conn()
    if conn is None:
        return
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO candle_sync_state (coin, timeframe, last_timestamp_ms) "
            "VALUES (%s, %s, %s) "
            "ON CONFLICT (coin, timeframe) DO UPDATE SET last_timestamp_ms=EXCLUDED.last_timestamp_ms",
            (coin, timeframe, ts),
        )
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        ret(conn)


# =====================================================================
#  Layer 2 — Feature Computation (pure numpy, no I/O)
# =====================================================================

def _log_returns(closes: np.ndarray) -> np.ndarray:
    """Log returns from a close price array."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(closes[1:] / closes[:-1])
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def _ema(data: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average via recursive formula."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    """Wilder RSI using simple average of gains/losses."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss < 1e-12:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
         period: int = 14) -> float:
    """Average True Range as fraction of last close."""
    n = len(closes)
    if n < period + 1:
        return 0.0
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    atr_val = np.mean(tr[-period:])
    return float(atr_val / closes[-1]) if closes[-1] > 0 else 0.0


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation (returns 0.0 on degenerate inputs)."""
    if len(x) < 5 or len(y) < 5:
        return 0.0
    minlen = min(len(x), len(y))
    x, y = x[-minlen:], y[-minlen:]
    sx, sy = np.std(x), np.std(y)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compute_features(
    coin: str,
    timeframe: str,
    candles: List[dict],
    funding_rate: float = 0.0,
    open_interest: float = 0.0,
    prev_open_interest: float = 0.0,
    btc_candles: List[dict] = None,
    eth_candles: List[dict] = None,
) -> Dict[str, float]:
    """Compute all features from a candle array.

    Requires >= 60 candles for full output.  Returns partial dict with
    available features if fewer candles are present.
    """
    result: Dict[str, float] = {}
    if not candles or len(candles) < 15:
        return result

    closes = np.array([float(c["c"]) for c in candles])
    highs = np.array([float(c["h"]) for c in candles])
    lows = np.array([float(c["l"]) for c in candles])
    volumes = np.array([float(c.get("v", 0)) for c in candles])
    n = len(closes)
    lr = _log_returns(closes)
    ppy = _PERIODS_PER_YEAR.get(timeframe, 8766)

    # ── Volatility ────────────────────────────────────────────
    if n >= 21:
        result["realized_vol_20"] = float(np.std(lr[-20:]) * math.sqrt(ppy))
    if n >= 51:
        result["realized_vol_50"] = float(np.std(lr[-50:]) * math.sqrt(ppy))
    atr14 = _atr(highs, lows, closes, 14)
    result["atr_14"] = atr14
    if n >= 51:
        atr50 = _atr(highs, lows, closes, 50)
        result["atr_ratio"] = atr14 / atr50 if atr50 > 1e-12 else 1.0
    if "realized_vol_20" in result and "realized_vol_50" in result:
        rv50 = result["realized_vol_50"]
        result["vol_regime"] = result["realized_vol_20"] / rv50 if rv50 > 1e-12 else 1.0

    # ── Momentum ──────────────────────────────────────────────
    if n >= 2:
        result["return_1"] = float(lr[-1])
    if n >= 6:
        result["return_5"] = float(np.sum(lr[-5:]))
    if n >= 21:
        result["return_20"] = float(np.sum(lr[-20:]))

    result["rsi_14"] = _rsi(closes, 14)

    if n >= 27:
        ema12 = _ema(closes, 12)
        ema26 = _ema(closes, 26)
        macd_line = ema12 - ema26
        signal_line = _ema(macd_line, 9)
        result["macd_histogram"] = float(
            (macd_line[-1] - signal_line[-1]) / closes[-1]
        ) if closes[-1] > 0 else 0.0

    if n >= 11:
        result["roc_10"] = float(
            (closes[-1] - closes[-11]) / closes[-11]
        ) if closes[-11] > 0 else 0.0

    if n >= 50:
        sma50 = float(np.mean(closes[-50:]))
        result["price_vs_sma50"] = (closes[-1] - sma50) / sma50 if sma50 > 0 else 0.0

    # ── Volume ────────────────────────────────────────────────
    if n >= 21 and np.sum(volumes[-21:-1]) > 0:
        avg_vol = np.mean(volumes[-21:-1])
        result["volume_ratio_20"] = float(volumes[-1] / avg_vol) if avg_vol > 0 else 1.0
    if n >= 20:
        short_v = np.mean(volumes[-5:]) if n >= 5 else volumes[-1]
        long_v = np.mean(volumes[-20:])
        result["volume_sma_ratio"] = float(short_v / long_v) if long_v > 0 else 1.0
    if n >= 20 and np.sum(volumes[-20:]) > 0:
        vwap = float(np.sum(closes[-20:] * volumes[-20:]) / np.sum(volumes[-20:]))
        result["vwap_distance"] = (closes[-1] - vwap) / closes[-1] if closes[-1] > 0 else 0.0

    # ── Funding & OI ──────────────────────────────────────────
    result["funding_rate"] = funding_rate
    result["funding_annualised"] = funding_rate * 3 * 365  # 8h periods/year
    if prev_open_interest > 0:
        result["oi_change_pct"] = (open_interest - prev_open_interest) / prev_open_interest
    else:
        result["oi_change_pct"] = 0.0

    # ── Cross-asset correlation ───────────────────────────────
    if btc_candles and len(btc_candles) >= 21:
        btc_c = np.array([float(c["c"]) for c in btc_candles])
        btc_lr = _log_returns(btc_c)
        minlen = min(len(lr), len(btc_lr), 20)
        if coin.upper() == "BTC":
            result["btc_corr_20"] = 1.0
        else:
            result["btc_corr_20"] = _pearson(lr[-minlen:], btc_lr[-minlen:])
        if "return_20" in result and len(btc_lr) >= 20:
            result["btc_rel_strength"] = result["return_20"] - float(np.sum(btc_lr[-20:]))

    if eth_candles and len(eth_candles) >= 21:
        eth_c = np.array([float(c["c"]) for c in eth_candles])
        eth_lr = _log_returns(eth_c)
        minlen = min(len(lr), len(eth_lr), 20)
        if coin.upper() == "ETH":
            result["eth_corr_20"] = 1.0
        else:
            result["eth_corr_20"] = _pearson(lr[-minlen:], eth_lr[-minlen:])
        if "return_20" in result and len(eth_lr) >= 20:
            result["eth_rel_strength"] = result["return_20"] - float(np.sum(eth_lr[-20:]))

    return result


# =====================================================================
#  Layer 2b — Store computed features
# =====================================================================

def store_features(coin: str, timeframe: str, timestamp_ms: int,
                   features: Dict[str, float]) -> int:
    """Batch upsert feature values. Returns count written."""
    if not features:
        return 0
    conn, ret = _pg_conn()
    if conn is None:
        return 0
    try:
        cur = conn.cursor()
        sql = (
            "INSERT INTO features (coin, timeframe, timestamp_ms, feature_name, value) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (coin, timeframe, timestamp_ms, feature_name) "
            "DO UPDATE SET value=EXCLUDED.value"
        )
        rows = [
            (coin, timeframe, timestamp_ms, name, float(val))
            for name, val in features.items()
            if val is not None and not math.isnan(val)
        ]
        cur.executemany(sql, rows)
        conn.commit()
        return len(rows)
    except Exception as exc:
        logger.warning("store_features(%s/%s) failed: %s", coin, timeframe, exc)
        try:
            conn.rollback()
        except Exception:
            pass
        return 0
    finally:
        ret(conn)


# =====================================================================
#  Layer 3 — Consumer API
# =====================================================================

def get_feature_vector(coin: str, timeframe: str = "1h") -> Optional[Dict[str, float]]:
    """Get the latest feature values for a coin.  Returns None if unavailable."""
    conn, ret = _pg_conn()
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        # Find the latest timestamp that has features
        cur.execute(
            "SELECT MAX(timestamp_ms) AS ts FROM features "
            "WHERE coin=%s AND timeframe=%s",
            (coin, timeframe),
        )
        row = cur.fetchone()
        if not row or row["ts"] is None:
            return None
        ts = row["ts"]
        cur.execute(
            "SELECT feature_name, value FROM features "
            "WHERE coin=%s AND timeframe=%s AND timestamp_ms=%s",
            (coin, timeframe, ts),
        )
        result = {"timestamp_ms": ts}
        for r in cur.fetchall():
            result[r["feature_name"]] = float(r["value"])
        return result if len(result) > 2 else None  # need at least a few features
    except Exception as exc:
        logger.debug("get_feature_vector(%s/%s) failed: %s", coin, timeframe, exc)
        return None
    finally:
        ret(conn)


def get_features(coin: str, timeframe: str = "1h", lookback: int = 20,
                 feature_names: Optional[List[str]] = None) -> List[Dict[str, float]]:
    """Get feature time-series for a coin, newest-first.

    Returns a list of dicts like::

        [{"timestamp_ms": ..., "rsi_14": 45.2, "realized_vol_20": 0.032}, ...]
    """
    conn, ret = _pg_conn()
    if conn is None:
        return []
    try:
        cur = conn.cursor()
        # Get distinct timestamps
        cur.execute(
            "SELECT DISTINCT timestamp_ms FROM features "
            "WHERE coin=%s AND timeframe=%s "
            "ORDER BY timestamp_ms DESC LIMIT %s",
            (coin, timeframe, lookback),
        )
        timestamps = [r["timestamp_ms"] for r in cur.fetchall()]
        if not timestamps:
            return []

        # Build WHERE clause for feature name filter
        if feature_names:
            placeholders = ",".join(["%s"] * len(feature_names))
            name_clause = f"AND feature_name IN ({placeholders})"
            name_params = list(feature_names)
        else:
            name_clause = ""
            name_params = []

        ts_placeholders = ",".join(["%s"] * len(timestamps))
        cur.execute(
            f"SELECT timestamp_ms, feature_name, value FROM features "
            f"WHERE coin=%s AND timeframe=%s "
            f"AND timestamp_ms IN ({ts_placeholders}) {name_clause} "
            f"ORDER BY timestamp_ms DESC",
            [coin, timeframe] + timestamps + name_params,
        )

        # Group by timestamp
        grouped: Dict[int, Dict[str, float]] = {}
        for r in cur.fetchall():
            ts = r["timestamp_ms"]
            if ts not in grouped:
                grouped[ts] = {"timestamp_ms": ts}
            grouped[ts][r["feature_name"]] = float(r["value"])

        return [grouped[ts] for ts in timestamps if ts in grouped]
    except Exception as exc:
        logger.debug("get_features(%s/%s) failed: %s", coin, timeframe, exc)
        return []
    finally:
        ret(conn)


def get_feature_matrix(coins: List[str], timeframe: str = "1h",
                       feature_names: Optional[List[str]] = None,
                       ) -> Dict[str, Dict[str, float]]:
    """Get latest features for multiple coins.

    Returns ``{coin: {feature_name: value, ...}, ...}``
    """
    result = {}
    for coin in coins:
        vec = get_feature_vector(coin, timeframe)
        if vec:
            if feature_names:
                vec = {k: v for k, v in vec.items()
                       if k in feature_names or k == "timestamp_ms"}
            result[coin] = vec
    return result


# =====================================================================
#  Candle collection helpers
# =====================================================================

def collect_candles_for_coin(coin: str, timeframe: str,
                             backfill_days: int = 0) -> int:
    """Fetch new candles from Hyperliquid and store them.

    If *backfill_days* > 0 and no sync watermark exists, fetches that
    many days of history.  Otherwise only fetches incrementally from
    the last watermark.

    Returns count of candles stored.
    """
    from src.data import hyperliquid_client as hl

    watermark = get_sync_watermark(coin, timeframe)
    now_ms = int(time.time() * 1000)

    if watermark == 0 and backfill_days > 0:
        # First-time backfill
        start_ms = now_ms - (backfill_days * 86_400_000)
    elif watermark > 0:
        # Incremental: start just after last known candle
        start_ms = watermark + 1
    else:
        # No backfill requested, start from 7 days ago default
        start_ms = now_ms - (7 * 86_400_000)

    raw = hl.get_candles(coin, interval=timeframe, start_time=start_ms, end_time=now_ms)
    if not raw:
        return 0

    stored = store_candles(coin, timeframe, raw)
    if stored > 0 and raw:
        # Update watermark to the latest candle timestamp
        max_ts = max(int(c["t"]) for c in raw)
        set_sync_watermark(coin, timeframe, max_ts)

    return stored


def get_candle_count(coin: str = None, timeframe: str = None) -> int:
    """Return total candle count, optionally filtered."""
    conn, ret = _pg_conn()
    if conn is None:
        return 0
    try:
        cur = conn.cursor()
        where_parts, params = [], []
        if coin:
            where_parts.append("coin=%s")
            params.append(coin)
        if timeframe:
            where_parts.append("timeframe=%s")
            params.append(timeframe)
        where = "WHERE " + " AND ".join(where_parts) if where_parts else ""
        cur.execute(f"SELECT COUNT(*) AS cnt FROM candles {where}", params)
        return cur.fetchone()["cnt"]
    except Exception:
        return 0
    finally:
        ret(conn)


def get_feature_count() -> int:
    """Return total feature row count."""
    conn, ret = _pg_conn()
    if conn is None:
        return 0
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS cnt FROM features")
        return cur.fetchone()["cnt"]
    except Exception:
        return 0
    finally:
        ret(conn)
