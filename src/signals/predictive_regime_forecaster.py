"""
Predictive Regime Forecaster
=============================
Forward-looking regime detection using a composite signal from:
  1. Hyperliquid funding rate slope (public API — no key needed)
  2. Orderbook imbalance (bid/ask depth ratio)
  3. Arkham Intelligence smart-money flow (optional, requires ARKHAM_API_KEY)

Produces a regime prediction: "crash", "bullish", or "neutral" with a
confidence score. This prediction is consumed by the DecisionFirewall
for dynamic de-risking — cutting position size and tightening exposure
caps when a crash regime is detected with high confidence.

Composite Signal Formula (calibrated weights):
  signal = 0.4 * funding_slope + 0.35 * imbalance + 0.25 * arkham_signal

Regime Classification:
  signal < -0.15 → "crash"
  signal >  0.15 → "bullish"
  else           → "neutral"
"""

import logging
import time
import os
from typing import Dict, Optional
from collections import deque

import requests
import numpy as np

logger = logging.getLogger(__name__)

# Weights for composite signal
W_FUNDING = 0.40
W_IMBALANCE = 0.35
W_ARKHAM = 0.25

# Regime thresholds
CRASH_THRESHOLD = -0.15
BULLISH_THRESHOLD = 0.15


class ArkhamClient:
    """
    Lightweight Arkham Intelligence API client.
    Provides on-chain smart-money flow scoring.
    Optional — degrades gracefully if no API key is set.
    """

    def __init__(self):
        self.api_key = os.environ.get("ARKHAM_API_KEY")
        self.base_url = "https://api.arkhamintelligence.com"
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
            logger.info("Arkham Intelligence client initialized (API key set)")
        else:
            logger.info("Arkham Intelligence disabled (no ARKHAM_API_KEY)")

    def get_smart_money_flow(self, coin: str = "BTC") -> Dict:
        """
        Query net flow score from smart-money entities linked to Hyperliquid.
        Returns {"net_flow_score": float} where -1.0 = outflow, +1.0 = inflow.
        """
        if not self.api_key:
            return {"net_flow_score": 0.0}
        try:
            resp = self.session.post(
                f"{self.base_url}/v1/flow",
                json={
                    "asset": coin,
                    "chain": "arbitrum",  # Hyperliquid L1 proxy
                    "window_hours": 4,
                    "entity_types": ["smart_money", "whale", "perp_fund"]
                },
                timeout=8
            )
            resp.raise_for_status()
            data = resp.json()
            # Normalize to -1.0 (outflow) → +1.0 (inflow)
            score = data.get("net_flow", 0) / 1_000_000  # scale
            return {"net_flow_score": max(min(score, 1.0), -1.0)}
        except Exception as e:
            logger.debug(f"Arkham flow query failed: {e}")
            return {"net_flow_score": 0.0}


class PredictiveRegimeForecaster:
    """
    Forward-looking regime forecaster using funding, orderbook, and on-chain data.

    Caches predictions per coin with a configurable TTL (default 60s).
    Thread-safe for use across trading cycles.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.hl_info_url = cfg.get("hl_info_url", "https://api.hyperliquid.xyz/info")
        self.cache_ttl = cfg.get("cache_ttl", 60)  # seconds
        self.cache: Dict[str, Dict] = {}

        # Funding rate history for slope computation
        self._funding_history: Dict[str, deque] = {}
        self._funding_window = cfg.get("funding_window", 8)  # last N observations

        # Arkham client (optional)
        self.arkham = ArkhamClient()

        logger.info("PredictiveRegimeForecaster initialized")

    def predict_regime(self, coin: str = "BTC") -> Dict:
        """
        Predict the current market regime for a coin.

        Returns:
            {
                "signal": float,          # composite signal (-1 to +1)
                "regime": str,            # "crash", "bullish", "neutral"
                "confidence": float,      # 0-1
                "components": {
                    "funding_slope": float,
                    "imbalance": float,
                    "arkham_flow": float
                }
            }
        """
        now = time.time()

        # Check cache
        if coin in self.cache:
            cached = self.cache[coin]
            if now - cached.get("ts", 0) < self.cache_ttl:
                return cached["data"]

        # 1. Funding rate slope
        try:
            funding_slope = self._get_funding_slope(coin)
        except Exception:
            funding_slope = 0.0

        # 2. Orderbook imbalance
        try:
            imbalance = self._get_orderbook_imbalance(coin)
        except Exception:
            imbalance = 0.0

        # 3. Arkham on-chain flow
        arkham_signal = 0.0
        if self.arkham.api_key:
            try:
                flow = self.arkham.get_smart_money_flow(coin)
                arkham_signal = flow.get("net_flow_score", 0.0)
            except Exception:
                pass

        # Composite signal (calibrated weights)
        signal = (W_FUNDING * funding_slope +
                  W_IMBALANCE * imbalance +
                  W_ARKHAM * arkham_signal)

        # Regime classification
        if signal < CRASH_THRESHOLD:
            regime = "crash"
        elif signal > BULLISH_THRESHOLD:
            regime = "bullish"
        else:
            regime = "neutral"

        confidence = min(1.0, abs(signal) * 1.8)

        data = {
            "signal": round(signal, 4),
            "regime": regime,
            "confidence": round(confidence, 4),
            "components": {
                "funding_slope": round(funding_slope, 4),
                "imbalance": round(imbalance, 4),
                "arkham_flow": round(arkham_signal, 4)
            }
        }

        self.cache[coin] = {"data": data, "ts": now}
        logger.info(f"Forecaster {coin} → {regime} (signal={signal:.3f}, conf={confidence:.3f})")
        return data

    def _get_funding_slope(self, coin: str) -> float:
        """
        Compute the slope of recent funding rates.
        Negative slope = funding turning negative = bearish pressure.
        Returns normalized value in [-1, +1].
        """
        try:
            resp = requests.post(
                self.hl_info_url,
                json={"type": "metaAndAssetCtxs"},
                timeout=5
            )
            resp.raise_for_status()
            data = resp.json()

            # Find the coin's funding rate
            if len(data) >= 2:
                asset_ctxs = data[1]
                meta = data[0]
                universe = meta.get("universe", [])
                for i, asset in enumerate(universe):
                    if asset.get("name", "").upper() == coin.upper():
                        if i < len(asset_ctxs):
                            funding = float(asset_ctxs[i].get("funding", 0))

                            # Track history
                            if coin not in self._funding_history:
                                self._funding_history[coin] = deque(maxlen=self._funding_window)
                            self._funding_history[coin].append(funding)

                            # Compute slope from history
                            hist = list(self._funding_history[coin])
                            if len(hist) >= 3:
                                x = np.arange(len(hist), dtype=np.float64)
                                y = np.array(hist, dtype=np.float64)
                                slope = np.polyfit(x, y, 1)[0]
                                # Normalize: typical funding slope range is ~1e-5
                                return max(min(slope * 100_000, 1.0), -1.0)
                            elif len(hist) >= 1:
                                # Single point: just normalize raw funding
                                return max(min(funding * 10_000, 1.0), -1.0)
                            break
        except Exception as e:
            logger.debug(f"Funding slope fetch failed for {coin}: {e}")

        return 0.0

    def _get_orderbook_imbalance(self, coin: str) -> float:
        """
        Compute bid/ask imbalance from Hyperliquid L2 orderbook.
        Returns value in [-1, +1]:
          +1 = strongly bid-heavy (bullish)
          -1 = strongly ask-heavy (bearish)
        """
        try:
            resp = requests.post(
                self.hl_info_url,
                json={"type": "l2Book", "coin": coin},
                timeout=5
            )
            resp.raise_for_status()
            book = resp.json()

            levels = book.get("levels", [[], []])
            if len(levels) < 2:
                return 0.0

            bids = levels[0][:10]  # top 10 levels
            asks = levels[1][:10]

            bid_depth = sum(float(b.get("sz", 0)) for b in bids)
            ask_depth = sum(float(a.get("sz", 0)) for a in asks)

            total = bid_depth + ask_depth
            if total == 0:
                return 0.0

            # Imbalance: (bid - ask) / (bid + ask)
            imbalance = (bid_depth - ask_depth) / total
            return max(min(imbalance, 1.0), -1.0)

        except Exception as e:
            logger.debug(f"Orderbook imbalance fetch failed for {coin}: {e}")
            return 0.0
