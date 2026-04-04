"""
Predictive Regime Forecaster (V2)
==================================
Forward-looking regime detection using a 5-input composite signal:
  1. Hyperliquid funding rate slope (public API — no key needed)
  2. Orderbook imbalance (bid/ask depth ratio)
  3. Arkham Intelligence smart-money flow (optional, requires ARKHAM_API_KEY)
  4. Polymarket prediction-market sentiment (event-driven forward signal)
  5. Options flow conviction (Deribit unusual activity net direction)

Produces a regime prediction: "crash", "bullish", or "neutral" with a
confidence score. This prediction is consumed by the DecisionFirewall
for dynamic de-risking — cutting position size and tightening exposure
caps when a crash regime is detected with high confidence.

V2 Composite Signal Formula (re-calibrated weights):
  signal = 0.30 * funding_slope
         + 0.25 * imbalance
         + 0.10 * arkham_signal
         + 0.20 * polymarket_sentiment
         + 0.15 * options_flow_conviction

When Polymarket / Options Flow are unavailable, their weight is
redistributed proportionally to the remaining active inputs so
the signal always sums to the correct scale.

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

# ─── V2 Weights (5-input model) ─────────────────────────────
# When all 5 sources are active these sum to 1.0.
# When Polymarket or Options Flow are unavailable, their weight
# is redistributed to the remaining active inputs.
W_FUNDING = 0.30
W_IMBALANCE = 0.25
W_ARKHAM = 0.10
W_POLYMARKET = 0.20
W_OPTIONS_FLOW = 0.15

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
    Forward-looking regime forecaster using funding, orderbook, on-chain,
    prediction-market, and options flow data.

    Caches predictions per coin with a configurable TTL (default 60s).
    Thread-safe for use across trading cycles.

    External data injection:
        Call `update_polymarket_sentiment(sentiment_dict)` each cycle to feed
        Polymarket data.  Call `update_options_flow(convictions_list)` to feed
        Deribit/Binance options flow.  Both degrade gracefully when absent.
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

        # ─── External data slots (set by main bot loop each cycle) ───
        self._polymarket_sentiment: Optional[Dict] = None
        self._polymarket_ts: float = 0.0
        self._options_convictions: list = []
        self._options_ts: float = 0.0
        self._external_data_ttl = cfg.get("external_data_ttl", 600)  # 10 min staleness limit

        logger.info("PredictiveRegimeForecaster V2 initialized (5-input model)")

    # ─── External Data Injection ─────────────────────────────────

    def update_polymarket_sentiment(self, sentiment: Dict) -> None:
        """
        Feed latest Polymarket sentiment into the forecaster.
        Expected format (from PolymarketScanner.get_market_sentiment()):
            {"sentiment": "bullish"|"bearish"|"neutral",
             "confidence": float, "bullish_probability": float,
             "bearish_probability": float, "markets_analyzed": int}
        """
        self._polymarket_sentiment = sentiment
        self._polymarket_ts = time.time()
        logger.debug(f"Polymarket sentiment updated: {sentiment.get('sentiment', '?')} "
                     f"(conf={sentiment.get('confidence', 0):.2f})")

    def update_options_flow(self, convictions: list) -> None:
        """
        Feed latest options flow top-conviction list into the forecaster.
        Expected format (from OptionsFlowScanner.top_convictions):
            [{"ticker": str, "direction": "BULLISH"|"BEARISH",
              "net_flow": float, "conviction_pct": float, ...}, ...]
        """
        self._options_convictions = convictions or []
        self._options_ts = time.time()
        logger.debug(f"Options flow updated: {len(self._options_convictions)} convictions")

    def predict_regime(self, coin: str = "BTC") -> Dict:
        """
        Predict the current market regime for a coin using the 5-input model.

        Returns:
            {
                "signal": float,          # composite signal (-1 to +1)
                "regime": str,            # "crash", "bullish", "neutral"
                "confidence": float,      # 0-1
                "components": {
                    "funding_slope": float,
                    "imbalance": float,
                    "arkham_flow": float,
                    "polymarket": float,
                    "options_flow": float,
                },
                "active_inputs": int,     # how many of 5 sources were active
            }
        """
        now = time.time()

        # Check cache
        if coin in self.cache:
            cached = self.cache[coin]
            if now - cached.get("ts", 0) < self.cache_ttl:
                return cached["data"]

        # Collect (weight, value) pairs — inactive sources are excluded
        # so their weight gets redistributed to active sources.
        components = {}
        active_weights = []

        # 1. Funding rate slope (always available via public HL API)
        try:
            funding_slope = self._get_funding_slope(coin)
        except Exception:
            funding_slope = 0.0
        components["funding_slope"] = funding_slope
        active_weights.append((W_FUNDING, funding_slope))

        # 2. Orderbook imbalance (always available via public HL API)
        try:
            imbalance = self._get_orderbook_imbalance(coin)
        except Exception:
            imbalance = 0.0
        components["imbalance"] = imbalance
        active_weights.append((W_IMBALANCE, imbalance))

        # 3. Arkham on-chain flow (optional — key-gated)
        arkham_signal = 0.0
        if self.arkham.api_key:
            try:
                flow = self.arkham.get_smart_money_flow(coin)
                arkham_signal = flow.get("net_flow_score", 0.0)
            except Exception:
                pass
        components["arkham_flow"] = arkham_signal
        if self.arkham.api_key:
            active_weights.append((W_ARKHAM, arkham_signal))

        # 4. Polymarket prediction-market sentiment
        pm_signal = self._get_polymarket_signal(now)
        components["polymarket"] = pm_signal
        if pm_signal != 0.0:
            active_weights.append((W_POLYMARKET, pm_signal))

        # 5. Options flow conviction (Deribit unusual activity)
        of_signal = self._get_options_flow_signal(coin, now)
        components["options_flow"] = of_signal
        if of_signal != 0.0:
            active_weights.append((W_OPTIONS_FLOW, of_signal))

        # Compute composite with dynamic weight re-normalization
        total_weight = sum(w for w, _ in active_weights)
        if total_weight > 0:
            signal = sum((w / total_weight) * v for w, v in active_weights)
        else:
            signal = 0.0

        confidence = min(1.0, abs(signal) * 1.8)

        # MED-FIX MED-4: when only near-zero inputs are available (e.g. funding
        # rate near 0, no Arkham/Polymarket data), the re-normalised signal is
        # pure noise.  Classifying that noise as "crash" would fire the 80%
        # size-reduction de-risk on false grounds.  Require a minimum confidence
        # before assigning a non-neutral regime.
        _MIN_CONFIDENCE_FOR_REGIME = 0.05
        if confidence < _MIN_CONFIDENCE_FOR_REGIME:
            regime = "neutral"
        elif signal < CRASH_THRESHOLD:
            regime = "crash"
        elif signal > BULLISH_THRESHOLD:
            regime = "bullish"
        else:
            regime = "neutral"

        data = {
            "signal": round(signal, 4),
            "regime": regime,
            "confidence": round(confidence, 4),
            "components": {k: round(v, 4) for k, v in components.items()},
            "active_inputs": len(active_weights),
        }

        self.cache[coin] = {"data": data, "ts": now}
        inputs_str = "/".join(k for k in components if components[k] != 0.0) or "funding+book"
        logger.info(f"Forecaster {coin} → {regime} (signal={signal:.3f}, "
                    f"conf={confidence:.3f}, inputs={len(active_weights)}: {inputs_str})")
        return data

    # ─── Internal: derive signals from external data ─────────────

    def _get_polymarket_signal(self, now: float) -> float:
        """
        Convert Polymarket sentiment into a -1 to +1 signal.
        Bullish sentiment → positive, bearish → negative.
        Returns 0 if data is stale or unavailable.
        """
        if (self._polymarket_sentiment is None or
                now - self._polymarket_ts > self._external_data_ttl):
            return 0.0

        sent = self._polymarket_sentiment
        sentiment = sent.get("sentiment", "neutral")
        conf = float(sent.get("confidence", 0))

        if sentiment == "bullish":
            return min(conf * 2.0, 1.0)   # scale confidence (0-0.5) → signal (0-1)
        elif sentiment == "bearish":
            return max(-conf * 2.0, -1.0)
        return 0.0

    def _get_options_flow_signal(self, coin: str, now: float) -> float:
        """
        Convert options flow conviction for a coin into a -1 to +1 signal.
        Strong bullish net flow → positive, bearish → negative.
        Returns 0 if data is stale, unavailable, or coin has no conviction entry.
        """
        if (not self._options_convictions or
                now - self._options_ts > self._external_data_ttl):
            return 0.0

        for conv in self._options_convictions:
            if conv.get("ticker", "").upper() == coin.upper():
                pct = conv.get("conviction_pct", 0)
                direction = conv.get("direction", "")
                # conviction_pct is 0-100; normalize to 0-1 then sign
                normalized = min(pct / 100.0, 1.0)
                if direction == "BULLISH":
                    return normalized
                elif direction == "BEARISH":
                    return -normalized
        return 0.0

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
