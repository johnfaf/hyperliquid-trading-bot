"""
Options Flow Scanner — Deribit + Binance Options
Detects unusual options activity: sweeps, blocks, mega prints.
Computes Volume/OI ratios, net directional flow per ticker, and
tags prints by size tier.

Data sources:
  - Deribit API v2 (primary — 90%+ of crypto options volume)
  - Binance Options API (secondary)
"""
import logging
import time
import json
import threading
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import requests
import config

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────
DERIBIT_BASE = "https://www.deribit.com/api/v2"
BINANCE_OPTIONS_BASE = "https://eapi.binance.com"

# Size tier thresholds (notional USD)
TIER_MEGA_BLOCK = 5_000_000
TIER_BLOCK = 2_000_000
TIER_SWEEP = 500_000
TIER_LARGE = 100_000

# Minimum filters
MIN_VOL_OI_RATIO = 0.10  # 10% of OI is already noteworthy for options
MIN_NOTIONAL = 25_000    # Lowered from 50K; options have smaller trade sizes

# Supported underlyings
# SOL removed: Deribit has no SOL options — every scan returned 0 trades.
TRACKED_CURRENCIES = ["BTC", "ETH"]

# Cache TTLs
INSTRUMENTS_CACHE_TTL = 300   # 5 min
TRADES_CACHE_TTL = 30         # 30 sec
OI_CACHE_TTL = 600            # 10 min (was 5 min; OI doesn't change fast enough for frequent refreshes)


class OptionsFlowScanner:
    """
    Scans Deribit + Binance for unusual options prints.
    Maintains a rolling window of recent unusual activity.
    """

    # Expose as class attribute so tests and callers can access via instance
    TRACKED_CURRENCIES = TRACKED_CURRENCIES

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # Caches
        self._instruments_cache = {}  # currency -> {instruments, ts}
        self._oi_cache = {}           # instrument_name -> {oi, ts}
        self._spot_cache = {}         # currency -> {price, ts}

        # Rolling unusual prints (last 24h, capped at 500)
        self.unusual_prints: List[Dict] = []
        self._lock = threading.Lock()

        # Net flow tracking per ticker
        self.net_flow: Dict[str, Dict] = defaultdict(lambda: {
            "bullish_notional": 0, "bearish_notional": 0,
            "bullish_prints": 0, "bearish_prints": 0,
            "prints": [],
        })

        # Conviction rankings
        self.top_convictions: List[Dict] = []
        self.min_conviction_pct = float(
            getattr(config, "OPTIONS_FLOW_MIN_CONVICTION_PCT", 30.0)
        )
        self._venue_last_request = {"deribit": 0.0, "binance_options": 0.0}
        self._venue_min_interval = {"deribit": 0.2, "binance_options": 0.2}

        logger.info("OptionsFlowScanner initialized")

    def _throttle_venue(self, venue: str) -> None:
        """Simple per-venue rate limiter for direct external API calls."""
        min_interval = self._venue_min_interval.get(venue, 0.2)
        last = self._venue_last_request.get(venue, 0.0)
        now = time.time()
        elapsed = now - last
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._venue_last_request[venue] = time.time()

    # ─── Deribit API Helpers ──────────────────────────────────

    def _deribit_get(self, method: str, params: dict = None) -> Optional[dict]:
        """Call a Deribit public API method (JSON-RPC over HTTP GET).
        Includes rate-limit handling: backs off on 429 with exponential delay.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._throttle_venue("deribit")
                url = f"{DERIBIT_BASE}/{method}"
                resp = self.session.get(url, params=params or {}, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("result", data)
                elif resp.status_code == 429:
                    backoff = min(30, 2 ** (attempt + 1))
                    logger.warning(f"Deribit 429 on {method}, backing off {backoff}s "
                                   f"(attempt {attempt+1}/{max_retries})")
                    time.sleep(backoff)
                    continue
                elif resp.status_code >= 500:
                    backoff = min(10, 2 ** attempt)
                    logger.warning(
                        "Deribit %s returned %d (attempt %d/%d); retrying in %.1fs",
                        method, resp.status_code, attempt + 1, max_retries, backoff,
                    )
                    time.sleep(backoff)
                    continue
                else:
                    logger.warning(f"Deribit {method} returned {resp.status_code}")
                    return None
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                backoff = min(10, 2 ** attempt)
                logger.warning(
                    "Deribit %s network error (attempt %d/%d): %s; retrying in %.1fs",
                    method, attempt + 1, max_retries, e, backoff,
                )
                time.sleep(backoff)
                continue
            except Exception as e:
                logger.debug(f"Deribit {method} error: {e}")
                backoff = min(10, 2 ** attempt)
                time.sleep(backoff)
        logger.warning(f"Deribit {method} failed after {max_retries} attempts")
        return None

    def _binance_options_get(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Call Binance Options API."""
        max_retries = 3
        for attempt in range(max_retries):
            self._throttle_venue("binance_options")
            try:
                url = f"{BINANCE_OPTIONS_BASE}{endpoint}"
                resp = self.session.get(url, params=params or {}, timeout=10)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429 or resp.status_code >= 500:
                    backoff = min(10, 2 ** attempt)
                    logger.warning(
                        "Binance Options %s returned %d (attempt %d/%d); retrying in %.1fs",
                        endpoint, resp.status_code, attempt + 1, max_retries, backoff,
                    )
                    time.sleep(backoff)
                    continue
                logger.debug(f"Binance Options {endpoint} returned {resp.status_code}")
                return None
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                backoff = min(10, 2 ** attempt)
                logger.warning(
                    "Binance Options %s network error (attempt %d/%d): %s; retrying in %.1fs",
                    endpoint, attempt + 1, max_retries, e, backoff,
                )
                time.sleep(backoff)
            except Exception as e:
                logger.debug(f"Binance Options error: {e}")
                return None
        return None

    # ─── Instrument Discovery ─────────────────────────────────

    def get_instruments(self, currency: str) -> List[Dict]:
        """Get all active options instruments for a currency. Cached 5 min."""
        now = time.time()
        cached = self._instruments_cache.get(currency)
        if cached and (now - cached["ts"]) < INSTRUMENTS_CACHE_TTL:
            return cached["instruments"]

        instruments = []

        # Deribit
        data = self._deribit_get("public/get_instruments", {
            "currency": currency, "kind": "option", "expired": "false"
        })
        if data and isinstance(data, list):
            for inst in data:
                instruments.append({
                    "name": inst.get("instrument_name", ""),
                    "underlying": currency,
                    "strike": inst.get("strike", 0),
                    "option_type": inst.get("option_type", ""),  # call/put
                    "expiry": inst.get("expiration_timestamp", 0),
                    "expiry_date": datetime.fromtimestamp(
                        inst.get("expiration_timestamp", 0) / 1000
                    ).strftime("%Y-%m-%d") if inst.get("expiration_timestamp") else "",
                    "tick_size": inst.get("tick_size", 0),
                    "min_trade_amount": inst.get("min_trade_amount", 0),
                    "source": "deribit",
                })
            logger.info(f"Deribit: {len(instruments)} active {currency} options")

        self._instruments_cache[currency] = {"instruments": instruments, "ts": now}
        return instruments

    # ─── Spot Prices ──────────────────────────────────────────

    def get_spot_price(self, currency: str) -> float:
        """Get current spot/index price. Cached 30s."""
        now = time.time()
        cached = self._spot_cache.get(currency)
        if cached and (now - cached["ts"]) < 30:
            return cached["price"]

        price = 0.0
        data = self._deribit_get("public/get_index_price", {
            "index_name": f"{currency.lower()}_usd"
        })
        if data and "index_price" in data:
            price = float(data["index_price"])

        if price > 0:
            self._spot_cache[currency] = {"price": price, "ts": now}
        return price

    # ─── Open Interest ────────────────────────────────────────

    def get_open_interest(self, instrument_name: str) -> float:
        """Get open interest for a specific instrument. Cached 10 min. Throttles API only on cache miss."""
        now = time.time()
        cached = self._oi_cache.get(instrument_name)
        if cached and (now - cached["ts"]) < OI_CACHE_TTL:
            return cached["oi"]

        # Cache miss — throttle and fetch from API
        oi = 0.0
        time.sleep(0.25)  # Throttle OI lookups — Deribit rate limit is ~5 req/s
        data = self._deribit_get("public/get_order_book", {
            "instrument_name": instrument_name, "depth": 1
        })
        if data:
            oi = float(data.get("open_interest", 0))

        self._oi_cache[instrument_name] = {"oi": oi, "ts": now}
        return oi

    # ─── Recent Trades / Prints ───────────────────────────────

    def get_recent_trades(self, currency: str, count: int = 500) -> List[Dict]:
        """Get recent options trades from Deribit for a currency."""
        trades = []

        data = self._deribit_get("public/get_last_trades_by_currency", {
            "currency": currency, "kind": "option", "count": min(count, 1000)
        })
        if data and "trades" in data:
            for t in data["trades"]:
                inst_name = t.get("instrument_name", "")
                # Parse instrument name: BTC-28MAR26-90000-C
                parts = inst_name.split("-")
                if len(parts) >= 4:
                    strike = float(parts[2]) if parts[2].isdigit() else 0
                    opt_type = "call" if parts[3] == "C" else "put"
                    expiry_str = parts[1]
                else:
                    strike = 0
                    opt_type = "unknown"
                    expiry_str = ""

                price_usd = float(t.get("price", 0))
                amount = float(t.get("amount", 0))
                # Deribit prices are in BTC for BTC options
                spot = self.get_spot_price(currency)
                notional = price_usd * amount * spot if spot > 0 else price_usd * amount

                trades.append({
                    "instrument": inst_name,
                    "underlying": currency,
                    "strike": strike,
                    "option_type": opt_type,
                    "expiry": expiry_str,
                    "side": t.get("direction", ""),  # buy/sell
                    "price": price_usd,
                    "amount": amount,
                    "notional": notional,
                    "timestamp": t.get("timestamp", 0),
                    "time_str": datetime.fromtimestamp(
                        t.get("timestamp", 0) / 1000
                    ).strftime("%H:%M:%S") if t.get("timestamp") else "",
                    "iv": float(t.get("iv", 0)),
                    "mark_price": float(t.get("mark_price", 0)),
                    "index_price": float(t.get("index_price", 0)),
                    "tick_direction": t.get("tick_direction", 0),
                    "trade_id": t.get("trade_id", ""),
                    "source": "deribit",
                })

            time.sleep(0.15)  # Rate limit

        return trades

    # ─── Flow Classification ──────────────────────────────────

    def classify_print(self, trade: Dict) -> Dict:
        """
        Classify a single options print:
        - Compute Vol/OI ratio
        - Assign size tier (MEGA/BLOCK/SWEEP/LARGE)
        - Determine flow direction (bullish/bearish)
        - Tag strike relative to spot (OTM/ATM/ITM)

        OPTIMIZATION: Early return for sub-notional trades to avoid OI API calls.
        """
        instrument = trade["instrument"]
        notional = trade["notional"]
        amount = trade["amount"]
        opt_type = trade["option_type"]
        side = trade["side"]
        strike = trade["strike"]
        underlying = trade["underlying"]

        # Early exit for low-notional trades — skip expensive OI lookup
        if notional < MIN_NOTIONAL:
            return {
                **trade,
                "vol_oi_ratio": 0.0,
                "open_interest": 0.0,
                "tier": "NORMAL",
                "direction": "neutral",
                "moneyness": "unknown",
                "pct_from_spot": 0,
                "spot_price": 0.0,
                "expiry_window": self._classify_expiry(trade.get("expiry", "")),
                "is_unusual": False,
            }

        # Get OI for this instrument (only called for notional >= MIN_NOTIONAL)
        oi = self.get_open_interest(instrument)
        vol_oi_ratio = amount / oi if oi > 0 else 999.0

        # Size tier
        if notional >= TIER_MEGA_BLOCK:
            tier = "MEGA_BLOCK"
        elif notional >= TIER_BLOCK:
            tier = "BLOCK"
        elif notional >= TIER_SWEEP:
            tier = "SWEEP"
        elif notional >= TIER_LARGE:
            tier = "LARGE"
        else:
            tier = "NORMAL"

        # Flow direction
        # Buy call or Sell put = bullish; Buy put or Sell call = bearish
        if (side == "buy" and opt_type == "call") or (side == "sell" and opt_type == "put"):
            direction = "bullish"
        elif (side == "buy" and opt_type == "put") or (side == "sell" and opt_type == "call"):
            direction = "bearish"
        else:
            direction = "neutral"

        # Strike vs spot (moneyness)
        spot = self.get_spot_price(underlying)
        if spot > 0 and strike > 0:
            pct_from_spot = (strike - spot) / spot * 100
            if opt_type == "call":
                if pct_from_spot > 5:
                    moneyness = "OTM"
                elif pct_from_spot < -5:
                    moneyness = "deep_ITM"
                elif abs(pct_from_spot) <= 5:
                    moneyness = "ATM"
                else:
                    moneyness = "ITM"
            else:  # put
                if pct_from_spot < -5:
                    moneyness = "OTM"
                elif pct_from_spot > 5:
                    moneyness = "deep_ITM"
                elif abs(pct_from_spot) <= 5:
                    moneyness = "ATM"
                else:
                    moneyness = "ITM"
        else:
            moneyness = "unknown"
            pct_from_spot = 0

        # Expiry window classification
        expiry_window = self._classify_expiry(trade.get("expiry", ""))

        return {
            **trade,
            "vol_oi_ratio": round(vol_oi_ratio, 2),
            "open_interest": oi,
            "tier": tier,
            "direction": direction,
            "moneyness": moneyness,
            "pct_from_spot": round(pct_from_spot, 2),
            "spot_price": spot,
            "expiry_window": expiry_window,
            "is_unusual": (vol_oi_ratio >= MIN_VOL_OI_RATIO and notional >= MIN_NOTIONAL),
        }

    def _classify_expiry(self, expiry_str: str) -> str:
        """Classify expiry into weekly/monthly/quarterly."""
        if not expiry_str:
            return "unknown"
        try:
            # Deribit format: 28MAR26
            from datetime import datetime
            exp_date = datetime.strptime(expiry_str, "%d%b%y")
            days_to_expiry = (exp_date - datetime.now(timezone.utc)).days
            if days_to_expiry <= 7:
                return "weekly"
            elif days_to_expiry <= 35:
                return "monthly"
            elif days_to_expiry <= 100:
                return "quarterly"
            else:
                return "leap"
        except Exception:
            return "unknown"

    # ─── Scan Cycle ───────────────────────────────────────────

    def scan_flow(self) -> Dict:
        """
        Run a full flow scan cycle:
        1. Fetch recent trades for all tracked currencies
        2. Classify each print
        3. Filter unusual activity
        4. Update net flow per ticker
        5. Compute top convictions
        Returns summary dict.
        """
        logger.info("Starting options flow scan...")
        all_unusual = []

        for currency in TRACKED_CURRENCIES:
            try:
                trades = self.get_recent_trades(currency, count=500)
                logger.info(f"{currency}: fetched {len(trades)} recent options trades")

                passed_notional = 0
                for trade in trades:
                    # Quick pre-filter: skip trades below notional minimum
                    # This avoids expensive OI lookups for low-value trades
                    if trade.get("notional", 0) < MIN_NOTIONAL:
                        continue

                    passed_notional += 1
                    classified = self.classify_print(trade)
                    if classified["is_unusual"]:
                        all_unusual.append(classified)

                logger.info(f"{currency}: {passed_notional}/{len(trades)} trades passed notional filter ($>{MIN_NOTIONAL}), "
                           f"found {sum(1 for t in all_unusual if t['underlying'] == currency)} unusual")

                time.sleep(0.3)  # Rate limiting between currencies

            except Exception as e:
                logger.error(f"Error scanning {currency} flow: {e}")

        # Update state
        with self._lock:
            # Merge with existing, dedup by trade_id, keep last 500
            existing_ids = {p["trade_id"] for p in self.unusual_prints}
            for p in all_unusual:
                if p["trade_id"] not in existing_ids:
                    self.unusual_prints.append(p)
                    existing_ids.add(p["trade_id"])

            # Trim to last 24h and cap at 500
            cutoff = int((datetime.now(timezone.utc) - timedelta(hours=24)).timestamp() * 1000)
            self.unusual_prints = [
                p for p in self.unusual_prints if p.get("timestamp", 0) > cutoff
            ][-500:]

            # Sort by notional (biggest first)
            self.unusual_prints.sort(key=lambda x: x.get("notional", 0), reverse=True)

            # Rebuild net flow
            self._rebuild_net_flow()

            # Compute top convictions
            self._compute_convictions()

        summary = {
            "total_scanned": sum(1 for _ in TRACKED_CURRENCIES),
            "unusual_prints": len(all_unusual),
            "total_tracked": len(self.unusual_prints),
            "top_convictions": len(self.top_convictions),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(f"Flow scan complete: {len(all_unusual)} new unusual prints, "
                    f"{len(self.unusual_prints)} total tracked")
        return summary

    def _rebuild_net_flow(self):
        """Rebuild net directional flow per underlying from unusual prints."""
        self.net_flow = defaultdict(lambda: {
            "bullish_notional": 0, "bearish_notional": 0,
            "bullish_prints": 0, "bearish_prints": 0,
            "prints": [],
        })
        for p in self.unusual_prints:
            ticker = p["underlying"]
            direction = p["direction"]
            notional = p["notional"]
            if direction == "bullish":
                self.net_flow[ticker]["bullish_notional"] += notional
                self.net_flow[ticker]["bullish_prints"] += 1
            elif direction == "bearish":
                self.net_flow[ticker]["bearish_notional"] += notional
                self.net_flow[ticker]["bearish_prints"] += 1
            self.net_flow[ticker]["prints"].append(p)

    def _compute_convictions(self):
        """
        Compute top conviction rankings — tickers with strongest
        net one-sided flow. Not just "someone bought calls" but
        "net $3.2M bullish across 8 prints on BTC."
        """
        convictions = []
        for ticker, flow in self.net_flow.items():
            net = flow["bullish_notional"] - flow["bearish_notional"]
            total = flow["bullish_notional"] + flow["bearish_notional"]
            total_prints = flow["bullish_prints"] + flow["bearish_prints"]

            if total < MIN_NOTIONAL:
                continue

            conviction_pct = abs(net) / total if total > 0 else 0
            convictions.append({
                "ticker": ticker,
                "net_flow": net,
                "direction": "BULLISH" if net > 0 else "BEARISH",
                "bullish_notional": flow["bullish_notional"],
                "bearish_notional": flow["bearish_notional"],
                "total_prints": total_prints,
                "conviction_pct": round(conviction_pct * 100, 1),
                "spot_price": self.get_spot_price(ticker),
            })

        # Sort by absolute net flow
        convictions.sort(key=lambda x: abs(x["net_flow"]), reverse=True)
        self.top_convictions = convictions[:10]

    # ─── Data Access (for dashboard) ──────────────────────────

    def get_dashboard_data(self) -> Dict:
        """Return all data needed by the options flow dashboard."""
        with self._lock:
            # Heatmap data: ticker × expiry_window → net notional
            heatmap = defaultdict(lambda: defaultdict(float))
            for p in self.unusual_prints:
                ticker = p["underlying"]
                window = p.get("expiry_window", "unknown")
                net_sign = 1 if p["direction"] == "bullish" else -1
                heatmap[ticker][window] += p["notional"] * net_sign

            # Convert to serializable format
            heatmap_data = []
            for ticker in heatmap:
                for window in ["weekly", "monthly", "quarterly", "leap"]:
                    val = heatmap[ticker].get(window, 0)
                    heatmap_data.append({
                        "ticker": ticker,
                        "window": window,
                        "net_notional": round(val, 2),
                    })

            # Net flow bar chart data
            flow_bars = []
            for ticker, flow in self.net_flow.items():
                net = flow["bullish_notional"] - flow["bearish_notional"]
                flow_bars.append({
                    "ticker": ticker,
                    "net_flow": round(net, 2),
                    "bullish": round(flow["bullish_notional"], 2),
                    "bearish": round(flow["bearish_notional"], 2),
                })
            flow_bars.sort(key=lambda x: x["net_flow"], reverse=True)

            # Top unusual prints (limit to 100 for dashboard)
            top_prints = []
            for p in self.unusual_prints[:100]:
                top_prints.append({
                    "time": p.get("time_str", ""),
                    "ticker": p["underlying"],
                    "instrument": p["instrument"],
                    "strike": p["strike"],
                    "option_type": p["option_type"],
                    "expiry": p.get("expiry", ""),
                    "expiry_window": p.get("expiry_window", ""),
                    "side": p.get("side", ""),
                    "amount": p.get("amount", 0),
                    "notional": round(p.get("notional", 0), 2),
                    "vol_oi_ratio": p.get("vol_oi_ratio", 0),
                    "tier": p.get("tier", ""),
                    "direction": p.get("direction", ""),
                    "moneyness": p.get("moneyness", ""),
                    "pct_from_spot": p.get("pct_from_spot", 0),
                    "iv": p.get("iv", 0),
                    "spot_price": p.get("spot_price", 0),
                })

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "convictions": self.top_convictions,
                "heatmap": heatmap_data,
                "flow_bars": flow_bars,
                "unusual_prints": top_prints,
                "spot_prices": {c: self.get_spot_price(c) for c in TRACKED_CURRENCIES},
                "summary": {
                    "total_unusual": len(self.unusual_prints),
                    "currencies_tracked": len(TRACKED_CURRENCIES),
                },
            }

    def get_flow_signal(self, ticker: str) -> Optional[Dict]:
        """
        Get a flow-based trading signal for a ticker.
        Used by the main bot's strategy engine for integration.
        Returns signal dict or None if no strong conviction.
        """
        with self._lock:
            for conv in self.top_convictions:
                try:
                    if (
                        conv.get("ticker") == ticker
                        and conv.get("conviction_pct", 0) >= self.min_conviction_pct
                    ):
                        return {
                            "ticker": ticker,
                            "side": "long" if conv.get("direction") == "BULLISH" else "short",
                            "confidence": conv.get("conviction_pct", 0) / 100,
                            "net_flow": conv.get("net_flow", 0),
                            "total_prints": conv.get("total_prints", 0),
                            "source": "options_flow",
                        }
                except (KeyError, TypeError) as e:
                    logger.debug(f"Options flow signal parse error: {e}")
                    continue
        return None
