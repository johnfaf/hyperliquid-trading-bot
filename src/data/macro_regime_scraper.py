"""
Macro Regime Scraper
====================
Slow-polling, heavily-cached scraper that collects macro / sentiment data
from external sources and produces a **protective regime overlay**.

This is NOT a signal generator — it adjusts the bot's risk posture
(size modifiers, confidence drags, entry blocking) rather than producing
trade orders.

Sources (all free / public):
  1. CoinAnk  — order-book depth + liquidation heatmap
  2. FinancialJuice — real-time macro headlines
  3. ForexFactory — economic calendar (high-impact events)
  4. Nasdaq — upcoming earnings for macro-correlated names
  5. Finviz — futures snapshot (S&P, Nasdaq, VIX, DXY, Gold, Oil, Crypto)
  6. SentimentTrader — public sentiment gauges (if available)

Design choices:
  - One requests.Session with retry + exponential backoff
  - Per-source TTL caches (5 min → 4 h depending on volatility)
  - Thread-safe (single lock around the snapshot dict)
  - Graceful degradation: every source can fail independently
  - Each sub-score is -1.0 (max bearish) to +1.0 (max bullish)
  - The aggregated output is a risk-posture dict consumed by the
    trading cycle (not individual trade signals)
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from html import unescape
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests

import config

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

ET_TZ = ZoneInfo("America/New_York")


# ─── Source URLs ─────────────────────────────────────────────────────────────
FINVIZ_FUTURES_URL = "https://finviz.com/futures.ashx"
FOREXFACTORY_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FINANCIAL_JUICE_URL = "https://www.financialjuice.com/home"
NASDAQ_EARNINGS_URL = "https://www.nasdaq.com/market-activity/earnings"
COINANK_DEPTH_URL = "https://coinank.com/api/indicator/order-depth"
COINANK_LIQ_URL = "https://coinank.com/api/liqMap/total"
SENTIMENTRADER_URL = "https://sentimentrader.com"


# ─── Cache TTLs (seconds) ───────────────────────────────────────────────────
_TTL_FINVIZ = 300          # 5 min — futures change quickly
_TTL_FOREXFACTORY = 3600   # 1 h — calendar is weekly
_TTL_FINANCIAL_JUICE = 600 # 10 min — headline sentiment
_TTL_NASDAQ_EARNINGS = 14400  # 4 h — daily changes
_TTL_COINANK = 900         # 15 min — orderbook / liq data
_TTL_SENTIMENTRADER = 1800 # 30 min


# ─── Sentiment keyword banks ────────────────────────────────────────────────
_BEARISH_KEYWORDS = (
    "crash", "recession", "default", "collapse", "downgrade", "layoff",
    "bankruptcy", "plunge", "sell-off", "selloff", "hawkish", "rate hike",
    "inflation surge", "crisis", "war", "sanction", "tariff", "miss",
    "worse than expected", "disappointing", "contraction", "decline",
    "drops", "tumbles", "slump", "fear", "panic", "outflow",
)
_BULLISH_KEYWORDS = (
    "rally", "surge", "breakout", "all-time high", "ath", "dovish",
    "rate cut", "stimulus", "easing", "beat", "better than expected",
    "recovery", "bull", "inflow", "strong", "growth", "expansion",
    "upgrade", "optimism", "boost", "gains", "soars", "record",
)


# ─── Finviz futures tickers we care about ────────────────────────────────────
_FINVIZ_TICKERS = {
    "S&P 500": "sp500",
    "Dow Jones": "dow",
    "Nasdaq": "nasdaq",
    "Russell 2000": "russell",
    "Crude Oil": "oil",
    "Gold": "gold",
    "Silver": "silver",
    "US Dollar Index": "dxy",
    "Bitcoin": "btc",
    "Ethereum": "eth",
    "VIX": "vix",
    "10-Year Bond": "us10y",
    "Euro": "euro",
    "Nat Gas": "natgas",
}


# ─── Risk Level Enum ─────────────────────────────────────────────────────────
RISK_LEVELS = {
    "low":      {"size_mod": 1.00, "conf_drag": 0.00},
    "normal":   {"size_mod": 0.90, "conf_drag": -0.03},
    "elevated": {"size_mod": 0.70, "conf_drag": -0.07},
    "high":     {"size_mod": 0.45, "conf_drag": -0.12},
    "extreme":  {"size_mod": 0.20, "conf_drag": -0.20},
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class MacroRegimeScraper:
    """
    Collects macro data from external sources and produces a protective
    risk-posture overlay for the trading cycle.

    Usage:
        scraper = MacroRegimeScraper()
        posture = scraper.get_risk_posture()
        # posture["macro_risk_level"]  -> "elevated"
        # posture["size_modifier"]     -> 0.70
        # posture["confidence_drag"]   -> -0.07
        # posture["block_new_entries"] -> False
    """

    def __init__(self, config_override: Optional[Dict] = None):
        cfg = dict(config_override or {})
        self.enabled = bool(cfg.get(
            "enabled",
            getattr(config, "MACRO_REGIME_ENABLED", True),
        ))
        self.refresh_seconds = int(cfg.get(
            "refresh_seconds",
            getattr(config, "MACRO_REGIME_REFRESH_SECONDS", 900),
        ))
        self.block_at_level = str(cfg.get(
            "block_at_level",
            getattr(config, "MACRO_REGIME_BLOCK_AT_LEVEL", "extreme"),
        ))

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/json,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })

        self._lock = threading.Lock()
        self._cache: Dict[str, Tuple[float, Any]] = {}  # key -> (expire_ts, data)
        self._snapshot: Optional[Dict] = None
        self._last_refresh = 0.0
        self._source_status: Dict[str, Dict] = {}

        logger.info("MacroRegimeScraper initialized (enabled=%s)", self.enabled)

    # ─── Cache Helpers ───────────────────────────────────────────────────

    def _get_cached(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry and time.time() < entry[0]:
            return entry[1]
        return None

    def _set_cached(self, key: str, data: Any, ttl: int) -> None:
        self._cache[key] = (time.time() + ttl, data)

    # ─── Fetch Helpers ───────────────────────────────────────────────────

    def _fetch(self, url: str, timeout: int = 20, max_retries: int = 2,
               as_json: bool = False) -> Optional[Any]:
        """Fetch with retry + exponential backoff."""
        for attempt in range(max_retries + 1):
            try:
                resp = self.session.get(url, timeout=timeout)
                if resp.status_code == 429:
                    wait = min(30, 2 ** (attempt + 1))
                    logger.debug("Rate limited on %s, waiting %ds", url, wait)
                    time.sleep(wait)
                    continue
                if resp.status_code >= 500:
                    if attempt < max_retries:
                        time.sleep(min(10, 2 ** attempt))
                        continue
                    return None
                resp.raise_for_status()
                return resp.json() if as_json else resp.text
            except requests.exceptions.JSONDecodeError:
                logger.debug("JSON decode failed for %s", url)
                return None
            except Exception as exc:
                if attempt < max_retries:
                    time.sleep(min(10, 2 ** attempt))
                    continue
                logger.debug("Fetch failed for %s: %s", url, exc)
                return None
        return None

    def _set_source(self, name: str, ok: bool, detail: str = "") -> None:
        self._source_status[name] = {
            "ok": ok,
            "detail": detail[:200] if detail else "",
            "updated_at": _utc_now().isoformat(),
        }

    # =====================================================================
    # Source 1: Finviz Futures
    # =====================================================================

    def _scrape_finviz_futures(self) -> Dict[str, float]:
        """
        Scrape Finviz futures page for daily % changes.
        Returns dict of ticker_key -> daily_pct_change.
        """
        cached = self._get_cached("finviz")
        if cached is not None:
            return cached

        html = self._fetch(FINVIZ_FUTURES_URL)
        if not html:
            self._set_source("finviz_futures", False, "fetch_failed")
            return {}

        result: Dict[str, float] = {}
        try:
            # Find all % change cells — pattern: >+0.45%< or >-1.23%<
            # Finviz renders futures in table rows with name and % change
            rows = re.findall(
                r'<td[^>]*class="[^"]*futures[^"]*"[^>]*>.*?</tr>',
                html, re.DOTALL | re.IGNORECASE,
            )
            if not rows:
                # Fallback: find label + percentage pairs
                pairs = re.findall(
                    r'(?:title|alt)="([^"]+)"[^>]*>.*?'
                    r'([+-]?\d+\.\d+)%',
                    html, re.DOTALL,
                )
                for label, pct in pairs:
                    for display_name, key in _FINVIZ_TICKERS.items():
                        if display_name.lower() in label.lower():
                            result[key] = float(pct)
                            break

            if not result:
                # Second fallback: brute-force find all percentage values near known names
                for display_name, key in _FINVIZ_TICKERS.items():
                    pattern = re.compile(
                        re.escape(display_name) + r'.*?([+-]?\d+\.\d+)%',
                        re.DOTALL | re.IGNORECASE,
                    )
                    match = pattern.search(html)
                    if match:
                        result[key] = float(match.group(1))

            self._set_cached("finviz", result, _TTL_FINVIZ)
            self._set_source("finviz_futures", True, f"{len(result)} tickers")
        except Exception as exc:
            self._set_source("finviz_futures", False, str(exc))

        return result

    def _score_finviz(self) -> Tuple[float, Dict]:
        """
        Convert futures snapshot to a -1..+1 sentiment score.
        Equity risk-on vs VIX/DXY risk-off divergence is the key signal.
        """
        futures = self._scrape_finviz_futures()
        if not futures:
            return 0.0, {}

        # Risk-on components (positive = bullish)
        equity_avg = 0.0
        equity_count = 0
        for key in ("sp500", "nasdaq", "dow", "russell"):
            if key in futures:
                equity_avg += futures[key]
                equity_count += 1
        if equity_count:
            equity_avg /= equity_count

        crypto_avg = 0.0
        crypto_count = 0
        for key in ("btc", "eth"):
            if key in futures:
                crypto_avg += futures[key]
                crypto_count += 1
        if crypto_count:
            crypto_avg /= crypto_count

        # Risk-off components (positive = bearish for equities)
        vix_change = futures.get("vix", 0.0)
        dxy_change = futures.get("dxy", 0.0)
        gold_change = futures.get("gold", 0.0)

        # Score: equities up + crypto up + VIX down + DXY down = bullish
        # Scale: ±2% daily move maps to ±1.0
        risk_on = _clamp(equity_avg / 2.0) * 0.35
        risk_on += _clamp(crypto_avg / 3.0) * 0.25
        risk_off = _clamp(-vix_change / 5.0) * 0.20   # VIX up = bearish
        risk_off += _clamp(-dxy_change / 1.5) * 0.10   # DXY up = bearish for crypto
        risk_off += _clamp(gold_change / 2.0) * 0.10   # gold up = mild flight to safety

        score = _clamp(risk_on + risk_off)
        return score, futures

    # =====================================================================
    # Source 2: ForexFactory Calendar
    # =====================================================================

    def _scrape_forexfactory(self) -> List[Dict]:
        """
        Fetch ForexFactory economic calendar (JSON feed).
        Returns list of high/medium impact events.
        """
        cached = self._get_cached("forexfactory")
        if cached is not None:
            return cached

        data = self._fetch(FOREXFACTORY_CALENDAR_URL, as_json=True)
        if not data or not isinstance(data, list):
            self._set_source("forexfactory", False, "fetch_failed")
            return []

        events = []
        now = _utc_now()
        try:
            for item in data:
                impact = str(item.get("impact", "")).strip()
                if impact not in ("High", "Medium"):
                    continue
                title = str(item.get("title", "")).strip()
                country = str(item.get("country", "")).strip()
                date_str = str(item.get("date", "")).strip()

                event_time = None
                try:
                    event_time = datetime.fromisoformat(
                        date_str.replace("Z", "+00:00")
                    ).astimezone(timezone.utc)
                except Exception:
                    pass

                if not event_time:
                    continue

                # Only care about events within ±48h
                delta = (event_time - now).total_seconds()
                if abs(delta) > 48 * 3600:
                    continue

                actual = item.get("actual", "")
                forecast = item.get("forecast", "")
                previous = item.get("previous", "")

                events.append({
                    "title": title,
                    "country": country,
                    "impact": impact,
                    "time": event_time.isoformat(),
                    "minutes_until": int(delta / 60),
                    "actual": actual,
                    "forecast": forecast,
                    "previous": previous,
                    "is_released": bool(actual),
                })

            self._set_cached("forexfactory", events, _TTL_FOREXFACTORY)
            self._set_source("forexfactory", True, f"{len(events)} events")
        except Exception as exc:
            self._set_source("forexfactory", False, str(exc))

        return events

    def _score_forexfactory(self) -> Tuple[float, Dict]:
        """
        Score the macro calendar.
        Imminent high-impact events = defensive drag.
        Surprise beats/misses after release = directional score.
        """
        events = self._scrape_forexfactory()
        if not events:
            return 0.0, {}

        imminent_high = 0
        imminent_medium = 0
        surprise_score = 0.0
        surprise_count = 0

        for ev in events:
            mins = ev.get("minutes_until", 0)
            impact = ev.get("impact", "")
            is_released = ev.get("is_released", False)

            # Pre-release: defensive drag for approaching events
            if not is_released and 0 < mins <= 120:
                if impact == "High":
                    imminent_high += 1
                elif impact == "Medium":
                    imminent_medium += 1

            # Post-release: actual vs forecast surprise
            if is_released and ev.get("actual") and ev.get("forecast"):
                try:
                    actual = float(re.sub(r"[^\d.\-]", "", str(ev["actual"])))
                    forecast = float(re.sub(r"[^\d.\-]", "", str(ev["forecast"])))
                    if forecast != 0:
                        surprise = (actual - forecast) / abs(forecast)
                        surprise_score += _clamp(surprise / 0.1)  # 10% surprise = max
                        surprise_count += 1
                except (ValueError, ZeroDivisionError):
                    pass

        # Defensive drag: -0.15 per imminent high-impact, -0.05 per medium
        event_drag = -(imminent_high * 0.15 + imminent_medium * 0.05)
        event_drag = max(event_drag, -0.6)

        # Surprise component
        surprise_avg = (surprise_score / surprise_count) if surprise_count else 0.0

        score = _clamp(event_drag + surprise_avg * 0.3)
        detail = {
            "imminent_high": imminent_high,
            "imminent_medium": imminent_medium,
            "surprise_count": surprise_count,
            "surprise_avg": round(surprise_avg, 3) if surprise_count else None,
        }
        return score, detail

    # =====================================================================
    # Source 3: FinancialJuice Headlines
    # =====================================================================

    def _scrape_financial_juice(self) -> List[str]:
        """Scrape recent macro headlines from FinancialJuice."""
        cached = self._get_cached("financialjuice")
        if cached is not None:
            return cached

        html = self._fetch(FINANCIAL_JUICE_URL)
        if not html:
            self._set_source("financialjuice", False, "fetch_failed")
            return []

        headlines = []
        try:
            # Extract headline text from HTML
            # FinancialJuice displays headlines in divs/spans
            raw_headlines = re.findall(
                r'class="[^"]*(?:headline|news-item|feed-item|story)[^"]*"[^>]*>'
                r'\s*(.*?)\s*<',
                html, re.DOTALL | re.IGNORECASE,
            )
            if not raw_headlines:
                # Broader fallback: find text in list items or p tags
                raw_headlines = re.findall(
                    r'<(?:li|p|div|span)[^>]*class="[^"]*"[^>]*>\s*'
                    r'([A-Z][^<]{20,200})\s*<',
                    html,
                )

            for h in raw_headlines[:50]:
                cleaned = unescape(re.sub(r"<[^>]+>", "", h)).strip()
                if len(cleaned) > 15:
                    headlines.append(cleaned)

            self._set_cached("financialjuice", headlines, _TTL_FINANCIAL_JUICE)
            self._set_source("financialjuice", True, f"{len(headlines)} headlines")
        except Exception as exc:
            self._set_source("financialjuice", False, str(exc))

        return headlines

    def _score_financial_juice(self) -> Tuple[float, Dict]:
        """Simple keyword sentiment on macro headlines."""
        headlines = self._scrape_financial_juice()
        if not headlines:
            return 0.0, {}

        bullish = 0
        bearish = 0
        for h in headlines:
            h_lower = h.lower()
            if any(kw in h_lower for kw in _BULLISH_KEYWORDS):
                bullish += 1
            if any(kw in h_lower for kw in _BEARISH_KEYWORDS):
                bearish += 1

        total = bullish + bearish
        if total == 0:
            return 0.0, {"headlines_scanned": len(headlines)}

        # Net sentiment: bullish fraction minus bearish fraction
        net = (bullish - bearish) / total
        score = _clamp(net * 0.5)  # Dampen — keyword sentiment is noisy
        return score, {
            "headlines_scanned": len(headlines),
            "bullish_hits": bullish,
            "bearish_hits": bearish,
        }

    # =====================================================================
    # Source 4: Nasdaq Earnings Calendar
    # =====================================================================

    def _scrape_nasdaq_earnings(self) -> Dict:
        """Check for high-profile earnings that could move indices/crypto."""
        cached = self._get_cached("nasdaq_earnings")
        if cached is not None:
            return cached

        html = self._fetch(NASDAQ_EARNINGS_URL)
        if not html:
            self._set_source("nasdaq_earnings", False, "fetch_failed")
            return {}

        result: Dict = {"mega_cap_today": 0, "total_today": 0}
        try:
            # Look for mega-cap names that move markets
            mega_caps = (
                "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META",
                "TSLA", "COIN", "MSTR", "BRK", "JPM", "GS", "MS",
                "V", "MA", "CRM", "NFLX", "AMD", "INTC",
            )
            # Count how many mega-caps appear on the page today
            for ticker in mega_caps:
                if re.search(rf"\b{ticker}\b", html):
                    result["mega_cap_today"] += 1

            # Rough count of total earnings rows
            total_matches = re.findall(r"(?:earnings|report)", html, re.IGNORECASE)
            result["total_today"] = min(len(total_matches), 200)

            self._set_cached("nasdaq_earnings", result, _TTL_NASDAQ_EARNINGS)
            self._set_source("nasdaq_earnings", True,
                             f"{result['mega_cap_today']} mega-cap, {result['total_today']} total")
        except Exception as exc:
            self._set_source("nasdaq_earnings", False, str(exc))

        return result

    def _score_nasdaq_earnings(self) -> Tuple[float, Dict]:
        """
        Mega-cap earnings days = higher uncertainty = defensive drag.
        This isn't directional — we just reduce risk.
        """
        data = self._scrape_nasdaq_earnings()
        mega = data.get("mega_cap_today", 0)
        if mega == 0:
            return 0.0, data

        # Each mega-cap earning = -0.08 drag (they move everything)
        drag = -min(mega * 0.08, 0.4)
        return drag, data

    # =====================================================================
    # Source 5: CoinAnk Order Depth
    # =====================================================================

    def _scrape_coinank_depth(self) -> Dict:
        """
        Attempt to fetch CoinAnk order depth data.
        Falls back gracefully if JS-rendered or API changes.
        """
        cached = self._get_cached("coinank_depth")
        if cached is not None:
            return cached

        data = self._fetch(COINANK_DEPTH_URL, as_json=True)
        if not data or not isinstance(data, dict):
            # Try HTML fallback
            html = self._fetch("https://coinank.com/chart/indicator/order-depth")
            if html:
                self._set_source("coinank_depth", True, "html_only")
                result = {"html_available": True, "data": None}
            else:
                self._set_source("coinank_depth", False, "fetch_failed")
                result = {}
            self._set_cached("coinank_depth", result, _TTL_COINANK)
            return result

        try:
            self._set_cached("coinank_depth", data, _TTL_COINANK)
            self._set_source("coinank_depth", True, "api_ok")
        except Exception as exc:
            self._set_source("coinank_depth", False, str(exc))
        return data

    # =====================================================================
    # Source 6: CoinAnk Liquidation Map
    # =====================================================================

    def _scrape_coinank_liq(self) -> Dict:
        """
        Attempt to fetch CoinAnk liquidation map data.
        Falls back gracefully if JS-rendered or API changes.
        """
        cached = self._get_cached("coinank_liq")
        if cached is not None:
            return cached

        data = self._fetch(COINANK_LIQ_URL, as_json=True)
        if not data or not isinstance(data, dict):
            html = self._fetch("https://coinank.com/chart/derivatives/liq-map")
            if html:
                self._set_source("coinank_liq", True, "html_only")
                result = {"html_available": True, "data": None}
            else:
                self._set_source("coinank_liq", False, "fetch_failed")
                result = {}
            self._set_cached("coinank_liq", result, _TTL_COINANK)
            return result

        try:
            self._set_cached("coinank_liq", data, _TTL_COINANK)
            self._set_source("coinank_liq", True, "api_ok")
        except Exception as exc:
            self._set_source("coinank_liq", False, str(exc))
        return data

    def _score_coinank(self) -> Tuple[float, Dict]:
        """
        Score CoinAnk data (order depth + liquidation map).
        If APIs are JS-gated we return neutral — the score will improve
        once we reverse-engineer the actual JSON endpoints or add a
        browser-based approach.
        """
        depth = self._scrape_coinank_depth()
        liq = self._scrape_coinank_liq()

        # Try to extract useful data if the API returned something
        score = 0.0
        detail: Dict = {"depth_available": bool(depth), "liq_available": bool(liq)}

        # If we got actual API data, try to read bid/ask depth ratio
        if isinstance(depth, dict) and depth.get("data"):
            try:
                d = depth["data"]
                bid_depth = float(d.get("bidDepth", 0) or 0)
                ask_depth = float(d.get("askDepth", 0) or 0)
                if bid_depth + ask_depth > 0:
                    ratio = (bid_depth - ask_depth) / (bid_depth + ask_depth)
                    score += _clamp(ratio * 0.3)
                    detail["depth_ratio"] = round(ratio, 3)
            except Exception:
                pass

        # If we got liq data, check long/short imbalance
        if isinstance(liq, dict) and liq.get("data"):
            try:
                d = liq["data"]
                long_liq = float(d.get("longLiquidation", 0) or 0)
                short_liq = float(d.get("shortLiquidation", 0) or 0)
                if long_liq + short_liq > 0:
                    # More long liqs = bearish pressure
                    imbalance = (short_liq - long_liq) / (long_liq + short_liq)
                    score += _clamp(imbalance * 0.2)
                    detail["liq_imbalance"] = round(imbalance, 3)
            except Exception:
                pass

        return _clamp(score), detail

    # =====================================================================
    # Source 7: SentimentTrader
    # =====================================================================

    def _scrape_sentimentrader(self) -> Dict:
        """
        Attempt to scrape public sentiment indicators from SentimentTrader.
        Most data is behind a paywall — we extract whatever is publicly visible.
        """
        cached = self._get_cached("sentimentrader")
        if cached is not None:
            return cached

        html = self._fetch(SENTIMENTRADER_URL)
        if not html:
            self._set_source("sentimentrader", False, "fetch_failed")
            return {}

        result: Dict = {"available": False}
        try:
            # Look for any public sentiment gauges / numbers
            # SentimentTrader may show "Smart Money / Dumb Money" confidence
            smart_money = re.search(
                r"(?:smart\s*money|institutional)[^<]*?(\d{1,3})%",
                html, re.IGNORECASE,
            )
            dumb_money = re.search(
                r"(?:dumb\s*money|retail)[^<]*?(\d{1,3})%",
                html, re.IGNORECASE,
            )

            if smart_money:
                result["smart_money_confidence"] = int(smart_money.group(1))
                result["available"] = True
            if dumb_money:
                result["dumb_money_confidence"] = int(dumb_money.group(1))
                result["available"] = True

            # Fear/greed or other sentiment gauges
            fear_greed = re.search(
                r"(?:fear|greed)\s*(?:&amp;|and|/)\s*(?:greed|fear)[^<]*?(\d{1,3})",
                html, re.IGNORECASE,
            )
            if fear_greed:
                result["fear_greed_index"] = int(fear_greed.group(1))
                result["available"] = True

            self._set_cached("sentimentrader", result, _TTL_SENTIMENTRADER)
            self._set_source("sentimentrader", result["available"], str(result))
        except Exception as exc:
            self._set_source("sentimentrader", False, str(exc))

        return result

    def _score_sentimentrader(self) -> Tuple[float, Dict]:
        """Score sentiment data if available."""
        data = self._scrape_sentimentrader()
        if not data.get("available"):
            return 0.0, data

        score = 0.0
        # Smart money high confidence = follow the smart money direction
        if "smart_money_confidence" in data:
            sm = data["smart_money_confidence"]
            # >75% = bullish, <25% = bearish
            score += _clamp((sm - 50) / 50.0) * 0.3

        # Dumb money high confidence = contrarian bearish
        if "dumb_money_confidence" in data:
            dm = data["dumb_money_confidence"]
            # Dumb money very confident = contrarian signal (bearish)
            score -= _clamp((dm - 50) / 50.0) * 0.15

        # Fear/Greed index
        if "fear_greed_index" in data:
            fg = data["fear_greed_index"]
            # 0=extreme fear (bullish contrarian), 100=extreme greed (bearish)
            # We use it as a defensive signal: extreme greed = reduce risk
            score -= _clamp((fg - 50) / 50.0) * 0.2

        return _clamp(score), data

    # =====================================================================
    # Aggregation
    # =====================================================================

    def _compute_components(self) -> Dict[str, Tuple[float, Dict]]:
        """Collect all component scores. Each source runs independently."""
        components = {}

        # Finviz futures — most reliable source
        try:
            components["futures_sentiment"] = self._score_finviz()
        except Exception as exc:
            logger.debug("Finviz scoring error: %s", exc)
            components["futures_sentiment"] = (0.0, {"error": str(exc)})

        # ForexFactory calendar
        try:
            components["macro_calendar"] = self._score_forexfactory()
        except Exception as exc:
            logger.debug("ForexFactory scoring error: %s", exc)
            components["macro_calendar"] = (0.0, {"error": str(exc)})

        # FinancialJuice headlines
        try:
            components["headline_sentiment"] = self._score_financial_juice()
        except Exception as exc:
            logger.debug("FinancialJuice scoring error: %s", exc)
            components["headline_sentiment"] = (0.0, {"error": str(exc)})

        # Nasdaq earnings
        try:
            components["earnings_risk"] = self._score_nasdaq_earnings()
        except Exception as exc:
            logger.debug("Nasdaq earnings scoring error: %s", exc)
            components["earnings_risk"] = (0.0, {"error": str(exc)})

        # CoinAnk (depth + liq)
        try:
            components["order_depth_liq"] = self._score_coinank()
        except Exception as exc:
            logger.debug("CoinAnk scoring error: %s", exc)
            components["order_depth_liq"] = (0.0, {"error": str(exc)})

        # SentimentTrader
        try:
            components["institutional_sentiment"] = self._score_sentimentrader()
        except Exception as exc:
            logger.debug("SentimentTrader scoring error: %s", exc)
            components["institutional_sentiment"] = (0.0, {"error": str(exc)})

        return components

    def _aggregate_score(self, components: Dict[str, Tuple[float, Dict]]) -> float:
        """
        Weighted average of all component scores.
        Weights reflect reliability and importance.
        """
        weights = {
            "futures_sentiment": 0.30,       # Most reliable, real-time market data
            "macro_calendar": 0.25,          # High-impact events are critical
            "headline_sentiment": 0.10,      # Noisy, low weight
            "earnings_risk": 0.10,           # Important but infrequent
            "order_depth_liq": 0.15,         # Crypto-specific, very relevant
            "institutional_sentiment": 0.10, # Often unavailable, low weight
        }

        weighted_sum = 0.0
        weight_sum = 0.0

        for key, (score, _detail) in components.items():
            w = weights.get(key, 0.1)
            if score != 0.0 or key in ("futures_sentiment", "macro_calendar"):
                # Always include futures & calendar even if 0 (they're reliable zeros)
                weighted_sum += score * w
                weight_sum += w

        if weight_sum == 0:
            return 0.0

        return _clamp(weighted_sum / weight_sum)

    def _score_to_risk_level(self, score: float) -> str:
        """Convert aggregate score to a risk level string."""
        # score is -1.0 (max bearish) to +1.0 (max bullish)
        # More negative = higher risk
        if score >= 0.3:
            return "low"
        elif score >= 0.05:
            return "normal"
        elif score >= -0.15:
            return "elevated"
        elif score >= -0.40:
            return "high"
        else:
            return "extreme"

    # =====================================================================
    # Public API
    # =====================================================================

    def get_risk_posture(self, force: bool = False) -> Dict:
        """
        Return the current macro risk posture.

        Returns:
            Dict with keys:
                macro_risk_level: str ("low"/"normal"/"elevated"/"high"/"extreme")
                macro_score: float (-1..+1, negative = bearish)
                size_modifier: float (0.2..1.0)
                confidence_drag: float (-0.20..0.0)
                block_new_entries: bool
                reasons: list of str
                components: dict of component scores
                sources: dict of source health status
                timestamp: str
        """
        if not self.enabled:
            return self._neutral_posture("macro_regime_disabled")

        with self._lock:
            if (not force
                    and self._snapshot
                    and (time.time() - self._last_refresh) < self.refresh_seconds):
                return self._snapshot

        # Scrape all sources (outside lock — can be slow)
        components = self._compute_components()
        agg_score = self._aggregate_score(components)
        risk_level = self._score_to_risk_level(agg_score)
        level_params = RISK_LEVELS[risk_level]

        # Build reasons
        reasons = []
        for key, (score, detail) in components.items():
            if score <= -0.15:
                reasons.append(f"{key}: bearish ({score:+.2f})")
            elif score >= 0.15:
                reasons.append(f"{key}: bullish ({score:+.2f})")

        block_levels = {"extreme"} if self.block_at_level == "extreme" else {"extreme", "high"}
        should_block = risk_level in block_levels

        posture = {
            "macro_risk_level": risk_level,
            "macro_score": round(agg_score, 3),
            "size_modifier": level_params["size_mod"],
            "confidence_drag": level_params["conf_drag"],
            "block_new_entries": should_block,
            "reasons": reasons[:5],
            "components": {
                k: {"score": round(s, 3), "detail": d}
                for k, (s, d) in components.items()
            },
            "sources": dict(self._source_status),
            "timestamp": _utc_now().isoformat(),
        }

        with self._lock:
            self._snapshot = posture
            self._last_refresh = time.time()

        return posture

    def _neutral_posture(self, reason: str = "") -> Dict:
        return {
            "macro_risk_level": "normal",
            "macro_score": 0.0,
            "size_modifier": 1.0,
            "confidence_drag": 0.0,
            "block_new_entries": False,
            "reasons": [reason] if reason else [],
            "components": {},
            "sources": {},
            "timestamp": _utc_now().isoformat(),
        }

    def get_stats(self) -> Dict:
        """Compact stats for health/reporting."""
        posture = self.get_risk_posture()
        return {
            "macro_risk_level": posture["macro_risk_level"],
            "macro_score": posture["macro_score"],
            "size_modifier": posture["size_modifier"],
            "block_new_entries": posture["block_new_entries"],
            "sources_ok": sum(1 for s in posture.get("sources", {}).values() if s.get("ok")),
            "sources_total": len(posture.get("sources", {})),
            "timestamp": posture.get("timestamp"),
        }

    def get_dashboard_data(self) -> Dict:
        """Full posture data for dashboard display."""
        return self.get_risk_posture()
