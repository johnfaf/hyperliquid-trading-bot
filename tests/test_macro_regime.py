"""Tests for the macro regime protective overlay."""
import time
from unittest.mock import MagicMock

from src.data.macro_regime_scraper import (
    MacroRegimeScraper,
    RISK_LEVELS,
    _clamp,
)


# ─── Unit: _clamp ────────────────────────────────────────────────────────────

def test_clamp_within_range():
    assert _clamp(0.5) == 0.5
    assert _clamp(-0.3) == -0.3

def test_clamp_above():
    assert _clamp(2.0) == 1.0
    assert _clamp(1.5, lo=0.0, hi=1.0) == 1.0

def test_clamp_below():
    assert _clamp(-5.0) == -1.0


# ─── Unit: risk levels ───────────────────────────────────────────────────────

def test_risk_levels_structure():
    for level in ("low", "normal", "elevated", "high", "extreme"):
        assert level in RISK_LEVELS
        params = RISK_LEVELS[level]
        assert "size_mod" in params
        assert "conf_drag" in params
        assert 0.0 <= params["size_mod"] <= 1.0
        assert params["conf_drag"] <= 0.0


# ─── Unit: score_to_risk_level ───────────────────────────────────────────────

def test_score_to_risk_level():
    scraper = MacroRegimeScraper({"enabled": False})
    assert scraper._score_to_risk_level(0.5) == "low"
    assert scraper._score_to_risk_level(0.1) == "normal"
    assert scraper._score_to_risk_level(0.0) == "elevated"
    assert scraper._score_to_risk_level(-0.2) == "high"
    assert scraper._score_to_risk_level(-0.5) == "extreme"


# ─── Unit: disabled returns neutral ─────────────────────────────────────────

def test_disabled_returns_neutral_posture():
    scraper = MacroRegimeScraper({"enabled": False})
    posture = scraper.get_risk_posture()
    assert posture["macro_risk_level"] == "normal"
    assert posture["size_modifier"] == 1.0
    assert posture["confidence_drag"] == 0.0
    assert posture["block_new_entries"] is False


# ─── Unit: caching works ────────────────────────────────────────────────────

def test_cache_hit():
    scraper = MacroRegimeScraper({"enabled": True})
    scraper._set_cached("test_key", {"value": 42}, ttl=60)
    assert scraper._get_cached("test_key") == {"value": 42}

def test_cache_expired():
    scraper = MacroRegimeScraper({"enabled": True})
    scraper._set_cached("test_key", {"value": 42}, ttl=0)
    time.sleep(0.01)
    assert scraper._get_cached("test_key") is None


# ─── Integration: finviz scoring ─────────────────────────────────────────────

def test_score_finviz_bullish():
    scraper = MacroRegimeScraper({"enabled": True})
    # Inject cached data: equities up, VIX down
    scraper._set_cached("finviz", {
        "sp500": 1.5, "nasdaq": 2.0, "dow": 0.8, "russell": 1.0,
        "btc": 3.0, "eth": 4.0,
        "vix": -5.0, "dxy": -0.5, "gold": -0.3,
    }, ttl=300)
    score, detail = scraper._score_finviz()
    assert score > 0.1, f"Expected bullish score, got {score}"

def test_score_finviz_bearish():
    scraper = MacroRegimeScraper({"enabled": True})
    scraper._set_cached("finviz", {
        "sp500": -2.0, "nasdaq": -2.5, "dow": -1.5,
        "btc": -5.0, "eth": -6.0,
        "vix": 15.0, "dxy": 1.5, "gold": 2.0,
    }, ttl=300)
    score, detail = scraper._score_finviz()
    assert score < -0.1, f"Expected bearish score, got {score}"


# ─── Integration: forexfactory scoring ───────────────────────────────────────

def test_score_forexfactory_imminent_event():
    scraper = MacroRegimeScraper({"enabled": True})
    # Inject cached data with imminent high-impact event
    scraper._set_cached("forexfactory", [
        {"title": "NFP", "impact": "High", "minutes_until": 30,
         "is_released": False, "actual": "", "forecast": "200K", "previous": "180K"},
    ], ttl=300)
    score, detail = scraper._score_forexfactory()
    assert score < 0, f"Expected defensive drag, got {score}"
    assert detail["imminent_high"] == 1

def test_score_forexfactory_no_events():
    scraper = MacroRegimeScraper({"enabled": True})
    scraper._set_cached("forexfactory", [], ttl=300)
    score, detail = scraper._score_forexfactory()
    assert score == 0.0


# ─── Integration: headline sentiment ────────────────────────────────────────

def test_score_headlines_bearish():
    scraper = MacroRegimeScraper({"enabled": True})
    scraper._set_cached("financialjuice", [
        "Markets crash on recession fears",
        "Fed signals more rate hikes ahead",
        "Inflation surge exceeds expectations",
        "Tech stocks plunge in sell-off",
    ], ttl=300)
    score, detail = scraper._score_financial_juice()
    assert score < 0, f"Expected bearish, got {score}"
    assert detail["bearish_hits"] > 0

def test_score_headlines_bullish():
    scraper = MacroRegimeScraper({"enabled": True})
    scraper._set_cached("financialjuice", [
        "Markets rally on strong earnings",
        "Fed hints at rate cut in September",
        "Recovery gains momentum as growth beats forecast",
    ], ttl=300)
    score, detail = scraper._score_financial_juice()
    assert score > 0, f"Expected bullish, got {score}"


# ─── Integration: earnings risk ──────────────────────────────────────────────

def test_score_earnings_mega_cap():
    scraper = MacroRegimeScraper({"enabled": True})
    scraper._set_cached("nasdaq_earnings", {
        "mega_cap_today": 3, "total_today": 50,
    }, ttl=300)
    score, detail = scraper._score_nasdaq_earnings()
    assert score < 0, "Mega-cap earnings should produce defensive drag"


# ─── Integration: aggregation ────────────────────────────────────────────────

def test_aggregate_score_weighted():
    scraper = MacroRegimeScraper({"enabled": True})
    components = {
        "futures_sentiment": (0.5, {}),
        "macro_calendar": (-0.3, {}),
        "headline_sentiment": (0.0, {}),
        "earnings_risk": (0.0, {}),
        "order_depth_liq": (0.0, {}),
        "institutional_sentiment": (0.0, {}),
    }
    score = scraper._aggregate_score(components)
    # Futures has 0.30 weight, calendar 0.25 — net should be slightly positive
    assert -1.0 <= score <= 1.0


# ─── Integration: full posture with mocked sources ──────────────────────────

def test_full_posture_bearish_scenario():
    scraper = MacroRegimeScraper({"enabled": True, "refresh_seconds": 0})
    # Inject all caches to avoid network calls
    scraper._set_cached("finviz", {
        "sp500": -2.5, "nasdaq": -3.0, "btc": -8.0, "eth": -10.0,
        "vix": 20.0, "dxy": 2.0, "gold": 3.0,
    }, ttl=300)
    scraper._set_cached("forexfactory", [
        {"title": "FOMC", "impact": "High", "minutes_until": 15,
         "is_released": False, "actual": "", "forecast": "", "previous": ""},
    ], ttl=300)
    scraper._set_cached("financialjuice", [
        "Markets crash as panic spreads globally",
        "Recession confirmed by leading indicators",
    ], ttl=300)
    scraper._set_cached("nasdaq_earnings", {"mega_cap_today": 4, "total_today": 80}, ttl=300)
    scraper._set_cached("coinank_depth", {}, ttl=300)
    scraper._set_cached("coinank_liq", {}, ttl=300)
    scraper._set_cached("sentimentrader", {"available": False}, ttl=300)

    posture = scraper.get_risk_posture(force=True)
    assert posture["macro_risk_level"] in ("high", "extreme")
    assert posture["size_modifier"] < 1.0
    assert posture["confidence_drag"] < 0
    assert len(posture["components"]) > 0

def test_full_posture_bullish_scenario():
    scraper = MacroRegimeScraper({"enabled": True, "refresh_seconds": 0})
    scraper._set_cached("finviz", {
        "sp500": 1.5, "nasdaq": 2.0, "btc": 5.0, "eth": 6.0,
        "vix": -3.0, "dxy": -1.0, "gold": -0.5,
    }, ttl=300)
    scraper._set_cached("forexfactory", [], ttl=300)
    scraper._set_cached("financialjuice", [
        "Bull run continues with strong inflows",
        "Recovery accelerates across all sectors",
    ], ttl=300)
    scraper._set_cached("nasdaq_earnings", {"mega_cap_today": 0, "total_today": 10}, ttl=300)
    scraper._set_cached("coinank_depth", {}, ttl=300)
    scraper._set_cached("coinank_liq", {}, ttl=300)
    scraper._set_cached("sentimentrader", {"available": False}, ttl=300)

    posture = scraper.get_risk_posture(force=True)
    assert posture["macro_risk_level"] in ("low", "normal")
    assert posture["size_modifier"] >= 0.9


# ─── Integration: trading_cycle overlay ──────────────────────────────────────

def test_apply_macro_regime_overlay_stamps_regime_data():
    from src.core.cycles.trading_cycle import _apply_macro_regime_overlay

    mock_macro = MagicMock()
    mock_macro.get_risk_posture.return_value = {
        "macro_risk_level": "elevated",
        "macro_score": -0.1,
        "size_modifier": 0.70,
        "confidence_drag": -0.07,
        "block_new_entries": False,
        "reasons": ["futures_sentiment: bearish (-0.25)"],
    }
    container = MagicMock()
    container.macro_regime = mock_macro

    regime_data = {
        "overall_regime": "ranging",
        "strategy_guidance": {"size_modifier": 1.0, "pause": [], "activate": ["momentum"]},
    }

    result = _apply_macro_regime_overlay(container, regime_data)
    assert result["macro_risk_level"] == "elevated"
    assert result["macro_size_modifier"] == 0.70
    assert result["macro_confidence_drag"] == -0.07
    # Size modifier should be multiplied: 1.0 * 0.70 = 0.70
    assert result["strategy_guidance"]["size_modifier"] == 0.70

def test_apply_macro_regime_overlay_extreme_blocks():
    from src.core.cycles.trading_cycle import _apply_macro_regime_overlay

    mock_macro = MagicMock()
    mock_macro.get_risk_posture.return_value = {
        "macro_risk_level": "extreme",
        "macro_score": -0.6,
        "size_modifier": 0.20,
        "confidence_drag": -0.20,
        "block_new_entries": True,
        "reasons": ["futures_sentiment: bearish (-0.50)", "macro_calendar: bearish (-0.40)"],
    }
    container = MagicMock()
    container.macro_regime = mock_macro

    regime_data = {
        "overall_regime": "ranging",
        "strategy_guidance": {"size_modifier": 0.8, "pause": [], "activate": ["momentum"]},
    }

    result = _apply_macro_regime_overlay(container, regime_data)
    assert result["macro_block_new_entries"] is True
    assert "all" in result["strategy_guidance"]["pause"]

def test_apply_macro_regime_overlay_no_scraper():
    from src.core.cycles.trading_cycle import _apply_macro_regime_overlay

    container = MagicMock()
    container.macro_regime = None

    regime_data = {"overall_regime": "trending_up", "strategy_guidance": {"size_modifier": 1.0}}
    result = _apply_macro_regime_overlay(container, regime_data)
    assert "macro_risk_level" not in result  # Unchanged


# ─── Stats / dashboard ──────────────────────────────────────────────────────

def test_get_stats():
    scraper = MacroRegimeScraper({"enabled": False})
    stats = scraper.get_stats()
    assert "macro_risk_level" in stats
    assert "size_modifier" in stats

def test_get_dashboard_data():
    scraper = MacroRegimeScraper({"enabled": False})
    data = scraper.get_dashboard_data()
    assert isinstance(data, dict)
    assert "macro_risk_level" in data
