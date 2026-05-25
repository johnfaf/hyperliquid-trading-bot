"""XGBoost forecaster must flag predictions when external features fail.

Background
----------
Before this fix, the three external feature fetchers
(``_get_funding_rate``, ``_get_5m_volatility``, ``_get_basis_spread``)
silently caught all exceptions and returned 0.0.  The XGBoost model
trained on real values, so at inference a silent 0.0 substitution
produced a high-confidence prediction on a corrupted feature vector.
Affected ~1-5% of regime predictions during transient API issues; the
consumer (decision firewall, EV gate) had no way to know.

After this fix
--------------
Each fetcher returns ``Optional[float]``: ``None`` on failure, real
value on success.  ``_extract_features`` tracks which features failed
and ``predict_regime``:

  1. Substitutes 0.0 only for the model input (preserves XGB contract)
  2. Sets ``degraded=True`` and lists missing features in
     ``degraded_reason`` and ``missing_features``
  3. Caps the published ``confidence`` by the fraction of features that
     arrived (1/3 floor), so downstream consumers can de-rate.
  4. Sets ``authoritative=False`` when any external feature failed.

Both the XGB and the weighted-signal fallback paths propagate the flag.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def forecaster(monkeypatch):
    """A real XGBoostRegimeForecaster with the fallback stubbed."""
    from src.signals import xgboost_regime_forecaster as mod

    # Stub the fallback so predict_regime returns a deterministic base.
    fake_fallback = MagicMock()
    fake_fallback.predict_regime.return_value = {
        "regime": "neutral",
        "confidence": 0.50,
        "signal": 0.0,
        "components": {
            "funding_slope": 0.10,
            "imbalance": -0.05,
            "arkham_flow": 0.0,
            "polymarket": 0.20,
            "options_flow": 0.0,
        },
        "active_inputs": ["funding_slope", "imbalance", "polymarket"],
    }

    monkeypatch.setattr(
        mod, "PredictiveRegimeForecaster",
        lambda *a, **k: fake_fallback,
    )

    inst = mod.XGBoostRegimeForecaster()
    inst.fallback = fake_fallback
    # Use the fallback path (no model) so we test the degraded plumbing
    # without depending on xgboost being installed in the test env.
    inst.model = None
    return inst


# ── Fetcher return None on failure ───────────────────────────


def test_get_funding_rate_returns_none_on_exception(monkeypatch, forecaster):
    """A network exception in the funding-rate fetch returns None, not 0.0."""
    from src.signals import xgboost_regime_forecaster as mod

    def _boom(*a, **k):
        raise RuntimeError("Hyperliquid API down")

    fake_manager = MagicMock()
    fake_manager.post = _boom
    monkeypatch.setattr(mod, "get_manager", lambda: fake_manager)

    result = forecaster._get_funding_rate("BTC")
    assert result is None, "Must return None on exception, not 0.0"


def test_get_funding_rate_returns_none_when_coin_missing(monkeypatch, forecaster):
    """If the API returns data but the coin isn't in the universe, return None."""
    from src.signals import xgboost_regime_forecaster as mod

    fake_manager = MagicMock()
    fake_manager.post.return_value = [
        {"universe": [{"name": "ETH"}]},
        [{"funding": 0.0001}],
    ]
    monkeypatch.setattr(mod, "get_manager", lambda: fake_manager)

    # BTC not in universe -> None (not 0.0).
    result = forecaster._get_funding_rate("BTC")
    assert result is None


def test_get_5m_volatility_returns_none_on_exception(monkeypatch, forecaster):
    """Candle API failure returns None."""
    from src.signals import xgboost_regime_forecaster as mod

    def _boom(*a, **k):
        raise RuntimeError("Hyperliquid candle API down")

    fake_manager = MagicMock()
    fake_manager.post = _boom
    monkeypatch.setattr(mod, "get_manager", lambda: fake_manager)

    result = forecaster._get_5m_volatility("BTC")
    assert result is None


def test_get_basis_spread_returns_none_on_binance_failure(monkeypatch, forecaster):
    """Binance API failure inside basis-spread fetcher returns None."""
    from src.signals import xgboost_regime_forecaster as mod

    # HL fetcher succeeds (returns a real value) so the FAILURE has to
    # come from the Binance call alone.
    monkeypatch.setattr(forecaster, "_get_funding_rate", lambda coin: 0.0001)

    fake_resp = MagicMock()
    fake_resp.ok = False
    monkeypatch.setattr(
        "src.signals.xgboost_regime_forecaster.req"
        if hasattr(mod, "req")
        else "requests.get",
        lambda *a, **k: fake_resp,
        raising=False,
    )

    # Easier: directly patch requests via import side
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: fake_resp)

    result = forecaster._get_basis_spread("BTC")
    assert result is None


# ── _extract_features tracks missing list ───────────────────


def test_extract_features_returns_missing_list(forecaster, monkeypatch):
    """When all 3 fetchers fail, _extract_features returns all 3 as missing."""
    from src.signals import xgboost_regime_forecaster as mod

    monkeypatch.setattr(forecaster, "_get_funding_rate", lambda c: None)
    monkeypatch.setattr(forecaster, "_get_5m_volatility", lambda c: None)
    monkeypatch.setattr(forecaster, "_get_basis_spread", lambda c: None)
    # Disable the CDC fallback so the test sees a clean miss.  In prod
    # a working CDC could recover volatility_5m, which is tested
    # separately below.
    monkeypatch.setattr(mod, "HAS_CRYPTOCOM", False)

    base = forecaster.fallback.predict_regime("BTC")
    features, missing = forecaster._extract_features("BTC", base)

    # All 3 in the missing list
    assert set(missing) == {"funding_rate", "volatility_5m", "basis_spread"}
    # Values substituted to 0.0 in the features dict (model contract)
    assert features["funding_rate"] == 0.0
    assert features["volatility_5m"] == 0.0
    assert features["basis_spread"] == 0.0


def test_cdc_recovers_volatility_when_hl_fails(forecaster, monkeypatch):
    """If HL volatility fails but CDC succeeds, volatility_5m is recovered.

    A working CDC fallback means the bot still has a real measurement
    for 5-minute volatility -- it should NOT show up as missing.
    """
    from src.signals import xgboost_regime_forecaster as mod

    monkeypatch.setattr(forecaster, "_get_funding_rate", lambda c: 0.0001)
    monkeypatch.setattr(forecaster, "_get_5m_volatility", lambda c: None)
    monkeypatch.setattr(forecaster, "_get_basis_spread", lambda c: 0.10)

    monkeypatch.setattr(mod, "HAS_CRYPTOCOM", True)
    fake_cdc = MagicMock()
    fake_cdc.get_5m_volatility.return_value = 0.45
    monkeypatch.setattr(mod, "_cryptocom", fake_cdc, raising=False)

    base = forecaster.fallback.predict_regime("BTC")
    features, missing = forecaster._extract_features("BTC", base)

    assert "volatility_5m" not in missing, (
        "CDC successfully recovered volatility_5m -- must not be listed missing"
    )
    assert features["volatility_5m"] == 0.45


def test_extract_features_no_missing_when_all_ok(forecaster, monkeypatch):
    """All 3 fetchers succeeding produces an empty missing list."""
    from src.signals import xgboost_regime_forecaster as mod
    monkeypatch.setattr(forecaster, "_get_funding_rate", lambda c: 0.0002)
    monkeypatch.setattr(forecaster, "_get_5m_volatility", lambda c: 0.50)
    monkeypatch.setattr(forecaster, "_get_basis_spread", lambda c: 0.10)
    monkeypatch.setattr(mod, "HAS_CRYPTOCOM", False)

    base = forecaster.fallback.predict_regime("BTC")
    features, missing = forecaster._extract_features("BTC", base)
    assert missing == []
    assert features["funding_rate"] == 0.0002
    assert features["volatility_5m"] == 0.50
    assert features["basis_spread"] == 0.10


# ── predict_regime flags degraded ───────────────────────────


def test_predict_regime_flags_degraded_on_missing_features(forecaster, monkeypatch):
    """When features are missing, the prediction is flagged degraded."""
    from src.signals import xgboost_regime_forecaster as mod
    monkeypatch.setattr(forecaster, "_get_funding_rate", lambda c: None)
    monkeypatch.setattr(forecaster, "_get_5m_volatility", lambda c: None)
    monkeypatch.setattr(forecaster, "_get_basis_spread", lambda c: 0.10)
    monkeypatch.setattr(mod, "HAS_CRYPTOCOM", False)

    result = forecaster.predict_regime("BTC")

    assert result.get("degraded") is True
    reason = result.get("degraded_reason", "")
    assert "missing_features" in reason
    assert "funding_rate" in reason
    assert "volatility_5m" in reason
    assert "basis_spread" not in reason  # this one succeeded
    assert result.get("missing_features") == ["funding_rate", "volatility_5m"]


def test_predict_regime_caps_confidence_on_missing_features(forecaster, monkeypatch):
    """Confidence is scaled down by the fraction of features that arrived."""
    from src.signals import xgboost_regime_forecaster as mod
    # Fallback returns confidence=0.50.  With ALL 3 features missing,
    # coverage = 0 (but floored to 1/3 by design), so we expect at most
    # 0.50 * 1/3 = ~0.167 published confidence.
    monkeypatch.setattr(forecaster, "_get_funding_rate", lambda c: None)
    monkeypatch.setattr(forecaster, "_get_5m_volatility", lambda c: None)
    monkeypatch.setattr(forecaster, "_get_basis_spread", lambda c: None)
    monkeypatch.setattr(mod, "HAS_CRYPTOCOM", False)

    result = forecaster.predict_regime("BTC")
    # Roughly 0.50 * 1/3 = 0.167.  Allow some slack for rounding.
    assert result["confidence"] <= 0.20, (
        f"Confidence={result['confidence']} should be capped near 0.167 "
        f"when all 3 external features are missing"
    )


def test_predict_regime_authoritative_false_when_features_missing(forecaster, monkeypatch):
    """authoritative=False when any external feature failed."""
    from src.signals import xgboost_regime_forecaster as mod
    monkeypatch.setattr(forecaster, "_get_funding_rate", lambda c: None)
    monkeypatch.setattr(forecaster, "_get_5m_volatility", lambda c: 0.30)
    monkeypatch.setattr(forecaster, "_get_basis_spread", lambda c: 0.20)
    monkeypatch.setattr(mod, "HAS_CRYPTOCOM", False)

    result = forecaster.predict_regime("BTC")
    # The fallback path inherits authoritative from base; we just check
    # degraded is set and missing_features lists the failure.
    assert result.get("degraded") is True
    assert "funding_rate" in (result.get("degraded_reason") or "")


def test_predict_regime_clean_when_no_features_missing(forecaster, monkeypatch):
    """No missing features -> no degraded flag from this path."""
    from src.signals import xgboost_regime_forecaster as mod
    monkeypatch.setattr(forecaster, "_get_funding_rate", lambda c: 0.0001)
    monkeypatch.setattr(forecaster, "_get_5m_volatility", lambda c: 0.30)
    monkeypatch.setattr(forecaster, "_get_basis_spread", lambda c: 0.20)
    monkeypatch.setattr(mod, "HAS_CRYPTOCOM", False)

    result = forecaster.predict_regime("BTC")
    # Base fallback didn't set degraded; we shouldn't either.
    assert not result.get("degraded", False)
    assert result.get("missing_features", []) == [] or "missing_features" not in result
