"""Data readiness gate tests."""
from __future__ import annotations

import pytest

import config
from src.signals import data_readiness
from src.signals.data_readiness import assess_signal_readiness, is_signal_data_ready


class _FakeSide:
    def __init__(self, v): self.value = v


def _signal(*, coin="BTC", features=None, source_health=None):
    s = type("Sig", (), {})()
    s.coin = coin
    s.side = _FakeSide("long")
    ctx = {}
    if features is not None:
        ctx["features"] = features
    if source_health is not None:
        ctx["source_health"] = source_health
    s.context = ctx
    return s


@pytest.fixture(autouse=True)
def _enable_and_stub(monkeypatch):
    monkeypatch.setattr(config, "DATA_READINESS_GATE_ENABLED", True, raising=False)
    monkeypatch.setattr(
        config, "DATA_READINESS_REQUIRED_COMPONENTS",
        "candles,funding,spread,feature_vector",
        raising=False,
    )
    # Stub external data sources to controlled defaults; each test
    # overrides what it cares about.
    monkeypatch.setattr(
        "src.data.feature_store.get_candles",
        lambda coin, tf, limit=200: [{"c": 100.0}] * 6,
    )
    monkeypatch.setattr(
        "src.data.hyperliquid_client.get_asset_contexts",
        lambda: {"BTC": {"funding": 0.0001, "open_interest": 1000.0}},
    )
    monkeypatch.setattr(
        "src.data.hyperliquid_client.get_all_mids",
        lambda: {"BTC": 67000.0},
    )


def test_full_readiness_passes():
    features = {
        "rsi": 55.0, "volatility": 0.02, "volume_ratio": 1.1,
        "trend_strength": 0.5, "funding_rate": 0.0001,
    }
    sig = _signal(features=features)
    ok, reason = is_signal_data_ready(sig)
    assert ok, reason
    assert reason == "ok"


def test_disabled_gate_passes_any_signal(monkeypatch):
    monkeypatch.setattr(config, "DATA_READINESS_GATE_ENABLED", False, raising=False)
    sig = _signal()
    ok, reason = is_signal_data_ready(sig)
    assert ok and reason == "gate_disabled"


def test_missing_candles_blocks(monkeypatch):
    monkeypatch.setattr("src.data.feature_store.get_candles", lambda *a, **k: [])
    features = {"rsi": 55.0, "volatility": 0.02, "volume_ratio": 1.1}
    sig = _signal(features=features)
    ok, reason = is_signal_data_ready(sig)
    assert not ok
    assert "candles" in reason


def test_missing_feature_vector_blocks():
    sig = _signal(features=None)  # no features dict
    ok, reason = is_signal_data_ready(sig)
    assert not ok
    assert "feature_vector" in reason


def test_sparse_feature_vector_blocks():
    sig = _signal(features={"rsi": 55.0})  # only 1 feature, MIN_OVERLAP=3
    ok, reason = is_signal_data_ready(sig)
    assert not ok
    assert "feature_vector" in reason


def test_payload_shape():
    features = {"rsi": 55.0, "volatility": 0.02, "volume_ratio": 1.1}
    sig = _signal(features=features)
    payload = assess_signal_readiness(sig)
    assert payload["ready"] in (True, False)
    assert "details" in payload
    assert "required" in payload
    for component in payload["required"]:
        assert component in payload["details"]
