"""Funding vs price divergence safety brake tests."""
from __future__ import annotations

import pytest

import config
from src.signals import funding_divergence
from src.signals.funding_divergence import (
    get_market_divergence,
    reset_cache_for_tests,
    should_block_side,
)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    reset_cache_for_tests()
    monkeypatch.setattr(config, "FUNDING_DIVERGENCE_ENABLED", True, raising=False)
    monkeypatch.setattr(
        config, "FUNDING_DIVERGENCE_FUNDING_THRESHOLD", 0.00015, raising=False
    )
    monkeypatch.setattr(
        config, "FUNDING_DIVERGENCE_PRICE_DEV_THRESHOLD", 0.005, raising=False
    )
    monkeypatch.setattr(
        config, "FUNDING_DIVERGENCE_CACHE_TTL_S", 0.001, raising=False
    )
    yield
    reset_cache_for_tests()


def _patch_market(monkeypatch, funding: dict, closes_by_coin: dict) -> None:
    monkeypatch.setattr(
        funding_divergence, "_fetch_funding_rates", lambda: dict(funding)
    )
    monkeypatch.setattr(
        funding_divergence,
        "_recent_closes",
        lambda coin, n=4: list(closes_by_coin.get(coin, [])),
    )


def test_block_longs_when_crowded_into_selloff(monkeypatch):
    """BTC + ETH both: funding > threshold AND price < 4h MA × (1-deviation)."""
    closes = {
        "BTC": [70_000, 69_800, 69_500, 68_500],
        "ETH": [3500, 3490, 3470, 3400],
    }
    funding = {"BTC": 0.0003, "ETH": 0.0004}
    _patch_market(monkeypatch, funding, closes)

    payload = get_market_divergence(force_refresh=True)
    assert payload["side_to_block"] == "long"
    assert payload["confidence"] >= 0.55

    block, reason = should_block_side("long")
    assert block is True
    assert "funding_divergence_blocks_long" in reason

    block_short, _ = should_block_side("short")
    assert block_short is False


def test_block_shorts_when_crowded_into_rally(monkeypatch):
    """Symmetric: negative funding and price above 4h MA."""
    closes = {
        "BTC": [60_000, 60_500, 61_000, 62_000],
        "ETH": [3000, 3030, 3060, 3100],
    }
    funding = {"BTC": -0.00030, "ETH": -0.00050}
    _patch_market(monkeypatch, funding, closes)

    block, reason = should_block_side("short")
    assert block is True
    assert "funding_divergence_blocks_short" in reason


def test_no_block_when_funding_aligns_with_price(monkeypatch):
    """Funding positive + price rising = no divergence."""
    closes = {
        "BTC": [60_000, 60_500, 61_000, 62_000],
        "ETH": [3000, 3030, 3060, 3100],
    }
    funding = {"BTC": 0.0003, "ETH": 0.0004}
    _patch_market(monkeypatch, funding, closes)
    block, reason = should_block_side("long")
    assert block is False
    assert reason == "no_divergence"


def test_disabled_gate_passes_through(monkeypatch):
    monkeypatch.setattr(config, "FUNDING_DIVERGENCE_ENABLED", False, raising=False)
    block, reason = should_block_side("long")
    assert block is False
    assert reason == "gate_disabled"


def test_non_directional_side_passes(monkeypatch):
    block, reason = should_block_side("neutral")
    assert block is False
    assert reason == "non_directional"


def test_partial_market_data_blocks_at_lower_confidence(monkeypatch):
    """Only one of BTC/ETH showing divergence gives 0.55 confidence."""
    closes = {
        # BTC has clean divergence; ETH has no candles (insufficient data)
        "BTC": [70_000, 69_800, 69_500, 68_500],
        "ETH": [],
    }
    funding = {"BTC": 0.0003, "ETH": 0.0004}
    _patch_market(monkeypatch, funding, closes)

    payload = get_market_divergence(force_refresh=True)
    assert payload["side_to_block"] == "long"
    assert payload["confidence"] == pytest.approx(0.55, abs=1e-6)


def test_no_block_when_funding_below_threshold(monkeypatch):
    """Funding at exchange baseline (~0) shouldn't trigger even with price drop."""
    closes = {
        "BTC": [70_000, 69_800, 69_500, 68_500],
        "ETH": [3500, 3490, 3470, 3400],
    }
    funding = {"BTC": 0.00005, "ETH": 0.00005}
    _patch_market(monkeypatch, funding, closes)
    block, _ = should_block_side("long")
    assert block is False
