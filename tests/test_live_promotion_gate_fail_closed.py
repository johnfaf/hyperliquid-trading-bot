"""The legacy scoring and firewall flags cannot authorize real entries."""
from types import SimpleNamespace

import pytest

from src.trading.lighter_live_trader import LighterLiveTrader
from src.trading.live_trader import LiveTrader


@pytest.mark.parametrize("trader", [LiveTrader, LighterLiveTrader])
@pytest.mark.parametrize("bypass", [False, True])
def test_real_adapters_block_before_reading_candidate_or_contacting_venue(monkeypatch, trader, bypass):
    monkeypatch.setenv("LIVE_PROMOTION_GATE_ENABLED", "false")
    monkeypatch.setenv("LIVE_PROMOTION_GATE_FAIL_OPEN", "1")
    monkeypatch.setenv("PROMOTION_BOOTSTRAP_TIER_ENABLED", "true")
    # The stub intentionally has no execution, wallet or network methods.
    assert trader.execute_signal(SimpleNamespace(dry_run=False), object(), bypass_firewall=bypass) is None
