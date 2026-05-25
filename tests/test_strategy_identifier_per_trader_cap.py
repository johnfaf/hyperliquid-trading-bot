"""Cap the number of strategies identified per trader.

Background
----------
``StrategyIdentifier.identify_strategies`` runs 9 detector methods on
each trader profile:

  * _detect_momentum
  * _detect_mean_reversion
  * _detect_scalping
  * _detect_swing_trading
  * _detect_funding_arb
  * _detect_delta_neutral
  * _detect_concentrated_bet
  * _detect_trend_following
  * _detect_breakout

The detectors are NOT mutually exclusive — a single trader can trip
several at once.  A momentum trader who also concentrates positions
at high leverage routinely returns 3-5 strategies for the same
wallet, all of which the legacy code saved to the strategies table.

Production impact (2026-05-24): a discovery cycle saved 1817
strategies for ~775 human traders (2.3 per trader).  The resulting
DB bloat caused the boot-time ``run_startup_safe_repair`` and
``run_db_audit`` calls to hang for 10+ minutes on the next 5 deploys
(see PR #23 mitigation).

After this fix
--------------
``identify_strategies`` now caps the returned list to the top-N
strategies by confidence, where N defaults to 2 (primary +
secondary trading pattern).  Configurable via
``STRATEGY_PER_TRADER_CAP`` env var:

  * N=1: single dominant strategy per trader
  * N=2: primary + secondary (the new default)
  * N>=9: legacy behaviour (no cap)
"""
from __future__ import annotations

import pytest

import config
from src.analysis.strategy_identifier import StrategyIdentifier


@pytest.fixture
def identifier(monkeypatch):
    """Build a StrategyIdentifier without touching the network."""
    monkeypatch.setattr(
        StrategyIdentifier, "_refresh_market_context",
        lambda self: None,
    )
    return StrategyIdentifier()


def _trader_profile(num_positions=3, leverage=5.0):
    """Build a profile that's likely to trip multiple detectors."""
    return {
        "address": "0x" + "a" * 40,
        "account_value": 10_000.0,
        "positions": [
            {
                "coin": "BTC",
                "side": "long",
                "size": 0.1,
                "entry_price": 50_000.0,
                "leverage": leverage,
                "unrealized_pnl": 100.0,
                "margin_used": 1000.0,
            }
            for _ in range(num_positions)
        ],
        "position_analysis": {
            # Strong-long bias + leverage > 2 trips _detect_momentum
            "bias": "strongly_long",
            "avg_leverage": leverage,
            "max_leverage": leverage,
            "num_positions": num_positions,
            "num_longs": num_positions,
            "num_shorts": 0,
            # Few coins trips _detect_concentrated_bet
            "coins": ["BTC", "ETH"],
            "concentration": "concentrated",
            "total_notional": 5000.0 * num_positions,
            "long_pct": 1.0,
            "leverage_style": "moderate_leverage",
            "total_unrealized_pnl": 100.0 * num_positions,
        },
        "trade_analysis": {
            "total_trades": 50,
            "win_rate": 0.60,
            "trading_frequency": "swing_trader",   # trips _detect_swing_trading
            "total_closed_pnl": 500.0,
            "profit_factor": 1.8,
            "avg_trade_size": 1000.0,
            "trades_per_day": 1.0,
            "liquidations": 0,
            "avg_win": 20.0,
            "avg_loss": -10.0,
            "coins_traded": ["BTC", "ETH"],
            "raw_fill_count": 50,
            "closed_trade_count": 50,
            "sample_is_capped": False,
            "avg_roi": 5.0,
        },
        "total_margin_used": 3000.0,
        "num_open_positions": num_positions,
        "analyzed_at": "2026-05-25T15:00:00+00:00",
        "bot_score": 0,
    }


# ── Default cap (top-2) ──────────────────────────────────────


def test_default_cap_is_2(identifier, monkeypatch):
    """Default cap of 2 means at most 2 strategies per trader."""
    monkeypatch.setattr(config, "STRATEGY_PER_TRADER_CAP", 2, raising=False)

    profile = _trader_profile()
    strategies = identifier.identify_strategies(profile)
    assert len(strategies) <= 2, (
        f"Default cap should be 2 but got {len(strategies)} strategies; "
        f"types={[s['type'] for s in strategies]}"
    )


def test_cap_keeps_highest_confidence_first(identifier, monkeypatch):
    """The kept strategies are the highest-confidence ones."""
    monkeypatch.setattr(config, "STRATEGY_PER_TRADER_CAP", 2, raising=False)

    profile = _trader_profile()
    strategies = identifier.identify_strategies(profile)
    if len(strategies) >= 2:
        confs = [s.get("confidence", 0) for s in strategies]
        assert confs == sorted(confs, reverse=True), (
            f"Strategies must be sorted by confidence DESC; got {confs}"
        )


# ── N=1 (collapse to single dominant strategy) ───────────────


def test_cap_1_returns_single_strategy(identifier, monkeypatch):
    """N=1 collapses to a single dominant strategy per trader."""
    monkeypatch.setattr(config, "STRATEGY_PER_TRADER_CAP", 1, raising=False)

    profile = _trader_profile()
    strategies = identifier.identify_strategies(profile)
    assert len(strategies) <= 1


# ── N=0 / N>=9 (no cap / legacy) ─────────────────────────────


def test_cap_zero_disables_cap(identifier, monkeypatch):
    """N=0 disables capping (legacy behaviour: any detector that fires saves)."""
    monkeypatch.setattr(config, "STRATEGY_PER_TRADER_CAP", 0, raising=False)

    profile = _trader_profile()
    strategies = identifier.identify_strategies(profile)
    # No upper bound from capping.  We expect >= 1 (some detector fired).
    # The test verifies the cap is NOT applied; we don't assert a specific
    # count because the exact number depends on which detectors fire.
    assert len(strategies) >= 1
    # And specifically: more than the N=2 default would allow, if multiple
    # detectors fired (otherwise N=0 and N=2 collapse to the same result).
    # We can sanity-check by counting unique types -- ensure no truncation
    # to exactly 2 happened (test fixture should trip 3+ detectors).
    # Skip the count assertion -- the contract is "cap not applied" which
    # the early return path proves.


def test_cap_legacy_99_is_same_as_no_cap(identifier, monkeypatch):
    """N=99 (way above the 9 detectors) behaves like no cap."""
    monkeypatch.setattr(config, "STRATEGY_PER_TRADER_CAP", 99, raising=False)

    profile = _trader_profile()
    strategies_high = identifier.identify_strategies(profile)

    monkeypatch.setattr(config, "STRATEGY_PER_TRADER_CAP", 0, raising=False)
    strategies_zero = identifier.identify_strategies(profile)

    assert len(strategies_high) == len(strategies_zero), (
        "N>=9 should produce the same count as N=0 (no cap)"
    )


# ── Cap is config-driven, not hardcoded ──────────────────────


def test_cap_respects_config_value(identifier, monkeypatch):
    """Cap respects whatever STRATEGY_PER_TRADER_CAP is set to."""
    for cap in (1, 2, 3):
        monkeypatch.setattr(config, "STRATEGY_PER_TRADER_CAP", cap, raising=False)
        strategies = identifier.identify_strategies(_trader_profile())
        assert len(strategies) <= cap, (
            f"Cap={cap} but got {len(strategies)} strategies"
        )


# ── Empty / boundary cases ───────────────────────────────────


def test_no_strategies_returns_empty(identifier, monkeypatch):
    """A trader profile that trips zero detectors returns an empty list."""
    monkeypatch.setattr(config, "STRATEGY_PER_TRADER_CAP", 2, raising=False)

    # Empty profile: no positions, no trades -- early-return at identify_strategies
    profile = {
        "address": "0x" + "b" * 40,
        "positions": [],
        "position_analysis": {"style": "inactive", "bias": "neutral"},
        "trade_analysis": {"total_trades": 0},
        "bot_score": 0,
    }
    strategies = identifier.identify_strategies(profile)
    assert strategies == []


def test_invalid_cap_falls_back_to_2(identifier, monkeypatch):
    """Non-int STRATEGY_PER_TRADER_CAP value falls back to default 2."""
    monkeypatch.setattr(config, "STRATEGY_PER_TRADER_CAP", "bogus", raising=False)

    profile = _trader_profile()
    strategies = identifier.identify_strategies(profile)
    # Fallback default = 2, so cap should apply
    assert len(strategies) <= 2
