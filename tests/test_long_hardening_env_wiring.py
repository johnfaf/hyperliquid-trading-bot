"""LONG_HARDENING_* env wiring.

The long-side throttle in ``decision_firewall._apply_long_hardening``
defaults to ON with hard-coded thresholds. Without env wiring there
was no way to disable it when the rolling-12-trade window is
contaminated (e.g. by a deploy thrash) and the operator wants to
let new longs back through to refresh the sample.

This test suite locks the wiring: every ``LONG_HARDENING_*`` env var
must reach the firewall through ``build_subsystems`` -> ``DecisionFirewall(cfg)``,
with the same defaults as the raw firewall init.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import config
from src.signals.decision_firewall import DecisionFirewall


@pytest.fixture
def base_cfg():
    """Minimal cfg dict accepted by DecisionFirewall.__init__."""
    return {
        "forecaster": None,
        "agent_scorer": None,
        "event_scanner": None,
    }


def test_default_long_hardening_state(base_cfg):
    """No env override -> defaults match the firewall's hard-coded init."""
    fw = DecisionFirewall(base_cfg)
    assert fw.long_hardening_enabled is True
    assert fw.long_hardening_lookback_trades == 120
    assert fw.long_hardening_min_closed_trades == 12
    assert fw.long_hardening_degrade_win_rate == pytest.approx(0.48)
    assert fw.long_hardening_block_win_rate == pytest.approx(0.40)
    assert fw.long_hardening_block_net_pnl == pytest.approx(-0.5)
    assert fw.long_hardening_confidence_multiplier == pytest.approx(0.80)
    assert fw.long_hardening_size_multiplier == pytest.approx(0.50)


def test_long_hardening_enabled_false_disables_gate(base_cfg):
    """The override path operators care about: setting enabled=False
    must short-circuit the gate so a contaminated recent-trade window
    cannot block new longs."""
    cfg = {**base_cfg, "long_hardening_enabled": False}
    fw = DecisionFirewall(cfg)
    assert fw.long_hardening_enabled is False
    # _apply_long_hardening's first branch returns (True, "") when
    # enabled is False, regardless of the underlying policy.
    from src.signals.signal_schema import TradeSignal, SignalSide, SignalSource
    fake_signal = TradeSignal(
        coin="BTC",
        side=SignalSide.LONG,
        confidence=0.5,
        source=SignalSource.COPY_TRADE,
        reason="test",
        size=1.0,
        entry_price=50_000,
    )
    ok, reason = fw._apply_long_hardening(fake_signal)
    assert ok is True
    assert reason == ""


def test_all_long_hardening_knobs_are_overridable(base_cfg):
    """Every cfg key DecisionFirewall reads is configurable from outside."""
    overrides = {
        "long_hardening_enabled": False,
        "long_hardening_lookback_trades": 50,
        "long_hardening_min_closed_trades": 5,
        "long_hardening_degrade_win_rate": 0.35,
        "long_hardening_block_win_rate": 0.20,
        "long_hardening_block_net_pnl": -10.0,
        "long_hardening_confidence_multiplier": 0.60,
        "long_hardening_size_multiplier": 0.25,
    }
    fw = DecisionFirewall({**base_cfg, **overrides})
    assert fw.long_hardening_enabled is False
    assert fw.long_hardening_lookback_trades == 50
    assert fw.long_hardening_min_closed_trades == 5
    assert fw.long_hardening_degrade_win_rate == pytest.approx(0.35)
    assert fw.long_hardening_block_win_rate == pytest.approx(0.20)
    assert fw.long_hardening_block_net_pnl == pytest.approx(-10.0)
    assert fw.long_hardening_confidence_multiplier == pytest.approx(0.60)
    assert fw.long_hardening_size_multiplier == pytest.approx(0.25)


def test_long_hardening_lookback_floored_to_10(base_cfg):
    """The firewall enforces a 10-trade minimum lookback as a sanity floor.
    Confirm the cfg pass-through doesn't accidentally let smaller values
    through; same protection as the short-side knob."""
    fw = DecisionFirewall({**base_cfg, "long_hardening_lookback_trades": 5})
    assert fw.long_hardening_lookback_trades == 10  # floored


# ── Env var -> config -> cfg dict propagation ──


def test_env_vars_exist_in_config_module():
    """Each LONG_HARDENING_* env var must be readable from config so that
    subsystem_registry can copy it into the cfg dict."""
    assert hasattr(config, "LONG_HARDENING_ENABLED")
    assert hasattr(config, "LONG_HARDENING_LOOKBACK_TRADES")
    assert hasattr(config, "LONG_HARDENING_MIN_CLOSED_TRADES")
    assert hasattr(config, "LONG_HARDENING_DEGRADE_WIN_RATE")
    assert hasattr(config, "LONG_HARDENING_BLOCK_WIN_RATE")
    assert hasattr(config, "LONG_HARDENING_BLOCK_NET_PNL")
    assert hasattr(config, "LONG_HARDENING_CONFIDENCE_MULTIPLIER")
    assert hasattr(config, "LONG_HARDENING_SIZE_MULTIPLIER")


def test_subsystem_registry_wires_long_hardening_to_firewall_cfg():
    """The actual wire we care about: the cfg dict built in
    subsystem_registry.py for DecisionFirewall must include every
    ``long_hardening_*`` key, so operator env overrides reach the gate.

    Source-level structural test -- the firewall is created via a lazy
    local import inside ``build_subsystems`` so we can't easily stub it.
    Reading the module source proves each cfg key is wired to its env
    var, which is what we'd otherwise verify by booting the bot.
    """
    import inspect
    from src.core import subsystem_registry as sr

    source = inspect.getsource(sr)
    expected = {
        "LONG_HARDENING_ENABLED":               "long_hardening_enabled",
        "LONG_HARDENING_LOOKBACK_TRADES":       "long_hardening_lookback_trades",
        "LONG_HARDENING_MIN_CLOSED_TRADES":     "long_hardening_min_closed_trades",
        "LONG_HARDENING_DEGRADE_WIN_RATE":      "long_hardening_degrade_win_rate",
        "LONG_HARDENING_BLOCK_WIN_RATE":        "long_hardening_block_win_rate",
        "LONG_HARDENING_BLOCK_NET_PNL":         "long_hardening_block_net_pnl",
        "LONG_HARDENING_CONFIDENCE_MULTIPLIER": "long_hardening_confidence_multiplier",
        "LONG_HARDENING_SIZE_MULTIPLIER":       "long_hardening_size_multiplier",
    }
    for env_name, cfg_key in expected.items():
        assert env_name in source, (
            f"{env_name} not referenced in subsystem_registry"
        )
        assert f'"{cfg_key}"' in source, (
            f"{cfg_key} cfg-dict key missing in subsystem_registry"
        )
