"""Confluence gate: require N independent confirmations for non-copy entries
(signal #3)."""
from __future__ import annotations

from types import SimpleNamespace

from src.signals.confluence import confluence_ok, count_confirmations


def _sig(source="strategy", strategy_type="momentum_long",
         ofa=False, vc=False, ctx=None):
    return SimpleNamespace(source=source, strategy_type=strategy_type,
                           options_flow_aligned=ofa, volume_confirmed=vc,
                           context=ctx or {})


def test_count_confirmations():
    assert count_confirmations(_sig()) == 0
    assert count_confirmations(_sig(ofa=True, vc=True)) == 2
    assert count_confirmations(
        _sig(ofa=True, ctx={"regime_aligned": True, "cross_venue_confirmed": True})) == 3


def test_off_when_threshold_zero():
    assert confluence_ok(_sig(), 0)[0] is True


def test_blocks_lone_non_copy_signal():
    ok, why = confluence_ok(_sig(ofa=True), 2)   # only 1 confirmation < 2
    assert ok is False and "confluence" in why


def test_passes_with_enough_confirmations():
    assert confluence_ok(_sig(ofa=True, vc=True), 2)[0] is True


def test_copy_trade_is_exempt():
    # copy_trade is the proven edge -> never gated, even with 0 confirmations
    assert confluence_ok(_sig(source="copy_trade"), 3)[0] is True
