"""Copy-slippage / replication-lag telemetry (signal #2).

_copy_lag_bps measures how much worse our copy entry is than the source trader's
entry, signed so positive = adverse (we chased the move). Observe-only metadata;
no behaviour change.
"""
from __future__ import annotations

from src.trading.copy_trader import CopyTrader


def test_long_chased_is_adverse_positive():
    # we entered at 101 vs source 100 on a long -> paid 100 bps worse
    assert CopyTrader._copy_lag_bps(101.0, 100.0, "long") == 100.0


def test_long_better_entry_is_favorable_negative():
    assert CopyTrader._copy_lag_bps(99.0, 100.0, "long") == -100.0


def test_short_lower_fill_is_adverse_positive():
    # short: selling at 99 vs source 100 is worse -> positive (adverse)
    assert CopyTrader._copy_lag_bps(99.0, 100.0, "short") == 100.0


def test_short_higher_fill_is_favorable_negative():
    assert CopyTrader._copy_lag_bps(101.0, 100.0, "short") == -100.0


def test_invalid_inputs_zero():
    assert CopyTrader._copy_lag_bps(0, 100, "long") == 0.0
    assert CopyTrader._copy_lag_bps(100, 0, "long") == 0.0
    assert CopyTrader._copy_lag_bps("x", 100, "long") == 0.0
