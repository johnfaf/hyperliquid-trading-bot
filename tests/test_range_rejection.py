from dataclasses import replace

import pytest

from src.backtest import candle_entry_replay
from src.backtest.candle_exit_replay import BAR_MS, Bar, Decision
from src.backtest.range_rejection_replay import range_rejection_decisions
from src.signals.range_rejection import failed_break_side


@pytest.mark.parametrize("high, low, close, volume_ratio, expected", [
    (111, 100, 108, 1.5, -1),  # Rejection of the upper edge.
    (100, 89, 92, 2, 1),      # Rejection of the lower edge.
    (111, 100, 109.5, 2, 0),  # Shallow reentry.
    (111, 100, 110.1, 2, 0),  # Closed outside the range.
    (111, 89, 105, 2, 0),     # Both edges crossed; no directional evidence.
    (111, 89.9, 108, 1.49, 0),
    (110.2, 100, 108, 3, 0),  # Wick did not clear the ATR buffer.
])
def test_rejection_requires_one_clear_boundary_and_high_volume(high, low, close, volume_ratio, expected):
    assert failed_break_side(high, low, close, range_high=110, range_low=90,
                             prior_atr=4, volume_ratio=volume_ratio) == expected


def test_features_exclude_current_bar_from_references_and_ignore_future_data():
    original = [Bar(i * BAR_MS, 100, 101, 99, 100, 10) for i in range(110)]
    original[96] = replace(original[96], high=103, volume=20)
    changed = [b if i <= 96 else replace(b, high=400, close=300, volume=1000) for i, b in enumerate(original)]
    signals, ratios = range_rejection_decisions(original)
    later_signals, later_ratios = range_rejection_decisions(changed)
    assert not any(d.side for d in signals[:96])
    assert signals[96].side == -1 and ratios[96] == 2
    assert signals[96].atr_pct == .02
    assert signals[:97] == later_signals[:97] and ratios[:97] == later_ratios[:97]


def test_matched_control_has_the_same_high_volume_flag_and_mature_label(monkeypatch):
    monkeypatch.setattr(candle_entry_replay, "LOOKBACK_MS", 12 * BAR_MS)
    bars = [Bar(i * BAR_MS, 100, 101, 99, 100, 10) for i in range(27)]
    decisions = [Decision(0, 0, None, "choppy") for _ in bars]
    decisions[4] = Decision(0, 0, .011, "choppy")
    decisions[6] = Decision(0, 0, .010, "choppy")  # Closer ATR, but normal volume.
    decisions[9] = Decision(0, 0, .010, "choppy")  # High volume, outcome unavailable.
    decisions[12] = Decision(-1, 0, .010, "choppy")
    volume_strata = [i in (4, 9, 12) for i in range(len(bars))]
    result, _ = candle_entry_replay.match_entries(
        bars, decisions, 13 * BAR_MS, 21 * BAR_MS, 4, set(), strata=volume_strata,
    )
    assert result == [candle_entry_replay.Match(12, 4)]
