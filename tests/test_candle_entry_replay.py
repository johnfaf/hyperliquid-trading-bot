from dataclasses import replace

import pytest

from scripts.run_hyperliquid_entry_replay import date_ms, windows
from src.backtest import candle_entry_replay as entry
from src.backtest.candle_exit_replay import BAR_MS, HOUR_MS, Bar, Decision


def bar(i, price=100.):
    return Bar(i * BAR_MS, price, price, price, price, 1.)


def decision(side=0, atr=None):
    return Decision(side, .55 if side else 0., atr, "choppy")


def test_matching_uses_only_mature_past_neutral_bars_of_same_stratum(monkeypatch):
    monkeypatch.setattr(entry, "LOOKBACK_MS", 12 * BAR_MS)
    bars = [bar(i) for i in range(28)]
    decisions = [decision() for _ in bars]
    decisions[3] = decision(0, .05)  # Outside the volatility tolerance.
    decisions[5] = decision(0, .011)
    decisions[6] = decision(0, .01)
    decisions[10] = decision(0, .01)  # Not mature yet.
    decisions[12] = decision(1, .01)
    decisions[14] = decision(1, .01)  # Overlaps the first signal.
    decisions[18] = decision(0, .01)
    used = set()
    matched, counts = entry.match_entries(bars, decisions, 13 * BAR_MS, 22 * BAR_MS, 4, used)
    assert matched == [entry.Match(12, 6)]
    assert counts["overlapping_signals"] == 1 and counts["matched"] == 1
    assert used == set(range(7, 12))
    assert entry.match_entries(bars, decisions, 13 * BAR_MS, 22 * BAR_MS, 4, used)[0] == [entry.Match(12, None)]


def test_a_filtered_signal_does_not_crowd_out_a_later_eligible_signal():
    bars = [bar(i) for i in range(20)]
    decisions = [decision() for _ in bars]
    decisions[3] = replace(decision(1, .01), confidence=.51)
    decisions[4] = decision(1, .01)
    assert entry.eligible_entries(bars, decisions, 0, 10 * BAR_MS, 4)[0] == [3]
    picked, counts = entry.eligible_entries(bars, decisions, 0, 10 * BAR_MS, 4, entry.FILTERS["confidence_053"])
    assert picked == [4] and counts["filtered"] == 1
    assert entry.eligible_entries(bars, decisions, 0, 9 * BAR_MS, 4, entry.FILTERS["confidence_053"])[0] == []


def test_control_uses_signal_direction_and_known_execution_costs():
    bars = [bar(i, 101 if i == 9 else 100) for i in range(18)]
    funding = {i * BAR_MS: 0 for i in (4, 8, 12, 16)}
    decisions = [decision() if i != 4 else decision(-1, .01) for i in range(18)]
    rows = entry.materialize("BTC", bars, decisions, funding, [entry.Match(4, 0)], 4, 5, 12)
    item, = rows
    assert item["side"] == -1 and item["base"] < 0
    assert item["control_base"] == pytest.approx(-19, abs=.02)
    assert item["control_adverse"] == pytest.approx(-33, abs=.05)
    assert entry.summarize(rows)["adverse"]["lift"] == pytest.approx(item["adverse"] - item["control_adverse"])


def test_outcomes_after_training_cutoff_cannot_change_the_selected_filter():
    bars = [bar(i, 100 * 1.001 ** i) for i in range(1600)]
    funding = {i * BAR_MS: 0 for i in range(0, 1600, 4)}
    decisions = [decision(1, .01)] * len(bars)
    boundary = 1500 * BAR_MS
    selected = entry.fit_filter([(bars, decisions, funding)], boundary, 12)
    future_changed = [b if b.t < boundary - BAR_MS else bar(i, 10) for i, b in enumerate(bars)]
    assert selected == entry.fit_filter([(future_changed, decisions, funding)], boundary, 12)
    assert selected[0] != "cash" and selected[2][-1][1] == boundary - BAR_MS
    flat = [bar(i) for i in range(1600)]
    assert entry.fit_filter([(flat, decisions, funding)], boundary, 12)[0] == "cash"


def test_uncertainty_bands_keep_correlated_assets_in_day_blocks():
    rows = [{"t": day * entry.DAY_MS, "adverse": n, "control_adverse": n - 4}
            for day in range(8) for n in (10., -10.)]
    assert entry.block_intervals(rows, draws=100) == {"signal": (0., 0.), "lift": (4., 4.)}
    assert entry.block_intervals(rows[:10], draws=100) is None


def test_prospective_cutoff_cannot_call_incomplete_week_a_holdout():
    start = date_ms("2026-10-01T13:00:00Z")
    end = date_ms("2026-11-12T13:00:00Z")
    meta = {"start_ms": start, "end_ms": end, "first_ms": start - HOUR_MS}
    assert len(windows(meta)) == 3
    with pytest.raises(ValueError, match="no complete seven-day window"):
        windows(meta, end - HOUR_MS)
    with pytest.raises(ValueError, match="control context"):
        windows(meta, start + HOUR_MS)
