from dataclasses import replace

import pytest

from scripts import fetch_hyperliquid_replay_market as fetch
from src.backtest.candle_exit_replay import (
    BAR_MS, HOUR_MS, Bar, Decision, aggregate, arena_decisions, select, simulate, trigger,
)


def candle(i, o=100., h=100.1, low=99.9, c=100.):
    return Bar(i * BAR_MS, o, h, low, c, 10.)


def signal(side=0, atr=.01):
    return Decision(side, .7 if side else 0., atr, "choppy")


def run(bars, decisions, policy="fixed", funding=None, log=None):
    return simulate(bars, decisions, funding or {}, bars[0].t, bars[-1].t + BAR_MS,
                    policy=policy, slip_bps=0, capital=1000, trace=log)


def test_next_open_not_decision_close_and_fee_is_charged_both_sides():
    bars = [candle(0), candle(1, 110., 110.5, 109.9, 110.3), candle(2, 110., 110.3, 109., 109.5)]
    result = run(bars, [signal(1), signal(), signal()])
    trade, = result["trades"]
    assert trade["entry_bar"] == BAR_MS and trade["signal_bar"] == 0
    assert trade["entry_price"] == 110. and trade["exit_reason"] == "window_end"
    assert trade["entry_fee"] > 0 and trade["exit_fee"] > 0
    assert aggregate([result], 1000)["net"] == pytest.approx(trade["pnl"])


def test_intrabar_dual_touch_and_gap_fill_both_sides():
    assert trigger(candle(0, 100, 105, 96, 100), 1, 98, 104) == ("stop_loss", 98, True)
    assert trigger(candle(0, 100, 105, 95, 100), -1, 102, 96) == ("stop_loss", 102, True)
    assert trigger(candle(0, 95, 106, 93, 100), 1, 98, 104) == ("stop_gap", 95, False)
    assert trigger(candle(0, 104, 105, 95, 100), -1, 102, 96) == ("stop_gap", 104, False)


def test_trail_uses_completed_close_only_and_never_changes_earlier_bar():
    bars = [candle(0), candle(1, 100, 103, 99, 102.5), candle(2, 101, 102, 100, 101)]
    seen = []
    result = run(bars, [signal(1), signal(0, atr=.005), signal()], "atr_2x_tighten", log=seen.append)
    trade, = result["trades"]
    update, = (event for event in seen if event["kind"] == "stop_update")
    assert update["t"] == 2 * BAR_MS and update["stop"] > update["previous"]
    assert trade["exit_bar"] == 2 * BAR_MS and trade["exit_reason"] == "stop_gap"


@pytest.mark.parametrize("side, sign", [(1, 1), (-1, -1)])
def test_hourly_funding_is_directional_and_not_double_counted(side, sign):
    bars = [candle(i) for i in range(6)]
    result = run(bars, [signal(side)] + [signal()] * 5, funding={HOUR_MS: .001})
    trade, = result["trades"]
    assert trade["funding"] == pytest.approx(sign * .05)
    assert result["equity"][-1] - 1000 == pytest.approx(trade["pnl"])


def test_future_market_data_cannot_change_earlier_arena_votes_or_atr():
    from math import sin

    original = [candle(i, (c := 100 + sin(i / 4) + i / 100), c + .2, c - .2, c) for i in range(52)]
    poisoned = [b if i < 45 else replace(b, open=500., close=500., high=501., low=498.) for i, b in enumerate(original)]
    assert arena_decisions("BTC", original)[:45] == arena_decisions("BTC", poisoned)[:45]
    assert any(d.side for d in arena_decisions("BTC", original))


def test_selection_preserves_fixed_without_positive_stress_and_bounded_drawdown():
    fixed = {"n": 40, "net": -1, "drawdown_pct": .7}
    too_risky = {"n": 50, "net": 3, "drawdown_pct": .9}
    too_few = {"n": 29, "net": 4, "drawdown_pct": .6}
    assert select({"fixed": fixed, "atr_2x": too_risky, "atr_3x": too_few}) == "fixed"
    assert select({"fixed": fixed, "atr_2x": {**too_risky, "drawdown_pct": .6}}) == "atr_2x"


def test_fetch_ignores_next_unfinished_bar_and_rejects_malformed_closed_bar(monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda _: None)
    sample = {"t": 0, "T": BAR_MS - 1, "s": "BTC", "i": "15m", "o": "100", "h": "101", "l": "99", "c": "100", "v": "1"}
    rows = [sample, {**sample, "t": BAR_MS, "T": 2 * BAR_MS - 1}]
    assert list(fetch.fetch_candles("BTC", 0, BAR_MS, lambda _: rows)) == [0]
    with pytest.raises(ValueError, match="malformed candle"):
        fetch.fetch_candles("BTC", 0, BAR_MS, lambda _: [{**sample, "T": BAR_MS}])


def test_rank_uses_notional_and_skips_delisted_spot_and_synthetic_hyperps():
    names = [{"name": f"C{i}"} for i in range(11)]
    rejected = [{"name": "xyz:TEST"}, {"name": "@15"}, {"name": "DONE", "isDelisted": True},
                {"name": "SYN", "isHyperp": True}]
    data = [{"dayNtlVlm": str(i)} for i in range(1, 12)] + [{"dayNtlVlm": "1000"}] * len(rejected)
    assert [c for c, _ in fetch.ranking(({"universe": names + rejected}, data))] == [f"C{i}" for i in range(10, 0, -1)]
