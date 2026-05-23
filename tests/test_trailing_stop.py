"""Trailing stop policy (default OFF).

Each gate exercised independently; HWM/LWM dynamics covered.
"""
from __future__ import annotations

from typing import List

import pytest

import config
import src.trading.trailing_stop as ts
from src.trading.trailing_stop import (
    _compute_trailing_sl,
    _water_marks,
    evaluate_trailing_stop,
)


# ── Stubs ────────────────────────────────────────────────────


class _StubTrader:
    def __init__(self, positions, orders=None, cancel_ok=True, place_status="success"):
        self._positions = list(positions)
        self._orders = list(orders or [])
        self.cancel_calls: List[tuple] = []
        self.place_calls: List[tuple] = []
        self._cancel_ok = cancel_ok
        self._place_status = place_status

    def is_live_enabled(self): return True
    def is_deployable(self): return True

    def get_positions(self, force_fresh=False):
        return list(self._positions)

    def get_open_orders(self, force_fresh=False):
        return list(self._orders)

    def cancel_order(self, coin, order_id):
        self.cancel_calls.append((coin, int(order_id)))
        if self._cancel_ok:
            self._orders = [o for o in self._orders if int(o.get("oid", 0)) != int(order_id)]
        return bool(self._cancel_ok)

    def place_trigger_order(self, coin, side, size, trigger_price, tp_or_sl="sl"):
        self.place_calls.append((coin, side, size, trigger_price, tp_or_sl))
        return {"status": self._place_status, "oid": 777_000_001}


class _StubContainer:
    def __init__(self, trader):
        self.live_trader = trader
        self.lighter_live_trader = None


@pytest.fixture(autouse=True)
def _reset_water_marks():
    _water_marks.clear()
    yield
    _water_marks.clear()


@pytest.fixture(autouse=True)
def _gate_defaults(monkeypatch):
    monkeypatch.setattr(config, "TRAILING_STOP_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "TRAILING_STOP_DRY_RUN", False, raising=False)
    monkeypatch.setattr(config, "TRAILING_STOP_ACTIVATION_PROFIT_PCT", 0.01, raising=False)
    monkeypatch.setattr(config, "TRAILING_STOP_OFFSET_PCT", 0.01, raising=False)
    monkeypatch.setattr(config, "TRAILING_STOP_MIN_STEP_PCT", 0.002, raising=False)


# ── _compute_trailing_sl ─────────────────────────────────────


def test_compute_trailing_sl_long():
    # HWM 100, offset 1% -> SL at 99
    assert _compute_trailing_sl("long", 100.0, 0.01) == pytest.approx(99.0)


def test_compute_trailing_sl_short():
    # LWM 100, offset 1% -> SL at 101
    assert _compute_trailing_sl("short", 100.0, 0.01) == pytest.approx(101.0)


def test_compute_trailing_sl_invalid():
    assert _compute_trailing_sl("long", 0, 0.01) == 0
    assert _compute_trailing_sl("long", 100, 0) == 0


# ── Gates ────────────────────────────────────────────────────


def test_disabled_short_circuits(monkeypatch):
    monkeypatch.setattr(config, "TRAILING_STOP_ENABLED", False, raising=False)
    monkeypatch.setattr(ts, "get_all_mids", lambda: {"BTC": 80_000.0})
    trader = _StubTrader(
        positions=[{"coin": "BTC", "side": "long", "entry_price": 70_000, "size": 0.001}],
        orders=[{"coin": "BTC", "side": "A", "triggerPx": 68_000, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 1}],
    )
    container = _StubContainer(trader)
    evaluate_trailing_stop(container)
    assert trader.cancel_calls == []
    assert trader.place_calls == []


def test_below_activation_profit_does_not_act(monkeypatch):
    """Position only +0.5% -- below 1% activation threshold."""
    monkeypatch.setattr(ts, "get_all_mids", lambda: {"BTC": 100_500.0})
    trader = _StubTrader(
        positions=[{"coin": "BTC", "side": "long", "entry_price": 100_000, "size": 0.001}],
        orders=[{"coin": "BTC", "side": "A", "triggerPx": 95_000, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 1}],
    )
    container = _StubContainer(trader)
    evaluate_trailing_stop(container)
    assert trader.place_calls == []


def test_min_step_throttle_blocks_tiny_moves(monkeypatch):
    """Position +5%, HWM 105_000, SL trail computes to 103_950 (-1% off HWM).
    Current SL at 103_900.  Diff = 50/103900 ~= 0.048%, BELOW 0.2% step."""
    monkeypatch.setattr(ts, "get_all_mids", lambda: {"BTC": 105_000.0})
    trader = _StubTrader(
        positions=[{"coin": "BTC", "side": "long", "entry_price": 100_000, "size": 0.001}],
        orders=[{"coin": "BTC", "side": "A", "triggerPx": 103_900, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 1}],
    )
    container = _StubContainer(trader)
    evaluate_trailing_stop(container)
    assert trader.place_calls == []  # blocked by min-step


def test_full_gate_pass_trails_sl_long(monkeypatch):
    """Long at 100k, current 110k (+10%, HWM 110k).  Trail offset 1% ->
    new SL = 110_000 * 0.99 = 108_900.  Current SL at 95_000 (way below).
    Move from 95_000 -> 108_900."""
    monkeypatch.setattr(ts, "get_all_mids", lambda: {"BTC": 110_000.0})
    trader = _StubTrader(
        positions=[{"coin": "BTC", "side": "long", "entry_price": 100_000, "size": 0.001}],
        orders=[{"coin": "BTC", "side": "A", "triggerPx": 95_000, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 1234}],
    )
    container = _StubContainer(trader)
    evaluate_trailing_stop(container)
    assert trader.cancel_calls == [("BTC", 1234)]
    coin, side, size, trigger, tp_or_sl = trader.place_calls[0]
    assert side == "sell"
    assert trigger == pytest.approx(108_900.0)


def test_full_gate_pass_trails_sl_short(monkeypatch):
    """Short at 100k, current 90k (+10% favourable, LWM 90k).  Trail offset 1%
    -> new SL = 90_000 * 1.01 = 90_900.  Current SL at 105_000.  Move down."""
    monkeypatch.setattr(ts, "get_all_mids", lambda: {"BTC": 90_000.0})
    trader = _StubTrader(
        positions=[{"coin": "BTC", "side": "short", "entry_price": 100_000, "size": 0.001}],
        orders=[{"coin": "BTC", "side": "B", "triggerPx": 105_000, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 7777}],
    )
    container = _StubContainer(trader)
    evaluate_trailing_stop(container)
    coin, side, size, trigger, tp_or_sl = trader.place_calls[0]
    assert side == "buy"  # closing a short
    assert trigger == pytest.approx(90_900.0)


def test_wrong_side_guard_blocks_short_when_price_reverses(monkeypatch):
    """REGRESSION: production dry-run captured a SHORT SOL trailing-stop
    proposal where new_sl (82.47) was BELOW current_price (84.17) -- the
    SL would have auto-triggered on placement.

    Scenario: short SOL at 85.05 entry, LWM hit 81.65 earlier, but price
    has since reversed back up to 84.17.  With offset=1.00%, the
    computed trail SL is 81.65 * 1.01 = 82.47 -- on the WRONG side of
    current price for a short (must be ABOVE current).

    The side-of-price guard must skip this move.  Without the guard,
    the position would have closed instantly when SL was placed."""
    container = _StubContainer(_StubTrader([], []))

    # Cycle 1: price drops to 81.65, hwm/lwm tracks the low.
    monkeypatch.setattr(ts, "get_all_mids", lambda: {"SOL": 81.65})
    container.live_trader = _StubTrader(
        positions=[{"coin": "SOL", "side": "short", "entry_price": 85.05, "size": 1.0}],
        orders=[{"coin": "SOL", "side": "B", "triggerPx": 86.57, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 11111}],
    )
    evaluate_trailing_stop(container)
    # First cycle SHOULD move the SL because price (81.65) is well below
    # the proposed new SL (82.47), so SL is correctly above price.
    assert container.live_trader.place_calls, (
        "First cycle should move SL when LWM = current price"
    )
    container.live_trader.place_calls.clear()
    container.live_trader.cancel_calls.clear()

    # Cycle 2: price has reversed back up to 84.17 (above the locked-in
    # SL of 82.47).  HWM/LWM stays at 81.65 (favourable-only update).
    # The computed new SL based on LWM is still 82.47 -- but 82.47 is
    # now BELOW current 84.17 = wrong side for a short.  Must skip.
    monkeypatch.setattr(ts, "get_all_mids", lambda: {"SOL": 84.17})
    # Pretend the previous cycle moved SL to ~82.47; here we keep the
    # ORIGINAL 86.57 in order to exercise the wrong-side guard.  In
    # practice the previous cycle would have done the same skip the
    # first time price reversed past the proposed SL.
    container.live_trader = _StubTrader(
        positions=[{"coin": "SOL", "side": "short", "entry_price": 85.05, "size": 1.0}],
        orders=[{"coin": "SOL", "side": "B", "triggerPx": 86.57, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 11111}],
    )
    evaluate_trailing_stop(container)
    assert container.live_trader.place_calls == [], (
        "Second cycle must NOT place an SL on the wrong side of price"
    )
    assert container.live_trader.cancel_calls == [], (
        "Second cycle must NOT cancel the existing SL just to skip placement"
    )


def test_wrong_side_guard_blocks_long_when_price_reverses(monkeypatch):
    """Symmetric: long BTC trailing SL at 99k when price reversed back
    down to 98k means the proposed SL (99k) would be ABOVE current
    price (98k) -- a long SL must be BELOW current price."""
    container = _StubContainer(_StubTrader([], []))

    # Establish HWM = 100k.
    monkeypatch.setattr(ts, "get_all_mids", lambda: {"BTC": 100_000.0})
    container.live_trader = _StubTrader(
        positions=[{"coin": "BTC", "side": "long", "entry_price": 90_000, "size": 0.001}],
        orders=[{"coin": "BTC", "side": "A", "triggerPx": 85_000, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 22222}],
    )
    evaluate_trailing_stop(container)
    container.live_trader.place_calls.clear()
    container.live_trader.cancel_calls.clear()

    # Price reverses to 98k.  HWM stays at 100k.  Proposed trail SL =
    # 100_000 * 0.99 = 99_000.  But 99_000 > 98_000 = wrong side for a long.
    monkeypatch.setattr(ts, "get_all_mids", lambda: {"BTC": 98_000.0})
    container.live_trader = _StubTrader(
        positions=[{"coin": "BTC", "side": "long", "entry_price": 90_000, "size": 0.001}],
        orders=[{"coin": "BTC", "side": "A", "triggerPx": 85_000, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 22222}],
    )
    evaluate_trailing_stop(container)
    assert container.live_trader.place_calls == [], (
        "Trailing SL on the wrong side of price (long) must be skipped"
    )


def test_hwm_persists_across_cycles_long(monkeypatch):
    """Long at 100k.  Cycle 1: price 110k -> HWM 110k.  Cycle 2: price 108k
    (drop, but still above entry+1%).  HWM should STAY at 110k (favourable-only
    update), so trailing SL stays at 108_900 instead of recomputing from 108k."""
    container = _StubContainer(_StubTrader([], []))

    # Cycle 1
    monkeypatch.setattr(ts, "get_all_mids", lambda: {"BTC": 110_000.0})
    container.live_trader = _StubTrader(
        positions=[{"coin": "BTC", "side": "long", "entry_price": 100_000, "size": 0.001}],
        orders=[{"coin": "BTC", "side": "A", "triggerPx": 95_000, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 1}],
    )
    evaluate_trailing_stop(container)
    sl1 = container.live_trader.place_calls[0][3]
    assert sl1 == pytest.approx(108_900.0)
    assert _water_marks[("BTC", "long")] == 110_000.0

    # Cycle 2 - price ticks down to 108k.  HWM should hold at 110k.
    monkeypatch.setattr(ts, "get_all_mids", lambda: {"BTC": 108_000.0})
    # New trader stub representing fresh exchange state -- SL is now
    # the 108_900 placed last cycle.
    container.live_trader = _StubTrader(
        positions=[{"coin": "BTC", "side": "long", "entry_price": 100_000, "size": 0.001}],
        orders=[{"coin": "BTC", "side": "A", "triggerPx": 108_900, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 2}],
    )
    evaluate_trailing_stop(container)
    # HWM still 110_000; trailing SL still 108_900; current SL already
    # at 108_900 -> sl_is_tighter returns False -> no action.
    assert container.live_trader.place_calls == []
    assert _water_marks[("BTC", "long")] == 110_000.0


def test_water_mark_resets_when_position_closes(monkeypatch):
    """When the position is no longer in fetched positions, the water mark
    for that (coin, side) is dropped so a fresh re-entry starts clean."""
    monkeypatch.setattr(ts, "get_all_mids", lambda: {"BTC": 110_000.0})
    # First cycle: position exists, establishes water mark.
    container = _StubContainer(_StubTrader(
        positions=[{"coin": "BTC", "side": "long", "entry_price": 100_000, "size": 0.001}],
        orders=[{"coin": "BTC", "side": "A", "triggerPx": 95_000, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 1}],
    ))
    evaluate_trailing_stop(container)
    assert _water_marks[("BTC", "long")] == 110_000.0

    # Second cycle: position has closed -- empty positions list.
    container.live_trader = _StubTrader(positions=[], orders=[])
    evaluate_trailing_stop(container)
    # Water mark gone (or at least no longer present for BTC).
    assert ("BTC", "long") not in _water_marks


def test_dry_run_logs_but_never_acts(monkeypatch):
    monkeypatch.setattr(config, "TRAILING_STOP_DRY_RUN", True, raising=False)
    monkeypatch.setattr(ts, "get_all_mids", lambda: {"BTC": 110_000.0})
    trader = _StubTrader(
        positions=[{"coin": "BTC", "side": "long", "entry_price": 100_000, "size": 0.001}],
        orders=[{"coin": "BTC", "side": "A", "triggerPx": 95_000, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 1}],
    )
    container = _StubContainer(trader)
    evaluate_trailing_stop(container)
    assert trader.cancel_calls == []
    assert trader.place_calls == []


def test_cancel_failure_does_not_place_new_sl(monkeypatch):
    monkeypatch.setattr(ts, "get_all_mids", lambda: {"BTC": 110_000.0})
    trader = _StubTrader(
        positions=[{"coin": "BTC", "side": "long", "entry_price": 100_000, "size": 0.001}],
        orders=[{"coin": "BTC", "side": "A", "triggerPx": 95_000, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 1234}],
        cancel_ok=False,
    )
    container = _StubContainer(trader)
    evaluate_trailing_stop(container)
    assert trader.cancel_calls == [("BTC", 1234)]
    assert trader.place_calls == []


def test_no_active_sl_skips(monkeypatch):
    monkeypatch.setattr(ts, "get_all_mids", lambda: {"BTC": 110_000.0})
    trader = _StubTrader(
        positions=[{"coin": "BTC", "side": "long", "entry_price": 100_000, "size": 0.001}],
        orders=[],  # no SL
    )
    container = _StubContainer(trader)
    evaluate_trailing_stop(container)
    assert trader.cancel_calls == []
    assert trader.place_calls == []
