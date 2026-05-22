"""Break-even stop policy + shared sl_management primitives (default OFF).

Each safety gate is exercised independently so a future refactor that
removes one fails loudly.  No real DB / exchange calls -- the trader
+ container + mids are stubbed.
"""
from __future__ import annotations

from typing import List

import pytest

import config
import src.trading.break_even_stop as bes
from src.trading.break_even_stop import (
    _compute_break_even_sl,
    evaluate_break_even_stop,
)
from src.trading.sl_management import (
    find_sl_order,
    position_entry_price,
    position_side,
    position_size,
    profit_pct,
    sl_is_tighter,
)


# ── Stubs ────────────────────────────────────────────────────


class _StubTrader:
    """LiveTrader-shaped stub with controllable positions + open orders."""

    def __init__(self, positions, orders=None,
                 cancel_ok=True, place_status="success"):
        self._positions = list(positions)
        self._orders = list(orders or [])
        self.cancel_calls: List[tuple] = []
        self.place_calls: List[tuple] = []
        self._cancel_ok = cancel_ok
        self._place_status = place_status

    def is_live_enabled(self):
        return True

    def is_deployable(self):
        return True

    def get_positions(self, force_fresh=False):
        return list(self._positions)

    def get_open_orders(self, force_fresh=False):
        return list(self._orders)

    def cancel_order(self, coin, order_id):
        self.cancel_calls.append((coin, int(order_id)))
        if self._cancel_ok:
            # Pretend the exchange removed it from open orders.
            self._orders = [o for o in self._orders if int(o.get("oid", 0)) != int(order_id)]
        return bool(self._cancel_ok)

    def place_trigger_order(self, coin, side, size, trigger_price, tp_or_sl="sl"):
        self.place_calls.append((coin, side, size, trigger_price, tp_or_sl))
        return {"status": self._place_status, "oid": 999_000_001}


class _StubContainer:
    def __init__(self, trader):
        self.live_trader = trader
        self.lighter_live_trader = None


@pytest.fixture(autouse=True)
def _gate_defaults(monkeypatch):
    """Force the policy ON with canonical thresholds.  DRY_RUN starts
    OFF so the close path is exercised; individual tests flip it back
    on when they want to verify dry-run behaviour."""
    monkeypatch.setattr(config, "BREAK_EVEN_STOP_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "BREAK_EVEN_STOP_DRY_RUN", False, raising=False)
    monkeypatch.setattr(config, "BREAK_EVEN_STOP_TRIGGER_PCT", 0.01, raising=False)
    monkeypatch.setattr(config, "BREAK_EVEN_STOP_BUFFER_PCT", 0.001, raising=False)


@pytest.fixture
def _mid_btc_77k(monkeypatch):
    """Stub get_all_mids() with BTC at $77000 (unless overridden)."""
    monkeypatch.setattr(bes, "get_all_mids", lambda: {"BTC": 77_000.0})


# ── sl_management primitives ─────────────────────────────────


def test_position_side_normalisation():
    assert position_side({"side": "long"}) == "long"
    assert position_side({"side": "SHORT"}) == "short"
    assert position_side({"szi": 0.5}) == "long"
    assert position_side({"szi": -0.5}) == "short"
    assert position_side({"szi": 0}) == ""
    assert position_side({}) == ""


def test_position_entry_price_handles_all_field_names():
    assert position_entry_price({"entry_price": 100}) == 100.0
    assert position_entry_price({"entryPx": "200"}) == 200.0
    assert position_entry_price({"avgEntryPx": 300.5}) == 300.5
    assert position_entry_price({}) == 0.0
    assert position_entry_price({"entry_price": "bad"}) == 0.0


def test_position_size_returns_absolute():
    assert position_size({"size": 0.5}) == 0.5
    assert position_size({"size": -0.5}) == 0.5  # abs
    assert position_size({"szi": -1.0}) == 1.0
    assert position_size({}) == 0.0


def test_profit_pct_signs():
    # Long, price went up 10%
    assert profit_pct(100, 110, "long") == pytest.approx(0.10)
    # Long, price went down 10%
    assert profit_pct(100, 90, "long") == pytest.approx(-0.10)
    # Short, price went down 10% (favourable)
    assert profit_pct(100, 90, "short") == pytest.approx(0.10)
    # Short, price went up 10% (adverse)
    assert profit_pct(100, 110, "short") == pytest.approx(-0.10)
    # Degenerate inputs
    assert profit_pct(0, 100, "long") == 0.0
    assert profit_pct(100, 100, "") == 0.0


def test_sl_is_tighter_long():
    # Long: tighter SL is HIGHER (closer to current/entry).
    assert sl_is_tighter(side="long", new_sl=76_000, current_sl=75_000) is True
    assert sl_is_tighter(side="long", new_sl=75_000, current_sl=76_000) is False
    assert sl_is_tighter(side="long", new_sl=75_000, current_sl=75_000) is False  # equal


def test_sl_is_tighter_short():
    # Short: tighter SL is LOWER (closer to current/entry).
    assert sl_is_tighter(side="short", new_sl=80_000, current_sl=81_000) is True
    assert sl_is_tighter(side="short", new_sl=82_000, current_sl=81_000) is False
    assert sl_is_tighter(side="short", new_sl=81_000, current_sl=81_000) is False


def test_sl_is_tighter_rejects_zero_or_negative():
    assert sl_is_tighter(side="long", new_sl=0, current_sl=100) is False
    assert sl_is_tighter(side="long", new_sl=100, current_sl=0) is False


def test_find_sl_order_picks_correct_leg_for_long():
    """Long position: SL is a reduce-only SELL trigger below entry."""
    orders = [
        # Open buy order (entry) — not protective
        {"coin": "BTC", "side": "B", "triggerPx": None, "reduceOnly": False, "oid": 1},
        # TP trigger (reduce-only SELL above entry)
        {"coin": "BTC", "side": "A", "triggerPx": 80_000, "reduceOnly": True,
         "orderType": "Take Profit Market", "oid": 2},
        # SL trigger (reduce-only SELL below entry) — what we want
        {"coin": "BTC", "side": "A", "triggerPx": 75_000, "reduceOnly": True,
         "orderType": "Stop Market", "oid": 3},
    ]
    found = find_sl_order(orders, "BTC", "long")
    assert found is not None
    assert found["oid"] == 3
    assert float(found["triggerPx"]) == 75_000


def test_find_sl_order_returns_none_when_no_match():
    orders = [
        {"coin": "ETH", "side": "A", "triggerPx": 2000, "reduceOnly": True, "orderType": "Stop Market", "oid": 5},
    ]
    assert find_sl_order(orders, "BTC", "long") is None


def test_find_sl_order_handles_empty():
    assert find_sl_order([], "BTC", "long") is None
    assert find_sl_order(None, "BTC", "long") is None  # type: ignore[arg-type]


# ── break_even_stop policy gates ──────────────────────────────


def test_disabled_short_circuits(_mid_btc_77k, monkeypatch):
    monkeypatch.setattr(config, "BREAK_EVEN_STOP_ENABLED", False, raising=False)
    trader = _StubTrader(
        positions=[{"coin": "BTC", "side": "long", "entry_price": 76_000, "size": 0.001}],
        orders=[{"coin": "BTC", "side": "A", "triggerPx": 73_000, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 100}],
    )
    container = _StubContainer(trader)
    evaluate_break_even_stop(container)
    assert trader.cancel_calls == []
    assert trader.place_calls == []


def test_compute_new_sl_long_at_entry_with_buffer():
    sl = _compute_break_even_sl("long", 100.0, 0.001)
    assert sl == pytest.approx(100.1)


def test_compute_new_sl_short_at_entry_with_buffer():
    sl = _compute_break_even_sl("short", 100.0, 0.001)
    assert sl == pytest.approx(99.9)


def test_below_trigger_threshold_does_not_act(_mid_btc_77k):
    """Long at 76_500, current 77_000 → only +0.65% profit (< 1% trigger)."""
    trader = _StubTrader(
        positions=[{"coin": "BTC", "side": "long", "entry_price": 76_500, "size": 0.001}],
        orders=[{"coin": "BTC", "side": "A", "triggerPx": 73_000, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 100}],
    )
    container = _StubContainer(trader)
    evaluate_break_even_stop(container)
    assert trader.cancel_calls == []
    assert trader.place_calls == []


def test_no_active_sl_skips(_mid_btc_77k):
    """Profitable position but no SL on books → defer to orphan-protection."""
    trader = _StubTrader(
        positions=[{"coin": "BTC", "side": "long", "entry_price": 70_000, "size": 0.001}],
        orders=[],  # no SL
    )
    container = _StubContainer(trader)
    evaluate_break_even_stop(container)
    assert trader.cancel_calls == []
    assert trader.place_calls == []


def test_sl_already_at_or_above_entry_skips(_mid_btc_77k):
    """Position profitable; SL already at 76_900 > entry 76_000.  Don't move."""
    trader = _StubTrader(
        positions=[{"coin": "BTC", "side": "long", "entry_price": 76_000, "size": 0.001}],
        orders=[{"coin": "BTC", "side": "A", "triggerPx": 76_900, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 100}],
    )
    container = _StubContainer(trader)
    evaluate_break_even_stop(container)
    # New SL would be 76_076 (entry * 1.001), which is < current 76_900,
    # so sl_is_tighter returns False -- no action.
    assert trader.cancel_calls == []
    assert trader.place_calls == []


def test_dry_run_logs_but_never_acts(_mid_btc_77k, monkeypatch):
    monkeypatch.setattr(config, "BREAK_EVEN_STOP_DRY_RUN", True, raising=False)
    trader = _StubTrader(
        positions=[{"coin": "BTC", "side": "long", "entry_price": 75_000, "size": 0.001}],
        orders=[{"coin": "BTC", "side": "A", "triggerPx": 73_000, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 100}],
    )
    container = _StubContainer(trader)
    evaluate_break_even_stop(container)
    assert trader.cancel_calls == []
    assert trader.place_calls == []


def test_full_gate_pass_promotes_sl_to_entry_plus_buffer(_mid_btc_77k):
    """Long entered at 75_000, current 77_000 (+2.67% > 1% trigger).
    SL was at 73_000.  Should cancel + place new SL at 75_075 (entry * 1.001)."""
    trader = _StubTrader(
        positions=[{"coin": "BTC", "side": "long", "entry_price": 75_000, "size": 0.001}],
        orders=[{"coin": "BTC", "side": "A", "triggerPx": 73_000, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 12345}],
    )
    container = _StubContainer(trader)
    evaluate_break_even_stop(container)
    # Cancelled old SL by oid 12345
    assert trader.cancel_calls == [("BTC", 12345)]
    # Placed new SL: SELL (closing long), size 0.001, trigger 75_075, tp_or_sl="sl"
    assert len(trader.place_calls) == 1
    coin, side, size, trigger, tp_or_sl = trader.place_calls[0]
    assert coin == "BTC"
    assert side == "sell"
    assert size == pytest.approx(0.001)
    assert trigger == pytest.approx(75_075.0)
    assert tp_or_sl == "sl"


def test_short_position_promotion(_mid_btc_77k, monkeypatch):
    """Short at 79_000, current 77_000 (+2.53% favourable).  SL was 81_000.
    Should move to 78_921 (entry * 0.999)."""
    monkeypatch.setattr(bes, "get_all_mids", lambda: {"BTC": 77_000.0})
    trader = _StubTrader(
        positions=[{"coin": "BTC", "side": "short", "entry_price": 79_000, "size": 0.001}],
        orders=[{"coin": "BTC", "side": "B", "triggerPx": 81_000, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 555}],
    )
    container = _StubContainer(trader)
    evaluate_break_even_stop(container)
    assert trader.cancel_calls == [("BTC", 555)]
    coin, side, size, trigger, tp_or_sl = trader.place_calls[0]
    assert side == "buy"  # closing a short → BUY
    assert trigger == pytest.approx(78_921.0)


def test_cancel_failure_does_not_place_new_sl(_mid_btc_77k):
    """If cancel returns False, the new SL is NOT placed -- position keeps
    the original SL intact rather than ending up unprotected."""
    trader = _StubTrader(
        positions=[{"coin": "BTC", "side": "long", "entry_price": 75_000, "size": 0.001}],
        orders=[{"coin": "BTC", "side": "A", "triggerPx": 73_000, "reduceOnly": True,
                 "orderType": "Stop Market", "oid": 12345}],
        cancel_ok=False,
    )
    container = _StubContainer(trader)
    evaluate_break_even_stop(container)
    assert trader.cancel_calls == [("BTC", 12345)]
    assert trader.place_calls == []  # never attempted new SL after cancel failed
