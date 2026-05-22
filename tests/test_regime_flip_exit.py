"""Regime-flip exit policy (default OFF).

Each gate is exercised independently so a future refactor that
accidentally removes one of them fails loudly.  All tests use a stub
trader + stub container + monkeypatched ``db.get_open_paper_trades``
so no real DB / exchange calls happen.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

import config
import src.trading.regime_flip_exit as rfe
from src.trading.regime_flip_exit import (
    _against_cycle_counters,
    _coin_confidence,
    _coin_direction,
    evaluate_regime_flip_exits,
)


# ── Stubs ────────────────────────────────────────────────────


class _StubTrader:
    """Minimal LiveTrader-shaped stub: tracks close_position calls."""

    def __init__(self, positions, close_result=None):
        self._positions = list(positions)
        self.close_calls = []
        self._close_result = close_result or {"status": "success"}

    # Mirror the real LiveTrader API surface used by regime_flip_exit
    def is_live_enabled(self):
        return True

    def is_deployable(self):
        return True

    def get_positions(self, force_fresh=False):
        return list(self._positions)

    def close_position(self, coin):
        self.close_calls.append(coin)
        # Simulate the exchange closing the position so a subsequent
        # ``get_positions`` returns an empty list.
        self._positions = [
            p for p in self._positions
            if str(p.get("coin", "")).upper() != coin.upper()
        ]
        return self._close_result


class _StubContainer:
    """LIVE-mode container with a configurable forecaster + trader."""

    def __init__(self, trader, forecaster=None):
        self.live_trader = trader
        self.lighter_live_trader = None
        self.predictive_forecaster = forecaster


class _StubForecaster:
    def __init__(self, signals):
        # signals: {"BTC": -0.45, "ETH": 0.10, ...}
        self.last_signals = {
            coin: {"signal": sig} for coin, sig in signals.items()
        }


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_counters():
    _against_cycle_counters.clear()
    yield
    _against_cycle_counters.clear()


@pytest.fixture(autouse=True)
def _gate_defaults(monkeypatch):
    """Force the gate ON with canonical thresholds.  DRY_RUN is the
    operator-toggle the test cases flip explicitly."""
    monkeypatch.setattr(config, "REGIME_FLIP_EXIT_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "REGIME_FLIP_EXIT_DRY_RUN", False, raising=False)
    monkeypatch.setattr(config, "REGIME_FLIP_EXIT_MIN_CONFIDENCE", 0.70, raising=False)
    monkeypatch.setattr(
        config, "REGIME_FLIP_EXIT_MIN_CONSECUTIVE_CYCLES", 2, raising=False,
    )
    monkeypatch.setattr(config, "REGIME_FLIP_EXIT_MIN_HOLD_SECONDS", 300, raising=False)
    monkeypatch.setattr(
        config, "REGIME_FLIP_EXIT_REQUIRE_FORECASTER_AGREE", False, raising=False,
    )
    monkeypatch.setattr(
        config, "REGIME_FLIP_EXIT_FORECASTER_MIN_SIGNAL", 0.20, raising=False,
    )


@pytest.fixture
def _stub_old_paper_trade(monkeypatch):
    """Pretend the paper trade for BTC opened 30 minutes ago."""
    from datetime import datetime, timezone, timedelta
    opened_at = (
        datetime.now(timezone.utc) - timedelta(minutes=30)
    ).isoformat()
    monkeypatch.setattr(
        rfe.db, "get_open_paper_trades",
        lambda: [{"coin": "BTC", "opened_at": opened_at}],
    )


# ── Helpers / direction normalisation ────────────────────────


@pytest.mark.parametrize("data,expected", [
    ({"direction": "up"}, "up"),
    ({"direction": "DOWN"}, "down"),
    ({"regime": "trending_up"}, "up"),
    ({"regime": "crash"}, "down"),
    ({"momentum": 0.05}, "up"),
    ({"momentum": -0.05}, "down"),
    ({"momentum": 0.005}, ""),     # below noise floor
    ({}, ""),
    ({"direction": ""}, ""),
])
def test_coin_direction_normalisation(data, expected):
    assert _coin_direction(data) == expected


@pytest.mark.parametrize("data,expected", [
    ({"confidence": 0.85}, 0.85),
    ({"confidence": 85}, 0.85),  # percent form
    ({"regime_confidence": 0.5}, 0.5),
    ({}, 0.0),
    ({"confidence": "bad"}, 0.0),
    ({"confidence": 1.5}, 1.0),  # clamped
    ({"confidence": -0.1}, 0.0),  # clamped
])
def test_coin_confidence_normalisation(data, expected):
    assert _coin_confidence(data) == pytest.approx(expected)


# ── Master switch ────────────────────────────────────────────


def test_disabled_short_circuits_with_no_side_effects(monkeypatch):
    monkeypatch.setattr(config, "REGIME_FLIP_EXIT_ENABLED", False, raising=False)
    trader = _StubTrader([{"coin": "BTC", "side": "long"}])
    container = _StubContainer(trader)
    evaluate_regime_flip_exits(
        container,
        {"per_coin": {"BTC": {"direction": "down", "confidence": 0.99}}},
    )
    assert trader.close_calls == []


def test_no_regime_data_is_noop():
    trader = _StubTrader([{"coin": "BTC", "side": "long"}])
    container = _StubContainer(trader)
    evaluate_regime_flip_exits(container, None)
    evaluate_regime_flip_exits(container, {})
    evaluate_regime_flip_exits(container, {"per_coin": None})
    assert trader.close_calls == []


def test_no_positions_is_noop():
    trader = _StubTrader([])
    container = _StubContainer(trader)
    evaluate_regime_flip_exits(
        container,
        {"per_coin": {"BTC": {"direction": "down", "confidence": 0.99}}},
    )
    assert trader.close_calls == []


# ── Gate 1: min hold time ────────────────────────────────────


def test_fresh_position_below_min_hold_is_not_closed(monkeypatch):
    fresh_open = datetime.now(timezone.utc).isoformat()  # opened now
    monkeypatch.setattr(
        rfe.db, "get_open_paper_trades",
        lambda: [{"coin": "BTC", "opened_at": fresh_open}],
    )
    trader = _StubTrader([{"coin": "BTC", "side": "long"}])
    container = _StubContainer(trader)
    # Multiple cycles in rapid succession should NOT close while age < 300s.
    for _ in range(5):
        evaluate_regime_flip_exits(
            container,
            {"per_coin": {"BTC": {"direction": "down", "confidence": 0.99}}},
        )
    assert trader.close_calls == []


def test_orphan_position_without_paper_row_is_skipped(monkeypatch):
    """When ``get_open_paper_trades`` returns no row for the live coin,
    the gate fails OPEN (skips close) -- defer to SL backstop."""
    monkeypatch.setattr(rfe.db, "get_open_paper_trades", lambda: [])
    trader = _StubTrader([{"coin": "BTC", "side": "long"}])
    container = _StubContainer(trader)
    for _ in range(5):
        evaluate_regime_flip_exits(
            container,
            {"per_coin": {"BTC": {"direction": "down", "confidence": 0.99}}},
        )
    assert trader.close_calls == []


# ── Gate 2 + 3: regime against position at high confidence ──


def test_regime_not_against_does_not_count(_stub_old_paper_trade):
    """LONG position + regime says UP -> no count, no close even after
    many cycles."""
    trader = _StubTrader([{"coin": "BTC", "side": "long"}])
    container = _StubContainer(trader)
    for _ in range(10):
        evaluate_regime_flip_exits(
            container,
            {"per_coin": {"BTC": {"direction": "up", "confidence": 0.99}}},
        )
    assert trader.close_calls == []


def test_low_confidence_against_regime_does_not_count(_stub_old_paper_trade):
    """LONG position + regime DOWN but confidence below threshold -> no count."""
    trader = _StubTrader([{"coin": "BTC", "side": "long"}])
    container = _StubContainer(trader)
    for _ in range(10):
        evaluate_regime_flip_exits(
            container,
            {"per_coin": {"BTC": {"direction": "down", "confidence": 0.50}}},
        )
    assert trader.close_calls == []


# ── Gate 4: forecaster agreement (optional) ──


def test_forecaster_disagrees_does_not_count(_stub_old_paper_trade, monkeypatch):
    """When forecaster gate is ON and the forecaster says UP while
    regime says DOWN, the cycle does not count."""
    monkeypatch.setattr(
        config, "REGIME_FLIP_EXIT_REQUIRE_FORECASTER_AGREE", True, raising=False,
    )
    trader = _StubTrader([{"coin": "BTC", "side": "long"}])
    forecaster = _StubForecaster({"BTC": +0.30})  # bullish forecaster
    container = _StubContainer(trader, forecaster=forecaster)
    for _ in range(10):
        evaluate_regime_flip_exits(
            container,
            {"per_coin": {"BTC": {"direction": "down", "confidence": 0.99}}},
        )
    assert trader.close_calls == []


def test_missing_forecaster_with_require_true_defers(_stub_old_paper_trade, monkeypatch):
    """When forecaster gate is ON but no read is available, the cycle
    does NOT count (defers to next read) -- and crucially does NOT
    close based on regime alone."""
    monkeypatch.setattr(
        config, "REGIME_FLIP_EXIT_REQUIRE_FORECASTER_AGREE", True, raising=False,
    )
    trader = _StubTrader([{"coin": "BTC", "side": "long"}])
    container = _StubContainer(trader, forecaster=None)
    for _ in range(10):
        evaluate_regime_flip_exits(
            container,
            {"per_coin": {"BTC": {"direction": "down", "confidence": 0.99}}},
        )
    assert trader.close_calls == []


# ── Gate 5: persistence ──────────────────────────────────────


def test_single_against_cycle_does_not_close(_stub_old_paper_trade):
    """One cycle of against-regime is not enough -- need MIN_CONSECUTIVE_CYCLES."""
    trader = _StubTrader([{"coin": "BTC", "side": "long"}])
    container = _StubContainer(trader)
    evaluate_regime_flip_exits(
        container,
        {"per_coin": {"BTC": {"direction": "down", "confidence": 0.99}}},
    )
    assert trader.close_calls == []
    assert _against_cycle_counters[("BTC", "long")] == 1


def test_two_consecutive_against_cycles_close(_stub_old_paper_trade):
    """Two consecutive cycles of against-regime at high confidence -> close."""
    trader = _StubTrader([{"coin": "BTC", "side": "long"}])
    container = _StubContainer(trader)
    regime = {"per_coin": {"BTC": {"direction": "down", "confidence": 0.99}}}
    evaluate_regime_flip_exits(container, regime)  # cycle 1
    assert trader.close_calls == []
    evaluate_regime_flip_exits(container, regime)  # cycle 2 -> close
    assert trader.close_calls == ["BTC"]


def test_counter_resets_when_regime_flips_back(_stub_old_paper_trade):
    """Cycle 1 against, cycle 2 with regime in our favor -> counter resets;
    cycle 3 against again should NOT close (only 1 consecutive)."""
    trader = _StubTrader([{"coin": "BTC", "side": "long"}])
    container = _StubContainer(trader)
    evaluate_regime_flip_exits(
        container,
        {"per_coin": {"BTC": {"direction": "down", "confidence": 0.99}}},
    )
    assert _against_cycle_counters[("BTC", "long")] == 1
    evaluate_regime_flip_exits(
        container,
        {"per_coin": {"BTC": {"direction": "up", "confidence": 0.99}}},
    )
    assert ("BTC", "long") not in _against_cycle_counters
    evaluate_regime_flip_exits(
        container,
        {"per_coin": {"BTC": {"direction": "down", "confidence": 0.99}}},
    )
    # Only one consecutive against -> no close yet
    assert trader.close_calls == []


# ── Dry-run mode ─────────────────────────────────────────────


def test_dry_run_never_closes(_stub_old_paper_trade, monkeypatch):
    """With DRY_RUN=true, the gate evaluates fully and logs, but never
    calls close_position."""
    monkeypatch.setattr(config, "REGIME_FLIP_EXIT_DRY_RUN", True, raising=False)
    trader = _StubTrader([{"coin": "BTC", "side": "long"}])
    container = _StubContainer(trader)
    regime = {"per_coin": {"BTC": {"direction": "down", "confidence": 0.99}}}
    for _ in range(10):
        evaluate_regime_flip_exits(container, regime)
    assert trader.close_calls == []


# ── End-to-end with forecaster agreement ─────────────────────


def test_full_gate_pass_closes_when_everything_aligns(_stub_old_paper_trade, monkeypatch):
    """Position underwater + regime says DOWN at high conf + forecaster
    confirms DOWN + 2 consecutive cycles -> close."""
    monkeypatch.setattr(
        config, "REGIME_FLIP_EXIT_REQUIRE_FORECASTER_AGREE", True, raising=False,
    )
    trader = _StubTrader([{"coin": "BTC", "side": "long"}])
    forecaster = _StubForecaster({"BTC": -0.45})  # bearish, well past threshold
    container = _StubContainer(trader, forecaster=forecaster)
    regime = {"per_coin": {"BTC": {"direction": "down", "confidence": 0.85}}}
    evaluate_regime_flip_exits(container, regime)
    assert trader.close_calls == []
    evaluate_regime_flip_exits(container, regime)
    assert trader.close_calls == ["BTC"]
    # Counter should have been reset on successful close
    assert ("BTC", "long") not in _against_cycle_counters


def test_short_position_against_regime_up(_stub_old_paper_trade):
    """Symmetric check: SHORT position + regime UP at high conf for 2
    cycles -> close."""
    trader = _StubTrader([{"coin": "BTC", "side": "short"}])
    container = _StubContainer(trader)
    regime = {"per_coin": {"BTC": {"direction": "up", "confidence": 0.99}}}
    evaluate_regime_flip_exits(container, regime)
    evaluate_regime_flip_exits(container, regime)
    assert trader.close_calls == ["BTC"]


def test_close_failure_keeps_counter_for_retry(_stub_old_paper_trade):
    """If close_position returns an error, the counter is NOT reset so
    the next cycle retries -- never silently drop a flip signal."""
    trader = _StubTrader(
        [{"coin": "BTC", "side": "long"}],
        close_result={"status": "error", "message": "exchange unreachable"},
    )
    container = _StubContainer(trader)
    regime = {"per_coin": {"BTC": {"direction": "down", "confidence": 0.99}}}
    evaluate_regime_flip_exits(container, regime)
    evaluate_regime_flip_exits(container, regime)
    # close_position WAS called, but returned error
    assert trader.close_calls == ["BTC"]
    assert _against_cycle_counters[("BTC", "long")] == 2
