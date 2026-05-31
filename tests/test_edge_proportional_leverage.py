"""Edge-proportional leverage (algo #5).

Legacy: leverage = min(followed-trader avg_leverage, MAX) -- the bot inherits
the source's leverage habit, which is how the *unproven* strategy bucket became
the 5x-leveraged loser. When enabled, leverage scales with the signal's
(calibrated) edge confidence: no proven edge -> 1x; ramps to MAX only as
confidence rises. OFF reproduces the legacy inheritance.
"""
from __future__ import annotations

import config
from src.trading.paper_trader import PaperTrader


def _enable_edge(monkeypatch, mn=0.50, full=0.65, max_lev=5):
    monkeypatch.setattr(config, "LEVERAGE_EDGE_PROPORTIONAL_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "LEVERAGE_EDGE_MIN_CONF", mn, raising=False)
    monkeypatch.setattr(config, "LEVERAGE_EDGE_FULL_CONF", full, raising=False)
    monkeypatch.setattr(config, "PAPER_TRADING_MAX_LEVERAGE", max_lev, raising=False)


def test_legacy_inherits_avg_leverage(monkeypatch):
    monkeypatch.setattr(config, "LEVERAGE_EDGE_PROPORTIONAL_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "PAPER_TRADING_MAX_LEVERAGE", 5, raising=False)
    assert PaperTrader._resolve_leverage(0.50, 3) == 3.0       # inherits trader habit
    assert PaperTrader._resolve_leverage(0.90, 8) == 5.0       # capped at MAX
    assert PaperTrader._resolve_leverage(0.90, 0) == 1.0       # floored at 1x


def test_edge_unproven_source_is_1x(monkeypatch):
    _enable_edge(monkeypatch)
    # confidence at/below the floor => no proven edge => 1x, regardless of habit
    assert PaperTrader._resolve_leverage(0.50, 5) == 1.0
    assert PaperTrader._resolve_leverage(0.40, 5) == 1.0


def test_edge_scales_with_confidence(monkeypatch):
    _enable_edge(monkeypatch)
    # midpoint conf -> halfway leverage: 1 + 0.5*(5-1) = 3.0
    assert abs(PaperTrader._resolve_leverage(0.575, 2) - 3.0) < 1e-6
    # at/above FULL_CONF -> max leverage
    assert PaperTrader._resolve_leverage(0.65, 2) == 5.0
    assert PaperTrader._resolve_leverage(0.90, 2) == 5.0


def test_edge_mode_ignores_trader_leverage_habit(monkeypatch):
    """The inversion fix: a high-habit-leverage but low-confidence source must
    NOT get leveraged up."""
    _enable_edge(monkeypatch)
    assert PaperTrader._resolve_leverage(0.50, 8) == 1.0   # habit 8x ignored
