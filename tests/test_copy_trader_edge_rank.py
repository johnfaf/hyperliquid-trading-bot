"""Copy-trader edge-ranked selection (algo #4).

scan_top_traders takes the top-N copyable wallets by raw PnL. When enabled, it
re-ranks by a SHRUNK win-rate edge (beta-binomial toward 0.5 by trade_count) so
a lucky 3/3 wallet can't outrank a proven 45/60, and drops wallets below a
minimum edge. Flag-gated default OFF.
"""
from __future__ import annotations

import config
from src.trading.copy_trader import CopyTrader


class _Ranker:
    """Borrow the two ranking methods without constructing a full CopyTrader."""
    _shrunk_edge = staticmethod(CopyTrader._shrunk_edge)
    _rank_traders_by_edge = CopyTrader._rank_traders_by_edge


def test_shrunk_edge_pulls_thin_samples_toward_half(monkeypatch):
    monkeypatch.setattr(config, "COPY_TRADER_EDGE_SHRINKAGE", 20.0, raising=False)
    e_thin = CopyTrader._shrunk_edge(1.0, 3)      # (3 + 10)/23  = 0.565
    e_thick = CopyTrader._shrunk_edge(0.75, 60)   # (45 + 10)/80 = 0.6875
    assert e_thick > e_thin, "a proven 75% must beat a lucky 3/3"
    assert 0.5 < e_thin < 0.6


def test_rank_orders_by_edge_and_drops_losers(monkeypatch):
    monkeypatch.setattr(config, "COPY_TRADER_EDGE_SHRINKAGE", 20.0, raising=False)
    monkeypatch.setattr(config, "COPY_TRADER_MIN_SHRUNK_EDGE", 0.55, raising=False)
    traders = [
        {"address": "0xthin", "win_rate": 1.00, "trade_count": 3},    # ~0.565 keep
        {"address": "0xthick", "win_rate": 0.75, "trade_count": 60},  # ~0.688 keep, top
        {"address": "0xloser", "win_rate": 0.30, "trade_count": 50},  # ~0.357 drop
    ]
    ranked = _Ranker()._rank_traders_by_edge(traders, top_n=10)
    addrs = [t["address"] for t in ranked]
    assert addrs and addrs[0] == "0xthick", "highest shrunk edge ranks first"
    assert "0xloser" not in addrs, "below-min-edge wallet dropped"
    assert "0xthin" in addrs


def test_rank_respects_top_n(monkeypatch):
    monkeypatch.setattr(config, "COPY_TRADER_EDGE_SHRINKAGE", 20.0, raising=False)
    monkeypatch.setattr(config, "COPY_TRADER_MIN_SHRUNK_EDGE", 0.0, raising=False)
    traders = [{"address": f"0x{i}", "win_rate": 0.6, "trade_count": 40} for i in range(10)]
    ranked = _Ranker()._rank_traders_by_edge(traders, top_n=3)
    assert len(ranked) == 3
