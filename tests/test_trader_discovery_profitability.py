"""Tests for the 90-day profitability filter on trader discovery.

The filter exists because the hardcoded SEED_TRADER_ADDRESSES list is
bull-market vintage — most addresses got onto a "successful HL trader"
list during the post-2022 BTC run and are 70-80% long-biased. Without a
profitability gate, the bot perpetually re-injects survivor-biased long
sources back into the candidate pool and produces ~88-90% long executions.
"""
from __future__ import annotations

from unittest.mock import patch

import src.discovery.trader_discovery as td


def _fill(*, dir_: str, closed_pnl: float, fee: float = 0.0) -> dict:
    """RAW Hyperliquid-shaped fill record (camelCase, as the public API
    returns it before normalization)."""
    return {"dir": dir_, "closedPnl": closed_pnl, "fee": fee}


def _norm_fill(*, dir_: str, closed_pnl: float, fee: float = 0.0) -> dict:
    """NORMALIZED fill record -- the shape ``hl.get_user_fills`` actually
    returns in production (snake_case ``direction``/``closed_pnl``).  The
    profitability filter MUST handle this shape; a regression here silently
    scores net_pnl = -fees for every real trader and inverts the gate."""
    return {"direction": dir_, "closed_pnl": closed_pnl, "fee": fee}


def test_evaluate_profitability_window_marks_winning_trader_profitable():
    fills = [
        _fill(dir_="Open Long", closed_pnl=-0.01, fee=0.01),
        _fill(dir_="Close Long", closed_pnl=2.50, fee=0.02),
        _fill(dir_="Open Short", closed_pnl=-0.01, fee=0.01),
        _fill(dir_="Close Short", closed_pnl=1.20, fee=0.02),
        _fill(dir_="Open Long", closed_pnl=-0.01, fee=0.01),
        _fill(dir_="Close Long", closed_pnl=0.80, fee=0.02),
        # Pad to >= min_trades.
        _fill(dir_="Open Long", closed_pnl=-0.01, fee=0.01),
        _fill(dir_="Close Long", closed_pnl=0.50, fee=0.02),
        _fill(dir_="Open Short", closed_pnl=-0.01, fee=0.01),
        _fill(dir_="Close Short", closed_pnl=0.30, fee=0.02),
        _fill(dir_="Open Long", closed_pnl=-0.01, fee=0.01),
        _fill(dir_="Close Long", closed_pnl=0.10, fee=0.02),
    ]
    with patch.object(td.hl, "get_user_fills", return_value=fills):
        verdict = td.evaluate_trader_profitability_window(
            "0xabcdef0000000000000000000000000000000001",
            window_days=90, min_trades=5, min_net_pnl_usd=0.0,
        )
    assert verdict["verdict"] == "profitable"
    assert verdict["trades"] >= 6
    assert verdict["net_pnl"] > 0


def test_evaluate_profitability_window_handles_normalized_production_shape():
    """REGRESSION: production ``hl.get_user_fills`` returns NORMALIZED fills
    (``direction``/``closed_pnl``), not the raw ``dir``/``closedPnl``. The
    filter previously read only the raw keys, so every real trader scored
    net_pnl = -fees and was wrongly rejected as 'unprofitable', leaving the
    discovery pool full of thin/dormant 0%-ROI accounts. A clearly winning
    trader in the NORMALIZED shape must be graded 'profitable'."""
    fills = [
        _norm_fill(dir_="Open Long", closed_pnl=-0.01, fee=0.01),
        _norm_fill(dir_="Close Long", closed_pnl=3.00, fee=0.02),
    ] * 12  # 12 closing fills of +$3 each, well above min_trades
    with patch.object(td.hl, "get_user_fills", return_value=fills):
        verdict = td.evaluate_trader_profitability_window(
            "0x" + "c" * 40, window_days=90, min_trades=10, min_net_pnl_usd=0.0,
        )
    assert verdict["verdict"] == "profitable", verdict
    assert verdict["trades"] >= 10
    assert verdict["net_pnl"] > 0


def test_evaluate_profitability_window_marks_losing_trader_unprofitable():
    fills = [
        _fill(dir_="Open Long", closed_pnl=-0.01, fee=0.01),
        _fill(dir_="Close Long", closed_pnl=-2.50, fee=0.02),
        _fill(dir_="Open Long", closed_pnl=-0.01, fee=0.01),
        _fill(dir_="Close Long", closed_pnl=-1.50, fee=0.02),
        _fill(dir_="Open Long", closed_pnl=-0.01, fee=0.01),
        _fill(dir_="Close Long", closed_pnl=-0.80, fee=0.02),
        _fill(dir_="Open Long", closed_pnl=-0.01, fee=0.01),
        _fill(dir_="Close Long", closed_pnl=0.20, fee=0.02),
        _fill(dir_="Open Long", closed_pnl=-0.01, fee=0.01),
        _fill(dir_="Close Long", closed_pnl=-0.30, fee=0.02),
    ]
    with patch.object(td.hl, "get_user_fills", return_value=fills):
        verdict = td.evaluate_trader_profitability_window(
            "0xbad0000000000000000000000000000000000000",
            window_days=90, min_trades=4, min_net_pnl_usd=0.0,
        )
    assert verdict["verdict"] == "unprofitable"
    assert verdict["net_pnl"] < 0


def test_evaluate_profitability_window_returns_insufficient_for_thin_history():
    # Only 2 closing fills, below the default min_trades.
    fills = [
        _fill(dir_="Open Long", closed_pnl=-0.01, fee=0.01),
        _fill(dir_="Close Long", closed_pnl=10.0, fee=0.02),
        _fill(dir_="Open Long", closed_pnl=-0.01, fee=0.01),
        _fill(dir_="Close Long", closed_pnl=5.0, fee=0.02),
    ]
    with patch.object(td.hl, "get_user_fills", return_value=fills):
        verdict = td.evaluate_trader_profitability_window(
            "0xnewbie000000000000000000000000000000000",
            window_days=90, min_trades=10,
        )
    assert verdict["verdict"] == "insufficient"
    assert verdict["trades"] == 2


def test_filter_profitable_addresses_keeps_profitable_drops_losers(monkeypatch):
    # Use 12 closes per side to clear the default min_trades=10 threshold.
    fills_profit = [
        _fill(dir_="Open Long", closed_pnl=-0.01, fee=0.01),
        _fill(dir_="Close Long", closed_pnl=2.0, fee=0.02),
    ] * 12
    fills_loss = [
        _fill(dir_="Open Long", closed_pnl=-0.01, fee=0.01),
        _fill(dir_="Close Long", closed_pnl=-2.0, fee=0.02),
    ] * 12
    profitable_addr = "0x" + "a" * 40
    losing_addr = "0x" + "b" * 40

    def fake_fills(addr, start_time=None):
        return fills_profit if addr == profitable_addr else fills_loss

    with patch.object(td.hl, "get_user_fills", side_effect=fake_fills):
        out = td.filter_profitable_addresses([profitable_addr, losing_addr])

    # Profitable address kept, loser dropped.
    assert profitable_addr in out
    assert losing_addr not in out


def test_filter_profitable_addresses_keeps_insufficient_by_default():
    fills_thin = [
        _fill(dir_="Open Long", closed_pnl=-0.01, fee=0.01),
        _fill(dir_="Close Long", closed_pnl=1.0, fee=0.02),
    ]  # only 1 close — below min_trades=10

    with patch.object(td.hl, "get_user_fills", return_value=fills_thin):
        out_kept = td.filter_profitable_addresses(
            ["0x" + "a" * 40], keep_insufficient=True,
        )
        out_strict = td.filter_profitable_addresses(
            ["0x" + "a" * 40], keep_insufficient=False,
        )
    assert out_kept == ["0x" + "a" * 40]
    assert out_strict == []


def test_filter_handles_fill_fetch_errors_gracefully():
    def boom(addr, start_time=None):
        raise RuntimeError("hl api 500")

    with patch.object(td.hl, "get_user_fills", side_effect=boom):
        out_kept = td.filter_profitable_addresses(
            ["0x" + "a" * 40], keep_errored=True,
        )
        out_strict = td.filter_profitable_addresses(
            ["0x" + "a" * 40], keep_errored=False,
        )
    assert out_kept == ["0x" + "a" * 40]
    assert out_strict == []
