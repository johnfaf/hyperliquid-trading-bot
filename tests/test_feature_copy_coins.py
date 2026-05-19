"""#3 Feature precompute covers recent copy-trade candidate coins.

Copy signals on the broad tracked-trader coin set (ALGO/FARTCOIN/POL/
ADA/PUMP/STRK...) were dropped with
data_readiness_missing:candles,feature_vector because those coins were
never in the watched/feature-precompute universe. They now are.
"""
from __future__ import annotations

from contextlib import contextmanager


import src.core.cycles.feature_cycle as fc
import src.data.database as db
import src.data.hyperliquid_client as hl


class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _Conn:
    def execute(self, sql, params=()):
        s = " ".join(str(sql).split())
        if "source LIKE 'copy_trade%'" in s:
            return _Cursor([
                {"coin": "FARTCOIN", "m": "2026-05-19T05:00:00"},
                {"coin": "POL", "m": "2026-05-19T04:59:00"},
                {"coin": "PUMP", "m": "2026-05-19T04:58:00"},
            ])
        return _Cursor([])  # positions / strategies / other -> empty


@contextmanager
def _fake_conn(*a, **k):
    yield _Conn()


def test_copy_candidate_coins_enter_watched_universe(monkeypatch):
    monkeypatch.setattr(db, "get_connection", _fake_conn)
    monkeypatch.setattr(db, "table_exists", lambda name: True)
    # keep the volume fallback from adding noise / network
    monkeypatch.setattr(hl, "get_all_coins", lambda: [])

    coins = fc._get_watched_coins()
    assert "BTC" in coins and "ETH" in coins      # always-on base
    assert "FARTCOIN" in coins                     # folded-in copy candidates
    assert "POL" in coins
    assert "PUMP" in coins


def test_copy_candidate_cap_respected(monkeypatch):
    # cap=0 -> the copy block is skipped entirely (no copy coins added)
    monkeypatch.setattr(db, "get_connection", _fake_conn)
    monkeypatch.setattr(db, "table_exists", lambda name: True)
    monkeypatch.setattr(hl, "get_all_coins", lambda: [])
    import config
    monkeypatch.setattr(config, "FEATURE_COPY_CANDIDATE_COINS_MAX", 0, raising=False)

    coins = fc._get_watched_coins()
    assert "FARTCOIN" not in coins
    assert "BTC" in coins  # base still present
