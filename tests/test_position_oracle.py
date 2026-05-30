"""Tests for the replay TraderPositionOracle (copy-trade source positions)."""
from __future__ import annotations

import sqlite3

from src.backtest.replay.position_oracle import (
    TraderPositionOracle,
    _signed_size,
)

ADDR = "0x" + "ab" * 20


def _fill(coin, direction, size, px, t, side=""):
    return {"coin": coin, "direction": direction, "size": size,
            "original_price": px, "time_ms": t, "side": side}


# ── _signed_size ────────────────────────────────────────────────


def test_signed_size_directions():
    assert _signed_size("Open Long", "", 2) == 2
    assert _signed_size("Close Long", "", 2) == -2
    assert _signed_size("Open Short", "", 3) == -3
    assert _signed_size("Close Short", "", 3) == 3


def test_signed_size_side_fallback():
    assert _signed_size("", "B", 1) == 1
    assert _signed_size("", "buy", 1) == 1
    assert _signed_size("", "A", 1) == -1
    assert _signed_size("", "sell", 1) == -1
    assert _signed_size("", "", 1) == 0


# ── position reconstruction ─────────────────────────────────────


def test_open_long_appears():
    o = TraderPositionOracle({ADDR: [_fill("BTC", "Open Long", 2.0, 100.0, 1000)]})
    pos = o._net_by_coin(ADDR, 2000)
    assert "BTC" in pos and pos["BTC"][0] == 2.0
    assert pos["BTC"][1] == 100.0  # entry px


def test_partial_close_reduces():
    o = TraderPositionOracle({ADDR: [
        _fill("BTC", "Open Long", 4.0, 100.0, 1000),
        _fill("BTC", "Close Long", 1.0, 110.0, 2000),
    ]})
    assert o._net_by_coin(ADDR, 3000)["BTC"][0] == 3.0


def test_full_close_removes_position():
    o = TraderPositionOracle({ADDR: [
        _fill("BTC", "Open Long", 2.0, 100.0, 1000),
        _fill("BTC", "Close Long", 2.0, 110.0, 2000),
    ]})
    assert "BTC" not in o._net_by_coin(ADDR, 3000)


def test_time_filtering_excludes_future_fills():
    o = TraderPositionOracle({ADDR: [
        _fill("BTC", "Open Long", 2.0, 100.0, 5000),
    ]})
    assert o._net_by_coin(ADDR, 1000) == {}        # before the fill
    assert "BTC" in o._net_by_coin(ADDR, 5000)      # at/after the fill


def test_short_position_is_negative():
    o = TraderPositionOracle({ADDR: [_fill("ETH", "Open Short", 5.0, 50.0, 1000)]})
    assert o._net_by_coin(ADDR, 2000)["ETH"][0] == -5.0


def test_entry_px_is_size_weighted():
    o = TraderPositionOracle({ADDR: [
        _fill("BTC", "Open Long", 1.0, 100.0, 1000),
        _fill("BTC", "Open Long", 3.0, 200.0, 2000),
    ]})
    # (1*100 + 3*200) / 4 = 175
    assert o._net_by_coin(ADDR, 3000)["BTC"][1] == 175.0


def test_multiple_coins_independent():
    o = TraderPositionOracle({ADDR: [
        _fill("BTC", "Open Long", 2.0, 100.0, 1000),
        _fill("SOL", "Open Short", 10.0, 5.0, 1500),
    ]})
    pos = o._net_by_coin(ADDR, 2000)
    assert pos["BTC"][0] == 2.0 and pos["SOL"][0] == -10.0


# ── clearinghouse_state shape ───────────────────────────────────


def test_clearinghouse_state_shape():
    o = TraderPositionOracle({ADDR: [_fill("BTC", "Open Long", 2.0, 100.0, 1000)]})
    chs = o.clearinghouse_state(ADDR, 2000)
    assert "assetPositions" in chs
    ap = chs["assetPositions"]
    assert len(ap) == 1
    p = ap[0]["position"]
    assert p["coin"] == "BTC" and p["szi"] == "2.0"


def test_unknown_address_is_empty():
    o = TraderPositionOracle({ADDR: [_fill("BTC", "Open Long", 2.0, 100.0, 1000)]})
    chs = o.clearinghouse_state("0xdead", 2000)
    assert chs["assetPositions"] == []


# ── from_db ─────────────────────────────────────────────────────


def test_from_db_loads_and_filters(tmp_path):
    db = tmp_path / "wf.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE wallet_fills (wallet_address TEXT, coin TEXT, side TEXT, "
        "original_price REAL, size REAL, time_ms INTEGER, direction TEXT)"
    )
    conn.executemany(
        "INSERT INTO wallet_fills VALUES (?,?,?,?,?,?,?)",
        [
            (ADDR, "BTC", "B", 100.0, 2.0, 1000, "Open Long"),
            ("0xother", "ETH", "A", 50.0, 1.0, 1000, "Open Short"),
        ],
    )
    conn.commit()
    conn.close()

    # No filter: both wallets loaded.
    o = TraderPositionOracle.from_db(str(db))
    assert set(o.addresses()) == {ADDR.lower(), "0xother"}
    # Address filter: only the requested wallet.
    o2 = TraderPositionOracle.from_db(str(db), addresses=[ADDR])
    assert o2.addresses() == [ADDR.lower()]
    assert o2._net_by_coin(ADDR, 2000)["BTC"][0] == 2.0


def test_from_db_missing_file_is_empty():
    o = TraderPositionOracle.from_db("/nonexistent/path.db")
    assert o.addresses() == []
