"""The replay P&L analyzer must mark open positions to market (no lookahead)
and report a TRUE win rate that includes the open book -- not the misleading
closed-only 100% that the old ad-hoc breakdown produced.
"""
from __future__ import annotations

import sqlite3

from scripts.analyze_replay_pnl import analyze


_PAPER_TRADES_DDL = """
CREATE TABLE paper_trades (
    id INTEGER PRIMARY KEY, strategy_id INTEGER, opened_at TEXT, closed_at TEXT,
    coin TEXT, side TEXT, entry_price REAL, exit_price REAL, size REAL,
    leverage REAL, pnl REAL, status TEXT, stop_loss REAL, take_profit REAL,
    client_order_id TEXT, metadata TEXT)
"""


def _mk_replay(path, rows):
    c = sqlite3.connect(str(path))
    c.execute(_PAPER_TRADES_DDL)
    c.executemany(
        "INSERT INTO paper_trades (id,strategy_id,opened_at,closed_at,coin,side,"
        "entry_price,exit_price,size,leverage,pnl,status,metadata) "
        "VALUES (:id,:strategy_id,:opened_at,:closed_at,:coin,:side,:entry_price,"
        ":exit_price,:size,:leverage,:pnl,:status,:metadata)", rows)
    c.commit()
    c.close()


def _mk_cache(path, candles):
    c = sqlite3.connect(str(path))
    c.execute("CREATE TABLE candles (coin TEXT, timeframe TEXT, timestamp_ms INTEGER, "
              "open REAL, high REAL, low REAL, close REAL, volume REAL)")
    c.executemany("INSERT INTO candles VALUES (?,?,?,?,?,?,?,?)", candles)
    c.commit()
    c.close()


def test_marks_open_to_market_and_true_win_rate(tmp_path):
    rdb = tmp_path / "replay.db"
    cache = tmp_path / "cache.db"
    AS_OF = 1_000_000

    _mk_replay(rdb, [
        # closed copy-long winner: +10
        dict(id=1, strategy_id=None, opened_at="2026-03-01T00:00:00+00:00",
             closed_at="2026-03-02T00:00:00+00:00", coin="BTC", side="long",
             entry_price=100, exit_price=110, size=1, leverage=1, pnl=10,
             status="closed", metadata='{"is_copy_trade": true, "source": "copy_trade"}'),
        # closed strategy-short loser: -10
        dict(id=2, strategy_id=1, opened_at="2026-03-01T00:00:00+00:00",
             closed_at="2026-03-02T00:00:00+00:00", coin="ETH", side="short",
             entry_price=200, exit_price=210, size=1, leverage=1, pnl=-10,
             status="closed", metadata='{"source": "strategy"}'),
        # OPEN copy-long: entry 100, size 2, lev 1 -> marked to BTC close at AS_OF
        dict(id=3, strategy_id=None, opened_at="2026-03-01T00:00:00+00:00",
             closed_at=None, coin="BTC", side="long", entry_price=100,
             exit_price=None, size=2, leverage=1, pnl=0, status="open",
             metadata='{"is_copy_trade": true, "source": "copy_trade"}'),
    ])
    _mk_cache(cache, [
        ("BTC", "1h", AS_OF - 3600_000, 100, 100, 100, 115, 1.0),   # earlier
        ("BTC", "1h", AS_OF,            100, 100, 100, 120, 1.0),   # the mark
        ("BTC", "1h", AS_OF + 3600_000, 100, 100, 100, 999, 1.0),   # FUTURE: must be ignored
    ])

    a = analyze(str(rdb), str(cache), as_of_ms=AS_OF)

    # realized = +10 -10 = 0
    assert a["realized_pnl"] == 0.0
    # open BTC long marked to 120 (NOT the future 999): 2*(120-100)*1 = +40
    assert a["unrealized_pnl"] == 40.0, a["unrealized_pnl"]
    assert a["combined_pnl"] == 40.0
    # closed-only win rate is the misleading 1/2; TRUE includes the open winner: 2/3
    assert a["closed_win_rate"] == 0.5
    assert a["true_win_rate"] == round(2 / 3, 4)
    assert a["true_graded"] == 3 and a["true_wins"] == 2
    # bucketing
    cl = a["buckets"]["copy_trade|long"]
    assert cl["closed_n"] == 1 and cl["open_n"] == 1
    assert cl["realized"] == 10.0 and cl["unrealized"] == 40.0
    assert cl["wins"] == 2 and cl["losses"] == 0
    ss = a["buckets"]["strategy|short"]
    assert ss["closed_n"] == 1 and ss["realized"] == -10.0 and ss["losses"] == 1


def test_unmarked_when_coin_absent_from_cache(tmp_path):
    rdb = tmp_path / "r.db"
    cache = tmp_path / "c.db"
    _mk_replay(rdb, [
        dict(id=1, strategy_id=None, opened_at="2026-03-01T00:00:00+00:00",
             closed_at=None, coin="DOGE", side="long", entry_price=1,
             exit_price=None, size=1, leverage=1, pnl=0, status="open",
             metadata='{"is_copy_trade": true}'),
    ])
    _mk_cache(cache, [("BTC", "1h", 1_000_000, 1, 1, 1, 1, 1.0)])  # no DOGE
    a = analyze(str(rdb), str(cache), as_of_ms=1_000_000)
    assert a["unmarked_open_ids"] == [1]
    assert a["unrealized_pnl"] == 0.0          # unmarked excluded from total
    assert a["true_graded"] == 0               # nothing gradable
