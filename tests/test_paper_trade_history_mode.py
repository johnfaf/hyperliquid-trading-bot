"""Tests for the paper-vs-live `mode` filter on get_paper_trade_history."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager

import config
import src.data.database as db


def _seed_paper_trades(conn: sqlite3.Connection, rows: list[dict]) -> None:
    conn.execute(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER,
            opened_at TEXT NOT NULL,
            closed_at TEXT,
            coin TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            size REAL NOT NULL,
            leverage REAL DEFAULT 1,
            pnl REAL DEFAULT 0,
            status TEXT DEFAULT 'open',
            stop_loss REAL,
            take_profit REAL,
            client_order_id TEXT,
            metadata TEXT DEFAULT '{}'
        )
        """
    )
    for r in rows:
        conn.execute(
            """
            INSERT INTO paper_trades
              (opened_at, closed_at, coin, side, entry_price, size, pnl, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'closed', ?)
            """,
            (
                r["opened_at"],
                r["closed_at"],
                r["coin"],
                r["side"],
                r["entry_price"],
                r["size"],
                r["pnl"],
                json.dumps(r.get("metadata") or {}),
            ),
        )
    conn.commit()


def test_get_paper_trade_history_mode_filters_live_vs_paper(tmp_path, monkeypatch):
    db_path = tmp_path / "trades.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _seed_paper_trades(
        conn,
        [
            {
                "opened_at": "2026-04-01T00:00:00Z",
                "closed_at": "2026-04-01T01:00:00Z",
                "coin": "BTC", "side": "long", "entry_price": 70000, "size": 0.001,
                "pnl": 1.0,
                "metadata": {"live_mirror": True, "source": "strategy:foo"},
            },
            {
                "opened_at": "2026-04-02T00:00:00Z",
                "closed_at": "2026-04-02T01:00:00Z",
                "coin": "ETH", "side": "short", "entry_price": 3000, "size": 0.01,
                "pnl": -2.0,
                "metadata": {"source": "strategy:bar"},  # paper-only
            },
            {
                "opened_at": "2026-04-03T00:00:00Z",
                "closed_at": "2026-04-03T01:00:00Z",
                "coin": "SOL", "side": "long", "entry_price": 80, "size": 0.1,
                "pnl": 0.5,
                "metadata": {"live_mirror": True, "source": "copy_trade:0xabc"},
            },
        ],
    )

    @contextmanager
    def _ctx(*, for_read: bool = False):
        yield conn

    monkeypatch.setattr(db, "get_connection", _ctx)

    any_rows = db.get_paper_trade_history(limit=10, mode="any")
    live_rows = db.get_paper_trade_history(limit=10, mode="live")
    paper_rows = db.get_paper_trade_history(limit=10, mode="paper")

    assert {r["coin"] for r in any_rows} == {"BTC", "ETH", "SOL"}
    assert {r["coin"] for r in live_rows} == {"BTC", "SOL"}
    assert {r["coin"] for r in paper_rows} == {"ETH"}


def test_resolve_history_mode_for_runtime_uses_live_flag(monkeypatch):
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", True, raising=False)
    assert db._resolve_history_mode_for_runtime() == "live"

    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", False, raising=False)
    assert db._resolve_history_mode_for_runtime() == "paper"
