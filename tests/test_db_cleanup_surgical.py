"""Tests for the surgical DB cleanup script."""
from __future__ import annotations

import importlib.util
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _load_module():
    here = Path(__file__).resolve().parent.parent
    path = here / "scripts" / "db_cleanup_surgical.py"
    spec = importlib.util.spec_from_file_location("db_cleanup_surgical", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _seed_test_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            opened_at TEXT NOT NULL,
            closed_at TEXT,
            coin TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            size REAL NOT NULL,
            pnl REAL DEFAULT 0,
            status TEXT DEFAULT 'open',
            metadata TEXT DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            strategy_type TEXT,
            current_score REAL DEFAULT 0,
            active INTEGER DEFAULT 1,
            discovered_at TEXT,
            last_scored TEXT
        )
        """
    )
    now = datetime.now(timezone.utc)
    very_old = (now - timedelta(days=60)).isoformat()
    recent = (now - timedelta(days=3)).isoformat()
    near_cutoff = (now - timedelta(days=22)).isoformat()
    open_trade = (now - timedelta(days=40)).isoformat()  # would be old, but status='open'

    conn.executemany(
        """
        INSERT INTO paper_trades
          (opened_at, closed_at, coin, side, entry_price, size, pnl, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (very_old, very_old, "BTC", "long", 70000, 0.001, 1.0, "closed"),
            (very_old, near_cutoff, "ETH", "short", 3000, 0.01, -2.0, "closed"),
            (recent, recent, "SOL", "long", 80, 0.1, 0.5, "closed"),
            (open_trade, None, "DOGE", "long", 0.1, 100, 0.0, "open"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO strategies (name, strategy_type, active, discovered_at, last_scored)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            ("old_inactive", "momentum_long", 0, very_old, very_old),  # delete
            ("recent_inactive", "mean_revert", 0, recent, recent),     # keep
            ("active_strategy", "momentum_short", 1, very_old, very_old),  # keep (active)
        ],
    )
    conn.commit()


@pytest.fixture()
def seeded_db(tmp_path, monkeypatch):
    db_path = tmp_path / "cleanup.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _seed_test_db(conn)

    @contextmanager
    def _ctx(*, for_read: bool = False):
        yield conn

    import src.data.database as db_mod
    monkeypatch.setattr(db_mod, "get_connection", _ctx)
    yield conn
    conn.close()


def test_dry_run_reports_counts_without_deleting(seeded_db, capsys):
    mod = _load_module()
    rc = mod.main(["scripts/db_cleanup_surgical.py", "--paper-days", "21", "--strategies-days", "30"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "[DRY-RUN]" in out
    # Nothing should have been deleted.
    paper_count = seeded_db.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    strat_count = seeded_db.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
    assert paper_count == 4
    assert strat_count == 3


def test_apply_deletes_old_closed_paper_trades_and_old_inactive_strategies(seeded_db, monkeypatch, capsys):
    mod = _load_module()
    monkeypatch.setattr(mod.config, "LIVE_TRADING_ENABLED", False, raising=False)

    rc = mod.main(["scripts/db_cleanup_surgical.py", "--apply"])
    assert rc == 0

    # Open trade preserved.
    open_trades = seeded_db.execute(
        "SELECT coin FROM paper_trades WHERE status = 'open'"
    ).fetchall()
    assert [t[0] for t in open_trades] == ["DOGE"]

    # Recent closed (3d) preserved; old closed (22d, 60d) deleted.
    closed = seeded_db.execute(
        "SELECT coin FROM paper_trades WHERE status = 'closed' ORDER BY coin"
    ).fetchall()
    assert [c[0] for c in closed] == ["SOL"]

    # Active strategy preserved; recent inactive preserved; very-old inactive deleted.
    strategies = seeded_db.execute(
        "SELECT name FROM strategies ORDER BY name"
    ).fetchall()
    names = {s[0] for s in strategies}
    assert "active_strategy" in names
    assert "recent_inactive" in names
    assert "old_inactive" not in names


def test_apply_refuses_in_live_mode_without_risks_flag(seeded_db, monkeypatch, capsys):
    mod = _load_module()
    monkeypatch.setattr(mod.config, "LIVE_TRADING_ENABLED", True, raising=False)

    rc = mod.main(["scripts/db_cleanup_surgical.py", "--apply"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "REFUSED" in out
    # Nothing deleted.
    paper_count = seeded_db.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    assert paper_count == 4
