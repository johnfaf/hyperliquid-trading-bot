from __future__ import annotations

import sqlite3
from contextlib import contextmanager

from src.analysis import shadow_tracker as shadow_tracker_module
from src.data.db import router
from src.trading import trade_memory as trade_memory_module


def _shared_sqlite_ctx(db_file):
    @contextmanager
    def _ctx(*, for_read: bool = False):
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    return _ctx


def test_dualwrite_read_only_path_skips_postgres(monkeypatch):
    monkeypatch.setattr(router.config, "DB_BACKEND", "dualwrite")
    monkeypatch.setattr(router, "_sqlite_connect", lambda: sqlite3.connect(":memory:"))

    def _unexpected_pg():
        raise AssertionError("dualwrite read path should not open Postgres")

    monkeypatch.setattr(router, "_pg_connect", _unexpected_pg)

    with router.get_connection(for_read=True) as conn:
        row = conn.execute("SELECT 1").fetchone()

    assert row[0] == 1


def test_trade_memory_uses_shared_runtime_database(monkeypatch, tmp_path):
    shared_db = tmp_path / "shared.db"
    monkeypatch.setattr(
        trade_memory_module.db,
        "get_connection",
        _shared_sqlite_ctx(str(shared_db)),
    )
    monkeypatch.setattr(trade_memory_module.db, "get_backend_name", lambda: "sqlite")

    memory = trade_memory_module.TradeMemory(db_path=str(tmp_path / "legacy.db"))
    memory.record_trade(
        trade_id="t-1",
        coin="BTC",
        side="long",
        strategy_type="momentum",
        entry_price=100.0,
        exit_price=105.0,
        pnl=5.0,
        return_pct=0.05,
        opened_at="2026-04-12T00:00:00+00:00",
        closed_at="2026-04-12T01:00:00+00:00",
        confidence=0.8,
        source="strategy",
        regime="trend",
        setup_type="full_confluence",
        features={"momentum_score": 1.0},
    )

    stats = memory.get_stats()
    assert stats["total_trades"] == 1

    conn = sqlite3.connect(shared_db)
    try:
        row = conn.execute("SELECT COUNT(*) FROM trade_memory").fetchone()
        assert row[0] == 1
    finally:
        conn.close()


def test_shadow_tracker_uses_shared_runtime_database(monkeypatch, tmp_path):
    shared_db = tmp_path / "shared.db"
    monkeypatch.setattr(
        shadow_tracker_module.db,
        "get_connection",
        _shared_sqlite_ctx(str(shared_db)),
    )
    monkeypatch.setattr(shadow_tracker_module.db, "get_backend_name", lambda: "sqlite")
    monkeypatch.setattr(shadow_tracker_module.db, "get_db_path", lambda: str(shared_db))

    tracker = shadow_tracker_module.ShadowTracker(db_path=str(tmp_path / "legacy-shadow.db"))
    tracker.record_trade(
        {
            "signal_source": "strategy:momentum",
            "coin": "ETH",
            "side": "long",
            "entry_price": 100.0,
            "exit_price": 102.5,
            "size": 1.0,
            "entry_ts": "2026-04-12T00:00:00+00:00",
            "exit_ts": "2026-04-12T01:00:00+00:00",
            "confidence": 0.7,
        }
    )

    summary = tracker.get_summary(days=30)
    assert summary["total_trades"] == 1

    conn = sqlite3.connect(shared_db)
    try:
        row = conn.execute("SELECT COUNT(*) FROM shadow_trades").fetchone()
        assert row[0] == 1
    finally:
        conn.close()
