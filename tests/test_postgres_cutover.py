from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager

import pytest

from src.analysis import shadow_tracker as shadow_tracker_module
from src.data.db import migrations as migrations_module
from src.data.db import postgres as postgres_module
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


def test_localhost_postgres_dsn_is_allowed_for_local_development():
    assert postgres_module.get_postgres_config_error(
        "postgres",
        "postgresql://postgres:secret@localhost:5432/hyperliquid_bot",
    ) == ""


def test_hosted_localhost_postgres_dsn_is_rejected_on_railway(monkeypatch):
    monkeypatch.setenv("RAILWAY_ENVIRONMENT_ID", "env_123")

    error = postgres_module.get_postgres_config_error(
        "postgres",
        "postgresql://postgres:secret@localhost:5432/hyperliquid_bot",
    )

    assert "localhost" in error
    assert "managed Postgres" in error


def test_dualwrite_init_postgres_schema_degrades_when_postgres_is_misconfigured(monkeypatch, caplog):
    monkeypatch.setenv("RAILWAY_ENVIRONMENT_ID", "env_123")
    monkeypatch.setattr(router.config, "DB_BACKEND", "dualwrite")
    monkeypatch.setattr(
        router.config,
        "POSTGRES_DSN",
        "postgresql://postgres:secret@localhost:5432/hyperliquid_bot",
    )

    called = {"value": False}

    def _should_not_run():
        called["value"] = True

    monkeypatch.setattr(migrations_module, "run_migrations", _should_not_run)

    with caplog.at_level(logging.WARNING):
        router.init_postgres_schema()

    assert called["value"] is False
    assert "SQLite will remain authoritative" in caplog.text


def test_postgres_init_postgres_schema_raises_when_postgres_is_misconfigured(monkeypatch):
    monkeypatch.setenv("RAILWAY_ENVIRONMENT_ID", "env_123")
    monkeypatch.setattr(router.config, "DB_BACKEND", "postgres")
    monkeypatch.setattr(
        router.config,
        "POSTGRES_DSN",
        "postgresql://postgres:secret@localhost:5432/hyperliquid_bot",
    )

    with pytest.raises(RuntimeError, match="localhost"):
        router.init_postgres_schema()


def test_dualwrite_init_postgres_schema_degrades_when_migrations_fail(monkeypatch, caplog):
    monkeypatch.delenv("RAILWAY_ENVIRONMENT_ID", raising=False)
    monkeypatch.setattr(router.config, "DB_BACKEND", "dualwrite")
    monkeypatch.setattr(
        router.config,
        "POSTGRES_DSN",
        "postgresql://postgres:secret@db.example.com:5432/hyperliquid_bot?sslmode=require",
    )

    def _boom():
        raise RuntimeError("db unreachable")

    monkeypatch.setattr(migrations_module, "run_migrations", _boom)

    with caplog.at_level(logging.WARNING):
        router.init_postgres_schema()

    assert "migrations could not run" in caplog.text
    assert "SQLite will remain authoritative" in caplog.text


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
