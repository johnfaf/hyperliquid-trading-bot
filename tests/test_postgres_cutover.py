from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager

import pytest

from src.analysis import shadow_tracker as shadow_tracker_module
from src.data import database as db
from src.data.db.connection import DualWriteAdapter, _translate_sql
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


class _RecordingPgCursor:
    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=()):
        self._conn.executed.append((sql, tuple(params or ())))

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    @property
    def rowcount(self):
        return 1


class _RecordingPgConn:
    def __init__(self):
        self.executed = []
        self.rollback_calls = 0
        self.commit_calls = 0

    def cursor(self):
        return _RecordingPgCursor(self)

    def rollback(self):
        self.rollback_calls += 1

    def commit(self):
        self.commit_calls += 1


class _MigrationTransaction:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        self._conn.transaction_entries += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self._conn.transaction_exits += 1
        return False


class _MigrationCursor:
    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=()):
        self._conn.executed.append((sql, tuple(params or ())))


class _MigrationConn:
    def __init__(self):
        self.executed = []
        self.commit_calls = 0
        self.rollback_calls = 0
        self.transaction_entries = 0
        self.transaction_exits = 0

    def transaction(self):
        return _MigrationTransaction(self)

    def cursor(self):
        return _MigrationCursor(self)

    def commit(self):
        self.commit_calls += 1

    def rollback(self):
        self.rollback_calls += 1


def test_dualwrite_read_only_path_skips_postgres(monkeypatch):
    monkeypatch.setattr(router.config, "DB_BACKEND", "dualwrite")
    monkeypatch.setattr(router, "_sqlite_connect", lambda: sqlite3.connect(":memory:"))

    def _unexpected_pg():
        raise AssertionError("dualwrite read path should not open Postgres")

    monkeypatch.setattr(router, "_pg_connect", _unexpected_pg)

    with router.get_connection(for_read=True) as conn:
        row = conn.execute("SELECT 1").fetchone()

    assert row[0] == 1


def test_dualwrite_insert_preserves_sqlite_generated_ids_in_postgres():
    sqlite_conn = sqlite3.connect(":memory:")
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_conn.execute(
        "CREATE TABLE strategies (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL)"
    )
    pg_conn = _RecordingPgConn()
    adapter = DualWriteAdapter(sqlite_conn, pg_conn)

    cursor = adapter.execute(
        "INSERT INTO strategies (name) VALUES (?)",
        ("momentum",),
    )

    assert cursor.lastrowid == 1
    assert pg_conn.executed[-1][0].startswith("INSERT INTO strategies (id, name) VALUES (%s, %s)")
    assert pg_conn.executed[-1][1] == (1, "momentum")
    assert pg_conn.commit_calls == 0
    adapter.commit()
    assert pg_conn.commit_calls == 1


def test_dualwrite_skips_sqlite_only_pragmas_for_postgres():
    sqlite_conn = sqlite3.connect(":memory:")
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_conn.execute(
        "CREATE TABLE strategies (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL)"
    )
    pg_conn = _RecordingPgConn()
    adapter = DualWriteAdapter(sqlite_conn, pg_conn)

    adapter.execute("PRAGMA table_info(strategies)")

    assert pg_conn.executed == []


def test_dualwrite_skips_selects_for_postgres_mirror():
    sqlite_conn = sqlite3.connect(":memory:")
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_conn.execute(
        "CREATE TABLE strategies (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL)"
    )
    sqlite_conn.execute("INSERT INTO strategies (name) VALUES (?)", ("momentum",))
    pg_conn = _RecordingPgConn()
    adapter = DualWriteAdapter(sqlite_conn, pg_conn)

    rows = adapter.execute(
        "SELECT * FROM strategies WHERE name = ?",
        ("momentum",),
    ).fetchall()

    assert len(rows) == 1
    assert pg_conn.executed == []


def test_table_exists_uses_backend_agnostic_query(monkeypatch):
    observed = {}

    class _Conn:
        def execute(self, sql, params=()):
            observed["sql"] = sql
            observed["params"] = params

            class _Cursor:
                @staticmethod
                def fetchone():
                    return {"name": "paper_trades"}

            return _Cursor()

    @contextmanager
    def _ctx(*, for_read: bool = False):
        observed["for_read"] = for_read
        yield _Conn()

    monkeypatch.setattr(db, "get_connection", _ctx)

    assert db.table_exists("paper_trades") is True
    assert observed["for_read"] is True
    assert "sqlite_master" in observed["sql"]
    assert observed["params"] == ("paper_trades",)


def test_update_paper_trade_metadata_raises_when_trade_is_missing(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metadata TEXT
        )
        """
    )

    @contextmanager
    def _ctx(*, for_read: bool = False):
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    monkeypatch.setattr(db, "get_connection", _ctx)

    with pytest.raises(LookupError, match="does not exist"):
        db.update_paper_trade_metadata(123, {"foo": "bar"})


def test_update_paper_account_raises_when_singleton_missing(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE paper_account (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            balance REAL NOT NULL,
            total_pnl REAL DEFAULT 0,
            total_trades INTEGER DEFAULT 0,
            winning_trades INTEGER DEFAULT 0,
            last_updated TEXT NOT NULL
        )
        """
    )

    @contextmanager
    def _ctx(*, for_read: bool = False):
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    monkeypatch.setattr(db, "get_connection", _ctx)

    with pytest.raises(LookupError, match="paper_account singleton"):
        db.update_paper_account(1000.0, 0.0, 0, 0)


def test_run_migrations_wraps_each_file_in_transaction(monkeypatch, tmp_path):
    migration_file = tmp_path / "0001_test.sql"
    migration_file.write_text("CREATE TABLE test_table (id INTEGER);", encoding="utf-8")
    conn = _MigrationConn()

    monkeypatch.setattr(migrations_module, "_ensure_schema_migrations_table", lambda _conn: None)
    monkeypatch.setattr(migrations_module, "_applied_versions", lambda _conn: set())
    monkeypatch.setattr(
        migrations_module,
        "_discover_migrations",
        lambda: [("0001", migration_file.name, str(migration_file))],
    )
    monkeypatch.setattr("src.data.db.postgres.get_connection", lambda: conn)
    monkeypatch.setattr("src.data.db.postgres.return_connection", lambda _conn: None)

    applied = migrations_module.run_migrations()

    assert applied == 1
    assert conn.transaction_entries == 1
    assert conn.transaction_exits == 1
    assert conn.commit_calls == 1
    assert conn.executed[0][0] == "CREATE TABLE test_table (id INTEGER);"
    assert conn.executed[1][0].startswith("INSERT INTO schema_migrations")


def test_translate_sql_only_rewrites_datetime_function_calls():
    sql = (
        "SELECT datetime('now') AS created_at, datetime('now', '-90 days') AS cutoff, "
        "datetime_now_column FROM metrics"
    )

    translated = _translate_sql(sql, "postgres")

    assert "CURRENT_TIMESTAMP AS created_at" in translated
    assert "(now() - INTERVAL '90 days') AS cutoff" in translated
    assert "datetime_now_column" in translated


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


def test_ensure_postgres_strategy_parent_backfills_from_sqlite_in_postgres_mode(monkeypatch, tmp_path):
    sqlite_db = tmp_path / "runtime.db"
    conn = sqlite3.connect(sqlite_db)
    try:
        conn.execute(
            """
            CREATE TABLE strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                strategy_type TEXT NOT NULL,
                parameters TEXT DEFAULT '{}',
                discovered_at TEXT NOT NULL,
                last_scored TEXT,
                current_score REAL DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                trade_count INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                active INTEGER DEFAULT 1
            )
            """
        )
        conn.execute(
            """
            INSERT INTO strategies
            (id, name, description, strategy_type, parameters, discovered_at,
             last_scored, current_score, total_pnl, trade_count, win_rate,
             sharpe_ratio, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                7,
                "btc_momentum",
                "test",
                "momentum",
                "{}",
                "2026-04-14T00:00:00+00:00",
                None,
                0.9,
                1.2,
                5,
                0.6,
                1.1,
                1,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    pg_conn = _RecordingPgConn()
    monkeypatch.setattr(db.config, "DB_BACKEND", "postgres")
    monkeypatch.setattr(db, "_DB_PATH", str(sqlite_db))
    monkeypatch.setattr(postgres_module, "get_connection", lambda: pg_conn)
    monkeypatch.setattr(postgres_module, "return_connection", lambda conn: None)

    db._ensure_postgres_strategy_parent(7)

    insert_sql, insert_params = pg_conn.executed[-2]
    assert "INSERT INTO strategies" in insert_sql
    assert insert_params[0] == 7
    assert insert_params[1] == "btc_momentum"
    assert insert_params[-1] is True
    assert "setval" in pg_conn.executed[-1][0]


def test_active_queries_bind_boolean_parameters(monkeypatch):
    class _DummyCursor:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class _DummyConn:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params=()):
            self.calls.append((sql, tuple(params or ())))
            return _DummyCursor([])

    dummy = _DummyConn()

    @contextmanager
    def _ctx(*, for_read: bool = False):
        yield dummy

    monkeypatch.setattr(db, "get_connection", _ctx)

    db.get_active_traders()
    db.get_active_strategies()
    db.get_known_bot_addresses()

    assert dummy.calls[0] == (
        "SELECT * FROM traders WHERE active = ? ORDER BY total_pnl DESC",
        (True,),
    )
    assert dummy.calls[1] == (
        "SELECT * FROM strategies WHERE active = ? ORDER BY current_score DESC",
        (True,),
    )
    assert dummy.calls[2] == (
        "SELECT address FROM traders WHERE active = ?",
        (False,),
    )


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
