from __future__ import annotations

import logging
import sqlite3
import threading
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


class _TrackingLock:
    def __init__(self):
        self.enter_count = 0
        self.exit_count = 0
        self.max_active = 0
        self._active = 0
        self._inner = threading.Lock()

    def __enter__(self):
        self._inner.acquire()
        self.enter_count += 1
        self._active += 1
        self.max_active = max(self.max_active, self._active)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._active -= 1
        self.exit_count += 1
        self._inner.release()
        return False


def test_sqlite_write_path_uses_process_write_guard(monkeypatch, tmp_path):
    db_path = tmp_path / "router-write-lock.db"
    monkeypatch.setattr(router.config, "DB_BACKEND", "sqlite")
    monkeypatch.setattr(router.config, "DB_PATH", str(db_path))
    tracking_lock = _TrackingLock()
    monkeypatch.setattr(router, "_SQLITE_WRITE_LOCK", tracking_lock)

    with router.get_connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS test_rows (id INTEGER PRIMARY KEY, value TEXT)")

    def _write_row(value: str):
        with router.get_connection() as conn:
            conn.execute("INSERT INTO test_rows (value) VALUES (?)", (value,))

    t1 = threading.Thread(target=_write_row, args=("a",))
    t2 = threading.Thread(target=_write_row, args=("b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    with router.get_connection(for_read=True) as conn:
        rows = conn.execute("SELECT value FROM test_rows ORDER BY id").fetchall()

    assert [row[0] for row in rows] == ["a", "b"]
    assert tracking_lock.enter_count == 3
    assert tracking_lock.exit_count == 3
    assert tracking_lock.max_active == 1


def test_dualwrite_read_only_path_skips_process_write_guard(monkeypatch):
    tracking_lock = _TrackingLock()
    monkeypatch.setattr(router, "_SQLITE_WRITE_LOCK", tracking_lock)
    monkeypatch.setattr(router.config, "DB_BACKEND", "dualwrite")
    monkeypatch.setattr(router, "_sqlite_connect", lambda: sqlite3.connect(":memory:"))

    def _unexpected_pg():
        raise AssertionError("dualwrite read path should not open Postgres")

    monkeypatch.setattr(router, "_pg_connect", _unexpected_pg)

    with router.get_connection(for_read=True) as conn:
        row = conn.execute("SELECT 1").fetchone()

    assert row[0] == 1
    assert tracking_lock.enter_count == 0
    assert tracking_lock.exit_count == 0


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


def test_dualwrite_metadata_cas_uses_jsonb_predicate_for_postgres():
    sqlite_conn = sqlite3.connect(":memory:")
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_conn.execute(
        "CREATE TABLE paper_trades (id INTEGER PRIMARY KEY AUTOINCREMENT, metadata TEXT)"
    )
    sqlite_conn.execute("INSERT INTO paper_trades (metadata) VALUES (?)", ("",))
    pg_conn = _RecordingPgConn()
    adapter = DualWriteAdapter(sqlite_conn, pg_conn)

    adapter.execute(
        "UPDATE paper_trades SET metadata = ? "
        "WHERE id = ? AND COALESCE(metadata, '') = COALESCE(?, '')",
        ('{"guard": true}', 1, ""),
    )

    sql, params = pg_conn.executed[-1]
    assert sql == (
        "UPDATE paper_trades SET metadata = %s::jsonb "
        "WHERE id = %s AND COALESCE(metadata, '{}'::jsonb) = "
        "COALESCE(%s::jsonb, '{}'::jsonb)"
    )
    assert params == ('{"guard": true}', 1, "{}")


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


def test_migration_reader_strips_utf8_bom(tmp_path):
    migration_file = tmp_path / "0009_test.sql"
    migration_file.write_bytes(b"\xef\xbb\xbf-- comment\nCREATE TABLE t (id INTEGER);")

    sql = migrations_module._read_migration_sql(str(migration_file))

    assert sql.startswith("-- comment")
    assert "\ufeff" not in sql


def test_translate_sql_only_rewrites_datetime_function_calls():
    sql = (
        "SELECT datetime('now') AS created_at, datetime('now', '-90 days') AS cutoff, "
        "datetime_now_column FROM metrics"
    )

    translated = _translate_sql(sql, "postgres")

    assert "CURRENT_TIMESTAMP AS created_at" in translated
    assert "(now() - INTERVAL '90 days') AS cutoff" in translated
    assert "datetime_now_column" in translated


def test_translate_sql_rewrites_paper_trade_metadata_cas_for_jsonb():
    sql = (
        "UPDATE paper_trades SET metadata = ? "
        "WHERE id = ? AND COALESCE(metadata, '') = COALESCE(?, '')"
    )

    translated = _translate_sql(sql, "postgres")

    assert translated == (
        "UPDATE paper_trades SET metadata = %s::jsonb "
        "WHERE id = %s AND COALESCE(metadata, '{}'::jsonb) = "
        "COALESCE(%s::jsonb, '{}'::jsonb)"
    )


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


# ─── H4 (audit): dualwrite readiness / health surface ──────────────


class _FlakyPgCursor:
    """Postgres cursor that raises on every ``execute`` — simulates outage."""

    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=()):
        self._conn.execute_calls += 1
        raise RuntimeError("postgres-down")


class _FlakyPgConn:
    def __init__(self):
        self.execute_calls = 0
        self.rollback_calls = 0
        self.commit_calls = 0

    def cursor(self):
        return _FlakyPgCursor(self)

    def rollback(self):
        self.rollback_calls += 1

    def commit(self):
        self.commit_calls += 1


def test_dualwrite_stats_records_failures_into_rolling_window():
    from src.data.db.connection import _DualWriteStats

    stats = _DualWriteStats()
    assert stats.is_healthy()

    for _ in range(3):
        stats.record_fail("outage")
    assert stats.recent_failures(window_s=60) == 3
    assert stats.is_healthy(window_s=60, max_failures=5)

    for _ in range(5):
        stats.record_fail("outage")
    assert stats.recent_failures(window_s=60) == 8
    assert not stats.is_healthy(window_s=60, max_failures=5)

    snap = stats.snapshot()
    assert snap["pg_writes_failed"] == 8
    assert snap["recent_failures_5m"] == 8


def test_dualwrite_adapter_mirror_failure_feeds_health_counters():
    from src.data.db.connection import dualwrite_stats

    dualwrite_stats._recent_failures.clear()
    before = dualwrite_stats.snapshot()
    before_failed = before["pg_writes_failed"]

    sqlite_conn = sqlite3.connect(":memory:")
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_conn.execute(
        "CREATE TABLE strategies (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL)"
    )
    pg_conn = _FlakyPgConn()
    adapter = DualWriteAdapter(sqlite_conn, pg_conn)

    for i in range(6):
        adapter.execute(
            "INSERT INTO strategies (name) VALUES (?)",
            (f"strat-{i}",),
        )

    after = dualwrite_stats.snapshot()
    assert after["pg_writes_failed"] - before_failed >= 6
    assert pg_conn.rollback_calls >= 6
    assert not dualwrite_stats.is_healthy(window_s=60, max_failures=5)


def test_database_dualwrite_is_healthy_wrapper_respects_backend(monkeypatch):
    from src.data import database as db_module
    from src.data.db.connection import dualwrite_stats

    dualwrite_stats._recent_failures.clear()
    # Non-dualwrite backends short-circuit to True even if the shared
    # counter has stale failures.
    monkeypatch.setattr(db_module.config, "DB_BACKEND", "sqlite")
    for _ in range(50):
        dualwrite_stats.record_fail("irrelevant")
    assert db_module.dualwrite_is_healthy() is True

    # Switching the active backend surfaces the recorded failures.
    monkeypatch.setattr(db_module.config, "DB_BACKEND", "dualwrite")
    assert not db_module.dualwrite_is_healthy(window_s=60, max_failures=5)


def test_live_trader_trips_kill_switch_on_unhealthy_dualwrite(monkeypatch, tmp_path):
    from src.data import database as db_module
    from src.data.db.connection import dualwrite_stats
    from src.trading import live_trader as live_trader_module
    from src.trading.live_trader import LiveTrader

    class _AllowAllFirewall:
        def validate(self, signal, **kw):
            return True, "ok"

    def _fake_credentials(self):
        self.signer = type("Signer", (), {"address": "0x" + "1" * 40})()
        self.agent_wallet_address = self.signer.address
        self.public_address = "0x" + "2" * 40
        self.status_reason = "credentials_loaded"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    # Run in dry_run=False to enable the live-only health trip while
    # bypassing the H3 downgrade guard (dualwrite backend, no DSN required
    # for this unit-level check).
    monkeypatch.setattr(db_module.config, "DB_BACKEND", "dualwrite")
    monkeypatch.setattr(
        db_module.config, "DB_BACKEND_DOWNGRADED", False, raising=False,
    )
    # Redirect the persisted kill-switch state file to an isolated tmp
    # path so any residual file from a prior run in this repo cannot
    # flip the fresh trader into kill-switch-on before the test starts.
    ks_file = tmp_path / "ks_state.json"
    monkeypatch.setattr(
        live_trader_module.config,
        "LIVE_KILL_SWITCH_STATE_FILE",
        str(ks_file),
        raising=False,
    )

    trader = LiveTrader(
        firewall=_AllowAllFirewall(),
        dry_run=False,
        max_order_usd=1_000_000,
    )

    # Clean slate, then load 20 failures into the rolling window.
    dualwrite_stats._recent_failures.clear()
    for _ in range(20):
        dualwrite_stats.record_fail("pg-down")

    assert not trader._kill_switch_is_active()
    tripped = trader._check_dualwrite_health()
    assert tripped is True
    assert trader._kill_switch_is_active()
    assert trader._dualwrite_unhealthy_tripped is True
    # Second call in the same unhealthy episode must not double-fire.
    assert trader._check_dualwrite_health() is False


# ─── H5 (audit): paper_trades idempotency key ──────────────────────


def test_open_paper_trade_idempotency_key_deduplicates(monkeypatch, tmp_path):
    """Re-running open_paper_trade with the same key must return the same id."""
    from src.data import database as db_module

    db_file = tmp_path / "h5.db"
    monkeypatch.setattr(db_module.config, "DB_BACKEND", "sqlite")
    monkeypatch.setattr(db_module.config, "DB_PATH", str(db_file), raising=False)
    monkeypatch.setattr(
        db_module.config,
        "FIREWALL_MAX_SAME_SIDE_POSITIONS_PER_COIN",
        10,
        raising=False,
    )
    monkeypatch.setattr(db_module, "_RESOLVED_DB_PATH", str(db_file), raising=False)

    db_module.init_db()

    tid1 = db_module.open_paper_trade(
        None, "ETH", "long", 100.0, 1.0, idempotency_key="signal:abc",
    )
    tid2 = db_module.open_paper_trade(
        None, "ETH", "long", 100.0, 1.0, idempotency_key="signal:abc",
    )
    assert tid1 == tid2, (tid1, tid2)

    tid3 = db_module.open_paper_trade(
        None, "ETH", "long", 100.0, 1.0, idempotency_key="signal:xyz",
    )
    assert tid3 != tid1

    # No key at all -> legacy "always insert" behavior preserved.
    tid_a = db_module.open_paper_trade(None, "ETH", "long", 100.0, 1.0)
    tid_b = db_module.open_paper_trade(None, "ETH", "long", 100.0, 1.0)
    assert tid_a != tid_b

    # Empty / whitespace-only keys are normalized to None and therefore
    # always insert (do NOT collide with each other on the empty string).
    tid_x = db_module.open_paper_trade(
        None, "ETH", "long", 100.0, 1.0, idempotency_key="",
    )
    tid_y = db_module.open_paper_trade(
        None, "ETH", "long", 100.0, 1.0, idempotency_key="   ",
    )
    assert tid_x != tid_y


def test_open_paper_trade_idempotency_key_schema_migration(monkeypatch, tmp_path):
    """An existing SQLite DB without client_order_id must migrate cleanly."""
    import sqlite3 as _sqlite3

    from src.data import database as db_module

    db_file = tmp_path / "legacy.db"

    # Build a legacy schema: paper_trades WITHOUT client_order_id.
    conn = _sqlite3.connect(str(db_file))
    conn.executescript(
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
            metadata TEXT DEFAULT '{}'
        );
        INSERT INTO paper_trades
            (strategy_id, opened_at, coin, side, entry_price, size, leverage)
        VALUES (NULL, '2026-04-01T00:00:00+00:00', 'BTC', 'long', 50000, 0.1, 1);
        """
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(db_module.config, "DB_BACKEND", "sqlite")
    monkeypatch.setattr(db_module.config, "DB_PATH", str(db_file), raising=False)
    monkeypatch.setattr(db_module, "_RESOLVED_DB_PATH", str(db_file), raising=False)

    # Must not raise — the new init_db pre-migrates the column before
    # CREATE UNIQUE INDEX references it.
    db_module.init_db()

    # Column present and legacy row preserved with NULL client_order_id.
    raw = _sqlite3.connect(str(db_file))
    raw.row_factory = _sqlite3.Row
    cols = [r["name"] for r in raw.execute("PRAGMA table_info(paper_trades)")]
    assert "client_order_id" in cols
    row = raw.execute(
        "SELECT coin, client_order_id FROM paper_trades WHERE id=1"
    ).fetchone()
    assert row["coin"] == "BTC"
    assert row["client_order_id"] is None
    raw.close()

    # New inserts with a key still deduplicate.
    tid1 = db_module.open_paper_trade(
        None, "ETH", "long", 100.0, 1.0, idempotency_key="k1",
    )
    tid2 = db_module.open_paper_trade(
        None, "ETH", "long", 100.0, 1.0, idempotency_key="k1",
    )
    assert tid1 == tid2


def test_open_paper_trade_enforces_same_side_cap(monkeypatch, tmp_path):
    from src.data import database as db_module

    db_file = tmp_path / "cap.db"
    monkeypatch.setattr(db_module.config, "DB_BACKEND", "sqlite")
    monkeypatch.setattr(db_module.config, "DB_PATH", str(db_file), raising=False)
    monkeypatch.setattr(
        db_module.config,
        "FIREWALL_MAX_SAME_SIDE_POSITIONS_PER_COIN",
        2,
        raising=False,
    )
    monkeypatch.setattr(db_module, "_RESOLVED_DB_PATH", str(db_file), raising=False)

    db_module.init_db()
    db_module.open_paper_trade(None, "ETH", "long", 100.0, 1.0)
    db_module.open_paper_trade(None, "ETH", "long", 100.0, 1.0)

    with pytest.raises(ValueError, match="Pyramiding blocked"):
        db_module.open_paper_trade(None, "ETH", "long", 100.0, 1.0)
