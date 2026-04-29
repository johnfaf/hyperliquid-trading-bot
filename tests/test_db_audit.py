import sqlite3
from contextlib import contextmanager

from src.data import database as db
from src.data import db_audit
from src.data.db import router


@contextmanager
def _connection_ctx(conn):
    try:
        yield conn
    finally:
        pass


def test_sqlite_router_enables_foreign_keys(monkeypatch, tmp_path):
    db_path = tmp_path / "bot.db"
    monkeypatch.setattr(router.config, "DB_PATH", str(db_path))

    conn = router._sqlite_connect()
    try:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        conn.close()


def test_db_audit_detects_unprotected_open_trades_and_account_mismatch(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE traders (address TEXT PRIMARY KEY);
        CREATE TABLE strategies (id INTEGER PRIMARY KEY);
        CREATE TABLE bot_state (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE paper_account (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            balance REAL NOT NULL,
            total_pnl REAL DEFAULT 0,
            total_trades INTEGER DEFAULT 0,
            winning_trades INTEGER DEFAULT 0,
            last_updated TEXT NOT NULL
        );
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
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        );
        CREATE TABLE audit_trail (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT DEFAULT '{}'
        );
        CREATE TABLE decision_snapshots (
            decision_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            coin TEXT,
            firewall_decision TEXT,
            final_status TEXT NOT NULL DEFAULT 'candidate'
        );
        CREATE TABLE source_inventory (
            source_name TEXT PRIMARY KEY,
            required INTEGER NOT NULL DEFAULT 0,
            expected_freshness_seconds INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE data_source_health_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            observed_at TEXT NOT NULL,
            source_name TEXT NOT NULL,
            status TEXT NOT NULL,
            freshness_seconds REAL,
            reason TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO paper_account "
        "(id, balance, total_pnl, total_trades, winning_trades, last_updated) "
        "VALUES (1, 10000, 10, 99, 0, '2026-04-23T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO paper_trades "
        "(opened_at, coin, side, entry_price, size, leverage, status) "
        "VALUES ('2026-04-23T00:00:00+00:00', 'ETH', 'long', 3000, 0.1, 5, 'open')"
    )

    monkeypatch.setattr(db, "get_backend_name", lambda: "sqlite")
    monkeypatch.setattr(db, "get_db_path", lambda: "test.db")
    monkeypatch.setattr(db, "get_connection", lambda for_read=False: _connection_ctx(conn))

    report = db_audit.run_db_audit(include_candle_cache=False, include_code_scan=False)
    checks = {finding.check for finding in report.findings}

    assert "open_trades_missing_protection" in checks
    assert "paper_account_trade_count" in checks
    assert report.findings_at_or_above("high")


def test_db_repair_backfills_safe_local_state(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE traders (
            address TEXT PRIMARY KEY,
            first_seen TEXT NOT NULL,
            last_updated TEXT NOT NULL,
            total_pnl REAL DEFAULT 0,
            roi_pct REAL DEFAULT 0,
            account_value REAL DEFAULT 0,
            win_rate REAL DEFAULT 0,
            trade_count INTEGER DEFAULT 0,
            active INTEGER DEFAULT 1,
            metadata TEXT DEFAULT '{}'
        );
        CREATE TABLE strategies (
            id INTEGER PRIMARY KEY,
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
        );
        CREATE TABLE strategy_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            score REAL DEFAULT 0,
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        );
        CREATE TABLE position_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trader_address TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            coin TEXT NOT NULL,
            side TEXT NOT NULL,
            size REAL NOT NULL,
            entry_price REAL NOT NULL,
            leverage REAL DEFAULT 1,
            unrealized_pnl REAL DEFAULT 0,
            margin_used REAL DEFAULT 0,
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (trader_address) REFERENCES traders(address)
        );
        CREATE TABLE bot_state (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE paper_account (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            balance REAL NOT NULL,
            total_pnl REAL DEFAULT 0,
            total_trades INTEGER DEFAULT 0,
            winning_trades INTEGER DEFAULT 0,
            last_updated TEXT NOT NULL
        );
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
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        );
        CREATE TABLE audit_trail (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            coin TEXT,
            side TEXT,
            price REAL,
            size REAL,
            pnl REAL,
            source TEXT,
            details TEXT DEFAULT '{}'
        );
        CREATE TABLE decision_snapshots (
            decision_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            signal_id TEXT,
            coin TEXT,
            firewall_decision TEXT,
            final_status TEXT NOT NULL DEFAULT 'candidate',
            rejection_reason TEXT,
            paper_trade_id INTEGER,
            live_order_id TEXT,
            metadata TEXT DEFAULT '{}'
        );
        CREATE TABLE source_inventory (
            source_name TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            required INTEGER NOT NULL DEFAULT 0,
            supports_live INTEGER NOT NULL DEFAULT 1,
            supports_historical INTEGER NOT NULL DEFAULT 0,
            point_in_time_safe INTEGER NOT NULL DEFAULT 0,
            min_history_days INTEGER NOT NULL DEFAULT 0,
            expected_freshness_seconds INTEGER NOT NULL DEFAULT 0,
            owner TEXT NOT NULL DEFAULT 'bot',
            notes TEXT NOT NULL DEFAULT '',
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE data_source_health_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            observed_at TEXT NOT NULL,
            source_name TEXT NOT NULL,
            status TEXT NOT NULL,
            freshness_seconds REAL,
            reason TEXT,
            metadata TEXT NOT NULL DEFAULT '{}'
        );
        """
    )
    conn.execute(
        "INSERT INTO paper_account "
        "(id, balance, total_pnl, total_trades, winning_trades, last_updated) "
        "VALUES (1, 10000, 0, 0, 0, '2026-04-23T00:00:00+00:00')"
    )
    conn.execute(
        """
        INSERT INTO paper_trades
        (opened_at, coin, side, entry_price, size, leverage, status, stop_loss, take_profit, metadata)
        VALUES ('2026-04-20T12:38:54+00:00', 'ETH', 'long', 100, 1, 1, 'open', NULL, NULL, '{}')
        """
    )
    conn.execute(
        """
        INSERT INTO paper_trades
        (opened_at, closed_at, coin, side, entry_price, exit_price, size, leverage,
         pnl, status, stop_loss, take_profit, metadata)
        VALUES ('2026-04-20T10:00:00+00:00', '2026-04-20T11:00:00+00:00',
                'BTC', 'long', 100, 105, 1, 1, 5, 'closed', 90, 110, '{}')
        """
    )
    conn.execute(
        """
        INSERT INTO paper_trades
        (opened_at, closed_at, coin, side, entry_price, exit_price, size, leverage,
         pnl, status, stop_loss, take_profit, metadata)
        VALUES ('2026-04-20T11:00:00+00:00', '2026-04-20T12:00:00+00:00',
                'ETH', 'short', 100, 102, 1, 1, -2, 'closed', 105, 90, '{}')
        """
    )
    conn.execute(
        """
        INSERT INTO decision_snapshots
        (decision_id, created_at, updated_at, signal_id, coin, firewall_decision, final_status, metadata)
        VALUES ('dec-1', '2026-04-20T12:00:00+00:00', '2026-04-20T12:00:00+00:00',
                'sig-1', 'ETH', 'pending', 'candidate', '{}')
        """
    )
    conn.execute(
        """
        INSERT INTO audit_trail (timestamp, action, coin, side, source, details)
        VALUES ('2026-04-10T03:21:33+00:00', 'signal_approved', 'BTC', 'short', 'strategy', '{}')
        """
    )
    conn.commit()
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute(
        """
        INSERT INTO position_snapshots
        (trader_address, timestamp, coin, side, size, entry_price, leverage, unrealized_pnl, margin_used, metadata)
        VALUES ('0x1111111111111111111111111111111111111111', '2026-04-09T13:10:16+00:00',
                'BTC', 'long', 1, 100, 1, 0, 0, '{}')
        """
    )
    conn.execute(
        """
        INSERT INTO strategy_scores (strategy_id, timestamp, score)
        VALUES (6, '2026-04-09T13:14:22+00:00', 0.5)
        """
    )
    conn.commit()
    conn.execute("PRAGMA foreign_keys=ON")

    monkeypatch.setattr(db, "get_backend_name", lambda: "sqlite")
    monkeypatch.setattr(db, "get_db_path", lambda: "test.db")
    monkeypatch.setattr(db, "get_connection", lambda for_read=False: _connection_ctx(conn))
    from src.learning import source_inventory as source_inventory_module

    monkeypatch.setattr(
        source_inventory_module,
        "persist_source_health_snapshot",
        lambda *args, **kwargs: 5,
    )

    report = db_audit.run_db_repair(
        include_candle_cache=False,
        include_code_scan=False,
        repair_live_data=False,
    )

    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    ).fetchone()[0] == 1
    repaired_trade = conn.execute(
        "SELECT stop_loss, take_profit, metadata FROM paper_trades WHERE id = 1"
    ).fetchone()
    assert repaired_trade["stop_loss"] > 0
    assert repaired_trade["take_profit"] > repaired_trade["stop_loss"]

    repaired_account = conn.execute(
        "SELECT total_pnl, total_trades, winning_trades FROM paper_account WHERE id = 1"
    ).fetchone()
    assert repaired_account["total_pnl"] == 3
    assert repaired_account["total_trades"] == 2
    assert repaired_account["winning_trades"] == 1

    repaired_decision = conn.execute(
        "SELECT final_status, firewall_decision FROM decision_snapshots WHERE decision_id = 'dec-1'"
    ).fetchone()
    assert repaired_decision["final_status"] == "expired"
    assert repaired_decision["firewall_decision"] == "expired"

    repaired_trader = conn.execute(
        "SELECT active FROM traders WHERE address = '0x1111111111111111111111111111111111111111'"
    ).fetchone()
    assert repaired_trader["active"] == 0

    repaired_strategy = conn.execute(
        "SELECT active, strategy_type FROM strategies WHERE id = 6"
    ).fetchone()
    assert repaired_strategy is None
    assert conn.execute(
        "SELECT COUNT(*) FROM strategy_scores WHERE strategy_id = 6"
    ).fetchone()[0] == 0
    assert any(
        action.action == "orphan_strategy_score_parents"
        and action.status == "applied"
        and action.details.get("deleted") == 1
        for action in report.actions
    )

    assert conn.execute("SELECT COUNT(*) FROM source_inventory").fetchone()[0] > 0
    assert not report.post_audit.findings_at_or_above("high")


def test_strategy_repair_prunes_synthetic_and_caps_missing_source_rows(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE strategies (
            id INTEGER PRIMARY KEY,
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
        );
        CREATE TABLE strategy_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            score REAL DEFAULT 0
        );
        """
    )
    conn.executemany(
        """
        INSERT INTO strategies
        (id, name, description, strategy_type, parameters, discovered_at, last_scored, active)
        VALUES (?, ?, '', ?, ?, ?, ?, ?)
        """,
        [
            (
                1,
                "valid_inactive",
                "momentum_long",
                '{"source_wallet":"0x1111111111111111111111111111111111111111"}',
                "2026-04-10T00:00:00+00:00",
                "2026-04-20T00:00:00+00:00",
                0,
            ),
            (
                2,
                "recovered_strategy_2",
                "retired_placeholder",
                '{"auto_repaired":true,"repair_reason":"orphan_strategy_score_parent"}',
                "2026-04-10T00:00:00+00:00",
                "2026-04-10T00:00:00+00:00",
                0,
            ),
            (
                10,
                "old_missing_source",
                "momentum_long",
                '{}',
                "2026-04-01T00:00:00+00:00",
                "2026-04-01T00:00:00+00:00",
                0,
            ),
            (
                11,
                "new_missing_source",
                "momentum_long",
                '{}',
                "2026-04-02T00:00:00+00:00",
                "2026-04-02T00:00:00+00:00",
                0,
            ),
        ],
    )
    conn.executemany(
        "INSERT INTO strategy_scores (strategy_id, timestamp, score) VALUES (?, ?, ?)",
        [
            (1, "2026-04-20T00:00:00+00:00", 0.8),
            (2, "2026-04-10T00:00:00+00:00", 0.1),
            (10, "2026-04-01T00:00:00+00:00", 0.2),
            (11, "2026-04-02T00:00:00+00:00", 0.3),
        ],
    )
    conn.commit()

    monkeypatch.setattr(db, "get_connection", lambda for_read=False: _connection_ctx(conn))
    monkeypatch.setattr(db_audit.config, "DB_REPAIR_KEEP_MISSING_SOURCE_STRATEGIES", 1)

    actions = []
    db_audit._repair_invalid_strategy_bloat(actions)
    db_audit._repair_missing_source_wallet_strategy_bloat(actions)

    remaining_ids = {
        row["id"] for row in conn.execute("SELECT id FROM strategies ORDER BY id").fetchall()
    }
    assert remaining_ids == {1, 11}
    remaining_score_ids = {
        row["strategy_id"]
        for row in conn.execute("SELECT strategy_id FROM strategy_scores ORDER BY strategy_id")
    }
    assert remaining_score_ids == {1, 11}
    assert any(action.action == "invalid_strategy_bloat" for action in actions)
    assert any(action.action == "missing_source_strategy_bloat" for action in actions)


def test_startup_strategy_repair_is_bounded(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE strategies (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            strategy_type TEXT NOT NULL,
            parameters TEXT DEFAULT '{}',
            discovered_at TEXT NOT NULL,
            active INTEGER DEFAULT 1
        );
        CREATE TABLE strategy_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            score REAL DEFAULT 0
        );
        """
    )
    for strategy_id in range(1, 4):
        conn.execute(
            """
            INSERT INTO strategies
            (id, name, strategy_type, parameters, discovered_at, active)
            VALUES (?, ?, 'retired_placeholder', '{"auto_repaired":true}',
                    '2026-04-10T00:00:00+00:00', 0)
            """,
            (strategy_id, f"recovered_{strategy_id}"),
        )
        conn.execute(
            "INSERT INTO strategy_scores (strategy_id, timestamp, score) VALUES (?, ?, ?)",
            (strategy_id, "2026-04-10T00:00:00+00:00", 0.1),
        )
    conn.commit()

    monkeypatch.setattr(db, "get_connection", lambda for_read=False: _connection_ctx(conn))

    actions = []
    db_audit._repair_invalid_strategy_bloat(actions, max_rows=1)

    assert conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM strategy_scores").fetchone()[0] == 2
    action = next(action for action in actions if action.action == "invalid_strategy_bloat")
    assert action.details["deleted_strategies"] == 1
    assert action.details["limit"] == 1


def test_db_audit_same_side_open_trades_only_flag_above_cap(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE traders (address TEXT PRIMARY KEY);
        CREATE TABLE strategies (id INTEGER PRIMARY KEY);
        CREATE TABLE bot_state (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE paper_account (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            balance REAL NOT NULL,
            total_pnl REAL DEFAULT 0,
            total_trades INTEGER DEFAULT 0,
            winning_trades INTEGER DEFAULT 0,
            last_updated TEXT NOT NULL
        );
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
            metadata TEXT DEFAULT '{}',
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        );
        CREATE TABLE audit_trail (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT DEFAULT '{}'
        );
        CREATE TABLE decision_snapshots (
            decision_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            coin TEXT,
            firewall_decision TEXT,
            final_status TEXT NOT NULL DEFAULT 'candidate'
        );
        CREATE TABLE source_inventory (
            source_name TEXT PRIMARY KEY,
            required INTEGER NOT NULL DEFAULT 0,
            expected_freshness_seconds INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE data_source_health_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            observed_at TEXT NOT NULL,
            source_name TEXT NOT NULL,
            status TEXT NOT NULL,
            freshness_seconds REAL,
            reason TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO paper_account "
        "(id, balance, total_pnl, total_trades, winning_trades, last_updated) "
        "VALUES (1, 10000, 0, 0, 0, '2026-04-23T00:00:00+00:00')"
    )
    for opened_at in ("2026-04-23T00:00:00+00:00", "2026-04-23T00:05:00+00:00"):
        conn.execute(
            """
            INSERT INTO paper_trades
            (opened_at, coin, side, entry_price, size, leverage, status, stop_loss, take_profit)
            VALUES (?, 'BTC', 'short', 70000, 0.01, 2, 'open', 72000, 65000)
            """,
            (opened_at,),
        )
    conn.commit()

    monkeypatch.setattr(db, "get_backend_name", lambda: "sqlite")
    monkeypatch.setattr(db, "get_db_path", lambda: "test.db")
    monkeypatch.setattr(db, "get_connection", lambda for_read=False: _connection_ctx(conn))
    monkeypatch.setattr(
        db_audit.config,
        "FIREWALL_MAX_SAME_SIDE_POSITIONS_PER_COIN",
        2,
        raising=False,
    )

    report = db_audit.run_db_audit(include_candle_cache=False, include_code_scan=False)
    checks = {finding.check for finding in report.findings}
    assert "duplicate_same_side_open_trades" not in checks

    conn.execute(
        """
        INSERT INTO paper_trades
        (opened_at, coin, side, entry_price, size, leverage, status, stop_loss, take_profit)
        VALUES ('2026-04-23T00:10:00+00:00', 'BTC', 'short', 69900, 0.01, 2, 'open', 72000, 65000)
        """
    )
    conn.commit()

    report = db_audit.run_db_audit(include_candle_cache=False, include_code_scan=False)
    checks = {finding.check for finding in report.findings}
    assert "duplicate_same_side_open_trades" in checks


def test_non_active_regime_repair_prunes_stale_predicted_rows(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE regime_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            coin TEXT NOT NULL,
            label_source TEXT DEFAULT 'predicted'
        );
        """
    )
    conn.execute(
        """
        INSERT INTO regime_history (timestamp, coin, label_source)
        VALUES ('2026-04-10T00:00:00+00:00', 'SOL', 'predicted')
        """
    )
    conn.commit()

    monkeypatch.setattr(db, "get_backend_name", lambda: "sqlite")
    monkeypatch.setattr(db, "get_db_path", lambda: "test.db")
    monkeypatch.setattr(db, "get_connection", lambda for_read=False: _connection_ctx(conn))
    monkeypatch.setattr(
        db_audit.config,
        "DB_AUDIT_NON_ACTIVE_REGIME_RETENTION_DAYS",
        7.0,
        raising=False,
    )

    actions = []
    db_audit._repair_non_active_regime_history(actions, [{"coin": "SOL"}])

    assert conn.execute("SELECT COUNT(*) FROM regime_history").fetchone()[0] == 0
    assert actions[0].status == "applied"
    assert actions[0].details["pruned_by_coin"] == {"SOL": 1}
