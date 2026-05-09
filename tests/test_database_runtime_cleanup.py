from contextlib import contextmanager
import json
import sqlite3

import src.data.database as db


def test_quarantine_invalid_traders_marks_bad_rows_inactive(tmp_path, monkeypatch):
    db_path = tmp_path / "runtime.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
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
        )
        """
    )
    conn.execute(
        "INSERT INTO traders (address, first_seen, last_updated, active, metadata) VALUES (?, ?, ?, ?, ?)",
        ("0xalpha_momentum_001", "2026-01-01", "2026-01-01", 1, "{}"),
    )
    conn.execute(
        "INSERT INTO traders (address, first_seen, last_updated, active, metadata) VALUES (?, ?, ?, ?, ?)",
        ("0x" + "1" * 40, "2026-01-01", "2026-01-01", 1, "{}"),
    )
    conn.commit()
    conn.close()

    @contextmanager
    def _fake_connection(*, for_read=False):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    monkeypatch.setattr(db, "get_connection", _fake_connection)

    invalid = db.quarantine_invalid_traders()
    active = db.get_active_traders(valid_only=True)

    assert invalid == ["0xalpha_momentum_001"]
    assert [row["address"] for row in active] == ["0x" + "1" * 40]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT active, metadata FROM traders WHERE address = ?",
        ("0xalpha_momentum_001",),
    ).fetchone()
    conn.close()

    assert row["active"] == 0
    assert "invalid_address_quarantined" in row["metadata"]


def test_quarantine_contaminated_runtime_data_disconnects_fixture_sources(tmp_path, monkeypatch):
    db_path = tmp_path / "runtime.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE traders (
            address TEXT PRIMARY KEY,
            first_seen TEXT NOT NULL,
            last_updated TEXT NOT NULL,
            active INTEGER DEFAULT 1,
            metadata TEXT DEFAULT '{}'
        );
        CREATE TABLE golden_wallets (
            address TEXT PRIMARY KEY,
            bot_score INTEGER DEFAULT 0,
            is_golden INTEGER DEFAULT 0,
            connected_to_live INTEGER DEFAULT 0
        );
        CREATE TABLE strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            strategy_type TEXT NOT NULL,
            parameters TEXT DEFAULT '{}',
            discovered_at TEXT NOT NULL,
            last_scored TEXT,
            current_score REAL DEFAULT 0,
            active INTEGER DEFAULT 1
        );
        """
    )
    valid_wallet = "0x" + "1" * 40
    conn.execute(
        "INSERT INTO traders (address, first_seen, last_updated, active, metadata) VALUES (?, ?, ?, ?, ?)",
        ("0xalpha_momentum_001", "2026-01-01", "2026-01-01", 1, "{}"),
    )
    conn.execute(
        "INSERT INTO golden_wallets (address, bot_score, is_golden, connected_to_live) VALUES (?, ?, ?, ?)",
        ("0xalpha_momentum_001", 0, 1, 1),
    )
    conn.execute(
        "INSERT INTO golden_wallets (address, bot_score, is_golden, connected_to_live) VALUES (?, ?, ?, ?)",
        (valid_wallet, 0, 1, 1),
    )
    conn.execute(
        """
        INSERT INTO strategies
        (name, description, strategy_type, parameters, discovered_at, current_score, active)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("alpha_momentum_btc", "fixture", "momentum_long", "{}", "2026-01-01", 0.9, 1),
    )
    conn.execute(
        """
        INSERT INTO strategies
        (name, description, strategy_type, parameters, discovered_at, current_score, active)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "momentum_long_0x111111",
            "real",
            "momentum_long",
            json.dumps({"source_wallet": valid_wallet}),
            "2026-01-01",
            0.8,
            1,
        ),
    )
    conn.commit()
    conn.close()

    @contextmanager
    def _fake_connection(*, for_read=False):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    monkeypatch.setattr(db, "get_connection", _fake_connection)

    summary = db.quarantine_contaminated_runtime_data()

    assert summary["invalid_traders"] == ["0xalpha_momentum_001"]
    assert summary["invalid_golden_wallets"] == ["0xalpha_momentum_001"]
    assert [item["name"] for item in summary["invalid_strategies"]] == ["alpha_momentum_btc"]

    active = db.get_active_strategies()
    assert [row["name"] for row in active] == ["momentum_long_0x111111"]

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    bad_wallet = conn.execute(
        "SELECT is_golden, connected_to_live, bot_score FROM golden_wallets WHERE address = ?",
        ("0xalpha_momentum_001",),
    ).fetchone()
    bad_strategy = conn.execute(
        "SELECT active, current_score FROM strategies WHERE name = ?",
        ("alpha_momentum_btc",),
    ).fetchone()
    conn.close()

    assert dict(bad_wallet) == {"is_golden": 0, "connected_to_live": 0, "bot_score": 10}
    assert dict(bad_strategy) == {"active": 0, "current_score": 0.0}


def test_get_active_strategies_filters_missing_or_bot_like_source(tmp_path, monkeypatch):
    db_path = tmp_path / "runtime.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            strategy_type TEXT NOT NULL,
            parameters TEXT DEFAULT '{}',
            discovered_at TEXT NOT NULL,
            current_score REAL DEFAULT 0,
            active INTEGER DEFAULT 1
        )
        """
    )
    valid_wallet = "0x" + "2" * 40
    rows = [
        ("unsourced", "momentum_long", "{}"),
        ("bot_sourced", "momentum_long", json.dumps({"source_wallet": valid_wallet, "source_wallet_bot_score": 9})),
        ("valid_sourced", "momentum_long", json.dumps({"source_wallet": valid_wallet, "source_wallet_bot_score": 0})),
    ]
    for name, stype, params in rows:
        conn.execute(
            """
            INSERT INTO strategies
            (name, description, strategy_type, parameters, discovered_at, current_score, active)
            VALUES (?, '', ?, ?, '2026-01-01', 0.5, 1)
            """,
            (name, stype, params),
        )
    conn.commit()
    conn.close()

    @contextmanager
    def _fake_connection(*, for_read=False):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    monkeypatch.setattr(db, "get_connection", _fake_connection)

    assert [row["name"] for row in db.get_active_strategies()] == ["valid_sourced"]
    assert len(db.get_active_strategies(validated_only=False)) == 3


def test_get_active_strategies_filters_synthetic_placeholder_metrics(tmp_path, monkeypatch):
    db_path = tmp_path / "runtime.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
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
            active INTEGER DEFAULT 1
        )
        """
    )
    valid_wallet = "0x" + "4" * 40
    rows = [
        ("synthetic_edge", 0.7955, 2000, 0.998),
        ("real_edge", 0.62, 42, 0.57),
    ]
    for name, score, trade_count, win_rate in rows:
        conn.execute(
            """
            INSERT INTO strategies
            (name, description, strategy_type, parameters, discovered_at,
             last_scored, current_score, total_pnl, trade_count, win_rate, active)
            VALUES (?, '', 'momentum_long', ?, '2026-01-01', '2026-01-02',
                    ?, 250, ?, ?, 1)
            """,
            (
                name,
                json.dumps({"source_wallet": valid_wallet, "source_wallet_bot_score": 0}),
                score,
                trade_count,
                win_rate,
            ),
        )
    conn.commit()
    conn.close()

    @contextmanager
    def _fake_connection(*, for_read=False):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    monkeypatch.setattr(db, "get_connection", _fake_connection)

    assert [row["name"] for row in db.get_active_strategies()] == ["real_edge"]

    summary = db.quarantine_contaminated_runtime_data()

    assert summary["invalid_strategies"] == [
        {"id": 1, "name": "synthetic_edge", "reason": "synthetic_placeholder_metrics"}
    ]
    assert [row["name"] for row in db.get_active_strategies(validated_only=False)] == ["real_edge"]


def test_recover_valid_inactive_strategies_keeps_quarantine_guard(tmp_path, monkeypatch):
    db_path = tmp_path / "runtime.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
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
            active INTEGER DEFAULT 1
        )
        """
    )
    valid_wallet = "0x" + "3" * 40
    rows = [
        ("valid_inactive", json.dumps({"source_wallet": valid_wallet}), 0.7, 0),
        ("fixture_inactive", "{}", 0.9, 0),
        ("active_valid", json.dumps({"source_wallet": valid_wallet}), 0.5, 1),
    ]
    for name, params, score, active in rows:
        conn.execute(
            """
            INSERT INTO strategies
            (name, description, strategy_type, parameters, discovered_at, current_score, active)
            VALUES (?, '', 'momentum_long', ?, '2026-01-01', ?, ?)
            """,
            (name, params, score, active),
        )
    conn.commit()
    conn.close()

    @contextmanager
    def _fake_connection(*, for_read=False):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    monkeypatch.setattr(db, "get_connection", _fake_connection)

    status_before = db.get_strategy_runtime_status()
    recovered = db.recover_valid_inactive_strategies(limit=5)
    active_names = [row["name"] for row in db.get_active_strategies()]

    assert status_before["inactive_valid"] == 1
    assert [row["name"] for row in recovered] == ["valid_inactive"]
    assert active_names == ["valid_inactive", "active_valid"]
