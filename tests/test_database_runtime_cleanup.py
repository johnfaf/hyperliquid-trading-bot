from contextlib import contextmanager
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
