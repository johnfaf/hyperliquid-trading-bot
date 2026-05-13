import contextlib
import sqlite3

from src.data import database as db
from src.learning.audit_source_analysis import analyze_audit_sources
from src.learning.schema import ensure_sqlite_schema


@contextlib.contextmanager
def _sqlite_ctx(conn):
    yield conn
    conn.commit()


def test_analyze_audit_sources_reports_coverage_and_warmup(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_sqlite_schema(conn)
    conn.execute(
        """
        CREATE TABLE audit_trail (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            action TEXT,
            source TEXT,
            details TEXT,
            pnl REAL
        )
        """
    )
    for idx in range(30):
        conn.execute(
            """
            INSERT INTO decision_snapshots
            (decision_id, created_at, updated_at, source, source_key, final_status, rejection_reason, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"d{idx}",
                "2026-05-10T00:00:00+00:00",
                "2026-05-10T00:00:00+00:00",
                "copy_trade",
                "copy_trade:0x" + "1" * 40,
                "rejected_confidence",
                "allocator warmup fixed cap",
                "{}",
            ),
        )
    conn.execute(
        """
        INSERT INTO decision_outcomes
        (decision_id, created_at, updated_at, source, source_key, action_taken, label_win, outcome_pnl)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "o1",
            "2026-05-10T00:00:00+00:00",
            "2026-05-10T00:00:00+00:00",
            "strategy",
            "strategy:momentum",
            1,
            1,
            3.5,
        ),
    )
    conn.commit()
    monkeypatch.setattr(db, "get_connection", lambda for_read=False: _sqlite_ctx(conn))

    report = analyze_audit_sources(
        days=10,
        limit=100,
        warmup_days=1,
        warmup_min_rejections=20,
    )

    assert report["summary"]["rejections"] == 30
    assert report["top_sources_to_threshold"][0]["source_key"].startswith("copy_trade:0x")
    assert report["warmup_stuck_alerts"]
    assert any(row["source_key"] == "strategy:momentum" and row["net_positive"] for row in report["sources"])
