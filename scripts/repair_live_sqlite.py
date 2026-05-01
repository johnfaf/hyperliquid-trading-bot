"""Minimal live SQLite repair for blocking audit findings.

This script is intentionally narrower than ``main.py --db-repair``:

- it talks directly to SQLite instead of the routed DB layer
- it avoids network-backed refreshes
- it retries ``BEGIN IMMEDIATE`` long enough to win an exclusive write window

Use it on a deployed volume when the app process exists but the general repair
path cannot get through due to lock contention.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path(cli_path: str | None) -> str:
    if cli_path:
        return cli_path
    env_path = str(os.environ.get("HL_BOT_DB", "") or "").strip()
    if env_path:
        return env_path
    return "/data/bot.db"


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _wait_for_immediate(conn: sqlite3.Connection, *, attempts: int = 90, sleep_s: float = 2.0) -> int:
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            conn.execute("BEGIN IMMEDIATE")
            return attempt
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
            last_error = exc
            time.sleep(sleep_s)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Could not acquire SQLite write lock")


def _fk_summary(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("PRAGMA foreign_key_check").fetchall()
    summary: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["table"]), str(row["parent"]))
        summary[key] = summary.get(key, 0) + 1
    return [
        {"table": table, "parent": parent, "count": count}
        for (table, parent), count in sorted(summary.items(), key=lambda item: item[1], reverse=True)
    ]


def _ensure_schema_migrations(conn: sqlite3.Connection) -> bool:
    if _table_exists(conn, "schema_migrations"):
        return False
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    return True


def _repair_orphan_traders(conn: sqlite3.Connection) -> int:
    if not (_table_exists(conn, "position_snapshots") and _table_exists(conn, "traders")):
        return 0
    rows = conn.execute(
        """
        SELECT trader_address, COUNT(*) AS snapshot_count,
               MIN(timestamp) AS first_seen,
               MAX(timestamp) AS last_seen
        FROM position_snapshots
        WHERE trader_address NOT IN (SELECT address FROM traders)
        GROUP BY trader_address
        ORDER BY trader_address
        """
    ).fetchall()
    repaired = 0
    for row in rows:
        address = str(row["trader_address"] or "").strip()
        if not address:
            continue
        metadata = json.dumps(
            {
                "auto_repaired": True,
                "repair_reason": "orphan_position_snapshot_parent",
                "placeholder": True,
                "snapshot_count": int(row["snapshot_count"] or 0),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        conn.execute(
            """
            INSERT INTO traders
            (address, first_seen, last_updated, total_pnl, roi_pct,
             account_value, win_rate, trade_count, active, metadata)
            VALUES (?, ?, ?, 0, 0, 0, 0, 0, 0, ?)
            ON CONFLICT(address) DO UPDATE SET
                active = 0,
                metadata = excluded.metadata
            """,
            (
                address,
                str(row["first_seen"] or _now()),
                str(row["last_seen"] or _now()),
                metadata,
            ),
        )
        repaired += 1
    return repaired


def _repair_orphan_strategies(conn: sqlite3.Connection) -> int:
    if not (_table_exists(conn, "strategy_scores") and _table_exists(conn, "strategies")):
        return 0
    rows = conn.execute(
        """
        SELECT strategy_id, COUNT(*) AS score_count,
               MIN(timestamp) AS first_seen,
               MAX(timestamp) AS last_seen
        FROM strategy_scores
        WHERE strategy_id NOT IN (SELECT id FROM strategies)
        GROUP BY strategy_id
        ORDER BY strategy_id
        """
    ).fetchall()
    repaired = 0
    for row in rows:
        strategy_id = int(row["strategy_id"] or 0)
        if strategy_id <= 0:
            continue
        score_count = int(row["score_count"] or 0)
        first_seen = str(row["first_seen"] or _now())
        last_seen = str(row["last_seen"] or first_seen)
        conn.execute(
            """
            INSERT INTO strategies
            (id, name, description, strategy_type, parameters, discovered_at,
             last_scored, current_score, total_pnl, trade_count, win_rate,
             sharpe_ratio, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 0, 0)
            ON CONFLICT(id) DO UPDATE SET
                last_scored = excluded.last_scored,
                active = 0
            """,
            (
                strategy_id,
                f"recovered_strategy_{strategy_id}",
                (
                    "Auto-repaired placeholder strategy created to preserve "
                    f"{score_count} historical strategy_scores rows."
                ),
                "retired_placeholder",
                json.dumps(
                    {
                        "auto_repaired": True,
                        "repair_reason": "orphan_strategy_score_parent",
                        "score_count": score_count,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                first_seen,
                last_seen,
            ),
        )
        repaired += 1
    return repaired


def _repair_paper_account(conn: sqlite3.Connection) -> dict[str, Any]:
    if not _table_exists(conn, "paper_trades"):
        return {"updated": False, "reason": "paper_trades_missing"}

    closed = conn.execute(
        """
        SELECT COUNT(*) AS closed_count,
               COALESCE(SUM(pnl), 0) AS closed_pnl,
               COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) AS closed_wins
        FROM paper_trades
        WHERE LOWER(COALESCE(status, '')) = 'closed'
        """
    ).fetchone()
    closed_count = int(closed["closed_count"] or 0)
    closed_pnl = float(closed["closed_pnl"] or 0.0)
    closed_wins = int(closed["closed_wins"] or 0)

    if not _table_exists(conn, "paper_account"):
        return {
            "updated": False,
            "reason": "paper_account_missing",
            "closed_count": closed_count,
            "closed_pnl": closed_pnl,
            "closed_wins": closed_wins,
        }

    account = conn.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
    balance = float(account["balance"] or 0.0) if account is not None else 0.0
    now = _now()
    if account is None:
        conn.execute(
            """
            INSERT INTO paper_account
            (id, balance, total_pnl, total_trades, winning_trades, last_updated)
            VALUES (1, ?, ?, ?, ?, ?)
            """,
            (balance, closed_pnl, closed_count, closed_wins, now),
        )
    else:
        conn.execute(
            """
            UPDATE paper_account
            SET total_pnl = ?, total_trades = ?, winning_trades = ?, last_updated = ?
            WHERE id = 1
            """,
            (closed_pnl, closed_count, closed_wins, now),
        )
    return {
        "updated": True,
        "balance": balance,
        "closed_count": closed_count,
        "closed_pnl": closed_pnl,
        "closed_wins": closed_wins,
    }


def run(path: str) -> dict[str, Any]:
    conn = _connect(path)
    try:
        before = _fk_summary(conn)
        lock_attempt = _wait_for_immediate(conn)
        created_schema_migrations = _ensure_schema_migrations(conn)
        repaired_traders = _repair_orphan_traders(conn)
        repaired_strategies = _repair_orphan_strategies(conn)
        paper_account = _repair_paper_account(conn)
        conn.commit()
        after = _fk_summary(conn)
        return {
            "db_path": path,
            "lock_attempt": lock_attempt,
            "created_schema_migrations": created_schema_migrations,
            "repaired_traders": repaired_traders,
            "repaired_strategies": repaired_strategies,
            "paper_account": paper_account,
            "foreign_key_before": {
                "count": sum(int(item["count"]) for item in before),
                "summary": before[:20],
            },
            "foreign_key_after": {
                "count": sum(int(item["count"]) for item in after),
                "summary": after[:20],
            },
        }
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair blocking live SQLite audit findings.")
    parser.add_argument("--db-path", default=None, help="SQLite DB path (defaults to HL_BOT_DB or /data/bot.db)")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    args = parser.parse_args(argv)

    result = run(_db_path(args.db_path))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"db_path: {result['db_path']}")
        print(f"lock_attempt: {result['lock_attempt']}")
        print(f"created_schema_migrations: {result['created_schema_migrations']}")
        print(f"repaired_traders: {result['repaired_traders']}")
        print(f"repaired_strategies: {result['repaired_strategies']}")
        print(f"paper_account: {json.dumps(result['paper_account'], sort_keys=True)}")
        print(
            "foreign_key_before:",
            result["foreign_key_before"]["count"],
            result["foreign_key_before"]["summary"][:5],
        )
        print(
            "foreign_key_after:",
            result["foreign_key_after"]["count"],
            result["foreign_key_after"]["summary"][:5],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
