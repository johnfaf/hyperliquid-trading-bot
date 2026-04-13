#!/usr/bin/env python3
"""
Continuous parity checker for dual-write mode.

Compares SQLite and Postgres row counts, paper_account state, latest
audit rows, and open trade counts.  Run this alongside the bot during
the dual-write observation window (24-48h).

Usage::

    # Run once
    python scripts/check_dualwrite_parity.py

    # Run continuously every 60 seconds
    python scripts/check_dualwrite_parity.py --loop 60

    # Show dual-write health counters
    python scripts/check_dualwrite_parity.py --stats
"""
import argparse
import logging
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("parity")

TABLES = [
    "traders",
    "position_snapshots",
    "strategies",
    "strategy_scores",
    "paper_account",
    "paper_trades",
    "research_logs",
    "bot_state",
    "audit_trail",
    "golden_wallets",
    "wallet_fills",
    "calibration_records",
    "agent_scores",
    "arena_agents",
    "arena_rounds",
    "trade_memory",
    "shadow_trades",
    "shadow_attribution",
]


def _sq_conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _pg_conn():
    from src.data.db.postgres import get_connection, return_connection
    return get_connection(), return_connection


def _table_exists_sq(conn, name):
    return conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _table_exists_pg(cur, name):
    cur.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='public' AND table_name=%s", (name,)
    )
    return cur.fetchone() is not None


def run_parity_check() -> dict:
    """Run a single parity check. Returns a dict of results."""
    sq = _sq_conn()
    pg_raw, pg_return = _pg_conn()
    cur = pg_raw.cursor()
    results = {"tables": {}, "ok": True, "timestamp": time.time()}

    try:
        # 1. Row counts per table
        for table in TABLES:
            sq_exists = _table_exists_sq(sq, table)
            pg_exists = _table_exists_pg(cur, table)

            if not sq_exists and not pg_exists:
                continue

            sq_count = 0
            pg_count = 0
            if sq_exists:
                sq_count = sq.execute(f"SELECT COUNT(*) as c FROM {table}").fetchone()["c"]
            if pg_exists:
                cur.execute(f"SELECT COUNT(*) as c FROM {table}")
                pg_count = cur.fetchone()["c"]

            delta = pg_count - sq_count
            ok = delta == 0
            results["tables"][table] = {
                "sqlite": sq_count, "postgres": pg_count,
                "delta": delta, "ok": ok,
            }
            if not ok:
                results["ok"] = False

            status = "OK" if ok else f"DRIFT {delta:+d}"
            logger.info("  %-25s sq=%-6d pg=%-6d %s", table, sq_count, pg_count, status)

        # 2. Paper account balance check
        sq_acc = sq.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
        if sq_acc:
            cur.execute("SELECT * FROM paper_account WHERE id = 1")
            pg_acc = cur.fetchone()
            if pg_acc:
                for key in ("balance", "total_pnl", "total_trades", "winning_trades"):
                    sq_val = float(sq_acc[key] or 0)
                    pg_val = float(pg_acc[key] or 0)
                    drift = abs(sq_val - pg_val)
                    ok = drift < 0.01
                    if not ok:
                        results["ok"] = False
                    status = "OK" if ok else f"DRIFT {drift:.4f}"
                    logger.info("  paper_account.%-15s sq=%-12s pg=%-12s %s",
                                key, sq_val, pg_val, status)
                    results[f"paper_{key}"] = {"sqlite": sq_val, "postgres": pg_val, "ok": ok}
            else:
                logger.warning("  paper_account: exists in SQLite but NOT in Postgres")
                results["ok"] = False

        # 3. Open trades count
        sq_open = sq.execute(
            "SELECT COUNT(*) as c FROM paper_trades WHERE status = 'open'"
        ).fetchone()["c"]
        cur.execute("SELECT COUNT(*) as c FROM paper_trades WHERE status = 'open'")
        pg_open = cur.fetchone()["c"]
        open_ok = sq_open == pg_open
        if not open_ok:
            results["ok"] = False
        logger.info("  open_trades                 sq=%-6d pg=%-6d %s",
                     sq_open, pg_open, "OK" if open_ok else "MISMATCH")
        results["open_trades"] = {"sqlite": sq_open, "postgres": pg_open, "ok": open_ok}

        # 4. Latest audit trail IDs
        sq_latest = sq.execute(
            "SELECT MAX(id) as m FROM audit_trail"
        ).fetchone()
        sq_max = sq_latest["m"] if sq_latest else 0

        cur.execute("SELECT MAX(id) as m FROM audit_trail")
        pg_latest = cur.fetchone()
        pg_max = pg_latest["m"] if pg_latest else 0

        audit_ok = sq_max == pg_max
        if not audit_ok:
            results["ok"] = False
        logger.info("  audit_trail.max_id          sq=%-6s pg=%-6s %s",
                     sq_max, pg_max, "OK" if audit_ok else f"DRIFT {(pg_max or 0) - (sq_max or 0):+d}")
        results["audit_max_id"] = {"sqlite": sq_max, "postgres": pg_max, "ok": audit_ok}

        pg_raw.commit()

    finally:
        sq.close()
        pg_return(pg_raw)

    return results


def show_dualwrite_stats():
    """Print the DualWriteAdapter health counters."""
    try:
        from src.data.db.connection import dualwrite_stats
        snap = dualwrite_stats.snapshot()
        total = snap["pg_writes_ok"] + snap["pg_writes_failed"]
        fail_pct = (snap["pg_writes_failed"] / total * 100) if total > 0 else 0.0
        logger.info("=== DUAL-WRITE HEALTH ===")
        logger.info("  Postgres writes OK:     %d", snap["pg_writes_ok"])
        logger.info("  Postgres writes FAILED: %d (%.1f%%)", snap["pg_writes_failed"], fail_pct)
        if snap["pg_last_error"]:
            age = time.time() - snap["pg_last_error_ts"]
            logger.info("  Last error (%ds ago):    %s", int(age), snap["pg_last_error"])
        return snap
    except Exception as exc:
        logger.error("Could not read dualwrite stats: %s", exc)
        return {}


def main():
    parser = argparse.ArgumentParser(description="Dual-write parity checker")
    parser.add_argument("--loop", type=int, default=0,
                        help="Re-run every N seconds (0 = run once)")
    parser.add_argument("--stats", action="store_true",
                        help="Show dual-write health counters and exit")
    args = parser.parse_args()

    if not config.POSTGRES_DSN:
        logger.error("POSTGRES_DSN is not set.")
        sys.exit(1)

    if args.stats:
        show_dualwrite_stats()
        return

    if args.loop > 0:
        logger.info("Parity checker running every %ds. Ctrl-C to stop.", args.loop)
        while True:
            logger.info("=== PARITY CHECK %s ===", time.strftime("%Y-%m-%d %H:%M:%S"))
            result = run_parity_check()
            show_dualwrite_stats()
            if result["ok"]:
                logger.info(">>> ALL CHECKS PASSED <<<")
            else:
                logger.warning(">>> PARITY ISSUES DETECTED <<<")
            logger.info("")
            time.sleep(args.loop)
    else:
        logger.info("=== PARITY CHECK ===")
        result = run_parity_check()
        if result["ok"]:
            logger.info("\n>>> ALL CHECKS PASSED <<<")
        else:
            logger.warning("\n>>> PARITY ISSUES DETECTED <<<")
            sys.exit(1)


if __name__ == "__main__":
    main()
