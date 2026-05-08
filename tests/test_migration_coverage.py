"""Regression test that prevents new on-demand `CREATE TABLE IF NOT EXISTS`
patterns in src/ from rotting the Postgres dualwrite mirror.

Background
----------
The dualwrite adapter at src/data/db/connection.py deliberately skips DDL
statements (CREATE / ALTER / DROP) from the Postgres mirror — schema is
expected to flow through migrations/. When a module creates tables on
demand at runtime via `CREATE TABLE IF NOT EXISTS ...` and forgets to add
a matching migration, the table only ever exists in SQLite. Every INSERT
into that table then fails on the Postgres side with
"UndefinedTable: relation 'X' does not exist".

That manifested as 207 dualwrite warnings in 7.5 minutes for the backtest
tables (fixed in 0015) and arena_trade_events / readiness_probe (fixed
in 0016). This test guards against the next instance.

Mechanism
---------
Walk every `CREATE TABLE IF NOT EXISTS <name>` literal in src/, then
verify each `<name>` has a matching `CREATE TABLE IF NOT EXISTS <name>`
in some file under migrations/*.sql.

If a new table needs to legitimately stay SQLite-only (rare; nothing
should), add it to ``KNOWN_SQLITE_ONLY_TABLES`` below with a comment
explaining why.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
MIGRATIONS = ROOT / "migrations"

# Tables that intentionally do NOT have a Postgres migration.
# When you add an entry here, leave a comment with the *reason* so the next
# person doesn't whitelist a real schema-rot bug by mistake.
KNOWN_SQLITE_ONLY_TABLES: set[str] = {
    # Backtest experiment results — written via direct sqlite3.connect to
    # config.DB_PATH (NOT the dualwrite router). Local research artefact;
    # never mirrored to Postgres because the writes don't go through
    # src/data/db/connection.py at all. See src/backtest/backtester.py.
    "experiments",
    # Candle cache fetch log — written to data/candle_cache.db, a separate
    # SQLite file with its own dedicated connection. Same reason as above:
    # bypasses the dualwrite adapter, so there's no Postgres write to
    # mirror. See src/backtest/data_fetcher.py.
    "fetch_log",
}

# Tables created by the schema_migrations machinery itself or by the
# dualwrite probes — these are bookkeeping concerns, not bot data.
INTERNAL_TABLES: set[str] = {
    "schema_migrations",       # written by src/data/db/migrations.py
    "_migration_guard",        # transient guard inside golden_wallet migration
    "_dualwrite_probe",        # dualwrite write-probe table
}

# `CREATE TABLE IF NOT EXISTS <name> (` — capture <name> only.
# Require an open paren after the name so we don't match natural-language
# fragments like "# Create table if not exists" followed by an unrelated
# identifier on the next line (e.g. ``conn``).
_CREATE_RE = re.compile(
    r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.IGNORECASE,
)


def _collect_create_table_names(root: Path, suffix: str) -> set[str]:
    names: set[str] = set()
    for path in root.rglob(f"*{suffix}"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for match in _CREATE_RE.finditer(text):
            names.add(match.group(1).lower())
    return names


def test_every_runtime_create_table_has_a_postgres_migration():
    src_tables = _collect_create_table_names(SRC, ".py")
    migration_tables = _collect_create_table_names(MIGRATIONS, ".sql")

    # Strip internal/whitelisted entries from both sides.
    src_tables -= INTERNAL_TABLES
    src_tables -= KNOWN_SQLITE_ONLY_TABLES

    missing = sorted(src_tables - migration_tables)
    assert not missing, (
        "The following tables are created on-demand in src/ but have no "
        "matching CREATE TABLE in migrations/. Each of these will spam the "
        "Postgres dualwrite mirror with UndefinedTable warnings on every "
        "INSERT. Add a migration file (or, if the table is genuinely "
        "SQLite-only, add it to KNOWN_SQLITE_ONLY_TABLES with a reason).\n"
        f"  Missing: {missing}"
    )
