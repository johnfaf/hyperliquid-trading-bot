"""
Shared pytest configuration.

Adds the project root to sys.path once so individual test files
don't need their own sys.path.insert hacks.
"""
import sys
import os
import sqlite3

# Add project root to path so `import config` and `from src.xxx` work
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import pytest  # noqa: E402  -- must follow the sys.path shim above


_RESET_TABLES = (
    "decision_stage_events",
    "decision_outcomes",
    "shadow_trades",
    "paper_trades",
    "bot_state",
)


def _clear_test_state():
    from src.data import database as _db

    with _db.get_connection() as _conn:
        for table in _RESET_TABLES:
            try:
                if _db.table_exists(table):
                    _conn.execute(f"DELETE FROM {table}")
            except sqlite3.OperationalError as exc:
                if "no such table" in str(exc).lower():
                    continue
                raise


@pytest.fixture(autouse=True)
def _reset_persistent_runtime_state():
    """Ensure persistent runtime rows do not leak between tests.

    LiveTrader and learning/replay paths persist kill-switch, dedup, paper
    trade, shadow, and decision state. Clear the volatile tables before and
    after each test so ordering does not hide state bugs.
    """
    try:
        _clear_test_state()
    except Exception as exc:
        pytest.fail(f"Could not reset persistent test DB state before test: {exc}")
    yield
    try:
        _clear_test_state()
    except Exception as exc:
        pytest.fail(f"Could not reset persistent test DB state after test: {exc}")
