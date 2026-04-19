"""
Shared pytest configuration.

Adds the project root to sys.path once so individual test files
don't need their own sys.path.insert hacks.
"""
import sys
import os

# Add project root to path so `import config` and `from src.xxx` work
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import pytest


@pytest.fixture(autouse=True)
def _reset_bot_state_kv():
    """Ensure bot_state KV rows don't leak between tests.

    The LiveTrader now persists canary/dedup state to the bot_state table
    (H5, E7).  Without isolation, a prior test's rows would be loaded by
    the next test's LiveTrader() and throw off counters.  We clear the
    table before AND after each test so ordering doesn't matter.
    """
    try:
        from src.data import database as _db
        with _db.get_connection() as _conn:
            _conn.execute("DELETE FROM bot_state")
    except Exception:
        pass
    yield
    try:
        from src.data import database as _db
        with _db.get_connection() as _conn:
            _conn.execute("DELETE FROM bot_state")
    except Exception:
        pass
