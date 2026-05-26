"""feature_cycle._get_watched_coins coverage + priority guarantees.

Background
----------
Production 2026-05-26: 27% of decision-firewall rejections were
``data_readiness_missing:candles`` because signals fired on coins
(BCH, VVV observed) that were NOT in the feature-store watched
universe.  Two structural bugs:

  1. The 30-coin cap was too small for the bot's tracked universe
     (~1300 traders, dozens of distinct coins emit signals on a
     given day).
  2. The post-collection truncation was ``sorted(coins)[:30]`` --
     alphabetical -- so an ACTIVE position on ZRX could be dropped
     to make room for a BOOTSTRAP coin starting with 'A'.

This PR:
  * Raises ``FEATURE_STORE_MAX_COINS`` 30 -> 80
  * Raises ``FEATURE_COPY_CANDIDATE_COINS_MAX`` 25 -> 60
  * Adds a new ``FEATURE_POSITION_SNAPSHOT_COINS_MAX`` (50) source
    that reads coins from the bot's tracked-trader
    ``position_snapshots`` table -- so when a tracked wallet opens
    a position on BCH, BCH joins the watched set BEFORE the bot's
    next signal evaluation
  * Replaces alphabetic truncation with priority-ordered
    truncation: ACTIVE > CANDIDATE > BOOTSTRAP
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager

import config
from src.core.cycles import feature_cycle


def _seeded_test_db(tmp_path, monkeypatch, *, traders_holding=()):
    """Build a minimal DB with the schema feature_cycle reads from."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY,
            coin TEXT,
            status TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE strategies (
            id INTEGER PRIMARY KEY,
            name TEXT,
            active INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE decision_snapshots (
            id INTEGER PRIMARY KEY,
            coin TEXT,
            source TEXT,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE position_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trader_address TEXT,
            timestamp TEXT,
            coin TEXT,
            side TEXT,
            size REAL,
            entry_price REAL,
            leverage REAL,
            unrealized_pnl REAL,
            margin_used REAL,
            metadata TEXT
        )
    """)
    # Seed the position_snapshots with the coins the tracked traders hold
    # right now (within the last 12 hours).  This is the new data path.
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()
    for coin in traders_holding:
        conn.execute(
            "INSERT INTO position_snapshots "
            "(trader_address, timestamp, coin, side, size, entry_price, "
            " leverage, unrealized_pnl, margin_used, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("0xfeed", now_iso, coin, "long", 1.0, 100.0, 1.0, 0.0, 100.0, "{}"),
        )
    conn.commit()
    conn.close()

    @contextmanager
    def fake_get_connection(*a, **k):
        c = sqlite3.connect(str(db_path))
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def fake_table_exists(name):
        return name in ("paper_trades", "strategies", "decision_snapshots",
                        "position_snapshots")

    from src.data import database as real_db
    monkeypatch.setattr(real_db, "get_connection", fake_get_connection)
    monkeypatch.setattr(real_db, "table_exists", fake_table_exists)

    return db_path


# ── New default cap is 80 ────────────────────────────────────


def test_max_coins_default_is_80(monkeypatch):
    """The new default cap is 80 (was 30)."""
    monkeypatch.delenv("FEATURE_STORE_MAX_COINS", raising=False)
    # Reload config in a fresh process would be ideal; instead just check
    # the value parses to 80 when env is unset.
    import importlib
    import config as _cfg
    importlib.reload(_cfg)
    assert _cfg.FEATURE_STORE_MAX_COINS == 80


# ── position_snapshots feeds the watched set ────────────────


def test_tracked_trader_positions_join_watched_set(tmp_path, monkeypatch):
    """A coin in position_snapshots (last 12h) lands in the watched set."""
    _seeded_test_db(tmp_path, monkeypatch, traders_holding=["BCH", "VVV"])
    monkeypatch.setattr(config, "FEATURE_STORE_COINS", "", raising=False)
    monkeypatch.setattr(config, "FEATURE_STORE_MAX_COINS", 80, raising=False)
    monkeypatch.setattr(config, "FEATURE_POSITION_SNAPSHOT_COINS_MAX", 50, raising=False)
    monkeypatch.setattr(config, "FEATURE_COPY_CANDIDATE_COINS_MAX", 60, raising=False)
    # Stub get_all_coins so the bootstrap fallback doesn't actually hit HL.
    monkeypatch.setattr(
        "src.data.hyperliquid_client.get_all_coins",
        lambda: [],
    )

    coins = feature_cycle._get_watched_coins()
    # The bug we're fixing: BCH and VVV must be in the set even though
    # they're not in paper_trades, strategies, or decision_snapshots.
    assert "BCH" in coins, f"BCH missing from watched set: {coins}"
    assert "VVV" in coins, f"VVV missing from watched set: {coins}"
    assert "BTC" in coins and "ETH" in coins   # always-on base


def test_position_snapshot_cap_zero_disables(tmp_path, monkeypatch):
    """FEATURE_POSITION_SNAPSHOT_COINS_MAX=0 disables the new path."""
    _seeded_test_db(tmp_path, monkeypatch, traders_holding=["BCH"])
    monkeypatch.setattr(config, "FEATURE_STORE_COINS", "", raising=False)
    monkeypatch.setattr(config, "FEATURE_STORE_MAX_COINS", 80, raising=False)
    monkeypatch.setattr(config, "FEATURE_POSITION_SNAPSHOT_COINS_MAX", 0, raising=False)
    monkeypatch.setattr(config, "FEATURE_COPY_CANDIDATE_COINS_MAX", 60, raising=False)
    monkeypatch.setattr(
        "src.data.hyperliquid_client.get_all_coins",
        lambda: [],
    )

    coins = feature_cycle._get_watched_coins()
    assert "BCH" not in coins, "BCH should be absent when path disabled"
    assert "BTC" in coins and "ETH" in coins


# ── Priority-ordered truncation ─────────────────────────────


def test_priority_truncation_keeps_active_over_alphabetic(tmp_path, monkeypatch):
    """At-cap collection: ACTIVE coins survive even if alphabetically last.

    Before this PR: ``sorted(coins)[:cap]`` truncated alphabetically
    -- ZRX (active position) could be dropped to make room for AAVE
    (bootstrap-only coin).  After: ACTIVE coins are kept first.
    """
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE paper_trades (id INTEGER PRIMARY KEY, coin TEXT, status TEXT)"
    )
    # Single ACTIVE position on ZRX (alphabetically last).
    conn.execute(
        "INSERT INTO paper_trades (coin, status) VALUES (?, ?)",
        ("ZRX", "open"),
    )
    conn.execute("CREATE TABLE strategies (id INTEGER PRIMARY KEY, name TEXT, active INTEGER)")
    conn.execute("CREATE TABLE decision_snapshots (id INTEGER PRIMARY KEY, coin TEXT, source TEXT, created_at TEXT)")
    conn.execute("CREATE TABLE position_snapshots (id INTEGER PRIMARY KEY, trader_address TEXT, timestamp TEXT, coin TEXT, side TEXT, size REAL, entry_price REAL, leverage REAL, unrealized_pnl REAL, margin_used REAL, metadata TEXT)")
    conn.commit()
    conn.close()

    @contextmanager
    def fake_get_connection(*a, **k):
        c = sqlite3.connect(str(db_path))
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    from src.data import database as real_db
    monkeypatch.setattr(real_db, "get_connection", fake_get_connection)
    monkeypatch.setattr(real_db, "table_exists", lambda n: True)

    # Force a TINY cap so truncation is exercised.  Bootstrap supplies
    # ~30 coins starting with 'A' which would alphabetically displace
    # ZRX under the old code.
    monkeypatch.setattr(config, "FEATURE_STORE_COINS", "", raising=False)
    monkeypatch.setattr(config, "FEATURE_STORE_MAX_COINS", 5, raising=False)
    monkeypatch.setattr(config, "FEATURE_STORE_BOOTSTRAP_TOP_COINS", 80, raising=False)
    monkeypatch.setattr(config, "FEATURE_POSITION_SNAPSHOT_COINS_MAX", 0, raising=False)
    monkeypatch.setattr(config, "FEATURE_COPY_CANDIDATE_COINS_MAX", 0, raising=False)
    monkeypatch.setattr(
        "src.data.hyperliquid_client.get_all_coins",
        lambda: ["AAVE", "AXS", "ADA", "ATOM", "ALGO", "APE", "AVAX", "ARB"],
    )

    # Reload feature_cycle module-level _MAX_COINS pickup.
    import importlib
    importlib.reload(feature_cycle)
    monkeypatch.setattr(feature_cycle, "_MAX_COINS", 5)
    monkeypatch.setattr(feature_cycle, "_BOOTSTRAP_TOP_COINS", 80)

    coins = feature_cycle._get_watched_coins()
    # ZRX is ACTIVE; cap is 5; bootstrap supplies 8 'A' coins that
    # would alphabetically beat ZRX.  ZRX MUST survive.
    assert "ZRX" in coins, (
        f"ZRX (active) was dropped by truncation; got: {coins}.  "
        f"Bug: alphabetic truncation dropped active positions in favour "
        f"of bootstrap coins"
    )
    assert "BTC" in coins and "ETH" in coins
    assert len(coins) <= 5


# ── Regression: existing paths still work ──────────────────


def test_open_paper_positions_are_watched(tmp_path, monkeypatch):
    """Open paper trades' coins land in watched (regression check)."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE paper_trades (id INTEGER PRIMARY KEY, coin TEXT, status TEXT)"
    )
    conn.execute(
        "INSERT INTO paper_trades (coin, status) VALUES (?, ?)",
        ("DOGE", "open"),
    )
    conn.execute("CREATE TABLE strategies (id INTEGER PRIMARY KEY, name TEXT, active INTEGER)")
    conn.execute("CREATE TABLE decision_snapshots (id INTEGER PRIMARY KEY, coin TEXT, source TEXT, created_at TEXT)")
    conn.execute("CREATE TABLE position_snapshots (id INTEGER PRIMARY KEY, trader_address TEXT, timestamp TEXT, coin TEXT, side TEXT, size REAL, entry_price REAL, leverage REAL, unrealized_pnl REAL, margin_used REAL, metadata TEXT)")
    conn.commit()
    conn.close()

    @contextmanager
    def fake_get_connection(*a, **k):
        c = sqlite3.connect(str(db_path))
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    from src.data import database as real_db
    monkeypatch.setattr(real_db, "get_connection", fake_get_connection)
    monkeypatch.setattr(real_db, "table_exists", lambda n: True)
    monkeypatch.setattr(
        "src.data.hyperliquid_client.get_all_coins",
        lambda: [],
    )
    monkeypatch.setattr(config, "FEATURE_STORE_COINS", "", raising=False)
    monkeypatch.setattr(config, "FEATURE_STORE_MAX_COINS", 80, raising=False)

    coins = feature_cycle._get_watched_coins()
    assert "DOGE" in coins


def test_explicit_env_override_takes_priority(tmp_path, monkeypatch):
    """Operator-pinned FEATURE_STORE_COINS land as ACTIVE coins."""
    _seeded_test_db(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "FEATURE_STORE_COINS", "WIF,JUP,BONK", raising=False)
    monkeypatch.setattr(config, "FEATURE_STORE_MAX_COINS", 80, raising=False)
    monkeypatch.setattr(
        "src.data.hyperliquid_client.get_all_coins",
        lambda: [],
    )

    coins = feature_cycle._get_watched_coins()
    assert "WIF" in coins
    assert "JUP" in coins
    assert "BONK" in coins
