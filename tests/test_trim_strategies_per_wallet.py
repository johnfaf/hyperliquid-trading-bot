"""Tests for the retroactive strategy-trim CLI.

The trim CLI matches PR #24's runtime cap to the historical strategies
table: for each ``source_wallet``, keep the top-N strategies by
``current_score`` and deactivate the rest.  This unblocks removing the
temporary ``DB_SAFE_AUTO_REPAIR_ON_BOOT=false`` and
``BOOT_DB_AUDIT_SKIP=true`` env-var bypasses (PR #23).

Tests cover:
  * Pure planning logic (no DB) -- which ids would be kept vs dropped
  * Apply-path -- the UPDATE actually flips ``active=false`` on the
    targeted ids and leaves the keepers untouched
  * Edge cases -- wallet extraction from ``parameters.source_wallet``
    JSON vs the name-suffix fallback, missing scores, no-wallet rows
"""
from __future__ import annotations

import json

from scripts.trim_strategies_per_wallet import (
    _extract_source_wallet,
    apply_trim,
    plan_trim,
)


# ── _extract_source_wallet ──────────────────────────────────


def test_extracts_from_parameters_json_string():
    s = {
        "name": "momentum_long_0xabcdef",
        "parameters": json.dumps({"source_wallet": "0xABCDEF" + "0" * 34}),
    }
    assert _extract_source_wallet(s) == "0xabcdef" + "0" * 34


def test_extracts_from_parameters_dict():
    s = {
        "name": "momentum_long_0xabcdef",
        "parameters": {"source_wallet": "0xCAFE" + "0" * 36},
    }
    assert _extract_source_wallet(s) == "0xcafe" + "0" * 36


def test_falls_back_to_name_suffix_when_params_missing():
    s = {"name": "momentum_long_0xabcde0", "parameters": {}}
    assert _extract_source_wallet(s) == "0xabcde0"


def test_returns_none_when_no_wallet_resolvable():
    s = {"name": "unnamed_strategy", "parameters": {}}
    assert _extract_source_wallet(s) is None


def test_handles_invalid_parameters_json_string():
    s = {"name": "momentum_long_0xabcde0", "parameters": "this is not json"}
    # Falls back to the name-suffix path.
    assert _extract_source_wallet(s) == "0xabcde0"


# ── plan_trim ───────────────────────────────────────────────


def _strat(id_: int, wallet: str, type_: str, score: float):
    return {
        "id": id_,
        "name": f"{type_}_{wallet[:8]}",
        "strategy_type": type_,
        "parameters": {"source_wallet": wallet},
        "current_score": score,
        "active": True,
    }


def test_plan_keeps_top_n_per_wallet():
    """Wallet with 4 strategies, cap=2 -> keep 2 highest-score ids."""
    wallet = "0x" + "a" * 40
    strategies = [
        _strat(1, wallet, "momentum_long", 0.30),
        _strat(2, wallet, "concentrated_bet", 0.90),  # top
        _strat(3, wallet, "trend_following", 0.20),
        _strat(4, wallet, "swing_trading", 0.70),  # second
    ]
    keep, deactivate, report = plan_trim(strategies, cap=2)
    # Top 2 by score = ids 2 and 4
    assert set(keep) == {2, 4}
    assert set(deactivate) == {1, 3}
    r = report[wallet]
    assert r["before"] == 4 and r["after"] == 2 and r["dropped"] == 2
    assert r["kept_types"] == ["concentrated_bet", "swing_trading"]
    assert set(r["dropped_types"]) == {"momentum_long", "trend_following"}


def test_plan_cap_1_collapses_to_single():
    """Cap=1 keeps only the highest-confidence per wallet."""
    wallet = "0x" + "b" * 40
    strategies = [
        _strat(10, wallet, "momentum_long", 0.55),
        _strat(11, wallet, "scalping", 0.80),  # winner
        _strat(12, wallet, "breakout", 0.40),
    ]
    keep, deactivate, _ = plan_trim(strategies, cap=1)
    assert keep == [11]
    assert set(deactivate) == {10, 12}


def test_plan_wallet_under_cap_keeps_all():
    """Wallet with 2 strategies, cap=3 -> keep both, no drops."""
    wallet = "0x" + "c" * 40
    strategies = [
        _strat(20, wallet, "momentum_long", 0.40),
        _strat(21, wallet, "scalping", 0.70),
    ]
    keep, deactivate, report = plan_trim(strategies, cap=3)
    assert set(keep) == {20, 21}
    assert deactivate == []
    assert report[wallet]["dropped"] == 0


def test_plan_groups_by_wallet_independently():
    """Two wallets at cap=2: each gets its own top-2 selection."""
    w1 = "0x" + "d" * 40
    w2 = "0x" + "e" * 40
    strategies = [
        _strat(30, w1, "momentum_long", 0.30),
        _strat(31, w1, "scalping", 0.90),
        _strat(32, w1, "breakout", 0.20),
        _strat(33, w2, "momentum_long", 0.50),
        _strat(34, w2, "swing_trading", 0.80),
        _strat(35, w2, "concentrated_bet", 0.10),
    ]
    keep, deactivate, _ = plan_trim(strategies, cap=2)
    # w1 keeps: 31 (0.90), 30 (0.30)
    # w2 keeps: 34 (0.80), 33 (0.50)
    assert set(keep) == {31, 30, 34, 33}
    assert set(deactivate) == {32, 35}


def test_plan_no_wallet_strategies_are_left_alone():
    """Strategies without a resolvable source_wallet stay active."""
    strategies = [
        {
            "id": 100,
            "name": "weird_name_no_addr",
            "strategy_type": "momentum_long",
            "parameters": {},
            "current_score": 0.5,
        },
    ]
    keep, deactivate, report = plan_trim(strategies, cap=2)
    assert keep == [] and deactivate == []
    assert "__no_wallet__" in report
    assert report["__no_wallet__"]["dropped"] == 0


def test_plan_handles_none_scores_by_treating_as_zero():
    """Strategies with current_score=None sort last (treated as 0.0)."""
    wallet = "0x" + "f" * 40
    strategies = [
        {**_strat(50, wallet, "momentum_long", 0.10), "current_score": None},
        _strat(51, wallet, "scalping", 0.05),
    ]
    keep, deactivate, _ = plan_trim(strategies, cap=1)
    # 51 has score 0.05; the None-score one is treated as 0.0 -> 51 wins.
    assert keep == [51]
    assert deactivate == [50]


# ── apply_trim ──────────────────────────────────────────────


def _setup_test_db(tmp_path, monkeypatch):
    """Point ``db.get_connection`` at an isolated SQLite file."""
    import sqlite3
    from contextlib import contextmanager

    db_path = tmp_path / "test.db"
    # Reuse the production schema for ``strategies`` + ``strategy_scores``
    # so the SQL the script issues binds correctly.
    conn0 = sqlite3.connect(str(db_path))
    conn0.row_factory = sqlite3.Row
    conn0.execute("""
        CREATE TABLE strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, description TEXT, strategy_type TEXT,
            parameters TEXT DEFAULT '{}',
            discovered_at TEXT, last_scored TEXT,
            current_score REAL DEFAULT 0,
            total_pnl REAL DEFAULT 0, trade_count INTEGER DEFAULT 0,
            win_rate REAL DEFAULT 0, sharpe_ratio REAL DEFAULT 0,
            active INTEGER DEFAULT 1
        )
    """)
    conn0.execute("""
        CREATE TABLE strategy_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER NOT NULL,
            timestamp TEXT, score REAL
        )
    """)
    conn0.commit()
    conn0.close()

    @contextmanager
    def fake_get_connection(*a, **k):
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    from src.data import database as real_db
    monkeypatch.setattr(real_db, "get_connection", fake_get_connection)

    # Also patch the import the script uses (it imports db at module
    # load), so the in-script ``db.get_connection`` references our fake.
    from scripts import trim_strategies_per_wallet as script
    monkeypatch.setattr(script.db, "get_connection", fake_get_connection)

    return db_path


def test_apply_deactivates_only_targeted_ids(tmp_path, monkeypatch):
    """Apply path flips ``active=0`` on targeted ids and leaves the rest."""
    db_path = _setup_test_db(tmp_path, monkeypatch)
    import sqlite3

    # Seed: 4 strategies, deactivate ids 1 and 3.
    conn = sqlite3.connect(str(db_path))
    for sid in (1, 2, 3, 4):
        conn.execute(
            "INSERT INTO strategies (id, name, strategy_type, active) "
            "VALUES (?, ?, ?, 1)",
            (sid, f"name_{sid}", "momentum_long"),
        )
    conn.commit()
    conn.close()

    counts = apply_trim([1, 3])
    assert counts["deactivated"] == 2

    conn = sqlite3.connect(str(db_path))
    rows = list(
        conn.execute("SELECT id, active FROM strategies ORDER BY id")
    )
    conn.close()
    assert dict(rows) == {1: 0, 2: 1, 3: 0, 4: 1}


def test_apply_purges_scores_when_requested(tmp_path, monkeypatch):
    """``--purge-scores`` deletes strategy_scores rows for deactivated ids."""
    db_path = _setup_test_db(tmp_path, monkeypatch)
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    for sid in (1, 2):
        conn.execute(
            "INSERT INTO strategies (id, name, strategy_type, active) "
            "VALUES (?, ?, ?, 1)",
            (sid, f"name_{sid}", "momentum_long"),
        )
        for k in range(3):
            conn.execute(
                "INSERT INTO strategy_scores (strategy_id, timestamp, score) "
                "VALUES (?, ?, ?)",
                (sid, "2026-01-01T00:00:00+00:00", float(k)),
            )
    conn.commit()
    conn.close()

    counts = apply_trim([1], purge_scores=True)
    assert counts["deactivated"] == 1
    # Strategy 1 had 3 score rows; they should all be gone.
    assert counts["scores_purged"] == 3

    conn = sqlite3.connect(str(db_path))
    remaining_scores = conn.execute(
        "SELECT strategy_id, COUNT(*) FROM strategy_scores GROUP BY strategy_id"
    ).fetchall()
    conn.close()
    # Only strategy 2's 3 rows remain.
    assert remaining_scores == [(2, 3)]


def test_apply_with_no_ids_is_noop(tmp_path, monkeypatch):
    """Empty deactivate list returns zero counts and writes nothing."""
    _setup_test_db(tmp_path, monkeypatch)
    counts = apply_trim([])
    assert counts == {"deactivated": 0, "scores_purged": 0}
