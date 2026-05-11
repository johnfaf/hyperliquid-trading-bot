"""Tests for ReplayDB lifecycle and strategy_seed insertion."""
import json
import os
import sqlite3

import pytest

from src.backtest.replay.replay_db import ReplayDB, ReplayDBError
from src.backtest.replay.strategy_seed import (
    SeedSnapshot, SeedTrader, SeedStrategy,
    build_default_smoke_snapshot,
    seed_into, load_snapshot, save_snapshot, export_from_live_db,
)


# --- ReplayDB ---------------------------------------------------------

def test_replay_db_creates_unique_path_per_run(tmp_path):
    db1 = ReplayDB(data_dir=str(tmp_path))
    db2 = ReplayDB(data_dir=str(tmp_path))
    assert db1.run_id != db2.run_id
    assert db1.db_path != db2.db_path


def test_replay_db_install_sets_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("HL_BOT_DB", raising=False)
    with ReplayDB(run_id="t01", data_dir=str(tmp_path)) as db:
        assert os.environ["HL_BOT_DB"] == str(db.db_path)
    # After exit: env restored
    assert os.environ.get("HL_BOT_DB") is None


def test_replay_db_install_preserves_prior_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HL_BOT_DB", "/tmp/somethingelse.db")
    with ReplayDB(run_id="t02", data_dir=str(tmp_path)):
        assert os.environ["HL_BOT_DB"].endswith("replay_t02.db")
    assert os.environ["HL_BOT_DB"] == "/tmp/somethingelse.db"


def test_replay_db_init_schema_creates_expected_tables(tmp_path, monkeypatch):
    """init_schema should build the production tables in the replay DB."""
    monkeypatch.delenv("HL_BOT_DB", raising=False)
    with ReplayDB(run_id="t03", data_dir=str(tmp_path)) as db:
        db.init_schema()
        tables = db.list_tables()
        # Spot-check the critical ones
        for t in ("traders", "strategies", "paper_trades", "audit_trail"):
            assert t in tables, f"missing table {t}: have {tables}"


def test_replay_db_reset_clears_runtime_but_not_strategies(tmp_path, monkeypatch):
    """After reset, runtime tables are empty but strategies/traders persist."""
    monkeypatch.delenv("HL_BOT_DB", raising=False)
    with ReplayDB(run_id="t04", data_dir=str(tmp_path)) as db:
        db.init_schema()
        seed_into(str(db.db_path), build_default_smoke_snapshot())

        # Insert a paper trade row
        with sqlite3.connect(str(db.db_path)) as conn:
            conn.execute(
                """INSERT INTO paper_trades (coin, side, entry_price, size, status, opened_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("BTC", "buy", 50_000.0, 0.1, "OPEN", "2025-04-05T00:00:00Z"),
            )

        # Confirm rows exist
        assert db.table_count("paper_trades") == 1
        assert db.table_count("strategies") == 10
        assert db.table_count("traders") == 3

        # Reset: paper_trades cleared, reference data kept
        db.reset_runtime_state()
        assert db.table_count("paper_trades") == 0
        assert db.table_count("strategies") == 10
        assert db.table_count("traders") == 3


def test_replay_db_snapshot_to_copies_file(tmp_path, monkeypatch):
    monkeypatch.delenv("HL_BOT_DB", raising=False)
    dest = tmp_path / "archive.db"
    with ReplayDB(run_id="t05", data_dir=str(tmp_path)) as db:
        db.init_schema()
        db.snapshot_to(str(dest))
    assert dest.exists()
    # Sanity: it has the same tables
    with sqlite3.connect(f"file:{dest}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='strategies'"
        ).fetchall()
        assert rows


def test_replay_db_keep_on_exit_default_preserves_file(tmp_path, monkeypatch):
    monkeypatch.delenv("HL_BOT_DB", raising=False)
    with ReplayDB(run_id="t06", data_dir=str(tmp_path)) as db:
        db.init_schema()
        path = db.db_path
    assert path.exists(), "File should be kept by default for forensic analysis"


def test_replay_db_keep_on_exit_false_deletes(tmp_path, monkeypatch):
    monkeypatch.delenv("HL_BOT_DB", raising=False)
    with ReplayDB(run_id="t07", data_dir=str(tmp_path), keep_on_exit=False) as db:
        db.init_schema()
        path = db.db_path
        assert path.exists()
    assert not path.exists()


# --- Strategy seed ----------------------------------------------------

def test_default_smoke_snapshot_has_expected_shape():
    snap = build_default_smoke_snapshot()
    assert len(snap.traders) == 3
    assert len(snap.strategies) == 10
    # Mix of long/short, mix of BTC/ETH
    coins = {s.parameters.get("coin") for s in snap.strategies}
    types = {s.strategy_type for s in snap.strategies}
    assert "BTC" in coins and "ETH" in coins
    assert "momentum" in types and "mean_reversion" in types


def test_snapshot_roundtrip(tmp_path):
    snap = build_default_smoke_snapshot()
    path = tmp_path / "snap.json"
    save_snapshot(snap, str(path))
    loaded = load_snapshot(str(path))
    assert loaded.snapshot_date == snap.snapshot_date
    assert len(loaded.traders) == len(snap.traders)
    assert len(loaded.strategies) == len(snap.strategies)


def test_seed_into_inserts_rows(tmp_path, monkeypatch):
    monkeypatch.delenv("HL_BOT_DB", raising=False)
    with ReplayDB(run_id="t08", data_dir=str(tmp_path)) as db:
        db.init_schema()
        snap = build_default_smoke_snapshot()
        result = seed_into(str(db.db_path), snap)
        assert result["traders"] == 3
        assert result["strategies"] == 10
        assert db.table_count("traders") == 3
        assert db.table_count("strategies") == 10


def test_seed_into_replace_mode_clears_existing(tmp_path, monkeypatch):
    monkeypatch.delenv("HL_BOT_DB", raising=False)
    with ReplayDB(run_id="t09", data_dir=str(tmp_path)) as db:
        db.init_schema()
        seed_into(str(db.db_path), build_default_smoke_snapshot())
        # Seed again with replace=True; total count should equal new snapshot
        new_snap = SeedSnapshot(
            snapshot_date="2026-01-01",
            description="single",
            traders=[SeedTrader(address="0x1234")],
            strategies=[SeedStrategy(name="only_one", strategy_type="momentum")],
        )
        seed_into(str(db.db_path), new_snap, replace=True)
        assert db.table_count("traders") == 1
        assert db.table_count("strategies") == 1


def test_seed_into_strategy_parameters_round_trip_as_json(tmp_path, monkeypatch):
    """Parameter dict must serialize / deserialize correctly."""
    monkeypatch.delenv("HL_BOT_DB", raising=False)
    with ReplayDB(run_id="t10", data_dir=str(tmp_path)) as db:
        db.init_schema()
        custom = SeedSnapshot(
            snapshot_date="2025-04-01",
            description="param test",
            traders=[],
            strategies=[SeedStrategy(
                name="rsi_custom",
                strategy_type="rsi",
                parameters={"period": 28, "ob": 75, "os": 25, "coin": "BTC"},
            )],
        )
        seed_into(str(db.db_path), custom)
        with sqlite3.connect(str(db.db_path)) as conn:
            row = conn.execute("SELECT parameters FROM strategies WHERE name = 'rsi_custom'").fetchone()
            params = json.loads(row[0])
            assert params == {"period": 28, "ob": 75, "os": 25, "coin": "BTC"}


def test_export_from_live_db(tmp_path, monkeypatch):
    """export_from_live_db should round-trip through a snapshot."""
    monkeypatch.delenv("HL_BOT_DB", raising=False)
    with ReplayDB(run_id="t11", data_dir=str(tmp_path)) as db:
        db.init_schema()
        original = build_default_smoke_snapshot()
        seed_into(str(db.db_path), original)
        exported = export_from_live_db(str(db.db_path), snapshot_date="2025-09-01")
        # Same counts (active=1 in default snapshot)
        assert len(exported.traders) == 3
        assert len(exported.strategies) == 10
