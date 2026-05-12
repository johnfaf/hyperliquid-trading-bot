"""Tests for the audit-trail diff tool.

These are unit tests of the matching logic, not integration tests --
the matcher is the part that's easy to get wrong (off-by-one on time
windows, double-counting consumed replay rows, etc.).
"""
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Import the diff module directly (it lives in scripts/).
import importlib.util
import sys
SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "replay_audit_diff.py"
spec = importlib.util.spec_from_file_location("replay_audit_diff", SCRIPT_PATH)
audit_diff = importlib.util.module_from_spec(spec)
sys.modules["replay_audit_diff"] = audit_diff
spec.loader.exec_module(audit_diff)


def _make_audit_db(tmp_path, name, rows):
    """rows: list of (timestamp_iso, action, coin, side, source, details_dict)"""
    db = tmp_path / name
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE audit_trail (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            coin TEXT, side TEXT, price REAL, size REAL, pnl REAL,
            source TEXT, details TEXT DEFAULT '{}'
        )""")
    conn.executemany(
        "INSERT INTO audit_trail (timestamp, action, coin, side, source, details) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [(ts, act, coin, side, source, json.dumps(det))
         for (ts, act, coin, side, source, det) in rows],
    )
    conn.commit()
    conn.close()
    return str(db)


def _ts(year, month, day, hour=0, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc).isoformat()


def test_diff_exact_match(tmp_path):
    """Identical rows in both DBs -> 100% match."""
    rows = [
        (_ts(2025, 8, 1, 12, 0), "signal_approved", "BTC", "buy", "momentum", {}),
        (_ts(2025, 8, 1, 13, 0), "signal_rejected", "ETH", "buy", "rsi", {"reason": "cooldown"}),
    ]
    live = _make_audit_db(tmp_path, "live.db", rows)
    replay = _make_audit_db(tmp_path, "replay.db", rows)

    live_rows = audit_diff._load_audit(live, _ts(2025, 8, 1), _ts(2025, 8, 2))
    replay_rows = audit_diff._load_audit(replay, _ts(2025, 8, 1), _ts(2025, 8, 2))
    diff = audit_diff.diff_audit_trails(live_rows, replay_rows, match_window_s=60.0)
    assert diff.total_live == 2
    assert diff.total_replay == 2
    assert diff.matched == 2
    assert diff.live_only == 0
    assert diff.replay_only == 0


def test_diff_within_window_matches(tmp_path):
    """Live row at 12:00, replay row at 12:05 -> match with window=600s."""
    live = _make_audit_db(tmp_path, "live.db", [
        (_ts(2025, 8, 1, 12, 0), "signal_approved", "BTC", "buy", "momentum", {}),
    ])
    replay = _make_audit_db(tmp_path, "replay.db", [
        (_ts(2025, 8, 1, 12, 5), "signal_approved", "BTC", "buy", "momentum", {}),
    ])
    lr = audit_diff._load_audit(live, _ts(2025, 8, 1), _ts(2025, 8, 2))
    rr = audit_diff._load_audit(replay, _ts(2025, 8, 1), _ts(2025, 8, 2))
    diff = audit_diff.diff_audit_trails(lr, rr, match_window_s=600.0)
    assert diff.matched == 1
    assert diff.live_only == 0


def test_diff_outside_window_no_match(tmp_path):
    """Live row at 12:00, replay row at 13:00 (1h apart) with 5min window -> no match."""
    live = _make_audit_db(tmp_path, "live.db", [
        (_ts(2025, 8, 1, 12, 0), "signal_approved", "BTC", "buy", "momentum", {}),
    ])
    replay = _make_audit_db(tmp_path, "replay.db", [
        (_ts(2025, 8, 1, 13, 0), "signal_approved", "BTC", "buy", "momentum", {}),
    ])
    lr = audit_diff._load_audit(live, _ts(2025, 8, 1), _ts(2025, 8, 2))
    rr = audit_diff._load_audit(replay, _ts(2025, 8, 1), _ts(2025, 8, 2))
    diff = audit_diff.diff_audit_trails(lr, rr, match_window_s=300.0)
    assert diff.matched == 0
    assert diff.live_only == 1
    assert diff.replay_only == 1


def test_diff_different_coin_no_match(tmp_path):
    """Same timestamp/action but different coin -> no match."""
    live = _make_audit_db(tmp_path, "live.db", [
        (_ts(2025, 8, 1, 12, 0), "signal_approved", "BTC", "buy", "momentum", {}),
    ])
    replay = _make_audit_db(tmp_path, "replay.db", [
        (_ts(2025, 8, 1, 12, 0), "signal_approved", "ETH", "buy", "momentum", {}),
    ])
    lr = audit_diff._load_audit(live, _ts(2025, 8, 1), _ts(2025, 8, 2))
    rr = audit_diff._load_audit(replay, _ts(2025, 8, 1), _ts(2025, 8, 2))
    diff = audit_diff.diff_audit_trails(lr, rr, match_window_s=600.0)
    assert diff.matched == 0
    assert diff.live_only == 1
    assert diff.replay_only == 1


def test_diff_action_breakdown(tmp_path):
    """by_action counters should reflect each action category."""
    live = _make_audit_db(tmp_path, "live.db", [
        (_ts(2025, 8, 1, 12, 0), "signal_approved", "BTC", "buy", "momentum", {}),
        (_ts(2025, 8, 1, 12, 5), "signal_rejected", "BTC", "buy", "rsi", {"reason": "cooldown"}),
        (_ts(2025, 8, 1, 12, 10), "trade_closed", "BTC", "sell", "momentum", {}),
    ])
    replay = _make_audit_db(tmp_path, "replay.db", [
        (_ts(2025, 8, 1, 12, 0), "signal_approved", "BTC", "buy", "momentum", {}),
    ])
    lr = audit_diff._load_audit(live, _ts(2025, 8, 1), _ts(2025, 8, 2))
    rr = audit_diff._load_audit(replay, _ts(2025, 8, 1), _ts(2025, 8, 2))
    diff = audit_diff.diff_audit_trails(lr, rr, match_window_s=60.0)
    assert diff.by_action_live["signal_approved"] == 1
    assert diff.by_action_live["signal_rejected"] == 1
    assert diff.by_action_live["trade_closed"] == 1
    assert diff.by_action_matched["signal_approved"] == 1


def test_diff_reject_reason_tracked(tmp_path):
    """Live-only rows record their `details.reason` so the report shows why."""
    live = _make_audit_db(tmp_path, "live.db", [
        (_ts(2025, 8, 1, 12, 0), "signal_rejected", "BTC", "buy", "rsi",
         {"reason": "cooldown_active"}),
        (_ts(2025, 8, 1, 12, 5), "signal_rejected", "ETH", "buy", "rsi",
         {"reason": "regime_mismatch"}),
    ])
    replay = _make_audit_db(tmp_path, "replay.db", [])
    lr = audit_diff._load_audit(live, _ts(2025, 8, 1), _ts(2025, 8, 2))
    rr = audit_diff._load_audit(replay, _ts(2025, 8, 1), _ts(2025, 8, 2))
    diff = audit_diff.diff_audit_trails(lr, rr, match_window_s=60.0)
    assert diff.live_only_reasons["cooldown_active"] == 1
    assert diff.live_only_reasons["regime_mismatch"] == 1


def test_diff_missing_audit_table_is_empty_not_crash(tmp_path):
    """A DB without audit_trail should return [] rather than crash."""
    db = tmp_path / "no_audit.db"
    sqlite3.connect(str(db)).close()
    rows = audit_diff._load_audit(str(db), _ts(2025, 8, 1), _ts(2025, 8, 2))
    assert rows == []


def test_diff_missing_db_file_is_empty(tmp_path):
    rows = audit_diff._load_audit(str(tmp_path / "nope.db"),
                                  _ts(2025, 8, 1), _ts(2025, 8, 2))
    assert rows == []


def test_diff_replay_only_when_live_silent(tmp_path):
    """Replay invents activity that live didn't see -> replay_only."""
    live = _make_audit_db(tmp_path, "live.db", [])
    replay = _make_audit_db(tmp_path, "replay.db", [
        (_ts(2025, 8, 1, 12, 0), "signal_approved", "BTC", "buy", "momentum", {}),
        (_ts(2025, 8, 1, 12, 5), "signal_approved", "ETH", "buy", "rsi", {}),
    ])
    lr = audit_diff._load_audit(live, _ts(2025, 8, 1), _ts(2025, 8, 2))
    rr = audit_diff._load_audit(replay, _ts(2025, 8, 1), _ts(2025, 8, 2))
    diff = audit_diff.diff_audit_trails(lr, rr, match_window_s=60.0)
    assert diff.matched == 0
    assert diff.live_only == 0
    assert diff.replay_only == 2


def test_diff_one_replay_row_matches_one_live_row(tmp_path):
    """If live has 2 rows but replay has 1, only one should match."""
    live = _make_audit_db(tmp_path, "live.db", [
        (_ts(2025, 8, 1, 12, 0), "signal_approved", "BTC", "buy", "momentum", {}),
        (_ts(2025, 8, 1, 12, 1), "signal_approved", "BTC", "buy", "momentum", {}),
    ])
    replay = _make_audit_db(tmp_path, "replay.db", [
        (_ts(2025, 8, 1, 12, 0), "signal_approved", "BTC", "buy", "momentum", {}),
    ])
    lr = audit_diff._load_audit(live, _ts(2025, 8, 1), _ts(2025, 8, 2))
    rr = audit_diff._load_audit(replay, _ts(2025, 8, 1), _ts(2025, 8, 2))
    diff = audit_diff.diff_audit_trails(lr, rr, match_window_s=600.0)
    # The first live row matches the single replay row; second is live_only.
    assert diff.matched == 1
    assert diff.live_only == 1
    assert diff.replay_only == 0
