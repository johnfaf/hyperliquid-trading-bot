"""Tests for src/learning/calibration_bootstrap.py."""
from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.learning import calibration_bootstrap as cb


@contextlib.contextmanager
def _sqlite_ctx(conn: sqlite3.Connection):
    yield conn
    conn.commit()


@pytest.fixture
def paper_trades_db(monkeypatch):
    """In-memory DB with the minimum paper_trades columns the bootstrap reads."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY,
            coin TEXT,
            side TEXT,
            pnl REAL,
            status TEXT,
            closed_at TEXT,
            metadata TEXT
        )
        """
    )
    conn.commit()
    monkeypatch.setattr(cb.db, "get_connection", lambda for_read=False: _sqlite_ctx(conn))
    yield conn


def _insert_trade(conn, *, tid, coin, side, pnl, metadata=None,
                  hours_ago=1, status="closed"):
    closed_at = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    conn.execute(
        "INSERT INTO paper_trades (id, coin, side, pnl, status, closed_at, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (tid, coin, side, pnl, status, closed_at, json.dumps(metadata or {})),
    )
    conn.commit()


class _FakeCalibration:
    """Counting stub modeling the small subset of CalibrationTracker the
    bootstrap touches."""
    def __init__(self, *, prefilled=None):
        self.records = []
        # Optional pre-filled sample sizes per resolved key, to test
        # the idempotency skip.
        self.prefilled = prefilled or {}

    def _resolve_key(self, source_key, *, side=None, regime=None):
        parts = [source_key]
        if side:
            parts.append(side)
        if regime:
            parts.append(regime)
        return "|".join(parts)

    def get_sample_size(self, key):
        return float(self.prefilled.get(key, 0))

    def record(self, *, source_key, predicted_confidence, actual_win,
               pnl=0.0, coin="", side="", regime=None):
        self.records.append({
            "source_key": source_key, "side": side, "regime": regime,
            "confidence": predicted_confidence, "win": actual_win,
            "pnl": pnl, "coin": coin,
        })


def test_returns_zero_when_no_calibration_tracker():
    stats = cb.bootstrap_calibration_from_history(None)
    assert stats == {"trades_read": 0, "records_seeded": 0,
                     "buckets_skipped_full": 0, "buckets_seeded": 0}


def test_returns_zero_when_no_clean_trades(paper_trades_db):
    cal = _FakeCalibration()
    stats = cb.bootstrap_calibration_from_history(cal)
    assert stats["trades_read"] == 0
    assert stats["records_seeded"] == 0
    assert cal.records == []


def test_seeds_calibration_from_closed_trades(paper_trades_db):
    _insert_trade(
        paper_trades_db, tid=1, coin="BTC", side="long", pnl=2.5,
        metadata={"source_key": "strategy:momentum_long",
                  "confidence": 0.65, "regime": "trending_up"},
    )
    _insert_trade(
        paper_trades_db, tid=2, coin="ETH", side="short", pnl=-1.2,
        metadata={"source_key": "strategy:momentum_short",
                  "confidence": 0.55, "regime": "ranging"},
    )
    cal = _FakeCalibration()
    stats = cb.bootstrap_calibration_from_history(cal)

    assert stats["trades_read"] == 2
    assert stats["records_seeded"] == 2
    assert stats["buckets_seeded"] == 2
    assert len(cal.records) == 2
    btc = next(r for r in cal.records if r["coin"] == "BTC")
    assert btc["source_key"] == "strategy:momentum_long"
    assert btc["side"] == "long"
    assert btc["regime"] == "trending_up"
    assert btc["win"] is True
    eth = next(r for r in cal.records if r["coin"] == "ETH")
    assert eth["win"] is False


def test_excludes_tainted_trades(paper_trades_db):
    _insert_trade(
        paper_trades_db, tid=1, coin="BTC", side="long", pnl=-50.0,
        metadata={"source_key": "strategy:momentum_long",
                  "tainted": True, "taint_reason": "reconciler_kill_pre_fix"},
    )
    _insert_trade(
        paper_trades_db, tid=2, coin="BTC", side="long", pnl=2.5,
        metadata={"source_key": "strategy:momentum_long",
                  "confidence": 0.6},
    )
    cal = _FakeCalibration()
    stats = cb.bootstrap_calibration_from_history(cal)
    assert stats["trades_read"] == 1, "tainted trades must be filtered out"
    assert stats["records_seeded"] == 1
    assert cal.records[0]["win"] is True  # only the clean +2.5 trade


def test_idempotent_skip_when_bucket_already_full(paper_trades_db):
    _insert_trade(
        paper_trades_db, tid=1, coin="BTC", side="long", pnl=2.5,
        metadata={"source_key": "strategy:momentum_long",
                  "confidence": 0.6, "regime": "trending_up"},
    )
    cal = _FakeCalibration(prefilled={
        # resolved key matches what _FakeCalibration._resolve_key returns
        "strategy:momentum_long|long|trending_up": 999,  # > skip_threshold (default 100)
    })
    stats = cb.bootstrap_calibration_from_history(cal)
    assert stats["records_seeded"] == 0
    assert stats["buckets_skipped_full"] == 1
    assert cal.records == []


def test_caps_records_per_bucket(paper_trades_db):
    # 20 winning BTC longs from the same source -- should be capped
    # to cap_per_bucket=5 across one bucket.
    for i in range(20):
        _insert_trade(
            paper_trades_db, tid=i + 1, coin="BTC", side="long", pnl=1.0 + i,
            metadata={"source_key": "strategy:momentum_long",
                      "confidence": 0.6, "regime": "trending_up"},
            hours_ago=1 + i,
        )
    cal = _FakeCalibration()
    stats = cb.bootstrap_calibration_from_history(cal, cap_per_bucket=5)
    assert stats["trades_read"] == 20
    assert stats["records_seeded"] == 5
    assert stats["buckets_seeded"] == 1


def test_lookback_window_excludes_old_trades(paper_trades_db):
    _insert_trade(
        paper_trades_db, tid=1, coin="BTC", side="long", pnl=1.0,
        metadata={"source_key": "strategy:m", "confidence": 0.5},
        hours_ago=1,  # recent
    )
    _insert_trade(
        paper_trades_db, tid=2, coin="BTC", side="long", pnl=1.0,
        metadata={"source_key": "strategy:m", "confidence": 0.5},
        hours_ago=24 * 60,  # 60 days ago
    )
    cal = _FakeCalibration()
    stats = cb.bootstrap_calibration_from_history(cal, lookback_days_v=30)
    assert stats["trades_read"] == 1
    assert stats["records_seeded"] == 1


def test_derives_copy_trade_address_for_source_key(paper_trades_db):
    _insert_trade(
        paper_trades_db, tid=1, coin="HYPE", side="long", pnl=1.5,
        metadata={"source": "copy_trade",
                  "source_trader": "0xABCDEF",
                  "confidence": 0.6},
    )
    cal = _FakeCalibration()
    cb.bootstrap_calibration_from_history(cal)
    assert cal.records[0]["source_key"] == "copy_trade:0xabcdef"


def test_skips_trades_missing_critical_fields(paper_trades_db):
    # Missing coin
    _insert_trade(
        paper_trades_db, tid=1, coin="", side="long", pnl=1.0,
        metadata={"source_key": "strategy:m"},
    )
    # Bad side
    _insert_trade(
        paper_trades_db, tid=2, coin="BTC", side="neutral", pnl=1.0,
        metadata={"source_key": "strategy:m"},
    )
    cal = _FakeCalibration()
    stats = cb.bootstrap_calibration_from_history(cal)
    assert stats["trades_read"] == 2  # query found them
    assert stats["records_seeded"] == 0  # but the per-row filter rejected
    assert cal.records == []


def test_env_default_off():
    """Bootstrap must NOT run on boot unless the env flag is set."""
    import os
    saved = os.environ.pop("CALIBRATION_BOOTSTRAP_ON_BOOT", None)
    try:
        assert cb.bootstrap_enabled_on_boot() is False
    finally:
        if saved is not None:
            os.environ["CALIBRATION_BOOTSTRAP_ON_BOOT"] = saved


def test_env_truthy_values(monkeypatch):
    monkeypatch.setenv("CALIBRATION_BOOTSTRAP_ON_BOOT", "1")
    assert cb.bootstrap_enabled_on_boot() is True
    monkeypatch.setenv("CALIBRATION_BOOTSTRAP_ON_BOOT", "true")
    assert cb.bootstrap_enabled_on_boot() is True
    monkeypatch.setenv("CALIBRATION_BOOTSTRAP_ON_BOOT", "0")
    assert cb.bootstrap_enabled_on_boot() is False
