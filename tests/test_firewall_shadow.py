"""Tests for src/signals/firewall_shadow.py.

Covers:
- Sampling fraction (0.0 = off, 1.0 = always)
- Schema bootstrap
- Recording path (signal -> DB row)
- Evaluator: aged signals only, win/loss labelling, calibration feed
- Failure modes are fail-open (no exception bubbles)
"""
from __future__ import annotations

import contextlib
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.signals import firewall_shadow


# ── Test fixtures ───────────────────────────────────────────────


@contextlib.contextmanager
def _sqlite_ctx(conn: sqlite3.Connection):
    yield conn
    conn.commit()


@pytest.fixture
def shadow_db(monkeypatch):
    """In-memory SQLite with the firewall_shadow_signals schema, wired
    into the shared db connection helper."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    monkeypatch.setattr(
        firewall_shadow.db,
        "get_connection",
        lambda for_read=False: _sqlite_ctx(conn),
    )
    monkeypatch.setattr(
        firewall_shadow.db,
        "get_backend_name",
        lambda: "sqlite",
        raising=False,
    )
    # Each test gets a fresh in-memory DB, so force schema creation
    # past the process-level _SCHEMA_READY guard.
    firewall_shadow._SCHEMA_READY = False
    firewall_shadow._ensure_schema(conn, force=True)
    yield conn
    firewall_shadow._SCHEMA_READY = False


def _signal(coin="BTC", side="long", confidence=0.65, source="strategy",
            strategy_type="momentum_long", entry_price=70_000.0,
            trader_address=""):
    """Minimal duck-typed signal that exercises record_shadow_signal."""
    side_obj = SimpleNamespace(value=side)
    return SimpleNamespace(
        coin=coin,
        side=side_obj,
        confidence=confidence,
        source=SimpleNamespace(value=source),
        strategy_type=strategy_type,
        entry_price=entry_price,
        trader_address=trader_address,
    )


class _FakeCalibration:
    def __init__(self):
        self.records: list[dict] = []

    def record(self, *, source_key, predicted_confidence, actual_win,
               pnl=0.0, coin="", side="", regime=None):
        self.records.append({
            "source_key": source_key,
            "predicted_confidence": predicted_confidence,
            "actual_win": actual_win,
            "pnl": pnl,
            "coin": coin,
            "side": side,
            "regime": regime,
        })


# ── Config helpers ──────────────────────────────────────────────


def test_shadow_fraction_default_zero(monkeypatch):
    monkeypatch.delenv("FIREWALL_SHADOW_MODE_FRACTION", raising=False)
    assert firewall_shadow.shadow_fraction() == 0.0


def test_shadow_fraction_clamps_to_range(monkeypatch):
    monkeypatch.setenv("FIREWALL_SHADOW_MODE_FRACTION", "2.5")
    assert firewall_shadow.shadow_fraction() == 1.0
    monkeypatch.setenv("FIREWALL_SHADOW_MODE_FRACTION", "-0.1")
    assert firewall_shadow.shadow_fraction() == 0.0


def test_shadow_hold_minutes_default(monkeypatch):
    monkeypatch.delenv("FIREWALL_SHADOW_HOLD_MINUTES", raising=False)
    assert firewall_shadow.shadow_hold_minutes() == 60


# ── Recording ───────────────────────────────────────────────────


def test_record_off_when_fraction_zero(shadow_db, monkeypatch):
    """Default config -> no recording, no DB write."""
    monkeypatch.delenv("FIREWALL_SHADOW_MODE_FRACTION", raising=False)
    result = firewall_shadow.record_shadow_signal(
        _signal(), "Recent strategy:momentum_short underperforming"
    )
    assert result is False
    n = shadow_db.execute("SELECT COUNT(*) FROM firewall_shadow_signals").fetchone()[0]
    assert n == 0


def test_record_at_fraction_1_always_records(shadow_db, monkeypatch):
    monkeypatch.setenv("FIREWALL_SHADOW_MODE_FRACTION", "1.0")
    rng = random.Random(42)
    for _ in range(5):
        assert firewall_shadow.record_shadow_signal(
            _signal(), "test_reason", rng=rng,
        ) is True
    n = shadow_db.execute("SELECT COUNT(*) FROM firewall_shadow_signals").fetchone()[0]
    assert n == 5


def test_record_persists_canonical_fields(shadow_db, monkeypatch):
    monkeypatch.setenv("FIREWALL_SHADOW_MODE_FRACTION", "1.0")
    sig = _signal(coin="ETH", side="short", confidence=0.42,
                  source="strategy", strategy_type="mean_reversion",
                  entry_price=2_100.0)
    firewall_shadow.record_shadow_signal(
        sig, "Source allocator paused strategy:mean_reversion",
        regime="trending_down",
    )
    row = shadow_db.execute(
        "SELECT coin, side, confidence, source_key, entry_price, "
        "rejection_reason, regime, evaluated FROM firewall_shadow_signals"
    ).fetchone()
    assert dict(row) == {
        "coin": "ETH",
        "side": "short",
        "confidence": 0.42,
        "source_key": "strategy:mean_reversion",
        "entry_price": 2_100.0,
        "rejection_reason": "Source allocator paused strategy:mean_reversion",
        "regime": "trending_down",
        "evaluated": 0,
    }


def test_record_uses_copy_trade_address_for_source_key(shadow_db, monkeypatch):
    monkeypatch.setenv("FIREWALL_SHADOW_MODE_FRACTION", "1.0")
    sig = _signal(source="copy_trade", strategy_type="",
                  trader_address="0xABC123")
    firewall_shadow.record_shadow_signal(sig, "ev_below_threshold")
    row = shadow_db.execute(
        "SELECT source_key FROM firewall_shadow_signals"
    ).fetchone()
    assert row["source_key"] == "copy_trade:0xabc123"


def test_record_skips_invalid_side(shadow_db, monkeypatch):
    monkeypatch.setenv("FIREWALL_SHADOW_MODE_FRACTION", "1.0")
    sig = _signal()
    sig.side = SimpleNamespace(value="neutral")
    result = firewall_shadow.record_shadow_signal(sig, "test")
    assert result is False
    assert shadow_db.execute("SELECT COUNT(*) FROM firewall_shadow_signals").fetchone()[0] == 0


def test_record_skips_zero_entry_price(shadow_db, monkeypatch):
    monkeypatch.setenv("FIREWALL_SHADOW_MODE_FRACTION", "1.0")
    sig = _signal(entry_price=0.0)
    result = firewall_shadow.record_shadow_signal(sig, "test")
    assert result is False


def test_record_fail_open_on_db_error(monkeypatch):
    """If the DB write throws, record_shadow_signal must not raise."""
    monkeypatch.setenv("FIREWALL_SHADOW_MODE_FRACTION", "1.0")

    @contextlib.contextmanager
    def _broken_conn(*_args, **_kwargs):
        raise RuntimeError("DB is on fire")
        yield None  # unreachable

    monkeypatch.setattr(firewall_shadow.db, "get_connection", _broken_conn)
    # Must not raise.
    assert firewall_shadow.record_shadow_signal(_signal(), "test") is False


# ── Evaluator ───────────────────────────────────────────────────


def test_evaluator_off_when_fraction_zero(shadow_db, monkeypatch):
    monkeypatch.delenv("FIREWALL_SHADOW_MODE_FRACTION", raising=False)
    stats = firewall_shadow.evaluate_pending_shadow_signals()
    assert stats == {"evaluated": 0, "wins": 0, "losses": 0, "skipped": 0}


def test_evaluator_only_aged_rows(shadow_db, monkeypatch):
    """Only rows older than FIREWALL_SHADOW_HOLD_MINUTES are evaluated."""
    monkeypatch.setenv("FIREWALL_SHADOW_MODE_FRACTION", "1.0")
    monkeypatch.setenv("FIREWALL_SHADOW_HOLD_MINUTES", "30")

    now = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
    old_ts = (now - timedelta(hours=2)).isoformat()
    new_ts = (now - timedelta(minutes=5)).isoformat()
    shadow_db.execute(
        "INSERT INTO firewall_shadow_signals (coin, side, confidence, "
        "source_key, entry_price, rejection_reason, opened_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("BTC", "long", 0.5, "strategy:m", 70_000.0, "r1", old_ts),
    )
    shadow_db.execute(
        "INSERT INTO firewall_shadow_signals (coin, side, confidence, "
        "source_key, entry_price, rejection_reason, opened_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("BTC", "long", 0.5, "strategy:m", 70_000.0, "r2", new_ts),
    )

    monkeypatch.setattr(
        "src.signals.firewall_shadow.get_all_mids",
        lambda: {"BTC": 70_500.0},  # +71 bps -> win
    )
    cal = _FakeCalibration()
    stats = firewall_shadow.evaluate_pending_shadow_signals(
        calibration_tracker=cal, now=now,
    )
    assert stats == {"evaluated": 1, "wins": 1, "losses": 0, "skipped": 0}
    assert len(cal.records) == 1
    assert cal.records[0]["actual_win"] is True


def test_evaluator_labels_long_win(shadow_db, monkeypatch):
    monkeypatch.setenv("FIREWALL_SHADOW_MODE_FRACTION", "1.0")
    monkeypatch.setenv("FIREWALL_SHADOW_WIN_BPS", "20")

    now = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
    shadow_db.execute(
        "INSERT INTO firewall_shadow_signals (coin, side, confidence, "
        "source_key, entry_price, rejection_reason, opened_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("BTC", "long", 0.65, "strategy:m", 70_000.0, "r", (now - timedelta(hours=2)).isoformat()),
    )
    monkeypatch.setattr(
        "src.signals.firewall_shadow.get_all_mids",
        lambda: {"BTC": 70_140.1},  # +20 bps exactly
    )
    cal = _FakeCalibration()
    stats = firewall_shadow.evaluate_pending_shadow_signals(
        calibration_tracker=cal, now=now,
    )
    assert stats["wins"] == 1
    assert cal.records[0]["actual_win"] is True
    assert cal.records[0]["coin"] == "BTC"
    assert cal.records[0]["side"] == "long"


def test_evaluator_labels_short_win(shadow_db, monkeypatch):
    """A short signal wins when price moves DOWN -- sign convention."""
    monkeypatch.setenv("FIREWALL_SHADOW_MODE_FRACTION", "1.0")
    monkeypatch.setenv("FIREWALL_SHADOW_WIN_BPS", "20")

    now = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
    shadow_db.execute(
        "INSERT INTO firewall_shadow_signals (coin, side, confidence, "
        "source_key, entry_price, rejection_reason, opened_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("BTC", "short", 0.55, "strategy:m_s", 70_000.0, "r", (now - timedelta(hours=2)).isoformat()),
    )
    monkeypatch.setattr(
        "src.signals.firewall_shadow.get_all_mids",
        lambda: {"BTC": 69_500.0},  # price down by ~71 bps -> short wins
    )
    cal = _FakeCalibration()
    stats = firewall_shadow.evaluate_pending_shadow_signals(
        calibration_tracker=cal, now=now,
    )
    assert stats["wins"] == 1
    assert stats["losses"] == 0


def test_evaluator_labels_loss(shadow_db, monkeypatch):
    monkeypatch.setenv("FIREWALL_SHADOW_MODE_FRACTION", "1.0")
    monkeypatch.setenv("FIREWALL_SHADOW_WIN_BPS", "20")

    now = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
    shadow_db.execute(
        "INSERT INTO firewall_shadow_signals (coin, side, confidence, "
        "source_key, entry_price, rejection_reason, opened_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("BTC", "long", 0.65, "strategy:m", 70_000.0, "r", (now - timedelta(hours=2)).isoformat()),
    )
    monkeypatch.setattr(
        "src.signals.firewall_shadow.get_all_mids",
        lambda: {"BTC": 69_500.0},  # price down -> long loses
    )
    cal = _FakeCalibration()
    stats = firewall_shadow.evaluate_pending_shadow_signals(
        calibration_tracker=cal, now=now,
    )
    assert stats["losses"] == 1
    assert cal.records[0]["actual_win"] is False
    assert cal.records[0]["pnl"] < 0


def test_evaluator_marks_row_evaluated(shadow_db, monkeypatch):
    monkeypatch.setenv("FIREWALL_SHADOW_MODE_FRACTION", "1.0")
    now = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
    shadow_db.execute(
        "INSERT INTO firewall_shadow_signals (coin, side, confidence, "
        "source_key, entry_price, rejection_reason, opened_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("BTC", "long", 0.65, "strategy:m", 70_000.0, "r", (now - timedelta(hours=2)).isoformat()),
    )
    monkeypatch.setattr(
        "src.signals.firewall_shadow.get_all_mids",
        lambda: {"BTC": 70_500.0},
    )
    firewall_shadow.evaluate_pending_shadow_signals(now=now)

    row = shadow_db.execute(
        "SELECT evaluated, evaluated_at, simulated_win, simulated_exit_price, "
        "simulated_pnl_pct FROM firewall_shadow_signals"
    ).fetchone()
    assert row["evaluated"] == 1
    assert row["evaluated_at"] is not None
    assert row["simulated_win"] == 1
    assert row["simulated_exit_price"] == 70_500.0


def test_evaluator_skips_missing_mid(shadow_db, monkeypatch):
    """If no mid is available for the coin, leave row pending for next pass."""
    monkeypatch.setenv("FIREWALL_SHADOW_MODE_FRACTION", "1.0")
    now = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
    shadow_db.execute(
        "INSERT INTO firewall_shadow_signals (coin, side, confidence, "
        "source_key, entry_price, rejection_reason, opened_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("DOESNOTEXIST", "long", 0.5, "strategy:m", 100.0, "r",
         (now - timedelta(hours=2)).isoformat()),
    )
    monkeypatch.setattr(
        "src.signals.firewall_shadow.get_all_mids",
        lambda: {"BTC": 70_000.0},  # DOESNOTEXIST absent
    )
    stats = firewall_shadow.evaluate_pending_shadow_signals(now=now)
    assert stats["skipped"] == 1
    assert stats["evaluated"] == 0
    # Row is still pending
    row = shadow_db.execute(
        "SELECT evaluated FROM firewall_shadow_signals"
    ).fetchone()
    assert row["evaluated"] == 0


def test_evaluator_max_per_call_bound(shadow_db, monkeypatch):
    monkeypatch.setenv("FIREWALL_SHADOW_MODE_FRACTION", "1.0")
    now = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(10):
        shadow_db.execute(
            "INSERT INTO firewall_shadow_signals (coin, side, confidence, "
            "source_key, entry_price, rejection_reason, opened_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("BTC", "long", 0.5, "strategy:m", 70_000.0, f"r{i}",
             (now - timedelta(hours=2, seconds=i)).isoformat()),
        )
    monkeypatch.setattr(
        "src.signals.firewall_shadow.get_all_mids",
        lambda: {"BTC": 70_500.0},
    )
    stats = firewall_shadow.evaluate_pending_shadow_signals(
        max_per_call=3, now=now,
    )
    assert stats["evaluated"] == 3
    # The remaining 7 are still pending.
    pending = shadow_db.execute(
        "SELECT COUNT(*) FROM firewall_shadow_signals WHERE evaluated = 0"
    ).fetchone()[0]
    assert pending == 7
