import contextlib
import json
import sqlite3

from src.backtest.stress_test import DecisionStressTestEngine
from src.data import database as db
from src.data import decision_journal
from src.learning.conservative_calibrator import ConservativeDecisionCalibrator
from src.learning.dataset_builder import DecisionDatasetBuilder, DatasetBuildResult, LearningExample
from src.learning.decision_outcomes import compute_forward_labels
from src.learning.replay_backtester import ReplayPolicy
from src.learning.schema import ensure_sqlite_schema


@contextlib.contextmanager
def _sqlite_ctx(conn):
    yield conn
    conn.commit()


def _memory_db(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_sqlite_schema(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY,
            pnl REAL,
            status TEXT,
            closed_at TEXT,
            exit_price REAL,
            entry_price REAL,
            size REAL,
            leverage REAL,
            metadata TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candles (
            coin TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (coin, timeframe, timestamp_ms)
        )
        """
    )
    monkeypatch.setattr(db, "get_connection", lambda for_read=False: _sqlite_ctx(conn))
    monkeypatch.setattr(decision_journal.config, "DB_BACKEND", "sqlite")
    decision_journal._SCHEMA_READY = False
    decision_journal._SCHEMA_WARNED = False
    return conn


def _insert_rejected_decision(conn):
    conn.execute(
        """
        INSERT INTO decision_snapshots
        (decision_id, created_at, updated_at, signal_timestamp, coin, side, source,
         source_key, strategy_type, calibrated_confidence, final_status,
         rejection_reason, entry_price, proposed_size_usd, proposed_leverage,
         features, regime, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "d_reject",
            "2026-04-21T10:00:00+00:00",
            "2026-04-21T10:00:00+00:00",
            "2026-04-21T10:00:00+00:00",
            "BTC",
            "long",
            "strategy",
            "strategy:momentum",
            "momentum",
            0.72,
            "candidate",
            "",
            100.0,
            50.0,
            3.0,
            '{"momentum":0.8}',
            '{"overall_regime":"trending_up"}',
            "{}",
        ),
    )


def test_rejected_decision_gets_stage_outcome_and_dataset_label(monkeypatch):
    conn = _memory_db(monkeypatch)
    _insert_rejected_decision(conn)
    conn.commit()

    assert decision_journal.finalize_decision(
        "d_reject",
        final_status="firewall_prescreen_rejected",
        stage="firewall_prescreen",
        reason="confidence_floor",
        firewall_decision="rejected",
        confidence=0.72,
    )
    assert decision_journal.record_decision_outcome(
        "d_reject",
        forward_labels={
            "forward_return_1h": 0.02,
            "would_have_won": 1,
            "side_correct": 1,
            "missed_profit_usd": 3.0,
        },
    )

    stage_count = conn.execute("SELECT COUNT(*) AS n FROM decision_stage_events").fetchone()["n"]
    outcome = conn.execute("SELECT * FROM decision_outcomes WHERE decision_id = ?", ("d_reject",)).fetchone()
    assert stage_count == 1
    assert outcome["action_taken"] == 0
    assert "confidence_floor" in outcome["explanation"]

    dataset = DecisionDatasetBuilder().build(limit=10, persist=False)
    assert len(dataset.examples) == 1
    example = dataset.examples[0]
    assert example.executed is False
    assert example.label_win == 1
    assert example.outcome_pnl == 3.0
    assert example.rejection_reason == "confidence_floor"
    assert ReplayPolicy("default").accepts(example) is False
    assert ReplayPolicy("with_rejections", include_rejected=True).accepts(example) is True


def test_forward_labels_read_feature_store_candles(monkeypatch):
    conn = _memory_db(monkeypatch)
    conn.execute(
        """
        INSERT INTO candles
        (coin, timeframe, timestamp_ms, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("BTC", "5m", 1776766500000, 100.0, 103.0, 99.0, 102.0, 10.0),
    )
    conn.commit()

    labels = compute_forward_labels(
        {
            "decision_id": "d1",
            "coin": "BTC",
            "side": "long",
            "created_at": "2026-04-21T10:00:00+00:00",
            "entry_price": 100.0,
            "proposed_size_usd": 50.0,
            "proposed_leverage": 3.0,
        },
        primary_timeframes=("5m",),
    )

    assert round(labels["forward_return_15m"], 4) == 0.02
    assert labels["would_have_won"] == 1
    assert round(labels["missed_profit_usd"], 2) == 3.0


def test_conservative_calibrator_and_decision_stress_persist(monkeypatch):
    conn = _memory_db(monkeypatch)
    examples = [
        LearningExample(
            decision_id=f"d{idx}",
            coin="BTC",
            side="long",
            source="strategy",
            created_at=f"2026-04-21T10:{idx:02d}:00+00:00",
            features={"momentum": 1.0},
            confidence=0.7,
            executed=True,
            label_win=1 if idx < 18 else 0,
            outcome_pnl=2.0 if idx < 18 else -1.0,
            paper_trade_id=idx,
            source_key="strategy:momentum",
            strategy_type="momentum",
            outcome_return_pct=0.01 if idx < 18 else -0.005,
        )
        for idx in range(24)
    ]
    dataset = DatasetBuildResult("ds_decision", examples, ["momentum"], {"rows": 24})

    result = ConservativeDecisionCalibrator(min_group_examples=10).fit(dataset, persist=True)

    assert result.global_stats["win_rate"] > 0.7
    assert result.source_stats["strategy:momentum|long"]["action"] == "eligible_small_boost"
    assert conn.execute("SELECT COUNT(*) AS n FROM learning_decision_calibrators").fetchone()["n"] == 1

    row = {
        "decision_id": "d1",
        "side": "long",
        "action_taken": 1,
        "outcome_pnl": 2.0,
        "outcome_return_pct": 0.01,
        "hold_minutes": 60.0,
        "decision_metadata": json.dumps({"proposed_size_usd": 50.0, "proposed_leverage": 3.0}),
    }
    report = DecisionStressTestEngine(initial_balance=1000.0).run([row], scenarios=["flash_crash"])

    assert report.decision_count == 1
    assert report.scenarios[0].scenario_key == "flash_crash"
    assert report.scenarios[0].stressed_pnl < report.baseline_pnl
