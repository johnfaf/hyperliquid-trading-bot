import contextlib
import sqlite3

from src.data import database as db
from src.learning.dataset_builder import DecisionDatasetBuilder, DatasetBuildResult, LearningExample
from src.learning.improvement_loop import OfflineImprovementRunner
from src.learning.promotion import ManualPromotionController
from src.learning.replay_backtester import DecisionReplayBacktester, ReplayBacktestResult, ReplayPolicy
from src.learning.schema import ensure_sqlite_schema
from src.learning.shadow_evaluator import ShadowPolicyEvaluator


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
            exit_price REAL
        )
        """
    )
    monkeypatch.setattr(db, "get_connection", lambda for_read=False: _sqlite_ctx(conn))
    return conn


def _insert_decision(conn, decision_id, confidence, pnl, paper_trade_id, created_at):
    conn.execute(
        "INSERT INTO paper_trades (id, pnl, status, closed_at, exit_price) VALUES (?, ?, ?, ?, ?)",
        (paper_trade_id, pnl, "closed", created_at, 101.0),
    )
    conn.execute(
        """
        INSERT INTO decision_snapshots
        (decision_id, created_at, updated_at, coin, side, source,
         calibrated_confidence, final_status, paper_trade_id, features, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            decision_id,
            created_at,
            created_at,
            "BTC",
            "long",
            "strategy",
            confidence,
            "paper_opened",
            paper_trade_id,
            '{"return_3":0.01,"spread_bps":2.0}',
            "{}",
        ),
    )


def test_phase6_dataset_builder_labels_and_persists(monkeypatch):
    conn = _memory_db(monkeypatch)
    _insert_decision(conn, "d1", 0.70, 12.5, 1, "2026-04-21T10:00:00+00:00")
    _insert_decision(conn, "d2", 0.60, -3.0, 2, "2026-04-21T11:00:00+00:00")
    conn.commit()

    result = DecisionDatasetBuilder().build(limit=10, persist=True)

    assert len(result.examples) == 2
    assert result.quality_report["labelled"] == 2
    assert result.quality_report["positive_labels"] == 1
    assert {"return_3", "spread_bps"}.issubset(set(result.feature_names))
    row = conn.execute("SELECT * FROM learning_datasets WHERE dataset_id = ?", (result.dataset_id,)).fetchone()
    assert row is not None
    assert row["row_count"] == 2


def test_phase7_replay_backtester_records_policy_result(monkeypatch):
    conn = _memory_db(monkeypatch)
    examples = [
        LearningExample("d1", "BTC", "long", "strategy", "t1", {}, 0.70, True, 1, 10.0, 1),
        LearningExample("d2", "BTC", "long", "strategy", "t2", {}, 0.55, True, 0, -4.0, 2),
        LearningExample("d3", "BTC", "long", "strategy", "t3", {}, 0.75, True, 1, 8.0, 3),
        LearningExample("d4", "BTC", "long", "strategy", "t4", {}, 0.50, True, 0, -2.0, 4),
    ]
    dataset = DatasetBuildResult("ds1", examples, [], {"rows": 4})

    result = DecisionReplayBacktester(min_trades=1, train_fraction=0.5, min_test_trades=1).run(
        dataset,
        ReplayPolicy("candidate_conf_0p60", min_confidence=0.60),
        persist=True,
    )

    assert result.trade_count == 1
    assert result.total_pnl == 8.0
    assert result.passed is True
    assert result.metrics["split"]["train_passed"] is True
    assert result.metrics["split"]["test_passed"] is True
    row = conn.execute("SELECT * FROM learning_backtest_runs WHERE run_id = ?", (result.run_id,)).fetchone()
    assert row is not None


def test_phase7_replay_backtester_rejects_in_sample_only_edge(monkeypatch):
    _memory_db(monkeypatch)
    examples = [
        LearningExample("d1", "BTC", "long", "strategy", "t1", {}, 0.80, True, 1, 10.0, 1),
        LearningExample("d2", "BTC", "long", "strategy", "t2", {}, 0.75, True, 1, 5.0, 2),
        LearningExample("d3", "BTC", "long", "strategy", "t3", {}, 0.85, True, 0, -6.0, 3),
        LearningExample("d4", "BTC", "long", "strategy", "t4", {}, 0.90, True, 0, -4.0, 4),
    ]
    dataset = DatasetBuildResult("ds_oos_reject", examples, [], {"rows": 4})

    result = DecisionReplayBacktester(min_trades=1, train_fraction=0.5, min_test_trades=1).run(
        dataset,
        ReplayPolicy("candidate_conf_0p70", min_confidence=0.70),
        persist=False,
    )

    assert result.metrics["split"]["train_passed"] is True
    assert result.metrics["split"]["test_passed"] is False
    assert result.passed is False
    assert result.total_pnl == -10.0


def _bt(run_id, candidate, trades, pnl, avg, dd, pf, wr=0.6, passed=True):
    return ReplayBacktestResult(
        run_id=run_id,
        dataset_id="ds1",
        policy_id="champion_policy_v1",
        candidate_policy_id=candidate,
        trade_count=trades,
        win_rate=wr,
        total_pnl=pnl,
        avg_pnl=avg,
        max_drawdown=dd,
        profit_factor=pf,
        sharpe_like=1.0,
        metrics={},
        parameters={},
        passed=passed,
    )


def test_phase9_score_prefers_oos_quality_over_raw_dollar_pnl():
    raw_dollar_but_bad_risk = _bt("raw", "candidate_raw", 100, 5000.0, 1.0, 3000.0, 2.0, wr=0.6)
    smaller_but_clean = _bt("clean", "candidate_clean", 10, 100.0, 10.0, 5.0, 2.0, wr=0.7)

    assert OfflineImprovementRunner._score(smaller_but_clean) > OfflineImprovementRunner._score(raw_dollar_but_bad_risk)


def test_phase8_shadow_evaluator_and_phase10_manual_promotion(monkeypatch):
    conn = _memory_db(monkeypatch)
    champion = _bt("champ", "champion_policy_v1", 3, 5.0, 1.0, 1.0, 1.2)
    challenger = _bt("chall", "candidate_conf_0p65", 3, 12.0, 2.0, 1.0, 1.8)

    evaluation = ShadowPolicyEvaluator(
        {
            "min_out_of_sample_trades": 1,
            "must_beat_champion_after_fees_bps": 0,
            "max_drawdown_worse_than_champion_bps": 0,
        }
    ).evaluate(champion, challenger, persist=True)

    assert evaluation.gates_passed is True
    assert evaluation.verdict == "eligible_for_shadow"
    pending = ManualPromotionController().decide(evaluation, manual_approval=False, persist=True)
    approved = ManualPromotionController().decide(evaluation, manual_approval=True, persist=True)
    assert pending.approved is False
    assert pending.decision == "pending_manual_approval"
    assert approved.approved is True
    assert approved.rollback_policy_id == "champion_policy_v1"
    assert conn.execute("SELECT COUNT(*) AS n FROM learning_promotion_decisions").fetchone()["n"] == 2


def test_phase9_improvement_runner_records_offline_search(monkeypatch):
    conn = _memory_db(monkeypatch)
    examples = []
    for idx in range(80):
        winner = idx % 4 != 0
        confidence = 0.85 if winner else 0.55
        pnl = 2.0 if winner else -3.0
        examples.append(
            LearningExample(f"d{idx}", "BTC", "long", "strategy", f"t{idx}", {}, confidence, True, 1 if pnl > 0 else 0, pnl, idx)
        )
    dataset = DatasetBuildResult("ds_search", examples, [], {"rows": len(examples)})

    result = OfflineImprovementRunner(thresholds=[0.50, 0.80]).run(dataset=dataset, persist=True)

    assert result.best_candidate_policy_id == "candidate_conf_0p80"
    assert result.next_action == "shadow_evaluation"
    row = conn.execute(
        "SELECT * FROM learning_improvement_runs WHERE improvement_id = ?",
        (result.improvement_id,),
    ).fetchone()
    assert row is not None
    assert row["next_action"] == "shadow_evaluation"
