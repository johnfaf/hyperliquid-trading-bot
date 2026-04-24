import contextlib
import json
import sqlite3

from src.data import database as db
from src.learning.candidate_registry import CandidatePolicyRegistry
from src.learning.data_quality import DatasetQualityAuditor
from src.learning.dataset_builder import DatasetBuildResult, LearningExample
from src.learning.drift_monitor import FeatureDriftMonitor
from src.learning.feature_attribution import FeatureAttributionAnalyzer
from src.learning.orchestrator import ContinuousLearningOrchestrator
from src.learning.policy_registry import CHAMPION_POLICY_ID
from src.learning.promotion import ManualPromotionController
from src.learning.promotion_package import (
    PromotionPackageBuilder,
    RollbackReadinessChecker,
    ShadowPeriodPlanner,
)
from src.learning.replay_backtester import DecisionReplayBacktester, ReplayPolicy
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
    monkeypatch.setattr(db, "get_connection", lambda for_read=False: _sqlite_ctx(conn))
    return conn


def _dataset(dataset_id="ds_phase_11_19", n=60, shift=0.0):
    examples = []
    for idx in range(n):
        winner = idx % 4 != 0
        confidence = 0.86 if winner else 0.54
        pnl = 2.0 if winner else -0.5
        examples.append(
            LearningExample(
                decision_id=f"{dataset_id}_d{idx}",
                coin="BTC",
                side="long",
                source="strategy",
                created_at=f"2026-04-21T10:{idx % 60:02d}:00+00:00",
                features={
                    "momentum": (1.0 if winner else -1.0) + shift,
                    "spread_bps": 2.0 + shift,
                },
                confidence=confidence,
                executed=True,
                label_win=1 if winner else 0,
                outcome_pnl=pnl,
                paper_trade_id=idx + 1,
            )
        )
    return DatasetBuildResult(
        dataset_id,
        examples,
        ["momentum", "spread_bps"],
        {"rows": n, "labelled": n},
    )


def test_phase11_quality_auditor_records_blocking_report(monkeypatch):
    conn = _memory_db(monkeypatch)
    dataset = _dataset(n=8)

    report = DatasetQualityAuditor(min_rows=50, min_labelled=30).audit(dataset, persist=True)

    assert report.status == "fail"
    assert report.blocks_training is True
    row = conn.execute(
        "SELECT * FROM learning_data_quality_reports WHERE report_id = ?",
        (report.report_id,),
    ).fetchone()
    assert row is not None
    assert row["status"] == "fail"
    assert json.loads(row["summary"])["rows"] == 8


def test_phase12_candidate_registry_blocks_non_trainable_risk_knobs(monkeypatch):
    conn = _memory_db(monkeypatch)

    safe = CandidatePolicyRegistry().register(
        ReplayPolicy("candidate_conf_0p80", min_confidence=0.80),
        metrics={"profit_factor": 2.0},
        persist=True,
    )
    unsafe = CandidatePolicyRegistry().register(
        {"policy_id": "candidate_bad", "min_confidence": 0.8, "max_order_usd": 5000},
        persist=True,
    )

    assert safe.status == "candidate"
    assert unsafe.status == "blocked_unsafe_parameters"
    row = conn.execute(
        "SELECT * FROM learning_policy_candidates WHERE candidate_policy_id = ?",
        ("candidate_bad",),
    ).fetchone()
    assert "max_order_usd" in json.loads(row["safety_report"])["unsafe_keys"]


def test_phase13_feature_attribution_ranks_winning_signal_features(monkeypatch):
    conn = _memory_db(monkeypatch)
    dataset = _dataset()

    result = FeatureAttributionAnalyzer().analyze(dataset, candidate_policy_id="candidate_conf_0p80", persist=True)

    assert result.top_features[0]["feature"] == "momentum"
    assert result.top_features[0]["direction"] == "higher_when_winning"
    assert conn.execute("SELECT COUNT(*) AS n FROM learning_feature_attributions").fetchone()["n"] == 1


def test_phase14_drift_monitor_blocks_large_feature_shift(monkeypatch):
    conn = _memory_db(monkeypatch)
    baseline = _dataset("baseline", shift=0.0)
    current = _dataset("current", shift=10.0)

    report = FeatureDriftMonitor(warn_z=0.5, block_z=1.0, max_blocked_feature_ratio=0.1).compare(
        baseline,
        current,
        persist=True,
    )

    assert report.status == "block"
    assert report.blocks_promotion is True
    assert conn.execute("SELECT COUNT(*) AS n FROM learning_drift_reports").fetchone()["n"] == 1


def test_phase15_to_17_and_19_promotion_package_stays_manual(monkeypatch):
    conn = _memory_db(monkeypatch)
    dataset = _dataset()
    backtester = DecisionReplayBacktester(min_trades=1)
    champion = backtester.run(dataset, ReplayPolicy(CHAMPION_POLICY_ID, min_confidence=0.0), persist=True)
    challenger = backtester.run(dataset, ReplayPolicy("candidate_conf_0p80", min_confidence=0.80), persist=True)
    evaluation = ShadowPolicyEvaluator(
        {
            "min_out_of_sample_trades": 1,
            "must_beat_champion_after_fees_bps": -1_000_000,
            "max_drawdown_worse_than_champion_bps": 1_000_000,
        }
    ).evaluate(champion, challenger, persist=True)
    decision = ManualPromotionController().decide(evaluation, manual_approval=False, persist=True)
    shadow = ShadowPeriodPlanner().plan("candidate_conf_0p80", persist=True)
    rollback = RollbackReadinessChecker().check("candidate_conf_0p80", persist=True)

    package = PromotionPackageBuilder().build(
        evaluation=evaluation,
        promotion_decision=decision,
        shadow_plan=shadow,
        rollback_check=rollback,
        persist=True,
    )

    assert package.readiness == "blocked"
    assert "manual_approval" in package.evidence["blockers"]
    assert package.requires_manual_approval is True
    assert conn.execute("SELECT COUNT(*) AS n FROM learning_promotion_packages").fetchone()["n"] == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM learning_operator_reports").fetchone()["n"] == 1


def test_phase18_orchestrator_records_offline_cycle_without_live_mutation(monkeypatch):
    conn = _memory_db(monkeypatch)
    dataset = _dataset()
    orchestrator = ContinuousLearningOrchestrator(
        quality_auditor=DatasetQualityAuditor(min_rows=20, min_labelled=20),
        replay_backtester=DecisionReplayBacktester(min_trades=1),
        shadow_evaluator=ShadowPolicyEvaluator(
            {
                "min_out_of_sample_trades": 1,
                "must_beat_champion_after_fees_bps": -1_000_000,
                "max_drawdown_worse_than_champion_bps": 1_000_000,
            }
        ),
    )

    result = orchestrator.run_offline_cycle(dataset=dataset, manual_approval=False, persist=True)

    assert result.status == "completed"
    assert result.dataset_id == dataset.dataset_id
    assert result.package_id is not None
    assert result.artifacts["promotion_package"]["metadata"]["no_live_config_mutation"] is True
    assert conn.execute("SELECT COUNT(*) AS n FROM learning_scheduler_runs").fetchone()["n"] == 1
