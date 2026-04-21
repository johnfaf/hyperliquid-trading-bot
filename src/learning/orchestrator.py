"""Phase 18: offline continuous-learning orchestrator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.learning.candidate_registry import CandidatePolicyRegistry
from src.learning.data_quality import DatasetQualityAuditor
from src.learning.dataset_builder import DatasetBuildResult, DecisionDatasetBuilder
from src.learning.feature_attribution import FeatureAttributionAnalyzer
from src.learning.improvement_loop import OfflineImprovementRunner
from src.learning.policy_registry import CHAMPION_POLICY_ID
from src.learning.promotion import ManualPromotionController
from src.learning.promotion_package import (
    PromotionPackage,
    PromotionPackageBuilder,
    RollbackReadinessChecker,
    ShadowPeriodPlanner,
)
from src.learning.replay_backtester import DecisionReplayBacktester, ReplayPolicy
from src.learning.shadow_evaluator import ShadowPolicyEvaluator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True, separators=(",", ":"), default=str)


def _stable_id(prefix: str, payload: Any) -> str:
    raw = _json(payload)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class LearningCycleResult:
    schedule_run_id: str
    status: str
    dataset_id: Optional[str]
    improvement_id: Optional[str]
    package_id: Optional[str]
    metrics: Dict[str, Any]
    errors: List[str]
    artifacts: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class ContinuousLearningOrchestrator:
    """Runs the offline learning loop end-to-end.

    The orchestrator intentionally stops at an operator package. It does not
    write runtime config, env vars, source policies, or live risk limits.
    """

    def __init__(
        self,
        *,
        dataset_builder: Optional[DecisionDatasetBuilder] = None,
        quality_auditor: Optional[DatasetQualityAuditor] = None,
        improvement_runner: Optional[OfflineImprovementRunner] = None,
        replay_backtester: Optional[DecisionReplayBacktester] = None,
        shadow_evaluator: Optional[ShadowPolicyEvaluator] = None,
    ):
        self.dataset_builder = dataset_builder or DecisionDatasetBuilder()
        self.quality_auditor = quality_auditor or DatasetQualityAuditor()
        self.improvement_runner = improvement_runner or OfflineImprovementRunner()
        self.replay_backtester = replay_backtester or DecisionReplayBacktester()
        self.shadow_evaluator = shadow_evaluator or ShadowPolicyEvaluator()

    @staticmethod
    def _policy_from_candidate(candidate_policy_id: str, candidate_metrics: Dict[str, Any]) -> ReplayPolicy:
        params = dict(candidate_metrics.get("parameters") or {})
        return ReplayPolicy(
            policy_id=candidate_policy_id,
            min_confidence=float(params.get("min_confidence", 0.0) or 0.0),
            allowed_sources=params.get("allowed_sources"),
            allowed_sides=params.get("allowed_sides"),
        )

    def run_offline_cycle(
        self,
        *,
        dataset: Optional[DatasetBuildResult] = None,
        limit: int = 5000,
        manual_approval: bool = False,
        persist: bool = True,
    ) -> LearningCycleResult:
        started_at = _now()
        errors: List[str] = []
        artifacts: Dict[str, Any] = {}
        dataset_id: Optional[str] = None
        improvement_id: Optional[str] = None
        package: Optional[PromotionPackage] = None
        status = "completed"
        try:
            dataset = dataset or self.dataset_builder.build(limit=limit, persist=persist)
            dataset_id = dataset.dataset_id
            quality = self.quality_auditor.audit(dataset, persist=persist)
            artifacts["quality_report"] = quality.to_dict()
            if quality.blocks_training:
                status = "blocked_data_quality"
                return self._finalize(
                    started_at=started_at,
                    status=status,
                    dataset_id=dataset_id,
                    improvement_id=None,
                    package=None,
                    metrics={"quality_status": quality.status},
                    errors=errors,
                    artifacts=artifacts,
                    persist=persist,
                )

            improvement = self.improvement_runner.run(dataset=dataset, persist=persist)
            improvement_id = improvement.improvement_id
            artifacts["improvement"] = improvement.to_dict()
            if not improvement.best_candidate_policy_id:
                status = "blocked_no_candidate"
                return self._finalize(
                    started_at=started_at,
                    status=status,
                    dataset_id=dataset_id,
                    improvement_id=improvement_id,
                    package=None,
                    metrics={"next_action": improvement.next_action},
                    errors=errors,
                    artifacts=artifacts,
                    persist=persist,
                )

            candidate_record = CandidatePolicyRegistry().register(
                self._policy_from_candidate(
                    improvement.best_candidate_policy_id,
                    improvement.selected_metrics,
                ),
                source_improvement_id=improvement.improvement_id,
                metrics=improvement.selected_metrics,
                persist=persist,
            )
            artifacts["candidate"] = candidate_record.to_dict()
            if candidate_record.status != "candidate":
                status = "blocked_candidate_safety"
                return self._finalize(
                    started_at=started_at,
                    status=status,
                    dataset_id=dataset_id,
                    improvement_id=improvement_id,
                    package=None,
                    metrics={"candidate_status": candidate_record.status},
                    errors=errors,
                    artifacts=artifacts,
                    persist=persist,
                )

            champion = self.replay_backtester.run(
                dataset,
                ReplayPolicy(CHAMPION_POLICY_ID, min_confidence=0.0),
                persist=persist,
            )
            challenger = self.replay_backtester.run(
                dataset,
                self._policy_from_candidate(
                    improvement.best_candidate_policy_id,
                    improvement.selected_metrics,
                ),
                persist=persist,
            )
            evaluation = self.shadow_evaluator.evaluate(champion, challenger, persist=persist)
            promotion_decision = ManualPromotionController().decide(
                evaluation,
                manual_approval=manual_approval,
                persist=persist,
            )
            attribution = FeatureAttributionAnalyzer().analyze(
                dataset,
                candidate_policy_id=improvement.best_candidate_policy_id,
                persist=persist,
            )
            shadow_plan = ShadowPeriodPlanner().plan(
                improvement.best_candidate_policy_id,
                persist=persist,
            )
            rollback_check = RollbackReadinessChecker().check(
                improvement.best_candidate_policy_id,
                persist=persist,
            )
            package = PromotionPackageBuilder().build(
                evaluation=evaluation,
                promotion_decision=promotion_decision,
                quality_report=quality.to_dict(),
                attribution=attribution.to_dict(),
                shadow_plan=shadow_plan,
                rollback_check=rollback_check,
                persist=persist,
            )
            artifacts.update(
                {
                    "champion_backtest": champion.to_dict(),
                    "challenger_backtest": challenger.to_dict(),
                    "shadow_evaluation": evaluation.to_dict(),
                    "promotion_decision": promotion_decision.to_dict(),
                    "feature_attribution": attribution.to_dict(),
                    "shadow_plan": shadow_plan.to_dict(),
                    "rollback_check": rollback_check.to_dict(),
                    "promotion_package": package.to_dict(),
                }
            )
            status = "completed"
        except Exception as exc:
            status = "failed"
            errors.append(str(exc))

        return self._finalize(
            started_at=started_at,
            status=status,
            dataset_id=dataset_id,
            improvement_id=improvement_id,
            package=package,
            metrics={
                "artifact_count": len(artifacts),
                "manual_approval": manual_approval,
            },
            errors=errors,
            artifacts=artifacts,
            persist=persist,
        )

    @staticmethod
    def _finalize(
        *,
        started_at: str,
        status: str,
        dataset_id: Optional[str],
        improvement_id: Optional[str],
        package: Optional[PromotionPackage],
        metrics: Dict[str, Any],
        errors: List[str],
        artifacts: Dict[str, Any],
        persist: bool,
    ) -> LearningCycleResult:
        finished_at = _now()
        schedule_run_id = _stable_id(
            "lsr",
            {
                "started_at": started_at,
                "status": status,
                "dataset_id": dataset_id,
                "improvement_id": improvement_id,
                "package_id": package.package_id if package else None,
                "errors": errors,
            },
        )
        result = LearningCycleResult(
            schedule_run_id=schedule_run_id,
            status=status,
            dataset_id=dataset_id,
            improvement_id=improvement_id,
            package_id=package.package_id if package else None,
            metrics=metrics,
            errors=errors,
            artifacts=artifacts,
        )
        if persist:
            ContinuousLearningOrchestrator.record_schedule_run(
                result,
                started_at=started_at,
                finished_at=finished_at,
            )
        return result

    @staticmethod
    def record_schedule_run(result: LearningCycleResult, *, started_at: str, finished_at: str) -> None:
        from src.data import database as db

        with db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO learning_scheduler_runs
                (schedule_run_id, created_at, run_type, status, dataset_id,
                 improvement_id, package_id, started_at, finished_at,
                 metrics, errors, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(schedule_run_id) DO UPDATE SET
                    created_at = EXCLUDED.created_at,
                    status = EXCLUDED.status,
                    package_id = EXCLUDED.package_id,
                    finished_at = EXCLUDED.finished_at,
                    metrics = EXCLUDED.metrics,
                    errors = EXCLUDED.errors,
                    metadata = EXCLUDED.metadata
                """,
                (
                    result.schedule_run_id,
                    _now(),
                    "offline_learning_cycle",
                    result.status,
                    result.dataset_id,
                    result.improvement_id,
                    result.package_id,
                    started_at,
                    finished_at,
                    _json(result.metrics),
                    _json(result.errors),
                    _json({"no_live_config_mutation": True}),
                ),
            )
