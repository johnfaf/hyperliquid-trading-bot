"""Phase 15-17/19: shadow planning, rollback checks, and operator package."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.learning.policy_registry import CHAMPION_POLICY_ID, default_rollback_rules
from src.learning.promotion import PromotionDecision
from src.learning.shadow_evaluator import ShadowEvaluationResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True, separators=(",", ":"), default=str)


def _stable_id(prefix: str, payload: Any) -> str:
    raw = _json(payload)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class ShadowPeriodPlan:
    shadow_id: str
    candidate_policy_id: str
    champion_policy_id: str
    min_shadow_days: int
    status: str
    metrics: Dict[str, Any]
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RollbackCheckResult:
    rollback_check_id: str
    candidate_policy_id: str
    rollback_policy_id: str
    status: str
    checks: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class PromotionPackage:
    package_id: str
    candidate_policy_id: str
    dataset_id: Optional[str]
    shadow_evaluation_id: Optional[str]
    promotion_decision_id: Optional[str]
    readiness: str
    evidence: Dict[str, Any]
    operator_summary: str
    requires_manual_approval: bool
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class ShadowPeriodPlanner:
    """Records a required shadow period before any live promotion."""

    def plan(
        self,
        candidate_policy_id: str,
        *,
        champion_policy_id: str = CHAMPION_POLICY_ID,
        min_shadow_days: int = 7,
        persist: bool = True,
    ) -> ShadowPeriodPlan:
        result = ShadowPeriodPlan(
            shadow_id=_stable_id(
                "lsp",
                {
                    "candidate": candidate_policy_id,
                    "champion": champion_policy_id,
                    "min_shadow_days": min_shadow_days,
                },
            ),
            candidate_policy_id=candidate_policy_id,
            champion_policy_id=champion_policy_id,
            min_shadow_days=int(min_shadow_days),
            status="planned",
            metrics={"required_before_live_promotion": True},
            notes="Shadow mode only; no live config mutation.",
        )
        if persist:
            self.record_plan(result)
        return result

    @staticmethod
    def record_plan(result: ShadowPeriodPlan) -> None:
        from src.data import database as db

        with db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO learning_shadow_periods
                (shadow_id, created_at, candidate_policy_id, champion_policy_id,
                 min_shadow_days, status, metrics, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(shadow_id) DO UPDATE SET
                    created_at = EXCLUDED.created_at,
                    status = EXCLUDED.status,
                    metrics = EXCLUDED.metrics,
                    notes = EXCLUDED.notes
                """,
                (
                    result.shadow_id,
                    _now(),
                    result.candidate_policy_id,
                    result.champion_policy_id,
                    result.min_shadow_days,
                    result.status,
                    _json(result.metrics),
                    result.notes,
                ),
            )


class RollbackReadinessChecker:
    """Verifies rollback metadata exists before a promotion package is usable."""

    def check(
        self,
        candidate_policy_id: str,
        *,
        rollback_policy_id: str = CHAMPION_POLICY_ID,
        persist: bool = True,
    ) -> RollbackCheckResult:
        rules = default_rollback_rules()
        checks = {
            "rollback_policy_known": {
                "passed": bool(rollback_policy_id),
                "actual": rollback_policy_id,
                "required": "non-empty rollback policy id",
            },
            "auto_rollback_rules_present": {
                "passed": bool(rules.get("auto_rollback_enabled")),
                "actual": rules.get("auto_rollback_enabled"),
                "required": True,
            },
            "operator_alert_channels_present": {
                "passed": bool(rules.get("operator_alert_channels")),
                "actual": rules.get("operator_alert_channels"),
                "required": "at least one alert channel",
            },
        }
        status = "pass" if all(item["passed"] for item in checks.values()) else "fail"
        result = RollbackCheckResult(
            rollback_check_id=_stable_id(
                "lrc",
                {"candidate": candidate_policy_id, "rollback": rollback_policy_id, "checks": checks},
            ),
            candidate_policy_id=candidate_policy_id,
            rollback_policy_id=rollback_policy_id,
            status=status,
            checks=checks,
            metadata={"rollback_rules": rules, "offline_only": True},
        )
        if persist:
            self.record_result(result)
        return result

    @staticmethod
    def record_result(result: RollbackCheckResult) -> None:
        from src.data import database as db

        with db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO learning_rollback_checks
                (rollback_check_id, created_at, candidate_policy_id,
                 rollback_policy_id, status, checks, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(rollback_check_id) DO UPDATE SET
                    created_at = EXCLUDED.created_at,
                    status = EXCLUDED.status,
                    checks = EXCLUDED.checks,
                    metadata = EXCLUDED.metadata
                """,
                (
                    result.rollback_check_id,
                    _now(),
                    result.candidate_policy_id,
                    result.rollback_policy_id,
                    result.status,
                    _json(result.checks),
                    _json(result.metadata),
                ),
            )


class PromotionPackageBuilder:
    """Builds an operator-facing evidence package; never promotes live config."""

    def build(
        self,
        *,
        evaluation: ShadowEvaluationResult,
        promotion_decision: PromotionDecision,
        quality_report: Optional[Dict[str, Any]] = None,
        attribution: Optional[Dict[str, Any]] = None,
        drift_report: Optional[Dict[str, Any]] = None,
        shadow_plan: Optional[ShadowPeriodPlan] = None,
        rollback_check: Optional[RollbackCheckResult] = None,
        persist: bool = True,
    ) -> PromotionPackage:
        blockers = []
        if quality_report and quality_report.get("blocks_training"):
            blockers.append("data_quality")
        if drift_report and drift_report.get("blocks_promotion"):
            blockers.append("feature_drift")
        if rollback_check and rollback_check.status != "pass":
            blockers.append("rollback_readiness")
        if not evaluation.gates_passed:
            blockers.append("shadow_gates")
        if not promotion_decision.approved:
            blockers.append("manual_approval")
        readiness = "ready_for_operator_cutover" if not blockers else "blocked"
        evidence = {
            "shadow_evaluation": evaluation.to_dict(),
            "promotion_decision": promotion_decision.to_dict(),
            "quality_report": quality_report or {},
            "feature_attribution": attribution or {},
            "drift_report": drift_report or {},
            "shadow_plan": shadow_plan.to_dict() if shadow_plan else {},
            "rollback_check": rollback_check.to_dict() if rollback_check else {},
            "blockers": blockers,
            "live_config_mutated": False,
        }
        operator_summary = (
            f"Candidate {promotion_decision.candidate_policy_id}: {readiness}. "
            f"Manual approval required before any live cutover. "
            f"Blockers: {', '.join(blockers) if blockers else 'none'}."
        )
        result = PromotionPackage(
            package_id=_stable_id(
                "lpp",
                {
                    "candidate": promotion_decision.candidate_policy_id,
                    "evaluation": evaluation.evaluation_id,
                    "decision": promotion_decision.decision_id,
                    "readiness": readiness,
                },
            ),
            candidate_policy_id=promotion_decision.candidate_policy_id,
            dataset_id=evaluation.dataset_id,
            shadow_evaluation_id=evaluation.evaluation_id,
            promotion_decision_id=promotion_decision.decision_id,
            readiness=readiness,
            evidence=evidence,
            operator_summary=operator_summary,
            requires_manual_approval=True,
            metadata={"offline_only": True, "no_live_config_mutation": True},
        )
        if persist:
            self.record_package(result)
            self.record_operator_report(result)
        return result

    @staticmethod
    def record_package(result: PromotionPackage) -> None:
        from src.data import database as db

        with db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO learning_promotion_packages
                (package_id, created_at, candidate_policy_id, dataset_id,
                 shadow_evaluation_id, promotion_decision_id, readiness,
                 evidence, operator_summary, requires_manual_approval, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(package_id) DO UPDATE SET
                    created_at = EXCLUDED.created_at,
                    readiness = EXCLUDED.readiness,
                    evidence = EXCLUDED.evidence,
                    operator_summary = EXCLUDED.operator_summary,
                    metadata = EXCLUDED.metadata
                """,
                (
                    result.package_id,
                    _now(),
                    result.candidate_policy_id,
                    result.dataset_id,
                    result.shadow_evaluation_id,
                    result.promotion_decision_id,
                    result.readiness,
                    _json(result.evidence),
                    result.operator_summary,
                    bool(result.requires_manual_approval),
                    _json(result.metadata),
                ),
            )

    @staticmethod
    def record_operator_report(result: PromotionPackage) -> None:
        from src.data import database as db

        report_id = _stable_id("lor", {"package": result.package_id, "summary": result.operator_summary})
        with db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO learning_operator_reports
                (report_id, created_at, package_id, report_type, title, body, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(report_id) DO UPDATE SET
                    created_at = EXCLUDED.created_at,
                    body = EXCLUDED.body,
                    metadata = EXCLUDED.metadata
                """,
                (
                    report_id,
                    _now(),
                    result.package_id,
                    "promotion_readiness",
                    f"Promotion package {result.package_id}",
                    result.operator_summary,
                    _json({"candidate_policy_id": result.candidate_policy_id}),
                ),
            )
