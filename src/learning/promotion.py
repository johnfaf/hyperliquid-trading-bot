"""Phase 10: manual promotion records and rollback-safe gating."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.learning.policy_registry import CHAMPION_POLICY_ID
from src.learning.shadow_evaluator import ShadowEvaluationResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"), default=str)


def _stable_id(prefix: str, payload: Any) -> str:
    raw = _json(payload)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class PromotionDecision:
    decision_id: str
    candidate_policy_id: str
    decision: str
    approved: bool
    requires_manual_approval: bool
    target_policy_id: Optional[str]
    shadow_evaluation_id: Optional[str]
    rollback_policy_id: str
    reason: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class ManualPromotionController:
    """Records promotion intent without mutating live config.

    A challenger can be marked ready only if the shadow gates passed and the
    caller explicitly supplies ``manual_approval=True``. Even then, this class
    only records approval; deployment/cutover remains a separate operator step.
    """

    def decide(
        self,
        evaluation: ShadowEvaluationResult,
        *,
        manual_approval: bool = False,
        requested_by: str = "codex",
        persist: bool = True,
    ) -> PromotionDecision:
        if not evaluation.gates_passed:
            decision = "blocked"
            approved = False
            reason = "shadow gates failed"
        elif not manual_approval:
            decision = "pending_manual_approval"
            approved = False
            reason = "manual approval required"
        else:
            decision = "approved_for_shadow_cutover"
            approved = True
            reason = "shadow gates passed and manual approval supplied"
        result = PromotionDecision(
            decision_id=_stable_id(
                "lpd",
                {
                    "evaluation_id": evaluation.evaluation_id,
                    "decision": decision,
                    "manual_approval": manual_approval,
                },
            ),
            candidate_policy_id=evaluation.challenger_policy_id,
            target_policy_id=evaluation.challenger_policy_id if approved else None,
            decision=decision,
            approved=approved,
            requires_manual_approval=True,
            shadow_evaluation_id=evaluation.evaluation_id,
            rollback_policy_id=CHAMPION_POLICY_ID,
            reason=reason,
            metadata={
                "requested_by": requested_by,
                "no_live_config_mutation": True,
                "rollback_policy_id": CHAMPION_POLICY_ID,
            },
        )
        if persist:
            self.record_result(result, requested_by=requested_by)
        return result

    @staticmethod
    def record_result(result: PromotionDecision, requested_by: str = "codex") -> None:
        from src.data import database as db

        with db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO learning_promotion_decisions
                (decision_id, created_at, candidate_policy_id, target_policy_id,
                 requested_by, decision, approved, requires_manual_approval,
                 shadow_evaluation_id, rollback_policy_id, reason, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(decision_id) DO UPDATE SET
                    created_at = EXCLUDED.created_at,
                    decision = EXCLUDED.decision,
                    approved = EXCLUDED.approved,
                    reason = EXCLUDED.reason,
                    metadata = EXCLUDED.metadata
                """,
                (
                    result.decision_id,
                    _now(),
                    result.candidate_policy_id,
                    result.target_policy_id,
                    requested_by,
                    result.decision,
                    bool(result.approved),
                    bool(result.requires_manual_approval),
                    result.shadow_evaluation_id,
                    result.rollback_policy_id,
                    result.reason,
                    _json(result.metadata),
                ),
            )
