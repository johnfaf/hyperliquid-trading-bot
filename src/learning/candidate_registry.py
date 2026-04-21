"""Phase 12: safe candidate-policy registry for offline learning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.learning.policy_registry import (
    CHAMPION_POLICY_ID,
    default_non_trainable_limits,
)
from src.learning.replay_backtester import ReplayPolicy


TRAINABLE_POLICY_KEYS = {
    "min_confidence",
    "allowed_sources",
    "allowed_sides",
    "source_weights",
    "source_confidence_offsets",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True, separators=(",", ":"), default=str)


def _stable_id(prefix: str, payload: Any) -> str:
    raw = _json(payload)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class CandidatePolicyRecord:
    candidate_policy_id: str
    parent_policy_id: str
    source_improvement_id: Optional[str]
    status: str
    trainable_parameters: Dict[str, Any]
    non_trainable_limits_snapshot: Dict[str, Any]
    metrics: Dict[str, Any]
    safety_report: Dict[str, Any]
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class CandidatePolicyRegistry:
    """Registers challenger policies without touching live config.

    Only trainable knobs are accepted. Anything that tries to change leverage,
    max order size, kill switches, TP/SL basis, or other non-trainable risk
    controls is rejected and persisted as a blocked safety report.
    """

    def __init__(self, parent_policy_id: str = CHAMPION_POLICY_ID):
        self.parent_policy_id = parent_policy_id

    @staticmethod
    def _policy_to_parameters(policy: ReplayPolicy | Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(policy, ReplayPolicy):
            return policy.to_dict()
        return dict(policy or {})

    @staticmethod
    def _safety_report(parameters: Dict[str, Any]) -> Dict[str, Any]:
        keys = set(parameters)
        unsafe_keys = sorted(keys - TRAINABLE_POLICY_KEYS - {"policy_id"})
        return {
            "safe": not unsafe_keys,
            "allowed_trainable_keys": sorted(TRAINABLE_POLICY_KEYS),
            "unsafe_keys": unsafe_keys,
            "live_config_mutation": False,
        }

    def register(
        self,
        policy: ReplayPolicy | Dict[str, Any],
        *,
        source_improvement_id: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        persist: bool = True,
    ) -> CandidatePolicyRecord:
        parameters = self._policy_to_parameters(policy)
        safety = self._safety_report(parameters)
        candidate_policy_id = str(parameters.get("policy_id") or "") or _stable_id(
            "candidate",
            {"parent": self.parent_policy_id, "params": parameters},
        )
        status = "candidate" if safety["safe"] else "blocked_unsafe_parameters"
        record = CandidatePolicyRecord(
            candidate_policy_id=candidate_policy_id,
            parent_policy_id=self.parent_policy_id,
            source_improvement_id=source_improvement_id,
            status=status,
            trainable_parameters={
                key: value for key, value in parameters.items() if key in TRAINABLE_POLICY_KEYS
            },
            non_trainable_limits_snapshot=default_non_trainable_limits(),
            metrics=dict(metrics or {}),
            safety_report=safety,
            notes="Offline candidate; live promotion requires manual approval.",
        )
        if persist:
            self.record_candidate(record)
        return record

    @staticmethod
    def record_candidate(record: CandidatePolicyRecord) -> None:
        from src.data import database as db

        with db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO learning_policy_candidates
                (candidate_policy_id, created_at, parent_policy_id,
                 source_improvement_id, status, trainable_parameters,
                 non_trainable_limits_snapshot, metrics, safety_report, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_policy_id) DO UPDATE SET
                    created_at = EXCLUDED.created_at,
                    source_improvement_id = EXCLUDED.source_improvement_id,
                    status = EXCLUDED.status,
                    trainable_parameters = EXCLUDED.trainable_parameters,
                    non_trainable_limits_snapshot = EXCLUDED.non_trainable_limits_snapshot,
                    metrics = EXCLUDED.metrics,
                    safety_report = EXCLUDED.safety_report,
                    notes = EXCLUDED.notes
                """,
                (
                    record.candidate_policy_id,
                    _now(),
                    record.parent_policy_id,
                    record.source_improvement_id,
                    record.status,
                    _json(record.trainable_parameters),
                    _json(record.non_trainable_limits_snapshot),
                    _json(record.metrics),
                    _json(record.safety_report),
                    record.notes,
                ),
            )


def get_candidate(candidate_policy_id: str) -> Optional[Dict[str, Any]]:
    from src.data import database as db

    with db.get_connection(for_read=True) as conn:
        row = conn.execute(
            "SELECT * FROM learning_policy_candidates WHERE candidate_policy_id = ?",
            (candidate_policy_id,),
        ).fetchone()
    return dict(row) if row else None
