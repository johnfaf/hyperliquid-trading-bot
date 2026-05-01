"""Phase 14: dataset drift monitoring for continuous learning."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Any, Dict, List

from src.learning.dataset_builder import DatasetBuildResult, LearningExample


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True, separators=(",", ":"), default=str)


def _stable_id(prefix: str, payload: Any) -> str:
    raw = _json(payload)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class DriftReport:
    drift_id: str
    baseline_dataset_id: str
    current_dataset_id: str
    status: str
    feature_drift: List[Dict[str, Any]]
    summary: Dict[str, Any]
    blocks_promotion: bool
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class FeatureDriftMonitor:
    """Compares two datasets using standardized mean shift per feature."""

    def __init__(self, *, warn_z: float = 1.5, block_z: float = 3.0, max_blocked_feature_ratio: float = 0.25):
        self.warn_z = float(warn_z)
        self.block_z = float(block_z)
        self.max_blocked_feature_ratio = float(max_blocked_feature_ratio)

    @staticmethod
    def _feature_values(examples: List[LearningExample], feature: str) -> List[float]:
        values = []
        for item in examples:
            try:
                values.append(float(item.features[feature]))
            except (KeyError, TypeError, ValueError):
                continue
        return values

    @staticmethod
    def _standardized_shift(baseline: List[float], current: List[float]) -> float:
        if not baseline or not current:
            return 0.0
        b_mean = mean(baseline)
        c_mean = mean(current)
        pooled = math.sqrt((pstdev(baseline) ** 2 + pstdev(current) ** 2) / 2.0)
        if pooled <= 1e-12:
            return 0.0 if abs(c_mean - b_mean) <= 1e-12 else 1e9
        return (c_mean - b_mean) / pooled

    def compare(
        self,
        baseline: DatasetBuildResult,
        current: DatasetBuildResult,
        *,
        persist: bool = True,
    ) -> DriftReport:
        features = sorted(set(baseline.feature_names or []) & set(current.feature_names or []))
        drift_rows: List[Dict[str, Any]] = []
        for feature in features:
            base_values = self._feature_values(baseline.examples, feature)
            current_values = self._feature_values(current.examples, feature)
            shift = self._standardized_shift(base_values, current_values)
            abs_shift = abs(shift) if math.isfinite(shift) else float("inf")
            severity = "ok"
            if abs_shift >= self.block_z:
                severity = "block"
            elif abs_shift >= self.warn_z:
                severity = "warn"
            drift_rows.append(
                {
                    "feature": feature,
                    "standardized_mean_shift": shift,
                    "abs_shift": abs_shift,
                    "severity": severity,
                    "baseline_count": len(base_values),
                    "current_count": len(current_values),
                }
            )
        drift_rows.sort(key=lambda item: item["abs_shift"], reverse=True)
        blocked = [row for row in drift_rows if row["severity"] == "block"]
        warned = [row for row in drift_rows if row["severity"] == "warn"]
        blocked_ratio = len(blocked) / len(drift_rows) if drift_rows else 0.0
        blocks_promotion = bool(blocked and blocked_ratio >= self.max_blocked_feature_ratio)
        status = "block" if blocks_promotion else ("warn" if warned or blocked else "pass")
        summary = {
            "feature_count": len(drift_rows),
            "warn_features": len(warned),
            "block_features": len(blocked),
            "blocked_feature_ratio": blocked_ratio,
            "max_abs_shift": drift_rows[0]["abs_shift"] if drift_rows else 0.0,
        }
        result = DriftReport(
            drift_id=_stable_id(
                "ldr",
                {
                    "baseline": baseline.dataset_id,
                    "current": current.dataset_id,
                    "summary": summary,
                },
            ),
            baseline_dataset_id=baseline.dataset_id,
            current_dataset_id=current.dataset_id,
            status=status,
            feature_drift=drift_rows,
            summary=summary,
            blocks_promotion=blocks_promotion,
            metadata={"method": "standardized_mean_shift", "offline_only": True},
        )
        if persist:
            self.record_report(result)
        return result

    @staticmethod
    def record_report(result: DriftReport) -> None:
        from src.data import database as db

        with db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO learning_drift_reports
                (drift_id, created_at, baseline_dataset_id, current_dataset_id,
                 status, feature_drift, summary, blocks_promotion, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(drift_id) DO UPDATE SET
                    created_at = EXCLUDED.created_at,
                    status = EXCLUDED.status,
                    feature_drift = EXCLUDED.feature_drift,
                    summary = EXCLUDED.summary,
                    blocks_promotion = EXCLUDED.blocks_promotion,
                    metadata = EXCLUDED.metadata
                """,
                (
                    result.drift_id,
                    _now(),
                    result.baseline_dataset_id,
                    result.current_dataset_id,
                    result.status,
                    _json(result.feature_drift),
                    _json(result.summary),
                    bool(result.blocks_promotion),
                    _json(result.metadata),
                ),
            )
