"""Phase 11: offline data-quality gates for learning datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.learning.dataset_builder import DatasetBuildResult, LearningExample


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True, separators=(",", ":"), default=str)


def _stable_id(prefix: str, payload: Any) -> str:
    raw = _json(payload)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class DataQualityReport:
    report_id: str
    dataset_id: str
    status: str
    checks: Dict[str, Any]
    summary: Dict[str, Any]
    blocks_training: bool
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class DatasetQualityAuditor:
    """Checks whether an offline dataset is safe enough to train/evaluate.

    These gates are intentionally conservative. A failed report never changes
    live behavior; it only blocks the automated learning loop from producing a
    promotion package from weak or biased evidence.
    """

    def __init__(
        self,
        *,
        min_rows: int = 50,
        min_labelled: int = 30,
        max_missing_feature_ratio: float = 0.35,
        min_positive_ratio: float = 0.05,
        max_positive_ratio: float = 0.95,
    ):
        self.min_rows = int(min_rows)
        self.min_labelled = int(min_labelled)
        self.max_missing_feature_ratio = float(max_missing_feature_ratio)
        self.min_positive_ratio = float(min_positive_ratio)
        self.max_positive_ratio = float(max_positive_ratio)

    @staticmethod
    def _feature_missing_ratio(examples: List[LearningExample], feature_names: List[str]) -> float:
        if not examples or not feature_names:
            return 0.0
        total = len(examples) * len(feature_names)
        missing = sum(1 for item in examples for name in feature_names if name not in item.features)
        return missing / total if total else 0.0

    def audit(self, dataset: DatasetBuildResult, *, persist: bool = True) -> DataQualityReport:
        examples = list(dataset.examples or [])
        labelled = [item for item in examples if item.label_win is not None]
        positives = sum(1 for item in labelled if item.label_win == 1)
        positive_ratio = positives / len(labelled) if labelled else 0.0
        missing_ratio = self._feature_missing_ratio(examples, list(dataset.feature_names or []))
        checks = {
            "min_rows": {
                "passed": len(examples) >= self.min_rows,
                "actual": len(examples),
                "required": self.min_rows,
            },
            "min_labelled": {
                "passed": len(labelled) >= self.min_labelled,
                "actual": len(labelled),
                "required": self.min_labelled,
            },
            "missing_feature_ratio": {
                "passed": missing_ratio <= self.max_missing_feature_ratio,
                "actual": missing_ratio,
                "required_max": self.max_missing_feature_ratio,
            },
            "positive_label_balance": {
                "passed": self.min_positive_ratio <= positive_ratio <= self.max_positive_ratio,
                "actual": positive_ratio,
                "required_min": self.min_positive_ratio,
                "required_max": self.max_positive_ratio,
            },
        }
        failed = [name for name, payload in checks.items() if not payload["passed"]]
        status = "pass" if not failed else "fail"
        summary = {
            "rows": len(examples),
            "labelled": len(labelled),
            "positive_labels": positives,
            "positive_ratio": positive_ratio,
            "feature_count": len(dataset.feature_names or []),
            "missing_feature_ratio": missing_ratio,
            "failed_checks": failed,
        }
        report = DataQualityReport(
            report_id=_stable_id(
                "ldq",
                {"dataset_id": dataset.dataset_id, "checks": checks, "summary": summary},
            ),
            dataset_id=dataset.dataset_id,
            status=status,
            checks=checks,
            summary=summary,
            blocks_training=bool(failed),
            metadata={"auditor": type(self).__name__, "offline_only": True},
        )
        if persist:
            self.record_report(report)
        return report

    @staticmethod
    def record_report(report: DataQualityReport) -> None:
        from src.data import database as db

        with db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO learning_data_quality_reports
                (report_id, created_at, dataset_id, status, checks, summary,
                 blocks_training, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(report_id) DO UPDATE SET
                    created_at = EXCLUDED.created_at,
                    status = EXCLUDED.status,
                    checks = EXCLUDED.checks,
                    summary = EXCLUDED.summary,
                    blocks_training = EXCLUDED.blocks_training,
                    metadata = EXCLUDED.metadata
                """,
                (
                    report.report_id,
                    _now(),
                    report.dataset_id,
                    report.status,
                    _json(report.checks),
                    _json(report.summary),
                    bool(report.blocks_training),
                    _json(report.metadata),
                ),
            )


def latest_quality_report(dataset_id: str) -> Optional[Dict[str, Any]]:
    from src.data import database as db

    with db.get_connection(for_read=True) as conn:
        row = conn.execute(
            """
            SELECT * FROM learning_data_quality_reports
            WHERE dataset_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (dataset_id,),
        ).fetchone()
    return dict(row) if row else None
