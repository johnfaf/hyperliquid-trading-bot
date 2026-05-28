"""Phase 11: offline data-quality gates for learning datasets."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.learning.dataset_builder import DatasetBuildResult, LearningExample


# ── Env-driven threshold overrides ──────────────────────────────


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    """Return env value clamped to [lo, hi], or default if unset/invalid."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return max(lo, min(hi, float(raw)))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    try:
        return max(lo, min(hi, int(float(raw))))
    except (TypeError, ValueError):
        return int(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


# ── Threshold presets ───────────────────────────────────────────


# Strict / production defaults.  Mirror the legacy hard-coded values
# so existing operators see no behaviour change unless they opt in.
_STRICT_THRESHOLDS = dict(
    min_rows=50,
    min_labelled=30,
    max_missing_feature_ratio=0.15,
    min_positive_ratio=0.20,
    max_positive_ratio=0.80,
    max_data_gap_ratio=0.05,
)


# Research-mode preset.  Used during the cold-start window where
# decision_outcomes is sparse (only paper-closed trades have labels),
# the bot has been losing (positive ratio near 0%), and signal cadence
# is uneven (high data_gap_ratio).  Set
# ``LEARNING_QUALITY_RESEARCH_MODE=1`` to apply.
#
# Looser bars are SAFE because the auditor only gates the offline
# learning loop's progression to building promotion packages -- it
# never touches live behaviour.  Even with a passing dataset, every
# downstream stage (replay backtest, shadow eval, promotion decision)
# applies its own safety gates before a candidate policy would ever
# affect live trading.
_RESEARCH_THRESHOLDS = dict(
    min_rows=50,
    min_labelled=20,
    max_missing_feature_ratio=0.60,
    min_positive_ratio=0.05,
    max_positive_ratio=0.95,
    max_data_gap_ratio=0.80,
)


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
        min_rows: Optional[int] = None,
        min_labelled: Optional[int] = None,
        max_missing_feature_ratio: Optional[float] = None,
        # ★ M21 FIX: was 0.05 / 0.95.  A dataset of 95% wins (or 5%) was
        # passing the balance check, even though such an extreme ratio is
        # strongly suggestive of label leakage, broken close-detection, or
        # a degenerate strategy that's never genuinely losing -- none of
        # which should pass quality gate.  Tightened to 0.20 / 0.80 so
        # we still allow asymmetric edges (e.g. high-PF setups with
        # plenty of small losses) while rejecting suspect distributions.
        # Also tightened max_missing_feature_ratio 0.35 -> 0.15 (mirrors
        # M11 in dataset_builder).
        #
        # ★ PHASE 4 FIX: all six thresholds are now operator-tunable
        # via env so production can run with looser bars while the
        # upstream data shape improves.  Set
        # ``LEARNING_QUALITY_RESEARCH_MODE=1`` to apply the looser
        # research preset in one switch.  See _RESEARCH_THRESHOLDS for
        # the values and the safety rationale.
        min_positive_ratio: Optional[float] = None,
        max_positive_ratio: Optional[float] = None,
        max_data_gap_ratio: Optional[float] = None,
    ):
        preset = (
            _RESEARCH_THRESHOLDS
            if _env_bool("LEARNING_QUALITY_RESEARCH_MODE", default=False)
            else _STRICT_THRESHOLDS
        )
        # Resolution order: explicit kwarg > per-knob env > preset.
        self.min_rows = int(
            min_rows
            if min_rows is not None
            else _env_int(
                "LEARNING_QUALITY_MIN_ROWS", preset["min_rows"], 1, 1_000_000,
            )
        )
        self.min_labelled = int(
            min_labelled
            if min_labelled is not None
            else _env_int(
                "LEARNING_QUALITY_MIN_LABELLED",
                preset["min_labelled"], 1, 1_000_000,
            )
        )
        self.max_missing_feature_ratio = float(
            max_missing_feature_ratio
            if max_missing_feature_ratio is not None
            else _env_float(
                "LEARNING_QUALITY_MAX_MISSING_FEATURE_RATIO",
                preset["max_missing_feature_ratio"], 0.0, 1.0,
            )
        )
        self.min_positive_ratio = float(
            min_positive_ratio
            if min_positive_ratio is not None
            else _env_float(
                "LEARNING_QUALITY_MIN_POSITIVE_RATIO",
                preset["min_positive_ratio"], 0.0, 1.0,
            )
        )
        self.max_positive_ratio = float(
            max_positive_ratio
            if max_positive_ratio is not None
            else _env_float(
                "LEARNING_QUALITY_MAX_POSITIVE_RATIO",
                preset["max_positive_ratio"], 0.0, 1.0,
            )
        )
        self.max_data_gap_ratio = float(
            max_data_gap_ratio
            if max_data_gap_ratio is not None
            else _env_float(
                "LEARNING_QUALITY_MAX_DATA_GAP_RATIO",
                preset["max_data_gap_ratio"], 0.0, 1.0,
            )
        )

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
        data_gap_payloads: List[Dict[str, Any]] = []
        try:
            from src.learning.decision_replay_report import detect_example_data_gaps

            data_gap_payloads = [
                detect_example_data_gaps(item, required_feature_names=list(dataset.feature_names or []))
                for item in examples
            ]
        except Exception:
            data_gap_payloads = []
        data_gap_count = sum(1 for item in data_gap_payloads if item.get("has_gap"))
        data_gap_ratio = data_gap_count / len(examples) if examples else 0.0
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
            "data_gap_ratio": {
                "passed": data_gap_ratio <= self.max_data_gap_ratio,
                "actual": data_gap_ratio,
                "required_max": self.max_data_gap_ratio,
                "gap_count": data_gap_count,
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
            "data_gap_ratio": data_gap_ratio,
            "data_gap_count": data_gap_count,
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
