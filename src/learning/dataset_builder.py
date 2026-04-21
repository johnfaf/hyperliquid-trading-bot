"""Phase 6: build supervised learning datasets from decision snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from src.learning.policy_registry import CHAMPION_POLICY_ID


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"), default=str)


def _loads(value: Any, fallback: Any = None) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value or "{}")
    except Exception:
        return fallback if fallback is not None else {}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _stable_id(prefix: str, payload: Any) -> str:
    raw = _json(payload)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class LearningExample:
    decision_id: str
    coin: str
    side: str
    source: str
    created_at: str
    features: Dict[str, float]
    confidence: float
    executed: bool
    label_win: Optional[int]
    outcome_pnl: float
    paper_trade_id: Optional[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "coin": self.coin,
            "side": self.side,
            "source": self.source,
            "created_at": self.created_at,
            "features": self.features,
            "confidence": self.confidence,
            "executed": self.executed,
            "label_win": self.label_win,
            "outcome_pnl": self.outcome_pnl,
            "paper_trade_id": self.paper_trade_id,
            "metadata": self.metadata,
        }


@dataclass
class DatasetBuildResult:
    dataset_id: str
    examples: List[LearningExample]
    feature_names: List[str]
    quality_report: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "row_count": len(self.examples),
            "feature_names": self.feature_names,
            "quality_report": self.quality_report,
            "examples": [item.to_dict() for item in self.examples],
        }


class DecisionDatasetBuilder:
    """Builds labels from journaled decisions and paper outcomes.

    This is deliberately offline-only: it reads completed/paper decisions and
    stores dataset manifests. It does not alter live policies or sizing.
    """

    LABEL_DEFINITION = {
        "label_win": "1 when linked paper trade has pnl > 0, 0 when pnl <= 0",
        "executed": "decision_snapshots.paper_trade_id is present",
        "outcome_pnl": "paper_trades.pnl after fees/funding when available",
    }

    def __init__(self, source_policy_id: str = CHAMPION_POLICY_ID):
        self.source_policy_id = source_policy_id

    def _load_rows(self, limit: int = 5000, min_created_at: Optional[str] = None) -> List[Dict[str, Any]]:
        from src.data import database as db

        where = "WHERE ds.final_status IN ('paper_opened', 'approved', 'rejected', 'firewall_prescreen_rejected', 'firewall_prescreen_approved')"
        params: List[Any] = []
        if min_created_at:
            where += " AND ds.created_at >= ?"
            params.append(min_created_at)
        params.append(int(limit))
        with db.get_connection(for_read=True) as conn:
            rows = conn.execute(
                f"""
                SELECT ds.*, pt.pnl AS paper_pnl, pt.status AS paper_status,
                       pt.closed_at AS paper_closed_at, pt.exit_price AS paper_exit_price
                FROM decision_snapshots ds
                LEFT JOIN paper_trades pt ON pt.id = ds.paper_trade_id
                {where}
                ORDER BY ds.created_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _row_to_example(row: Dict[str, Any]) -> LearningExample:
        features = _loads(row.get("features"), {}) or {}
        features = {str(k): _float(v) for k, v in dict(features).items()}
        pnl = _float(row.get("paper_pnl"), 0.0)
        executed = row.get("paper_trade_id") is not None
        label_win: Optional[int] = None
        if executed and row.get("paper_pnl") is not None:
            label_win = 1 if pnl > 0 else 0
        return LearningExample(
            decision_id=str(row.get("decision_id") or ""),
            coin=str(row.get("coin") or ""),
            side=str(row.get("side") or ""),
            source=str(row.get("source") or ""),
            created_at=str(row.get("created_at") or ""),
            features=features,
            confidence=_float(row.get("calibrated_confidence", row.get("raw_confidence", 0.0))),
            executed=bool(executed),
            label_win=label_win,
            outcome_pnl=pnl,
            paper_trade_id=int(row["paper_trade_id"]) if row.get("paper_trade_id") is not None else None,
            metadata={
                "final_status": row.get("final_status"),
                "firewall_decision": row.get("firewall_decision"),
                "strategy_type": row.get("strategy_type"),
                "source_key": row.get("source_key"),
            },
        )

    @staticmethod
    def _quality(examples: List[LearningExample], feature_names: List[str]) -> Dict[str, Any]:
        rows = len(examples)
        executed = sum(1 for item in examples if item.executed)
        labelled = [item for item in examples if item.label_win is not None]
        positives = sum(1 for item in labelled if item.label_win == 1)
        missing_feature_ratio = 0.0
        if rows and feature_names:
            total = rows * len(feature_names)
            missing = sum(1 for item in examples for name in feature_names if name not in item.features)
            missing_feature_ratio = missing / total if total else 0.0
        return {
            "rows": rows,
            "executed": executed,
            "labelled": len(labelled),
            "positive_labels": positives,
            "win_rate": positives / len(labelled) if labelled else 0.0,
            "missing_feature_ratio": missing_feature_ratio,
            "ready_for_training": len(labelled) >= 50 and missing_feature_ratio <= 0.35,
        }

    def build(self, limit: int = 5000, min_created_at: Optional[str] = None, persist: bool = True) -> DatasetBuildResult:
        rows = self._load_rows(limit=limit, min_created_at=min_created_at)
        examples = [self._row_to_example(row) for row in rows]
        feature_names = sorted({name for item in examples for name in item.features})
        quality = self._quality(examples, feature_names)
        dataset_id = _stable_id(
            "lds",
            {
                "source_policy_id": self.source_policy_id,
                "count": len(examples),
                "first": examples[-1].decision_id if examples else "",
                "last": examples[0].decision_id if examples else "",
                "feature_names": feature_names,
            },
        )
        result = DatasetBuildResult(dataset_id, examples, feature_names, quality)
        if persist:
            self.record_manifest(result)
        return result

    def record_manifest(self, result: DatasetBuildResult) -> None:
        from src.data import database as db

        examples = result.examples
        start_ts = examples[-1].created_at if examples else None
        end_ts = examples[0].created_at if examples else None
        positives = sum(1 for item in examples if item.label_win == 1)
        executed = sum(1 for item in examples if item.executed)
        with db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO learning_datasets
                (dataset_id, created_at, source_policy_id, start_ts, end_ts,
                 row_count, positive_labels, executed_count, feature_names,
                 label_definition, quality_report, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                    created_at = EXCLUDED.created_at,
                    row_count = EXCLUDED.row_count,
                    positive_labels = EXCLUDED.positive_labels,
                    executed_count = EXCLUDED.executed_count,
                    feature_names = EXCLUDED.feature_names,
                    quality_report = EXCLUDED.quality_report,
                    metadata = EXCLUDED.metadata
                """,
                (
                    result.dataset_id,
                    _now(),
                    self.source_policy_id,
                    start_ts,
                    end_ts,
                    len(examples),
                    positives,
                    executed,
                    _json(result.feature_names),
                    _json(self.LABEL_DEFINITION),
                    _json(result.quality_report),
                    _json({"builder": type(self).__name__}),
                ),
            )


def examples_to_dicts(examples: Iterable[LearningExample]) -> List[Dict[str, Any]]:
    return [item.to_dict() for item in examples]
