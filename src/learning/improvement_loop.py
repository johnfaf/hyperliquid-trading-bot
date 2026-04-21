"""Phase 9: offline automated improvement cycle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from src.learning.dataset_builder import DecisionDatasetBuilder, DatasetBuildResult
from src.learning.replay_backtester import DecisionReplayBacktester, ReplayBacktestResult, ReplayPolicy


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"), default=str)


def _stable_id(prefix: str, payload: Any) -> str:
    raw = _json(payload)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class ImprovementRunResult:
    improvement_id: str
    dataset_id: str
    best_candidate_policy_id: Optional[str]
    candidate_results: List[Dict[str, Any]]
    selected_metrics: Dict[str, Any]
    next_action: str
    status: str = "completed"

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class OfflineImprovementRunner:
    """Searches safe policy knobs offline and records the best challenger.

    The first implementation only searches confidence thresholds. That improves
    selectivity without changing SL/TP, leverage, position size, kill-switch, or
    any non-trainable safety limit.
    """

    def __init__(self, thresholds: Optional[Iterable[float]] = None):
        self.thresholds = list(
            thresholds if thresholds is not None else [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        )

    @staticmethod
    def _score(result: ReplayBacktestResult) -> float:
        if result.trade_count <= 0:
            return float("-inf")
        return (
            result.total_pnl
            + result.profit_factor * 0.25
            + result.win_rate * 0.25
            - result.max_drawdown * 0.50
        )

    def run(
        self,
        *,
        dataset: Optional[DatasetBuildResult] = None,
        limit: int = 5000,
        persist: bool = True,
    ) -> ImprovementRunResult:
        if dataset is None:
            dataset = DecisionDatasetBuilder().build(limit=limit, persist=persist)
        backtester = DecisionReplayBacktester()
        candidates: List[ReplayBacktestResult] = []
        for threshold in self.thresholds:
            policy = ReplayPolicy(
                policy_id=f"candidate_conf_{float(threshold):.2f}".replace(".", "p"),
                min_confidence=float(threshold),
            )
            candidates.append(backtester.run(dataset, policy, persist=persist))
        eligible = [item for item in candidates if item.passed]
        best = max(eligible or candidates, key=self._score, default=None)
        next_action = "shadow_evaluation" if best and best.passed else "collect_more_data"
        result = ImprovementRunResult(
            improvement_id=_stable_id(
                "lir",
                {
                    "dataset_id": dataset.dataset_id,
                    "thresholds": self.thresholds,
                    "best": best.run_id if best else "",
                },
            ),
            dataset_id=dataset.dataset_id,
            best_candidate_policy_id=best.candidate_policy_id if best else None,
            candidate_results=[item.to_dict() for item in candidates],
            selected_metrics=best.to_dict() if best else {},
            next_action=next_action,
        )
        if persist:
            self.record_result(result, search_space={"thresholds": self.thresholds})
        return result

    @staticmethod
    def record_result(result: ImprovementRunResult, search_space: Dict[str, Any]) -> None:
        from src.data import database as db

        with db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO learning_improvement_runs
                (improvement_id, created_at, dataset_id, best_candidate_policy_id,
                 mode, status, search_space, candidate_results, selected_metrics,
                 next_action, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(improvement_id) DO UPDATE SET
                    created_at = EXCLUDED.created_at,
                    best_candidate_policy_id = EXCLUDED.best_candidate_policy_id,
                    status = EXCLUDED.status,
                    candidate_results = EXCLUDED.candidate_results,
                    selected_metrics = EXCLUDED.selected_metrics,
                    next_action = EXCLUDED.next_action,
                    metadata = EXCLUDED.metadata
                """,
                (
                    result.improvement_id,
                    _now(),
                    result.dataset_id,
                    result.best_candidate_policy_id,
                    "offline",
                    result.status,
                    _json(search_space),
                    _json(result.candidate_results),
                    _json(result.selected_metrics),
                    result.next_action,
                    _json({"safety": "no_live_mutation"}),
                ),
            )
