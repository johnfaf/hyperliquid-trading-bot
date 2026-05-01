"""Conservative empirical calibration from decision outcomes.

The calibrator produces offline recommendations only. It records source/side
confidence multipliers for operator review and candidate-policy packaging, but
does not mutate live configuration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from src.learning.dataset_builder import DatasetBuildResult, LearningExample


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True, separators=(",", ":"), default=str)


def _stable_id(prefix: str, payload: Any) -> str:
    raw = _json(payload)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def _mean(values: Iterable[float]) -> float:
    data = [float(v) for v in values]
    return sum(data) / len(data) if data else 0.0


@dataclass
class DecisionCalibratorResult:
    calibrator_id: str
    dataset_id: str
    source_stats: Dict[str, Any]
    global_stats: Dict[str, Any]
    gates: Dict[str, Any]
    feature_names: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class ConservativeDecisionCalibrator:
    """Fit cautious confidence multipliers from labelled decision outcomes."""

    def __init__(
        self,
        *,
        min_group_examples: int = 20,
        derisk_win_rate: float = 0.45,
        promote_win_rate: float = 0.55,
        max_boost: float = 1.15,
        min_multiplier: float = 0.50,
    ):
        self.min_group_examples = int(min_group_examples)
        self.derisk_win_rate = float(derisk_win_rate)
        self.promote_win_rate = float(promote_win_rate)
        self.max_boost = float(max_boost)
        self.min_multiplier = float(min_multiplier)

    @staticmethod
    def _group_key(example: LearningExample) -> str:
        source_key = example.source_key or example.source or "unknown"
        side = example.side or "unknown"
        return f"{source_key}|{side}"

    def _stats_for_group(self, examples: List[LearningExample]) -> Dict[str, Any]:
        labelled = [item for item in examples if item.label_win is not None]
        wins = sum(1 for item in labelled if item.label_win == 1)
        losses = len(labelled) - wins
        avg_pnl = _mean(item.outcome_pnl for item in labelled)
        avg_return = _mean(item.outcome_return_pct for item in labelled)
        win_rate = wins / len(labelled) if labelled else 0.0
        if len(labelled) < self.min_group_examples:
            multiplier = 0.75
            action = "collect_more_data"
        elif win_rate < self.derisk_win_rate or avg_return < 0 or avg_pnl < 0:
            multiplier = self.min_multiplier
            action = "derisk"
        elif win_rate >= self.promote_win_rate and avg_return > 0 and avg_pnl > 0:
            multiplier = min(self.max_boost, 1.0 + (win_rate - self.promote_win_rate))
            action = "eligible_small_boost"
        else:
            multiplier = 1.0
            action = "hold"
        return {
            "count": len(labelled),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "avg_return_pct": avg_return,
            "confidence_multiplier": multiplier,
            "action": action,
            "executed_count": sum(1 for item in labelled if item.executed),
            "rejected_labelled_count": sum(1 for item in labelled if not item.executed),
        }

    def fit(self, dataset: DatasetBuildResult, *, persist: bool = True) -> DecisionCalibratorResult:
        labelled = [item for item in dataset.examples if item.label_win is not None]
        groups: Dict[str, List[LearningExample]] = {}
        for example in labelled:
            groups.setdefault(self._group_key(example), []).append(example)

        source_stats = {
            key: self._stats_for_group(items)
            for key, items in sorted(groups.items())
        }
        global_stats = self._stats_for_group(labelled)
        gates = {
            "min_group_examples": self.min_group_examples,
            "derisk_win_rate": self.derisk_win_rate,
            "promote_win_rate": self.promote_win_rate,
            "safe_output_only": True,
        }
        result = DecisionCalibratorResult(
            calibrator_id=_stable_id(
                "ldc",
                {
                    "dataset_id": dataset.dataset_id,
                    "source_stats": source_stats,
                    "global_stats": global_stats,
                    "gates": gates,
                },
            ),
            dataset_id=dataset.dataset_id,
            source_stats=source_stats,
            global_stats=global_stats,
            gates=gates,
            feature_names=list(dataset.feature_names),
            metadata={
                "model_type": "conservative_empirical",
                "no_live_config_mutation": True,
            },
        )
        if persist:
            self.record_result(result)
        return result

    @staticmethod
    def record_result(result: DecisionCalibratorResult) -> None:
        from src.data import database as db

        with db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO learning_decision_calibrators
                (calibrator_id, created_at, dataset_id, model_type, feature_names,
                 source_stats, global_stats, gates, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(calibrator_id) DO UPDATE SET
                    created_at = EXCLUDED.created_at,
                    source_stats = EXCLUDED.source_stats,
                    global_stats = EXCLUDED.global_stats,
                    gates = EXCLUDED.gates,
                    metadata = EXCLUDED.metadata
                """,
                (
                    result.calibrator_id,
                    _now(),
                    result.dataset_id,
                    "conservative_empirical",
                    _json(result.feature_names),
                    _json(result.source_stats),
                    _json(result.global_stats),
                    _json(result.gates),
                    _json(result.metadata),
                ),
            )
