"""Phase 13: lightweight feature attribution for decision datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
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
class FeatureAttributionResult:
    attribution_id: str
    dataset_id: str
    candidate_policy_id: Optional[str]
    method: str
    feature_scores: List[Dict[str, Any]]
    top_features: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class FeatureAttributionAnalyzer:
    """Ranks features by simple win/loss mean delta.

    This avoids heavy ML dependencies while still giving the operator useful
    visibility into which feature columns separate winning from losing paper
    outcomes in a point-in-time dataset.
    """

    method = "win_loss_mean_delta"

    @staticmethod
    def _values(examples: List[LearningExample], feature: str) -> List[float]:
        values = []
        for item in examples:
            if feature not in item.features:
                continue
            try:
                values.append(float(item.features[feature]))
            except (TypeError, ValueError):
                continue
        return values

    def analyze(
        self,
        dataset: DatasetBuildResult,
        *,
        candidate_policy_id: Optional[str] = None,
        top_n: int = 10,
        persist: bool = True,
    ) -> FeatureAttributionResult:
        labelled = [item for item in dataset.examples if item.label_win is not None]
        winners = [item for item in labelled if item.label_win == 1]
        losers = [item for item in labelled if item.label_win == 0]
        scores: List[Dict[str, Any]] = []
        for feature in sorted(dataset.feature_names or []):
            win_values = self._values(winners, feature)
            loss_values = self._values(losers, feature)
            if not win_values or not loss_values:
                continue
            win_mean = mean(win_values)
            loss_mean = mean(loss_values)
            delta = win_mean - loss_mean
            pooled_abs = max(abs(win_mean) + abs(loss_mean), 1e-12)
            score = abs(delta) / pooled_abs
            scores.append(
                {
                    "feature": feature,
                    "score": score,
                    "direction": "higher_when_winning" if delta >= 0 else "lower_when_winning",
                    "win_mean": win_mean,
                    "loss_mean": loss_mean,
                    "delta": delta,
                    "coverage": (len(win_values) + len(loss_values)) / len(labelled) if labelled else 0.0,
                }
            )
        scores.sort(key=lambda item: (item["score"], item["coverage"]), reverse=True)
        top_features = scores[: max(0, int(top_n))]
        result = FeatureAttributionResult(
            attribution_id=_stable_id(
                "lfa",
                {
                    "dataset_id": dataset.dataset_id,
                    "candidate_policy_id": candidate_policy_id,
                    "top": top_features,
                },
            ),
            dataset_id=dataset.dataset_id,
            candidate_policy_id=candidate_policy_id,
            method=self.method,
            feature_scores=scores,
            top_features=top_features,
            metadata={
                "labelled": len(labelled),
                "winners": len(winners),
                "losers": len(losers),
                "offline_only": True,
            },
        )
        if persist:
            self.record_result(result)
        return result

    @staticmethod
    def record_result(result: FeatureAttributionResult) -> None:
        from src.data import database as db

        with db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO learning_feature_attributions
                (attribution_id, created_at, dataset_id, candidate_policy_id,
                 method, feature_scores, top_features, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(attribution_id) DO UPDATE SET
                    created_at = EXCLUDED.created_at,
                    feature_scores = EXCLUDED.feature_scores,
                    top_features = EXCLUDED.top_features,
                    metadata = EXCLUDED.metadata
                """,
                (
                    result.attribution_id,
                    _now(),
                    result.dataset_id,
                    result.candidate_policy_id,
                    result.method,
                    _json(result.feature_scores),
                    _json(result.top_features),
                    _json(result.metadata),
                ),
            )
