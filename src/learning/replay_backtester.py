"""Phase 7: offline replay backtester for decision-journal datasets."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.learning.dataset_builder import DatasetBuildResult, LearningExample
from src.learning.policy_registry import CHAMPION_POLICY_ID


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"), default=str)


def _stable_id(prefix: str, payload: Any) -> str:
    raw = _json(payload)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class ReplayPolicy:
    policy_id: str
    min_confidence: float = 0.0
    allowed_sources: Optional[List[str]] = None
    allowed_sides: Optional[List[str]] = None

    def accepts(self, example: LearningExample) -> bool:
        if example.confidence < self.min_confidence:
            return False
        if self.allowed_sources and example.source not in self.allowed_sources:
            return False
        if self.allowed_sides and example.side not in self.allowed_sides:
            return False
        return example.executed and example.label_win is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "min_confidence": self.min_confidence,
            "allowed_sources": self.allowed_sources,
            "allowed_sides": self.allowed_sides,
        }


@dataclass
class ReplayBacktestResult:
    run_id: str
    dataset_id: str
    policy_id: str
    candidate_policy_id: str
    trade_count: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    max_drawdown: float
    profit_factor: float
    sharpe_like: float
    metrics: Dict[str, Any]
    parameters: Dict[str, Any]
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class DecisionReplayBacktester:
    """Replays policy filters over already-labelled decisions."""

    def __init__(
        self,
        min_trades: int = 20,
        min_profit_factor: float = 1.05,
        train_fraction: float = 0.70,
        min_test_trades: Optional[int] = None,
    ):
        self.min_trades = int(min_trades)
        self.min_profit_factor = float(min_profit_factor)
        self.train_fraction = min(max(float(train_fraction), 0.10), 0.90)
        if min_test_trades is None:
            min_test_trades = max(1, int(math.ceil(self.min_trades * (1.0 - self.train_fraction) - 1e-9)))
        self.min_test_trades = int(min_test_trades)

    @staticmethod
    def _max_drawdown(pnls: List[float]) -> float:
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnls:
            equity += pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
        return max_dd

    @staticmethod
    def _sharpe_like(pnls: List[float]) -> float:
        # ★ H25 FIX: route through canonical helper so every Sharpe
        # number in the codebase shares the same definition (sample
        # stdev, no annualization for per-trade observations).
        from src.analysis.sharpe import sharpe_per_trade
        return sharpe_per_trade(pnls)

    @staticmethod
    def _sort_key(example: LearningExample) -> Tuple[int, Any]:
        value = example.created_at
        if isinstance(value, (int, float)):
            return (0, float(value))
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return (1, parsed.timestamp())
        except Exception:
            return (2, str(value))

    def _split_examples(self, examples: List[LearningExample]) -> Tuple[List[LearningExample], List[LearningExample]]:
        ordered = sorted(examples, key=self._sort_key)
        if len(ordered) < 2:
            return ordered, []
        split_idx = int(math.floor(len(ordered) * self.train_fraction))
        split_idx = min(max(split_idx, 1), len(ordered) - 1)
        return ordered[:split_idx], ordered[split_idx:]

    def _evaluate_policy(
        self,
        examples: List[LearningExample],
        policy: ReplayPolicy,
    ) -> Dict[str, Any]:
        accepted = [item for item in examples if policy.accepts(item)]
        pnls = [float(item.outcome_pnl or 0.0) for item in accepted]
        wins = sum(1 for pnl in pnls if pnl > 0)
        losses = [abs(pnl) for pnl in pnls if pnl < 0]
        gains = [pnl for pnl in pnls if pnl > 0]
        total_pnl = sum(pnls)
        trade_count = len(pnls)
        profit_factor = sum(gains) / sum(losses) if losses and sum(losses) > 0 else (999.0 if gains else 0.0)
        return {
            "wins": wins,
            "losses": sum(1 for pnl in pnls if pnl <= 0),
            "total_gain": sum(gains),
            "total_loss_abs": sum(losses),
            "coverage_ratio": trade_count / len(examples) if examples else 0.0,
            "trade_count": trade_count,
            "win_rate": wins / trade_count if trade_count else 0.0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / trade_count if trade_count else 0.0,
            "max_drawdown": self._max_drawdown(pnls),
            "profit_factor": profit_factor,
            "sharpe_like": self._sharpe_like(pnls),
            "pnls": pnls,
        }

    def _slice_passed(self, metrics: Dict[str, Any], *, min_trades: int) -> bool:
        return (
            int(metrics.get("trade_count", 0) or 0) >= min_trades
            and float(metrics.get("profit_factor", 0.0) or 0.0) >= self.min_profit_factor
            and float(metrics.get("total_pnl", 0.0) or 0.0) > 0
        )

    def run(
        self,
        dataset: DatasetBuildResult,
        policy: ReplayPolicy,
        *,
        champion_policy_id: str = CHAMPION_POLICY_ID,
        persist: bool = True,
    ) -> ReplayBacktestResult:
        train_examples, test_examples = self._split_examples(dataset.examples)
        train_metrics = self._evaluate_policy(train_examples, policy)
        test_metrics = self._evaluate_policy(test_examples, policy)
        all_metrics = self._evaluate_policy(dataset.examples, policy)
        train_passed = self._slice_passed(train_metrics, min_trades=self.min_trades)
        test_passed = self._slice_passed(test_metrics, min_trades=self.min_test_trades)
        metrics = dict(test_metrics)
        metrics.pop("pnls", None)
        metrics.update({
            "train": {k: v for k, v in train_metrics.items() if k != "pnls"},
            "test": {k: v for k, v in test_metrics.items() if k != "pnls"},
            "all": {k: v for k, v in all_metrics.items() if k != "pnls"},
            "split": {
                "train_examples": len(train_examples),
                "test_examples": len(test_examples),
                "train_fraction": self.train_fraction,
                "min_train_trades": self.min_trades,
                "min_test_trades": self.min_test_trades,
                "train_passed": train_passed,
                "test_passed": test_passed,
            },
        })
        passed = train_passed and test_passed
        pnls = list(test_metrics["pnls"])
        result = ReplayBacktestResult(
            run_id=_stable_id(
                "lbt",
                {
                    "dataset_id": dataset.dataset_id,
                    "policy": policy.to_dict(),
                    "train_pnls": train_metrics["pnls"],
                    "test_pnls": pnls,
                    "train_fraction": self.train_fraction,
                },
            ),
            dataset_id=dataset.dataset_id,
            policy_id=champion_policy_id,
            candidate_policy_id=policy.policy_id,
            trade_count=int(test_metrics["trade_count"]),
            win_rate=float(test_metrics["win_rate"]),
            total_pnl=float(test_metrics["total_pnl"]),
            avg_pnl=float(test_metrics["avg_pnl"]),
            max_drawdown=float(test_metrics["max_drawdown"]),
            profit_factor=float(test_metrics["profit_factor"]),
            sharpe_like=float(test_metrics["sharpe_like"]),
            metrics=metrics,
            parameters=policy.to_dict(),
            passed=passed,
        )
        if persist:
            self.record_result(result)
        return result

    @staticmethod
    def record_result(result: ReplayBacktestResult) -> None:
        from src.data import database as db

        with db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO learning_backtest_runs
                (run_id, created_at, dataset_id, policy_id, candidate_policy_id,
                 backtest_type, trade_count, win_rate, total_pnl, avg_pnl,
                 max_drawdown, profit_factor, sharpe_like, metrics, parameters, passed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    created_at = EXCLUDED.created_at,
                    trade_count = EXCLUDED.trade_count,
                    win_rate = EXCLUDED.win_rate,
                    total_pnl = EXCLUDED.total_pnl,
                    avg_pnl = EXCLUDED.avg_pnl,
                    max_drawdown = EXCLUDED.max_drawdown,
                    profit_factor = EXCLUDED.profit_factor,
                    sharpe_like = EXCLUDED.sharpe_like,
                    metrics = EXCLUDED.metrics,
                    parameters = EXCLUDED.parameters,
                    passed = EXCLUDED.passed
                """,
                (
                    result.run_id,
                    _now(),
                    result.dataset_id,
                    result.policy_id,
                    result.candidate_policy_id,
                    "decision_replay",
                    result.trade_count,
                    result.win_rate,
                    result.total_pnl,
                    result.avg_pnl,
                    result.max_drawdown,
                    result.profit_factor,
                    result.sharpe_like,
                    _json(result.metrics),
                    _json(result.parameters),
                    bool(result.passed),
                ),
            )


def sweep_confidence_thresholds(
    dataset: DatasetBuildResult,
    thresholds: Iterable[float],
    *,
    source: Optional[str] = None,
    persist: bool = True,
) -> List[ReplayBacktestResult]:
    runner = DecisionReplayBacktester()
    results = []
    for threshold in thresholds:
        allowed_sources = [source] if source else None
        policy = ReplayPolicy(
            policy_id=f"candidate_conf_{threshold:.2f}".replace(".", "p"),
            min_confidence=float(threshold),
            allowed_sources=allowed_sources,
        )
        results.append(runner.run(dataset, policy, persist=persist))
    return results
