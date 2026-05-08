"""Phase 7: offline replay backtester for decision-journal datasets."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.learning.dataset_builder import (
    DatasetBuildResult,
    DecisionDatasetBuilder,
    LearningExample,
)
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
    include_rejected: bool = False
    allowed_source_keys: Optional[List[str]] = None
    allowed_statuses: Optional[List[str]] = None
    allowed_regimes: Optional[List[str]] = None
    blocked_rejection_reasons: Optional[List[str]] = None
    source_confidence_multipliers: Optional[Dict[str, float]] = None
    side_confidence_multipliers: Optional[Dict[str, float]] = None

    def effective_confidence(self, example: LearningExample) -> float:
        confidence = float(example.confidence or 0.0)
        source_key = example.source_key or example.metadata.get("source_key") or example.source
        if self.source_confidence_multipliers and source_key in self.source_confidence_multipliers:
            confidence *= float(self.source_confidence_multipliers[source_key])
        if self.side_confidence_multipliers and example.side in self.side_confidence_multipliers:
            confidence *= float(self.side_confidence_multipliers[example.side])
        return max(0.0, min(confidence, 1.0))

    def accepts(self, example: LearningExample) -> bool:
        if example.label_win is None:
            return False
        if not example.executed and not self.include_rejected:
            return False
        if self.effective_confidence(example) < self.min_confidence:
            return False
        if self.allowed_sources and example.source not in self.allowed_sources:
            return False
        source_key = example.source_key or example.metadata.get("source_key") or ""
        if self.allowed_source_keys and source_key not in self.allowed_source_keys:
            return False
        if self.allowed_sides and example.side not in self.allowed_sides:
            return False
        status = example.final_status or str(example.metadata.get("final_status") or "")
        if self.allowed_statuses and status not in self.allowed_statuses:
            return False
        regime = ""
        if isinstance(example.regime, dict):
            regime = str(example.regime.get("overall_regime") or example.regime.get("regime") or "")
        if self.allowed_regimes and regime not in self.allowed_regimes:
            return False
        reason = example.rejection_reason or str(example.metadata.get("rejection_reason") or "")
        if self.blocked_rejection_reasons and reason in self.blocked_rejection_reasons:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "min_confidence": self.min_confidence,
            "allowed_sources": self.allowed_sources,
            "allowed_sides": self.allowed_sides,
            "include_rejected": self.include_rejected,
            "allowed_source_keys": self.allowed_source_keys,
            "allowed_statuses": self.allowed_statuses,
            "allowed_regimes": self.allowed_regimes,
            "blocked_rejection_reasons": self.blocked_rejection_reasons,
            "source_confidence_multipliers": self.source_confidence_multipliers,
            "side_confidence_multipliers": self.side_confidence_multipliers,
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
        purge_fraction: float = 0.02,
    ):
        self.min_trades = int(min_trades)
        self.min_profit_factor = float(min_profit_factor)
        self.train_fraction = min(max(float(train_fraction), 0.10), 0.90)
        self.purge_fraction = min(max(float(purge_fraction), 0.0), 0.20)
        if min_test_trades is None:
            min_test_trades = max(1, int(math.ceil(self.min_trades * (1.0 - self.train_fraction) - 1e-9)))
        self.min_test_trades = int(min_test_trades)
        self._last_purged_examples = 0

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
            self._last_purged_examples = 0
            return ordered, []
        split_idx = int(math.floor(len(ordered) * self.train_fraction))
        split_idx = min(max(split_idx, 1), len(ordered) - 1)
        purge = int(math.floor(len(ordered) * self.purge_fraction)) if len(ordered) >= 50 else 0
        test_start = min(split_idx + purge, len(ordered) - 1)
        self._last_purged_examples = max(0, test_start - split_idx)
        return ordered[:split_idx], ordered[test_start:]

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

    def _walkforward_metrics(
        self,
        examples: List[LearningExample],
        policy: ReplayPolicy,
        *,
        windows: int = 4,
    ) -> List[Dict[str, Any]]:
        ordered = sorted(examples, key=self._sort_key)
        if len(ordered) < max(20, windows * 4):
            return []
        out: List[Dict[str, Any]] = []
        window_count = min(max(int(windows), 1), 8)
        step = max(1, len(ordered) // window_count)
        for idx in range(window_count):
            start = idx * step
            end = len(ordered) if idx == window_count - 1 else min(len(ordered), (idx + 1) * step)
            chunk = ordered[start:end]
            if len(chunk) < 2:
                continue
            train, test = self._split_examples(chunk)
            metrics = self._evaluate_policy(test, policy)
            out.append(
                {
                    "window": idx + 1,
                    "examples": len(chunk),
                    "train_examples": len(train),
                    "test_examples": len(test),
                    "purged_examples": self._last_purged_examples,
                    **{k: v for k, v in metrics.items() if k != "pnls"},
                    "passed": self._slice_passed(metrics, min_trades=max(1, self.min_test_trades)),
                }
            )
        return out

    @staticmethod
    def _candidate_score(metrics: Dict[str, Any]) -> float:
        trades = max(float(metrics.get("trade_count", 0) or 0), 1.0)
        avg_pnl = float(metrics.get("avg_pnl", 0.0) or 0.0)
        max_dd_per_trade = float(metrics.get("max_drawdown", 0.0) or 0.0) / trades
        profit_factor = min(float(metrics.get("profit_factor", 0.0) or 0.0), 10.0)
        win_rate = float(metrics.get("win_rate", 0.0) or 0.0)
        coverage = float(metrics.get("coverage_ratio", 0.0) or 0.0)
        return avg_pnl - (0.50 * max_dd_per_trade) + (0.10 * profit_factor) + (0.05 * win_rate) + (0.02 * coverage)

    def _threshold_grid(self, policy: ReplayPolicy) -> List[float]:
        base = float(policy.min_confidence or 0.0)
        raw = [base, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        return sorted({round(min(max(value, 0.0), 0.99), 4) for value in raw})

    def _select_policy_on_train(
        self,
        train: List[LearningExample],
        base_policy: ReplayPolicy,
    ) -> Tuple[ReplayPolicy, Dict[str, Any]]:
        best_policy = base_policy
        best_metrics: Optional[Dict[str, Any]] = None
        best_score = -1_000_000_000_000.0
        for threshold in self._threshold_grid(base_policy):
            candidate = replace(base_policy, min_confidence=threshold)
            metrics = self._evaluate_policy(train, candidate)
            if int(metrics.get("trade_count", 0) or 0) < max(1, self.min_trades):
                score = -1_000_000_000_000.0
            else:
                score = self._candidate_score(metrics)
            if best_metrics is None or score > best_score:
                best_policy = candidate
                best_metrics = metrics
                best_score = score
        best_metrics = best_metrics or self._evaluate_policy(train, best_policy)
        return best_policy, {k: v for k, v in best_metrics.items() if k != "pnls"} | {"score": best_score}

    def _walkforward_trained_metrics(
        self,
        examples: List[LearningExample],
        policy: ReplayPolicy,
        *,
        windows: int = 4,
    ) -> List[Dict[str, Any]]:
        """Expanding-window walk-forward validation.

        Each window trains only the confidence threshold on past decisions, then
        evaluates the selected threshold on the next future slice.
        """
        ordered = sorted(examples, key=self._sort_key)
        if len(ordered) < max(20, self.min_trades + self.min_test_trades + 2):
            return []
        window_count = min(max(int(windows), 1), 8)
        test_size = max(self.min_test_trades, len(ordered) // (window_count + 2), 1)
        initial_train = max(self.min_trades, int(math.floor(len(ordered) * self.train_fraction)) - test_size)
        initial_train = min(max(initial_train, 1), len(ordered) - 1)
        out: List[Dict[str, Any]] = []
        for idx in range(window_count):
            train_end = initial_train + idx * test_size
            test_end = min(len(ordered), train_end + test_size)
            if train_end >= len(ordered) or test_end <= train_end:
                break
            train = ordered[:train_end]
            test = ordered[train_end:test_end]
            if len(test) < 1:
                continue
            selected_policy, train_metrics = self._select_policy_on_train(train, policy)
            test_metrics = self._evaluate_policy(test, selected_policy)
            out.append(
                {
                    "window": idx + 1,
                    "train_examples": len(train),
                    "test_examples": len(test),
                    "selected_policy": selected_policy.to_dict(),
                    "selected_min_confidence": selected_policy.min_confidence,
                    "train": train_metrics,
                    "test": {k: v for k, v in test_metrics.items() if k != "pnls"},
                    "passed": self._slice_passed(test_metrics, min_trades=max(1, self.min_test_trades)),
                }
            )
        return out

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
        decision_report: Dict[str, Any]
        try:
            from src.learning.decision_replay_report import build_decision_replay_report

            decision_report = build_decision_replay_report(dataset, policy)
        except Exception as exc:
            decision_report = {
                "data_quality": {"passed": False},
                "error": f"{type(exc).__name__}: {exc}",
            }
        data_quality_passed = bool(decision_report.get("data_quality", {}).get("passed", False))
        metrics.update({
            "train": {k: v for k, v in train_metrics.items() if k != "pnls"},
            "test": {k: v for k, v in test_metrics.items() if k != "pnls"},
            "all": {k: v for k, v in all_metrics.items() if k != "pnls"},
            "split": {
                "train_examples": len(train_examples),
                "test_examples": len(test_examples),
                "purged_examples": self._last_purged_examples,
                "train_fraction": self.train_fraction,
                "min_train_trades": self.min_trades,
                "min_test_trades": self.min_test_trades,
                "train_passed": train_passed,
                "test_passed": test_passed,
                "data_quality_passed": data_quality_passed,
            },
            "walk_forward": self._walkforward_metrics(dataset.examples, policy),
            "walk_forward_trained": self._walkforward_trained_metrics(dataset.examples, policy),
            "decision_replay_report": decision_report,
        })
        passed = train_passed and test_passed and data_quality_passed
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

    def run_date_range(
        self,
        *,
        start: Optional[str],
        end: Optional[str],
        policy: ReplayPolicy,
        limit: int = 5000,
        persist: bool = True,
        use_outcomes: bool = True,
    ) -> ReplayBacktestResult:
        """Build and replay the exact decisions made inside a timestamp window."""
        dataset = DecisionDatasetBuilder().build(
            limit=limit,
            min_created_at=start,
            max_created_at=end,
            persist=persist,
            use_outcomes=use_outcomes,
        )
        result = self.run(dataset, policy, persist=False)
        result.metrics.setdefault("date_range", {"start": start, "end": end})
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
