"""Phase 8: shadow/champion gate evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

from src.learning.policy_registry import CHAMPION_POLICY_ID, default_promotion_gates
from src.learning.replay_backtester import ReplayBacktestResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"), default=str)


def _stable_id(prefix: str, payload: Any) -> str:
    raw = _json(payload)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class ShadowEvaluationResult:
    evaluation_id: str
    champion_policy_id: str
    challenger_policy_id: str
    dataset_id: str
    champion_run_id: str
    challenger_run_id: str
    verdict: str
    gates_passed: bool
    gate_results: Dict[str, Any]
    metrics_delta: Dict[str, Any]
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class ShadowPolicyEvaluator:
    """Compares a challenger backtest to the frozen champion gates."""

    def __init__(self, gates: Dict[str, Any] | None = None):
        self.gates = dict(gates or default_promotion_gates())

    def evaluate(
        self,
        champion: ReplayBacktestResult,
        challenger: ReplayBacktestResult,
        *,
        persist: bool = True,
    ) -> ShadowEvaluationResult:
        min_trades = int(self.gates.get("min_out_of_sample_trades", 100) or 100)
        must_beat_bps = float(self.gates.get("must_beat_champion_after_fees_bps", 0.0) or 0.0)
        max_dd_worse_bps = float(self.gates.get("max_drawdown_worse_than_champion_bps", 0.0) or 0.0)
        pnl_delta = challenger.total_pnl - champion.total_pnl
        avg_delta = challenger.avg_pnl - champion.avg_pnl
        drawdown_delta = challenger.max_drawdown - champion.max_drawdown
        improvement_bps = avg_delta * 10_000.0
        gate_results = {
            "min_out_of_sample_trades": {
                "passed": challenger.trade_count >= min_trades,
                "actual": challenger.trade_count,
                "required": min_trades,
            },
            "must_beat_champion_after_fees_bps": {
                "passed": improvement_bps >= must_beat_bps,
                "actual": improvement_bps,
                "required": must_beat_bps,
            },
            "max_drawdown_worse_than_champion_bps": {
                "passed": (drawdown_delta * 10_000.0) <= max_dd_worse_bps,
                "actual": drawdown_delta * 10_000.0,
                "required": max_dd_worse_bps,
            },
            "challenger_backtest_passed": {
                "passed": bool(challenger.passed),
                "actual": bool(challenger.passed),
                "required": True,
            },
        }
        gates_passed = all(item["passed"] for item in gate_results.values())
        verdict = "eligible_for_shadow" if gates_passed else "blocked"
        result = ShadowEvaluationResult(
            evaluation_id=_stable_id(
                "lse",
                {
                    "champion": champion.run_id,
                    "challenger": challenger.run_id,
                    "gates": gate_results,
                },
            ),
            champion_policy_id=CHAMPION_POLICY_ID,
            challenger_policy_id=challenger.candidate_policy_id,
            dataset_id=challenger.dataset_id,
            champion_run_id=champion.run_id,
            challenger_run_id=challenger.run_id,
            verdict=verdict,
            gates_passed=gates_passed,
            gate_results=gate_results,
            metrics_delta={
                "total_pnl": pnl_delta,
                "avg_pnl": avg_delta,
                "win_rate": challenger.win_rate - champion.win_rate,
                "profit_factor": challenger.profit_factor - champion.profit_factor,
                "max_drawdown": drawdown_delta,
            },
            notes="Manual approval still required before any live promotion.",
        )
        if persist:
            self.record_result(result)
        return result

    @staticmethod
    def record_result(result: ShadowEvaluationResult) -> None:
        from src.data import database as db

        with db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO learning_shadow_evaluations
                (evaluation_id, created_at, champion_policy_id, challenger_policy_id,
                 dataset_id, champion_run_id, challenger_run_id, verdict,
                 gates_passed, gate_results, metrics_delta, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evaluation_id) DO UPDATE SET
                    created_at = EXCLUDED.created_at,
                    verdict = EXCLUDED.verdict,
                    gates_passed = EXCLUDED.gates_passed,
                    gate_results = EXCLUDED.gate_results,
                    metrics_delta = EXCLUDED.metrics_delta,
                    notes = EXCLUDED.notes
                """,
                (
                    result.evaluation_id,
                    _now(),
                    result.champion_policy_id,
                    result.challenger_policy_id,
                    result.dataset_id,
                    result.champion_run_id,
                    result.challenger_run_id,
                    result.verdict,
                    bool(result.gates_passed),
                    _json(result.gate_results),
                    _json(result.metrics_delta),
                    result.notes,
                ),
            )
