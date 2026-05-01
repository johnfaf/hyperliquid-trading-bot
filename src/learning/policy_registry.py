"""Safety policy registry for continuous learning.

Phase 0 freezes the currently deployed behavior as a champion policy. Learned
models can be evaluated against it, but promotion remains manual and bounded by
non-trainable risk limits.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import config

logger = logging.getLogger(__name__)

CHAMPION_POLICY_ID = "champion_policy_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Dict[str, Any]) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"))


def default_non_trainable_limits() -> Dict[str, Any]:
    """Risk controls that learning code is never allowed to relax."""
    return {
        "live_trading_requires_dual_control": True,
        "dashboard_auth_required": True,
        "kill_switch_persistent": True,
        "max_live_order_usd": float(getattr(config, "LIVE_MAX_ORDER_USD", 0.0) or 0.0),
        "max_live_position_usd": float(getattr(config, "LIVE_MAX_POSITION_SIZE_USD", 0.0) or 0.0),
        "max_daily_loss_usd": float(getattr(config, "LIVE_MAX_DAILY_LOSS_USD", 0.0) or 0.0),
        "max_paper_positions": int(getattr(config, "PAPER_TRADING_MAX_OPEN_TRADES", 0) or 0),
        "max_paper_leverage": float(getattr(config, "PAPER_TRADING_MAX_LEVERAGE", 0.0) or 0.0),
        "min_firewall_confidence": float(getattr(config, "FIREWALL_MIN_CONFIDENCE", 0.0) or 0.0),
        "tp_sl_basis": "roe",
        "reward_to_risk_ratio": float(getattr(config, "PAPER_TRADING_REWARD_TO_RISK_RATIO", 5.0) or 5.0),
    }


def default_promotion_gates() -> Dict[str, Any]:
    return {
        "mode": "manual_approval_only",
        "min_walk_forward_folds": 5,
        "min_out_of_sample_trades": 100,
        "min_shadow_days": 7,
        "must_beat_champion_after_fees_bps": 15,
        "max_drawdown_worse_than_champion_bps": 0,
        "min_calibration_ece_improvement": 0.0,
        "require_no_missing_required_sources": True,
        "require_replay_point_in_time_safe": True,
    }


def default_rollback_rules() -> Dict[str, Any]:
    return {
        "auto_rollback_enabled": True,
        "rollback_to_policy_id": CHAMPION_POLICY_ID,
        "max_live_drawdown_bps": 100,
        "max_daily_loss_bps": 100,
        "max_consecutive_protected_order_failures": 1,
        "max_source_missing_ratio": 0.25,
        "operator_alert_channels": ["telegram", "logs"],
    }


def default_reporting_contract() -> Dict[str, Any]:
    return {
        "sections": [
            "policy_id",
            "training_window",
            "walk_forward_metrics",
            "shadow_vs_champion",
            "source_quality",
            "risk_limit_deltas",
            "promotion_decision",
        ],
        "required_metrics": [
            "win_rate",
            "profit_factor",
            "max_drawdown",
            "brier_score",
            "ece",
            "avg_return_after_fees_bps",
        ],
    }


def ensure_champion_policy(
    policy_id: str = CHAMPION_POLICY_ID,
    *,
    mirror_to_postgres: bool = True,
) -> str:
    """Create or refresh the champion policy definition."""
    from src.data import database as db

    now = _now()
    with db.get_connection(for_read=not mirror_to_postgres) as conn:
        row = conn.execute(
            "SELECT policy_id FROM continuous_learning_policies WHERE policy_id = ?",
            (policy_id,),
        ).fetchone()
        payload = (
            policy_id,
            "champion",
            "manual",
            None,
            now,
            now,
            None,
            _json(default_non_trainable_limits()),
            _json(default_promotion_gates()),
            _json(default_rollback_rules()),
            _json(default_reporting_contract()),
            _json({"live_policy": policy_id}),
            _json({}),
            "Frozen baseline policy for continuous-learning comparisons.",
        )
        if row:
            conn.execute(
                """
                UPDATE continuous_learning_policies
                SET status = ?, mode = ?, parent_policy_id = ?, updated_at = ?,
                    non_trainable_limits = ?, promotion_gates = ?, rollback_rules = ?,
                    reporting_contract = ?, model_versions = ?, notes = ?
                WHERE policy_id = ?
                """,
                (
                    payload[1], payload[2], payload[3], now,
                    payload[7], payload[8], payload[9], payload[10], payload[11],
                    payload[13], policy_id,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO continuous_learning_policies
                (policy_id, status, mode, parent_policy_id, created_at, updated_at,
                 promoted_at, non_trainable_limits, promotion_gates, rollback_rules,
                 reporting_contract, model_versions, metrics, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
    return policy_id


def get_policy(policy_id: str = CHAMPION_POLICY_ID) -> Optional[Dict[str, Any]]:
    from src.data import database as db

    with db.get_connection(for_read=True) as conn:
        row = conn.execute(
            "SELECT * FROM continuous_learning_policies WHERE policy_id = ?",
            (policy_id,),
        ).fetchone()
    if not row:
        return None
    result = dict(row)
    for key in (
        "non_trainable_limits",
        "promotion_gates",
        "rollback_rules",
        "reporting_contract",
        "model_versions",
        "metrics",
    ):
        try:
            result[key] = json.loads(result.get(key) or "{}")
        except Exception:
            result[key] = {}
    return result
