"""
Experiment-discipline tooling for replayable benchmark packs.

Replays stored decision-research cycles through the current decision engine
and challenger configurations, then reports out-of-sample promotion gates.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime
from typing import Dict, List, Optional

import config
from src.data import database as db
from src.signals.decision_engine import DecisionEngine


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def build_decision_engine_config(overrides: Optional[Dict] = None) -> Dict:
    cfg = {
        "w_score": config.DECISION_W_SCORE,
        "w_regime": config.DECISION_W_REGIME,
        "w_diversity": config.DECISION_W_DIVERSITY,
        "w_freshness": config.DECISION_W_FRESHNESS,
        "w_consensus": config.DECISION_W_CONSENSUS,
        "w_confidence": config.DECISION_W_CONFIDENCE,
        "w_source_quality": config.DECISION_W_SOURCE_QUALITY,
        "w_confirmation": config.DECISION_W_CONFIRMATION,
        "w_expected_value": config.DECISION_W_EXPECTED_VALUE,
        "min_decision_score": config.DECISION_MIN_SCORE,
        "min_signal_confidence": config.DECISION_MIN_CONFIDENCE,
        "min_source_weight": config.DECISION_MIN_SOURCE_WEIGHT,
        "min_expected_value_pct": config.DECISION_MIN_EXPECTED_VALUE_PCT,
        "max_trades_per_cycle": config.DECISION_MAX_TRADES_PER_CYCLE,
        "maker_fee_bps": config.DECISION_MAKER_FEE_BPS,
        "taker_fee_bps": config.DECISION_TAKER_FEE_BPS,
        "expected_slippage_bps": config.DECISION_EXPECTED_SLIPPAGE_BPS,
        "churn_penalty_bps": config.DECISION_CHURN_PENALTY_BPS,
        "default_execution_role": config.DECISION_DEFAULT_EXECUTION_ROLE,
        "persist_research": False,
        "execution_quality_enabled": config.DECISION_EXECUTION_QUALITY_ENABLED,
        "execution_quality_lookback_hours": config.DECISION_EXECUTION_QUALITY_LOOKBACK_HOURS,
        "execution_quality_min_events": config.DECISION_EXECUTION_QUALITY_MIN_EVENTS,
        "execution_rejection_penalty_bps": config.DECISION_EXECUTION_REJECTION_PENALTY_BPS,
        "execution_fill_gap_penalty_bps": config.DECISION_EXECUTION_FILL_GAP_PENALTY_BPS,
        "execution_protective_failure_penalty_bps": (
            config.DECISION_EXECUTION_PROTECTIVE_FAILURE_PENALTY_BPS
        ),
        "adaptive_learning_enabled": False,
    }
    if overrides:
        cfg.update(overrides)
    return cfg


def build_challenger_profiles() -> Dict[str, Dict]:
    current_max_trades = max(1, int(config.DECISION_MAX_TRADES_PER_CYCLE))
    return {
        "baseline_current": {},
        "challenger_selective": {
            "min_decision_score": round(min(0.95, config.DECISION_MIN_SCORE + 0.03), 4),
            "min_signal_confidence": round(min(0.95, config.DECISION_MIN_CONFIDENCE + 0.04), 4),
            "min_expected_value_pct": round(config.DECISION_MIN_EXPECTED_VALUE_PCT + 0.0007, 4),
            "max_trades_per_cycle": max(1, current_max_trades - 1),
        },
        "challenger_execution_strict": {
            "w_expected_value": round(min(0.45, config.DECISION_W_EXPECTED_VALUE + 0.06), 4),
            "w_score": round(max(0.05, config.DECISION_W_SCORE - 0.04), 4),
            "min_source_weight": round(min(0.95, config.DECISION_MIN_SOURCE_WEIGHT + 0.03), 4),
            "execution_rejection_penalty_bps": (
                config.DECISION_EXECUTION_REJECTION_PENALTY_BPS + 6.0
            ),
            "execution_fill_gap_penalty_bps": (
                config.DECISION_EXECUTION_FILL_GAP_PENALTY_BPS + 4.0
            ),
            "execution_protective_failure_penalty_bps": (
                config.DECISION_EXECUTION_PROTECTIVE_FAILURE_PENALTY_BPS + 6.0
            ),
        },
        "challenger_diversified": {
            "w_diversity": round(min(0.25, config.DECISION_W_DIVERSITY + 0.05), 4),
            "w_confirmation": round(min(0.20, config.DECISION_W_CONFIRMATION + 0.03), 4),
            "w_consensus": round(max(0.0, config.DECISION_W_CONSENSUS - 0.01), 4),
        },
    }


def _deepcopy_json(payload):
    return json.loads(json.dumps(payload, default=str))


def _replay_strategies_from_cycle(cycle: Dict) -> List[Dict]:
    strategies: List[Dict] = []
    for candidate in cycle.get("candidates", []) or []:
        raw = candidate.get("raw_candidate")
        if isinstance(raw, dict) and raw:
            strategies.append(_deepcopy_json(raw))
            continue

        metadata = {
            "source": candidate.get("source"),
            "source_key": candidate.get("source_key"),
        }
        strategies.append(
            {
                "name": candidate.get("name"),
                "source": candidate.get("source"),
                "source_key": candidate.get("source_key"),
                "strategy_type": candidate.get("strategy_type"),
                "confidence": candidate.get("confidence", 0.0),
                "current_score": candidate.get("composite_score", 0.0),
                "parameters": {"coins": [candidate.get("coin", "unknown")]},
                "metadata": metadata,
            }
        )
    return strategies


def _replay_cycle(engine: DecisionEngine, cycle: Dict) -> Dict:
    context = cycle.get("context", {}) if isinstance(cycle.get("context"), dict) else {}
    strategies = _replay_strategies_from_cycle(cycle)
    regime_data = context.get("regime_data", {}) if isinstance(context.get("regime_data"), dict) else {}
    open_positions = context.get("open_positions", []) if isinstance(context.get("open_positions"), list) else []
    kelly_stats = context.get("kelly_summary", {}) if isinstance(context.get("kelly_summary"), dict) else {}

    open_coins = {
        str(position.get("coin", "") or "").upper()
        for position in open_positions
        if position.get("coin")
    }
    available_slots = max(0, 8 - len(open_positions))

    scored: List[Dict] = []
    for strategy in strategies:
        candidate = copy.deepcopy(strategy)
        composite = engine._compute_composite_score(
            candidate,
            regime_data,
            open_coins,
            kelly_stats,
        )
        scored.append(
            {
                **candidate,
                "_composite_score": composite["total"],
                "_score_breakdown": composite,
            }
        )

    scored.sort(key=lambda item: item["_composite_score"], reverse=True)

    qualified: List[Dict] = []
    disqualified: List[Dict] = []
    for strategy in scored:
        blockers = engine._decision_blockers(strategy)
        strategy["_decision_blockers"] = blockers
        if blockers or strategy["_composite_score"] < engine.min_decision_score:
            disqualified.append(strategy)
        else:
            qualified.append(strategy)

    max_this_cycle = min(engine.max_trades_per_cycle, available_slots)
    selected = qualified[:max_this_cycle]
    overflow = qualified[max_this_cycle:]

    blocker_mix: Dict[str, int] = {}
    source_mix: Dict[str, int] = {}
    selected_candidates = []
    selected_ev_values: List[float] = []
    selected_cost_values: List[float] = []
    selected_score_values: List[float] = []

    for strategy in disqualified:
        blockers = strategy.get("_decision_blockers", []) or ["unknown"]
        for blocker in blockers:
            label = str(blocker or "unknown")
            blocker_mix[label] = blocker_mix.get(label, 0) + 1

    for strategy in selected:
        source_key = str(
            strategy.get("source_key")
            or strategy.get("source")
            or strategy.get("strategy_type")
            or "unknown"
        )
        source_mix[source_key] = source_mix.get(source_key, 0) + 1
        selected_ev_values.append(float(strategy.get("_expected_value_pct", 0.0) or 0.0))
        selected_cost_values.append(float(strategy.get("_execution_cost_pct", 0.0) or 0.0))
        selected_score_values.append(float(strategy.get("_composite_score", 0.0) or 0.0))
        selected_candidates.append(
            {
                "name": strategy.get("name"),
                "coin": strategy.get("_decision_coin"),
                "side": strategy.get("_decision_side"),
                "source": strategy.get("source"),
                "source_key": strategy.get("source_key"),
                "expected_value_pct": strategy.get("_expected_value_pct", 0.0),
                "execution_cost_pct": strategy.get("_execution_cost_pct", 0.0),
                "composite_score": strategy.get("_composite_score", 0.0),
            }
        )

    return {
        "candidate_count": len(scored),
        "selected_count": len(selected),
        "blocked_count": len(disqualified),
        "overflow_count": len(overflow),
        "ranked_count": max(0, len(scored) - len(disqualified) - len(selected) - len(overflow)),
        "no_trade": len(selected) == 0,
        "selected_ev_values": selected_ev_values,
        "selected_cost_values": selected_cost_values,
        "selected_score_values": selected_score_values,
        "blocker_mix": blocker_mix,
        "source_mix": source_mix,
        "selected_candidates": selected_candidates,
    }


def _summarize_partition(cycles: List[Dict], engine_config: Dict) -> Dict:
    engine = DecisionEngine(engine_config)
    aggregate = {
        "cycles": len(cycles),
        "candidate_count": 0,
        "selected_count": 0,
        "blocked_count": 0,
        "overflow_count": 0,
        "ranked_count": 0,
        "no_trade_cycles": 0,
        "_ev_values": [],
        "_cost_values": [],
        "_score_values": [],
        "blocker_mix": {},
        "source_mix": {},
    }

    for cycle in cycles:
        replayed = _replay_cycle(engine, cycle)
        aggregate["candidate_count"] += replayed["candidate_count"]
        aggregate["selected_count"] += replayed["selected_count"]
        aggregate["blocked_count"] += replayed["blocked_count"]
        aggregate["overflow_count"] += replayed["overflow_count"]
        aggregate["ranked_count"] += replayed["ranked_count"]
        if replayed["no_trade"]:
            aggregate["no_trade_cycles"] += 1
        aggregate["_ev_values"].extend(replayed["selected_ev_values"])
        aggregate["_cost_values"].extend(replayed["selected_cost_values"])
        aggregate["_score_values"].extend(replayed["selected_score_values"])
        for key, count in replayed["blocker_mix"].items():
            aggregate["blocker_mix"][key] = aggregate["blocker_mix"].get(key, 0) + count
        for key, count in replayed["source_mix"].items():
            aggregate["source_mix"][key] = aggregate["source_mix"].get(key, 0) + count

    ev_values = aggregate.pop("_ev_values")
    cost_values = aggregate.pop("_cost_values")
    score_values = aggregate.pop("_score_values")
    aggregate["selection_rate"] = round(
        aggregate["selected_count"] / aggregate["candidate_count"],
        4,
    ) if aggregate["candidate_count"] else 0.0
    aggregate["no_trade_rate"] = round(
        aggregate["no_trade_cycles"] / aggregate["cycles"],
        4,
    ) if aggregate["cycles"] else 0.0
    aggregate["avg_selected_ev_pct"] = round(sum(ev_values) / len(ev_values), 4) if ev_values else 0.0
    aggregate["avg_selected_execution_cost_pct"] = (
        round(sum(cost_values) / len(cost_values), 4) if cost_values else 0.0
    )
    aggregate["avg_selected_score"] = round(sum(score_values) / len(score_values), 4) if score_values else 0.0
    aggregate["source_mix"] = dict(
        sorted(aggregate["source_mix"].items(), key=lambda item: (-item[1], item[0]))[:10]
    )
    aggregate["blocker_mix"] = dict(
        sorted(aggregate["blocker_mix"].items(), key=lambda item: (-item[1], item[0]))[:10]
    )
    return aggregate


def _split_cycles(cycles: List[Dict], out_of_sample_ratio: float) -> Dict[str, List[Dict]]:
    if len(cycles) <= 1:
        return {"in_sample": list(cycles), "out_of_sample": []}
    ratio = _clamp(float(out_of_sample_ratio or 0.3), 0.1, 0.8)
    split_idx = int(round(len(cycles) * (1.0 - ratio)))
    split_idx = max(1, min(len(cycles) - 1, split_idx))
    return {
        "in_sample": cycles[:split_idx],
        "out_of_sample": cycles[split_idx:],
    }


def _assess_promotion(
    baseline_metrics: Dict,
    challenger_metrics: Dict,
    *,
    min_oos_cycles: int,
    min_ev_delta_pct: float,
) -> Dict:
    reasons: List[str] = []
    if challenger_metrics.get("cycles", 0) < min_oos_cycles:
        reasons.append(f"oos_cycles<{min_oos_cycles}")
    if challenger_metrics.get("selected_count", 0) <= 0:
        reasons.append("no_oos_selections")

    ev_delta = float(challenger_metrics.get("avg_selected_ev_pct", 0.0) or 0.0) - float(
        baseline_metrics.get("avg_selected_ev_pct", 0.0) or 0.0
    )
    cost_delta = float(challenger_metrics.get("avg_selected_execution_cost_pct", 0.0) or 0.0) - float(
        baseline_metrics.get("avg_selected_execution_cost_pct", 0.0) or 0.0
    )
    no_trade_delta = float(challenger_metrics.get("no_trade_rate", 0.0) or 0.0) - float(
        baseline_metrics.get("no_trade_rate", 0.0) or 0.0
    )

    if ev_delta < min_ev_delta_pct:
        reasons.append(f"ev_delta<{min_ev_delta_pct:.4f}")
    if cost_delta > 0.0010:
        reasons.append("higher_execution_cost")
    if no_trade_delta > 0.15:
        reasons.append("higher_no_trade_rate")

    return {
        "approved": len(reasons) == 0,
        "ev_delta_pct": round(ev_delta, 4),
        "execution_cost_delta_pct": round(cost_delta, 4),
        "no_trade_delta": round(no_trade_delta, 4),
        "reasons": reasons,
    }


def build_experiment_benchmark_pack(
    cycles: List[Dict],
    *,
    profiles: Optional[Dict[str, Dict]] = None,
    out_of_sample_ratio: Optional[float] = None,
    min_oos_cycles: Optional[int] = None,
    min_ev_delta_pct: Optional[float] = None,
) -> Dict:
    ordered_cycles = sorted(
        [cycle for cycle in cycles if isinstance(cycle, dict)],
        key=lambda cycle: str(cycle.get("timestamp") or ""),
    )
    profiles = profiles or build_challenger_profiles()
    out_of_sample_ratio = float(
        config.EXPERIMENT_OOS_RATIO if out_of_sample_ratio is None else out_of_sample_ratio
    )
    min_oos_cycles = int(
        config.EXPERIMENT_MIN_OOS_CYCLES if min_oos_cycles is None else min_oos_cycles
    )
    min_ev_delta_pct = float(
        config.EXPERIMENT_MIN_EV_DELTA_PCT if min_ev_delta_pct is None else min_ev_delta_pct
    )
    partitions = _split_cycles(ordered_cycles, out_of_sample_ratio)

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "cycle_count": len(ordered_cycles),
        "in_sample_cycles": len(partitions["in_sample"]),
        "out_of_sample_cycles": len(partitions["out_of_sample"]),
        "profiles": {},
        "promotion_gate": {
            "baseline": "baseline_current",
            "approved_profiles": [],
            "winner": "baseline_current",
        },
    }

    for profile_name, overrides in profiles.items():
        engine_config = build_decision_engine_config(overrides)
        report["profiles"][profile_name] = {
            "overrides": overrides,
            "in_sample": _summarize_partition(partitions["in_sample"], engine_config),
            "out_of_sample": _summarize_partition(partitions["out_of_sample"], engine_config),
        }

    baseline_oos = report["profiles"].get("baseline_current", {}).get("out_of_sample", {})
    approved = []
    for profile_name, profile_report in report["profiles"].items():
        if profile_name == "baseline_current":
            profile_report["promotion"] = {
                "approved": True,
                "ev_delta_pct": 0.0,
                "execution_cost_delta_pct": 0.0,
                "no_trade_delta": 0.0,
                "reasons": [],
            }
            continue
        profile_report["promotion"] = _assess_promotion(
            baseline_oos,
            profile_report["out_of_sample"],
            min_oos_cycles=min_oos_cycles,
            min_ev_delta_pct=min_ev_delta_pct,
        )
        if profile_report["promotion"]["approved"]:
            approved.append(profile_name)

    report["promotion_gate"]["approved_profiles"] = approved
    if approved:
        approved.sort(
            key=lambda name: report["profiles"][name]["promotion"]["ev_delta_pct"],
            reverse=True,
        )
        report["promotion_gate"]["winner"] = approved[0]
    return report


def run_experiment_benchmark_pack(
    *,
    limit_cycles: Optional[int] = None,
    out_of_sample_ratio: Optional[float] = None,
) -> Dict:
    limit_cycles = int(config.EXPERIMENT_REPORT_LIMIT_CYCLES if limit_cycles is None else limit_cycles)
    cycles = db.get_recent_decision_research(limit=limit_cycles, include_candidates=True)
    report = build_experiment_benchmark_pack(
        cycles,
        out_of_sample_ratio=out_of_sample_ratio,
    )
    report["decision_funnel"] = db.get_decision_funnel_summary(limit_cycles=limit_cycles)
    report["source_attribution"] = db.get_source_attribution_summary(
        limit_cycles=limit_cycles,
        lookback_hours=config.EXPERIMENT_ATTRIBUTION_LOOKBACK_HOURS,
    )
    report["divergence"] = db.get_runtime_divergence_summary(
        lookback_hours=config.EXPERIMENT_DIVERGENCE_LOOKBACK_HOURS,
    )
    return report
