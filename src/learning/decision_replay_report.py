"""Decision-level replay diagnostics for offline learning backtests.

The candle backtest answers "would a signal rule have made money?".
This module answers the more useful production question: "why did the bot
take or reject each recorded decision, and what happened after costs, data
quality, portfolio overlap, and live/paper execution drift are accounted for?"
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.learning.dataset_builder import DatasetBuildResult, LearningExample


BAD_SOURCE_STATUSES = {"down", "missing", "stale", "error", "unavailable", "offline"}
DEGRADED_SOURCE_STATUSES = {"degraded", "lagging", "partial"}
DATA_GAP_KEYS = {
    "data_gap",
    "data_gap_detected",
    "candles_incomplete",
    "features_incomplete",
    "missing_candles",
    "market_data_missing",
    "source_data_missing",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _loads(value: Any, fallback: Any = None) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        loaded = json.loads(value or "{}")
        return loaded
    except Exception:
        return fallback if fallback is not None else {}


def _as_dict(value: Any) -> Dict[str, Any]:
    loaded = _loads(value, {})
    return dict(loaded) if isinstance(loaded, dict) else {}


def _float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if value in (None, ""):
            return default
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_ts(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        raw = float(value)
        return raw / 1000.0 if raw > 10_000_000_000 else raw
    try:
        text = str(value).strip()
        if text.isdigit():
            raw = float(text)
            return raw / 1000.0 if raw > 10_000_000_000 else raw
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except Exception:
        return None


def _metadata_layers(example: LearningExample) -> List[Dict[str, Any]]:
    meta = _as_dict(example.metadata)
    decision_meta = _as_dict(meta.get("decision_metadata"))
    outcome_meta = _as_dict(meta.get("outcome_metadata"))
    paper_meta = _as_dict(outcome_meta.get("paper_metadata"))
    raw_signal = _as_dict(meta.get("raw_signal"))
    layers = [meta, decision_meta, outcome_meta, paper_meta, raw_signal]
    return [layer for layer in layers if layer]


def _first_float(
    layers: Iterable[Dict[str, Any]],
    keys: Iterable[str],
    default: Optional[float] = None,
) -> Optional[float]:
    for layer in layers:
        for key in keys:
            value = _float(layer.get(key), None)
            if value is not None:
                return value
    return default


def _first_value(layers: Iterable[Dict[str, Any]], keys: Iterable[str]) -> Any:
    for layer in layers:
        for key in keys:
            value = layer.get(key)
            if value not in (None, "", {}, []):
                return value
    return None


def _source_health_payload(example: LearningExample) -> Dict[str, Any]:
    for layer in _metadata_layers(example):
        value = layer.get("source_health")
        if value not in (None, "", {}, []):
            return _as_dict(value)
    return {}


def summarize_source_health(example: LearningExample) -> Dict[str, Any]:
    """Return compact source-health counts attached to a decision."""
    health = _source_health_payload(example)
    statuses: Counter[str] = Counter()
    unhealthy: List[str] = []
    degraded: List[str] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            status = str(value.get("status") or value.get("state") or "").strip().lower()
            if status:
                statuses[status] += 1
                label = prefix or str(value.get("source") or "source")
                if status in BAD_SOURCE_STATUSES:
                    unhealthy.append(f"{label}:{status}")
                elif status in DEGRADED_SOURCE_STATUSES:
                    degraded.append(f"{label}:{status}")
                return
            for key, child in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), child)
            return
        status = str(value or "").strip().lower()
        if status:
            statuses[status] += 1
            if status in BAD_SOURCE_STATUSES:
                unhealthy.append(f"{prefix}:{status}")
            elif status in DEGRADED_SOURCE_STATUSES:
                degraded.append(f"{prefix}:{status}")

    walk("", health)
    return {
        "attached": bool(health),
        "status_counts": dict(statuses),
        "unhealthy": sorted(set(unhealthy))[:20],
        "degraded": sorted(set(degraded))[:20],
        "unhealthy_count": len(set(unhealthy)),
        "degraded_count": len(set(degraded)),
    }


def detect_example_data_gaps(
    example: LearningExample,
    *,
    required_feature_names: Optional[Iterable[str]] = None,
    min_candle_coverage: float = 0.80,
) -> Dict[str, Any]:
    """Detect whether a replay example has incomplete decision data.

    The function is intentionally metadata-driven so historical rows can be
    audited without live API calls.
    """
    required = [str(name) for name in (required_feature_names or []) if str(name)]
    missing_features = [name for name in required if name not in (example.features or {})]
    reasons: List[str] = []
    if missing_features:
        reasons.append("missing_features")

    layers = _metadata_layers(example)
    explicit_flags: List[str] = []
    coverage: Dict[str, float] = {}
    for layer in layers:
        for key, value in layer.items():
            key_l = str(key).lower()
            if key_l in DATA_GAP_KEYS and _bool(value):
                explicit_flags.append(key_l)
            if key_l.startswith("data_coverage_"):
                cov = _float(value, None)
                if cov is not None:
                    coverage[key_l.replace("data_coverage_", "")] = float(cov)
            if key_l == "forward_label_metadata" and isinstance(value, dict):
                for child_key, child_val in value.items():
                    child_l = str(child_key).lower()
                    if child_l.startswith("data_coverage_"):
                        cov = _float(child_val, None)
                        if cov is not None:
                            coverage[child_l.replace("data_coverage_", "")] = float(cov)
                    if child_l in DATA_GAP_KEYS and _bool(child_val):
                        explicit_flags.append(child_l)
    if explicit_flags:
        reasons.append("explicit_data_gap_flag")
    low_coverage = {
        horizon: value
        for horizon, value in coverage.items()
        if value < float(min_candle_coverage)
    }
    if low_coverage:
        reasons.append("low_candle_coverage")

    health = summarize_source_health(example)
    if health["unhealthy_count"]:
        reasons.append("source_unhealthy")

    return {
        "has_gap": bool(reasons),
        "reasons": sorted(set(reasons)),
        "missing_features": missing_features[:50],
        "explicit_flags": sorted(set(explicit_flags)),
        "candle_coverage": coverage,
        "low_candle_coverage": low_coverage,
        "source_health": health,
    }


def _cost_breakdown(example: LearningExample) -> Dict[str, Any]:
    layers = _metadata_layers(example)
    expected_size = _first_float(
        layers,
        ("proposed_size_usd", "expected_size_usd", "notional_usd", "size_usd"),
    )
    actual_size = _first_float(
        layers,
        ("actual_size_usd", "actual_notional_usd", "filled_notional_usd", "live_notional_usd"),
    )
    fees = _first_float(layers, ("total_fees_paid", "fees_paid", "fee_paid", "fees", "total_fee_usd")) or 0.0
    slippage = _first_float(
        layers,
        ("total_slippage_cost", "slippage_cost", "slippage_usd", "slippage_paid"),
    ) or 0.0
    funding = _first_float(
        layers,
        ("funding_paid", "total_funding_paid", "funding_cost", "funding_cost_usd"),
    ) or 0.0
    spread = _first_float(layers, ("spread_cost", "spread_cost_usd", "total_spread_cost")) or 0.0
    spread_bps = _first_float(layers, ("spread_bps", "entry_spread_bps"), None)
    if not spread and spread_bps is not None and expected_size:
        spread = abs(expected_size) * float(spread_bps) / 10_000.0

    fill_ratio = _first_float(layers, ("fill_ratio", "filled_ratio", "live_fill_ratio"), None)
    if fill_ratio is None and expected_size and actual_size is not None:
        fill_ratio = max(0.0, min(abs(actual_size) / max(abs(expected_size), 1e-9), 1.0))
    partial_fill = bool(fill_ratio is not None and fill_ratio < 0.999)
    partial_fill = partial_fill or any(_bool(layer.get("partial_fill")) for layer in layers)
    missed_fill = any(_bool(layer.get("missed_fill")) or _bool(layer.get("no_fill")) for layer in layers)
    if fill_ratio == 0.0:
        missed_fill = True

    gross = _first_float(layers, ("gross_pnl_before_fees", "gross_pnl", "pnl_before_costs"), None)
    known_costs = float(fees) + float(slippage) + float(funding) + float(spread)
    outcome_pnl = float(example.outcome_pnl or 0.0)
    if gross is not None:
        adjusted = float(gross) - known_costs
        assumption = "gross_minus_known_costs"
    else:
        adjusted = outcome_pnl
        assumption = "outcome_pnl_assumed_net"

    return {
        "outcome_pnl": outcome_pnl,
        "adjusted_pnl": adjusted,
        "gross_pnl_before_fees": gross,
        "known_costs": known_costs,
        "fees": float(fees),
        "slippage": float(slippage),
        "funding": float(funding),
        "spread": float(spread),
        "spread_bps": spread_bps,
        "expected_size_usd": expected_size,
        "actual_size_usd": actual_size,
        "fill_ratio": fill_ratio,
        "partial_fill": partial_fill,
        "missed_fill": missed_fill,
        "assumption": assumption,
    }


def _live_paper_drift(example: LearningExample) -> Dict[str, Any]:
    layers = _metadata_layers(example)
    paper_pnl = _first_float(layers, ("paper_pnl", "paper_outcome_pnl"), None)
    if paper_pnl is None:
        paper_pnl = float(example.outcome_pnl or 0.0)
    live_pnl = _first_float(layers, ("live_pnl", "live_outcome_pnl", "realized_live_pnl"), None)
    paper_entry = _first_float(layers, ("paper_entry_price", "entry_price"), None)
    live_entry = _first_float(layers, ("live_entry_price", "actual_entry_price", "fill_price"), None)
    paper_exit = _first_float(layers, ("paper_exit_price", "exit_price"), None)
    live_exit = _first_float(layers, ("live_exit_price", "actual_exit_price"), None)
    has_live = live_pnl is not None or live_entry is not None or live_exit is not None
    entry_drift_bps = None
    exit_drift_bps = None
    if paper_entry and live_entry:
        entry_drift_bps = (float(live_entry) - float(paper_entry)) / float(paper_entry) * 10_000.0
    if paper_exit and live_exit:
        exit_drift_bps = (float(live_exit) - float(paper_exit)) / float(paper_exit) * 10_000.0
    pnl_drift = (float(live_pnl) - float(paper_pnl)) if live_pnl is not None else None
    return {
        "has_live_execution": bool(has_live),
        "paper_pnl": paper_pnl,
        "live_pnl": live_pnl,
        "pnl_drift": pnl_drift,
        "entry_drift_bps": entry_drift_bps,
        "exit_drift_bps": exit_drift_bps,
    }


def _regime_label(example: LearningExample) -> str:
    regime = example.regime or {}
    if not regime:
        for layer in _metadata_layers(example):
            regime = _as_dict(layer.get("regime"))
            if regime:
                break
    if not isinstance(regime, dict):
        return ""
    return str(regime.get("overall_regime") or regime.get("regime") or regime.get("label") or "")


def _counterfactual_bucket(example: LearningExample, costs: Dict[str, Any]) -> str:
    final_status = str(example.final_status or example.metadata.get("final_status") or "").lower()
    reason = str(example.rejection_reason or example.metadata.get("rejection_reason") or "").lower()
    exit_reason = str(_first_value(_metadata_layers(example), ("exit_reason", "close_reason")) or "").lower()
    if costs.get("missed_fill"):
        return "missed_fill"
    if not example.executed:
        return "rejected_would_win" if int(example.label_win or 0) == 1 else "rejected_would_lose"
    if "reverse" in exit_reason or "regime" in exit_reason:
        return "reversed"
    if any(token in exit_reason for token in ("manual", "early", "trailing", "time_exit")):
        return "closed_early"
    if "take_profit" in exit_reason or "stop_loss" in exit_reason or "tp" == exit_reason or "sl" == exit_reason:
        return "held_to_sl_tp"
    if costs.get("partial_fill"):
        return "partial_fill"
    if "reject" in final_status or reason:
        return "rejected_would_win" if int(example.label_win or 0) == 1 else "rejected_would_lose"
    return "accepted_executed"


def _why_entered(example: LearningExample, costs: Dict[str, Any], gaps: Dict[str, Any]) -> str:
    layers = _metadata_layers(example)
    decision_meta = _as_dict(example.metadata.get("decision_metadata") if isinstance(example.metadata, dict) else {})
    firewall = (
        example.metadata.get("firewall_decision")
        if isinstance(example.metadata, dict)
        else None
    ) or decision_meta.get("firewall_decision")
    source_key = example.source_key or example.source or "unknown"
    bits = [
        f"{example.side or '?'} {example.coin or '?'} from {source_key}",
        f"confidence={float(example.confidence or 0.0):.2f}",
    ]
    if example.strategy_type:
        bits.append(f"strategy={example.strategy_type}")
    regime = _regime_label(example)
    if regime:
        bits.append(f"regime={regime}")
    if firewall or example.final_status:
        bits.append(f"firewall={firewall or 'unknown'} status={example.final_status or 'unknown'}")
    if example.rejection_reason:
        bits.append(f"reason={example.rejection_reason}")
    risk_bits = []
    for label, keys in (
        ("size_usd", ("proposed_size_usd", "expected_size_usd")),
        ("lev", ("proposed_leverage", "leverage")),
        ("sl_roe", ("proposed_sl_roe",)),
        ("tp_roe", ("proposed_tp_roe",)),
        ("sl_price", ("proposed_sl_price",)),
        ("tp_price", ("proposed_tp_price",)),
    ):
        value = _first_value(layers, keys)
        if value not in (None, ""):
            risk_bits.append(f"{label}={value}")
    if risk_bits:
        bits.append("risk(" + ", ".join(risk_bits) + ")")
    health = gaps.get("source_health") or {}
    if health.get("unhealthy") or health.get("degraded"):
        bits.append(
            "source_health="
            + ",".join((health.get("unhealthy") or []) + (health.get("degraded") or []))
        )
    if gaps.get("has_gap"):
        bits.append("data_gap=" + ",".join(gaps.get("reasons") or []))
    if costs.get("partial_fill"):
        bits.append(f"partial_fill={costs.get('fill_ratio')}")
    if costs.get("missed_fill"):
        bits.append("missed_fill=true")
    if example.explanation:
        bits.append(f"journal='{example.explanation}'")
    return "; ".join(bits)


def _trade_row(
    example: LearningExample,
    *,
    accepted_by_policy: bool,
    required_feature_names: Iterable[str],
    min_candle_coverage: float,
) -> Dict[str, Any]:
    costs = _cost_breakdown(example)
    gaps = detect_example_data_gaps(
        example,
        required_feature_names=required_feature_names,
        min_candle_coverage=min_candle_coverage,
    )
    drift = _live_paper_drift(example)
    bucket = _counterfactual_bucket(example, costs)
    return {
        "decision_id": example.decision_id,
        "created_at": example.created_at,
        "coin": example.coin,
        "side": example.side,
        "source": example.source,
        "source_key": example.source_key,
        "strategy_type": example.strategy_type,
        "confidence": float(example.confidence or 0.0),
        "executed": bool(example.executed),
        "accepted_by_policy": bool(accepted_by_policy),
        "label_win": example.label_win,
        "final_status": example.final_status,
        "rejection_reason": example.rejection_reason,
        "regime": _regime_label(example),
        "outcome_pnl": costs["outcome_pnl"],
        "adjusted_pnl": costs["adjusted_pnl"],
        "hold_minutes": _first_float(_metadata_layers(example), ("hold_minutes",), None),
        "costs": costs,
        "data_gap": gaps,
        "counterfactual_bucket": bucket,
        "live_paper_drift": drift,
        "why_entered": _why_entered(example, costs, gaps),
    }


def _max_drawdown(pnls: Iterable[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        equity += float(pnl)
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


def _portfolio_summary(rows: List[Dict[str, Any]], *, initial_balance: float) -> Dict[str, Any]:
    accepted = [row for row in rows if row.get("accepted_by_policy")]
    pnls = [float(row.get("adjusted_pnl", 0.0) or 0.0) for row in accepted]
    by_coin: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0})
    equity = float(initial_balance)
    curve: List[Dict[str, Any]] = []
    events: List[Tuple[float, int, str]] = []
    for row, pnl in zip(accepted, pnls):
        coin = str(row.get("coin") or "UNKNOWN").upper()
        by_coin[coin]["trades"] += 1
        by_coin[coin]["pnl"] += pnl
        if pnl > 0:
            by_coin[coin]["wins"] += 1
        equity += pnl
        curve.append({"created_at": row.get("created_at"), "equity": round(equity, 6), "pnl": pnl})
        start = _parse_ts(row.get("created_at"))
        hold_minutes = _float(row.get("hold_minutes"), None)
        if start is not None and hold_minutes is not None and hold_minutes > 0:
            events.append((start, 1, coin))
            events.append((start + hold_minutes * 60.0, -1, coin))

    active = 0
    active_by_coin: Counter[str] = Counter()
    max_concurrent = 0
    max_concurrent_coins = 0
    for _, delta, coin in sorted(events, key=lambda item: (item[0], -item[1], item[2])):
        active += delta
        active_by_coin[coin] += delta
        if active_by_coin[coin] <= 0:
            active_by_coin.pop(coin, None)
        max_concurrent = max(max_concurrent, active)
        max_concurrent_coins = max(max_concurrent_coins, len(active_by_coin))

    coin_rows = []
    for coin, payload in by_coin.items():
        trades = int(payload["trades"])
        coin_rows.append(
            {
                "coin": coin,
                "trades": trades,
                "pnl": round(float(payload["pnl"]), 6),
                "win_rate": round(float(payload["wins"]) / trades, 4) if trades else 0.0,
            }
        )
    coin_rows.sort(key=lambda row: row["pnl"], reverse=True)
    return {
        "mode": "terminal_decision_portfolio_replay",
        "initial_balance": float(initial_balance),
        "final_balance": round(equity, 6),
        "total_pnl": round(sum(pnls), 6),
        "max_drawdown": round(_max_drawdown(pnls), 6),
        "coins": coin_rows,
        "coin_count": len(coin_rows),
        "multi_coin": len(coin_rows) > 1,
        "max_concurrent_positions": max_concurrent,
        "max_concurrent_coins": max_concurrent_coins,
        "equity_curve": curve[-250:],
    }
def build_decision_replay_report(
    dataset: DatasetBuildResult,
    policy: Any = None,
    *,
    initial_balance: float = 1_000.0,
    max_trade_rows: int = 500,
    min_candle_coverage: float = 0.80,
    max_accepted_data_gap_ratio: float = 0.0,
) -> Dict[str, Any]:
    """Build a compact audit report for a decision-snapshot replay."""
    examples = sorted(dataset.examples or [], key=lambda item: str(item.created_at or ""))
    required_features = list(dataset.feature_names or [])
    rows: List[Dict[str, Any]] = []
    for example in examples:
        accepted = bool(policy.accepts(example)) if policy is not None else bool(example.label_win is not None)
        rows.append(
            _trade_row(
                example,
                accepted_by_policy=accepted,
                required_feature_names=required_features,
                min_candle_coverage=min_candle_coverage,
            )
        )

    accepted_rows = [row for row in rows if row.get("accepted_by_policy")]
    accepted_gaps = [row for row in accepted_rows if row.get("data_gap", {}).get("has_gap")]
    all_gaps = [row for row in rows if row.get("data_gap", {}).get("has_gap")]
    cost_totals = {
        "fees": round(sum(float(row["costs"].get("fees", 0.0) or 0.0) for row in accepted_rows), 6),
        "slippage": round(sum(float(row["costs"].get("slippage", 0.0) or 0.0) for row in accepted_rows), 6),
        "funding": round(sum(float(row["costs"].get("funding", 0.0) or 0.0) for row in accepted_rows), 6),
        "spread": round(sum(float(row["costs"].get("spread", 0.0) or 0.0) for row in accepted_rows), 6),
    }
    cost_totals["known_costs"] = round(sum(cost_totals.values()), 6)

    buckets = Counter(str(row.get("counterfactual_bucket") or "unknown") for row in rows)
    live_rows = [row for row in accepted_rows if row.get("live_paper_drift", {}).get("has_live_execution")]
    drift_values = [
        float(row["live_paper_drift"]["pnl_drift"])
        for row in live_rows
        if row.get("live_paper_drift", {}).get("pnl_drift") is not None
    ]
    accepted_gap_ratio = len(accepted_gaps) / len(accepted_rows) if accepted_rows else 0.0
    data_quality_passed = accepted_gap_ratio <= float(max_accepted_data_gap_ratio)

    return {
        "created_at": _now(),
        "dataset_id": dataset.dataset_id,
        "summary": {
            "examples": len(rows),
            "accepted_by_policy": len(accepted_rows),
            "executed": sum(1 for row in rows if row.get("executed")),
            "rejected": sum(1 for row in rows if not row.get("executed")),
            "coins": sorted({str(row.get("coin") or "").upper() for row in rows if row.get("coin")}),
        },
        "data_quality": {
            "passed": data_quality_passed,
            "accepted_data_gap_count": len(accepted_gaps),
            "accepted_data_gap_ratio": accepted_gap_ratio,
            "all_data_gap_count": len(all_gaps),
            "max_accepted_data_gap_ratio": float(max_accepted_data_gap_ratio),
            "gap_reasons": dict(Counter(reason for row in all_gaps for reason in row["data_gap"].get("reasons", []))),
        },
        "execution_costs": cost_totals,
        "fills": {
            "partial_fill_count": sum(1 for row in accepted_rows if row["costs"].get("partial_fill")),
            "missed_fill_count": sum(1 for row in accepted_rows if row["costs"].get("missed_fill")),
        },
        "counterfactuals": dict(buckets),
        "live_paper_drift": {
            "coverage_count": len(live_rows),
            "coverage_ratio": len(live_rows) / len(accepted_rows) if accepted_rows else 0.0,
            "avg_pnl_drift": round(sum(drift_values) / len(drift_values), 6) if drift_values else 0.0,
            "total_pnl_drift": round(sum(drift_values), 6) if drift_values else 0.0,
        },
        "portfolio": _portfolio_summary(accepted_rows, initial_balance=initial_balance),
        "trade_reports": rows[: max(int(max_trade_rows), 0)],
    }
