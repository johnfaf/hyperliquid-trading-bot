"""Shadow certification pack for 7-day shadow deployments."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import config
from src.core.incident_visibility import build_runtime_incident_report
from src.core.time_utils import utc_now_iso, utc_now_naive
from src.data import database as db

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_READINESS_INCIDENT_KEYS = {
    "live_preflight_blocked",
    "live_activation_blocked",
    "live_readiness_blocked",
    "live_readiness_degraded",
}


def build_shadow_certification_config(cfg=config) -> Dict:
    lookback_days = max(int(getattr(cfg, "SHADOW_CERTIFICATION_LOOKBACK_DAYS", 7) or 7), 1)
    limit_cycles = int(getattr(cfg, "SHADOW_CERTIFICATION_LIMIT_CYCLES", 0) or 0)
    if limit_cycles <= 0:
        interval = max(int(getattr(cfg, "TRADING_CYCLE_INTERVAL", 900) or 900), 1)
        limit_cycles = max(int((lookback_days * 86400) / interval), 1)

    return {
        "enabled": bool(getattr(cfg, "SHADOW_CERTIFICATION_ENABLED", True)),
        "lookback_days": lookback_days,
        "limit_cycles": limit_cycles,
        "report_path": os.path.abspath(
            getattr(cfg, "SHADOW_CERTIFICATION_REPORT_PATH", "reports/shadow_certification_report.json")
        ),
        "min_shadow_trades": int(getattr(cfg, "SHADOW_CERTIFICATION_MIN_SHADOW_TRADES", 10) or 10),
        "max_open_gap_ratio": float(getattr(cfg, "SHADOW_CERTIFICATION_MAX_OPEN_GAP_RATIO", 0.40)),
        "max_realized_pnl_gap_ratio": float(
            getattr(cfg, "SHADOW_CERTIFICATION_MAX_REALIZED_PNL_GAP_RATIO", 0.45)
        ),
        "max_slippage_drift_bps": float(
            getattr(cfg, "SHADOW_CERTIFICATION_MAX_SLIPPAGE_DRIFT_BPS", 2.0)
        ),
        "max_degraded_source_ratio": float(
            getattr(cfg, "SHADOW_CERTIFICATION_MAX_DEGRADED_SOURCE_RATIO", 0.35)
        ),
        "max_readiness_interruption_runs": int(
            getattr(cfg, "SHADOW_CERTIFICATION_MAX_READINESS_INTERRUPTION_RUNS", 0)
        ),
    }


def _write_json(path: str, payload: Dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _parse_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _sorted_shadow_sources(attribution: Dict[str, Dict], limit: int = 10) -> List[Dict]:
    rows = []
    for source_key, metrics in (attribution or {}).items():
        payload = {"source_key": source_key}
        payload.update(metrics or {})
        rows.append(payload)
    rows.sort(
        key=lambda item: (
            -float(item.get("pnl", 0.0) or 0.0),
            -int(item.get("trades", 0) or 0),
            str(item.get("source_key", "")),
        )
    )
    return rows[:limit]


def _top_degraded_sources(snapshot: Dict, limit: int = 10) -> List[Dict]:
    profiles = list((snapshot.get("profiles", []) or []))
    degraded = [
        profile for profile in profiles
        if str(profile.get("status", "") or "").strip().lower() in {"caution", "blocked"}
    ]
    degraded.sort(
        key=lambda item: (
            str(item.get("status", "warming_up") or "warming_up") != "blocked",
            float(item.get("health_score", 0.0) or 0.0),
            str(item.get("source_key", "")),
        )
    )
    return degraded[:limit]


def _recent_readiness_interruptions(
    recent_runs: List[Dict],
    current_incident_keys: List[str],
    *,
    lookback_days: int,
) -> int:
    cutoff = utc_now_naive() - timedelta(days=max(int(lookback_days or 0), 1))
    count = 0
    for run in recent_runs:
        run_ts = _parse_timestamp(run.get("timestamp"))
        if run_ts and run_ts < cutoff:
            continue
        incident_keys = list(((run.get("metadata", {}) or {}).get("incident_keys", []) or []))
        if any(key in _READINESS_INCIDENT_KEYS for key in incident_keys):
            count += 1
    if any(key in _READINESS_INCIDENT_KEYS for key in current_incident_keys):
        count += 1
    return count


def build_shadow_certification_report(
    *,
    cfg: Optional[Dict] = None,
    shadow_tracker=None,
    live_trader=None,
    divergence_controller=None,
    capital_governor=None,
    adaptive_learning=None,
) -> Dict:
    settings = dict(cfg or build_shadow_certification_config(config))
    lookback_days = max(int(settings.get("lookback_days", 7) or 7), 1)
    lookback_hours = float(lookback_days * 24)
    limit_cycles = max(int(settings.get("limit_cycles", 1) or 1), 1)
    timestamp = utc_now_iso()

    divergence = db.get_runtime_divergence_summary(lookback_hours=lookback_hours)
    decision_funnel = db.get_decision_funnel_summary(limit_cycles=limit_cycles)
    source_attribution = db.get_source_attribution_summary(
        limit_cycles=limit_cycles,
        lookback_hours=lookback_hours,
    )
    execution_quality = db.get_execution_quality_summary(lookback_hours=lookback_hours)
    execution_by_source = db.get_execution_quality_by_source(
        lookback_hours=lookback_hours,
        limit=10,
    )
    capital_summary = db.get_capital_governor_summary(lookback_hours=lookback_hours)
    source_snapshot = db.get_latest_source_health_snapshot(limit=25)
    daily_research = db.get_latest_daily_research_run()
    recent_runs = db.get_recent_shadow_certification_runs(limit=max(lookback_days + 3, 10))

    shadow_summary = {}
    shadow_attribution = {}
    if shadow_tracker:
        try:
            shadow_summary = shadow_tracker.get_summary(days=lookback_days)
            shadow_attribution = shadow_tracker.get_attribution(days=lookback_days)
        except Exception as exc:
            logger.debug("shadow certification shadow tracker error: %s", exc)

    incident_report = build_runtime_incident_report(
        live_trader=live_trader,
        divergence_controller=divergence_controller,
        capital_governor=capital_governor,
        adaptive_learning=adaptive_learning,
        max_items=getattr(config, "RUNTIME_INCIDENT_MAX_ITEMS", 8),
    )
    incident_keys = [item.get("key", "") for item in (incident_report.get("incidents", []) or [])]
    readiness_interruptions = _recent_readiness_interruptions(
        recent_runs,
        incident_keys,
        lookback_days=lookback_days,
    )

    slippage_drift_bps = round(
        float(execution_quality.get("avg_realized_slippage_bps", 0.0) or 0.0)
        - float(execution_quality.get("avg_expected_slippage_bps", 0.0) or 0.0),
        4,
    )
    shadow_trades = int(shadow_summary.get("total_trades", 0) or 0)
    checks = [
        {
            "name": "shadow_trade_sample",
            "passed": shadow_trades >= int(settings.get("min_shadow_trades", 10) or 10),
            "value": shadow_trades,
            "threshold": int(settings.get("min_shadow_trades", 10) or 10),
        },
        {
            "name": "paper_live_open_gap_ratio",
            "passed": float(divergence.get("paper_live_open_gap_ratio", 0.0) or 0.0)
            <= float(settings.get("max_open_gap_ratio", 0.40) or 0.40),
            "value": float(divergence.get("paper_live_open_gap_ratio", 0.0) or 0.0),
            "threshold": float(settings.get("max_open_gap_ratio", 0.40) or 0.40),
        },
        {
            "name": "realized_pnl_gap_ratio",
            "passed": float(divergence.get("realized_pnl_gap_ratio", 0.0) or 0.0)
            <= float(settings.get("max_realized_pnl_gap_ratio", 0.45) or 0.45),
            "value": float(divergence.get("realized_pnl_gap_ratio", 0.0) or 0.0),
            "threshold": float(settings.get("max_realized_pnl_gap_ratio", 0.45) or 0.45),
        },
        {
            "name": "slippage_drift_bps",
            "passed": abs(slippage_drift_bps)
            <= float(settings.get("max_slippage_drift_bps", 2.0) or 2.0),
            "value": slippage_drift_bps,
            "threshold": float(settings.get("max_slippage_drift_bps", 2.0) or 2.0),
        },
        {
            "name": "degraded_source_ratio",
            "passed": float(capital_summary.get("degraded_source_ratio", 0.0) or 0.0)
            <= float(settings.get("max_degraded_source_ratio", 0.35) or 0.35),
            "value": float(capital_summary.get("degraded_source_ratio", 0.0) or 0.0),
            "threshold": float(settings.get("max_degraded_source_ratio", 0.35) or 0.35),
        },
        {
            "name": "readiness_interruptions",
            "passed": readiness_interruptions
            <= int(settings.get("max_readiness_interruption_runs", 0) or 0),
            "value": readiness_interruptions,
            "threshold": int(settings.get("max_readiness_interruption_runs", 0) or 0),
        },
    ]

    failed_checks = [check["name"] for check in checks if not check["passed"]]
    if shadow_trades < int(settings.get("min_shadow_trades", 10) or 10):
        status = "warming_up"
        certified = False
        summary = (
            f"Shadow certification warming up: {shadow_trades} shadow trades "
            f"vs required {int(settings.get('min_shadow_trades', 10) or 10)}."
        )
    elif failed_checks:
        status = "failed"
        certified = False
        summary = "Shadow certification failed checks: " + ", ".join(failed_checks)
    else:
        status = "certified"
        certified = True
        summary = (
            f"Shadow certification passed over {lookback_days}d with {shadow_trades} "
            f"shadow trades and slippage drift {slippage_drift_bps:+.2f}bps."
        )

    report = {
        "timestamp": timestamp,
        "lookback_days": lookback_days,
        "status": status,
        "certified": certified,
        "summary": summary,
        "checks": checks,
        "shadow_summary": shadow_summary,
        "shadow_top_sources": _sorted_shadow_sources(shadow_attribution, limit=10),
        "divergence": divergence,
        "decision_funnel": decision_funnel,
        "blocked_entry_reasons": dict(list((decision_funnel.get("blocker_mix", {}) or {}).items())[:10]),
        "source_attribution": source_attribution[:12],
        "execution_quality": {
            **execution_quality,
            "slippage_drift_bps": slippage_drift_bps,
            "by_source": execution_by_source,
        },
        "capital_summary": capital_summary,
        "source_health": {
            "snapshot": source_snapshot.get("snapshot"),
            "top_degraded_sources": _top_degraded_sources(source_snapshot, limit=10),
        },
        "runtime_incidents": incident_report,
        "readiness_interruptions": {
            "count": readiness_interruptions,
            "incident_keys": [key for key in incident_keys if key in _READINESS_INCIDENT_KEYS],
        },
        "daily_research": {
            "latest": daily_research,
        },
    }
    return report


def run_shadow_certification(
    *,
    shadow_tracker=None,
    live_trader=None,
    divergence_controller=None,
    capital_governor=None,
    adaptive_learning=None,
) -> Dict:
    settings = build_shadow_certification_config(config)
    report = build_shadow_certification_report(
        cfg=settings,
        shadow_tracker=shadow_tracker,
        live_trader=live_trader,
        divergence_controller=divergence_controller,
        capital_governor=capital_governor,
        adaptive_learning=adaptive_learning,
    )
    metadata = {
        "summary": report.get("summary"),
        "checks": report.get("checks", []),
        "incident_keys": [
            item.get("key", "")
            for item in ((report.get("runtime_incidents", {}) or {}).get("incidents", []) or [])
            if item.get("key")
        ],
        "blocked_entry_reasons": report.get("blocked_entry_reasons", {}),
        "readiness_interruptions": report.get("readiness_interruptions", {}),
        "shadow_summary": report.get("shadow_summary", {}),
    }
    run_id = db.save_shadow_certification_run(
        {
            "timestamp": report.get("timestamp"),
            "lookback_days": report.get("lookback_days"),
            "status": report.get("status"),
            "certified": report.get("certified", False),
            "metadata": metadata,
        }
    )
    report["run_id"] = run_id
    report_path = settings.get("report_path", "")
    if report_path:
        _write_json(report_path, report)
    return report
