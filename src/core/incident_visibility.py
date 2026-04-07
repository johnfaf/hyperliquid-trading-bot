"""Shared runtime incident collection for reporting, dashboard, and alerts."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import config
from src.core.time_utils import utc_now_iso

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_SEVERITY_RANKS = {
    "info": 1,
    "warning": 2,
    "critical": 3,
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _incident(
    key: str,
    title: str,
    summary: str,
    *,
    severity: str,
    source: str,
    status: str,
    blocking: bool,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized = str(severity or "warning").strip().lower() or "warning"
    if normalized not in _SEVERITY_RANKS:
        normalized = "warning"
    return {
        "key": key,
        "title": title,
        "summary": summary,
        "severity": normalized,
        "severity_rank": _SEVERITY_RANKS[normalized],
        "source": source,
        "status": str(status or "unknown"),
        "blocking": bool(blocking),
        "metadata": dict(metadata or {}),
    }


def _join_reasons(values: Any, fallback: str = "none") -> str:
    items = [str(item).strip() for item in (values or []) if str(item).strip()]
    return ", ".join(items[:4]) or fallback


def _collect_live_incidents(live_trader=None) -> List[Dict[str, Any]]:
    incidents: List[Dict[str, Any]] = []
    if not live_trader:
        return incidents

    try:
        stats = live_trader.get_stats()
    except Exception as exc:
        logger.debug("runtime incident collector live_trader error: %s", exc)
        return incidents

    live_requested = bool(stats.get("live_requested", stats.get("live_enabled", False)))
    live_enabled = bool(stats.get("live_enabled", False))
    kill_switch_active = bool(stats.get("kill_switch_active", False))
    if kill_switch_active and live_enabled:
        incidents.append(
            _incident(
                "live_kill_switch_active",
                "Live kill switch active",
                "New live entries are blocked until the kill switch is cleared.",
                severity="critical",
                source="live_trader",
                status="blocked",
                blocking=True,
                metadata={"status_reason": stats.get("status_reason", "")},
            )
        )

    if not live_requested or not live_enabled:
        return incidents

    preflight = stats.get("preflight", {}) or {}
    if preflight:
        blocking_checks = list(preflight.get("blocking_checks", []) or [])
        warning_checks = list(preflight.get("warning_checks", []) or [])
        if not bool(preflight.get("deployable", False)):
            incidents.append(
                _incident(
                    "live_preflight_blocked",
                    "Live preflight blocked",
                    f"Live deployability is blocked by: {_join_reasons(blocking_checks)}.",
                    severity="critical",
                    source="live_trader",
                    status=str(preflight.get("status", "blocked") or "blocked"),
                    blocking=True,
                    metadata={
                        "blocking_checks": blocking_checks,
                        "warning_checks": warning_checks,
                    },
                )
            )
        elif warning_checks:
            incidents.append(
                _incident(
                    "live_preflight_warnings",
                    "Live preflight warnings",
                    f"Live preflight has warnings: {_join_reasons(warning_checks)}.",
                    severity="warning",
                    source="live_trader",
                    status=str(preflight.get("status", "warning") or "warning"),
                    blocking=False,
                    metadata={"warning_checks": warning_checks},
                )
            )

    activation = stats.get("activation_guard", {}) or {}
    if activation:
        blocking_checks = list(activation.get("blocking_checks", []) or [])
        warning_checks = list(activation.get("warning_checks", []) or [])
        activation_status = str(activation.get("status", "not_run") or "not_run")
        if not bool(activation.get("deployable", False)):
            incidents.append(
                _incident(
                    "live_activation_blocked",
                    "Live activation blocked",
                    f"Live activation approval is not deployable: {_join_reasons(blocking_checks)}.",
                    severity="critical",
                    source="live_trader",
                    status=activation_status,
                    blocking=True,
                    metadata={
                        "approved_by": activation.get("approved_by", ""),
                        "approved_at": activation.get("approved_at", ""),
                        "expires_at": activation.get("expires_at", ""),
                        "blocking_checks": blocking_checks,
                    },
                )
            )
        elif "approval_expiring_soon" in warning_checks:
            incidents.append(
                _incident(
                    "live_activation_warning",
                    "Live activation expiring soon",
                    f"Activation approval expires soon: {_join_reasons(warning_checks)}.",
                    severity="warning",
                    source="live_trader",
                    status=activation_status,
                    blocking=False,
                    metadata={
                        "approved_by": activation.get("approved_by", ""),
                        "expires_at": activation.get("expires_at", ""),
                        "hours_remaining": activation.get("hours_remaining"),
                        "warning_checks": warning_checks,
                    },
                )
            )
        elif "activation_guard_enabled" in warning_checks or activation_status == "disabled":
            incidents.append(
                _incident(
                    "live_activation_guard_disabled",
                    "Live activation guard disabled",
                    "Live activation approval checks are disabled; rely on explicit runtime mode and preflight controls.",
                    severity="warning",
                    source="live_trader",
                    status=activation_status,
                    blocking=False,
                    metadata={
                        "approved_by": activation.get("approved_by", ""),
                        "warning_checks": warning_checks,
                    },
                )
            )
        elif warning_checks:
            incidents.append(
                _incident(
                    "live_activation_warning",
                    "Live activation warnings",
                    f"Activation approval has warnings: {_join_reasons(warning_checks)}.",
                    severity="warning",
                    source="live_trader",
                    status=activation_status,
                    blocking=False,
                    metadata={
                        "approved_by": activation.get("approved_by", ""),
                        "expires_at": activation.get("expires_at", ""),
                        "hours_remaining": activation.get("hours_remaining"),
                        "warning_checks": warning_checks,
                    },
                )
            )

    readiness = stats.get("live_readiness", {}) or {}
    readiness_status = str(readiness.get("status", "") or "").strip().lower()
    readiness_reason = str(readiness.get("status_reason", "") or "")
    if readiness_status in {"blocked", "degraded"}:
        known_reason_prefixes = ("preflight:", "activation:")
        if (
            not kill_switch_active
            and not readiness_reason.startswith(known_reason_prefixes)
        ):
            incidents.append(
                _incident(
                    f"live_readiness_{readiness_status}",
                    "Live readiness degraded",
                    f"Live readiness is {readiness_status}: {readiness_reason or 'unknown reason'}.",
                    severity="critical" if readiness_status == "blocked" else "warning",
                    source="live_trader",
                    status=readiness_status,
                    blocking=readiness_status == "blocked",
                    metadata={"status_reason": readiness_reason},
                )
            )
    return incidents


def _collect_divergence_incidents(divergence_controller=None) -> List[Dict[str, Any]]:
    incidents: List[Dict[str, Any]] = []
    if not divergence_controller:
        return incidents

    try:
        payload = divergence_controller.get_dashboard_payload(limit=8)
    except Exception as exc:
        logger.debug("runtime incident collector divergence error: %s", exc)
        return incidents

    global_profile = payload.get("global", {}) or {}
    status = str(global_profile.get("status", "healthy") or "healthy").strip().lower()
    reasons = list(global_profile.get("reasons", []) or [])
    affected_profiles = [
        item for item in (payload.get("profiles", []) or [])
        if str(item.get("status", "") or "").strip().lower() in {"blocked", "caution"}
    ]
    if status == "blocked":
        incidents.append(
            _incident(
                "divergence_runtime_blocked",
                "Divergence control blocked entries",
                f"Live/paper divergence is blocking new entries: {_join_reasons(reasons)}.",
                severity="critical",
                source="divergence_controller",
                status=status,
                blocking=True,
                metadata={
                    "reasons": reasons,
                    "tracked_sources": payload.get("tracked_sources", 0),
                    "affected_profiles": len(affected_profiles),
                },
            )
        )
    elif status == "caution":
        incidents.append(
            _incident(
                "divergence_runtime_caution",
                "Divergence control caution",
                f"Live/paper divergence is elevated: {_join_reasons(reasons)}.",
                severity="warning",
                source="divergence_controller",
                status=status,
                blocking=False,
                metadata={
                    "reasons": reasons,
                    "tracked_sources": payload.get("tracked_sources", 0),
                    "affected_profiles": len(affected_profiles),
                },
            )
        )
    return incidents


def _collect_capital_incidents(capital_governor=None) -> List[Dict[str, Any]]:
    incidents: List[Dict[str, Any]] = []
    if not capital_governor:
        return incidents

    try:
        payload = capital_governor.get_dashboard_payload()
    except Exception as exc:
        logger.debug("runtime incident collector capital governor error: %s", exc)
        return incidents

    runtime = payload.get("runtime", {}) or {}
    status = str(runtime.get("status", "healthy") or "healthy").strip().lower()
    reasons = list(runtime.get("reasons", []) or [])

    if bool(runtime.get("operator_risk_off_enabled", False)):
        incidents.append(
            _incident(
                "operator_risk_off_active",
                "Operator risk-off override active",
                f"Manual risk-off is blocking entries: {runtime.get('operator_risk_off_reason') or 'manual override'}.",
                severity="critical",
                source="capital_governor",
                status="blocked",
                blocking=True,
                metadata={
                    "set_by": runtime.get("operator_risk_off_set_by", ""),
                    "set_at": runtime.get("operator_risk_off_set_at", ""),
                },
            )
        )
        return incidents

    if status == "blocked":
        incidents.append(
            _incident(
                "capital_governor_blocked",
                "Capital governor blocked entries",
                f"Global capital posture is risk-off: {_join_reasons(reasons)}.",
                severity="critical",
                source="capital_governor",
                status=status,
                blocking=True,
                metadata={"reasons": reasons, "metrics": runtime.get("metrics", {})},
            )
        )
    elif status == "caution":
        incidents.append(
            _incident(
                "capital_governor_caution",
                "Capital governor caution",
                f"Global capital posture is constrained: {_join_reasons(reasons)}.",
                severity="warning",
                source="capital_governor",
                status=status,
                blocking=False,
                metadata={"reasons": reasons, "metrics": runtime.get("metrics", {})},
            )
        )
    return incidents


def _collect_adaptive_incidents(adaptive_learning=None) -> List[Dict[str, Any]]:
    incidents: List[Dict[str, Any]] = []
    if not adaptive_learning:
        return incidents

    try:
        payload = adaptive_learning.get_dashboard_payload(limit=10)
    except Exception as exc:
        logger.debug("runtime incident collector adaptive learning error: %s", exc)
        return incidents

    recalibration = payload.get("recalibration", {}) or {}
    demoted_count = _safe_int(recalibration.get("demoted_count", 0), 0)
    transition_count = _safe_int(recalibration.get("transition_count", 0), 0)
    if demoted_count > 0:
        run_id = str(recalibration.get("run_id", "") or "")
        incidents.append(
            _incident(
                f"adaptive_source_demotions:{run_id or demoted_count}",
                "Sources were demoted",
                f"{demoted_count} source(s) were demoted in recalibration {run_id or 'current run'}.",
                severity="warning",
                source="adaptive_learning",
                status="demoted",
                blocking=False,
                metadata={
                    "run_id": run_id,
                    "demoted_count": demoted_count,
                    "transition_count": transition_count,
                    "promoted_count": _safe_int(recalibration.get("promoted_count", 0), 0),
                },
            )
        )
    return incidents


def build_runtime_incident_report(
    *,
    live_trader=None,
    divergence_controller=None,
    capital_governor=None,
    adaptive_learning=None,
    max_items: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a unified runtime incident payload for operators."""
    incident_map: Dict[str, Dict[str, Any]] = {}
    for incident in _collect_live_incidents(live_trader):
        incident_map[incident["key"]] = incident
    for incident in _collect_divergence_incidents(divergence_controller):
        incident_map[incident["key"]] = incident
    for incident in _collect_capital_incidents(capital_governor):
        incident_map[incident["key"]] = incident
    for incident in _collect_adaptive_incidents(adaptive_learning):
        incident_map[incident["key"]] = incident

    incidents = list(incident_map.values())
    incidents.sort(
        key=lambda item: (
            -int(item.get("severity_rank", 0) or 0),
            -int(bool(item.get("blocking", False))),
            item.get("title", ""),
        )
    )

    display_limit = int(
        max_items
        if max_items is not None
        else getattr(config, "RUNTIME_INCIDENT_MAX_ITEMS", 8)
    )
    displayed = incidents[: max(1, display_limit)] if incidents else []

    summary = {
        "total_incidents": len(incidents),
        "displayed_incidents": len(displayed),
        "critical_count": sum(1 for item in incidents if item.get("severity") == "critical"),
        "warning_count": sum(1 for item in incidents if item.get("severity") == "warning"),
        "info_count": sum(1 for item in incidents if item.get("severity") == "info"),
        "blocking_count": sum(1 for item in incidents if item.get("blocking", False)),
        "overall_status": (
            "critical"
            if any(item.get("severity") == "critical" for item in incidents)
            else "warning"
            if any(item.get("severity") == "warning" for item in incidents)
            else "healthy"
        ),
    }

    return {
        "generated_at": utc_now_iso(suffix_z=True),
        "summary": summary,
        "incidents": displayed,
    }
