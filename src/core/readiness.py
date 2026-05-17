"""
Runtime readiness evaluation and incident alerting.

Provides a single source of truth for:
- liveness (`/api/health`)
- runtime readiness (`/api/ready`)
- live-trading readiness (`/api/live_ready`)
- transition-based Telegram incident alerts
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import config
from src.data import database as db

logger = logging.getLogger(__name__)

_DB_WRITE_PROBE_CACHE: Dict[str, Any] = {"ts": 0.0, "ok": False, "error": ""}
_DB_AUDIT_CACHE: Dict[str, Any] = {"ts": 0.0, "ok": True, "report": {}, "blockers": []}
_DB_WRITE_PROBE_CACHE_LOCK = threading.Lock()
_DB_AUDIT_CACHE_LOCK = threading.Lock()
_DB_AUDIT_BUILD_LOCK = threading.Lock()
_DB_AUDIT_SAFE_REPAIR_CHECKS = {
    "paper_account_singleton",
    "paper_account_trade_count",
    "paper_account_winning_trades",
    "paper_account_total_pnl",
    "stale_pending_decisions",
    "source_health_history",
    "stale_non_active_regime_history",
}


def _probe_db_readable() -> tuple[bool, str]:
    try:
        with db.get_connection() as conn:
            conn.execute("SELECT 1").fetchone()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _probe_db_writable(ttl_s: Optional[int] = None) -> tuple[bool, str]:
    ttl = max(1, int(ttl_s or config.READINESS_DB_WRITE_TTL_S))
    now = time.time()
    with _DB_WRITE_PROBE_CACHE_LOCK:
        if now - float(_DB_WRITE_PROBE_CACHE.get("ts", 0.0) or 0.0) < ttl:
            return bool(_DB_WRITE_PROBE_CACHE.get("ok", False)), str(
                _DB_WRITE_PROBE_CACHE.get("error", "") or ""
            )

    try:
        with db.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS readiness_probe (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    touched_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO readiness_probe (id, touched_at)
                VALUES (1, ?)
                ON CONFLICT(id) DO UPDATE SET touched_at = excluded.touched_at
                """,
                (datetime.now(timezone.utc).isoformat(),),
            )
        with _DB_WRITE_PROBE_CACHE_LOCK:
            _DB_WRITE_PROBE_CACHE.update({"ts": now, "ok": True, "error": ""})
        return True, ""
    except Exception as exc:
        error = str(exc)
        with _DB_WRITE_PROBE_CACHE_LOCK:
            _DB_WRITE_PROBE_CACHE.update({"ts": now, "ok": False, "error": error})
        return False, error


def _probe_db_audit(ttl_s: Optional[int] = None) -> tuple[bool, Dict[str, Any], list]:
    if not bool(getattr(config, "READINESS_DB_AUDIT_ENABLED", True)):
        return True, {"enabled": False, "ok": True, "finding_count": 0}, []

    ttl = max(5, int(ttl_s or getattr(config, "READINESS_DB_AUDIT_TTL_S", 300)))
    now = time.time()
    with _DB_AUDIT_CACHE_LOCK:
        if now - float(_DB_AUDIT_CACHE.get("ts", 0.0) or 0.0) < ttl:
            return (
                bool(_DB_AUDIT_CACHE.get("ok", True)),
                dict(_DB_AUDIT_CACHE.get("report", {}) or {}),
                list(_DB_AUDIT_CACHE.get("blockers", []) or []),
            )

    with _DB_AUDIT_BUILD_LOCK:
        now = time.time()
        with _DB_AUDIT_CACHE_LOCK:
            if now - float(_DB_AUDIT_CACHE.get("ts", 0.0) or 0.0) < ttl:
                return (
                    bool(_DB_AUDIT_CACHE.get("ok", True)),
                    dict(_DB_AUDIT_CACHE.get("report", {}) or {}),
                    list(_DB_AUDIT_CACHE.get("blockers", []) or []),
                )

        block_severity = str(getattr(config, "READINESS_DB_AUDIT_BLOCK_SEVERITY", "high")).lower()
        try:
            from src.data.db_audit import run_db_audit

            report = run_db_audit(include_candle_cache=True, include_code_scan=False)
            payload = report.to_dict(block_severity=block_severity)
            blockers = [finding.to_dict() for finding in report.findings_at_or_above(block_severity)]

            if (
                blockers
                and bool(getattr(config, "READINESS_DB_AUDIT_AUTO_REPAIR", True))
                and any(str(item.get("check", "")) in _DB_AUDIT_SAFE_REPAIR_CHECKS for item in blockers)
            ):
                try:
                    from src.data.db_audit import run_startup_safe_repair

                    actions = run_startup_safe_repair()
                    report = run_db_audit(include_candle_cache=True, include_code_scan=False)
                    payload = report.to_dict(block_severity=block_severity)
                    payload["auto_repair_actions"] = [action.to_dict() for action in actions]
                    blockers = [
                        finding.to_dict()
                        for finding in report.findings_at_or_above(block_severity)
                    ]
                    if not blockers:
                        logger.info("Readiness DB audit auto-repair cleared blocking findings")
                except Exception as repair_exc:
                    payload["auto_repair_error"] = str(repair_exc)
                    logger.warning("Readiness DB audit auto-repair failed: %s", repair_exc)
            ok = not blockers
        except Exception as exc:
            payload = {
                "enabled": True,
                "ok": False,
                "error": str(exc),
                "finding_count": 1,
            }
            blockers = [
                {
                    "check": "db_audit_runtime",
                    "severity": "critical",
                    "message": "Database audit failed to run.",
                    "details": {"error": str(exc)},
                }
            ]
            ok = False

        with _DB_AUDIT_CACHE_LOCK:
            _DB_AUDIT_CACHE.update(
                {"ts": time.time(), "ok": ok, "report": payload, "blockers": blockers}
            )
    return ok, payload, blockers


def _resolve_health_registry(health_registry: Optional[Any]) -> tuple[Optional[Any], str]:
    if health_registry is not None:
        return health_registry, "provided"
    try:
        from src.core.health_registry import registry as global_registry

        return global_registry, "global"
    except Exception:
        return None, "missing"


def evaluate_readiness(
    container: Optional[Any] = None,
    health_registry: Optional[Any] = None,
    stale_seconds: Optional[int] = None,
    include_db_audit: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate runtime and live-trading readiness using concrete local checks.

    `ready` answers: is the runtime healthy enough to keep serving and trading?
    `live_ready` answers: if live trading is requested, is it safe to deploy now?
    """

    now = datetime.now(timezone.utc)
    stale_s = max(30, int(stale_seconds or config.READINESS_STALE_SECONDS))

    payload: Dict[str, Any] = {
        "timestamp": now.isoformat(),
        "status": "not_ready",
        "ready": False,
        "live_ready": False,
        "reasons": [],
        "checks": {},
    }

    reasons = payload["reasons"]
    checks = payload["checks"]

    # DB probes
    db_readable, db_read_error = _probe_db_readable()
    db_writable, db_write_error = _probe_db_writable()
    checks["db_readable"] = db_readable
    checks["db_writable"] = db_writable
    checks["db_path"] = db.get_db_path()
    if not db_readable:
        reasons.append(f"db_read_failed:{db_read_error[:160]}")
    if not db_writable:
        reasons.append(f"db_write_failed:{db_write_error[:160]}")

    if include_db_audit:
        db_audit_ok, db_audit_report, db_audit_blockers = _probe_db_audit()
    else:
        db_audit_ok, db_audit_report, db_audit_blockers = (
            True,
            {"enabled": False, "ok": True, "skipped": "lightweight_readiness"},
            [],
        )
    checks["db_audit_ok"] = db_audit_ok
    checks["db_audit"] = db_audit_report
    if not db_audit_ok:
        for finding in db_audit_blockers[:5]:
            check = str(finding.get("check", "unknown"))
            severity = str(finding.get("severity", "unknown"))
            reasons.append(f"db_audit_{severity}:{check}")

    # Health registry / subsystem readiness
    health_registry, health_registry_source = _resolve_health_registry(health_registry)
    subsystem_safe = None
    stale_trading = []
    at_risk_trading = []
    subsystem_states: Dict[str, str] = {}
    if health_registry is not None:
        try:
            stale_map = health_registry.check_stale(timeout_seconds=stale_s)
            statuses = health_registry.get_all()
            subsystem_safe = bool(health_registry.is_all_trading_safe())
            checks["health_registry_present"] = True
            checks["health_registry_source"] = health_registry_source
            checks["health_registry_subsystem_count"] = len(statuses)
            if (
                not statuses
                and bool(getattr(config, "READINESS_REQUIRE_HEALTH_REGISTRY", False))
            ):
                subsystem_safe = False
                reasons.append("health_registry_unavailable")
            for name, status in statuses.items():
                subsystem_states[name] = status.state.value
                if status.affects_trading and stale_map.get(name):
                    stale_trading.append(name)
                if status.affects_trading and status.state.value not in {"HEALTHY", "DEGRADED"}:
                    at_risk_trading.append(name)
                if status.affects_trading and not bool(status.dependency_ready):
                    at_risk_trading.append(name)
        except Exception as exc:
            subsystem_safe = False
            reasons.append(f"health_registry_error:{str(exc)[:160]}")
    else:
        require_registry = bool(getattr(config, "READINESS_REQUIRE_HEALTH_REGISTRY", False))
        subsystem_safe = not require_registry
        checks["health_registry_present"] = False
        checks["health_registry_source"] = "missing"
        checks["health_registry_subsystem_count"] = 0
        if require_registry:
            reasons.append("health_registry_unavailable")

    checks["subsystems_safe"] = bool(subsystem_safe)
    checks["stale_trading_subsystems"] = sorted(set(stale_trading))
    checks["at_risk_trading_subsystems"] = sorted(set(at_risk_trading))
    checks["subsystem_states"] = subsystem_states
    if not subsystem_safe:
        reasons.append("trading_subsystems_not_safe")
    if stale_trading:
        reasons.append("stale_trading_heartbeats")

    # Live-trader checks
    live_trader = getattr(container, "live_trader", None) if container is not None else None
    live_requested = False
    live_stats: Dict[str, Any] = {}
    if live_trader is not None:
        try:
            live_stats = live_trader.get_stats() or {}
        except Exception as exc:
            live_stats = {"error": str(exc)}

    live_requested = bool(live_stats.get("live_enabled", False))
    deployable = bool(live_stats.get("deployable", False))
    signer_available = bool(live_stats.get("signer_available", False))
    kill_switch_active = bool(live_stats.get("kill_switch_active", False))
    live_status_reason = str(live_stats.get("status_reason", "") or "")
    wallet_balance = live_stats.get("wallet_balance", {}) if isinstance(live_stats, dict) else {}
    free_margin = None
    if isinstance(wallet_balance, dict):
        free_margin = wallet_balance.get("free_margin")
    if free_margin is None:
        free_margin = live_stats.get("free_margin")
    try:
        free_margin = float(free_margin) if free_margin is not None else None
    except (TypeError, ValueError):
        free_margin = None

    checks["live_requested"] = live_requested
    checks["deployable"] = deployable
    checks["signer_available"] = signer_available
    checks["kill_switch_active"] = kill_switch_active
    checks["live_status_reason"] = live_status_reason or None
    checks["free_margin"] = free_margin

    if live_requested:
        if not deployable:
            reasons.append(f"live_not_deployable:{live_status_reason or 'unknown'}")
        if not signer_available:
            reasons.append("missing_agent_wallet_signer")
        if kill_switch_active:
            reason = str(live_stats.get("kill_switch_reason", "") or "active")
            reasons.append(f"kill_switch_active:{reason}")
        if free_margin is not None and free_margin <= 0:
            reasons.append("live_free_margin_zero")

    ready = bool(db_readable and db_writable and db_audit_ok and subsystem_safe and not stale_trading)
    live_ready = bool(
        ready
        and live_requested
        and deployable
        and signer_available
        and not kill_switch_active
        and not (free_margin is not None and free_margin <= 0)
    )

    payload["ready"] = ready
    payload["live_ready"] = live_ready
    payload["status"] = "ready" if ready else "not_ready"

    # Deduplicate while preserving order.
    deduped = []
    seen = set()
    for reason in reasons:
        if reason and reason not in seen:
            seen.add(reason)
            deduped.append(reason)
    payload["reasons"] = deduped
    return payload


class RuntimeIncidentMonitor:
    """Send Telegram alerts when readiness transitions or materially changes."""

    def __init__(self, cooldown_s: Optional[int] = None):
        self.cooldown_s = max(30, int(cooldown_s or config.READINESS_ALERT_COOLDOWN_S))
        self._last_state: Optional[tuple[bool, bool]] = None
        self._last_signature = ""
        self._last_alert_ts = 0.0
        self._initialized = False

    def evaluate_and_alert(
        self,
        container: Optional[Any] = None,
        health_registry: Optional[Any] = None,
    ) -> Dict[str, Any]:
        snapshot = evaluate_readiness(container=container, health_registry=health_registry)
        state = (bool(snapshot.get("ready")), bool(snapshot.get("live_ready")))
        signature = "|".join(snapshot.get("reasons", []))
        now = time.time()

        if not self._initialized:
            self._initialized = True
            self._last_state = state
            self._last_signature = signature
            return snapshot

        should_alert = False
        resolved = False
        if state != self._last_state:
            should_alert = True
            resolved = state[0] and (
                state[1] or not bool(snapshot.get("checks", {}).get("live_requested", False))
            )
        elif not state[0] and signature != self._last_signature and (
            now - self._last_alert_ts
        ) >= self.cooldown_s:
            should_alert = True

        if should_alert:
            try:
                from src.notifications import telegram_bot as tg

                if tg.is_configured():
                    tg.notify_runtime_incident(snapshot, resolved=resolved)
                    self._last_alert_ts = now
            except Exception as exc:
                logger.warning("Runtime incident alert skipped: %s", exc)

        self._last_state = state
        self._last_signature = signature
        return snapshot
