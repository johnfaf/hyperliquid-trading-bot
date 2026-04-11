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
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import config
from src.data import database as db

logger = logging.getLogger(__name__)

_DB_WRITE_PROBE_CACHE: Dict[str, Any] = {"ts": 0.0, "ok": False, "error": ""}


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
        _DB_WRITE_PROBE_CACHE.update({"ts": now, "ok": True, "error": ""})
        return True, ""
    except Exception as exc:
        error = str(exc)
        _DB_WRITE_PROBE_CACHE.update({"ts": now, "ok": False, "error": error})
        return False, error


def evaluate_readiness(
    container: Optional[Any] = None,
    health_registry: Optional[Any] = None,
    stale_seconds: Optional[int] = None,
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

    # Health registry / subsystem readiness
    subsystem_safe = None
    stale_trading = []
    at_risk_trading = []
    subsystem_states: Dict[str, str] = {}
    if health_registry is not None:
        try:
            stale_map = health_registry.check_stale(timeout_seconds=stale_s)
            statuses = health_registry.get_all()
            subsystem_safe = bool(health_registry.is_all_trading_safe())
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
        subsystem_safe = False
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

    checks["live_requested"] = live_requested
    checks["deployable"] = deployable
    checks["signer_available"] = signer_available
    checks["kill_switch_active"] = kill_switch_active
    checks["live_status_reason"] = live_status_reason or None

    if live_requested:
        if not deployable:
            reasons.append(f"live_not_deployable:{live_status_reason or 'unknown'}")
        if not signer_available:
            reasons.append("missing_agent_wallet_signer")
        if kill_switch_active:
            reason = str(live_stats.get("kill_switch_reason", "") or "active")
            reasons.append(f"kill_switch_active:{reason}")

    ready = bool(db_readable and db_writable and subsystem_safe and not stale_trading)
    live_ready = bool(
        ready
        and live_requested
        and deployable
        and signer_available
        and not kill_switch_active
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
                logger.debug("Runtime incident alert skipped: %s", exc)

        self._last_state = state
        self._last_signature = signature
        return snapshot
