"""Runtime database audit, repair, and readiness checks.

The audit turns silent database drift into a structured report that can be
exposed by readiness endpoints and operator CLI commands before capital is
scaled.  Safe, operator-invoked repairs live alongside the audit so the bot can
repair stale local state without guessing about irreversible trade outcomes.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import config
from src.data import database as db


SEVERITY_ORDER = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


@dataclass(frozen=True)
class DbAuditFinding:
    check: str
    severity: str
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DbAuditReport:
    timestamp: str
    backend: str
    db_path: str
    checks: dict[str, Any]
    findings: list[DbAuditFinding]

    @property
    def ok(self) -> bool:
        return not self.findings_at_or_above("high")

    def findings_at_or_above(self, severity: str) -> list[DbAuditFinding]:
        floor = SEVERITY_ORDER.get(str(severity or "high").lower(), SEVERITY_ORDER["high"])
        return [
            finding
            for finding in self.findings
            if SEVERITY_ORDER.get(finding.severity, 0) >= floor
        ]

    def to_dict(self, *, block_severity: str = "high") -> dict[str, Any]:
        blocking = self.findings_at_or_above(block_severity)
        return {
            "timestamp": self.timestamp,
            "backend": self.backend,
            "db_path": self.db_path,
            "ok": not blocking,
            "block_severity": block_severity,
            "finding_count": len(self.findings),
            "blocking_finding_count": len(blocking),
            "checks": self.checks,
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class DbRepairAction:
    action: str
    status: str
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DbRepairReport:
    timestamp: str
    backend: str
    db_path: str
    pre_audit: DbAuditReport
    post_audit: DbAuditReport
    actions: list[DbRepairAction]

    def to_dict(self, *, block_severity: str = "high") -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "backend": self.backend,
            "db_path": self.db_path,
            "pre_audit": self.pre_audit.to_dict(block_severity=block_severity),
            "post_audit": self.post_audit.to_dict(block_severity=block_severity),
            "actions": [action.to_dict() for action in self.actions],
            "failed_actions": [
                action.to_dict() for action in self.actions if str(action.status).lower() == "failed"
            ],
        }


def severity_at_or_above(severity: str, floor: str) -> bool:
    return SEVERITY_ORDER.get(str(severity or "").lower(), 0) >= SEVERITY_ORDER.get(
        str(floor or "high").lower(),
        SEVERITY_ORDER["high"],
    )


def _row_get(row: Any, key: str, idx: int = 0, default: Any = None) -> Any:
    if row is None:
        return default
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[key]
    except Exception:
        pass
    try:
        return row[idx]
    except Exception:
        return default


def _scalar(row: Any, key: str = "c", default: Any = 0) -> Any:
    return _row_get(row, key, 0, default)


def _parse_dt(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    raw = str(value).strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(raw[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_exists(conn: Any, name: str) -> bool:
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        return row is not None
    except Exception:
        return False


def _count(conn: Any, table: str, where: str = "", params: Iterable[Any] = ()) -> int:
    if not _table_exists(conn, table):
        return 0
    sql = f"SELECT COUNT(*) AS c FROM {table}"
    if where:
        sql += f" WHERE {where}"
    row = conn.execute(sql, tuple(params)).fetchone()
    return int(_scalar(row, "c", 0) or 0)


def _add(
    findings: list[DbAuditFinding],
    check: str,
    severity: str,
    message: str,
    **details: Any,
) -> None:
    findings.append(
        DbAuditFinding(
            check=check,
            severity=str(severity).lower(),
            message=message,
            details=details,
        )
    )


def _record_action(
    actions: list[DbRepairAction],
    action: str,
    status: str,
    message: str,
    **details: Any,
) -> None:
    actions.append(
        DbRepairAction(
            action=action,
            status=str(status).lower(),
            message=message,
            details=details,
        )
    )


def _sqlite_integrity_checks(conn: Any, findings: list[DbAuditFinding], checks: dict[str, Any]) -> None:
    if getattr(conn, "backend", "sqlite") != "sqlite":
        checks["sqlite_integrity"] = {"skipped": True, "reason": "non_sqlite_backend"}
        return

    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        integrity = str(_row_get(row, "integrity_check", 0, "") or "")
        checks["sqlite_integrity"] = integrity
        if integrity.lower() != "ok":
            _add(
                findings,
                "sqlite_integrity",
                "critical",
                "SQLite integrity_check did not return ok.",
                result=integrity,
            )
    except Exception as exc:
        checks["sqlite_integrity"] = {"error": str(exc)}
        _add(
            findings,
            "sqlite_integrity",
            "critical",
            "Could not run SQLite integrity_check.",
            error=str(exc),
        )

    try:
        row = conn.execute("PRAGMA foreign_keys").fetchone()
        enabled = int(_scalar(row, "foreign_keys", 0) or 0)
        checks["sqlite_foreign_keys_enabled"] = bool(enabled)
        if not enabled:
            _add(
                findings,
                "sqlite_foreign_keys",
                "high",
                "SQLite foreign-key enforcement is disabled for this connection.",
            )
    except Exception as exc:
        checks["sqlite_foreign_keys_enabled"] = {"error": str(exc)}

    try:
        rows = conn.execute("PRAGMA foreign_key_check").fetchall()
        violations = [dict(row) if hasattr(row, "keys") else tuple(row) for row in rows]
        checks["sqlite_foreign_key_violations"] = {
            "count": len(violations),
            "examples": violations[:10],
        }
        if violations:
            _add(
                findings,
                "sqlite_foreign_key_violations",
                "high",
                "Foreign-key violations are already persisted in the database.",
                count=len(violations),
                examples=violations[:10],
            )
    except Exception as exc:
        checks["sqlite_foreign_key_violations"] = {"error": str(exc)}


def _schema_checks(conn: Any, findings: list[DbAuditFinding], checks: dict[str, Any]) -> None:
    required_tables = [
        "traders",
        "strategies",
        "paper_trades",
        "paper_account",
        "audit_trail",
        "bot_state",
    ]
    learning_tables = [
        "continuous_learning_policies",
        "source_inventory",
        "data_source_health_history",
        "decision_snapshots",
    ]
    missing_required = [name for name in required_tables if not _table_exists(conn, name)]
    missing_learning = [name for name in learning_tables if not _table_exists(conn, name)]
    checks["schema"] = {
        "missing_required_tables": missing_required,
        "missing_learning_tables": missing_learning,
        "schema_migrations_present": _table_exists(conn, "schema_migrations"),
    }
    if missing_required:
        _add(
            findings,
            "schema_required_tables",
            "critical",
            "Required runtime tables are missing.",
            missing=missing_required,
        )
    if missing_learning:
        _add(
            findings,
            "schema_learning_tables",
            "medium",
            "Learning/source observability tables are missing.",
            missing=missing_learning,
        )
    if not checks["schema"]["schema_migrations_present"]:
        severity = "medium" if db.get_backend_name() == "postgres" else "low"
        _add(
            findings,
            "schema_migrations",
            severity,
            "schema_migrations table is absent, so DB schema drift is harder to detect.",
        )

    if getattr(conn, "backend", "sqlite") == "sqlite":
        try:
            row = conn.execute("PRAGMA user_version").fetchone()
            checks["schema"]["sqlite_user_version"] = int(_scalar(row, "user_version", 0) or 0)
        except Exception:
            checks["schema"]["sqlite_user_version"] = None


def _paper_account_checks(conn: Any, findings: list[DbAuditFinding], checks: dict[str, Any]) -> None:
    if not (_table_exists(conn, "paper_account") and _table_exists(conn, "paper_trades")):
        return

    account = conn.execute("SELECT * FROM paper_account WHERE id = 1").fetchone()
    closed = conn.execute(
        """
        SELECT COUNT(*) AS c,
               COALESCE(SUM(pnl), 0) AS total_pnl,
               COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) AS wins
        FROM paper_trades
        WHERE LOWER(COALESCE(status, '')) = 'closed'
        """
    ).fetchone()
    closed_count = int(_scalar(closed, "c", 0) or 0)
    closed_pnl = float(_row_get(closed, "total_pnl", 1, 0.0) or 0.0)
    closed_wins = int(_row_get(closed, "wins", 2, 0) or 0)
    checks["paper_account"] = {
        "present": account is not None,
        "closed_count": closed_count,
        "closed_pnl": round(closed_pnl, 8),
        "closed_wins": closed_wins,
    }
    if account is None:
        _add(
            findings,
            "paper_account_singleton",
            "high",
            "paper_account singleton row id=1 is missing.",
        )
        return

    account_total_trades = int(_row_get(account, "total_trades", default=0) or 0)
    account_wins = int(_row_get(account, "winning_trades", default=0) or 0)
    account_pnl = float(_row_get(account, "total_pnl", default=0.0) or 0.0)
    checks["paper_account"].update(
        {
            "account_total_trades": account_total_trades,
            "account_total_pnl": round(account_pnl, 8),
            "account_winning_trades": account_wins,
        }
    )
    if account_total_trades != closed_count:
        _add(
            findings,
            "paper_account_trade_count",
            "high",
            "paper_account.total_trades does not match closed paper trades.",
            account_total_trades=account_total_trades,
            closed_count=closed_count,
        )
    if account_wins != closed_wins:
        _add(
            findings,
            "paper_account_winning_trades",
            "medium",
            "paper_account.winning_trades does not match closed winning trades.",
            account_winning_trades=account_wins,
            closed_winning_trades=closed_wins,
        )
    if abs(account_pnl - closed_pnl) > 0.01:
        _add(
            findings,
            "paper_account_total_pnl",
            "high",
            "paper_account.total_pnl does not match closed paper-trade PnL.",
            account_total_pnl=account_pnl,
            closed_pnl=closed_pnl,
        )


def _paper_trade_checks(conn: Any, findings: list[DbAuditFinding], checks: dict[str, Any]) -> set[str]:
    active_coins: set[str] = set()
    if not _table_exists(conn, "paper_trades"):
        return active_coins

    status_rows = conn.execute(
        """
        SELECT LOWER(COALESCE(status, '')) AS status, COUNT(*) AS c
        FROM paper_trades
        GROUP BY LOWER(COALESCE(status, ''))
        """
    ).fetchall()
    status_counts = {
        str(_row_get(row, "status", 0, "") or "unknown"): int(_row_get(row, "c", 1, 0) or 0)
        for row in status_rows
    }

    bad_numeric = conn.execute(
        """
        SELECT COUNT(*) AS c
        FROM paper_trades
        WHERE entry_price IS NULL OR entry_price <= 0
           OR size IS NULL OR size <= 0
           OR leverage IS NULL OR leverage <= 0
        """
    ).fetchone()
    bad_numeric_count = int(_scalar(bad_numeric, "c", 0) or 0)

    missing_protection_rows = conn.execute(
        """
        SELECT id, coin, side, entry_price, size, stop_loss, take_profit
        FROM paper_trades
        WHERE LOWER(COALESCE(status, '')) = 'open'
          AND (
              stop_loss IS NULL OR take_profit IS NULL
              OR stop_loss <= 0 OR take_profit <= 0
          )
        ORDER BY id DESC
        LIMIT 10
        """
    ).fetchall()
    missing_protection = [dict(row) for row in missing_protection_rows]

    open_coin_rows = conn.execute(
        """
        SELECT DISTINCT coin
        FROM paper_trades
        WHERE LOWER(COALESCE(status, '')) = 'open'
          AND coin IS NOT NULL AND coin != ''
        """
    ).fetchall()
    active_coins.update(str(_row_get(row, "coin", 0, "") or "").upper() for row in open_coin_rows)
    active_coins.discard("")

    dup_rows = conn.execute(
        """
        SELECT coin, side, COUNT(*) AS c
        FROM paper_trades
        WHERE LOWER(COALESCE(status, '')) = 'open'
        GROUP BY coin, side
        HAVING COUNT(*) > 1
        ORDER BY c DESC
        """
    ).fetchall()
    duplicate_same_side = [dict(row) for row in dup_rows]

    conflict_rows = conn.execute(
        """
        SELECT coin, COUNT(DISTINCT LOWER(COALESCE(side, ''))) AS side_count, COUNT(*) AS c
        FROM paper_trades
        WHERE LOWER(COALESCE(status, '')) = 'open'
        GROUP BY coin
        HAVING COUNT(DISTINCT LOWER(COALESCE(side, ''))) > 1
        """
    ).fetchall()
    conflicting_sides = [dict(row) for row in conflict_rows]

    checks["paper_trades"] = {
        "status_counts": status_counts,
        "bad_numeric_count": bad_numeric_count,
        "open_missing_protection_count": len(missing_protection),
        "open_missing_protection_examples": missing_protection,
        "duplicate_same_side_open": duplicate_same_side,
        "conflicting_open_sides": conflicting_sides,
        "active_open_coins": sorted(active_coins),
    }
    if bad_numeric_count:
        _add(
            findings,
            "paper_trade_numeric_values",
            "high",
            "Paper trades contain invalid entry, size, or leverage values.",
            count=bad_numeric_count,
        )
    if missing_protection:
        _add(
            findings,
            "open_trades_missing_protection",
            "high",
            "Open paper trades exist without valid stop-loss and take-profit values.",
            count=len(missing_protection),
            examples=missing_protection,
        )
    if duplicate_same_side:
        _add(
            findings,
            "duplicate_same_side_open_trades",
            "medium",
            "Multiple open paper trades exist for the same coin and side.",
            examples=duplicate_same_side,
        )
    if conflicting_sides:
        _add(
            findings,
            "conflicting_open_trade_sides",
            "high",
            "Open paper trades contain both long and short exposure for the same coin.",
            examples=conflicting_sides,
        )
    return active_coins


def _decision_journal_checks(conn: Any, findings: list[DbAuditFinding], checks: dict[str, Any]) -> set[str]:
    active_coins: set[str] = set()
    if not _table_exists(conn, "decision_snapshots"):
        return active_coins

    pending_rows = conn.execute(
        """
        SELECT decision_id, created_at, updated_at, coin, final_status, firewall_decision
        FROM decision_snapshots
        WHERE COALESCE(final_status, '') = 'candidate'
           OR COALESCE(firewall_decision, '') = 'pending'
        ORDER BY created_at ASC
        LIMIT 1000
        """
    ).fetchall()
    now = datetime.now(timezone.utc)
    stale_minutes = float(getattr(config, "DB_AUDIT_PENDING_DECISION_MAX_AGE_MINUTES", 30.0))
    stale_pending = []
    for row in pending_rows:
        coin = str(_row_get(row, "coin", 3, "") or "").upper()
        if coin:
            active_coins.add(coin)
        created = _parse_dt(_row_get(row, "created_at", 1, None))
        if created and (now - created).total_seconds() > stale_minutes * 60:
            stale_pending.append(dict(row))

    terminal_decisions = conn.execute(
        """
        SELECT COUNT(*) AS c
        FROM decision_snapshots
        WHERE COALESCE(firewall_decision, '') IN ('approved', 'rejected')
           OR COALESCE(final_status, '') IN (
                'approved', 'rejected',
                'firewall_validation_rejected',
                'firewall_prescreen_rejected',
                'firewall_prescreen_approved'
           )
        """
    ).fetchone()
    terminal_decision_count = int(_scalar(terminal_decisions, "c", 0) or 0)
    terminal_audit_count = 0
    linkable_terminal_audit_count = 0
    unresolved_linked_audits: list[dict[str, Any]] = []
    if _table_exists(conn, "audit_trail"):
        row = conn.execute(
            """
            SELECT COUNT(*) AS c
            FROM audit_trail
            WHERE action IN ('signal_approved', 'signal_rejected')
            """
        ).fetchone()
        terminal_audit_count = int(_scalar(row, "c", 0) or 0)
        audit_rows = conn.execute(
            """
            SELECT timestamp, action, coin, side, source, details
            FROM audit_trail
            WHERE action IN ('signal_approved', 'signal_rejected')
            ORDER BY timestamp DESC
            LIMIT 1000
            """
        ).fetchall()
        for audit_row in audit_rows:
            details = _row_get(audit_row, "details", 5, "{}")
            try:
                payload = json.loads(details or "{}")
            except Exception:
                payload = {}
            decision_key = str(payload.get("decision_id") or payload.get("signal_id") or "").strip()
            if not decision_key:
                continue
            linkable_terminal_audit_count += 1
            matched = conn.execute(
                """
                SELECT decision_id, final_status, firewall_decision
                FROM decision_snapshots
                WHERE decision_id = ? OR signal_id = ?
                LIMIT 1
                """,
                (decision_key, decision_key),
            ).fetchone()
            if matched is None:
                unresolved_linked_audits.append(
                    {
                        "timestamp": _row_get(audit_row, "timestamp", 0, ""),
                        "action": _row_get(audit_row, "action", 1, ""),
                        "coin": _row_get(audit_row, "coin", 2, ""),
                        "decision_key": decision_key,
                    }
                )
                continue
            matched_status = str(_row_get(matched, "final_status", 1, "") or "").lower()
            matched_firewall = str(_row_get(matched, "firewall_decision", 2, "") or "").lower()
            if matched_status == "candidate" or matched_firewall == "pending":
                unresolved_linked_audits.append(
                    {
                        "timestamp": _row_get(audit_row, "timestamp", 0, ""),
                        "action": _row_get(audit_row, "action", 1, ""),
                        "coin": _row_get(audit_row, "coin", 2, ""),
                        "decision_key": decision_key,
                        "matched_decision_id": _row_get(matched, "decision_id", 0, ""),
                        "final_status": matched_status,
                        "firewall_decision": matched_firewall,
                    }
                )

    checks["decision_journal"] = {
        "pending_count": len(pending_rows),
        "stale_pending_count": len(stale_pending),
        "terminal_decision_count": terminal_decision_count,
        "terminal_audit_count": terminal_audit_count,
        "linkable_terminal_audit_count": linkable_terminal_audit_count,
        "unresolved_linked_terminal_audits": unresolved_linked_audits[:10],
        "stale_pending_examples": stale_pending[:10],
    }
    if stale_pending:
        _add(
            findings,
            "stale_pending_decisions",
            "medium",
            "Decision snapshots are still pending past the freshness threshold.",
            count=len(stale_pending),
            threshold_minutes=stale_minutes,
            examples=stale_pending[:10],
        )
    if unresolved_linked_audits:
        _add(
            findings,
            "decision_audit_mismatch",
            "high",
            "Audit trail has terminal signal events that decision_snapshots do not reflect.",
            terminal_audit_count=terminal_audit_count,
            terminal_decision_count=terminal_decision_count,
            linkable_terminal_audit_count=linkable_terminal_audit_count,
            unresolved_linked_count=len(unresolved_linked_audits),
            examples=unresolved_linked_audits[:10],
        )
    elif terminal_audit_count > terminal_decision_count and pending_rows and linkable_terminal_audit_count == 0:
        _add(
            findings,
            "decision_audit_unlinked_history",
            "medium",
            "Audit trail contains terminal signal events with no stable decision ids, so older history cannot be fully reconciled.",
            terminal_audit_count=terminal_audit_count,
            terminal_decision_count=terminal_decision_count,
            pending_count=len(pending_rows),
        )
    return active_coins


def _source_health_checks(conn: Any, findings: list[DbAuditFinding], checks: dict[str, Any]) -> None:
    if not _table_exists(conn, "source_inventory"):
        return

    inventory_rows = conn.execute(
        """
        SELECT source_name, required, expected_freshness_seconds
        FROM source_inventory
        ORDER BY source_name
        """
    ).fetchall()
    inventory = [dict(row) for row in inventory_rows]
    checks["source_inventory"] = {
        "count": len(inventory),
        "required_count": sum(1 for row in inventory if bool(row.get("required"))),
    }
    if not inventory:
        _add(
            findings,
            "source_inventory_empty",
            "medium",
            "source_inventory is empty, so data-source requirements are unknown.",
        )
        return

    if not _table_exists(conn, "data_source_health_history"):
        return

    health_count = _count(conn, "data_source_health_history")
    checks["source_health"] = {"row_count": health_count, "missing_required": [], "stale": []}
    if health_count == 0:
        _add(
            findings,
            "source_health_history_empty",
            "medium",
            "No data-source health snapshots have been persisted.",
        )
        return

    stale_multiplier = float(getattr(config, "DB_AUDIT_SOURCE_STALE_MULTIPLIER", 2.0))
    for source in inventory:
        name = str(source.get("source_name") or "")
        if not name:
            continue
        latest = conn.execute(
            """
            SELECT observed_at, status, freshness_seconds, reason
            FROM data_source_health_history
            WHERE source_name = ?
            ORDER BY observed_at DESC
            LIMIT 1
            """,
            (name,),
        ).fetchone()
        required = bool(source.get("required"))
        if latest is None:
            if required:
                checks["source_health"]["missing_required"].append(name)
            continue
        status = str(_row_get(latest, "status", 1, "") or "").upper()
        if required and status in {"DOWN", "FAILED", "UNKNOWN"}:
            _add(
                findings,
                "required_source_down",
                "high",
                "A required data source is marked down or unknown.",
                source=name,
                status=status,
                reason=_row_get(latest, "reason", 3, ""),
            )
        expected = float(source.get("expected_freshness_seconds") or 0)
        freshness = _row_get(latest, "freshness_seconds", 2, None)
        try:
            freshness_f = float(freshness) if freshness is not None else None
        except (TypeError, ValueError):
            freshness_f = None
        if expected > 0 and freshness_f is not None and freshness_f > expected * stale_multiplier:
            checks["source_health"]["stale"].append(
                {
                    "source_name": name,
                    "freshness_seconds": freshness_f,
                    "expected_freshness_seconds": expected,
                }
            )

    if checks["source_health"]["missing_required"]:
        _add(
            findings,
            "required_source_health_missing",
            "medium",
            "Required sources have no persisted health snapshot.",
            sources=checks["source_health"]["missing_required"],
        )
    if checks["source_health"]["stale"]:
        _add(
            findings,
            "stale_source_health",
            "medium",
            "Persisted source-health freshness exceeds expected thresholds.",
            examples=checks["source_health"]["stale"][:10],
        )


def _historical_data_checks(conn: Any, findings: list[DbAuditFinding], checks: dict[str, Any]) -> None:
    tables = [
        "funding_history",
        "open_interest_history",
        "options_summary_history",
        "polymarket_markets",
        "polymarket_market_snapshots",
        "polymarket_trades",
    ]
    counts = {table: _count(conn, table) for table in tables if _table_exists(conn, table)}
    checks["historical_sources"] = counts
    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        _add(
            findings,
            "historical_source_tables_empty",
            "medium",
            "Historical source tables exist but have no persisted rows.",
            empty_tables=empty,
        )


def _regime_checks(
    conn: Any,
    findings: list[DbAuditFinding],
    checks: dict[str, Any],
    active_coins: set[str],
) -> None:
    if not _table_exists(conn, "regime_history"):
        return

    rows = conn.execute(
        """
        SELECT coin, MAX(timestamp) AS latest_ts, COUNT(*) AS c
        FROM regime_history
        GROUP BY coin
        ORDER BY coin
        """
    ).fetchall()
    now = datetime.now(timezone.utc)
    max_age_hours = float(getattr(config, "DB_AUDIT_REGIME_MAX_AGE_HOURS", 24.0))
    entries = []
    stale_active = []
    stale_other = []
    for row in rows:
        coin = str(_row_get(row, "coin", 0, "") or "").upper()
        latest = _parse_dt(_row_get(row, "latest_ts", 1, None))
        age_hours = None
        if latest:
            age_hours = max(0.0, (now - latest).total_seconds() / 3600.0)
        item = {
            "coin": coin,
            "latest_ts": str(_row_get(row, "latest_ts", 1, "") or ""),
            "rows": int(_row_get(row, "c", 2, 0) or 0),
            "age_hours": round(age_hours, 2) if age_hours is not None else None,
        }
        entries.append(item)
        if age_hours is not None and age_hours > max_age_hours:
            if coin in active_coins:
                stale_active.append(item)
            else:
                stale_other.append(item)
    checks["regime_history"] = {
        "coin_count": len(entries),
        "entries": entries,
        "stale_active": stale_active,
        "stale_other_count": len(stale_other),
        "max_age_hours": max_age_hours,
    }
    if not entries:
        _add(
            findings,
            "regime_history_empty",
            "medium",
            "regime_history exists but contains no rows.",
        )
    if stale_active:
        _add(
            findings,
            "stale_active_regime_history",
            "high",
            "Active/open coins have stale regime_history rows.",
            threshold_hours=max_age_hours,
            examples=stale_active[:10],
        )
    elif stale_other:
        _add(
            findings,
            "stale_regime_history",
            "medium",
            "Some non-active coins have stale regime_history rows.",
            threshold_hours=max_age_hours,
            examples=stale_other[:10],
        )


def _resolve_candle_cache_path() -> Optional[Path]:
    candidates = [
        Path(db.get_db_path()).with_name("candle_cache.db"),
        Path.cwd() / "data" / "candle_cache.db",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _candle_cache_checks(
    findings: list[DbAuditFinding],
    checks: dict[str, Any],
    active_coins: set[str],
) -> None:
    path = _resolve_candle_cache_path()
    if path is None:
        return
    if not path.exists():
        checks["candle_cache"] = {"path": str(path), "exists": False}
        _add(
            findings,
            "candle_cache_missing",
            "medium",
            "Candle cache database is missing.",
            path=str(path),
        )
        return

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity.lower() != "ok":
            _add(
                findings,
                "candle_cache_integrity",
                "critical",
                "Candle cache integrity_check did not return ok.",
                result=integrity,
            )
        table_row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='candles'"
        ).fetchone()
        if not table_row:
            checks["candle_cache"] = {"path": str(path), "exists": True, "candles_table": False}
            _add(
                findings,
                "candle_cache_schema",
                "medium",
                "Candle cache exists but candles table is missing.",
                path=str(path),
            )
            return
        summary = conn.execute(
            """
            SELECT coin, timeframe, COUNT(*) AS c,
                   MIN(timestamp_ms) AS min_ts,
                   MAX(timestamp_ms) AS max_ts
            FROM candles
            GROUP BY coin, timeframe
            ORDER BY coin, timeframe
            """
        ).fetchall()
        entries = [dict(row) for row in summary]
        cached_coins = {str(row["coin"]).upper() for row in entries}
        missing_active = sorted(coin for coin in active_coins if coin and coin not in cached_coins)
        min_coin_count = int(getattr(config, "DB_AUDIT_MIN_CANDLE_COINS", 2))
        checks["candle_cache"] = {
            "path": str(path),
            "exists": True,
            "integrity": integrity,
            "series": entries,
            "unique_coins": sorted(cached_coins),
            "missing_active_coins": missing_active,
        }
        if len(cached_coins) < min_coin_count:
            _add(
                findings,
                "candle_cache_low_coin_coverage",
                "medium",
                "Candle cache has fewer unique coins than the configured minimum.",
                unique_coins=sorted(cached_coins),
                minimum=min_coin_count,
            )
        if missing_active:
            _add(
                findings,
                "candle_cache_missing_active_coins",
                "high",
                "Open/active coins are missing from the candle cache.",
                missing_active_coins=missing_active,
            )
    finally:
        conn.close()


def _dualwrite_checks(findings: list[DbAuditFinding], checks: dict[str, Any]) -> None:
    backend = db.get_backend_name()
    if backend != "dualwrite":
        checks["dualwrite"] = {"active": False}
        return
    try:
        window_s = float(getattr(config, "DB_AUDIT_DUALWRITE_HEALTH_WINDOW_S", 300.0))
        max_failures = int(getattr(config, "DB_AUDIT_DUALWRITE_MAX_FAILURES", 5))
        healthy = db.dualwrite_is_healthy(window_s=window_s, max_failures=max_failures)
        stats = db.get_dualwrite_stats()
        checks["dualwrite"] = {
            "active": True,
            "healthy": healthy,
            "window_s": window_s,
            "max_failures": max_failures,
            "stats": stats,
        }
        if not healthy:
            _add(
                findings,
                "dualwrite_unhealthy",
                "critical",
                "Dualwrite Postgres mirror has recent failures.",
                stats=stats,
                window_s=window_s,
                max_failures=max_failures,
            )
    except Exception as exc:
        checks["dualwrite"] = {"active": True, "error": str(exc)}
        _add(
            findings,
            "dualwrite_health_check_failed",
            "high",
            "Could not evaluate dualwrite health.",
            error=str(exc),
        )


def _app_structure_checks(findings: list[DbAuditFinding], checks: dict[str, Any]) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    matches: list[dict[str, Any]] = []
    if not src_root.exists():
        checks["app_structure"] = {"direct_sqlite_consumers": []}
        return

    ignored = {
        str((src_root / "data" / "db" / "router.py").resolve()),
        str((src_root / "backtest" / "data_fetcher.py").resolve()),
    }
    patterns = (
        "sqlite3.connect(config.DB_PATH",
        "sqlite3.connect(db_path",
        "sqlite3.connect(_DB_PATH",
    )
    for path in src_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        resolved = str(path.resolve())
        if resolved in ignored:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for idx, line in enumerate(lines, start=1):
            if any(pattern in line for pattern in patterns):
                matches.append(
                    {
                        "path": str(path.relative_to(repo_root)),
                        "line": idx,
                        "text": line.strip()[:160],
                    }
                )
    checks["app_structure"] = {"direct_sqlite_consumers": matches}
    if matches:
        _add(
            findings,
            "direct_sqlite_consumers",
            "medium",
            "Some modules still connect directly to SQLite and can bypass DB_BACKEND.",
            examples=matches[:20],
        )


def _merge_metadata_blob(raw: Any, extra: dict[str, Any]) -> str:
    if isinstance(raw, dict):
        payload = dict(raw)
    else:
        try:
            payload = json.loads(raw or "{}")
            if not isinstance(payload, dict):
                payload = {}
        except Exception:
            payload = {}
    payload.update(extra or {})
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _repair_schema_migrations_table(actions: list[DbRepairAction]) -> None:
    try:
        with db.get_connection() as conn:
            if getattr(conn, "backend", "sqlite") != "sqlite":
                _record_action(
                    actions,
                    "schema_migrations_table",
                    "skipped",
                    "schema_migrations bootstrap is only needed for SQLite local state.",
                    backend=getattr(conn, "backend", "unknown"),
                )
                return
            if _table_exists(conn, "schema_migrations"):
                _record_action(
                    actions,
                    "schema_migrations_table",
                    "skipped",
                    "schema_migrations table already exists.",
                )
                return
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    filename TEXT NOT NULL DEFAULT '',
                    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            _record_action(
                actions,
                "schema_migrations_table",
                "applied",
                "Created SQLite schema_migrations table for local schema tracking.",
            )
    except Exception as exc:
        _record_action(
            actions,
            "schema_migrations_table",
            "failed",
            "Could not create schema_migrations table.",
            error=str(exc),
        )


def _repair_orphan_position_snapshot_parents(actions: list[DbRepairAction]) -> None:
    try:
        with db.get_connection() as conn:
            if not _table_exists(conn, "position_snapshots") or not _table_exists(conn, "traders"):
                _record_action(
                    actions,
                    "orphan_position_snapshot_parents",
                    "skipped",
                    "position_snapshots/traders tables are not present.",
                )
                return
            rows = conn.execute(
                """
                SELECT trader_address, COUNT(*) AS snapshot_count,
                       MIN(timestamp) AS first_seen,
                       MAX(timestamp) AS last_seen
                FROM position_snapshots
                WHERE trader_address NOT IN (SELECT address FROM traders)
                GROUP BY trader_address
                ORDER BY trader_address
                """
            ).fetchall()
            if not rows:
                _record_action(
                    actions,
                    "orphan_position_snapshot_parents",
                    "skipped",
                    "No orphan position snapshots were found.",
                )
                return
            for row in rows:
                address = str(_row_get(row, "trader_address", 0, "") or "").strip()
                if not address:
                    continue
                metadata = json.dumps(
                    {
                        "auto_repaired": True,
                        "repair_reason": "orphan_position_snapshot_parent",
                        "placeholder": True,
                        "snapshot_count": int(_row_get(row, "snapshot_count", 1, 0) or 0),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                conn.execute(
                    """
                    INSERT INTO traders
                    (address, first_seen, last_updated, total_pnl, roi_pct,
                     account_value, win_rate, trade_count, active, metadata)
                    VALUES (?, ?, ?, 0, 0, 0, 0, 0, 0, ?)
                    ON CONFLICT(address) DO UPDATE SET
                        active = 0,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        address,
                        str(_row_get(row, "first_seen", 2, "") or _now()),
                        str(_row_get(row, "last_seen", 3, "") or _now()),
                        metadata,
                    ),
                )
            _record_action(
                actions,
                "orphan_position_snapshot_parents",
                "applied",
                "Backfilled placeholder trader parents for orphan position snapshots.",
                repaired=len(rows),
            )
    except Exception as exc:
        _record_action(
            actions,
            "orphan_position_snapshot_parents",
            "failed",
            "Could not backfill placeholder trader parents.",
            error=str(exc),
        )


def _repair_orphan_strategy_score_parents(actions: list[DbRepairAction]) -> None:
    try:
        with db.get_connection() as conn:
            if not _table_exists(conn, "strategy_scores") or not _table_exists(conn, "strategies"):
                _record_action(
                    actions,
                    "orphan_strategy_score_parents",
                    "skipped",
                    "strategy_scores/strategies tables are not present.",
                )
                return
            rows = conn.execute(
                """
                SELECT strategy_id, COUNT(*) AS score_count,
                       MIN(timestamp) AS first_seen,
                       MAX(timestamp) AS last_seen
                FROM strategy_scores
                WHERE strategy_id NOT IN (SELECT id FROM strategies)
                GROUP BY strategy_id
                ORDER BY strategy_id
                """
            ).fetchall()
            if not rows:
                _record_action(
                    actions,
                    "orphan_strategy_score_parents",
                    "skipped",
                    "No orphan strategy_scores rows were found.",
                )
                return
            for row in rows:
                strategy_id = int(_row_get(row, "strategy_id", 0, 0) or 0)
                if strategy_id <= 0:
                    continue
                score_count = int(_row_get(row, "score_count", 1, 0) or 0)
                first_seen = str(_row_get(row, "first_seen", 2, "") or _now())
                last_seen = str(_row_get(row, "last_seen", 3, "") or first_seen)
                conn.execute(
                    """
                    INSERT INTO strategies
                    (id, name, description, strategy_type, parameters, discovered_at,
                     last_scored, current_score, total_pnl, trade_count, win_rate,
                     sharpe_ratio, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 0, 0)
                    ON CONFLICT(id) DO UPDATE SET
                        last_scored = EXCLUDED.last_scored,
                        active = 0
                    """,
                    (
                        strategy_id,
                        f"recovered_strategy_{strategy_id}",
                        (
                            "Auto-repaired placeholder strategy created to preserve "
                            f"{score_count} historical strategy_scores rows."
                        ),
                        "retired_placeholder",
                        json.dumps(
                            {
                                "auto_repaired": True,
                                "repair_reason": "orphan_strategy_score_parent",
                                "score_count": score_count,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        first_seen,
                        last_seen,
                    ),
                )
            _record_action(
                actions,
                "orphan_strategy_score_parents",
                "applied",
                "Backfilled placeholder strategies for orphan strategy_scores rows.",
                repaired=len(rows),
            )
    except Exception as exc:
        _record_action(
            actions,
            "orphan_strategy_score_parents",
            "failed",
            "Could not backfill placeholder strategy parents.",
            error=str(exc),
        )


def _repair_source_inventory(actions: list[DbRepairAction]) -> None:
    try:
        with db.get_connection(for_read=True) as conn:
            if not _table_exists(conn, "source_inventory"):
                _record_action(
                    actions,
                    "source_inventory",
                    "skipped",
                    "source_inventory table is not present.",
                )
                return
            existing = _count(conn, "source_inventory")
        if existing > 0:
            _record_action(
                actions,
                "source_inventory",
                "skipped",
                "source_inventory already contains rows.",
                count=existing,
            )
            return
        from src.learning.source_inventory import seed_source_inventory

        inserted = int(seed_source_inventory() or 0)
        _record_action(
            actions,
            "source_inventory",
            "applied",
            "Seeded source_inventory with the default registry.",
            inserted=inserted,
        )
    except Exception as exc:
        _record_action(
            actions,
            "source_inventory",
            "failed",
            "Could not seed source_inventory.",
            error=str(exc),
        )


def _resolve_repair_risk(metadata: dict[str, Any]) -> Any:
    from src.signals.signal_schema import RiskParams

    risk_payload = metadata.get("risk")
    if not isinstance(risk_payload, dict):
        risk_payload = {}
    merged = {}
    for source in (metadata, risk_payload):
        for key in (
            "stop_loss_pct",
            "take_profit_pct",
            "max_leverage",
            "trailing_stop",
            "trailing_pct",
            "time_limit_hours",
            "risk_basis",
            "reward_to_risk_ratio",
            "enforce_reward_to_risk",
            "break_even_at_r",
            "break_even_buffer_pct",
            "trail_activate_at_r",
        ):
            if key in source:
                merged[key] = source[key]
    return RiskParams(**merged)


def _repair_open_trade_protection(actions: list[DbRepairAction]) -> None:
    try:
        with db.get_connection() as conn:
            if not _table_exists(conn, "paper_trades"):
                _record_action(
                    actions,
                    "open_trade_protection",
                    "skipped",
                    "paper_trades table is not present.",
                )
                return
            rows = conn.execute(
                """
                SELECT id, coin, side, entry_price, leverage, metadata
                FROM paper_trades
                WHERE LOWER(COALESCE(status, '')) = 'open'
                  AND (
                      stop_loss IS NULL OR take_profit IS NULL
                      OR stop_loss <= 0 OR take_profit <= 0
                  )
                ORDER BY id ASC
                """
            ).fetchall()
            if not rows:
                _record_action(
                    actions,
                    "open_trade_protection",
                    "skipped",
                    "All open paper trades already have valid protection.",
                )
                return
            repaired = 0
            examples: list[dict[str, Any]] = []
            for row in rows:
                trade_id = int(_row_get(row, "id", 0, 0) or 0)
                entry_price = float(_row_get(row, "entry_price", 3, 0.0) or 0.0)
                leverage = float(_row_get(row, "leverage", 4, 1.0) or 1.0)
                side = str(_row_get(row, "side", 2, "") or "").strip().lower()
                if trade_id <= 0 or entry_price <= 0 or leverage <= 0 or side not in {"long", "short", "buy", "sell"}:
                    continue
                raw_meta = _row_get(row, "metadata", 5, "{}")
                try:
                    metadata = json.loads(raw_meta or "{}")
                    if not isinstance(metadata, dict):
                        metadata = {}
                except Exception:
                    metadata = {}
                risk = _resolve_repair_risk(metadata)
                stop_loss, take_profit = risk.resolve_trigger_prices(entry_price, side, leverage)
                merged_meta = _merge_metadata_blob(
                    metadata,
                    {
                        "auto_repaired": True,
                        "repair_reason": "missing_trade_protection",
                        "repair_stop_loss": float(stop_loss),
                        "repair_take_profit": float(take_profit),
                        "repair_stop_roe_pct": float(risk.resolve_roe_stop_loss_pct(leverage)),
                        "repair_take_profit_roe_pct": float(risk.resolve_roe_take_profit_pct(leverage)),
                        "repair_timestamp": _now(),
                    },
                )
                conn.execute(
                    """
                    UPDATE paper_trades
                    SET stop_loss = ?, take_profit = ?, metadata = ?
                    WHERE id = ?
                    """,
                    (float(stop_loss), float(take_profit), merged_meta, trade_id),
                )
                repaired += 1
                if len(examples) < 10:
                    examples.append(
                        {
                            "id": trade_id,
                            "coin": str(_row_get(row, "coin", 1, "") or ""),
                            "side": side,
                            "stop_loss": round(float(stop_loss), 6),
                            "take_profit": round(float(take_profit), 6),
                        }
                    )
            _record_action(
                actions,
                "open_trade_protection",
                "applied" if repaired else "skipped",
                "Filled missing SL/TP values on open paper trades." if repaired else "No repairable open paper trades were found.",
                repaired=repaired,
                examples=examples,
            )
    except Exception as exc:
        _record_action(
            actions,
            "open_trade_protection",
            "failed",
            "Could not repair missing SL/TP values on open paper trades.",
            error=str(exc),
        )


def _repair_stale_pending_decisions(actions: list[DbRepairAction]) -> None:
    try:
        with db.get_connection() as conn:
            if not _table_exists(conn, "decision_snapshots"):
                _record_action(
                    actions,
                    "stale_pending_decisions",
                    "skipped",
                    "decision_snapshots table is not present.",
                )
                return
            stale_minutes = float(getattr(config, "DB_AUDIT_PENDING_DECISION_MAX_AGE_MINUTES", 30.0))
            now = datetime.now(timezone.utc)
            rows = conn.execute(
                """
                SELECT decision_id, created_at, updated_at, metadata, paper_trade_id, live_order_id
                FROM decision_snapshots
                WHERE (COALESCE(final_status, '') = 'candidate'
                       OR COALESCE(firewall_decision, '') = 'pending')
                ORDER BY created_at ASC
                """
            ).fetchall()
            stale_rows = []
            for row in rows:
                if _row_get(row, "paper_trade_id", 4, None) is not None:
                    continue
                if _row_get(row, "live_order_id", 5, None):
                    continue
                created = _parse_dt(_row_get(row, "created_at", 1, None))
                if not created:
                    continue
                if (now - created).total_seconds() <= stale_minutes * 60:
                    continue
                stale_rows.append(row)
            if not stale_rows:
                _record_action(
                    actions,
                    "stale_pending_decisions",
                    "skipped",
                    "No stale unresolved decision snapshots were found.",
                )
                return
            examples: list[dict[str, Any]] = []
            now_iso = _now()
            for row in stale_rows:
                decision_id = str(_row_get(row, "decision_id", 0, "") or "").strip()
                raw_meta = _row_get(row, "metadata", 3, "{}")
                merged_meta = _merge_metadata_blob(
                    raw_meta,
                    {
                        "auto_repaired": True,
                        "repair_reason": "stale_pending_decision_expired",
                        "repair_timestamp": now_iso,
                    },
                )
                conn.execute(
                    """
                    UPDATE decision_snapshots
                    SET updated_at = ?, final_status = ?, firewall_decision = ?,
                        rejection_reason = ?, metadata = ?
                    WHERE decision_id = ?
                    """,
                    (
                        now_iso,
                        "expired",
                        "expired",
                        "auto_expired_unresolved_candidate",
                        merged_meta,
                        decision_id,
                    ),
                )
                if len(examples) < 10:
                    examples.append(
                        {
                            "decision_id": decision_id,
                            "created_at": str(_row_get(row, "created_at", 1, "") or ""),
                        }
                    )
            _record_action(
                actions,
                "stale_pending_decisions",
                "applied",
                "Expired stale candidate decision snapshots that never opened a trade.",
                repaired=len(stale_rows),
                threshold_minutes=stale_minutes,
                examples=examples,
            )
    except Exception as exc:
        _record_action(
            actions,
            "stale_pending_decisions",
            "failed",
            "Could not expire stale pending decision snapshots.",
            error=str(exc),
        )


def _repair_stale_regime_history(actions: list[DbRepairAction], coins: Iterable[str]) -> None:
    target_coins = sorted({str(coin or "").upper() for coin in coins if str(coin or "").strip()})
    if not target_coins:
        _record_action(
            actions,
            "stale_regime_history",
            "skipped",
            "No active coins require regime-history refresh.",
        )
        return
    try:
        from src.signals.xgboost_regime_forecaster import XGBoostRegimeForecaster

        forecaster = XGBoostRegimeForecaster()
        refreshed = []
        for coin in target_coins:
            result = forecaster.predict_regime(coin)
            refreshed.append(
                {
                    "coin": coin,
                    "regime": str(result.get("regime", "") or ""),
                    "confidence": float(result.get("confidence", 0.0) or 0.0),
                }
            )
        _record_action(
            actions,
            "stale_regime_history",
            "applied",
            "Refreshed stale regime_history rows for active coins.",
            refreshed=refreshed,
        )
    except Exception as exc:
        _record_action(
            actions,
            "stale_regime_history",
            "failed",
            "Could not refresh stale active regime history.",
            coins=target_coins,
            error=str(exc),
        )


def _repair_candle_cache(actions: list[DbRepairAction], coins: Iterable[str]) -> None:
    target_coins = sorted({str(coin or "").upper() for coin in coins if str(coin or "").strip()})
    if not target_coins:
        _record_action(
            actions,
            "candle_cache",
            "skipped",
            "No active coins are missing from the candle cache.",
        )
        return
    try:
        from src.backtest.data_fetcher import DataFetcher

        fetcher = DataFetcher()
        refreshed = []
        for coin in target_coins:
            candles = fetcher.fetch_candles(coin, "1h", use_cache=True)
            refreshed.append({"coin": coin, "candles": len(candles)})
        _record_action(
            actions,
            "candle_cache",
            "applied",
            "Fetched missing active-coin candles into the local cache.",
            refreshed=refreshed,
        )
    except Exception as exc:
        _record_action(
            actions,
            "candle_cache",
            "failed",
            "Could not refresh the candle cache for active coins.",
            coins=target_coins,
            error=str(exc),
        )


def run_db_repair(
    *,
    include_candle_cache: bool = True,
    include_code_scan: bool = True,
    repair_live_data: bool = True,
) -> DbRepairReport:
    pre_audit = run_db_audit(
        include_candle_cache=include_candle_cache,
        include_code_scan=include_code_scan,
    )
    actions: list[DbRepairAction] = []
    _repair_schema_migrations_table(actions)
    _repair_orphan_position_snapshot_parents(actions)
    _repair_orphan_strategy_score_parents(actions)
    _repair_source_inventory(actions)
    _repair_open_trade_protection(actions)
    _repair_stale_pending_decisions(actions)

    if repair_live_data:
        stale_regime = [
            str(item.get("coin") or "").upper()
            for item in (pre_audit.checks.get("regime_history", {}) or {}).get("stale_active", [])
        ]
        missing_candle_coins = [
            str(coin or "").upper()
            for coin in (pre_audit.checks.get("candle_cache", {}) or {}).get("missing_active_coins", [])
        ]
        _repair_stale_regime_history(actions, stale_regime)
        if include_candle_cache:
            _repair_candle_cache(actions, missing_candle_coins)
        else:
            _record_action(
                actions,
                "candle_cache",
                "skipped",
                "Candle-cache repair skipped because candle-cache audit was disabled.",
            )
    else:
        _record_action(
            actions,
            "live_data_refresh",
            "skipped",
            "Network-backed regime/candle refresh was disabled for this repair run.",
        )

    post_audit = run_db_audit(
        include_candle_cache=include_candle_cache,
        include_code_scan=include_code_scan,
    )
    return DbRepairReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        backend=db.get_backend_name(),
        db_path=db.get_db_path(),
        pre_audit=pre_audit,
        post_audit=post_audit,
        actions=actions,
    )


def run_db_audit(
    *,
    include_candle_cache: bool = True,
    include_code_scan: bool = True,
) -> DbAuditReport:
    checks: dict[str, Any] = {}
    findings: list[DbAuditFinding] = []
    active_coins: set[str] = set()
    backend = db.get_backend_name()
    db_path = db.get_db_path()

    try:
        with db.get_connection(for_read=True) as conn:
            checks["connection_backend"] = getattr(conn, "backend", backend)
            _sqlite_integrity_checks(conn, findings, checks)
            _schema_checks(conn, findings, checks)
            _paper_account_checks(conn, findings, checks)
            active_coins.update(_paper_trade_checks(conn, findings, checks))
            active_coins.update(_decision_journal_checks(conn, findings, checks))
            _source_health_checks(conn, findings, checks)
            _historical_data_checks(conn, findings, checks)
            _regime_checks(conn, findings, checks, active_coins)
    except Exception as exc:
        _add(
            findings,
            "db_audit_connection",
            "critical",
            "Database audit could not read the runtime database.",
            error=str(exc),
        )
        checks["connection_error"] = str(exc)

    checks["active_coins"] = sorted(active_coins)
    _dualwrite_checks(findings, checks)
    if include_candle_cache:
        try:
            _candle_cache_checks(findings, checks, active_coins)
        except Exception as exc:
            checks["candle_cache"] = {"error": str(exc)}
            _add(
                findings,
                "candle_cache_audit_failed",
                "medium",
                "Could not audit candle cache.",
                error=str(exc),
            )
    if include_code_scan:
        try:
            _app_structure_checks(findings, checks)
        except Exception as exc:
            checks["app_structure"] = {"error": str(exc)}

    return DbAuditReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        backend=backend,
        db_path=db_path,
        checks=checks,
        findings=findings,
    )


def format_db_audit_report(report: DbAuditReport, *, block_severity: str = "high") -> str:
    payload = report.to_dict(block_severity=block_severity)
    lines = [
        "Database Audit",
        f"  backend: {payload['backend']}",
        f"  db_path: {payload['db_path']}",
        f"  ok: {payload['ok']} (block severity: {block_severity})",
        f"  findings: {payload['finding_count']} total, {payload['blocking_finding_count']} blocking",
    ]
    if not report.findings:
        lines.append("  no findings")
        return "\n".join(lines)

    for finding in sorted(
        report.findings,
        key=lambda item: SEVERITY_ORDER.get(item.severity, 0),
        reverse=True,
    ):
        lines.append(f"  [{finding.severity.upper()}] {finding.check}: {finding.message}")
        if finding.details:
            details = json.dumps(finding.details, sort_keys=True, default=str)
            lines.append(f"    details: {details[:800]}")
    return "\n".join(lines)


def format_db_repair_report(report: DbRepairReport, *, block_severity: str = "high") -> str:
    pre = report.pre_audit.to_dict(block_severity=block_severity)
    post = report.post_audit.to_dict(block_severity=block_severity)
    lines = [
        "Database Repair",
        f"  backend: {report.backend}",
        f"  db_path: {report.db_path}",
        (
            "  blocking findings: "
            f"{pre['blocking_finding_count']} -> {post['blocking_finding_count']} "
            f"(severity: {block_severity})"
        ),
        f"  actions: {len(report.actions)}",
    ]
    if not report.actions:
        lines.append("  no actions")
    else:
        for action in report.actions:
            lines.append(f"  [{action.status.upper()}] {action.action}: {action.message}")
            if action.details:
                details = json.dumps(action.details, sort_keys=True, default=str)
                lines.append(f"    details: {details[:800]}")
    lines.append("")
    lines.append(format_db_audit_report(report.post_audit, block_severity=block_severity))
    return "\n".join(lines)


def audit_exit_code(report: DbAuditReport, *, block_severity: str = "high") -> int:
    return 2 if report.findings_at_or_above(block_severity) else 0


def run_periodic_audit_snapshot(interval_s: float = 300.0) -> dict[str, Any]:
    """Small helper for future background jobs without creating another cache."""
    report = run_db_audit()
    payload = report.to_dict(
        block_severity=str(getattr(config, "READINESS_DB_AUDIT_BLOCK_SEVERITY", "high"))
    )
    payload["next_run_after_epoch_s"] = time.time() + float(interval_s)
    return payload
