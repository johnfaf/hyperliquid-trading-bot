"""Production audit/source analysis helpers.

These helpers answer two operational questions:

* Which source keys explain most approvals/rejections?
* Which sources are stuck in allocator warmup for too long?

The module is deliberately SQL-light and does most filtering in Python so it
works across SQLite, dual-write, and Postgres adapters.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Optional

from src.data import database as db

logger = logging.getLogger(__name__)
_CACHE: Dict[str, Any] = {"key": None, "ts": 0.0, "report": None}


APPROVED_STATUSES = {
    "approved",
    "paper_opened",
    "live_opened",
    "executed",
    "accepted",
    "opened",
}


def _row_dict(row: Any) -> Dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    try:
        return {key: row[key] for key in row.keys()}
    except Exception:
        return {}


def _parse_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _source_key(row: Dict[str, Any]) -> str:
    details = _parse_json(row.get("details") or row.get("metadata"))
    raw = (
        row.get("source_key")
        or details.get("source_key")
        or details.get("source")
        or row.get("source")
        or row.get("strategy_type")
        or "unknown"
    )
    return str(raw or "unknown").strip().lower() or "unknown"


def _decision_kind(row: Dict[str, Any]) -> str:
    status = str(row.get("final_status") or row.get("firewall_decision") or "").strip().lower()
    reason = str(row.get("rejection_reason") or "").strip().lower()
    action = str(row.get("action") or "").strip().lower()
    if status in APPROVED_STATUSES or "approved" in action or "open" in action:
        return "approved"
    if status.startswith("reject") or reason or "reject" in action:
        return "rejected"
    return "other"


def _within_window(row: Dict[str, Any], cutoff: datetime) -> bool:
    ts = _parse_dt(row.get("created_at") or row.get("timestamp"))
    return ts is None or ts >= cutoff


def _fetch_rows(table: str, columns: str, limit: int) -> List[Dict[str, Any]]:
    if not db.table_exists(table):
        return []
    try:
        with db.get_connection(for_read=True) as conn:
            rows = conn.execute(
                f"SELECT {columns} FROM {table} ORDER BY 1 DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [_row_dict(row) for row in rows]
    except Exception as exc:
        logger.debug("Audit source analysis could not read %s: %s", table, exc)
        return []


def _source_buckets(rows: Iterable[Dict[str, Any]], cutoff: datetime) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "source_key": "",
            "approvals": 0,
            "rejections": 0,
            "other": 0,
            "total": 0,
            "warmup_rejections": 0,
            "first_seen": None,
            "last_seen": None,
        }
    )
    for row in rows:
        if not _within_window(row, cutoff):
            continue
        key = _source_key(row)
        bucket = buckets[key]
        bucket["source_key"] = key
        kind = _decision_kind(row)
        if kind == "approved":
            bucket["approvals"] += 1
        elif kind == "rejected":
            bucket["rejections"] += 1
        else:
            bucket["other"] += 1
        bucket["total"] += 1
        reason = str(row.get("rejection_reason") or row.get("details") or "").lower()
        if "warmup" in reason:
            bucket["warmup_rejections"] += 1
        ts = _parse_dt(row.get("created_at") or row.get("timestamp"))
        if ts:
            if bucket["first_seen"] is None or ts < bucket["first_seen"]:
                bucket["first_seen"] = ts
            if bucket["last_seen"] is None or ts > bucket["last_seen"]:
                bucket["last_seen"] = ts
    return buckets


def _attach_outcomes(buckets: Dict[str, Dict[str, Any]], limit: int) -> None:
    rows = _fetch_rows(
        "decision_outcomes",
        "created_at, source, source_key, strategy_type, action_taken, label_win, outcome_pnl",
        limit,
    )
    for row in rows:
        key = _source_key(row)
        bucket = buckets.setdefault(
            key,
            {
                "source_key": key,
                "approvals": 0,
                "rejections": 0,
                "other": 0,
                "total": 0,
                "warmup_rejections": 0,
                "first_seen": None,
                "last_seen": None,
            },
        )
        bucket["outcome_count"] = int(bucket.get("outcome_count", 0) or 0) + 1
        try:
            pnl = float(row.get("outcome_pnl", 0.0) or 0.0)
        except Exception:
            pnl = 0.0
        bucket["net_pnl"] = float(bucket.get("net_pnl", 0.0) or 0.0) + pnl
        if pnl > 0:
            bucket["wins"] = int(bucket.get("wins", 0) or 0) + 1
        elif pnl < 0:
            bucket["losses"] = int(bucket.get("losses", 0) or 0) + 1


def _finalize_rows(
    buckets: Dict[str, Dict[str, Any]],
    *,
    coverage_threshold: float,
    warmup_days: int,
    warmup_min_rejections: int,
) -> Dict[str, Any]:
    rows = []
    for bucket in buckets.values():
        wins = int(bucket.get("wins", 0) or 0)
        losses = int(bucket.get("losses", 0) or 0)
        total = int(bucket.get("total", 0) or 0)
        outcome_count = int(bucket.get("outcome_count", 0) or 0)
        net_pnl = round(float(bucket.get("net_pnl", 0.0) or 0.0), 4)
        first_seen = bucket.get("first_seen")
        last_seen = bucket.get("last_seen")
        rows.append(
            {
                "source_key": bucket.get("source_key", "unknown"),
                "approvals": int(bucket.get("approvals", 0) or 0),
                "rejections": int(bucket.get("rejections", 0) or 0),
                "other": int(bucket.get("other", 0) or 0),
                "total": total,
                "approval_share": round(bucket.get("approvals", 0) / total, 4) if total else 0.0,
                "warmup_rejections": int(bucket.get("warmup_rejections", 0) or 0),
                "outcome_count": outcome_count,
                "net_pnl": net_pnl,
                "win_rate": round(wins / (wins + losses), 4) if (wins + losses) else 0.0,
                "net_positive": net_pnl > 0,
                "first_seen": first_seen.isoformat() if first_seen else None,
                "last_seen": last_seen.isoformat() if last_seen else None,
            }
        )
    rows.sort(key=lambda row: (row["total"], abs(row["net_pnl"])), reverse=True)

    total_decisions = sum(row["total"] for row in rows)
    cumulative = 0
    coverage_rows = []
    for row in rows:
        cumulative += row["total"]
        enriched = dict(row)
        enriched["cumulative_share"] = round(
            cumulative / total_decisions,
            4,
        ) if total_decisions else 0.0
        coverage_rows.append(enriched)
        if total_decisions and cumulative / total_decisions >= coverage_threshold:
            break

    now = datetime.now(timezone.utc)
    warmup_alerts = []
    for row in rows:
        first_seen = _parse_dt(row.get("first_seen"))
        age_days = (now - first_seen).days if first_seen else 0
        if row["warmup_rejections"] >= warmup_min_rejections and age_days >= warmup_days:
            alert = dict(row)
            alert["age_days"] = age_days
            warmup_alerts.append(alert)

    return {
        "sources": rows,
        "coverage_threshold": coverage_threshold,
        "top_sources_to_threshold": coverage_rows,
        "warmup_stuck_alerts": warmup_alerts,
        "summary": {
            "source_count": len(rows),
            "total_decisions": total_decisions,
            "approvals": sum(row["approvals"] for row in rows),
            "rejections": sum(row["rejections"] for row in rows),
            "net_positive_sources": sum(1 for row in rows if row["net_positive"]),
            "warmup_stuck_sources": len(warmup_alerts),
        },
    }


def analyze_audit_sources(
    *,
    days: int = 14,
    limit: int = 10_000,
    coverage_threshold: float = 0.80,
    warmup_days: int = 2,
    warmup_min_rejections: int = 25,
    cleanup_short_copy_keys: bool = False,
    send_warmup_alerts: bool = False,
    cache_ttl_seconds: float = 0.0,
) -> Dict[str, Any]:
    """Return production source concentration, performance, and warmup health."""
    cache_key = (
        days,
        limit,
        round(float(coverage_threshold), 4),
        warmup_days,
        warmup_min_rejections,
        bool(cleanup_short_copy_keys),
    )
    if cache_ttl_seconds > 0:
        age = time.time() - float(_CACHE.get("ts", 0.0) or 0.0)
        if _CACHE.get("key") == cache_key and age <= cache_ttl_seconds and _CACHE.get("report"):
            return dict(_CACHE["report"])

    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))
    snapshot_rows = _fetch_rows(
        "decision_snapshots",
        "created_at, source, source_key, strategy_type, final_status, firewall_decision, rejection_reason, metadata",
        limit,
    )
    audit_rows = _fetch_rows(
        "audit_trail",
        "timestamp, action, source, details, pnl",
        limit,
    )
    buckets = _source_buckets([*snapshot_rows, *audit_rows], cutoff)
    _attach_outcomes(buckets, limit)
    report = _finalize_rows(
        buckets,
        coverage_threshold=max(0.05, min(float(coverage_threshold), 1.0)),
        warmup_days=max(1, int(warmup_days)),
        warmup_min_rejections=max(1, int(warmup_min_rejections)),
    )
    report["window_days"] = max(1, int(days))
    report["cleanup"] = db.cleanup_short_copy_trade_agent_scores(
        apply=bool(cleanup_short_copy_keys)
    )
    if send_warmup_alerts and report["warmup_stuck_alerts"]:
        report["warmup_alert_sent"] = send_source_warmup_alert(
            report["warmup_stuck_alerts"]
        )
    if cache_ttl_seconds > 0:
        _CACHE.update({"key": cache_key, "ts": time.time(), "report": dict(report)})
    return report


def send_source_warmup_alert(alerts: List[Dict[str, Any]]) -> bool:
    """Best-effort Telegram alert for sources stuck in allocator warmup."""
    try:
        from src.notifications.telegram_alerts import send_source_warmup_stuck_alert

        return bool(send_source_warmup_stuck_alert(alerts))
    except Exception as exc:
        logger.debug("Source warmup alert dispatch skipped: %s", exc)
        return False
