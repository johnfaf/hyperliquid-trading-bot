"""Calibration tab — the v2 dashboard's first concrete page.

Surfaces what the ``CalibrationTracker`` learned from recent trades:
global ECE/Brier, the live-pause flag, and a per-source breakdown
keyed on (source, side, regime). The new calibrator is the operator's
single best lens on whether confidence is meaningful — this page
makes that visible without reading logs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
import logging
import math
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.ui.v2.auth import require_auth
from src.ui.v2.state import get_components

logger = logging.getLogger(__name__)

router = APIRouter()


def _parse_ts(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    if out > 1.0:
        out = out / 100.0
    return min(max(out, 0.0), 1.0)


def _loads_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _ece(records: List[tuple[float, int]], bins: int = 10) -> Optional[float]:
    if not records:
        return None
    buckets: Dict[int, List[tuple[float, int]]] = defaultdict(list)
    for confidence, actual in records:
        idx = min(bins - 1, max(0, int(confidence * bins)))
        buckets[idx].append((confidence, actual))
    total = len(records)
    error = 0.0
    for items in buckets.values():
        avg_conf = sum(c for c, _ in items) / len(items)
        avg_actual = sum(a for _, a in items) / len(items)
        error += (len(items) / total) * abs(avg_conf - avg_actual)
    return error


def _ece_timeline_payload(days: int = 21, top_sources: int = 8) -> Dict[str, Any]:
    days = max(1, min(int(days or 21), 180))
    top_sources = max(1, min(int(top_sources or 8), 20))
    since = datetime.now(timezone.utc) - timedelta(days=days)
    try:
        from src.data import database as db

        if not db.table_exists("decision_outcomes"):
            return {"days": days, "sources": [], "points": [], "reason": "decision_outcomes_missing"}
        has_snapshots = db.table_exists("decision_snapshots")
        with db.get_connection(for_read=True) as conn:
            if has_snapshots:
                rows = conn.execute(
                    """
                    SELECT o.created_at, o.source, o.source_key, o.strategy_type,
                           o.label_win, o.decision_metadata,
                           s.raw_confidence, s.calibrated_confidence
                    FROM decision_outcomes o
                    LEFT JOIN decision_snapshots s ON s.decision_id = o.decision_id
                    WHERE o.created_at >= ? AND o.label_win IS NOT NULL
                    ORDER BY o.created_at ASC
                    """,
                    (since.isoformat(),),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT created_at, source, source_key, strategy_type,
                           label_win, decision_metadata
                    FROM decision_outcomes
                    WHERE created_at >= ? AND label_win IS NOT NULL
                    ORDER BY created_at ASC
                    """,
                    (since.isoformat(),),
                ).fetchall()
    except Exception as exc:
        logger.debug("ECE timeline query failed: %s", exc)
        return {"days": days, "sources": [], "points": [], "reason": "query_failed"}

    grouped: Dict[tuple[str, str], List[tuple[float, int]]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    for row in rows:
        item = dict(row)
        ts = _parse_ts(item.get("created_at"))
        if ts is None:
            continue
        try:
            label = int(item.get("label_win"))
        except (TypeError, ValueError):
            continue
        if label not in (0, 1):
            continue
        metadata = _loads_dict(item.get("decision_metadata"))
        confidence = None
        for candidate in (
            item.get("calibrated_confidence"),
            item.get("raw_confidence"),
            metadata.get("calibrated_confidence"),
            metadata.get("raw_confidence"),
            metadata.get("confidence"),
        ):
            confidence = _safe_float(candidate)
            if confidence is not None:
                break
        if confidence is None:
            continue
        source = str(
            item.get("source_key")
            or item.get("strategy_type")
            or item.get("source")
            or "unknown"
        ).strip() or "unknown"
        day = ts.astimezone(timezone.utc).date().isoformat()
        grouped[(day, source)].append((confidence, label))
        source_counts[source] += 1

    selected_sources = [src for src, _ in source_counts.most_common(top_sources)]
    points: List[Dict[str, Any]] = []
    for (day, source), records in grouped.items():
        if source not in selected_sources:
            continue
        ece = _ece(records)
        if ece is None:
            continue
        points.append(
            {
                "date": day,
                "source": source,
                "ece": round(ece, 6),
                "samples": len(records),
            }
        )
    points.sort(key=lambda p: (p["date"], p["source"]))
    return {"days": days, "sources": selected_sources, "points": points}


def _templates() -> Jinja2Templates:
    # Lazy so tests can override the search path.
    from src.ui.v2.app import get_templates
    return get_templates()


def _serialize_curve(cal, key: str) -> List[Dict[str, Any]]:
    try:
        return cal.get_calibration_curve(key)
    except Exception as exc:
        logger.debug("curve fetch failed for %s: %s", key, exc)
        return []


def _summary_payload(cal) -> Dict[str, Any]:
    try:
        global_ece = cal.get_ece("global")
    except Exception:
        global_ece = None
    try:
        global_brier = cal.get_brier("global")
    except Exception:
        global_brier = None
    try:
        live_paused = cal.is_live_paused()
    except Exception:
        live_paused = False
    try:
        quarantined = cal.get_quarantined_sources()
    except Exception:
        quarantined = []
    try:
        all_stats = cal.get_all_stats()
    except Exception:
        all_stats = {}

    rows: List[Dict[str, Any]] = []
    for key, stats in all_stats.items():
        if key == "global":
            continue
        # Decompose the (source, side, regime) key for display.
        try:
            from src.signals.calibration import decompose_calibration_key
            source, side, regime = decompose_calibration_key(key)
        except Exception:
            source, side, regime = key, "_", "any"
        rows.append({
            "key": key,
            "source": source,
            "side": side,
            "regime": regime,
            "samples": stats.get("total_records", 0),
            "ece": stats.get("ece"),
            "brier": stats.get("brier"),
            "quality": stats.get("calibration_quality"),
            "quarantined": bool(stats.get("quarantined")),
        })
    rows.sort(
        key=lambda r: (
            0 if r["quarantined"] else 1,
            -(r["ece"] if r["ece"] is not None else -1),
            -r["samples"],
        )
    )
    return {
        "global": {
            "ece": global_ece,
            "brier": global_brier,
            "live_paused": bool(live_paused),
            "n_sources": len(rows),
            "n_quarantined": len(quarantined),
            "thresholds": {
                "live_pause_ece": getattr(cal, "live_pause_ece", None),
                "quarantine_ece": getattr(cal, "quarantine_ece", None),
                "min_outcomes": getattr(cal, "min_outcomes", None),
                "isotonic_min_outcomes": getattr(cal, "isotonic_min_outcomes", None),
                "half_life_days": getattr(cal, "half_life_days", None),
            },
        },
        "sources": rows,
        "quarantined": quarantined,
    }


@router.get("/api/calibration", response_class=JSONResponse)
async def calibration_data(request: Request):
    redirect = require_auth(request)
    if redirect is not None:
        return JSONResponse({"error": "auth_required"}, status_code=401)

    cal = get_components().calibration
    if cal is None:
        return JSONResponse({"error": "calibration_unavailable"}, status_code=503)
    return JSONResponse(_summary_payload(cal))


@router.get("/api/calibration/curve", response_class=JSONResponse)
async def calibration_curve(request: Request, key: str = "global"):
    """Return the bin-by-bin calibration curve for one (source|side|regime) key."""
    redirect = require_auth(request)
    if redirect is not None:
        return JSONResponse({"error": "auth_required"}, status_code=401)

    cal = get_components().calibration
    if cal is None:
        return JSONResponse({"error": "calibration_unavailable"}, status_code=503)
    return JSONResponse({"key": key, "curve": _serialize_curve(cal, key)})


@router.get("/api/calibration/timeline", response_class=JSONResponse)
async def calibration_timeline(request: Request, days: int = 21, top_sources: int = 8):
    """Rolling daily ECE per source from decision outcomes."""
    redirect = require_auth(request)
    if redirect is not None:
        return JSONResponse({"error": "auth_required"}, status_code=401)
    return JSONResponse(_ece_timeline_payload(days=days, top_sources=top_sources))


@router.get("/calibration", response_class=HTMLResponse)
async def calibration_page(request: Request):
    redirect = require_auth(request)
    if redirect is not None:
        return redirect

    cal = get_components().calibration
    payload: Optional[Dict[str, Any]] = (
        _summary_payload(cal) if cal is not None else None
    )
    return _templates().TemplateResponse(
        request,
        "calibration.html",
        {
            "title": "Calibration",
            "data": payload,
            "available": cal is not None,
        },
    )
