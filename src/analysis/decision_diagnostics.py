"""Why-isn't-it-trading diagnostics.

Aggregates recent ``decision_outcomes`` into an at-a-glance breakdown of how
many candidate decisions executed vs were rejected, and the top rejection
reasons -- the data behind the "Decision summary" log line, but queryable so
the dashboard (and operators) can self-diagnose a quiet bot without log-greps.
Read-only.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List


def _normalise_reason(raw: Any) -> str:
    """Collapse a verbose rejection_reason to its stable family.

    e.g. ``ev_below_threshold:ev=1.8bps<=thr=20.7bps (cost=...)`` -> ``ev_below_threshold``
         ``Source allocator paused strategy:momentum_long (paused)`` -> ``Source allocator paused``
    """
    s = str(raw or "").strip()
    if not s:
        return "unknown"
    # 1) Drop a trailing source-key fragment (strategy:x / copy_trade:0x... / etc.)
    #    so "Source allocator paused strategy:momentum_long (paused)" -> "Source allocator paused".
    s = re.sub(
        r"\s+(strategy|copy_trade|options_flow|polymarket|funding_carry)[:|]\S.*$",
        "", s, flags=re.IGNORECASE,
    ).strip()
    # 2) Drop a trailing parenthetical detail "(...)".
    s = re.split(r"\s*\(", s, maxsplit=1)[0].strip()
    # 3) Collapse a key:detail family (no spaces before the colon),
    #    e.g. "ev_below_threshold:ev=1.8bps<=thr=20.7bps" -> "ev_below_threshold".
    if ":" in s and " " not in s.split(":", 1)[0]:
        s = s.split(":", 1)[0].strip()
    return s or "unknown"


def summarize_recent_decisions(hours: float = 6.0, top_n: int = 12) -> Dict[str, Any]:
    """Return {window_hours, total, executed, rejected, execution_rate,
    top_reasons:[{reason,count}], generated_at}.  Never raises -- returns an
    ``error`` key on failure so the endpoint can degrade gracefully."""
    out: Dict[str, Any] = {
        "window_hours": hours,
        "total": 0,
        "executed": 0,
        "rejected": 0,
        "execution_rate": 0.0,
        "top_reasons": [],
    }
    try:
        from src.data import database as db
        from src.data.database import get_connection
        from src.core import clock_provider

        out["generated_at"] = clock_provider.utc_now().isoformat()
        backend = db.get_backend_name()
        hrs = max(1, int(hours))
        cutoff = (
            f"now() - INTERVAL '{hrs} hours'"
            if backend == "postgres"
            else f"datetime('now', '-{hrs} hours')"
        )

        with get_connection(for_read=True) as conn:
            def _one(sql: str) -> int:
                # COUNT(*) -> always a single positional column.  (Don't index
                # by name: a sqlite3.Row names this column "COUNT(*)", not
                # "count", so r["count"] raised KeyError in prod.)
                r = conn.execute(sql).fetchone()
                return int(r[0]) if r and r[0] is not None else 0

            out["total"] = _one(
                f"SELECT COUNT(*) FROM decision_outcomes WHERE created_at > {cutoff}"
            )
            out["executed"] = _one(
                f"SELECT COUNT(*) FROM decision_outcomes "
                f"WHERE created_at > {cutoff} AND action_taken"
            )
            reason_rows = conn.execute(
                f"SELECT rejection_reason FROM decision_outcomes "
                f"WHERE created_at > {cutoff} "
                f"AND rejection_reason IS NOT NULL AND rejection_reason <> ''"
            ).fetchall()

        out["rejected"] = max(0, out["total"] - out["executed"])
        out["execution_rate"] = (
            round(out["executed"] / out["total"], 4) if out["total"] else 0.0
        )

        counts: Dict[str, int] = {}
        for r in reason_rows:
            raw = r[0]  # single positional column; works for tuple + sqlite3.Row
            fam = _normalise_reason(raw)
            counts[fam] = counts.get(fam, 0) + 1
        ranked: List[Dict[str, Any]] = sorted(
            ({"reason": k, "count": v} for k, v in counts.items()),
            key=lambda d: d["count"],
            reverse=True,
        )
        out["top_reasons"] = ranked[: max(1, int(top_n))]
    except Exception as exc:  # degrade gracefully
        out["error"] = str(exc)[:200]
    return out
