"""Read-only funding evidence checks for the paper research page."""
from contextlib import closing
from datetime import timedelta
from pathlib import Path
import sqlite3

from src.trading.paper_funding import FundingLedger, HOUR, MODEL, hours_between, ledger_path, millis, verified_cost
from src.trading.paper_reporting import fields, timestamp
from src.trading.trade_decision_memory import number

LIMIT = 2000
GRACE_MS = 5 * 60_000


def _open(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
    conn.row_factory = sqlite3.Row
    return conn


def funding_health(path, now):
    result = {
        "state": "Ledger unavailable", "observed_at": now.isoformat(), "last_oracle_hour": None,
        "active": 0, "open_due": 0, "pending_hours": 0, "overdue_hours": 0, "invalid_open": 0,
        "cross_hour_closes": 0, "verified_closes": 0, "unverified_closes": 0,
    }
    cutoff = now - timedelta(days=1)
    instant = millis(now)
    ledger = FundingLedger(ledger_path(path))
    columns = "coin, side, size, leverage, opened_at, closed_at, metadata"
    try:
        with closing(_open(path)) as conn:
            conn.execute("BEGIN")
            opened = conn.execute(
                f"SELECT {columns} FROM paper_trades WHERE status='open' LIMIT ?", (LIMIT + 1,)
            ).fetchall()
            closed = conn.execute(
                f"SELECT {columns} FROM paper_trades WHERE status='closed' AND closed_at>=? AND closed_at<? "
                "ORDER BY closed_at DESC LIMIT ?", (cutoff.isoformat(), now.isoformat(), LIMIT + 1)
            ).fetchall()
    except (OSError, sqlite3.Error):
        return result
    if max(len(opened), len(closed)) > LIMIT:
        return {**result, "state": "Report limit reached"}

    if ledger.path.exists():
        try:
            with closing(_open(ledger.path)) as conn:
                result["last_oracle_hour"] = conn.execute(
                    "SELECT max(hour_ms) FROM paper_oracle_samples WHERE hour_ms<=?",
                    ((instant//HOUR+1)*HOUR,),
                ).fetchone()[0]
        except (OSError, sqlite3.Error):
            return {**result, "state": "Funding sidecar unreadable"}
    for row in opened:
        trade = dict(row)
        if fields(trade["metadata"]).get("paper_funding_model") != MODEL:
            continue
        result["active"] += 1
        summary = ledger.position(trade, now=now)
        if summary["state"] not in ("modeled_complete", "incomplete"):
            result["invalid_open"] += 1
            continue
        if summary["missing_hours"] or summary["receipts"]:
            result["open_due"] += 1
        for hour in summary["missing_hours"]:
            key = "overdue_hours" if instant-hour >= GRACE_MS else "pending_hours"
            result[key] += 1
    for row in closed:
        meta = fields(row["metadata"])
        if meta.get("paper_funding_model") != MODEL:
            continue
        ended = timestamp(row["closed_at"])
        if ended is None or not cutoff <= ended < now:
            continue
        hours = hours_between(millis(row["opened_at"]), millis(ended))
        if hours == []:
            continue
        result["cross_hour_closes"] += 1
        coverage = fields(meta.get("funding_coverage"))
        signed = verified_cost(row, coverage)
        charged = number(meta.get("funding_accrued"))
        key = "verified_closes" if signed is not None and charged is not None and abs(signed-charged) <= .000051 else "unverified_closes"
        result[key] += 1

    if result["overdue_hours"] or result["unverified_closes"] or result["invalid_open"]:
        state = "Evidence gaps"
    elif result["pending_hours"]:
        state = "Waiting for published rates"
    elif result["verified_closes"]:
        state = "Funded closes verified"
    elif result["open_due"]:
        state = "Open settlements measured"
    else:
        state = "No cross-hour closes measured"
    return {**result, "state": state}
