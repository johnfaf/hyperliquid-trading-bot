"""Read-only funding evidence checks for the paper research page."""
from contextlib import closing
from datetime import timedelta
from pathlib import Path
import re
import sqlite3

from src.trading.paper_funding import FundingLedger, HOUR, MODEL, hours_between, ledger_path, millis, verified_cost
from src.trading.paper_reporting import fields, timestamp
from src.trading.trade_decision_memory import number

LIMIT = 2000
GRACE_MS = 5 * 60_000


def close_result(row, meta):
    coverage = fields(meta.get("funding_coverage"))
    signed = verified_cost(row, coverage)
    if signed is None:
        return "missing_receipt" if coverage.get("state") == "incomplete" else "invalid_receipt"
    pnl, gross, fees, charged = (number(v) for v in (
        row["pnl"], meta.get("gross_pnl_before_fees"), meta.get("total_fees_paid"), meta.get("funding_accrued"),
    ))
    if (None in (pnl, gross, fees, charged) or fees < 0 or abs(signed-charged) > .000051
            or abs(pnl-(gross-fees-signed)) > .011):
        return "cost_mismatch"
    return "verified"


def _open(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
    conn.row_factory = sqlite3.Row
    return conn


def funding_health(path, now):
    result = {
        "state": "Ledger unavailable", "observed_at": now.isoformat(), "last_oracle_hour": None,
        "active": 0, "open_due": 0, "pending_hours": 0, "overdue_hours": 0, "invalid_open": 0,
        "cross_hour_closes": 0, "verified_closes": 0, "unverified_closes": 0,
        "by_exit": {},
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
                f"SELECT {columns}, pnl FROM paper_trades WHERE status='closed' AND closed_at>=? AND closed_at<? "
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
        reason = meta.get("close_reason") or fields(meta.get("exit_decision")).get("reason")
        reason = reason if isinstance(reason, str) and re.fullmatch("[a-z0-9_]{1,48}", reason) else "unrecorded_or_other"
        if len(result["by_exit"]) >= 20 and reason not in result["by_exit"]:
            reason = "unrecorded_or_other"
        group = result["by_exit"].setdefault(reason, {"verified": 0, "missing_receipt": 0, "invalid_receipt": 0, "cost_mismatch": 0})
        outcome = close_result(row, meta)
        group[outcome] += 1
        key = "verified_closes" if outcome == "verified" else "unverified_closes"
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
