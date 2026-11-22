"""Read-only, dated paper-book statistics; independent of the bot runtime."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3


def timestamp(value):
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def fields(value):
    try:
        value = json.loads(value or "{}") if not isinstance(value, dict) else value
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def metrics(trades):
    closed = [t for t in trades if t["status"] == "closed"]
    priced = [t for t in closed if finite(t["pnl"]) is not None]
    values = [float(t["pnl"]) for t in priced]
    net = math.fsum(values) if values else None
    gain = math.fsum(v for v in values if v > 0)
    loss = -math.fsum(v for v in values if v < 0)
    running = peak = drawdown = 0.
    ordered = all(timestamp(t.get("closed_at")) for t in priced)
    if ordered:
        for trade in sorted(priced, key=lambda t: (timestamp(t["closed_at"]), t["id"])):
            running += float(trade["pnl"])
            peak = max(peak, running)
            drawdown = max(drawdown, peak - running)
    largest = max((v for v in values if v > 0), default=None)
    complete = bool(closed) and len(priced) == len(closed)
    status = ("Incomplete P&L" if not complete else "Insufficient sample" if len(priced) < 30 else
              "Negative mean net" if net <= 0 else "Winner concentrated" if largest and net - largest <= 0 else
              "Research review only")
    return {"profit_factor": gain / loss if loss else None, "no_losing_closes": bool(values) and loss == 0,
            "average": net / len(priced) if priced else None,
            "win_rate": 100 * sum(v > 0 for v in values) / len(values) if values else None,
            "realized_drawdown": drawdown if complete and ordered else None,
            "largest_winner": largest, "net_without_largest": net - largest if largest is not None else net,
            "evidence": status, "live_approved": False}


def completed_week(now):
    """Most recently completed Monday-to-Monday UTC week, never a moving window."""
    end = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    end -= timedelta(days=end.weekday())
    return end - timedelta(days=7), end


def valuation(path, now):
    result = {"cash": None, "unrealized": None, "equity": None, "observed_at": None, "open": None}
    try:
        with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN")
            account = db.execute("SELECT balance FROM paper_account WHERE id=1").fetchone()
            trades = db.execute("SELECT coin,side,entry_price,size,leverage FROM paper_trades WHERE status='open'").fetchall()
            result.update(cash=finite(account[0]) if account else None, open=len(trades))
            if not trades:
                result.update(unrealized=0., equity=result["cash"], observed_at=now.isoformat())
                return result
            row = db.execute("SELECT value FROM bot_state WHERE key='paper_mark_prices'").fetchone()
            marks = fields(row[0]) if row else {}
        observed = timestamp(marks.get("observed_at"))
        if observed is None or not 0 <= (now - observed).total_seconds() <= 900:
            return result
        mids = fields(marks.get("mids"))
        values = []
        for trade in trades:
            mark = finite(mids.get(trade["coin"]))
            entry, size, leverage = (finite(trade[k]) for k in ("entry_price", "size", "leverage"))
            if None in (mark, entry, size, leverage) or min(mark, entry, size, leverage) <= 0 or trade["side"] not in {"long", "short"}:
                return result
            values.append((mark - entry) * size * leverage * (1 if trade["side"] == "long" else -1))
        unrealized = math.fsum(values)
        result.update(unrealized=unrealized, equity=result["cash"] + unrealized if result["cash"] is not None else None,
                      observed_at=observed.isoformat())
    except (sqlite3.Error, OSError):
        pass
    return result


def weekly_report(path, now):
    start, end = completed_week(now)
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("""SELECT id,status,closed_at,pnl,metadata FROM paper_trades
            WHERE status='closed' AND julianday(closed_at)>=julianday(?) AND julianday(closed_at)<julianday(?)
            ORDER BY julianday(closed_at),id LIMIT 50001""", (start.isoformat(), end.isoformat())).fetchall()
    if len(rows) > 50000:
        raise ValueError("Weekly report exceeds bounded row limit")
    clean, excluded, groups = [], 0, {}
    for row in rows:
        trade = dict(row)
        trade["pnl"] = finite(trade["pnl"])
        meta = fields(trade.pop("metadata"))
        reason = fields(meta.get("exit_decision")).get("reason") or meta.get("close_reason")
        if meta.get("tainted") or meta.get("synthetic_reconciliation") or "live_reconciled_closed" in (reason, meta.get("reconciliation_reason")):
            excluded += 1
            continue
        trade["route"] = str(fields(meta.get("entry_decision")).get("route") or meta.get("source") or "Unrecorded")[:100]
        trade["fees"] = finite(meta.get("total_fees_paid", meta.get("fees_paid")))
        clean.append(trade)
        groups.setdefault(trade["route"], []).append(trade)

    def total(trades):
        values = [finite(t["pnl"]) for t in trades if finite(t["pnl"]) is not None]
        fees = [t["fees"] for t in trades if t["fees"] is not None]
        return {**metrics(trades), "closed": len(trades), "priced": len(values),
                "net": math.fsum(values) if values else None,
                "fees": math.fsum(fees) if fees else None, "fee_count": len(fees)}

    return {"version": 1, "start": start.isoformat(), "end_exclusive": end.isoformat(),
            "generated_at": now.isoformat(), "mode": "paper", "excluded": excluded,
            "summary": total(clean), "routes": {key: total(rows) for key, rows in sorted(groups.items())},
            "trades": clean, "equity_drawdown": None}
