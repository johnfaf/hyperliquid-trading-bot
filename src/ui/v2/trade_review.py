"""Read-only paper trade attribution and execution evidence."""
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3

import config
from src.trading.trade_decision_memory import COPY_CONCENTRATION_STUDY, is_copy_metadata, number
from src.ui.v2.desk import host_status, read_sqlite

COLUMNS = "id, coin, side, status, opened_at, closed_at, entry_price, exit_price, size, leverage, pnl, stop_loss, take_profit, metadata"
STOP_REASONS = {"stop_loss", "trailing_stop"}
MAX_ROWS = 5000


def as_dict(value):
    if not isinstance(value, dict):
        try:
            value = json.loads(value or "{}")
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def stamp(value):
    try:
        value = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def utc_text(value):
    parsed = stamp(value)
    return parsed.isoformat() if parsed else None


def source_label(key):
    if key.startswith("0x") and len(key) >= 16:
        return f"Wallet {key[:8]}...{key[-6:]}"
    return key or "Unattributed copy source"


def normalize(row):
    trade = dict(row)
    meta = as_dict(trade.pop("metadata"))
    entry = as_dict(meta.get("entry_decision"))
    exit = as_dict(meta.get("exit_decision"))
    key = str(meta.get("source_key") or "")
    is_copy = is_copy_metadata(meta)
    wallet = str(meta.get("source_trader") or (key.split(":", 1)[1] if key.startswith("copy_trade:") else ""))
    reason = str(exit.get("reason") or meta.get("close_reason") or (
        "manual_dashboard_close" if meta.get("manual_close") is True and meta.get("close_source") == "dashboard" else ""
    )).strip() or None
    trigger = number(exit.get("trigger", meta.get("stop_trigger_price")))
    initial_stop = number(entry.get("initial_stop", meta.get("initial_stop_loss")))
    fill = number(trade["exit_price"])
    side = str(trade["side"] or "").strip().lower()
    stop_type, gap = None, None
    if reason in STOP_REASONS:
        stop_type = "Unverified stop"
        if side in {"long", "short"} and trigger is not None and trigger > 0:
            if fill is not None and fill > 0:
                gap = max(0.0, (trigger - fill if side == "long" else fill - trigger) / trigger * 10_000)
            if initial_stop is not None and initial_stop > 0:
                if math.isclose(trigger, initial_stop):
                    stop_type = "Original stop"
                elif (side == "long" and trigger > initial_stop) or (side == "short" and trigger < initial_stop):
                    stop_type = "Tightened stop"
    study = as_dict(entry.get("copy_concentration_study"))
    peers = study.get("peers")
    trade.update(
        symbol=str(trade["coin"] or "Unknown").strip().upper(), side=side, meta=meta, entry=entry, exit=exit,
        copy=is_copy, wallet=wallet.lower(), reason=reason, stop_type=stop_type, stop_gap_bps=gap,
        source=source_label(wallet.lower()) if is_copy else str(meta.get("strategy_type") or meta.get("source") or "Unattributed"),
        trigger=trigger, initial_stop=initial_stop, pnl=number(trade["pnl"]),
        fees=number(meta.get("total_fees_paid", meta.get("fees_paid"))),
        excluded=bool(meta.get("tainted") or meta.get("synthetic_reconciliation") or
                      "live_reconciled_closed" in (reason, meta.get("reconciliation_reason"))),
        opened_at=utc_text(trade["opened_at"]), closed_at=utc_text(trade["closed_at"]),
        study_peers=peers if study.get("policy") == COPY_CONCENTRATION_STUDY and type(peers) is int and peers >= 0 else None,
    )
    return trade


def concentration(trades, start, end):
    """Count any nonzero same-asset, same-side overlap inside the review window."""
    keys = defaultdict(list)
    for trade in trades:
        opened, closed = stamp(trade["opened_at"]), stamp(trade["closed_at"])
        if not opened or trade["side"] not in {"long", "short"} or (trade["status"] == "closed" and not closed):
            continue
        lo, hi = max(opened, start), min(closed or end, end)
        if lo < hi:
            key = (trade["symbol"], trade["side"])
            keys[key].extend(((lo, 1, trade["id"]), (hi, -1, trade["id"])))
    overlap, peaks = set(), {}
    for key, points in keys.items():
        active = set()
        for _, action, trade_id in sorted(points):
            if action == -1:
                active.discard(trade_id)
            else:
                if active:
                    overlap.update(active)
                    overlap.add(trade_id)
                active.add(trade_id)
                peaks[key] = max(peaks.get(key, 0), len(active))
    return overlap, peaks


def summary(trades, overlap):
    closed = [t for t in trades if t["status"] == "closed"]
    priced = [t["pnl"] for t in closed if t["pnl"] is not None]
    fees = [t["fees"] for t in closed if t["fees"] is not None]
    stop_gaps = [t["stop_gap_bps"] for t in closed if t["stop_gap_bps"] is not None]
    return {
        "closed": len(closed), "open": sum(t["status"] == "open" for t in trades),
        "priced": len(priced), "wins": sum(pnl > 0 for pnl in priced),
        "pnl": math.fsum(priced) if priced else None, "fees": math.fsum(fees) if fees else None,
        "fee_count": len(fees), "overlap": sum(t["id"] in overlap for t in trades),
        "original": sum(t["stop_type"] == "Original stop" for t in closed),
        "tightened": sum(t["stop_type"] == "Tightened stop" for t in closed),
        "unverified_stops": sum(t["stop_type"] == "Unverified stop" for t in closed),
        "gap_count": len(stop_gaps), "max_gap": max(stop_gaps, default=None),
    }


def shadow_comparison(trades, start):
    """Matched exclusions only; no new orders or reallocation in the comparison."""
    entered = [t for t in trades if t["opened_at"] and stamp(t["opened_at"]) >= start]
    tracked = [t for t in entered if t["study_peers"] is not None]
    priced = [t for t in tracked if t["status"] == "closed" and t["pnl"] is not None and t["closed_at"]]
    retained = [t for t in priced if not t["study_peers"]]
    skipped = [t for t in priced if t["study_peers"]]

    def metrics(rows):
        pnl, peak, drawdown = 0., 0., 0.
        for trade in sorted(rows, key=lambda t: (t["closed_at"], t["id"])):
            pnl += trade["pnl"]
            peak = max(peak, pnl)
            drawdown = max(drawdown, peak - pnl)
        return {"pnl": pnl, "drawdown": drawdown, "closed": len(rows)}

    return {
        "tracked": len(tracked), "unmeasured": len(entered) - len(tracked),
        "pending": len(tracked) - len(priced), "skipped": len(skipped),
        "missed_winners": sum(t["pnl"] > 0 for t in skipped),
        "actual": metrics(priced) if priced else None,
        "exclusion": metrics(retained) if priced else None,
    }


def review(days=7, group="source", scope="copy", *, path=None, now=None):
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    path = Path(path or config.DB_PATH)
    base = {"days": days, "group": group, "scope": scope, "available": False, "limited": False,
            "summary": {}, "groups": [], "recent": [], "excluded": 0, "peak": None}
    if scope == "copy":
        status = host_status(path.with_name("dashboard_ops.json"), now)
        base.update(copy_daily=status.get("copy_daily"), copy_alerts=status.get("copy_alerts") or [])
    try:
        with closing(read_sqlite(path)) as conn:
            rows = conn.execute(f"""
                SELECT {COLUMNS} FROM paper_trades
                WHERE status='open' OR (status='closed' AND closed_at>=? AND closed_at<=?)
                ORDER BY COALESCE(closed_at, opened_at) DESC, id DESC LIMIT ?
                """, (start.isoformat(), now.isoformat(), MAX_ROWS + 1)).fetchall()
    except sqlite3.Error:
        return base
    trades = [normalize(row) for row in rows[:MAX_ROWS]]
    selection = [t for t in trades if scope == "all" or t["copy"]]
    base.update(available=True, limited=len(rows) > MAX_ROWS, recent=selection[:50],
                excluded=sum(t["excluded"] for t in selection))
    if base["limited"]:
        return base  # Never report truncated totals as whole-window results.
    eligible = [t for t in selection if not t["excluded"]]
    copy = [t for t in trades if t["copy"] and not t["excluded"]]
    overlap, peaks = concentration(copy, start, now)
    groups = defaultdict(list)
    for trade in eligible:
        key = (trade["symbol"], trade["side"]) if group == "asset" else trade["side"] if group == "side" else (
            ("copy", trade["wallet"]) if trade["copy"] else ("strategy", trade["source"])
        )
        trade["overlap"] = trade["id"] in overlap
        groups[key].append(trade)
    output = []
    for key, members in groups.items():
        label = (f"{key[0]} {key[1].capitalize()}" if group == "asset" else key.capitalize() or "Unknown"
                 if group == "side" else members[0]["source"].replace("_", " "))
        output.append({"label": label, **summary(members, overlap)})
    output.sort(key=lambda row: (row["pnl"] is None, row["pnl"] or 0, row["label"]))
    base.update(summary=summary(eligible, overlap), groups=output[:100], peak=max(peaks.values(), default=None),
                shadow=shadow_comparison(copy, start))
    return base


def detail(trade_id, *, path=None):
    with closing(read_sqlite(Path(path or config.DB_PATH))) as conn:
        row = conn.execute(f"SELECT {COLUMNS} FROM paper_trades WHERE id=?", (trade_id,)).fetchone()
        if not row:
            return None
        trade = normalize(row)
        decision_id = trade["meta"].get("decision_id")
        linked, events = {}, []
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='decision_snapshots'").fetchone():
            query = """
                SELECT decision_id, created_at, signal_timestamp, firewall_decision, final_status, metadata
                FROM decision_snapshots WHERE paper_trade_id=? OR (decision_id=? AND paper_trade_id IS NULL)
                ORDER BY CASE WHEN paper_trade_id=? THEN 0 ELSE 1 END, created_at LIMIT 2
            """
            matches = conn.execute(query, (trade_id, decision_id if isinstance(decision_id, str) else None, trade_id)).fetchall()
            if len(matches) == 1:
                linked = dict(matches[0])
                if conn.execute("SELECT 1 FROM sqlite_master WHERE name='decision_stage_events'").fetchone():
                    events = [dict(event) for event in conn.execute("""
                        SELECT created_at, stage, status, reason FROM decision_stage_events
                        WHERE decision_id=? ORDER BY created_at,event_id LIMIT 50
                        """, (linked["decision_id"],))]
                    for event in events:
                        event["created_at"] = utc_text(event["created_at"])
        evidence = as_dict(linked.get("metadata"))
        why = as_dict(evidence.get("why_entered"))
        regime = as_dict(evidence.get("market_read"))
        entry = trade["entry"]
        trade.update(
            linked=linked, events=events,
            decision_id=linked.get("decision_id") or decision_id,
            entry_reason=entry.get("reason") or why.get("signal_reason"),
            selected_at=utc_text(entry.get("selected_at")),
            signal_at=utc_text(entry.get("signal_at") or linked.get("signal_timestamp")),
            journal_at=utc_text(linked.get("created_at")),
            regime=entry.get("regime") or regime.get("overall_regime") or trade["meta"].get("regime"),
            firewall=entry.get("firewall") or linked.get("firewall_decision"),
            confidence=number(entry.get("confidence", trade["meta"].get("confidence"))),
        )
        return trade
