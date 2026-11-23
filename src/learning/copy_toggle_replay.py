"""Paired recorded-book attribution, not an alternate execution backtest."""
from contextlib import closing
from datetime import timedelta
import math

from src.data.decision_coverage import open_readonly
from src.trading.copy_shadow_report import EXTRA_COST_BPS, as_dict, clean_close, realized
from src.trading.paper_reporting import timestamp
from src.trading.trade_decision_memory import is_copy_metadata, number

MAX_ROWS = 12000


def _metrics(rows):
    result = realized([r["pnl"] for r in rows])
    costs = [r["stress_cost"] for r in rows]
    stressed = realized([r["pnl"]-cost for r,cost in zip(rows,costs)]) if None not in costs else None
    pnl = [r["pnl"] for r in rows]
    return {"base": result, "stress": stressed,
            "net_without_largest": result["net"] - max(0., *pnl) if pnl else 0.,
            "reported_fees": math.fsum(r["fees"] for r in rows) if all(r["fees"] is not None for r in rows) else None}


def compare(path, start, end):
    if any(t is None or t.utcoffset() is None for t in (start,end)) or not timedelta(0) < end-start <= timedelta(days=31):
        raise ValueError("use one bounded, timezone-aware interval")
    with closing(open_readonly(path)) as db:
        rows = db.execute("""
            SELECT id,opened_at,closed_at,entry_price,exit_price,size,leverage,pnl,metadata
            FROM paper_trades WHERE julianday(opened_at)<julianday(?)
            AND (closed_at IS NULL OR julianday(closed_at)>=julianday(?))
            ORDER BY julianday(closed_at),id LIMIT ?
        """, (end.isoformat(),start.isoformat(),MAX_ROWS+1)).fetchall()
    if len(rows) > MAX_ROWS:
        raise ValueError("copy attribution exceeds bounded ledger size")
    retained, omitted, all_scored = [],[],[]
    carry, deferred, open_at_end, excluded = 0,0,0,0
    for row in rows:
        meta = as_dict(row["metadata"])
        opened, closed = timestamp(row["opened_at"]), timestamp(row["closed_at"])
        if opened is None or opened >= end or (closed is not None and closed < opened):
            excluded += 1
            continue
        # The toggle is decided only from route and entry time, not P&L.
        copy = is_copy_metadata(meta)
        skip = copy and opened >= start
        if not closed or closed >= end:
            open_at_end += 1
            deferred += int(skip)
            continue
        pnl = number(row["pnl"])
        if closed < start or not clean_close(meta) or pnl is None:
            excluded += 1
            continue
        values = [number(row[k]) for k in ("entry_price","exit_price","size","leverage")]
        cost = None
        if None not in values and min(values) > 0:
            entry, exit_price, size, leverage = values
            cost = number((entry+exit_price)*.5*size*leverage*EXTRA_COST_BPS/10000)
        scored = {"pnl":pnl,"stress_cost":cost,"fees":number(meta.get("total_fees_paid"))}
        all_scored.append(scored)
        (omitted if skip else retained).append(scored)
        carry += int(copy and not skip)
    on, off = _metrics(all_scored), _metrics(retained)
    return {"version":1,"method":"fixed_recorded_fills_no_reallocation","start":start.isoformat(),"end":end.isoformat(),
            "copy_on":on,"copy_off":off,"omitted_copy":_metrics(omitted),"carry_in_copy_closes":carry,
            "open_at_end":open_at_end,"omitted_open_at_end":deferred,"excluded_closes":excluded,
            "delta_base":off["base"]["net"]-on["base"]["net"],
            "delta_stress":off["stress"]["net"]-on["stress"]["net"] if off["stress"] and on["stress"] else None,
            "extra_roundtrip_bps":EXTRA_COST_BPS,"alternate_admissions_measured":False,"live_approved":False}
