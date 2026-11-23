"""Investment research controls. No scorecard can authorize an order."""
from contextlib import closing
from datetime import timedelta
from pathlib import Path
import sqlite3

from src.trading.paper_reporting import fields, finite, timestamp

PROTOCOL = "investment_review_v1"
REFERENCE_CAPITAL = 1000.
MIN_CLOSES = 30
PASS, FAIL, UNKNOWN = "Pass", "Fail", "Insufficient evidence"
MAX_CLOSES = 50000


def check(title, status, note):
    return {"check": title, "status": status, "note": note}


def paper_checkpoint(path, now):
    """Observe both account counters and the independently summed ledger."""
    with closing(sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True, timeout=2)) as conn:
        conn.execute("BEGIN")
        acct = conn.execute("SELECT balance,total_pnl,total_trades,winning_trades FROM paper_account WHERE id=1").fetchone()
        n, priced, pnl, wins = conn.execute("""
            SELECT count(*),count(CASE WHEN typeof(pnl) IN ('real','integer') AND abs(pnl)<1e12 THEN 1 END),
                   sum(pnl),count(CASE WHEN pnl>0 THEN 1 END)
            FROM paper_trades WHERE status='closed'
        """).fetchone()
    if acct is None or n > MAX_CLOSES or n != priced or any(finite(v) is None for v in acct):
        raise ValueError("paper account or closed ledger incomplete")
    return {"observed_at": now.isoformat(), "balance": acct[0], "total_pnl": acct[1], "account_closes": acct[2],
            "account_wins": acct[3], "ledger_closes": n, "ledger_pnl": pnl if n else 0., "ledger_wins": wins}


def paper_cash_check(previous, current):
    label = "Crypto paper cash roll-forward"
    before, after = timestamp(previous.get("observed_at")), timestamp(current.get("observed_at"))
    if not before or not after or not timedelta(hours=18) <= after-before <= timedelta(hours=30):
        return check(label, UNKNOWN, "Two complete daily checkpoints are required.")
    keys = ("balance", "total_pnl", "account_closes", "account_wins", "ledger_closes", "ledger_pnl", "ledger_wins")
    values = [(finite(previous.get(k)), finite(current.get(k))) for k in keys]
    if any(a is None or b is None for a, b in values):
        return check(label, UNKNOWN, "Checkpoint fields incomplete.")
    cash, pnl, acct_n, acct_w, ledger_n, ledger_pnl, ledger_w = (b-a for a,b in values)
    if min(acct_n, acct_w, ledger_n, ledger_w) < 0:
        return check(label, FAIL, "Ledger or account counters decreased. Review reset or history change.")
    if abs(cash-ledger_pnl) > .011 or abs(pnl-ledger_pnl) > .011 or (acct_n, acct_w) != (ledger_n, ledger_w):
        return check(label, FAIL, "Cash, P&L or close counters differ from the ledger movement.")
    note = f"{int(ledger_n)} ledger closes matched cash and account counters."
    return check(label, PASS if ledger_n > 0 else UNKNOWN, note if ledger_n > 0 else "No closes; cash movement matched but execution was not exercised.")


def paper_protection_check(path, now):
    label = "Open paper protection and quote freshness"
    try:
        with closing(sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True, timeout=2)) as conn:
            conn.execute("BEGIN")
            rows = conn.execute("SELECT coin,side,stop_loss FROM paper_trades WHERE status='open' LIMIT 1001").fetchall()
            source = conn.execute("SELECT value FROM bot_state WHERE key='paper_mark_prices'").fetchone()
    except (OSError,sqlite3.Error):
        return check(label, UNKNOWN, "Paper positions or marks unavailable.")
    if not rows or len(rows) > 1000:
        return check(label, UNKNOWN, "No bounded open-position sample.")
    marks = fields(source[0]) if source else {}
    observed = timestamp(marks.get("observed_at"))
    if not observed or not timedelta(0) <= now-observed <= timedelta(minutes=15):
        return check(label, FAIL, "Open positions with missing or stale book marks.")
    mids = fields(marks.get("mids"))
    for coin, side, stop in rows:
        stop, price = finite(stop), finite(mids.get(coin))
        if side not in ("short","long") or None in (stop,price) or min(stop,price) <= 0:
            return check(label, FAIL, "At least one open position has an unusable stop or mark.")
        if (side == "long" and price <= stop) or (side == "short" and price >= stop):
            return check(label, FAIL, "A book mark was through an open position's stop at observation.")
    return check(label, PASS, f"{len(rows)} positions have usable marks and stops. No venue exit implied.")


def economics(forward, monthly_ops):
    label = "BTC frozen strategy economics"
    evidence = {"basis": REFERENCE_CAPITAL, "monthly_ops": monthly_ops, "period_ops": None, "scenarios": {}}
    if forward.get("stale") is not False:
        return check(label, UNKNOWN, "Frozen data unavailable or stale."), evidence
    started, measured, end = (timestamp(forward.get(k)) for k in ("start","evaluated_at","end"))
    if not started or not measured or not end or not started < measured <= end:
        return check(label, UNKNOWN, "Waiting for in-window observations."), evidence
    evidence.update(start=started.isoformat(), through=measured.isoformat())
    period_ops = None if monthly_ops is None else monthly_ops*12*(measured-started).total_seconds()/(365*86400)
    evidence["period_ops"] = period_ops
    scores = forward.get("results") or {}
    for name in ("base","stress"):
        row = scores.get(name) or {}
        values = {k: finite(row.get(k)) for k in ("net","cash","passive","net_without_largest","drawdown_pct")}
        count = row.get("closed")
        if any(v is None for v in values.values()) or type(count) is not int or count < 0:
            return check(label, UNKNOWN, "Both complete cost scenarios are required."), evidence
        evidence["scenarios"][name] = {**values, "closed": count,
            "after_ops": None if period_ops is None else values["net"]-period_ops}
    if forward.get("state") != "Window complete" or measured != end:
        return check(label, UNKNOWN, "The preregistered study is still running."), evidence
    if min(row["closed"] for row in evidence["scenarios"].values()) < MIN_CLOSES:
        return check(label, UNKNOWN, "Too few natural closes for the prespecified screen."), evidence
    for row in evidence["scenarios"].values():
        hurdle = max(row["cash"], row["passive"])
        if row["net"] <= hurdle or row["net_without_largest"] <= row["cash"]:
            return check(label, FAIL, "Failed a benchmark or largest-winner test before overhead."), evidence
        if period_ops is not None and (row["after_ops"] <= hurdle or row["net_without_largest"]-period_ops <= row["cash"]):
            return check(label, FAIL, "Failed a benchmark or largest-winner test after overhead."), evidence
    if period_ops is None:
        return check(label, UNKNOWN, "Actual monthly hosting and data costs not recorded."), evidence
    return check(label, PASS, "Passed the study screen only. Fresh replication and approval still required."), evidence


def build_week(now, weekly, host, cash_check, broker, monthly_ops, protection):
    """Weekly review memo with explicit as-of times for cumulative evidence."""
    from src.trading.paper_reporting import completed_week
    week_start, week_end = completed_week(now - timedelta(minutes=10))
    weekly = weekly or {}
    prior = weekly.get("summary") or {}
    btc, figures = economics(host.get("forward_shadow") or {}, monthly_ops)
    sequence = [btc, check("Stock strategy economics", UNKNOWN,
                      "Stock shadow observations are not broker-filled strategy returns."),
                cash_check, check("Crypto venue statement", UNKNOWN,
                      "No independently reconciled current live statement. Historical fill attribution is unproven.")]
    sequence.append(check("Stock paper identities", broker.get("status", UNKNOWN),
                          broker.get("note", "No fresh read-only broker audit.")))
    sequence.append(check("Stock cash and fault drill", UNKNOWN,
                          "Requires a supervised disposable paper account with reconciled fills and cash."))
    sequence.append(protection)
    funding = host.get("paper_funding") or {}
    if any(funding.get(k, 0) for k in ("overdue_hours","unverified_closes","invalid_open")):
        sequence.append(check("Crypto funding accounting", FAIL, "Unverified position-hours or close arithmetic."))
    elif funding.get("state") == "Funded closes verified" and funding.get("verified_closes",0) > 0:
        sequence.append(check("Crypto funding accounting", PASS, f"{funding['verified_closes']} cross-hour closes verified in 24 hours."))
    else:
        sequence.append(check("Crypto funding accounting", UNKNOWN, "No recent verified cross-hour close."))
    backup = host.get("offhost_restore") or {}
    verified = backup.get("stale") is False and bool(backup.get("verified_at"))
    sequence += [check("Encrypted recovery", PASS if verified else UNKNOWN,
                       "Recent verified off-host restore." if verified else "Restore evidence absent or stale."),
                 check("Independent operator and venue controls", UNKNOWN,
                       "Offline fault tests are not a signed broker/outage drill."),
                 check("Capital allocation", FAIL, "Research only. No live authorization.")]
    return {"version": 1, "protocol": PROTOCOL, "week_start": week_start.isoformat(),
            "week_end": week_end.isoformat(), "reviewed_at": now.isoformat(), "live_approved": False,
            "crypto_week": {k: prior.get(k) for k in ("net","closed","fees","net_without_largest")} if
            (weekly.get("start"), weekly.get("end_exclusive")) == (week_start.isoformat(), week_end.isoformat()) else None,
            "checks": sequence, "btc": figures, "stock": {"basis": REFERENCE_CAPITAL, "mode": "shadow"}}
