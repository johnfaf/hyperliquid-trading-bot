"""Durable, account-scoped equity observations. No broker or order APIs."""
from contextlib import closing
from datetime import datetime, timezone
import math
from pathlib import Path
import sqlite3


def amount(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError, OverflowError):
        return None


def utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        raise ValueError("equity observations require an explicit timezone")
    return value.astimezone(timezone.utc)


class EquityRiskLedger:
    def __init__(self, path):
        self.path = Path(path)

    def observe(self, account, *, now, equity, cash=None, exposure=None, complete=True, day_baseline=None):
        now = utc(now)
        stamp = now.isoformat()
        equity, cash, exposure = map(amount, (equity, cash, exposure))
        baseline = amount(day_baseline)
        complete = bool(complete and equity is not None)
        with closing(sqlite3.connect(self.path, timeout=5)) as conn, conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA synchronous=FULL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS account_equity_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, account TEXT NOT NULL, observed_at TEXT NOT NULL,
                    equity REAL, cash REAL, exposure REAL, complete INTEGER NOT NULL,
                    peak REAL, max_drawdown REAL, day_baseline REAL, baseline_at TEXT
                );
                CREATE INDEX IF NOT EXISTS equity_account_time ON account_equity_samples(account,observed_at,id);
            """)
            conn.execute("BEGIN IMMEDIATE")
            prior = conn.execute("SELECT * FROM account_equity_samples WHERE account=? ORDER BY observed_at DESC,id DESC LIMIT 1",
                                 (account,)).fetchone()
            if prior and prior["observed_at"] > stamp:
                raise ValueError("out-of-order equity observation")
            peak = prior["peak"] if prior else None
            drawdown = prior["max_drawdown"] if prior else None
            if complete:
                peak = max(peak, equity) if peak is not None else equity
                drawdown = max(drawdown or 0., peak - equity)
            first = conn.execute("""SELECT day_baseline,baseline_at FROM account_equity_samples
                WHERE account=? AND observed_at>=? AND day_baseline IS NOT NULL ORDER BY observed_at,id LIMIT 1""",
                                 (account, now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat())).fetchone()
            baseline_at = first["baseline_at"] if first else stamp
            if first:
                baseline = first["day_baseline"]
            elif baseline is None:
                last_good = conn.execute("""SELECT equity,observed_at FROM account_equity_samples
                    WHERE account=? AND complete=1 ORDER BY observed_at DESC,id DESC LIMIT 1""", (account,)).fetchone()
                baseline = last_good["equity"] if last_good else equity if complete else None
                baseline_at = last_good["observed_at"] if last_good else stamp
            conn.execute("""INSERT INTO account_equity_samples
                (account,observed_at,equity,cash,exposure,complete,peak,max_drawdown,day_baseline,baseline_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                         (account, stamp, equity, cash, exposure, complete, peak, drawdown, baseline, baseline_at))

    def status(self, account, *, now, daily_loss_pct=.03, max_drawdown_pct=.10, daily_loss_usd=None, max_age_seconds=900):
        now = utc(now)
        result = {"available": False, "blocked": True, "reason": "equity_history_unavailable"}
        try:
            with closing(sqlite3.connect(self.path.resolve().as_uri()+"?mode=ro", uri=True, timeout=2)) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("""SELECT * FROM account_equity_samples WHERE account=? AND observed_at<=?
                    ORDER BY observed_at DESC,id DESC LIMIT 1""", (account, now.isoformat())).fetchone()
                count, started = conn.execute("SELECT count(*),min(observed_at) FROM account_equity_samples WHERE account=? AND observed_at<=?",
                                              (account, now.isoformat())).fetchone()
        except (OSError, sqlite3.Error):
            return result
        if row is None:
            return result
        row = dict(row)
        age = (now-utc(row["observed_at"])).total_seconds()
        result.update(available=True, equity=row["equity"], cash=row["cash"], exposure=row["exposure"],
                      observed_at=row["observed_at"], capture_started=started, samples=count,
                      peak=row["peak"], max_drawdown=row["max_drawdown"], baseline_at=row["baseline_at"],
                      baseline=row["day_baseline"], age_seconds=age)
        reason = None
        if not 0 <= age <= max_age_seconds:
            reason = "equity_snapshot_stale"
        elif not row["complete"] or row["equity"] is None:
            reason = "equity_quotes_incomplete"
        elif row["equity"] <= 0:
            reason = "equity_not_positive"
        elif row["day_baseline"] is None or row["day_baseline"] <= 0:
            reason = "equity_baseline_unavailable"
        else:
            loss = max(0., row["day_baseline"]-row["equity"])
            current_dd = max(0., row["peak"]-row["equity"])
            budget = row["day_baseline"]*daily_loss_pct
            if daily_loss_usd is not None:
                budget = min(budget, daily_loss_usd) if budget > 0 else daily_loss_usd
            result.update(daily_loss=loss, daily_budget=budget, drawdown=current_dd,
                          drawdown_budget=row["peak"]*max_drawdown_pct)
            if budget > 0 and loss >= budget:
                reason = "daily_equity_loss_limit"
            elif max_drawdown_pct > 0 and current_dd >= row["peak"]*max_drawdown_pct:
                reason = "peak_equity_drawdown_limit"
        result.update(blocked=reason is not None, reason=reason or "active")
        return result

    def window(self, account, *, start, end, max_gap_seconds=900):
        start, end = utc(start), utc(end)
        result = {"complete": False, "samples": 0, "drawdown": None, "first": None, "last": None}
        try:
            with closing(sqlite3.connect(self.path.resolve().as_uri()+"?mode=ro", uri=True, timeout=2)) as conn:
                rows = conn.execute("""SELECT observed_at,equity,complete FROM account_equity_samples
                    WHERE account=? AND observed_at>=? AND observed_at<=? ORDER BY observed_at,id LIMIT 50001""",
                                    (account, start.isoformat(), end.isoformat())).fetchall()
        except (OSError, sqlite3.Error):
            return result
        if not rows or len(rows) > 50000:
            return result
        peak, drawdown, prior = None, 0., start
        complete = True
        for stamp, equity, measured in rows:
            stamp = utc(stamp)
            complete &= bool(measured) and (stamp-prior).total_seconds() <= max_gap_seconds
            prior = stamp
            if measured and equity is not None:
                peak = max(peak, equity) if peak is not None else equity
                drawdown = max(drawdown, peak-equity)
        complete &= (end-prior).total_seconds() <= max_gap_seconds
        result.update(complete=complete, samples=len(rows), drawdown=drawdown if peak is not None else None,
                      first=rows[0][0], last=rows[-1][0])
        return result


PAPER_ACCOUNT = "hyperliquid-paper:main"


def paper_ledger():
    import config
    return EquityRiskLedger(Path(config.DB_PATH).with_suffix(".risk.sqlite3"))


def paper_status(*, now):
    import config
    return paper_ledger().status(PAPER_ACCOUNT, now=now, daily_loss_pct=config.PAPER_DAILY_EQUITY_LOSS_PCT,
                                 max_drawdown_pct=config.PAPER_MAX_EQUITY_DRAWDOWN_PCT)


def observe_paper(mids, *, now):
    from src.data import database as db
    with db.get_connection(for_read=True) as conn:
        if conn.backend == "sqlite":
            conn.execute("BEGIN")
        account = conn.execute("SELECT balance FROM paper_account WHERE id=1").fetchone()
        trades = [dict(row) for row in conn.execute("SELECT coin,side,entry_price,size,leverage FROM paper_trades WHERE status='open'")]
        if not account:
            return
        account = dict(account)
    cash = amount(account.get("balance"))
    equity, exposure, complete = cash, 0., cash is not None
    for trade in trades:
        mark = amount(mids.get(trade["coin"]))
        entry, size, leverage = (amount(trade.get(k)) for k in ("entry_price", "size", "leverage"))
        if None in (mark, entry, size, leverage) or min(mark, entry, size, leverage) <= 0 or trade["side"] not in {"long", "short"}:
            complete = False
            continue
        if equity is not None:
            equity += (mark-entry)*size*leverage*(1 if trade["side"] == "long" else -1)
        exposure += mark*size*leverage
    paper_ledger().observe(PAPER_ACCOUNT, now=now, equity=equity if complete else None,
                           cash=cash, exposure=exposure if complete else None, complete=complete)


def ensure_paper_entry_risk(conn, *, coin, side, entry, size, leverage, stop, metadata, now):
    """Called while the authoritative paper admission transaction is locked."""
    import config
    from src.trading.paper_reporting import fields
    from src.trading.paper_risk import PaperEntryBlocked
    if not config.PAPER_PORTFOLIO_GUARD_ENABLED:
        return
    if fields(metadata.get("entry_decision")).get("firewall") != "approved":
        raise PaperEntryBlocked("approval_missing", "paper entry has no recorded firewall approval")
    state = paper_status(now=utc(now))
    if state["blocked"]:
        raise PaperEntryBlocked("portfolio_equity_risk", "paper portfolio risk: "+state["reason"])
    account = conn.execute("SELECT balance FROM paper_account WHERE id=1").fetchone()
    if not account or amount(account["balance"]) is None:
        raise PaperEntryBlocked("paper_balance_unavailable", "paper balance unavailable")
    basis = min(float(account["balance"]), state["equity"])
    marks_row = conn.execute("SELECT value FROM bot_state WHERE key='paper_mark_prices'").fetchone()
    marks = fields(marks_row["value"]) if marks_row else {}
    seen = marks.get("observed_at")
    positions = conn.execute("SELECT coin,side,entry_price,size,leverage,stop_loss FROM paper_trades WHERE status='open'").fetchall()
    if positions and (not seen or not 0 <= (utc(now)-utc(seen)).total_seconds() <= 900):
        raise PaperEntryBlocked("portfolio_marks_unavailable", "paper portfolio marks unavailable or stale")
    slip = max(config.PAPER_TRADING_SLIPPAGE_MIN_BPS, config.PAPER_TRADING_SLIPPAGE_MAX_BPS, 0.)/10000
    fee = max(config.PAPER_TRADING_TAKER_FEE_BPS, config.PAPER_TRADING_MAKER_FEE_BPS, 0.)/10000

    def risk(direction, mark, fill_entry, quantity, lev, trigger):
        values = tuple(amount(v) for v in (mark, fill_entry, quantity, lev, trigger))
        if None in values or min(values) <= 0 or direction not in {"long", "short"}:
            raise PaperEntryBlocked("position_risk_unmeasured", "paper position protection or marks are unmeasured")
        mark, fill_entry, quantity, lev, trigger = values
        fill = trigger*(1-slip if direction == "long" else 1+slip)
        distance = max(0., mark-fill if direction == "long" else fill-mark)
        return quantity*lev*(distance+(fill_entry+fill)*fee)

    if not ((side == "long" and stop is not None and stop < entry) or (side == "short" and stop is not None and stop > entry)):
        raise PaperEntryBlocked("initial_stop_invalid", "paper initial stop is invalid")
    candidate = risk(side, entry, entry, size, leverage, stop)
    if config.PAPER_MAX_PLANNED_STOP_LOSS_PCT > 0 and candidate > basis*config.PAPER_MAX_PLANNED_STOP_LOSS_PCT+1e-8:
        raise PaperEntryBlocked("per_entry_stop_budget", "paper per-entry stop risk exceeded")
    total = candidate
    for trade in positions:
        total += risk(trade["side"], fields(marks.get("mids")).get(trade["coin"]), trade["entry_price"],
                      trade["size"], trade["leverage"], trade["stop_loss"])
    if total > basis*config.PAPER_MAX_OPEN_STOP_RISK_PCT+1e-8:
        raise PaperEntryBlocked("portfolio_stop_budget", "paper portfolio stop-risk budget exceeded")
