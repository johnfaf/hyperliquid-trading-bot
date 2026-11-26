"""Stock paper decisions and read-only, fill-backed reporting."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from src.trading.alpaca_paper import DAILY_LOSS_USD, REFERENCE_BALANCE_USD
from src.core import clock_provider

BASE_CASH = Decimal(REFERENCE_BALANCE_USD)
NEW_YORK = ZoneInfo("America/New_York")
MODELED_FEE_RATE = Decimal("0.0002")
INCIDENTS = {
    "broker_position_mismatch", "unmanaged_or_unconfirmed_fill", "unmanaged_open_order",
    "submission_unknown", "submitting", "cycle_failed", "stale_broker_clock", "order_rejected", "paper_account_unverified",
}
NEUTRAL = {"outside_entry_or_exit_window", "no_order"}


def number(value):
    try:
        result = Decimal(str(value))
        return result if result.is_finite() and abs(result) <= Decimal("10000000") else None
    except (ArithmeticError, ValueError, TypeError):
        return None


def iso_date(raw):
    try:
        return datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None


class StockEvidence:
    def __init__(self, path: Path, account: str):
        self.path, self.account = path, account
        with closing(sqlite3.connect(path, timeout=8)) as conn, conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS stock_cycles (
                    id TEXT PRIMARY KEY, account TEXT NOT NULL, observed_at TEXT NOT NULL,
                    symbol TEXT NOT NULL, mode TEXT NOT NULL, purpose TEXT NOT NULL, session TEXT,
                    decision TEXT NOT NULL, state TEXT NOT NULL, client_id TEXT, quote_at TEXT,
                    quote_age REAL, quote_valid INTEGER NOT NULL, mark TEXT, cash TEXT, equity TEXT,
                    position TEXT, planned_qty TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_stock_cycles_account_time ON stock_cycles(account, observed_at DESC);
                CREATE INDEX IF NOT EXISTS idx_stock_cycles_session ON stock_cycles(account, session, observed_at);
                CREATE TABLE IF NOT EXISTS stock_cycle_attempts (
                    id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT, purpose TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_stock_cycle_attempts_time ON stock_cycle_attempts(started_at);
            """)

    def start(self, cycle_id: str, purpose: str, *, observed_at: datetime | None = None):
        now = (observed_at or clock_provider.utc_now()).astimezone(timezone.utc)
        with closing(sqlite3.connect(self.path, timeout=8)) as conn, conn:
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("INSERT INTO stock_cycle_attempts VALUES (?, ?, NULL, ?)", (cycle_id, now.isoformat(), purpose))

    def record(self, cycle_id: str, result: dict, *, symbol: str, mode: str, purpose: str,
               observed_at: datetime | None = None, finished: bool = False):
        now = observed_at or clock_provider.utc_now()
        intent = result.get("intent") or {}
        side = result.get("decision") or intent.get("side")
        state = result["state"]
        decision = side if side in {"buy", "sell"} else "no_signal" if state == "no_order" else "closed" if state in NEUTRAL else "blocked"
        quantities = [
            str(parsed) if (parsed := number(result.get(key))) is not None else None
            for key in ("mark", "account_cash", "account_equity", "quantity")
        ]
        with closing(sqlite3.connect(self.path, timeout=8)) as conn, conn:
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("""
                INSERT INTO stock_cycles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET state=CASE
                    WHEN excluded.state='cycle_failed' AND stock_cycles.state='submitting' THEN stock_cycles.state
                    ELSE excluded.state END
                """, (cycle_id, self.account, now.astimezone(timezone.utc).isoformat(), symbol, mode, purpose,
                      result.get("session") or now.astimezone(NEW_YORK).date().isoformat(),
                      decision, state, intent.get("client_order_id"), result.get("quote_at"),
                      result.get("quote_age"), bool(result.get("quote_valid")), *quantities,
                      str(number(intent.get("qty"))) if number(intent.get("qty")) is not None else None))
            if finished:
                conn.execute("UPDATE stock_cycle_attempts SET finished_at=? WHERE id=?",
                             (now.astimezone(timezone.utc).isoformat(), cycle_id))

    def ready_shadow(self, session: str, at: datetime) -> bool:
        at = at.astimezone(timezone.utc)
        with closing(sqlite3.connect(self.path)) as conn:
            return conn.execute("""
                SELECT 1 FROM stock_cycles WHERE account=? AND session=? AND observed_at>=? AND observed_at<=?
                AND mode='shadow' AND purpose='strategy' AND quote_valid=1 AND state IN ('no_order','shadow_intent')
                """, (self.account, session, (at - timedelta(minutes=15)).isoformat(), at.isoformat())).fetchone() is not None

    def drill_position(self, symbol: str) -> Decimal:
        with closing(_reader(self.path)) as conn:
            return sum(
                (Decimal(f["qty"]) * (1 if f["side"] == "buy" else -1)
                 for f in _paper_fills(conn, self.account) if f["purpose"] == "drill" and f["symbol"] == symbol),
                Decimal(0),
            )


def _reader(path: Path):
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
    conn.row_factory = sqlite3.Row
    return conn


def _scope(conn):
    row = conn.execute("SELECT account FROM stock_cycles WHERE account!='unbound' ORDER BY observed_at DESC LIMIT 1").fetchone()
    return row["account"] if row else "unbound"


def _paper_fills(conn, scope: str) -> list[dict]:
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='fills'").fetchone():
        return []
    result = []
    for row in conn.execute("""
        SELECT f.symbol,f.side,f.quantity,f.price,f.occurred_at,o.intent,o.client_id
        FROM fills f JOIN orders o ON o.account=f.account AND o.venue_id=f.order_id
        WHERE f.account=? ORDER BY COALESCE(f.occurred_at,''),f.rowid
        """, (scope,)):
        intent = json.loads(row["intent"])
        result.append({"symbol": row["symbol"], "side": row["side"], "qty": row["quantity"],
                       "price": row["price"], "occurred_at": row["occurred_at"],
                       "purpose": intent.get("_purpose") or "strategy", "ref": row["client_id"][:8]})
    return result


def _model_value(fills, stamp: str, symbol: str, mark: Decimal) -> Decimal | None:
    cash, qty = BASE_CASH, Decimal(0)
    for fill in fills:
        if fill["purpose"] != "strategy" or fill["symbol"] != symbol:
            continue
        if not fill["occurred_at"]:
            return None
        if fill["occurred_at"] > stamp:
            continue
        amount = Decimal(fill["qty"])
        notional = Decimal(fill["price"]) * amount
        direction = 1 if fill["side"] == "buy" else -1
        cash -= direction * notional + notional * MODELED_FEE_RATE
        qty += direction * amount
        if qty < 0:
            return None
    return cash + qty * mark


def scorecard(conn, scope: str, session: str | None) -> dict:
    rows = conn.execute("""
        SELECT * FROM stock_cycles
        WHERE (account=? OR (account='unbound' AND state='cycle_failed')) AND session=? AND purpose='strategy'
        ORDER BY observed_at, id
        """, (scope, session)).fetchall() if session else []
    marks = [row for row in rows if row["quote_valid"] and row["mark"] is not None]
    checked = sum(bool(row["quote_valid"]) or row["state"] == "quote_not_usable" for row in rows)
    result = {"session": session, "cycles": len(rows), "fresh_quotes": len(marks), "quote_checks": checked, "cash_move": None,
              "hold_move": None, "strategy_move": None, "basis_usd": float(BASE_CASH), "mode": None,
              "first_mark": None, "last_mark": None, "state": "Waiting for two fresh observations"}
    if rows:
        result["mode"] = rows[-1]["mode"]
    if len(marks) < 2 or marks[0]["quote_at"] == marks[-1]["quote_at"]:
        return result
    first, last = marks[0], marks[-1]
    start_mark, end_mark = Decimal(first["mark"]), Decimal(last["mark"])
    result.update(cash_move=0.0, hold_move=float(BASE_CASH * (end_mark / start_mark - 1)),
                  first_mark=first["observed_at"], last_mark=last["observed_at"])
    if any(row["mode"] != "paper" for row in rows):
        result["state"] = "Shadow or mixed session; strategy P&L unmeasured"
        return result
    if any(row["state"] in INCIDENTS for row in rows):
        result["state"] = "Execution incident; strategy comparison withheld"
        return result
    fills = _paper_fills(conn, scope)
    relevant = [f for f in fills if f["purpose"] == "strategy" and f["symbol"] == last["symbol"]]
    first_at = first["quote_at"]
    held = sum((Decimal(f["qty"]) * (1 if f["side"] == "buy" else -1) for f in relevant
                if f["occurred_at"] and f["occurred_at"] <= first_at), Decimal(0))
    if not held and not any(not f["occurred_at"] or first_at < f["occurred_at"] <= last["quote_at"] for f in relevant):
        result["state"] = "No confirmed strategy fills; execution unmeasured"
        return result
    start_value = _model_value(fills, first["quote_at"], first["symbol"], start_mark)
    end_value = _model_value(fills, last["quote_at"], last["symbol"], end_mark)
    if start_value is None or end_value is None or start_value <= 0:
        result["state"] = "Incomplete fill history; strategy P&L unmeasured"
        return result
    result.update(strategy_move=float(BASE_CASH * (end_value / start_value - 1)), state="Measured from paper fills")
    return result


def drill_status(conn, scope: str, fills: list[dict]) -> dict:
    rows = conn.execute("""
        SELECT c.observed_at,c.cash,o.status FROM stock_cycles c
        JOIN orders o ON o.account=c.account AND o.client_id=c.client_id
        WHERE c.account=? AND c.purpose='drill' ORDER BY c.observed_at
        """, (scope,)).fetchall()
    if not rows:
        return {"state": "Not run", "fills": 0}
    trades = [fill for fill in fills if fill["purpose"] == "drill"]
    bought = sum((Decimal(f["qty"]) for f in trades if f["side"] == "buy"), Decimal(0))
    sold = sum((Decimal(f["qty"]) for f in trades if f["side"] == "sell"), Decimal(0))
    result = {"state": "Awaiting fills and balance check", "fills": len(trades)}
    if bought <= 0 or bought != sold or not rows[0]["cash"] or not all(f["occurred_at"] for f in trades):
        return result
    last_fill = max(f["occurred_at"] for f in trades)
    after = conn.execute("""
        SELECT cash FROM stock_cycles WHERE account=? AND purpose='strategy' AND mode='shadow'
        AND observed_at>? AND state IN ('no_order','shadow_intent') AND position='0' AND cash IS NOT NULL
        ORDER BY observed_at LIMIT 1
        """, (scope, last_fill)).fetchone()
    if not after:
        return result
    actual = Decimal(after["cash"]) - Decimal(rows[0]["cash"])
    expected = sum((Decimal(f["qty"]) * Decimal(f["price"]) * (1 if f["side"] == "sell" else -1)
                    for f in trades), Decimal(0))
    result.update(state="Reconciled" if abs(expected - actual) <= Decimal(".01") else "Cash difference needs review",
                  observed_cash_move=float(actual), fill_cash_move=float(expected))
    return result


def execution_audit(conn, scope, now):
    """Account-scoped automatic orders only; use exact client IDs for joins."""
    rows = conn.execute("""
        WITH actions AS (
            SELECT client_id,
                   min(CASE WHEN purpose='strategy' THEN observed_at END) AS proposed_at,
                   max(CASE WHEN purpose='operator' THEN 1 ELSE 0 END) AS operator
            FROM stock_cycles
            WHERE account=? AND client_id IS NOT NULL AND mode='paper'
                AND state IN ('submitting','intent_sent','submission_unknown','order_rejected')
            GROUP BY client_id
        )
        SELECT o.client_id,o.venue_id,o.intent,o.status,a.proposed_at,a.operator
        FROM orders o LEFT JOIN actions a ON a.client_id=o.client_id WHERE o.account=? LIMIT 2001
    """, (scope,scope)).fetchall()
    fills = conn.execute("""
        SELECT order_id,quantity,price,occurred_at,symbol,side FROM fills
        WHERE account=? ORDER BY occurred_at,rowid LIMIT 4001
    """, (scope,)).fetchall()
    if len(rows) > 2000 or len(fills) > 4000:
        return {"state": "Execution audit limit reached", "orders": []}
    by_order = {}
    for fill in fills:
        by_order.setdefault(fill["order_id"], []).append(fill)
    orders, natural_fills = [], []
    counts = dict(reserved=0, broker_identified=0, filled=0, partial=0, working=0, rejected=0, canceled=0)
    stuck = evidence_gap = 0
    has_operator = False
    venue_counts = {}
    for row in rows:
        if row["venue_id"]:
            venue_counts[row["venue_id"]] = venue_counts.get(row["venue_id"], 0) + 1
    for row in rows:
        intent = json.loads(row["intent"])
        if intent.get("_purpose") != "strategy":
            continue
        if row["operator"]:
            has_operator = True
            continue
        counts["reserved"] += 1
        counts["broker_identified"] += bool(row["venue_id"])
        intended = number(intent.get("qty"))
        stamp = iso_date(row["proposed_at"])
        linked = by_order.get(row["venue_id"], [])
        sizes = [number(f["quantity"]) for f in linked]
        done = None if None in sizes else sum(sizes, Decimal(0))
        valid = (stamp is not None and stamp.utcoffset() is not None and intended is not None and intended > 0
                 and intent.get("side") in ("buy", "sell") and done is not None and 0 <= done <= intended
                 and (not row["venue_id"] or venue_counts[row["venue_id"]] == 1)
                 and all(size > 0 and f["symbol"] == intent.get("symbol") and f["side"] == intent["side"]
                         and number(f["price"]) is not None and Decimal(f["price"]) > 0
                         and (at := iso_date(f["occurred_at"])) is not None and at.utcoffset() is not None
                         and stamp - timedelta(seconds=10) <= at <= now for f,size in zip(linked,sizes)))
        status = row["status"]
        if not valid or (status == "filled" and done != intended):
            evidence_gap += 1
        elif linked:
            natural_fills.extend(linked)
        counts["filled"] += bool(valid and status == "filled" and done == intended)
        counts["partial"] += bool(valid and 0 < done < intended)
        counts["rejected"] += status == "rejected"
        counts["canceled"] += status == "canceled"
        active = status in {"submitting", "unknown", "accepted", "partial", "cancel_pending"}
        counts["working"] += active
        if active and stamp is not None and stamp.utcoffset() is not None:
            limit = 10 if status in {"submitting", "unknown", "cancel_pending"} else 30
            stuck += now-stamp > timedelta(minutes=limit)
        orders.append({"ref": row["client_id"][:8], "at": row["proposed_at"], "symbol": intent.get("symbol"),
                       "side": intent.get("side"), "status": status, "shares": str(done) if valid else None})
    inventory, round_trips = {}, 0
    for f in sorted(natural_fills, key=lambda f: iso_date(f["occurred_at"])):
        before = inventory.get(f["symbol"], Decimal(0))
        after = before + Decimal(f["quantity"]) * (1 if f["side"] == "buy" else -1)
        inventory[f["symbol"]] = after
        if after < 0:
            evidence_gap += 1
        round_trips += before > 0 and after == 0
    state = ("Review execution" if stuck or evidence_gap else "Natural fills recorded" if natural_fills else
             "Waiting for a natural fill" if orders else "No natural strategy orders yet")
    return {"state": state, **counts, "fills": len(natural_fills), "round_trips": None if has_operator else round_trips,
            "stuck": stuck, "evidence_gap": evidence_gap,
            "orders": sorted(orders, key=lambda row: row["at"] or "", reverse=True)[:8]}


def read_summary(path: Path, *, session: str | None = None) -> dict:
    """Expose only bounded, anonymous decision, fill and scorecard data."""
    with closing(_reader(path)) as conn:
        conn.execute("BEGIN")
        scope = _scope(conn)
        if session is None:
            latest = conn.execute("""
                SELECT max(session) FROM stock_cycles
                WHERE account=? AND purpose='strategy' AND state!='outside_entry_or_exit_window'
                AND state!='waiting_for_paper_credentials'
                """, (scope,)).fetchone()
            session = latest[0]
        cycles = []
        for row in conn.execute("""
            SELECT c.observed_at,c.symbol,c.mode,c.purpose,c.session,c.decision,c.state,c.quote_age,
                   c.quote_valid,c.position,c.planned_qty,c.client_id,o.status
            FROM stock_cycles c LEFT JOIN orders o ON o.account=c.account AND o.client_id=c.client_id
            AND (c.mode='paper' OR c.purpose='drill')
            AND c.state IN ('submitting','intent_sent','submission_unknown','order_rejected')
            WHERE c.account IN (?, 'unbound') ORDER BY c.observed_at DESC,c.rowid DESC LIMIT 30
            """, (scope,)):
            d = dict(row)
            cid = d.pop("client_id")
            d["ref"] = cid[:8] if cid else None
            cycles.append(d)
        fills = _paper_fills(conn, scope)
        coverage = None
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='stock_cycle_attempts'").fetchone():
            now = clock_provider.utc_now()
            row = conn.execute("""
                SELECT count(*), count(finished_at), count(CASE WHEN finished_at IS NULL AND started_at<? THEN 1 END)
                FROM stock_cycle_attempts WHERE purpose='strategy' AND started_at>=?
                """, ((now - timedelta(minutes=10)).isoformat(), (now - timedelta(hours=24)).isoformat())).fetchone()
            coverage = dict(zip(("started", "finished", "overdue"), row))
        return {"decisions": cycles, "fills": fills[-12:][::-1], "scorecard": scorecard(conn, scope, session),
                "drill": drill_status(conn, scope, fills), "coverage": coverage,
                "execution": execution_audit(conn, scope, clock_provider.utc_now()),
                "equity_risk": _equity_risk(path, scope)}


def _equity_risk(path, account):
    from src.trading.equity_risk import EquityRiskLedger
    return EquityRiskLedger(path).status(account, now=clock_provider.utc_now(), daily_loss_usd=float(DAILY_LOSS_USD))
