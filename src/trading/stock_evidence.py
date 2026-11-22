"""Stock paper decisions and read-only, fill-backed reporting."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from src.trading.alpaca_paper import REFERENCE_BALANCE_USD
from src.core import clock_provider

BASE_CASH = Decimal(REFERENCE_BALANCE_USD)
NEW_YORK = ZoneInfo("America/New_York")
MODELED_FEE_RATE = Decimal("0.0002")
INCIDENTS = {
    "broker_position_mismatch", "unmanaged_or_unconfirmed_fill", "unmanaged_open_order",
    "submission_unknown", "submitting", "cycle_failed", "stale_broker_clock", "order_rejected",
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
                "drill": drill_status(conn, scope, fills), "coverage": coverage}
