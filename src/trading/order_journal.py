"""Durable per-account execution identities and venue-confirmed fills."""
from contextlib import closing, contextmanager
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import sqlite3


@dataclass(frozen=True)
class Fill:
    fill_id: str
    order_id: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal


def client_id(*parts: str) -> str:
    return sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()[:32]


class OrderJournal:
    def __init__(self, path: Path, account: str):
        self.path = path
        self.account = account
        with self._connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS orders (
                    account TEXT NOT NULL, client_id TEXT NOT NULL, intent TEXT NOT NULL,
                    status TEXT NOT NULL, venue_id TEXT, entry INTEGER NOT NULL,
                    PRIMARY KEY (account, client_id)
                );
                CREATE TABLE IF NOT EXISTS fills (
                    account TEXT NOT NULL, fill_id TEXT NOT NULL, order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL, side TEXT NOT NULL, quantity TEXT NOT NULL, price TEXT NOT NULL,
                    PRIMARY KEY (account, fill_id)
                );
                CREATE TABLE IF NOT EXISTS pauses (
                    account TEXT PRIMARY KEY, reason TEXT NOT NULL
                );
            """)

    @contextmanager
    def _connection(self):
        with closing(sqlite3.connect(self.path, timeout=8)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA synchronous=FULL")
            yield db

    def reserve(self, cid: str, intent: dict, *, entry: bool) -> bool:
        """Claim before network I/O. A previous claim is never permission to resend."""
        encoded = json.dumps(intent, sort_keys=True, separators=(",", ":"))
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute(
                "SELECT intent FROM orders WHERE account=? AND client_id=?", (self.account, cid),
            ).fetchone()
            if previous:
                if previous["intent"] != encoded:
                    raise ValueError("client order ID reused for a different intent")
                return False
            if entry and (
                db.execute("SELECT 1 FROM pauses WHERE account=?", (self.account,)).fetchone()
                or db.execute(
                    "SELECT 1 FROM orders WHERE account=? AND status IN ('submitting','unknown','cancel_pending')",
                    (self.account,),
                ).fetchone()
            ):
                return False
            db.execute("INSERT INTO orders VALUES (?, ?, ?, 'submitting', NULL, ?)", (self.account, cid, encoded, entry))
            return True

    def observe(self, cid: str, status: str, venue_id: str | None = None) -> None:
        if status not in {"unknown", "accepted", "partial", "filled", "canceled", "rejected", "cancel_pending"}:
            raise ValueError("unrecognized order status")
        with self._connection() as db:
            cursor = db.execute(
                "UPDATE orders SET status=?, venue_id=COALESCE(?,venue_id) WHERE account=? AND client_id=? "
                "AND (venue_id IS NULL OR ? IS NULL OR venue_id=?)",
                (status, venue_id, self.account, cid, venue_id, venue_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("order observation did not match the reserved identity")

    def orders(self) -> list[dict]:
        with self._connection() as db:
            return [
                {**dict(row), "intent": json.loads(row["intent"])}
                for row in db.execute("SELECT * FROM orders WHERE account=? ORDER BY rowid", (self.account,))
            ]

    def unresolved(self) -> list[str]:
        with self._connection() as db:
            return [row[0] for row in db.execute(
                "SELECT client_id FROM orders WHERE account=? AND status IN "
                "('submitting','unknown','cancel_pending') ORDER BY rowid",
                (self.account,),
            )]

    def status(self, cid: str) -> str | None:
        with self._connection() as db:
            row = db.execute("SELECT status FROM orders WHERE account=? AND client_id=?", (self.account, cid)).fetchone()
            return row["status"] if row else None

    def set_pause(self, reason: str | None) -> None:
        with self._connection() as db:
            if reason:
                db.execute("INSERT INTO pauses VALUES (?, ?) ON CONFLICT(account) DO UPDATE SET reason=excluded.reason",
                           (self.account, reason))
            else:
                db.execute("DELETE FROM pauses WHERE account=?", (self.account,))

    def pause_reason(self) -> str | None:
        with self._connection() as db:
            row = db.execute("SELECT reason FROM pauses WHERE account=?", (self.account,)).fetchone()
            return row["reason"] if row else None

    def record_fills(self, fills: list[Fill]) -> int:
        added = 0
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            for fill in fills:
                if (
                    not all((fill.fill_id, fill.order_id, fill.symbol)) or fill.side not in {"buy", "sell"}
                    or not all(n.is_finite() and n > 0 for n in (fill.quantity, fill.price))
                ):
                    raise ValueError("invalid broker fill")
                values = (fill.order_id, fill.symbol, fill.side, str(fill.quantity.normalize()), str(fill.price.normalize()))
                prior = db.execute(
                    "SELECT order_id,symbol,side,quantity,price FROM fills WHERE account=? AND fill_id=?",
                    (self.account, fill.fill_id),
                ).fetchone()
                if prior:
                    if tuple(prior) != values:
                        raise ValueError("conflicting duplicate broker fill")
                else:
                    db.execute("INSERT INTO fills VALUES (?, ?, ?, ?, ?, ?, ?)", (self.account, fill.fill_id, *values))
                    added += 1
        return added

    def inventory(self) -> dict[str, Decimal]:
        result: dict[str, Decimal] = {}
        with self._connection() as db:
            for symbol, side, qty in db.execute(
                "SELECT symbol,side,quantity FROM fills WHERE account=? ORDER BY rowid", (self.account,),
            ):
                result[symbol] = result.get(symbol, Decimal(0)) + Decimal(qty) * (1 if side == "buy" else -1)
        return {symbol: qty for symbol, qty in result.items() if qty}
