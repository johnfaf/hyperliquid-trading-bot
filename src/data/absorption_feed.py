"""Compact, receive-time-aligned capture of public Hyperliquid trade and L2 data."""
from collections import Counter, deque
from dataclasses import astuple
import json
import math
import sqlite3
from uuid import uuid4

from src.signals.flow_absorption import Frame, HISTORY, MINUTE_MS, RULE_ID, SLOT_MS, absorption

COINS = ("BTC", "ETH")
FRESH_MS = 3_000
FLOW_FRESH_MS = 30_000
RETENTION_MS = 60 * 86_400_000
SCHEMA = """
    CREATE TABLE IF NOT EXISTS frames (
        coin TEXT, at INTEGER, session TEXT, state TEXT, bid REAL, ask REAL, bid_usd REAL, ask_usd REAL,
        bid_depth REAL, ask_depth REAL, buy REAL, sell REAL, trades INTEGER,
        book_venue_ms INTEGER, book_received_ms INTEGER, trade_venue_ms INTEGER, trade_received_ms INTEGER,
        sign_agrees INTEGER, sign_disagrees INTEGER, PRIMARY KEY(coin,at)
    ) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS decisions (
        coin TEXT, at INTEGER, rule TEXT, session TEXT, state TEXT, side INTEGER, pressure INTEGER, features TEXT,
        PRIMARY KEY(coin,at)
    ) WITHOUT ROWID;
"""


def storage(path):
    db = sqlite3.connect(path, timeout=4)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(SCHEMA)
    return db


def number(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 1e12:
        raise ValueError("invalid market value")
    return value


def sided_book(data, received):
    stamp = data["time"]
    sides = data["levels"]
    if type(stamp) is not int or not -500 <= received - stamp <= FRESH_MS or len(sides) != 2:
        raise ValueError("unexpected book time or shape")
    bids, asks = [[(number(level["px"]), number(level["sz"])) for level in side[:5]] for side in sides]
    bid, ask = bids[0][0], asks[0][0]
    if (len(bids) != 5 or len(asks) != 5 or bid >= ask or any(a[0] <= b[0] for a, b in zip(bids, bids[1:]))
            or any(a[0] >= b[0] for a, b in zip(asks, asks[1:]))):
        raise ValueError("crossed or unordered order book")
    buy, sell = (math.fsum(p * q for p, q in rows) for rows in (bids, asks))
    return stamp, received, bid, ask, number(bid * bids[0][1]), number(ask * asks[0][1]), number(buy), number(sell)


class Capture:
    """One independent connection epoch; a restart never fills its missing slots retroactively."""

    def __init__(self, db, start):
        self.db = db
        self.start = start
        self.session = uuid4().hex
        self.next_at = (start // SLOT_MS + 1) * SLOT_MS
        self.last_received = start
        self.books = {}
        self.flow = {coin: Counter() for coin in COINS}
        self.trade_times = {}
        self.last_trade_received = {}
        self.seen, self.ordered = set(), deque()
        self.history = {coin: deque(maxlen=HISTORY) for coin in COINS}
        self.bad = set()
        self.first = True

    def ingest(self, msg, received):
        self.advance(received)
        channel, data = msg.get("channel"), msg.get("data")
        if channel == "l2Book" and isinstance(data, dict) and (coin := data.get("coin")) in COINS:
            try:
                book = sided_book(data, received)
                if coin not in self.books or self.books[coin][0] <= book[0]:
                    self.books[coin] = book
            except (KeyError, ValueError, TypeError, IndexError, OverflowError):
                self.books.pop(coin, None)
                self.bad.add(coin)
        elif channel == "trades" and isinstance(data, list):
            while self.ordered and self.ordered[0][0] < received - 60_000:
                _, key = self.ordered.popleft()
                self.seen.discard(key)
            for event in data:
                if isinstance(event, dict) and (coin := event.get("coin")) in COINS:
                    self._trade(coin, event, received)
        elif channel in {"trades", "l2Book"}:
            self.bad.update(COINS)

    def _trade(self, coin, event, received):
        try:
            stamp, ident, side = event["time"], event["tid"], event["side"]
            if type(stamp) is not int or type(ident) is not int or ident < 0 or side not in ("A", "B"):
                raise ValueError("unexpected trade identity")
            key = coin, stamp, ident
            if stamp < self.start or key in self.seen:
                return
            if not 0 <= received - stamp <= FRESH_MS:
                raise ValueError("late trade")
            price = number(event["px"])
            usd = number(price * number(event["sz"]))
        except (KeyError, ValueError, TypeError, OverflowError):
            self.bad.add(coin)
            return
        self.seen.add(key)
        self.ordered.append((received, key))
        if len(self.seen) > 50_000:
            raise ValueError("trade buffer exceeded a safe size; start a new capture session")
        bucket = self.flow[coin]
        bucket["buy" if side == "B" else "sell"] += usd
        bucket["trades"] += 1
        self.trade_times[coin] = (stamp, received)
        self.last_trade_received[coin] = received
        book = self.books.get(coin)
        # This is a noisy sanity check, not a substitute for the venue side field.
        if book and 0 <= stamp - book[0] <= 1_000 and (price >= book[3] or price <= book[2]):
            bucket["sign_agrees" if (price >= book[3]) == (side == "B") else "sign_disagrees"] += 1

    def advance(self, now):
        if now < self.last_received:
            raise ValueError("wall clock moved backwards; start a new capture session")
        self.last_received = now
        if now < self.next_at:
            return
        at = self.next_at
        delay = now - at
        self.next_at = (now // SLOT_MS + 1) * SLOT_MS
        with self.db:
            for coin in COINS:
                frame, evidence = self._frame(coin, at, delay)
                self.db.execute("INSERT OR IGNORE INTO frames VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                (coin, *astuple(frame), *evidence))
                history = self.history[coin]
                history.append(frame)
                if at % MINUTE_MS == 0:
                    decision = absorption(history)
                    self.db.execute("INSERT OR IGNORE INTO decisions VALUES(?,?,?,?,?,?,?,?)",
                                    (coin, at, RULE_ID, self.session, decision["state"], decision["side"],
                                     decision["pressure"], json.dumps(decision, allow_nan=False)))
                self.flow[coin].clear()
            if at % 3_600_000 == 0:
                for table in ("frames", "decisions"):
                    for coin in COINS:
                        self.db.execute(f"DELETE FROM {table} WHERE coin=? AND at<?", (coin, at - RETENTION_MS))
        self.bad.clear()
        self.trade_times.clear()
        self.first = False

    def _frame(self, coin, at, delay):
        book = self.books.get(coin)
        flow = self.flow[coin]
        if self.first or delay >= 1_500:
            state = "partial_slot"
        elif coin in self.bad:
            state = "bad_message"
        elif (not book or not 0 <= at - book[0] <= FRESH_MS
                or not 0 < at - book[1] <= FRESH_MS):
            state = "book_stale"
        elif not 0 < at - self.last_trade_received.get(coin, at) <= FLOW_FRESH_MS:
            state = "flow_stale"
        else:
            state = "ready"
        frame = Frame(at, self.session, state, *(book[2:] if state == "ready" else (None,) * 6),
                      flow["buy"], flow["sell"], flow["trades"])
        return frame, (*(book[:2] if book else (None, None)), *self.trade_times.get(coin, (None, None)),
                       flow["sign_agrees"], flow["sign_disagrees"])
