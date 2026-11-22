"""Immutable public hourly rates and near-settlement oracle evidence for paper trades."""
from contextlib import closing
from datetime import datetime
import math
from pathlib import Path
import sqlite3

from src.trading.paper_reporting import fields
from src.trading.trade_decision_memory import number

HOUR = 3_600_000
ORACLE_MAX_AGE = 120_000
RATE_MAX_JITTER = 30_000
MAX_HOURS = 744
MODEL = "hyperliquid_settled_rate_sampled_oracle_v1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_oracle_samples (
    coin TEXT NOT NULL, hour_ms INTEGER NOT NULL, received_ms INTEGER NOT NULL, price REAL NOT NULL,
    PRIMARY KEY(coin,hour_ms)
);
CREATE TABLE IF NOT EXISTS paper_funding_rates (
    coin TEXT NOT NULL, hour_ms INTEGER NOT NULL, published_ms INTEGER NOT NULL,
    received_ms INTEGER NOT NULL, rate REAL NOT NULL, conflicted INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(coin,hour_ms)
);
"""


def millis(value):
    try:
        value = datetime.fromisoformat(value) if isinstance(value, str) else value
        return int(value.timestamp()*1000) if value is not None and value.utcoffset() is not None else None
    except (AttributeError, ValueError, TypeError, OverflowError):
        return None


def hours_between(opened, closed):
    if opened is None or closed is None or closed < opened:
        return None
    first = (opened//HOUR+1)*HOUR
    count = (closed-first)//HOUR+1
    return list(range(first, first+count*HOUR, HOUR)) if count <= MAX_HOURS else None


def ledger_path(bot_db):
    return Path(bot_db).with_suffix(".funding.sqlite3")


class FundingLedger:
    def __init__(self, path):
        self.path = Path(path)

    def record(self, *, observed_ms, oracle_prices=None, rates=()):
        if type(observed_ms) is not int or observed_ms <= 0:
            raise ValueError("observation requires an integer UTC timestamp")
        target = (observed_ms//HOUR+1)*HOUR
        with closing(sqlite3.connect(self.path, timeout=3)) as conn, conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.executescript(SCHEMA)
            if 0 < target-observed_ms <= ORACLE_MAX_AGE:
                for coin, raw in (oracle_prices or {}).items():
                    price = number(raw)
                    if coin and price is not None and price > 0:
                        conn.execute("""INSERT INTO paper_oracle_samples VALUES(?,?,?,?)
                          ON CONFLICT(coin,hour_ms) DO UPDATE SET price=excluded.price,received_ms=excluded.received_ms
                          WHERE excluded.received_ms>paper_oracle_samples.received_ms""",
                                     (coin, target, observed_ms, price))
            for item in rates:
                event = item.get("time")
                coin, rate = item.get("coin"), number(item.get("fundingRate"))
                if (not isinstance(coin, str) or not coin or type(event) is not int or not 0 < event <= observed_ms
                        or event % HOUR > RATE_MAX_JITTER or rate is None or abs(rate) > .10):
                    raise ValueError("malformed public funding settlement")
                hour = event//HOUR*HOUR
                conn.execute("""INSERT INTO paper_funding_rates VALUES(?,?,?,?,?,0)
                  ON CONFLICT(coin,hour_ms) DO UPDATE SET conflicted=1
                  WHERE paper_funding_rates.rate<>excluded.rate OR paper_funding_rates.published_ms<>excluded.published_ms""",
                             (coin, hour, event, observed_ms, rate))

    def position(self, trade, *, now):
        through, opened = millis(now), millis(trade.get("opened_at"))
        expected = hours_between(opened, through)
        size, lev = number(trade.get("size")), number(trade.get("leverage", 1))
        side, coin = trade.get("side"), trade.get("coin")
        if expected is None or None in (size, lev) or min(size, lev) <= 0 or side not in ("long", "short") or not coin:
            return {"model": MODEL, "state": "invalid_position_or_window", "receipts": [], "missing_hours": [], "known_cost": None}
        units = size*lev*(1 if side == "long" else -1)
        rows = {}
        if expected:
            try:
                with closing(sqlite3.connect(self.path.resolve().as_uri()+"?mode=ro", uri=True, timeout=1)) as conn:
                    rows = {row[0]: row[1:] for row in conn.execute("""
                        SELECT r.hour_ms,r.published_ms,r.received_ms,r.rate,o.received_ms,o.price
                        FROM paper_funding_rates r JOIN paper_oracle_samples o USING(coin,hour_ms)
                        WHERE r.coin=? AND r.hour_ms BETWEEN ? AND ? AND r.conflicted=0 AND r.received_ms<=?
                    """, (coin, expected[0], expected[-1], through))}
            except (OSError, sqlite3.Error):
                pass
        receipts, missing = [], []
        for hour in expected:
            row = rows.get(hour)
            if row is None:
                missing.append(hour)
                continue
            published, received, rate, oracle_ms, price = row
            receipts.append({"hour_ms": hour, "published_ms": published, "received_ms": received, "rate": rate,
                             "oracle_ms": oracle_ms, "oracle_price": price, "signed_units": units,
                             "cost": units*price*rate})
        return {"model": MODEL, "state": "incomplete" if missing else "modeled_complete",
                "receipts": receipts, "missing_hours": missing, "through_ms": through,
                "known_cost": math.fsum(r["cost"] for r in receipts) if receipts or not missing else None}


def funding_for_close(trade, *, now):
    import config

    # Some executors return the pre-insert metadata without DB-added provenance.
    if config.PAPER_FUNDING_LEDGER_ENABLED and trade.get("id"):
        from src.data import database as db
        try:
            with db.get_connection(for_read=True) as conn:
                row = conn.execute("SELECT coin,side,size,leverage,opened_at,metadata FROM paper_trades WHERE id=?",
                                   (trade["id"],)).fetchone()
                if row:
                    trade = dict(row)
        except Exception:
            if fields(trade.get("metadata")).get("paper_funding_model") != MODEL:
                return {"model": MODEL, "state": "incomplete", "receipts": [], "missing_hours": [], "known_cost": None}
    if fields(trade.get("metadata")).get("paper_funding_model") != MODEL:
        return None
    return FundingLedger(ledger_path(config.DB_PATH)).position(trade, now=now)


def verified_cost(trade, coverage):
    """Recheck the exact modeled receipts that were present when the close committed."""
    opened, closed = millis(trade["opened_at"]), millis(trade["closed_at"])
    hours = hours_between(opened, closed)
    if hours is None:
        return None
    if not hours and not coverage:
        return 0.
    if (coverage.get("model") != MODEL or coverage.get("state") != "modeled_complete"
            or coverage.get("through_ms") != closed or coverage.get("missing_hours") != []):
        return None
    receipts = coverage.get("receipts")
    units = number(trade["size"])
    lev = number(trade["leverage"])
    side = trade["side"]
    if (not isinstance(receipts, list) or len(receipts) != len(hours) or None in (units, lev)
            or min(units, lev) <= 0 or side not in ("long", "short")):
        return None
    signed = units*lev*(1 if side == "long" else -1)
    costs = []
    for hour, receipt in zip(hours, receipts, strict=True):
        if not isinstance(receipt, dict) or receipt.get("hour_ms") != hour:
            return None
        published, received, oracle_ms = (receipt.get(k) for k in ("published_ms", "received_ms", "oracle_ms"))
        rate, price, cost = (number(receipt.get(k)) for k in ("rate", "oracle_price", "cost"))
        if (any(type(n) is not int for n in (published, received, oracle_ms))
                or not hour <= published <= min(hour+RATE_MAX_JITTER, received) <= received <= closed
                or not 0 < hour-oracle_ms <= ORACLE_MAX_AGE or None in (rate, price, cost)
                or abs(rate) > .10 or price <= 0 or number(receipt.get("signed_units")) != signed
                or not math.isclose(cost, signed*price*rate, rel_tol=1e-9, abs_tol=1e-9)):
            return None
        costs.append(cost)
    return math.fsum(costs)
