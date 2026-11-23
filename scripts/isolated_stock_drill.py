"""Operator-driven Alpaca paper drill on an empty, separately bound account."""
from argparse import ArgumentParser
from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
from tempfile import NamedTemporaryFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stock_paper import reconcile, submit_once, timestamp
from src.trading.alpaca_paper import AlpacaPaper, DEFAULT_SYMBOL, MAX_QUOTE_AGE_SECONDS, MAX_QUOTE_SPREAD_FRACTION
from src.trading.order_journal import OrderJournal, client_id

PROTOCOL = "isolated_paper_round_trip_v1"
CAP = Decimal("10")


class Blocked(ValueError):
    """Safe operating reason; can be shown without credentials or account fields."""


class LostAcknowledgement:
    def __init__(self, broker):
        self.broker = broker

    def submit(self, intent):
        self.broker.submit(intent)
        raise TimeoutError("drill withheld acknowledgement")


def guard(broker, expected, excluded):
    account = broker.request("/v2/account")
    if (not expected or not excluded or any(not re.fullmatch("[a-f0-9]{64}", key) for key in excluded)
            or account.get("id") != expected):
        raise Blocked("Dedicated account binding or production exclusion unavailable.")
    if sha256(expected.encode()).hexdigest() in excluded:
        raise Blocked("The stock worker account cannot be used for this drill.")
    if account.get("status") != "ACTIVE" or account.get("trading_blocked") is not False:
        raise Blocked("Dedicated paper account is not active.")
    return account


def history(broker):
    rows = broker.request("/v2/orders", query={"status":"all","limit":100})
    if not isinstance(rows,list) or len(rows) >= 100:
        raise Blocked("Cannot verify full dedicated account order history.")
    pairs = {(r["client_order_id"],str(r["id"])) for r in rows}
    if len(rows) != len(pairs):
        raise Blocked("Duplicate order records require review.")
    return pairs


def price_for(broker, side, now):
    clock = broker.request("/v2/clock")
    observed = timestamp(clock["timestamp"])
    remaining = timestamp(clock["next_close"])-observed
    if (not clock["is_open"] or abs((now-observed).total_seconds()) > 30
            or remaining < timedelta(minutes=20 if side == "buy" else 5)):
        raise Blocked("Paper market window or broker clock not suitable.")
    asset = broker.request(f"/v2/assets/{DEFAULT_SYMBOL}")
    if asset.get("class") != "us_equity" or not asset.get("tradable") or not asset.get("fractionable"):
        raise Blocked("Drill instrument unavailable or not fractionable.")
    q = broker.request(f"/v2/stocks/{DEFAULT_SYMBOL}/quotes/latest",market_data=True,query={"feed":"iex"})["quote"]
    bid,ask = Decimal(str(q["bp"])),Decimal(str(q["ap"]))
    age = (now-timestamp(q["t"])).total_seconds()
    if (not -5 <= age <= MAX_QUOTE_AGE_SECONDS or not all(v.is_finite() and v > 0 for v in (bid,ask))
            or ask < bid or (ask-bid)/bid > MAX_QUOTE_SPREAD_FRACTION):
        raise Blocked("No fresh quote within the drill spread bound.")
    return (ask if side == "buy" else bid).quantize(Decimal(".01"),rounding=ROUND_UP if side=="buy" else ROUND_DOWN)


def state_at(path, scope, *, cash=None, ambiguous=False, acknowledge=False):
    with closing(sqlite3.connect(path,timeout=5)) as db, db:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("BEGIN IMMEDIATE")
        db.execute("""CREATE TABLE IF NOT EXISTS drill_baseline (
            id INTEGER PRIMARY KEY CHECK(id=1), account TEXT, cash TEXT,
            ambiguity_seen INTEGER, recovered INTEGER
        )""")
        if cash is not None:
            db.execute("INSERT OR IGNORE INTO drill_baseline VALUES (1,?,?,0,0)",(scope,str(cash)))
        row = db.execute("SELECT account,cash,ambiguity_seen,recovered FROM drill_baseline").fetchone()
        if row and row[0] != scope:
            raise Blocked("The isolated journal is already bound to another account.")
        if row and acknowledge:
            db.execute("UPDATE drill_baseline SET recovered=1 WHERE id=1")
        elif row and ambiguous:
            db.execute("UPDATE drill_baseline SET ambiguity_seen=1 WHERE id=1")
        return (Decimal(row[1]),bool(row[2]) or ambiguous,bool(row[3])) if row else (None,False,False)


def act(broker, path, action, *, expected, excluded, now, confirm=False):
    # Never instantiate a journal or place an order until distinct identity is proved.
    account = guard(broker,expected,excluded)
    scope = f"alpaca-paper:{expected}"
    journal = OrderJournal(path,scope)
    baseline,ambiguous,recovered = state_at(path,scope)
    orders = journal.orders()
    cids = {side:client_id(PROTOCOL,scope,side) for side in ("buy","sell")}
    by_id = {row["client_id"]:row for row in orders}
    if set(by_id)-set(cids.values()):
        raise Blocked("Unexpected order recorded in the drill journal.")
    if action in ("entry","exit") and not confirm:
        raise Blocked("Only a supervised explicit paper submission is permitted.")
    if action in ("preflight","entry") and not orders:
        if history(broker) or broker.request("/v2/positions") or broker.fills():
            raise Blocked("A new, flat paper account without order or fill history is required.")
        cash = Decimal(str(account["cash"]))
        if not cash.is_finite() or cash < CAP:
            raise Blocked("Paper cash insufficient for the bounded drill.")
        if baseline is not None and cash != baseline:
            raise Blocked("Paper cash changed since the drill baseline.")
        if action == "preflight":
            price_for(broker,"buy",now)
            return {"state":"Ready for supervised paper entry","passed":False}
        side = "buy"
        price = price_for(broker,side,now)
        qty = (CAP/price).quantize(Decimal(".000001"),rounding=ROUND_DOWN)
        if qty*price < 1:
            raise Blocked("Order below minimum at the fixed paper-notional cap.")
        state_at(path,scope,cash=cash)
    else:
        if not orders or baseline is None:
            raise Blocked("No durable paper entry reserved.")
        if action in ("entry","preflight"):
            return {"state":"Entry already reserved; use recover","passed":False}
        entry = by_id.get(cids["buy"])
        if not entry:
            raise Blocked("Missing exact entry identity.")
        if action == "recover" and entry["status"] in ("unknown","submitting"):
            _,ambiguous,_ = state_at(path,scope,ambiguous=True)
        positions,problem = reconcile(broker,journal)
        if problem:
            raise Blocked(f"Paper reconciliation pending: {problem}.")
        orders = journal.orders()
        if (any(o["status"] != "filled" or not o["venue_id"] for o in orders)
                or history(broker) != {(o["client_id"],o["venue_id"]) for o in orders}):
            raise Blocked("Expected filled orders must match complete paper history.")
        qty = positions.get(DEFAULT_SYMBOL,Decimal(0))
        buy_qty = Decimal(next(o for o in orders if o["entry"])["intent"]["qty"])
        if action == "recover":
            if len(orders) != 1 or qty != buy_qty or set(positions) != {DEFAULT_SYMBOL}:
                raise Blocked("Entry inventory does not match the one reserved buy.")
            if ambiguous or recovered:
                state_at(path,scope,acknowledge=True)
                return {"state":"Restart recovered one paper order without resubmission","passed":False}
            raise Blocked("No durable unknown acknowledgement to verify.")
        if not recovered:
            raise Blocked("Verify restart recovery before exiting or certifying.")
        if action == "verify":
            if len(orders) != 2 or positions:
                raise Blocked("Both fills and zero final inventory are required.")
            with closing(sqlite3.connect(path)) as db:
                fills = db.execute("SELECT side,quantity,price FROM fills WHERE account=?",(scope,)).fetchall()
            if {side for side,_,_ in fills} != {"buy","sell"}:
                raise Blocked("No matched buy and sell fill records.")
            fill_cash = sum((Decimal(q)*Decimal(p)*(1 if side=="sell" else -1) for side,q,p in fills),Decimal(0))
            observed = Decimal(str(guard(broker,expected,excluded)["cash"])) - baseline
            if not observed.is_finite() or abs(fill_cash-observed) > Decimal(".01"):
                raise Blocked("Final paper cash does not reconcile to the cent.")
            return {"state":"Reconciled","passed":True,"fills":len(fills),"cash_delta":str(observed),
                    "unique_orders":2,"flat":True,"restart_recovered":True}
        if action != "exit" or len(orders) != 1 or qty != buy_qty or set(positions) != {DEFAULT_SYMBOL}:
            raise Blocked("Only the verified drill inventory can be sold.")
        side = "sell"
        price = price_for(broker,side,now)
    # Guard again immediately before releasing the one permitted submission.
    latest = guard(broker,expected,excluded)
    if side=="buy" and Decimal(str(latest["cash"])) != cash:
        raise Blocked("Paper cash changed before submitting the drill entry.")
    intent = {"client_order_id":cids[side],"symbol":DEFAULT_SYMBOL,"side":side,"qty":str(qty),
              "limit_price":str(price),"type":"limit","time_in_force":"day","extended_hours":False}
    transport = LostAcknowledgement(broker) if side=="buy" else broker
    result = submit_once(transport,journal,intent,purpose="isolated_drill")
    return {"state":result,"passed":False}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--action",choices=["preflight","entry","recover","exit","verify"],required=True)
    parser.add_argument("--confirm-paper-submit",action="store_true")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    key,secret = os.getenv("DRILL_ALPACA_KEY"),os.getenv("DRILL_ALPACA_SECRET")
    if not key or not secret:
        result = {"state":"Awaiting a separate paper account","passed":False}
    else:
        try:
            result = act(AlpacaPaper(key,secret),args.db,args.action,expected=os.getenv("DRILL_ALPACA_ACCOUNT_ID"),
                         excluded=set(os.getenv("DRILL_EXCLUDED_ACCOUNTS_SHA256","").split()),now=now,confirm=args.confirm_paper_submit)
        except Blocked as exc:
            result = {"state":str(exc),"passed":False}
        except (OSError,ArithmeticError,ValueError,KeyError,TypeError,RuntimeError,sqlite3.Error):
            result = {"state":"Paper broker or journal evidence unavailable; do not retry an order.","passed":False}
    result.update(protocol=PROTOCOL,observed_at=now.isoformat(),action=args.action)
    if result["passed"]:
        with NamedTemporaryFile(dir=args.db.parent,mode="w",encoding="utf-8",delete=False) as f:
            json.dump(result,f)
            temp = Path(f.name)
        try:
            temp.replace(args.db.with_name("verified.json"))
        finally:
            temp.unlink(missing_ok=True)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
