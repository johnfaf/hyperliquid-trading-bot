"""Read-only Alpaca paper observation, separate from the stock strategy worker."""
from argparse import ArgumentParser
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import sqlite3
import sys
from tempfile import NamedTemporaryFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.trading.alpaca_paper import AlpacaPaper, paper_account_pin

LIMIT = 2000


def _exact_index(records, key):
    result = {}
    for row in records:
        identity = key(row)
        if not identity or (identity in result and result[identity] != row):
            raise ValueError("ambiguous venue identity")
        result[identity] = row
    return result


def _position(rows):
    values = {}
    for row in rows:
        quantity = Decimal(row["qty"])
        symbol = row["symbol"]
        if not symbol or not quantity.is_finite() or symbol in values:
            raise ValueError("invalid broker position snapshot")
        values[symbol] = quantity
    return {symbol: qty for symbol, qty in values.items() if qty}


def compare(broker, path, *, required_flat_account=None):
    first = broker.request("/v2/account")
    if first.get("status") != "ACTIVE" or first.get("trading_blocked") is not False or not first.get("id"):
        return {"status": "Insufficient evidence", "note": "Paper account not available for verification."}
    if required_flat_account is not None and first["id"] != required_flat_account:
        return {"status":"Fail","note":"The paper account changed during activation."}
    scope = "alpaca-paper:" + first["id"]
    positions = _position(broker.request("/v2/positions"))
    open_orders = broker.request("/v2/orders", query={"status":"open","limit":500})
    if not isinstance(open_orders, list) or len(open_orders) >= 500:
        raise ValueError("open order history not complete")
    fills = _exact_index(broker.fills(), lambda row: row.fill_id)
    last = broker.request("/v2/account")
    if last.get("id") != first["id"] or first.get("cash") != last.get("cash") or positions != _position(broker.request("/v2/positions")):
        return {"status": "Insufficient evidence", "note": "Broker cash or positions changed during the check."}
    with closing(sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True, timeout=2)) as db:
        db.execute("BEGIN")
        known = db.execute("SELECT 1 FROM stock_cycles WHERE account=? LIMIT 1", (scope,)).fetchone()
        if not known:
            return {"status": "Insufficient evidence", "note": "The paper account is not bound to the local journal."}
        order_rows = db.execute("SELECT client_id, venue_id, status FROM orders WHERE account=? LIMIT ?",
                                (scope,LIMIT+1)).fetchall()
        fill_rows = db.execute("""
            SELECT fill_id,order_id,symbol,side,quantity,price FROM fills WHERE account=? LIMIT ?
        """,(scope,LIMIT+1)).fetchall()
    if max(len(order_rows), len(fill_rows), len(fills)) > LIMIT:
        return {"status": "Insufficient evidence", "note": "Journal or broker history exceeds the bounded audit window."}
    orders = {venue: (client,status) for client,venue,status in order_rows if venue}
    if len(orders) != sum(bool(venue) for _,venue,_ in order_rows):
        return {"status": "Fail", "note": "Multiple client IDs claim the same venue order."}
    actual_open = {(str(row["id"]), row["client_order_id"]) for row in open_orders}
    expected_open = {(venue, client) for venue,(client,status) in orders.items() if status in {"accepted","partial"}}
    if actual_open != expected_open or any(
            status in {"submitting","unknown","cancel_pending"} for _,_,status in order_rows):
        return {"status": "Fail", "note": "Unknown order, unresolved submission or open-order mismatch."}
    local_fills = {fid: (oid,symbol,side,Decimal(qty),Decimal(px)) for fid,oid,symbol,side,qty,px in fill_rows}
    remote_fills = {fid: (f.order_id,f.symbol,f.side,f.quantity,f.price) for fid,f in fills.items()}
    if local_fills != remote_fills or any(oid not in orders for oid,_,_,_,_ in remote_fills.values()):
        return {"status": "Fail", "note": "Broker fill IDs or economics do not agree with the scoped journal."}
    inventory = {}
    for _,symbol,side,qty,price in local_fills.values():
        if side not in ("buy","sell") or not qty.is_finite() or not price.is_finite() or min(qty,price) <= 0:
            raise ValueError("invalid fill economics")
        inventory[symbol] = inventory.get(symbol,Decimal(0)) + (qty if side=="buy" else -qty)
    if {k:v for k,v in inventory.items() if v} != positions:
        return {"status":"Fail","note":"Broker position quantities do not match the local fill ledger."}
    if required_flat_account is not None:
        if positions or open_orders:
            return {"status":"Fail","note":"Initial activation requires a flat account with no open orders."}
        return {"status":"Pass","note":"Flat paper account and durable order journal verified."}
    if not local_fills:
        return {"status":"Insufficient evidence","note":"Connected paper account is flat and matched; no fills to verify."}
    return {"status":"Pass","note":f"{len(local_fills)} fill IDs, open orders and positions matched; cash drill separate."}


def arm(broker, ledger, output, *, drill_account):
    account = broker.request("/v2/account")["id"]
    if not drill_account or account == drill_account:
        raise ValueError("strategy and drill paper accounts must be distinct")
    pin = paper_account_pin(account)
    proof = compare(broker, ledger, required_flat_account=account)
    if proof["status"] != "Pass":
        raise ValueError(proof["note"])
    temporary = None
    try:
        with NamedTemporaryFile(dir=output.parent, mode="w", encoding="ascii", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(f"STOCK_PAPER_SEND_ORDERS=true\nSTOCK_PAPER_ACCOUNT_SHA256={pin}\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return proof


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--out", type=Path)
    output.add_argument("--arm-output", type=Path)
    args = parser.parse_args()
    key, secret = os.getenv("STOCK_PAPER_KEY"), os.getenv("STOCK_PAPER_SECRET")
    if args.arm_output:
        if os.getenv("STOCK_PAPER_SEND_ORDERS") != "false" or not key or not secret:
            parser.error("Arming requires the shared paper credentials file in read-only mode.")
        try:
            print(json.dumps(arm(AlpacaPaper(key, secret), args.db, args.arm_output,
                                 drill_account=os.getenv("DRILL_ALPACA_ACCOUNT_ID"))))
        except (ValueError, KeyError, TypeError, OSError, ArithmeticError, RuntimeError, sqlite3.Error):
            parser.error("Paper activation preflight failed; no permission change made.")
        return
    if os.getenv("STOCK_PAPER_SEND_ORDERS") != "false" or not key or not secret:
        result = {"status":"Insufficient evidence","note":"Shadow mode or paper credentials not confirmed."}
    else:
        try:
            result = compare(AlpacaPaper(key, secret), args.db)
        except (ValueError, KeyError, TypeError, OSError, ArithmeticError, RuntimeError, sqlite3.Error):
            result = {"status":"Insufficient evidence","note":"Broker read or journal verification unavailable."}
    result = {"observed_at":datetime.now(timezone.utc).isoformat(), **result}
    with NamedTemporaryFile(dir=args.out.parent, mode="w", encoding="utf-8", delete=False) as output:
        json.dump(result, output)
        temporary = Path(output.name)
    try:
        temporary.replace(args.out)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
