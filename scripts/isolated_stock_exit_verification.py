"""Supervised sell-acknowledgment verification on the verified Alpaca Paper drill account."""

import argparse
from contextlib import closing
from datetime import UTC, datetime
from decimal import Decimal, ROUND_DOWN
import json
import os
from pathlib import Path
import sqlite3
import sys
from tempfile import NamedTemporaryFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.isolated_stock_drill import (
    CAP,
    PROTOCOL as LEGACY,
    Blocked,
    LostAcknowledgement,
    guard,
    history,
    price_for,
)
from scripts.run_stock_paper import reconcile, submit_once
from src.trading.alpaca_paper import AlpacaPaper, DEFAULT_SYMBOL
from src.trading.order_journal import OrderJournal, client_id

PROTOCOL = "isolated_paper_sell_ack_v2"
ACTIONS = ("preflight", "entry", "exit", "verify")


def seed(legacy_dir, target, scope):
    receipt = json.loads((legacy_dir / "legacy-receipt").read_text(encoding="utf-8"))
    if (
        receipt.get("protocol") != LEGACY
        or receipt.get("action") != "verify"
        or receipt.get("passed") is not True
        or receipt.get("flat") is not True
        or receipt.get("unique_orders") != 2
        or receipt.get("restart_recovered") is not True
    ):
        raise Blocked("A verified, flat previous paper drill is required.")
    source = legacy_dir / "legacy-db"
    with closing(
        sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=5)
    ) as db:
        rows = db.execute("SELECT account,client_id,status FROM orders").fetchall()
        required = {
            (scope, client_id(LEGACY, scope, side), "filled")
            for side in ("buy", "sell")
        }
        if len(rows) != 2 or set(rows) != required:
            raise Blocked(
                "Previous drill journal does not match this dedicated account."
            )
        if not target.exists():
            with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
            try:
                with closing(sqlite3.connect(temporary)) as copy:
                    db.backup(copy)
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    pass
            finally:
                temporary.unlink(missing_ok=True)


def checked_amount(value):
    amount = Decimal(value)
    if not amount.is_finite() or amount < 0:
        raise Blocked("Invalid amount in broker evidence.")
    return amount


def bind_cash(path, scope, balance=None):
    with closing(sqlite3.connect(path, timeout=5)) as db, db:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("BEGIN IMMEDIATE")
        db.execute("""CREATE TABLE IF NOT EXISTS exit_verification (
            id INTEGER PRIMARY KEY CHECK(id=1), account TEXT NOT NULL, cash TEXT NOT NULL
        )""")
        if balance is not None:
            db.execute(
                "INSERT OR IGNORE INTO exit_verification VALUES(1,?,?)",
                (scope, str(balance)),
            )
        baseline = db.execute(
            "SELECT account,cash FROM exit_verification WHERE id=1"
        ).fetchone()
        if baseline and baseline[0] != scope:
            raise Blocked("Exit verification journal is bound to another account.")
        return checked_amount(baseline[1]) if baseline else None


def validate_orders(broker, journal, orders, expected):
    known = {(row["client_id"], row["venue_id"]) for row in orders}
    if set(row["client_id"] for row in orders) - expected or history(broker) != known:
        raise Blocked("Complete paper order history does not match the journal.")
    with closing(sqlite3.connect(journal.path)) as db:
        fills = db.execute(
            "SELECT order_id,side,quantity,price FROM fills WHERE account=?",
            (journal.account,),
        ).fetchall()
    quantities = {}
    for oid, _, qty, _ in fills:
        quantities[oid] = quantities.get(oid, Decimal(0)) + checked_amount(qty)
    for row in orders:
        remote = broker.lookup(row["client_id"])
        if (
            row["status"] != "filled"
            or not remote
            or remote.get("status") != "filled"
            or remote.get("id") != row["venue_id"]
            or remote.get("client_order_id") != row["client_id"]
            or checked_amount(remote["filled_qty"])
            != checked_amount(row["intent"]["qty"])
            or quantities.get(row["venue_id"]) != checked_amount(row["intent"]["qty"])
        ):
            raise Blocked(
                "Exact broker fills are not verified; do not send another order."
            )
    return fills


def execute(
    broker, path, action, *, expected, excluded, now, legacy_dir, confirm=False
):
    if action not in ACTIONS or (action in {"entry", "exit"} and not confirm):
        raise Blocked("Explicit supervised paper submission is required.")
    account = guard(broker, expected, excluded)
    scope = f"alpaca-paper:{expected}"
    seed(legacy_dir, path, scope)
    journal = OrderJournal(path, scope)
    old = {client_id(LEGACY, scope, side) for side in ("buy", "sell")}
    buy, sell = (client_id(PROTOCOL, scope, side) for side in ("buy", "sell"))
    before = {row["client_id"]: row for row in journal.orders()}
    if not old <= before.keys() or before.keys() - (old | {buy, sell}):
        raise Blocked("Unexpected order identity in isolated journal.")
    if (action == "entry" and buy in before) or (action == "exit" and sell in before):
        return {"state": "already_reserved_do_not_resubmit", "passed": False}
    if action == "verify" and (before.get(sell) or {}).get("status") in {
        "submitting",
        "unknown",
    }:
        # Record the pre-reconciliation state so fill lag cannot erase the recovery evidence.
        with closing(sqlite3.connect(path, timeout=5)) as db, db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS sell_recovery (cid TEXT PRIMARY KEY)"
            )
            db.execute("INSERT OR IGNORE INTO sell_recovery VALUES(?)", (sell,))
    positions, problem = reconcile(broker, journal)
    if problem:
        return {"state": problem, "passed": False}
    orders = journal.orders()
    fills = validate_orders(broker, journal, orders, old | {buy, sell})
    qty = positions.get(DEFAULT_SYMBOL, Decimal(0))
    balance = checked_amount(account["cash"])
    baseline = bind_cash(path, scope)
    if action == "preflight":
        if positions or set(before) != old or balance < CAP:
            raise Blocked(
                "Preflight requires a flat dedicated account with only its verified history."
            )
        clock = broker.request("/v2/clock")
        if type(clock.get("is_open")) is not bool:
            raise Blocked("Paper market clock could not be verified.")
        if clock["is_open"]:
            price_for(broker, "buy", now)
        return {
            "state": "preflight_verified_open"
            if clock["is_open"]
            else "preflight_verified_market_closed",
            "passed": False,
            "account_verified": True,
            "flat": True,
            "open_orders": 0,
        }
    if action == "verify":
        if baseline is None or set(before) != old | {buy, sell} or positions:
            raise Blocked(
                "Verification requires both new orders filled and zero inventory."
            )
        with closing(sqlite3.connect(path)) as db:
            recovered = (
                db.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='sell_recovery'"
                ).fetchone()
                and db.execute(
                    "SELECT 1 FROM sell_recovery WHERE cid=?", (sell,)
                ).fetchone()
            )
        if not recovered:
            raise Blocked("No durable evidence of recovery from an unknown sell.")
        new_ids = {row["venue_id"] for row in orders if row["client_id"] in {buy, sell}}
        delta = sum(
            (
                Decimal(q) * Decimal(p) * (1 if side == "sell" else -1)
                for oid, side, q, p in fills
                if oid in new_ids
            ),
            Decimal(0),
        )
        observed = checked_amount(guard(broker, expected, excluded)["cash"]) - baseline
        if abs(delta - observed) > Decimal(".01"):
            raise Blocked("Paper cash does not reconcile within one cent.")
        sell_id = next(row["venue_id"] for row in orders if row["client_id"] == sell)
        sell_fills = sum(oid == sell_id for oid, *_ in fills)
        return {
            "state": "sell_ack_recovered_and_flat",
            "passed": True,
            "flat": True,
            "open_orders": 0,
            "new_orders": 2,
            "sell_fill_records": sell_fills,
            "partial_sell_fill_observed": sell_fills > 1,
        }
    if action == "entry":
        if (
            set(before) != old
            or positions
            or balance < CAP
            or (baseline is not None and baseline != balance)
        ):
            raise Blocked("Entry requires the unchanged, flat isolated account.")
        side, cid = "buy", buy
        price = price_for(broker, side, now)
        qty = (CAP / price).quantize(Decimal(".000001"), rounding=ROUND_DOWN)
        if qty * price < 1:
            raise Blocked("Entry below minimum at the fixed paper budget.")
        if bind_cash(path, scope, balance) != balance:
            raise Blocked("Paper cash changed during entry preparation.")
    else:
        if (
            baseline is None
            or set(before) != old | {buy}
            or set(positions) != {DEFAULT_SYMBOL}
            or qty != Decimal(before[buy]["intent"]["qty"])
        ):
            raise Blocked("Only the fully verified v2 drill inventory can be sold.")
        side, cid = "sell", sell
        price = price_for(broker, side, now)
    if checked_amount(guard(broker, expected, excluded)["cash"]) != balance:
        raise Blocked("Account cash changed before submission.")
    order = {
        "client_order_id": cid,
        "symbol": DEFAULT_SYMBOL,
        "side": side,
        "qty": str(qty),
        "type": "limit",
        "limit_price": str(price),
        "time_in_force": "day",
        "extended_hours": False,
    }
    transport = LostAcknowledgement(broker) if side == "sell" else broker
    return {
        "state": submit_once(transport, journal, order, purpose=PROTOCOL),
        "passed": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action", choices=ACTIONS, required=True)
    p.add_argument("--state-dir", type=Path, required=True)
    p.add_argument("--confirm-paper-submit", action="store_true")
    args = p.parse_args()
    now = datetime.now(UTC)
    try:
        key, secret = os.environ["DRILL_ALPACA_KEY"], os.environ["DRILL_ALPACA_SECRET"]
        if not key or not secret:
            raise Blocked("Dedicated paper credentials are unavailable.")
        result = execute(
            AlpacaPaper(key, secret),
            args.state_dir / "verification.sqlite3",
            args.action,
            expected=os.environ["DRILL_ALPACA_ACCOUNT_ID"],
            excluded=set(os.environ["DRILL_EXCLUDED_ACCOUNTS_SHA256"].split()),
            now=now,
            legacy_dir=Path(os.environ["CREDENTIALS_DIRECTORY"]),
            confirm=args.confirm_paper_submit,
        )
    except Blocked as exc:
        result = {"state": str(exc), "passed": False}
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        ArithmeticError,
        RuntimeError,
        sqlite3.Error,
    ):
        result = {
            "state": "Account or journal evidence unavailable; do not resubmit.",
            "passed": False,
        }
    result.update(
        protocol=PROTOCOL,
        action=args.action,
        observed_at=now.isoformat(),
        paper_approval=False,
        live_approval=False,
    )
    if result["passed"]:
        with NamedTemporaryFile(
            dir=args.state_dir, mode="w", delete=False, encoding="utf-8"
        ) as stream:
            json.dump(result, stream)
            temporary = Path(stream.name)
        try:
            temporary.replace(args.state_dir / "verified.json")
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
