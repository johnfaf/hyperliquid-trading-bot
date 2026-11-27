#!/usr/bin/env python3
"""Run one low-frequency US stock paper cycle, or an operator control."""
from argparse import ArgumentParser
from datetime import datetime, UTC, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import json
import os
from pathlib import Path
import re
import sys
from uuid import uuid4
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trading.alpaca_paper import (
    AlpacaPaper, DAILY_LOSS_USD, DEFAULT_SYMBOL, MAX_ENTRY_USD,
    MAX_QUOTE_AGE_SECONDS, MAX_QUOTE_SPREAD_FRACTION, STRATEGY_ID, paper_account_pin,
)
from src.trading.order_journal import OrderJournal, client_id
from src.trading.stock_evidence import StockEvidence
from src.trading.stock_intraday_pilot import PURPOSE as ORB_PURPOSE, Pilot, approved
from src.trading.stock_orb import MAX_CHASE as ORB_CHASE, MAX_SPREAD as ORB_SPREAD, RULE as ORB_RULE
from src.trading.stock_signals import completed_closes, sma20_from_closes

NY = ZoneInfo("America/New_York")
ACTIVE = {"submitting", "unknown", "accepted", "partial"}


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def positive(value):
    value = Decimal(value)
    if not value.is_finite() or value <= 0:
        raise ValueError("bad positive amount from broker")
    return value


def observe(journal, cid, remote):
    if remote["client_order_id"] != cid or not remote.get("id"):
        raise ValueError("broker observation does not match the client order")
    status = {
        "filled": "filled", "partially_filled": "partial", "canceled": "canceled",
        "expired": "canceled", "rejected": "rejected",
        "new": "accepted", "accepted": "accepted", "pending_new": "accepted",
        "pending_cancel": "accepted", "done_for_day": "accepted",
    }.get(remote["status"], "unknown")
    journal.observe(cid, status, remote["id"])
    return status


def submit_once(broker, journal, intent, *, purpose=None, exclusive_entry=False):
    cid = intent["client_order_id"]
    if not journal.reserve(cid, intent, entry=intent["side"] == "buy", purpose=purpose, exclusive_entry=exclusive_entry):
        return "intent_already_recorded_or_paused"
    try:
        status = observe(journal, cid, broker.submit(intent))
    except Exception:
        # Even a transport exception after POST can mean the broker accepted the order.
        journal.observe(cid, "unknown")
        return "submission_unknown"
    return "order_rejected" if status == "rejected" else "intent_sent"


def reconcile(broker, journal):
    for row in journal.orders():
        if row["status"] in ACTIVE:
            remote = broker.lookup(row["client_id"])
            if remote is not None:
                observe(journal, row["client_id"], remote)
    orders = [x for x in journal.orders() if x["venue_id"]]
    known = {x["venue_id"]: x for x in orders}
    if len(known) != len(orders):
        return {}, "unmanaged_open_order"
    fills = broker.fills()
    for fill in fills:
        order = known.get(fill.order_id)
        if not order or (fill.symbol, fill.side) != (order["intent"]["symbol"], order["intent"]["side"]):
            return {}, "unmanaged_or_unconfirmed_fill"
    journal.record_fills(fills)
    rows = broker.request("/v2/positions")
    positions = {row["symbol"]: Decimal(row["qty"]) for row in rows}
    if len(positions) != len(rows) or any(not qty.is_finite() or qty <= 0 for qty in positions.values()) or positions != journal.inventory():
        return {}, "broker_position_mismatch"
    open_orders = broker.request("/v2/orders", query={"status": "open", "limit": 500})
    if not isinstance(open_orders, list) or len(open_orders) >= 500 or any(
        row["id"] not in known or row.get("client_order_id") != known[row["id"]]["client_id"] for row in open_orders
    ):
        return {}, "unmanaged_open_order"
    if journal.unresolved():
        return positions, "submission_unknown"
    if open_orders or any(row["status"] in ACTIVE for row in journal.orders()):
        return positions, "order_pending"
    return positions, None


def tick(broker, journal, *, symbol, orders_enabled=False, flatten=False,
         evidence=None, remember=None, drill=None, expected_account=None, pilot=None):
    if flatten:
        journal.set_pause("operator_flatten")
    positions, problem = reconcile(broker, journal)
    if problem:
        return {"state": problem}
    account = broker.request("/v2/account")
    if expected_account is not None and account.get("id") != expected_account:
        return {"state": "paper_account_unverified"}
    if set(positions) - {symbol}:
        return {"state": "unexpected_symbol"}
    if account.get("trading_blocked") or account.get("status") != "ACTIVE":
        return {"state": "broker_trading_blocked"}
    clock = broker.request("/v2/clock")
    now = timestamp(clock["timestamp"])
    if abs((datetime.now(UTC) - now).total_seconds()) > 30:
        return {"state": "stale_broker_clock"}
    from src.trading.equity_risk import EquityRiskLedger
    risk_ledger = EquityRiskLedger(journal.path)
    risk_ledger.observe(journal.account, now=now, equity=account["equity"], cash=account.get("cash"),
                        day_baseline=account.get("last_equity"))
    equity_risk = risk_ledger.status(journal.account, now=now, daily_loss_usd=float(DAILY_LOSS_USD))
    day = now.astimezone(NY).date()
    today = day.isoformat()
    closed = timestamp(clock["next_close"])
    if not clock["is_open"] or closed - now < timedelta(minutes=5):
        return {"state": "outside_entry_or_exit_window", "session": today}
    equity = positive(account["equity"])
    delta = equity - positive(account["last_equity"])
    qty = positions.get(symbol, Decimal(0))
    orb_qty = journal.owned_inventory(ORB_PURPOSE).get(symbol,Decimal(0)) if qty else Decimal(0)
    if orb_qty and (orb_qty != qty or drill):
        return {"state":"intraday_inventory_unverified","session":today}
    risk_exit = qty and (delta <= -DAILY_LOSS_USD or equity_risk["reason"] in {
        "daily_equity_loss_limit", "peak_equity_drawdown_limit",
    })
    orb_exit = ("operator_flatten" if flatten else "account_risk" if risk_exit else
                pilot.protect(broker,journal,now,closed) if pilot else "intraday_plan_unavailable") if orb_qty else None
    closing = qty and (risk_exit or flatten or drill == "sell" or orb_exit)
    asset = broker.request(f"/v2/assets/{symbol}")
    if asset.get("class") != "us_equity" or not asset.get("tradable") or not asset.get("fractionable"):
        return {"state": "unsupported_asset"}
    signals = None
    daily = None
    if not closing and not orb_qty:
        lookback = (day - timedelta(days=50)).isoformat()
        calendar = broker.request("/v2/calendar", query={"start": lookback, "end": today})
        sessions = sorted(row["date"] for row in calendar)
        if today not in sessions:
            return {"state": "not_a_trading_session"}
        bars = broker.request(f"/v2/stocks/{symbol}/bars", market_data=True, query={
            "timeframe": "1Day", "feed": "iex", "start": lookback, "end": today, "adjustment": "split", "limit": 100,
        })
        closes = None if bars.get("next_page_token") else completed_closes(bars["bars"],sessions,today)
        if closes is not None:
            signals = sma20_from_closes(closes)
            daily = {"last_session":closes[-1][0], "last_close":str(closes[-1][1]),
                     "sma20":str(sum(value for _,value in closes[1:])/20),
                     "prior_close":str(closes[-2][1]),"prior_sma20":str(sum(value for _,value in closes[:-1])/20),
                     "entry":signals[0],"exit":signals[1]}
    if signals is None and not closing and not orb_qty:
        return {"state": "not_enough_completed_bars"}
    entry_paused = bool(journal.pause_reason()) or equity_risk["blocked"]
    drill_qty = evidence.drill_position(symbol) if evidence and qty else Decimal(0)
    paper_guard = qty and drill_qty and not (drill or flatten)
    if qty:
        side = "sell" if closing or (signals and signals[1]) else None
    else:
        side = "buy" if not flatten and not entry_paused and signals and signals[0] and delta > -DAILY_LOSS_USD else None
    if drill:
        side = drill
        if drill == "buy" and (
            not evidence or not evidence.ready_shadow(today, now) or qty or entry_paused
            or delta <= -DAILY_LOSS_USD or timestamp(clock["next_close"]) - now < timedelta(minutes=15)
            or any(row["entry"] and row["intent"].get("_purpose") == "drill" for row in journal.orders())
        ):
            return {"state": "drill_entry_not_ready", "session": today}
        if drill == "sell" and (qty <= 0 or drill_qty != qty):
            return {"state": "drill_position_unverified", "session": today}
    elif paper_guard:
        return {"state": "drill_position_open", "session": today}
    orb_entry = None
    if (orders_enabled and expected_account is not None and pilot and symbol == DEFAULT_SYMBOL and not qty
            and not side and not drill and not flatten and not entry_paused and delta > -DAILY_LOSS_USD):
        orb_entry=pilot.candidate(broker,now,closed)
        if orb_entry:
            side="buy"
    status = "equity_risk_blocked" if equity_risk["blocked"] else "entries_paused" if entry_paused else "daily_limit_reached" if delta <= -DAILY_LOSS_USD else "no_order"
    report = {"state": status, "session": today, "symbol": symbol, "quantity": str(qty),
              "account_equity": str(equity), "account_cash": str(account["cash"]) if account.get("cash") is not None else None,
              "account_daily_change": str(delta), "mode": "paper" if orders_enabled else "shadow",
              "decision": side, "equity_risk": equity_risk}
    if daily is not None:
        report["daily_signal"] = daily
    if orb_qty or orb_entry:
        report.update(route=ORB_PURPOSE, intraday_reason=orb_exit)
    row = broker.request(f"/v2/stocks/{symbol}/quotes/latest", market_data=True, query={"feed": "iex"})["quote"]
    try:
        quote_at = timestamp(row["t"]).astimezone(UTC)
        bid, ask = positive(row["bp"]), positive(row["ap"])
    except (ValueError, ArithmeticError, TypeError, KeyError):
        return {**report, "state": "quote_not_usable", "decision": None}
    quote_age = (datetime.now(UTC) - quote_at).total_seconds()
    report.update(quote_at=quote_at.isoformat(), quote_age=quote_age)
    if not -10 <= quote_age <= MAX_QUOTE_AGE_SECONDS or ask < bid or (ask - bid) / bid > MAX_QUOTE_SPREAD_FRACTION:
        return {**report, "state": "quote_not_usable", "decision": None}
    report.update(quote_valid=True, mark=str((bid + ask) / 2))
    if not side:
        return report
    price = (ask if side == "buy" else bid).quantize(Decimal("0.01"), rounding=ROUND_UP if side == "buy" else ROUND_DOWN)
    if price <= 0:
        return {**report, "state": "invalid_limit_price"}
    if side == "buy":
        if orb_entry and (not orb_entry.at+2 <= quote_at.timestamp() <= orb_entry.at+30
                          or price > orb_entry.price*(1+ORB_CHASE) or ask > bid*(1+ORB_SPREAD)):
            return {**report,"state":"intraday_quote_outside_plan","decision":None}
        cash = min(Decimal(account["cash"]), Decimal(account["buying_power"]), Decimal(10) if drill else MAX_ENTRY_USD)
        if not cash.is_finite() or cash < 1:
            return {**report, "state": "insufficient_cash"}
        qty = (cash / price).quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
        if qty * price < 1:
            return {**report, "state": "order_below_minimum"}
    intent = {"symbol": symbol, "side": side, "qty": str(qty), "limit_price": str(price),
              "type": "limit", "time_in_force": "day", "extended_hours": False,
              "client_order_id": client_id(ORB_RULE if orb_qty or orb_entry else STRATEGY_ID, today, symbol, side,
                                          "drill" if drill else "flatten" if flatten else "routine")}
    if not orders_enabled and not drill:
        return {**report, "state": "shadow_intent", "intent": intent}
    if any(order["client_id"] == intent["client_order_id"] for order in journal.orders()):
        return {**report, "state": "session_intent_already_recorded"}
    if orb_entry:
        pilot.reserve_plan(journal.account,intent["client_order_id"],orb_entry)
    if remember:
        remember({**report, "state": "submitting", "intent": intent})
    purpose = "drill" if drill or (flatten and drill_qty) else ORB_PURPOSE if orb_qty or orb_entry else "strategy"
    return {**report, "state": submit_once(broker, journal, intent, purpose=purpose,exclusive_entry=True), "intent": intent}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--action", choices=["cycle", "pause", "resume", "flatten", "drill-buy", "drill-sell"], default="cycle")
    parser.add_argument("--confirm-paper-drill", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Z]{1,5}", args.symbol):
        parser.error("symbol must be a simple US equity ticker")
    key, secret = os.getenv("STOCK_PAPER_KEY"), os.getenv("STOCK_PAPER_SECRET")
    orders_enabled = os.getenv("STOCK_PAPER_SEND_ORDERS") == "true"
    allowed_pin = os.getenv("STOCK_PAPER_ACCOUNT_SHA256")
    drill = args.action.removeprefix("drill-") if args.action.startswith("drill-") else None
    if drill and (not args.confirm_paper_drill or orders_enabled):
        parser.error("the supervised drill requires --confirm-paper-drill and shadow mode")
    account, cycle_id = "unbound", uuid4().hex
    mode = "paper" if orders_enabled else "shadow"
    purpose = "drill" if drill else "operator" if args.action != "cycle" else "strategy"
    evidence = None

    def remember(result, *, finished=False):
        evidence.record(cycle_id, result, symbol=args.symbol, mode=mode, purpose=purpose, finished=finished)

    try:
        evidence = StockEvidence(args.db, account)
        evidence.start(cycle_id, purpose)
        if not key or not secret:
            OrderJournal(args.db, account)
            result = {"mode": "paper_only", "state": "waiting_for_paper_credentials"}
            remember(result, finished=True)
            print(json.dumps(result))
            return
        broker = AlpacaPaper(key, secret)
        account_id = broker.request("/v2/account")["id"]
        if orders_enabled and paper_account_pin(account_id) != allowed_pin:
            result = {"state": "paper_account_unverified"}
            remember(result, finished=True)
            print(json.dumps({"mode": mode, **result}))
            return
        account = f"alpaca-paper:{account_id}"
        journal = OrderJournal(args.db, account)
        evidence = StockEvidence(args.db, account)
        if args.action in {"pause", "resume"}:
            journal.set_pause("operator_pause" if args.action == "pause" else None)
            result = {"state": args.action}
        else:
            pilot = Pilot(args.db.with_suffix(".orb.sqlite3"), enabled=(
                orders_enabled and os.getenv("STOCK_ORB_PAPER_ENABLE")=="true"
                and approved(Path("/opt/hyperliquid-stock-paper/intraday-proof.json"),datetime.now(UTC))))
            result = tick(
                broker, journal, symbol=args.symbol, orders_enabled=orders_enabled,
                flatten=args.action == "flatten", evidence=evidence, remember=remember, drill=drill,
                expected_account=account_id if orders_enabled else None,
                pilot=pilot,
            )
        remember(result, finished=True)
        print(json.dumps({"mode": mode, **result}))
    except Exception as exc:
        # Exception messages from HTTP clients can contain deployment-sensitive details.
        print(json.dumps({"state": "cycle_failed", "error_type": type(exc).__name__}), file=sys.stderr)
        try:
            if evidence is None:
                OrderJournal(args.db, account)
                evidence = StockEvidence(args.db, account)
            remember({"state": "cycle_failed"}, finished=True)
        except Exception:
            pass
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
