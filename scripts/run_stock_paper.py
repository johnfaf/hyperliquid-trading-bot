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
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trading.alpaca_paper import (
    AlpacaPaper, DAILY_LOSS_USD, DEFAULT_SYMBOL, MAX_ENTRY_USD,
    MAX_QUOTE_AGE_SECONDS, MAX_QUOTE_SPREAD_FRACTION, STRATEGY_ID,
)
from src.trading.order_journal import OrderJournal, client_id

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


def submit_once(broker, journal, intent):
    cid = intent["client_order_id"]
    if not journal.reserve(cid, intent, entry=intent["side"] == "buy"):
        return "intent_already_recorded_or_paused"
    try:
        observe(journal, cid, broker.submit(intent))
    except Exception:
        # Even a transport exception after POST can mean the broker accepted the order.
        journal.observe(cid, "unknown")
        return "submission_unknown"
    return "intent_sent"


def reconcile(broker, journal):
    for row in journal.orders():
        if row["status"] in ACTIVE:
            remote = broker.lookup(row["client_id"])
            if remote is not None:
                observe(journal, row["client_id"], remote)
    known = {x["venue_id"]: x for x in journal.orders() if x["venue_id"]}
    fills = broker.fills()
    for fill in fills:
        order = known.get(fill.order_id)
        if not order or (fill.symbol, fill.side) != (order["intent"]["symbol"], order["intent"]["side"]):
            return {}, "unmanaged_or_unconfirmed_fill"
    journal.record_fills(fills)
    rows = broker.request("/v2/positions")
    positions = {row["symbol"]: Decimal(row["qty"]) for row in rows}
    if any(not qty.is_finite() or qty <= 0 for qty in positions.values()) or positions != journal.inventory():
        return {}, "broker_position_mismatch"
    open_orders = broker.request("/v2/orders", query={"status": "open", "limit": 500})
    if any(row["id"] not in known for row in open_orders):
        return {}, "unmanaged_open_order"
    if journal.unresolved():
        return positions, "submission_unknown"
    if open_orders or any(row["status"] in ACTIVE for row in journal.orders()):
        return positions, "order_pending"
    return positions, None


def trend(bars, sessions, today):
    previous = [day for day in sessions if day < today][-21:]
    if len(previous) != 21:
        return None
    closes = {}
    for bar in bars:
        day = timestamp(bar["t"]).astimezone(NY).date().isoformat()
        if day < today:
            if day in closes:
                return None
            closes[day] = positive(bar["c"])
    if any(day not in closes for day in previous):
        return None
    prices = [closes[day] for day in previous]
    return prices[-2] * 20 <= sum(prices[:-1]) and prices[-1] * 20 > sum(prices[1:]), prices[-1] * 20 < sum(prices[1:])


def tick(broker, journal, *, symbol, orders_enabled=False, flatten=False):
    if flatten:
        journal.set_pause("operator_flatten")
    positions, problem = reconcile(broker, journal)
    if problem:
        return {"state": problem}
    account = broker.request("/v2/account")
    if set(positions) - {symbol}:
        return {"state": "unexpected_symbol"}
    if account.get("trading_blocked") or account.get("status") != "ACTIVE":
        return {"state": "broker_trading_blocked"}
    clock = broker.request("/v2/clock")
    now = timestamp(clock["timestamp"])
    if abs((datetime.now(UTC) - now).total_seconds()) > 30:
        return {"state": "stale_broker_clock"}
    day = now.astimezone(NY).date()
    today = day.isoformat()
    if not clock["is_open"] or timestamp(clock["next_close"]) - now < timedelta(minutes=5):
        return {"state": "outside_entry_or_exit_window", "session": today}
    lookback = (day - timedelta(days=50)).isoformat()
    calendar = broker.request("/v2/calendar", query={"start": lookback, "end": today})
    sessions = sorted(row["date"] for row in calendar)
    if today not in sessions:
        return {"state": "not_a_trading_session"}
    asset = broker.request(f"/v2/assets/{symbol}")
    if asset.get("class") != "us_equity" or not asset.get("tradable") or not asset.get("fractionable"):
        return {"state": "unsupported_asset"}
    bars = broker.request(f"/v2/stocks/{symbol}/bars", market_data=True, query={
        "timeframe": "1Day", "feed": "iex", "start": lookback, "end": today, "adjustment": "split", "limit": 100,
    })
    signals = None if bars.get("next_page_token") else trend(bars["bars"], sessions, today)
    qty = positions.get(symbol, Decimal(0))
    if signals is None and not (flatten and qty):
        return {"state": "not_enough_completed_bars"}
    equity = positive(account["equity"])
    delta = equity - positive(account["last_equity"])
    entry_paused = bool(journal.pause_reason())
    if qty:
        side = "sell" if flatten or (signals and signals[1]) or delta <= -DAILY_LOSS_USD else None
    else:
        side = "buy" if not flatten and not entry_paused and signals and signals[0] and delta > -DAILY_LOSS_USD else None
    status = "entries_paused" if entry_paused else "daily_limit_reached" if delta <= -DAILY_LOSS_USD else "no_order"
    report = {"state": status, "session": today, "symbol": symbol, "quantity": str(qty),
              "account_equity": str(equity), "account_daily_change": str(delta), "mode": "paper" if orders_enabled else "shadow"}
    if not side:
        return report
    row = broker.request(f"/v2/stocks/{symbol}/quotes/latest", market_data=True, query={"feed": "iex"})["quote"]
    bid, ask = positive(row["bp"]), positive(row["ap"])
    quote_age = (now - timestamp(row["t"])).total_seconds()
    if not -10 <= quote_age <= MAX_QUOTE_AGE_SECONDS or ask < bid or (ask - bid) / bid > MAX_QUOTE_SPREAD_FRACTION:
        return {**report, "state": "quote_not_usable"}
    price = (ask if side == "buy" else bid).quantize(Decimal("0.01"), rounding=ROUND_UP if side == "buy" else ROUND_DOWN)
    if price <= 0:
        return {**report, "state": "invalid_limit_price"}
    if side == "buy":
        cash = min(Decimal(account["cash"]), Decimal(account["buying_power"]), MAX_ENTRY_USD)
        if not cash.is_finite() or cash < 1:
            return {**report, "state": "insufficient_cash"}
        qty = (cash / price).quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
        if qty * price < 1:
            return {**report, "state": "order_below_minimum"}
    intent = {"symbol": symbol, "side": side, "qty": str(qty), "limit_price": str(price),
              "type": "limit", "time_in_force": "day", "extended_hours": False,
              "client_order_id": client_id(STRATEGY_ID, today, symbol, side, "flatten" if flatten else "routine")}
    if not orders_enabled:
        return {**report, "state": "shadow_intent", "intent": intent}
    if any(order["client_id"] == intent["client_order_id"] for order in journal.orders()):
        return {**report, "state": "session_intent_already_recorded"}
    return {**report, "state": submit_once(broker, journal, intent), "intent": intent}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--action", choices=["cycle", "pause", "resume", "flatten"], default="cycle")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Z]{1,5}", args.symbol):
        parser.error("symbol must be a simple US equity ticker")
    key, secret = os.getenv("STOCK_PAPER_KEY"), os.getenv("STOCK_PAPER_SECRET")
    if not key or not secret:
        print(json.dumps({"mode": "paper_only", "state": "waiting_for_paper_credentials"}))
        return
    broker = AlpacaPaper(key, secret)
    try:
        account_id = broker.request("/v2/account")["id"]
        journal = OrderJournal(args.db, f"alpaca-paper:{account_id}")
        if args.action in {"pause", "resume"}:
            journal.set_pause("operator_pause" if args.action == "pause" else None)
            result = {"state": args.action}
        else:
            result = tick(
                broker, journal, symbol=args.symbol, orders_enabled=os.getenv("STOCK_PAPER_SEND_ORDERS") == "true",
                flatten=args.action == "flatten",
            )
        print(json.dumps({"mode": "paper" if os.getenv("STOCK_PAPER_SEND_ORDERS") == "true" else "shadow", **result}))
    except Exception as exc:
        # Exception messages from HTTP clients can contain deployment-sensitive details.
        print(json.dumps({"state": "cycle_failed", "error_type": type(exc).__name__}), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
