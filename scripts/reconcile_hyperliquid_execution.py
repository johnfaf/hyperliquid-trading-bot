"""Offline reconciliation of Hyperliquid exports, venue fills and bot intents."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from contextlib import closing
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import sqlite3

from src.backtest.history_archive import SCHEMA, import_fills as archive_csv, timestamp
from src.data.decision_coverage import open_readonly
from src.trading.live_trader_helpers import make_cloid
from src.trading.paper_reporting import fields

VENUE = "https://api.hyperliquid.xyz/info"
LIMIT = 200_000


def summarize_csv(paths):
    # Occurrences are retained by source; totals are never summed across overlapping exports.
    with closing(sqlite3.connect(":memory:")) as db:
        db.executescript(SCHEMA)
        result = archive_csv(db, paths, timezone_name=None)
        exports = []
        for ident, name in db.execute("SELECT id,filename FROM fill_sources ORDER BY id"):
            rows = db.execute("""SELECT direction,notional,fee,closed_pnl
                FROM fill_occurrences JOIN fill_signatures USING(fingerprint) WHERE source_id=?""", (ident,)).fetchall()
            opens = [row for row in rows if row[0] in ("Open Long", "Open Short")]
            exports.append({"file": name, "rows": len(rows), "directions": dict(Counter(row[0] for row in rows)),
                            "notional": str(sum(Decimal(row[1]) for row in rows)),
                            "reported_fees": str(sum(Decimal(row[2]) for row in rows)),
                            "reported_closed_pnl": str(sum(Decimal(row[3]) for row in rows)),
                            "opening_rows_with_pnl_equal_negative_fee": sum(Decimal(r[3]) == -Decimal(r[2]) for r in opens),
                            "opening_rows": len(opens)})
    return {**result, "exports": exports, "exact_links_possible": False, "funding_included": "unverified"}


def read_rows(path, query, args=()):
    with closing(open_readonly(Path(path))) as db:
        rows = db.execute(query + " LIMIT ?", (*args, LIMIT+1)).fetchall()
    if len(rows) > LIMIT:
        raise ValueError("input too large; provide a bounded export")
    return rows


def _decision_index(path, last, wanted):
    if not path or not wanted:
        return {}
    index = defaultdict(list)
    for row in read_rows(path, """SELECT decision_id,coin,side,created_at,live_order_id FROM decision_snapshots
                               WHERE julianday(created_at)<=julianday(?, 'unixepoch')""", (last/1000,)):
        cid = make_cloid("entry", row["decision_id"])
        if cid in wanted:
            index[cid].append(row)
    return index


def match_execution(fill_path, account, *, order_path=None, decision_path=None):
    if not re.fullmatch("0x[0-9a-fA-F]{40}", account or ""):
        raise ValueError("a verified public account address is required for the raw fill ledger")
    account = account.lower()
    fills = read_rows(fill_path, "SELECT * FROM hyperliquid_fills WHERE account=? ORDER BY time_ms,tid,oid", (account,))
    orders = read_rows(order_path, "SELECT * FROM orders WHERE account=?", (f"hyperliquid:{VENUE}:{account}",)) if order_path else []
    by_oid, chunks = defaultdict(list), defaultdict(list)
    for order in orders:
        if order["venue_id"]:
            by_oid[str(order["venue_id"])].append(order)
    for fill in fills:
        chunks[str(fill["oid"])].append(fill)
    index = _decision_index(decision_path, fills[-1]["time_ms"], {row["client_id"] for row in orders}) if fills else {}
    states, fees, details = Counter(), defaultdict(Decimal), []
    venue_closed_pnl = Decimal(0)
    seen_tids = Counter((row["coin"], row["tid"]) for row in fills)
    for oid, batch in chunks.items():
        payloads = [fields(row["payload"]) for row in batch]
        order_rows = by_oid.get(oid, [])
        if any(seen_tids[(row["coin"], row["tid"])] != 1 for row in batch):
            state, decision_id = "batch_has_conflicting_fill_identity", None
        else:
            state, decision_id = _match_batch(oid, batch, payloads, order_rows, index)
        for row, payload in zip(batch, payloads, strict=True):
            if seen_tids[(row["coin"], row["tid"])] != 1:
                states["conflicting_fill_identity"] += 1
                details.append({"oid": oid, "tid": row["tid"], "decision_id": None, "state": "conflicting_fill_identity"})
                continue
            states[state] += 1
            fees[str(payload.get("feeToken") or "unspecified")] += Decimal(payload["fee"])
            venue_closed_pnl += Decimal(payload["closedPnl"])
            details.append({"oid": oid, "tid": row["tid"], "decision_id": decision_id, "state": state})
    return {"fills": len(fills), "venue_orders": len(chunks), "journal_orders": len(orders),
            "journal_orders_without_fills_in_input": sum(not o["venue_id"] or str(o["venue_id"]) not in chunks for o in orders),
            "first_time_ms": fills[0]["time_ms"] if fills else None, "last_time_ms": fills[-1]["time_ms"] if fills else None,
            "coverage": "unknown", "venue_reported_closed_pnl": str(venue_closed_pnl) if fills else None,
            "fees_by_token": {k: str(v) for k, v in fees.items()}, "states": dict(states), "records": details}


def _match_batch(oid, batch, payloads, order_rows, index):
    if not order_rows:
        return "no_journal_order", None
    if len(order_rows) != 1:
        return "ambiguous_order_id", None
    order = order_rows[0]
    if order["entry"] != 1:
        return "non_entry_order", None
    decisions = index.get(order["client_id"], [])
    if len(decisions) != 1:
        return "order_without_unique_decision", None
    decision = decisions[0]
    side = {"long": "B", "short": "A"}.get(decision["side"])
    intended = fields(order["intent"]).get("orders")
    if not isinstance(intended, list) or len(intended) != 1 or not isinstance(intended[0], dict):
        return "missing_intent_fields", None
    raw = intended[0]
    try:
        at = timestamp(decision["created_at"])
        qty = Decimal(str(raw["s"]))
        filled = sum(Decimal(row["sz"]) for row in payloads)
    except (KeyError, TypeError, ValueError, InvalidOperation):
        return "missing_intent_fields", None
    if (not isinstance(raw.get("b"), bool) or raw.get("r") is not False or raw.get("c") != order["client_id"]
            or not qty.is_finite() or not 0 < filled <= qty or not at or not side
            or raw["b"] != (side == "B") or str(decision["live_order_id"] or oid) != oid
            or any(row["coin"] != decision["coin"] or row["time_ms"] < int(at.timestamp()*1000)
                   for row in batch)
            or any(payload.get("side") != side for payload in payloads)):
        return "order_decision_fill_conflict", None
    return ("matched_full_entry" if filled == qty else "matched_partial_entry"), decision["decision_id"]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", type=Path)
    parser.add_argument("--fills", type=Path, help="Dedicated raw fill ledger from import_hyperliquid_fills")
    parser.add_argument("--account", help="Public address used for the raw import")
    parser.add_argument("--orders", type=Path, help="Local order journal snapshot")
    parser.add_argument("--decisions", type=Path, help="Bot ledger snapshot containing decision_snapshots")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (args.csv or args.fills) or (not args.fills and (args.account or args.orders or args.decisions)):
        parser.error("supply CSV exports or a scoped raw venue ledger")
    try:
        report = {"csv": summarize_csv(args.csv) if args.csv else None,
                  "raw": match_execution(args.fills, args.account, order_path=args.orders,
                                         decision_path=args.decisions) if args.fills else None,
                  "execution_attribution": "exact_ID_only", "statement_reconciled": False}
        with args.out.open("x", encoding="utf-8") as f:
            json.dump(report, f, indent=2, allow_nan=False)
    except (OSError, sqlite3.Error, ValueError, InvalidOperation) as exc:
        parser.exit(1, f"Reconciliation failed: {exc}\n")


if __name__ == "__main__":
    main()
