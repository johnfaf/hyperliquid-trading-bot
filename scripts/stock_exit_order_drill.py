#!/usr/bin/env python3
"""Run a network-free, paper-style stock exit drill across real process restarts."""

import argparse
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import mkdtemp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.run_stock_paper import observe, reconcile, submit_once
from scripts.run_stock_orb_replay import publish
from src.trading.order_journal import Fill, OrderJournal, client_id

PROTOCOL = "stock_exit_order_fault_drill_v1"
SYMBOL = "SPY"
INITIAL = Decimal(".10")
PRICE = "100.00"
REPLACEMENT_CRASH = 75
LEGS = ("entry", "primary", "replacement")
SCENARIOS = ("recovery", "exit_unverified", "cancel_unconfirmed")
SOURCE_FILES = (
    "scripts/stock_exit_order_drill.py",
    "scripts/run_stock_paper.py",
    "scripts/run_stock_orb_replay.py",
    "src/trading/order_journal.py",
)


class DrillFailure(ValueError):
    """The drill failed an invariant; no positive receipt can be issued."""


def check(condition, reason):
    if not condition:
        raise DrillFailure(reason)


def identity(scenario, leg):
    return client_id(PROTOCOL, scenario, leg)


def intent(scenario, leg, quantity):
    return {
        "client_order_id": identity(scenario, leg),
        "symbol": SYMBOL,
        "side": "buy" if leg == "entry" else "sell",
        "qty": str(quantity),
        "limit_price": PRICE,
        "type": "limit",
        "time_in_force": "day",
        "extended_hours": False,
    }


class FaultVenue:
    """Persistent venue fixture. A lost response does not undo the venue write."""

    def __init__(self, path, scenario):
        self.path, self.scenario = path, scenario
        self.data = json.loads(path.read_text(encoding="utf-8"))
        check(self.data["scenario"] == scenario, "fixture scope mismatch")

    @classmethod
    def seed(cls, directory, scenario):
        path = directory / "venue.json"
        entry = intent(scenario, "entry", INITIAL)
        publish(
            path,
            {
                "scenario": scenario,
                "orders": {
                    entry["client_order_id"]: {
                        **entry,
                        "id": "fixture-buy",
                        "status": "filled",
                        "filled_qty": str(INITIAL),
                    }
                },
                "fills": [
                    ["buy-fill", "fixture-buy", SYMBOL, "buy", str(INITIAL), PRICE]
                ],
                "posts": [],
                "hidden_lookup": None,
                "hidden_fill": None,
                "position_override": None,
            },
        )
        broker = cls(path, scenario)
        journal = order_journal(directory, scenario)
        check(
            journal.reserve(entry["client_order_id"], entry, entry=True),
            "entry seed present",
        )
        observe(
            journal, entry["client_order_id"], broker.lookup(entry["client_order_id"])
        )
        return broker

    def save(self):
        publish(self.path, self.data)

    def adjust(self, **values):
        self.data.update(values)
        self.save()

    def lookup(self, cid):
        if cid == self.data["hidden_lookup"]:
            return None
        return self.data["orders"].get(cid)

    def fills(self):
        return [
            Fill(*row[:4], Decimal(row[4]), Decimal(row[5]))
            for row in self.data["fills"]
            if row[0] != self.data["hidden_fill"]
        ]

    def inventory(self):
        return sum(
            (
                Decimal(row[4]) * (1 if row[3] == "buy" else -1)
                for row in self.data["fills"]
            ),
            Decimal(0),
        )

    def request(self, path, *, query=None):
        if path == "/v2/positions":
            quantity = self.data["position_override"]
            quantity = self.inventory() if quantity is None else Decimal(quantity)
            return [{"symbol": SYMBOL, "qty": str(quantity)}] if quantity else []
        if path == "/v2/orders":
            orders = list(self.data["orders"].values())
            if query["status"] == "open":
                return [
                    order
                    for order in orders
                    if order["status"] in {"new", "partially_filled", "pending_cancel"}
                ]
            if query["status"] == "all":
                return orders
        raise DrillFailure("unimplemented fixture endpoint")

    def change_primary(
        self, status, *, fill=None, hide_fill=False, position_override=None
    ):
        cid = identity(self.scenario, "primary")
        order = self.data["orders"][cid]
        if fill:
            fill_id = f"primary-fill-{len(self.data['fills'])}"
            self.data["fills"].append(
                [fill_id, order["id"], SYMBOL, "sell", str(fill), PRICE]
            )
            order["filled_qty"] = str(Decimal(order["filled_qty"]) + fill)
            if hide_fill:
                self.data["hidden_fill"] = fill_id
        order["status"] = status
        self.adjust(position_override=position_override)

    def submit(self, order):
        cid = order["client_order_id"]
        orders = self.data["orders"]
        self.data["posts"].append(cid)
        self.save()
        qty = Decimal(order["qty"])
        check(
            order["side"] == "sell"
            and 0 < qty <= self.inventory()
            and cid not in orders,
            "duplicate or excess POST reached the venue",
        )
        check(
            not self.request("/v2/orders", query={"status": "open"}),
            "overlapping exit reached the venue",
        )
        if cid == identity(self.scenario, "primary"):
            orders[cid] = {
                **order,
                "id": "fixture-primary",
                "status": "new",
                "filled_qty": "0",
            }
        elif cid == identity(self.scenario, "replacement"):
            orders[cid] = {
                **order,
                "id": "fixture-replacement",
                "status": "filled",
                "filled_qty": str(qty),
            }
            self.data["fills"].append(
                [
                    "replacement-fill",
                    "fixture-replacement",
                    SYMBOL,
                    "sell",
                    str(qty),
                    PRICE,
                ]
            )
        else:
            raise DrillFailure("unexpected order role")
        self.save()
        if cid == identity(self.scenario, "replacement"):
            raise SystemExit(REPLACEMENT_CRASH)
        raise TimeoutError(
            "fixture suppressed the order acknowledgement after acceptance"
        )


def order_journal(directory, scenario):
    return OrderJournal(directory / "orders.sqlite3", f"offline:{PROTOCOL}:{scenario}")


def source_fingerprint():
    return sha256(
        b"".join((ROOT / name).read_bytes() for name in SOURCE_FILES)
    ).hexdigest()


def terminal_evidence(broker, journal, rows):
    history = broker.request("/v2/orders", query={"status": "all"})
    by_id = {row["client_order_id"]: row for row in history}
    if (
        len(by_id) != len(history)
        or len(history) != len(rows)
        or any(
            row["client_id"] not in by_id
            or by_id[row["client_id"]]["id"] != row["venue_id"]
            for row in rows
        )
    ):
        return None
    with closing(sqlite3.connect(journal.path)) as db:
        fills = db.execute(
            "SELECT order_id, quantity FROM fills WHERE account=?", (journal.account,)
        ).fetchall()
    filled = Counter()
    for order_id, quantity in fills:
        filled[order_id] += Decimal(quantity)
    for row in rows:
        order_id = row["venue_id"]
        observed = by_id[row["client_id"]]
        requested = Decimal(row["intent"]["qty"])
        venue_qty = Decimal(observed["filled_qty"])
        if (
            observed["status"] != row["status"]
            or observed["status"] not in {"filled", "canceled"}
            or not venue_qty.is_finite()
            or not 0 <= venue_qty <= requested
            or filled[order_id] != venue_qty
            or (observed["status"] == "filled" and venue_qty != requested)
        ):
            return None
    return filled


def block(journal, reason, *, deadline):
    inventory = journal.inventory().get(SYMBOL, Decimal(0))
    state = {"state": "blocked", "reason": reason, "journal_qty": str(inventory)}
    if not deadline:
        return state
    alert = {
        "kind": "exit_exposure_unresolved",
        "reason": reason,
        "symbol": SYMBOL,
        "journal_qty": str(inventory),
        "verified_flat": False,
    }
    with closing(sqlite3.connect(journal.path, timeout=8)) as db, db:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("""CREATE TABLE IF NOT EXISTS drill_alerts (
            account TEXT PRIMARY KEY, payload TEXT NOT NULL
        )""")
        created = (
            db.execute(
                "INSERT OR IGNORE INTO drill_alerts VALUES(?,?)",
                (journal.account, json.dumps(alert)),
            ).rowcount
            == 1
        )
        saved = db.execute(
            "SELECT payload FROM drill_alerts WHERE account=?", (journal.account,)
        ).fetchone()
    return {
        **state,
        "state": "unresolved_exit",
        "new_alert": created,
        "alert": json.loads(saved[0]),
    }


def deadline_latched(journal):
    with closing(sqlite3.connect(journal.path)) as db:
        exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='drill_alerts'"
        ).fetchone()
        return bool(
            exists
            and db.execute(
                "SELECT 1 FROM drill_alerts WHERE account=?", (journal.account,)
            ).fetchone()
        )


def step(directory, scenario, *, deadline=False):
    journal = order_journal(directory, scenario)
    deadline = deadline or deadline_latched(journal)
    broker = FaultVenue(directory / "venue.json", scenario)
    positions, problem = reconcile(broker, journal)
    if problem:
        return block(journal, problem, deadline=deadline)
    rows = journal.orders()
    expected = [identity(scenario, leg) for leg in LEGS[: len(rows)]]
    if (
        not rows
        or [row["client_id"] for row in rows] != expected
        or set(positions) - {SYMBOL}
    ):
        return block(journal, "unexpected_inventory_or_order", deadline=deadline)
    filled = terminal_evidence(broker, journal, rows)
    if filled is None or filled[rows[0]["venue_id"]] != INITIAL:
        return block(journal, "terminal_fill_evidence_pending", deadline=deadline)
    quantity = positions.get(SYMBOL, Decimal(0))
    if not quantity and len(rows) > 1:
        return {"state": "flat_verified", "journal_qty": "0"}
    if deadline or len(rows) >= 3:
        return block(journal, "retry_deadline_or_budget", deadline=deadline)
    if len(rows) == 1 and quantity == INITIAL:
        leg = "primary"
    elif (
        len(rows) == 2
        and rows[1]["status"] == "canceled"
        and quantity == INITIAL - filled[rows[1]["venue_id"]]
        and 0 < quantity <= INITIAL
    ):
        leg = "replacement"
    else:
        return block(journal, "unverified_sell_quantity", deadline=deadline)
    result = submit_once(
        broker, journal, intent(scenario, leg, quantity), purpose=PROTOCOL
    )
    return {"state": result, "leg": leg, "requested_qty": str(quantity)}


def child_step(directory, scenario, *, deadline=False):
    command = [
        sys.executable,
        "-I",
        "-B",
        str(Path(__file__).resolve()),
        "--state-dir",
        str(directory),
        "--step",
        scenario,
    ]
    if deadline:
        command.append("--deadline")
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    if (
        result.returncode == REPLACEMENT_CRASH
        and not result.stdout
        and not result.stderr
    ):
        journal = order_journal(directory, scenario)
        replacement = journal.orders()[-1]
        venue = FaultVenue(directory / "venue.json", scenario)
        check(
            replacement["client_id"] == identity(scenario, "replacement")
            and replacement["status"] == "submitting"
            and venue.lookup(replacement["client_id"])["status"] == "filled",
            "replacement crash did not persist both execution records",
        )
        return {
            "state": "worker_exited_before_ack",
            "requested_qty": replacement["intent"]["qty"],
        }
    result.check_returncode()
    return json.loads(result.stdout)


def exercise(directory, scenario):
    venue = FaultVenue.seed(directory, scenario)
    trace = []

    def advance(label, expected, *, deadline=False):
        row = child_step(directory, scenario, deadline=deadline)
        check(row["state"] == expected, f"{scenario}: unexpected state at {label}")
        trace.append({"phase": label, **row})
        return row

    advance("primary_ack_lost_after_post", "submission_unknown")
    venue = FaultVenue(venue.path, scenario)
    primary = identity(scenario, "primary")
    venue.adjust(hidden_lookup=primary)
    if scenario != "exit_unverified":
        advance("lookup_missing_no_resubmit", "blocked")
        venue.adjust(hidden_lookup=None)
        advance("working_primary_no_resubmit", "blocked")
        venue.change_primary("partially_filled", fill=Decimal(".03"))
        advance("partial_fill_still_working", "blocked")
        venue.change_primary("pending_cancel")
        advance("cancel_requested_not_confirmed", "blocked")
        if scenario == "recovery":
            # Terminal order and two stale data feeds must not allow an oversized replacement.
            venue.change_primary(
                "canceled", fill=Decimal(".02"), hide_fill=True, position_override=".07"
            )
            blocked = advance("canceled_but_late_fill_not_in_ledger", "blocked")
            check(
                blocked["reason"] == "terminal_fill_evidence_pending",
                "late fill was not checked",
            )
            venue.adjust(position_override=None)
            blocked = advance("broker_position_updated_before_fill_receipt", "blocked")
            check(
                blocked["reason"] == "broker_position_mismatch",
                "position discrepancy was not checked",
            )
            venue.adjust(hidden_fill=None)
            retry = advance(
                "replacement_process_exited_after_post", "worker_exited_before_ack"
            )
            check(
                Decimal(retry["requested_qty"]) == Decimal(".05"),
                "replacement exceeded verified residual",
            )
            venue = FaultVenue(venue.path, scenario)
            venue.adjust(hidden_lookup=identity(scenario, "replacement"))
            advance("replacement_lookup_missing_no_resubmit", "blocked")
            venue.adjust(hidden_lookup=None)
            advance("restart_verified_flat", "flat_verified")
            advance("repeat_observation_remains_flat", "flat_verified", deadline=True)
    if scenario != "recovery":
        a = advance("deadline_exposure_unresolved", "unresolved_exit", deadline=True)
        b = advance(
            "restart_does_not_duplicate_alert", "unresolved_exit", deadline=True
        )
        check(
            a["new_alert"] and not b["new_alert"] and a["alert"] == b["alert"],
            "alert not durable or unique",
        )
        if scenario == "exit_unverified":
            venue.adjust(hidden_lookup=None)
            venue.change_primary("canceled")
            late = advance("late_confirmation_cannot_reset_deadline", "unresolved_exit")
            check(
                not late["new_alert"] and late["alert"] == a["alert"],
                "deadline did not remain latched",
            )
    venue = FaultVenue(venue.path, scenario)
    expected_posts = [identity(scenario, "primary")]
    if scenario == "recovery":
        expected_posts.append(identity(scenario, "replacement"))
    check(venue.data["posts"] == expected_posts, "unexpected sell POSTs")
    remaining = venue.inventory()
    expected_inventory = {
        "recovery": Decimal(0),
        "cancel_unconfirmed": Decimal(".07"),
        "exit_unverified": INITIAL,
    }
    check(
        remaining == expected_inventory[scenario],
        "fixture inventory differs from expected outcome",
    )
    return {
        "sell_posts": len(expected_posts),
        "remaining_qty": str(remaining),
        "trace": trace,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dir", type=Path, required=True)
    p.add_argument("--step", choices=SCENARIOS, help=argparse.SUPPRESS)
    p.add_argument("--deadline", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.step:
        print(json.dumps(step(args.state_dir, args.step, deadline=args.deadline)))
        return
    if args.deadline:
        p.error("internal step required")
    run_dir = Path(mkdtemp(prefix="run-", dir=args.state_dir.resolve()))
    scenarios = {}
    for scenario in SCENARIOS:
        directory = run_dir / scenario
        directory.mkdir()
        scenarios[scenario] = exercise(directory, scenario)
    result = {
        "protocol": PROTOCOL,
        "mode": "offline_fixture",
        "passed": True,
        "scenarios": scenarios,
        "run": run_dir.name,
        "broker_order_submissions": 0,
        "paper_approval": False,
        "live_approval": False,
        "source_sha256": source_fingerprint(),
        "observed_at": datetime.now(UTC).isoformat(),
    }
    publish(run_dir / "result.json", result)
    publish(args.state_dir / "latest.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
