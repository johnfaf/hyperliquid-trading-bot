"""Prospective audit roster and append-only execution checkpoints."""
from contextlib import closing, contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from src.trading.trade_decision_memory import COPY_CONCENTRATION_STUDY

log = logging.getLogger(__name__)
CURRENT = ContextVar("decision_coverage_cycle", default=None)


def path_for(path=None):
    if path is None:
        import config

        path = config.DB_PATH
    return Path(path).with_suffix(".decisions.sqlite3")


def json_value(values):
    return json.dumps(values, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)


class Cycle:
    def __init__(self, path, number):
        import config

        self.path = path
        self.cycle_id = uuid4().hex
        self.number = number
        self.usable = True
        self.failed = False
        self.sequence = 0
        self.refs = set()
        try:
            with closing(sqlite3.connect(path, timeout=2)) as db, db:
                db.executescript("""
                    CREATE TABLE IF NOT EXISTS evidence (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT, cycle_id TEXT NOT NULL,
                        reference TEXT, observed_at TEXT NOT NULL, event TEXT NOT NULL,
                        route TEXT, decision_id TEXT, payload TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS evidence_recent ON evidence(observed_at);
                    CREATE INDEX IF NOT EXISTS evidence_cycle ON evidence(cycle_id, sequence);
                """)
        except (OSError, sqlite3.Error):
            self.usable = False
            log.exception("Decision evidence unavailable: new entries blocked for this cycle")
        self.add("cycle_started", payload={
            "cycle_number": number, "revision": os.getenv("BOT_BUILD_REVISION", "unknown"),
            "cadence_seconds": config.TRADING_CYCLE_INTERVAL,
            "shadow_protocol": COPY_CONCENTRATION_STUDY,
        })

    def write(self, rows):
        if not self.usable:
            return False
        from src.core import clock_provider

        try:
            stamp = clock_provider.utc_now().isoformat()
            with closing(sqlite3.connect(self.path, timeout=2)) as db, db:
                db.execute("PRAGMA synchronous=FULL")
                db.executemany("INSERT INTO evidence(cycle_id,reference,observed_at,event,route,decision_id,payload) "
                               "VALUES (?,?,?,?,?,?,?)",
                               [(self.cycle_id, ref, stamp, event, route, decision, json_value(body))
                                for event, route, ref, decision, body in rows])
            return True
        except (OSError, sqlite3.Error, ValueError, TypeError):
            self.usable = False
            log.exception("Decision evidence write failed: new entries blocked for this cycle")
            return False

    def add(self, event, route=None, ref=None, decision=None, payload=None):
        return self.write([(event, route, ref, decision, payload or {})])

    def expect(self, route, candidates):
        self.sequence += 1
        refs = [sha256(f"{self.cycle_id}:{route}:{self.sequence}:{i}".encode()).hexdigest()[:32]
                for i in range(len(candidates))]
        roster = [("evaluation_expected", route, ref, None, {
            "symbol": str(candidate.get("coin") or candidate.get("_decision_coin") or candidate.get("ticker") or "")[:40],
            "side": str(candidate.get("side") or candidate.get("direction") or "")[:20],
            "strategy": str(candidate.get("strategy_type") or candidate.get("type") or "")[:70],
            "strategy_id": str(candidate.get("id") or "")[:70],
        }) for ref, candidate in zip(refs, candidates)]
        self.write([("roster", route, None, None, {"expected": len(refs)}), *roster])
        self.refs.update(refs)
        return refs


@contextmanager
def track_cycle(number):
    if os.getenv("DECISION_COVERAGE_ENABLED", "").lower() not in {"1", "true", "yes", "on"}:
        yield
        return
    cycle = Cycle(path_for(), number)
    token = CURRENT.set(cycle)
    try:
        yield
    except BaseException:
        cycle.failed = True
        raise
    finally:
        cycle.add("cycle_failed" if cycle.failed else "cycle_finished")
        CURRENT.reset(token)


def mark_failed():
    cycle = CURRENT.get()
    if cycle:
        cycle.failed = True


def expect(route, candidates):
    cycle = CURRENT.get()
    return cycle.expect(route, candidates) if cycle is not None else [None] * len(candidates)


def owns(ref):
    cycle = CURRENT.get()
    return cycle is not None and ref in cycle.refs


def resolve(ref, state, *, route=None, reason=None, decision=None, quote=None, context=None):
    cycle = CURRENT.get()
    if cycle is not None and ref in cycle.refs:
        cycle.add(state, route, ref, decision, {"reason": str(reason or "")[:250], "quote": quote, **(context or {})})


def entry_allowed():
    cycle = CURRENT.get()
    return cycle is None or cycle.usable


def bind_signal(ref, signal):
    def numeric(value):
        try:
            value = float(value)
            return value if math.isfinite(value) else None
        except (TypeError, ValueError, OverflowError):
            return None

    if not ref:
        return
    from src.data.decision_journal import resolve_decision_id

    context = signal.context or {}
    features = context.get("features")
    features = features if isinstance(features, dict) else {}
    resolve(ref, "decision_linked", decision=resolve_decision_id(signal), quote=numeric(signal.entry_price),
            reason=signal.reason, context={
                "symbol": signal.coin, "side": signal.side.value, "regime": signal.regime,
                "signal_at": signal.timestamp, "confidence": numeric(signal.confidence),
                "quote_age_s": None,
                "features": {k: numeric(features.get(k)) for k in (
                    "rsi", "volatility", "trend_strength", "volume_ratio", "momentum_score", "funding_rate",
                ) if features.get(k) is not None},
            })


def prepare_entry(metadata, *, symbol, side, price):
    cycle = CURRENT.get()
    if cycle is None:
        return
    ref = uuid4().hex
    entry = metadata.get("entry_decision")
    entry = entry if isinstance(entry, dict) else {}
    if not cycle.add("paper_entry_intent", str(metadata.get("source") or "paper"), ref,
                     str(metadata.get("decision_id") or "") or None, {
                         "symbol": str(symbol), "side": str(side), "price": str(price),
                         "evaluation_ref": metadata.get("evaluation_ref"),
                         "signal_at": entry.get("signal_at"), "regime": entry.get("regime"),
                         "reason": str(entry.get("reason") or "")[:250],
                     }):
        raise ValueError("paper entry blocked: durable decision evidence unavailable")
    metadata["coverage_intent_id"] = ref
    metadata["coverage_revision"] = os.getenv("BOT_BUILD_REVISION", "unknown")


def read_report(*, paper_path=None, order_path=None, now=None, window_start=None):
    """Read a bounded 24-hour report. Historic absence is explicitly unmeasured."""
    now = now or datetime.now(timezone.utc)
    if paper_path is None:
        import config

        paper_path = config.DB_PATH
    ledger = Path(paper_path)
    result = {"available": False, "cycles": 0, "finished": 0, "unresolved_cycles": 0,
              "expected": 0, "registered": 0, "terminal": 0, "unresolved_evaluations": 0,
              "intents": 0, "unresolved_intents": 0, "unresolved_orders": None, "routes": [],
              "latest_cycle": None, "capture_started": None, "gap_minutes": None, "late": False,
              "issue": "unavailable", "pending": [], "errors": 0, "revisions": [], "protocols": [],
              "schedule_contiguous": False, "failed_cycles": 0, "overdue_cycles": 0,
              "overdue_evaluations": 0, "overdue_intents": 0}
    try:
        with closing(open_readonly(Path(order_path or ledger.with_suffix(".orders.sqlite3")))) as db:
            result["unresolved_orders"] = db.execute(
                "SELECT count(*) FROM orders WHERE status IN ('submitting','unknown','cancel_pending')"
            ).fetchone()[0]
    except (sqlite3.Error, OSError):
        pass
    try:
        with closing(open_readonly(path_for(ledger))) as db:
            start_at = window_start or now - timedelta(hours=24)
            start = start_at.isoformat()
            limit = 150_000 if window_start is not None else 25_000
            result["capture_started"] = db.execute("SELECT min(observed_at) FROM evidence").fetchone()[0]
            rows = db.execute("SELECT * FROM evidence WHERE observed_at>=? AND observed_at<? ORDER BY sequence LIMIT ?",
                              (start, now.isoformat(), limit + 1)).fetchall()
            if len(rows) > limit:
                result["issue"] = "limit"
                return result
        with closing(open_readonly(ledger)) as db:
            db.execute("BEGIN")
            actual, actual_evaluations, actual_trades = set(), set(), {}
            snapshots = {}
            closed = db.execute("""
                SELECT id,metadata FROM paper_trades WHERE opened_at>=? AND opened_at<?
                """, (start, now.isoformat())).fetchall()
            for row in closed:
                value = _dict(row[1])
                if value.get("coverage_intent_id"):
                    actual.add(value["coverage_intent_id"])
                if value.get("evaluation_ref"):
                    actual_evaluations.add(value["evaluation_ref"])
                if value.get("decision_id"):
                    actual_trades[row[0]] = value["decision_id"]
            has_snapshots = db.execute("SELECT 1 FROM sqlite_master WHERE name='decision_snapshots'").fetchone()
            if has_snapshots:
                columns = {r[1] for r in db.execute("PRAGMA table_info(decision_snapshots)")}
                seen_at = "updated_at" if "updated_at" in columns else "created_at"
                snapshots = {r[0]: (r[1], r[2]) for r in db.execute(
                    "SELECT decision_id,final_status,paper_trade_id FROM decision_snapshots "
                    f"WHERE created_at>=? AND created_at<? AND {seen_at}<?",
                    (start, now.isoformat(), now.isoformat()),
                )}
    except (sqlite3.Error, OSError):
        return result
    groups = {}
    states, scheduled, finished, failed, intents = {}, {}, set(), set(), {}
    terminal = {"no_signal", "blocked", "error", "exit_applied"}
    candidates = {}
    revisions, protocols = set(), set()
    for row in rows:
        event, route, ref = row["event"], row["route"], row["reference"]
        if event == "cycle_started":
            payload = _dict(row["payload"])
            scheduled[row["cycle_id"]] = (row["observed_at"], payload.get("cadence_seconds"))
            revisions.add(payload.get("revision") or "unknown")
            protocols.add(payload.get("shadow_protocol") or "unknown")
        elif event == "cycle_finished":
            finished.add(row["cycle_id"])
        elif event == "cycle_failed":
            failed.add(row["cycle_id"])
        elif event == "roster":
            counts = groups.setdefault(route, {"route": route, "expected": 0, "registered": 0, "terminal": 0})
            counts["expected"] += _dict(row["payload"]).get("expected", 0)
        elif event == "evaluation_expected":
            states[ref] = (route, event, None)
            candidates[ref] = {"observed_at": row["observed_at"], "route": route,
                               "symbol": _dict(row["payload"]).get("symbol"), "reference": ref}
        elif event == "paper_entry_intent":
            intents[ref] = row["observed_at"]
        elif ref in states:
            states[ref] = (states[ref][0], event, row["decision_id"] or states[ref][2])
    for ref, (route, event, decision_id) in states.items():
        counts = groups[route]
        if event != "evaluation_expected" or ref in actual_evaluations:
            counts["registered"] += 1
        status, linked_paper = snapshots.get(decision_id, ("", None))
        status = status or ""
        known = (
            event in terminal or ref in actual_evaluations or
            (decision_id is not None and actual_trades.get(linked_paper) == decision_id) or
            status.endswith("rejected") or status in {"rejected", "rotation_victim_close_failed", "paper_open_failed"}
        )
        if known:
            counts["terminal"] += 1
            result["errors"] += int(event == "error" or status in {"rotation_victim_close_failed", "paper_open_failed"})
        else:
            result["pending"].append({**candidates[ref], "decision_id": decision_id, "state": status or event})
    overdue = (now - timedelta(minutes=20)).isoformat()
    result.update(available=True, issue=None, cycles=len(scheduled), finished=len(scheduled.keys() & finished),
                  unresolved_cycles=len(scheduled.keys() - finished), intents=len(intents),
                  unresolved_intents=len(intents.keys() - actual), routes=list(groups.values()),
                  revisions=sorted(revisions), protocols=sorted(protocols),
                  failed_cycles=len(scheduled.keys() & failed),
                  overdue_cycles=sum(scheduled[k][0] <= overdue for k in scheduled.keys() - finished - failed),
                  overdue_evaluations=sum(row["observed_at"] <= overdue for row in result["pending"]),
                  overdue_intents=sum(intents[k] <= overdue for k in intents.keys() - actual))
    if scheduled:
        latest, cadence = max(scheduled.values(), key=lambda row: row[0])
        result["latest_cycle"] = latest
        result["gap_minutes"] = round((now - datetime.fromisoformat(latest)).total_seconds() / 60, 1)
        result["late"] = bool(cadence and result["gap_minutes"] * 60 > cadence * 2)
        if window_start is not None:
            clock = start_at
            continuous = True
            for stamp, seconds in sorted(scheduled.values(), key=lambda row: row[0]):
                stamp = datetime.fromisoformat(stamp)
                continuous &= bool(seconds and (stamp - clock).total_seconds() <= seconds * 2)
                clock = stamp
            result["schedule_contiguous"] = continuous and not result["late"]
    elif result["capture_started"]:
        result["late"] = True
    for field in ("expected", "registered", "terminal"):
        result[field] = sum(row[field] for row in groups.values())
    result["unresolved_evaluations"] = result["expected"] - result["terminal"]
    result["pending"] = result["pending"][-20:][::-1]
    return result


def open_readonly(path):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
    db.row_factory = sqlite3.Row
    return db


def _dict(value):
    try:
        result = json.loads(value or "{}")
    except (ValueError, TypeError):
        return {}
    return result if isinstance(result, dict) else {}
