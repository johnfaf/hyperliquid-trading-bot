"""Completed paper-outcome evidence; never an authorization to trade."""
from collections import Counter, defaultdict
from contextlib import closing
from datetime import timedelta
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sqlite3

from src.data.decision_coverage import open_readonly
from src.signals.source_key import is_canonical_source_key
from src.trading.copy_shadow_report import clean_close
from src.trading.paper_reporting import fields, timestamp
from src.trading.trade_decision_memory import number

MIN_CLOSES = 30
MIN_DAYS = 7
STRESS_BPS = 16
MAX_ROWS = 12_000
UNVERIFIED_REASONS = {"unknown", "reconciliation", "live_reconciled_closed", "unverified"}


def rule_version(profile):
    rev = os.getenv("BOT_BUILD_REVISION", "")
    if not re.fullmatch("[0-9a-f]{40}", rev):
        return None
    encoded = json.dumps(profile, sort_keys=True, default=str, separators=(",", ":"))
    return f"{rev}:{sha256(encoded.encode()).hexdigest()[:16]}"


def stamp_paper_entry(meta, strategy_id):
    """Stamp new guarded paper observations only; never backfill missing lineage."""
    entry = fields(meta.get("entry_decision"))
    meta.pop("strategy_evidence", None)
    rules = meta.pop("evidence_rule_inputs", None)
    if os.getenv("BOT_RUN_MODE") != "paper" or entry.get("firewall") != "approved":
        return
    key, route = meta.get("source_key"), entry.get("route")
    key = str(key or "").strip()
    if key.startswith("copy_trade:") and not re.fullmatch("copy_trade:0x[0-9a-f]{40}", key):
        return
    if ":" not in key or not is_canonical_source_key(key) or key.split(":")[-1] in {"unknown", "untagged"} or not route:
        return
    import config
    controls = {k: getattr(config, k) for k in (
        "PAPER_MAX_PLANNED_STOP_LOSS_PCT", "PAPER_MAX_OPEN_STOP_RISK_PCT",
        "PAPER_TRADING_MAKER_FEE_BPS", "PAPER_TRADING_TAKER_FEE_BPS",
        "PAPER_TRADING_SLIPPAGE_MIN_BPS", "PAPER_TRADING_SLIPPAGE_MAX_BPS",
    )}
    instance = str(strategy_id or meta.get("agent_id") or key)
    version = rule_version({"instance": instance, "rules": rules, "controls": controls})
    if version:
        meta["strategy_evidence"] = {"source_key": key, "route": route, "instance": instance, "version": version}


def _outcome(row, meta):
    entry_time, close_time = timestamp(row["opened_at"]), timestamp(row["closed_at"])
    if not entry_time or not close_time or close_time <= entry_time:
        return None, "unverified_time"
    reason = str(meta.get("close_reason") or fields(meta.get("exit_decision")).get("reason") or "").lower()
    if not clean_close(meta) or not reason or reason in UNVERIFIED_REASONS or reason.startswith("live_reconcile"):
        return None, "unverified_close"
    pnl, entry, exit_px, size, lev = (number(row[k]) for k in ("pnl", "entry_price", "exit_price", "size", "leverage"))
    gross, fees, slip = (number(meta.get(k)) for k in
                         ("gross_pnl_before_fees", "total_fees_paid", "total_slippage_cost"))
    # The runtime's best-effort funding aggregate does not prove every hour was
    # observed. Only same-hour outcomes are verifiable without a funding ledger.
    if entry_time.replace(minute=0, second=0, microsecond=0) != close_time.replace(minute=0, second=0, microsecond=0):
        return None, "funding_coverage_unverified"
    funding = number(meta.get("funding_accrued", 0))
    if (None in (pnl, entry, exit_px, size, lev, gross, fees, slip, funding)
            or min(entry, exit_px, size, lev) <= 0 or fees < 0 or slip < 0 or funding != 0
            or row["side"] not in {"long", "short"}):
        return None, "cost_evidence_missing"
    notional = number((entry + exit_px)*size*lev)
    modeled = (exit_px-entry)*size*lev*(1 if row["side"] == "long" else -1)
    if notional is None or abs(modeled-gross) > .011 or abs(pnl-(gross-fees-funding)) > .011:
        return None, "cost_arithmetic_mismatch"
    return (pnl, pnl-notional*.5*STRESS_BPS/10_000, close_time.date()), None


def scorecard(path, *, now):
    start = now - timedelta(days=30)
    report = {"available": False, "reason": "ledger_unavailable", "rows": [], "live_approved": False,
              "min_closes": MIN_CLOSES, "min_days": MIN_DAYS, "stress_bps": STRESS_BPS}
    try:
        with closing(open_readonly(Path(path))) as conn:
            conn.execute("BEGIN")
            rows = conn.execute("""SELECT id,strategy_id,coin,side,status,opened_at,closed_at,pnl,entry_price,exit_price,size,leverage,metadata
                FROM paper_trades WHERE julianday(opened_at)<julianday(?)
                AND (status='open' OR (status='closed' AND julianday(closed_at)>=julianday(?) AND julianday(closed_at)<julianday(?)))
                ORDER BY closed_at,id LIMIT ?""", (now.isoformat(), start.isoformat(), now.isoformat(), MAX_ROWS + 1)).fetchall()
            links = set()
            if len(rows) <= MAX_ROWS and conn.execute("SELECT 1 FROM sqlite_master WHERE name='decision_snapshots'").fetchone():
                for key in range(0, len(rows), 400):
                    ids = [row["id"] for row in rows[key:key+400] if row["status"] == "closed"]
                    if ids:
                        marks = ",".join("?" for _ in ids)
                        links.update(tuple(x) for x in conn.execute(
                            f"SELECT decision_id,paper_trade_id FROM decision_snapshots WHERE paper_trade_id IN ({marks})",
                            ids,
                        ))
    except (OSError, sqlite3.Error):
        return report
    if len(rows) > MAX_ROWS:
        return {**report, "reason": "window_limit"}
    parsed = [(row, fields(row["metadata"])) for row in rows]
    decisions = Counter(meta.get("decision_id") for _, meta in parsed if isinstance(meta.get("decision_id"), str) and meta["decision_id"])
    cohorts = defaultdict(lambda: {"closed": 0, "open": 0, "excluded": Counter(), "outcomes": []})
    for row, meta in parsed:
        identity = fields(meta.get("strategy_evidence"))
        source = str(identity.get("source_key") or meta.get("source_key") or meta.get("strategy_type") or meta.get("source") or "unknown")
        if source.startswith("copy_trade:"):
            source = source[:17] + "..." + source[-6:]
        route = str(identity.get("route") or fields(meta.get("entry_decision")).get("route") or meta.get("source") or "unknown")
        instance = str(identity.get("instance") or row["strategy_id"] or meta.get("agent_id") or "")
        version = identity.get("version")
        version = version if isinstance(version, str) and re.fullmatch("[0-9a-f]{40}:[0-9a-f]{16}", version) else None
        # Keep the unredacted identity in the internal key, not the UI.
        key = (source, route, instance, version, str(identity.get("source_key") or meta.get("source_key") or ""))
        group = cohorts[key]
        if row["status"] == "open":
            group["open"] += 1
            continue
        group["closed"] += 1
        decision = meta.get("decision_id")
        if not version or identity.get("source_key") != meta.get("source_key"):
            group["excluded"]["unversioned"] += 1
        elif not isinstance(decision, str) or decisions.get(decision) != 1:
            group["excluded"]["missing_or_reused_decision"] += 1
        elif (decision, row["id"]) not in links:
            group["excluded"]["missing_decision_link"] += 1
        else:
            outcome, reason = _outcome(row, meta)
            if reason:
                group["excluded"][reason] += 1
            else:
                group["outcomes"].append(outcome)
    for (source, route, instance, version, raw_source), group in cohorts.items():
        outcomes = group.pop("outcomes")
        net = math.fsum(o[0] for o in outcomes) if outcomes else None
        stressed = math.fsum(o[1] for o in outcomes) if outcomes else None
        days = len({o[2] for o in outcomes})
        without_best = stressed - max(0., max(o[1] for o in outcomes)) if outcomes else None
        if not version:
            state = "Rule version missing"
        elif group["excluded"]:
            state = "Incomplete trade evidence"
        elif len(outcomes) < MIN_CLOSES or days < MIN_DAYS:
            state = "Insufficient verified sample"
        elif min(net, stressed, without_best) <= 0:
            state = "Fails cost or concentration check"
        else:
            state = "Ready for human review"
        report["rows"].append({**group, "source": source[:100], "route": route[:55],
                               "instance": instance[:24] if instance != raw_source else "",
                               "version": version, "verified": len(outcomes), "days": days, "net": net,
                               "stressed": stressed, "without_best": without_best, "state": state})
    report["rows"].sort(key=lambda r: (-r["closed"], r["source"], r["version"] or ""))
    return {**report, "available": True, "reason": None, "total_cohorts": len(report["rows"]), "rows": report["rows"][:150]}


def live_entry_hold() -> str:
    """The scorecard is review-only. No operator approval mechanism exists yet."""
    return "strategy_evidence_human_approval_required"
