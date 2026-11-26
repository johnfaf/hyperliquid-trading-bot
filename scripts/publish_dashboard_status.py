#!/usr/bin/env python3
"""Publish a small allowlisted host status snapshot to the dashboard bind mount."""
import argparse
from datetime import UTC, date, datetime, timedelta
import json
import math
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from time import monotonic
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen


UNITS = {
    "stock": "hyperliquid-stock-paper",
    "pilot_report": "hyperliquid-paper-pilot-report",
    "exit_scorecard": "hyperliquid-exit-scorecard",
    "study_backup": "hyperliquid-study-backup",
    "stock_report": "hyperliquid-stock-paper-report",
    "absorption_report": "hyperliquid-absorption-report",
    "paper_funding": "hyperliquid-paper-funding",
    "forward_shadow": "hyperliquid-forward-shadow",
    "stock_reconciliation": "hyperliquid-stock-reconciliation",
    "stock_forward": "hyperliquid-stock-forward",
    "copy_forward": "hyperliquid-copy-forward",
}
STOCK_STATES = {
    "waiting_for_paper_credentials", "no_order", "outside_entry_or_exit_window", "shadow_intent",
    "not_enough_completed_bars", "intent_sent", "order_pending", "submission_unknown",
    "broker_position_mismatch", "unmanaged_or_unconfirmed_fill", "unmanaged_open_order", "cycle_failed",
    "stale_broker_clock", "quote_not_usable", "unexpected_symbol", "broker_trading_blocked",
    "unsupported_asset", "not_a_trading_session", "insufficient_cash", "order_below_minimum",
    "intent_already_recorded_or_paused", "session_intent_already_recorded", "invalid_limit_price",
    "entries_paused", "daily_limit_reached",
    "order_rejected", "drill_entry_not_ready", "drill_position_unverified", "drill_position_open",
    "equity_risk_blocked", "paper_account_unverified",
}

ALERTS = {
    "submission_unknown": "An order status is unresolved. Entries remain blocked until reconciliation.",
    "broker_position_mismatch": "The paper ledger and broker positions do not match.",
    "unmanaged_or_unconfirmed_fill": "A broker paper fill could not be matched to an order.",
    "unmanaged_open_order": "An unexpected paper order is open at Alpaca.",
    "cycle_failed": "The stock paper worker failed.",
    "stale_worker": "The worker has not reported a cycle for more than 15 minutes.",
    "order_rejected": "Alpaca reported a rejected paper order.",
    "equity_risk_blocked": "The stock paper equity-risk budget is blocking new entries.",
    "paper_account_unverified": "The stock worker did not match the authorized Alpaca Paper account. No orders sent.",
    "strategy_order_aging": "An automatic Alpaca paper order is aging. Check broker status; do not resend it.",
    "quote_not_usable": "Three open-session cycles could not verify a usable quote.",
}

COPY_ALERTS = {
    "coverage_unavailable": "Decision coverage is unavailable.",
    "coverage_late": "No new trading cycle within two scheduled intervals.",
    "cycle_incomplete": "A trading cycle failed or did not finish within 20 minutes.",
    "evaluation_incomplete": "At least one evaluation is unresolved after 20 minutes.",
    "evaluation_error": "The decision journal recorded an execution or evaluation error.",
    "paper_intent": "A paper entry remains unmatched to the ledger after 20 minutes.",
    "venue_order": "The venue order journal contains an unresolved submission.",
    "report_unavailable": "The daily copy comparison could not be generated.",
}

FORWARD_GAPS = {
    "both_quotes": "both scheduled quotes missing",
    "decision_quote": "decision quote missing",
    "fill_quote": "execution quote missing",
    "signal_inputs": "completed-session signal inputs missing",
}


def host_command(*args):
    try:
        completed = subprocess.run(args, capture_output=True, check=False, text=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout if completed.returncode == 0 else ""


def collect_stock(log: str) -> dict:
    for line in reversed(log.splitlines()):
        try:
            event = json.loads(line)
            message = json.loads(event["MESSAGE"])
            if message.get("state") not in STOCK_STATES:
                continue
            observed = datetime.fromtimestamp(int(event["__REALTIME_TIMESTAMP"]) / 1_000_000, UTC).isoformat()
            mode = message.get("mode")
            return {"state": message["state"], "mode": mode if mode in {"paper", "shadow", "paper_only"} else None,
                    "observed_at": observed}
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
            continue
    return {"state": "no_status"}


def unit_info(name: str) -> dict:
    timer = host_command("systemctl", "show", name + ".timer", "-p", "ActiveState", "--value").strip()
    output = host_command("systemctl", "show", name + ".service", "-p", "Result", "-p", "ExecMainExitTimestamp",
                          "-p", "ExecMainExitTimestampMonotonic")
    fields = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    result = fields.get("Result")
    last_run = fields.get("ExecMainExitTimestamp")
    try:
        last_exit = int(fields.get("ExecMainExitTimestampMonotonic", "0")) / 1_000_000
        age = monotonic() - last_exit if last_exit else None
    except ValueError:
        age = None
    return {"timer_active": timer == "active", "result": result if result in {"success", "exit-code", "timeout", "signal"} else None,
            "last_run": last_run or None, "age_seconds": round(age) if age is not None and age >= 0 else None}


def latest_report(directory: Path, prefix: str) -> Path | None:
    return max((p for p in directory.glob(prefix + "*.md")
                if re.fullmatch(rf"{prefix}\d{{4}}-\d{{2}}-\d{{2}}\.md", p.name)), default=None)


def build_snapshot(study: Path) -> dict:
    jobs = {key: unit_info(value) for key, value in UNITS.items()}
    stock = collect_stock(host_command("journalctl", "-u", UNITS["stock"] + ".service", "-n", "60", "--no-pager", "-o",
                                       "json", "--output-fields=MESSAGE,__REALTIME_TIMESTAMP"))
    exit_report = latest_report(study / "exit-study/reports", "scorecard-")
    pilot = latest_report(study / "reports", "paper-")
    coverage = None
    if exit_report:
        content = exit_report.read_text(encoding="utf-8")[:16000]
        match = re.search(r"Since capture began: ([0-9,]+)/([0-9,]+) usable slots \(([\d.]+)%\)", content)
        if match:
            coverage = {"usable": int(match[1].replace(",", "")), "expected": int(match[2].replace(",", "")),
                        "percent": float(match[3])}
    return {"observed_at": datetime.now(UTC).isoformat(), "stock": stock, "jobs": jobs, "coverage": coverage,
            "exit_report": exit_report.name if exit_report else None, "pilot_report": pilot.name if pilot else None}


def publish_text(path: Path, text: str, *, mode=0o644) -> None:
    with NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as output:
        temporary = Path(output.name)
        output.write(text)
    try:
        # The parent is the private Docker bind mount, not a public web directory.
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def publish(path: Path, payload: dict, *, mode=0o644) -> None:
    publish_text(path, json.dumps(payload, allow_nan=False), mode=mode)


def load_object(path: Path, *, max_bytes=65536) -> dict:
    try:
        with path.open(encoding="utf-8") as stream:
            body = stream.read(max_bytes + 1)
        if len(body) > max_bytes:
            return {}
        value = json.loads(body)
        return value if isinstance(value, dict) else {}
    except (ValueError, OSError):
        return {}


def forward_shadow_status(path, now):
    data = load_object(path, max_bytes=8192)
    if data.get("rule") != "btc_long_24h_channel_v1" or data.get("capital") != 1000 or data.get("live_approval") is not False:
        return None
    stamps = {}
    try:
        for key in ("start", "end", "evaluated_at", "published_at"):
            value = datetime.fromisoformat(data[key])
            if value.utcoffset() is None:
                return None
            stamps[key] = value.astimezone(UTC).isoformat()
        age = (now-datetime.fromisoformat(stamps["published_at"])).total_seconds()
        if age < 0:
            return None
    except (KeyError, ValueError, TypeError):
        return None
    state = data.get("state")
    if state not in ("Observing", "Window complete", "Waiting for forward candles", "Incomplete market data"):
        return None
    results = {}
    scenarios = data.get("results")
    scenarios = scenarios if isinstance(scenarios, dict) else {}
    for name in ("base", "stress"):
        row = scenarios.get(name, {})
        if not isinstance(row, dict):
            continue
        values = {k: row.get(k) for k in ("net", "cash", "passive", "drawdown_pct", "net_without_largest", "closed")}
        if all(type(v) in (float, int) and abs(v) <= 1e12 and math.isfinite(v) for v in values.values()):
            results[name] = values
    return {"state": state, **stamps, "results": results if len(results) == 2 else None, "stale": age > 7200}


def stock_forward_status(path, now, monthly_ops):
    from zoneinfo import ZoneInfo

    data = load_object(path, max_bytes=8192)
    if (data.get("rule") != "spy_sma20_two_quote_forward_v1" or data.get("capital") != 1000
            or data.get("sleeve") != 75 or data.get("live_approval") is not False):
        return None
    try:
        published = datetime.fromisoformat(data["published_at"])
        if published.utcoffset() is None or published > now:
            return None
        published = published.astimezone(UTC)
    except (ValueError, KeyError, TypeError):
        return None
    stale = now-published > timedelta(hours=72)
    state = data.get("state")
    market = data.get("market_status")
    if market not in ("Missing shadow-only market configuration", "Frozen protocol or market data unavailable",
                      "Market data not usable in this slot"):
        market = None
    status = {"state": state, "stale": stale, "market_status": market, "published_at": published.isoformat(),
              "verdict": "Insufficient evidence"}
    if state == "Unavailable":
        return status
    if state not in ("Awaiting forward observations", "Observing", "Window complete"):
        return None
    try:
        start, end = (date.fromisoformat(data[k]) for k in ("start", "end"))
        if (end-start).days != 60:
            return None
        next_at = datetime.fromisoformat(data["next_slot"]) if data.get("next_slot") else None
        if next_at is not None and (next_at.utcoffset() is None or next_at < published-timedelta(seconds=75)):
            return None
    except (ValueError, KeyError, TypeError):
        return None
    coverage = {k: data.get(k) for k in ("sessions_planned", "sessions_due", "paired", "samples_due", "samples_captured")}
    if not all(type(v) is int and 0 <= v <= 120 for v in coverage.values()):
        return None
    planned, due, paired, samples_due, captured = coverage.values()
    if not (0 <= paired <= due <= planned <= 60 and paired*2 <= captured <= samples_due <= planned*2):
        return None
    local = ZoneInfo("America/New_York")
    elapsed = max(0, (min(datetime.combine(end, datetime.min.time(), local), published)
                      - datetime.combine(start, datetime.min.time(), local)).total_seconds()/86400)
    expense = None if monthly_ops is None else monthly_ops * elapsed / (365.25/12)
    results = {}
    raw = data.get("results")
    if raw is not None:
        if not isinstance(raw, dict):
            return None
        for name in ("base", "stress"):
            row = raw.get(name)
            if not isinstance(row, dict):
                return None
            values = {k: row.get(k) for k in ("net", "cash", "passive", "drawdown_pct", "net_without_largest", "closed", "entries")}
            if not all(type(v) in (int, float) and abs(v) <= 1e6 and math.isfinite(v) for v in values.values()):
                return None
            if (not all(type(values[k]) is int for k in ("closed", "entries"))
                    or not 0 <= values["closed"] <= values["entries"] <= paired
                    or not 0 <= values["drawdown_pct"] <= 100 or values["cash"] != 0):
                return None
            values["after_ops"] = None if expense is None else values["net"]-expense
            results[name] = values
    if (not stale and state == "Window complete" and planned == due and planned and paired/planned >= .9
            and data.get("final_pair") is True and expense is not None and results and results["stress"]["closed"] >= 10):
        stress = results["stress"]
        status["verdict"] = ("Initial screen passed; research only"
                             if min(stress["net"], stress["net_without_largest"])-expense > max(0, stress["passive"])
                             else "Did not clear the cost and benchmark screen")
    return {**status, **coverage, "start": start.isoformat(), "end": end.isoformat(),
            "next_slot": next_at.astimezone(UTC).isoformat() if next_at else None, "capital": 1000, "sleeve": 75,
            "results": results or None, "monthly_ops": monthly_ops, "operating_cost": expense if results else None}


def stock_forward_integrity(path, now):
    from scripts.stock_forward_evidence import current

    try:
        return current(path, now)
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError, ArithmeticError, OverflowError):
        return None


def forward_restore_status(path, integrity, now):
    if integrity is None:
        return None
    proof = load_object(path, max_bytes=8192)
    if (proof.get("verified") is not True or proof.get("protocol_sha256") != integrity["protocol_sha256"]
            or (proof.get("start"), proof.get("end")) != (integrity["start"], integrity["end"])
            or proof.get("sessions") != integrity["sessions"]):
        return None
    digest, revision, frames, paired = (proof.get(k) for k in ("encrypted_sha256", "revision", "frames", "paired_sessions"))
    if (not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest)
            or not isinstance(revision, str) or not re.fullmatch("[a-f0-9]{40}", revision)
            or type(frames) is not int or type(paired) is not int
            or not 0 <= paired <= min(integrity["paired_sessions"], frames//2)
            or not 0 <= frames <= integrity["frames"]):
        return None
    try:
        checked, verified = (datetime.fromisoformat(proof[k]) for k in ("checkpoint_at", "verified_at"))
        if checked.utcoffset() is None or verified.utcoffset() is None or not checked <= verified <= now:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return {"verified": True, "checkpoint_at": checked.astimezone(UTC).isoformat(),
            "verified_at": verified.astimezone(UTC).isoformat(), "revision": revision, "frames": frames,
            "paired_sessions": paired, "current": frames == integrity["frames"], "stale": now-checked > timedelta(hours=30)}


def forward_backup_lag(integrity, restore, now):
    if not integrity or not integrity["frames"]:
        return None
    covered = restore["frames"] if restore else 0
    if covered == integrity["frames"]:
        return {"missing": 0, "overdue": False, "status": "Encrypted checkpoint current"}
    oldest = integrity["capture_timestamps"][covered]
    overdue = now.timestamp()-oldest >= 12*3600
    return {"missing": integrity["frames"]-covered, "overdue": overdue,
            "status": "Encrypted checkpoint overdue" if overdue else "New observations awaiting the next checkpoint"}


def notify_backup_lag(lag, config, state_path, now):
    if not lag or not lag["overdue"]:
        return "quiet"
    history = load_object(state_path)
    last = history.get("backup_lag_at")
    if type(last) in (int, float) and 0 <= now.timestamp()-last < 86400:
        return "quiet"
    try:
        if send_telegram(config, f"SPY off-VPS restore overdue: {lag['missing']} captured observations lack a verified "
                                "encrypted checkpoint. Check the Windows recovery task and its AC power requirement."):
            publish(state_path, {"backup_lag_at":now.timestamp()}, mode=0o600)
            return "sent"
    except Exception:
        pass  # Do not log transport details or the Telegram URL.
    return "delivery_unavailable"


def copy_forward_status(path, now):
    data = load_object(path, max_bytes=8192)
    if data.get("rule") != "paired_copy_intrahour_v1" or data.get("live_approval") is not False:
        return None
    try:
        start, end, stamp = (datetime.fromisoformat(data[k]) for k in ("start","end","published_at"))
        if any(t.utcoffset() is None for t in (start,end,stamp)) or end-start != timedelta(days=28) or stamp > now:
            return None
        due, captured, candidates, copies = (data[k] for k in ("due_hours","capture_hours","candidates","copy_candidates"))
        if (any(type(x) is not int for x in (due,captured,candidates,copies))
                or not 0 <= captured <= due <= 672 or not 0 <= copies <= candidates <= 300*captured):
            return None
        if data["capital"] != 1000 or data["sleeve"] != 75 or data["state"] not in (
            "Observing","Window complete","Awaiting scheduled captures",
        ):
            return None
    except (KeyError, ValueError, TypeError):
        return None
    results = {}
    raw = data.get("results")
    if raw is not None:
        try:
            for scenario in ("base","stress"):
                sample = raw[scenario]
                counts = {k:sample[k] for k in ("scored_hours","unscored_hours","different_hours")}
                if (any(type(v) is not int or not 0 <= v <= due for v in counts.values())
                        or counts["scored_hours"] > captured or counts["different_hours"] > counts["scored_hours"]
                        or counts["scored_hours"]+counts["unscored_hours"] != due):
                    return None
                comparison = {}
                for name in ("copy_on","copy_off"):
                    row = {k:sample[name][k] for k in ("net","closes","marked_after_exit_drawdown","net_without_largest","copy_closes")}
                    if (not all(type(v) in (int,float) and abs(v) <= 1e6 and math.isfinite(v) for v in row.values())
                            or type(row["closes"]) is not int or type(row["copy_closes"]) is not int
                            or not 0 <= row["copy_closes"] <= row["closes"] <= counts["scored_hours"]):
                        return None
                    comparison[name] = row
                delta = comparison["copy_off"]["net"]-comparison["copy_on"]["net"]
                results[scenario] = {**counts, **comparison, "delta_off_minus_on":delta}
        except (KeyError, TypeError):
            return None
    state = data["state"]
    stale = now-stamp > timedelta(minutes=90)
    enough = (not stale and state == "Window complete" and due == 672 and results
              and all(r["scored_hours"] >= due*.9 and r["different_hours"] >= 20 for r in results.values()))
    return {"state":state, "coverage":captured, "due_hours":due, "candidates":candidates, "copy_candidates":copies,
            "results":results or None, "published_at":stamp.astimezone(UTC).isoformat(), "stale":stale,
            "start":start.astimezone(UTC).isoformat(), "end":end.astimezone(UTC).isoformat(),
            "verdict":"Exploratory comparison complete" if enough else "Insufficient evidence"}


def notify_forward_gaps(integrity, config, state_path):
    if not integrity or not integrity["missed"]:
        return "quiet"
    protocol = integrity["protocol_sha256"]
    history = load_object(state_path, max_bytes=8192)
    seen = history.get("sent", []) if history.get("protocol") == protocol else []
    seen = set(value for value in seen if isinstance(value, str)) if isinstance(seen, list) else set()
    pending = {day: reason for day, reason in integrity["missed"].items() if day not in seen}
    if not pending:
        return "quiet"
    body = "\n".join(f"- {day}: {FORWARD_GAPS[reason]}" for day, reason in pending.items())
    message = "SPY forward study coverage (New York dates):\n" + body + "\nThese sessions will not be backfilled. No orders placed."
    try:
        if send_telegram(config, message):
            publish(state_path, {"protocol": protocol, "sent": list(integrity["missed"])}, mode=0o600)
            return "sent"
    except (OSError, ValueError, TypeError):
        pass  # Avoid printing transport exceptions or the notification URL.
    return "delivery_unavailable"


def offhost_restore_status(path, now):
    data = load_object(path, max_bytes=4096)
    revision = data.get("revision")
    digest = data.get("encrypted_sha256")
    names = data.get("files")
    if (data.get("verified") is not True or not isinstance(revision, str) or not re.fullmatch("[a-f0-9]{40}", revision)
            or not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest) or not isinstance(names, list)
            or not all(isinstance(n, str) for n in names)
            or not {"bot.db", "decisions.db", "risk.db", "funding.db", "stock.db"} <= set(names)):
        return None
    try:
        verified = datetime.fromisoformat(data["verified_at"])
        snapshot = datetime.fromisoformat(data["snapshot_at"])
        if not (snapshot.tzinfo and verified.tzinfo and snapshot <= verified <= now):
            return None
        age = now - snapshot
    except (KeyError, ValueError, TypeError):
        return None
    return {"snapshot_at": snapshot.isoformat(), "verified_at": verified.isoformat(), "revision": revision,
            "stale": age > timedelta(days=7)}


def daily_copy_report(path, directory, now):
    from src.trading.copy_shadow_report import build_daily, markdown

    # Let the completed UTC day's asynchronous ledger writes settle before archiving.
    day = (now - timedelta(minutes=10)).date() - timedelta(days=1)
    basename = directory / f"copy-shadow-{day.isoformat()}"
    archive = basename.with_suffix(".json")
    result = load_object(archive)
    if result.get("date") == day.isoformat() and "baseline" in result:
        return result
    report = build_daily(path, day)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    publish_text(basename.with_suffix(".md"), markdown(report), mode=0o600)
    publish(archive, report, mode=0o600)
    return report


def fixed_weekly_report(path, directory, now):
    from src.trading.paper_reporting import completed_week, weekly_report

    cutoff = now - timedelta(minutes=10)
    start, end = completed_week(cutoff)
    archive = directory / f"paper-week-{start.date()}-to-{end.date()}.json"
    if archive.exists():
        existing = load_object(archive, max_bytes=32 * 1024 * 1024)
        if existing.get("start") != start.isoformat() or existing.get("end_exclusive") != end.isoformat():
            raise ValueError("Existing weekly archive must be inspected, not overwritten")
        return existing
    report = weekly_report(path, cutoff)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    publish(archive, report, mode=0o600)
    return report


def stock_broker_check(path, now):
    from src.trading.paper_reporting import timestamp

    data = load_object(path, max_bytes=4096)
    seen = timestamp(data.get("observed_at"))
    note = data.get("note")
    if (seen is None or not 0 <= (now-seen).total_seconds() <= 30*3600
            or data.get("status") not in ("Pass","Fail","Insufficient evidence")
            or not isinstance(note,str) or len(note) > 160):
        return {"status":"Insufficient evidence", "note":"No fresh read-only broker audit.", "observed_at":None}
    return {"status":data["status"], "note":note, "observed_at":seen.isoformat()}


def investment_review(path, directory, broker_path, weekly, host, now, monthly_ops, drill_path=None):
    from src.learning.copy_toggle_replay import compare
    from src.learning.investment_scorecard import build_week, economics, paper_cash_check, paper_checkpoint, paper_protection_check
    from src.trading.paper_reporting import completed_week, timestamp

    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    recent = now.date()
    checkpoint = directory / f"cash-{recent}.json"
    current = load_object(checkpoint, max_bytes=4096)
    if not checkpoint.exists():
        current = paper_checkpoint(path, now)
        publish(checkpoint, current, mode=0o600)
    elif not current:
        raise ValueError("invalid immutable daily accounting checkpoint")
    prior = load_object(directory / f"cash-{recent-timedelta(days=1)}.json", max_bytes=4096)
    cash = paper_cash_check(prior, current)
    protection = paper_protection_check(path, now)
    broker = stock_broker_check(broker_path, now)
    start, end = completed_week(now-timedelta(minutes=10))
    copy_path = directory / f"copy-toggle-{start.date()}-to-{end.date()}.json"
    copy = load_object(copy_path)
    if not copy and not copy_path.exists():
        try:
            copy = compare(path,start,end)
            publish(copy_path,copy,mode=0o600)
        except (OSError,ValueError,sqlite3.Error):
            copy = None
    if copy and (copy.get("start"),copy.get("end"),copy.get("method")) != (
            start.isoformat(),end.isoformat(),"fixed_recorded_fills_no_reallocation"):
        copy = None
    archive = directory / f"investment-{start.date()}-to-{end.date()}.json"
    if not archive.exists():
        memo = build_week(now, weekly, host, cash, broker, monthly_ops, protection)
        publish(archive, memo, mode=0o600)
    else:
        memo = load_object(archive)
        if (memo.get("version") != 1 or memo.get("week_start") != start.isoformat()
                or memo.get("week_end") != end.isoformat() or memo.get("live_approved") is not False):
            raise ValueError("invalid immutable investment memo")
    receipt = load_object(drill_path,max_bytes=4096) if drill_path is not None else {}
    observed = timestamp(receipt.get("observed_at"))
    passed = (receipt.get("protocol") == "isolated_paper_round_trip_v1" and receipt.get("action") == "verify"
              and receipt.get("passed") is True and receipt.get("flat") is True and receipt.get("restart_recovered") is True
              and receipt.get("unique_orders") == 2 and type(receipt.get("fills")) is int and receipt["fills"] >= 2
              and observed is not None and 0 <= (now-observed).total_seconds() <= 30*86400)
    verdict, figures = economics(host.get("forward_shadow") or {},monthly_ops)
    return {**memo, "current_cash_check": cash, "current_protection_check": protection,
            "current_stock_check": {"status":broker["status"], "note":broker["note"]},
            "current_drill": "Paper round trip and restart verified" if passed else "No verified isolated broker drill",
            "current_btc": figures, "current_btc_verdict": verdict["status"], "copy_toggle": copy}


def absorption_status(path, now):
    from src.trading.paper_reporting import timestamp

    report = load_object(path)
    observed = timestamp(report.get("observed_at"))
    if report.get("version") != 1 or observed is None:
        return None
    # Research data only; never relay arbitrary provider context or file contents.
    keys = ("observed_at", "start", "rule", "state", "healthy_at_scoring", "frames", "ready", "expected",
            "first_frame", "last_frame", "side_supported", "coverage", "horizons", "primary_intervals")
    return {**{key: report.get(key) for key in keys}, "live_approved": False,
            "stale": not 0 <= (now - observed).total_seconds() <= 26 * 3600}


def copy_incidents(coverage, *, report_available=True):
    reasons = []
    if not coverage["available"] or ("capture_started" in coverage and coverage["capture_started"] is None):
        reasons.append("coverage_unavailable")
    else:
        if coverage["late"]:
            reasons.append("coverage_late")
        if coverage["failed_cycles"] or coverage["overdue_cycles"]:
            reasons.append("cycle_incomplete")
        if coverage["overdue_evaluations"]:
            reasons.append("evaluation_incomplete")
        if coverage["errors"]:
            reasons.append("evaluation_error")
        if coverage["overdue_intents"]:
            reasons.append("paper_intent")
    if coverage["unresolved_orders"]:
        reasons.append("venue_order")
    if not report_available:
        reasons.append("report_unavailable")
    return reasons


def notify_copy(report, incidents, config, state_path, *, now):
    from src.trading.copy_shadow_report import digest

    history = load_object(state_path)
    alerts = {key: history[key] for key in COPY_ALERTS if isinstance(history.get(key), (float, int))}
    due = [key for key in incidents if key not in alerts or not 0 <= now.timestamp() - alerts[key] < 6 * 3600]
    report_due = report is not None and history.get("date") != report["date"]
    messages = [digest(report)] if report_due else []
    if due:
        messages.append("Crypto decision alerts:\n" + "\n".join("- " + COPY_ALERTS[key] for key in due))
    if not messages:
        return "quiet"
    try:
        if send_telegram(config, "\n\n".join(messages)):
            alerts.update({key: now.timestamp() for key in due})
            publish(state_path, {**alerts, "date": report["date"] if report_due else history.get("date")}, mode=0o600)
            return "sent"
    except Exception:
        # Transport errors can contain the secret-bearing URL.
        pass
    return "delivery_unavailable"


def stock_alert(snapshot: dict, now: datetime) -> str | None:
    stock = snapshot["stock"]
    unit = snapshot["jobs"].get("stock", {})
    state = stock.get("state")
    if unit.get("result") in {"exit-code", "timeout", "signal"}:
        return "cycle_failed"
    if state in {"waiting_for_paper_credentials", "no_status", None} or not unit.get("timer_active"):
        return None
    try:
        age = (now - datetime.fromisoformat(stock["observed_at"])).total_seconds()
    except (KeyError, ValueError, TypeError):
        return None
    if age > 900:
        return "stale_worker"
    if ((snapshot.get("stock_pilot") or {}).get("execution") or {}).get("stuck"):
        return "strategy_order_aging"
    recent = (snapshot.get("stock_pilot") or {}).get("decisions", [])
    if state == "quote_not_usable":
        return state if len(recent) >= 3 and all(row["state"] == state for row in recent[:3]) else None
    if state in {"broker_position_mismatch", "unmanaged_or_unconfirmed_fill", "unmanaged_open_order"}:
        # A fill arriving between two broker reads can cause one transient mismatch.
        return state if len(recent) >= 2 and all(row["state"] == state for row in recent[:2]) else None
    return state if state in ALERTS else None


def send_telegram(config: Path, message: str) -> bool:
    env = {}
    for line in config.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {"TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"}:
            env[key] = value.strip().strip("'\"")
    token, chat_id = env.get("TELEGRAM_BOT_TOKEN", ""), env.get("TELEGRAM_CHAT_ID", "")
    if not re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+", token) or not re.fullmatch(r"-?[0-9]+|@[A-Za-z0-9_]+", chat_id):
        return False
    request = Request(f"https://api.telegram.org/bot{token}/sendMessage", method="POST",
                      headers={"Content-Type": "application/json"},
                      data=json.dumps({"chat_id": chat_id, "text": message}).encode())
    with urlopen(request, timeout=8) as response:
        return json.load(response).get("ok") is True


def maybe_alert(snapshot: dict, config: Path, state_path: Path, *, now: datetime) -> str:
    reason = stock_alert(snapshot, now)
    if reason is None:
        return "quiet"
    history = load_object(state_path)
    previous = history.get(reason, 0)
    if isinstance(previous, (int, float)) and 0 <= now.timestamp() - previous < 3600:
        return "suppressed"
    try:
        if send_telegram(config, "Stock Pilot (paper): " + ALERTS[reason]):
            history[reason] = now.timestamp()
            publish(state_path, history, mode=0o600)
            return "sent"
    except Exception:
        # urllib exceptions can include the secret-bearing URL; do not log it.
        pass
    return "delivery_unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stock-db", type=Path)
    parser.add_argument("--telegram-env", type=Path)
    parser.add_argument("--alert-state", type=Path)
    parser.add_argument("--paper-db", type=Path)
    parser.add_argument("--copy-report-dir", type=Path)
    parser.add_argument("--absorption-report", type=Path, default=Path("/var/lib/hyperliquid-absorption/reports/latest.json"))
    parser.add_argument("--forward-report", type=Path, default=Path("/var/lib/hyperliquid-forward-shadow/latest.json"))
    parser.add_argument("--stock-forward-report", type=Path, default=Path("/var/lib/hyperliquid-stock-forward/latest.json"))
    parser.add_argument("--stock-forward-ledger", type=Path, default=Path("/var/lib/hyperliquid-stock-forward/market.sqlite3"))
    parser.add_argument("--stock-forward-restore", type=Path, default=Path("/var/lib/hyperliquid-dashboard/stock-forward-restore.json"))
    parser.add_argument("--copy-forward-report", type=Path, default=Path("/var/lib/hyperliquid-copy-forward/latest.json"))
    parser.add_argument("--offhost-restore", type=Path, default=Path("/var/lib/hyperliquid-dashboard/offhost-restore.json"))
    parser.add_argument("--stock-reconciliation", type=Path, default=Path("/var/lib/hyperliquid-stock-paper/broker-reconciliation.json"))
    parser.add_argument("--isolated-stock-drill", type=Path, default=Path("/var/lib/hyperliquid-stock-drill/verified.json"))
    parser.add_argument("--investment-dir", type=Path)
    parser.add_argument("--monthly-ops-usd", type=float)
    args = parser.parse_args()
    if bool(args.paper_db) != bool(args.copy_report_dir):
        parser.error("--paper-db and --copy-report-dir must be used together")
    if args.investment_dir and not args.paper_db:
        parser.error("--investment-dir requires --paper-db")
    if args.monthly_ops_usd is not None and not 0 <= args.monthly_ops_usd <= 100_000:
        parser.error("--monthly-ops-usd must be a finite nonnegative amount")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    snapshot = build_snapshot(args.study)
    if args.stock_db:
        from src.trading.stock_evidence import read_summary
        try:
            snapshot["stock_pilot"] = read_summary(args.stock_db)
        except (sqlite3.Error, ValueError, OSError):
            snapshot["stock_pilot"] = None
    now = datetime.now(UTC)
    if snapshot.get("stock_pilot"):
        snapshot["stock_pilot"]["broker_check"] = stock_broker_check(args.stock_reconciliation, now)
    snapshot["absorption"] = absorption_status(args.absorption_report, now)
    snapshot["forward_shadow"] = forward_shadow_status(args.forward_report, now)
    snapshot["stock_forward"] = stock_forward_status(args.stock_forward_report, now, args.monthly_ops_usd)
    integrity = stock_forward_integrity(args.stock_forward_ledger, now)
    snapshot["stock_forward_restore"] = forward_restore_status(args.stock_forward_restore, integrity, now)
    lag = forward_backup_lag(integrity, snapshot["stock_forward_restore"], now)
    snapshot["stock_forward_backup"] = lag
    snapshot["copy_forward"] = copy_forward_status(args.copy_forward_report, now)
    stock_forward = snapshot["stock_forward"]
    if stock_forward and stock_forward["state"] != "Unavailable":
        if integrity is None:
            stock_forward["market_status"] = "Independent coverage check unavailable"
        else:
            stock_forward["missing_sessions"] = len(integrity["missed"])
            if integrity["missed"]:
                last = max(integrity["missed"])
                stock_forward["market_status"] = f"Latest gap {last} NY: {FORWARD_GAPS[integrity['missed'][last]]}"
            if lag and lag["overdue"]:
                message = f"Encrypted checkpoint overdue: {lag['missing']} observations not verified off-VPS"
                stock_forward["market_status"] = " | ".join(x for x in (stock_forward["market_status"], message) if x)
    snapshot["offhost_restore"] = offhost_restore_status(args.offhost_restore, now)
    if args.paper_db:
        from src.data.decision_coverage import read_report
        from src.trading.paper_funding_health import funding_health

        snapshot["paper_funding"] = funding_health(args.paper_db, now)
        try:
            report = daily_copy_report(args.paper_db, args.copy_report_dir, now)
        except (sqlite3.Error, ValueError, OSError):
            report = None
        incidents = copy_incidents(read_report(paper_path=args.paper_db, now=now), report_available=report is not None)
        snapshot.update(copy_daily=report, copy_alerts=[COPY_ALERTS[key] for key in incidents])
        try:
            weekly = fixed_weekly_report(args.paper_db, args.copy_report_dir / "weekly", now)
            snapshot["weekly"] = {key: value for key, value in weekly.items() if key != "trades"}
        except (sqlite3.Error, ValueError, OSError):
            snapshot["weekly"] = None
        if args.investment_dir:
            try:
                snapshot["investment"] = investment_review(
                    args.paper_db, args.investment_dir, args.stock_reconciliation, snapshot["weekly"],
                    snapshot, now, args.monthly_ops_usd, args.isolated_stock_drill,
                )
            except (sqlite3.Error, ValueError, OSError):
                snapshot["investment"] = None
    publish(args.output, snapshot)
    if args.telegram_env and args.alert_state:
        delivery = notify_backup_lag(lag, args.telegram_env, args.alert_state.with_name("stock-forward-backup-alerts.json"), now)
        if delivery in {"sent", "delivery_unavailable"}:
            print("Stock forward backup:", delivery)
        delivery = notify_forward_gaps(integrity, args.telegram_env, args.alert_state.with_name("stock-forward-alerts.json"))
        if delivery in {"sent", "delivery_unavailable"}:
            print("Stock forward coverage:", delivery)
        delivery = maybe_alert(snapshot, args.telegram_env, args.alert_state, now=now)
        if delivery in {"sent", "delivery_unavailable"}:
            print("Stock Pilot alert:", delivery)
        if args.paper_db:
            delivery = notify_copy(report, incidents, args.telegram_env, args.alert_state.with_name("copy-alerts.json"), now=now)
            if delivery in {"sent", "delivery_unavailable"}:
                print("Copy comparison/alert:", delivery)


if __name__ == "__main__":
    main()
