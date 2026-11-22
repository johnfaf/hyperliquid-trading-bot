#!/usr/bin/env python3
"""Publish a small allowlisted host status snapshot to the dashboard bind mount."""
import argparse
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen


UNITS = {
    "stock": "hyperliquid-stock-paper",
    "pilot_report": "hyperliquid-paper-pilot-report",
    "exit_scorecard": "hyperliquid-exit-scorecard",
    "study_backup": "hyperliquid-study-backup",
    "stock_report": "hyperliquid-stock-paper-report",
    "absorption_report": "hyperliquid-absorption-report",
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
    "equity_risk_blocked",
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


def host_command(*args):
    completed = subprocess.run(args, capture_output=True, check=False, text=True, timeout=6)
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
    output = host_command("systemctl", "show", name + ".service", "-p", "Result", "-p", "ExecMainExitTimestamp")
    fields = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    result = fields.get("Result")
    last_run = fields.get("ExecMainExitTimestamp")
    return {"timer_active": timer == "active", "result": result if result in {"success", "exit-code", "timeout", "signal"} else None,
            "last_run": last_run or None}


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
    import math

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
    args = parser.parse_args()
    if bool(args.paper_db) != bool(args.copy_report_dir):
        parser.error("--paper-db and --copy-report-dir must be used together")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    snapshot = build_snapshot(args.study)
    if args.stock_db:
        from src.trading.stock_evidence import read_summary
        try:
            snapshot["stock_pilot"] = read_summary(args.stock_db)
        except (sqlite3.Error, ValueError, OSError):
            snapshot["stock_pilot"] = None
    now = datetime.now(UTC)
    snapshot["absorption"] = absorption_status(args.absorption_report, now)
    snapshot["forward_shadow"] = forward_shadow_status(args.forward_report, now)
    if args.paper_db:
        from src.data.decision_coverage import read_report

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
    publish(args.output, snapshot)
    if args.telegram_env and args.alert_state:
        delivery = maybe_alert(snapshot, args.telegram_env, args.alert_state, now=now)
        if delivery in {"sent", "delivery_unavailable"}:
            print("Stock Pilot alert:", delivery)
        if args.paper_db:
            delivery = notify_copy(report, incidents, args.telegram_env, args.alert_state.with_name("copy-alerts.json"), now=now)
            if delivery in {"sent", "delivery_unavailable"}:
                print("Copy comparison/alert:", delivery)


if __name__ == "__main__":
    main()
