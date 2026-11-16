#!/usr/bin/env python3
"""Publish a small allowlisted host status snapshot to the dashboard bind mount."""
import argparse
from datetime import UTC, datetime
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
}

ALERTS = {
    "submission_unknown": "An order status is unresolved. Entries remain blocked until reconciliation.",
    "broker_position_mismatch": "The paper ledger and broker positions do not match.",
    "unmanaged_or_unconfirmed_fill": "A broker paper fill could not be matched to an order.",
    "unmanaged_open_order": "An unexpected paper order is open at Alpaca.",
    "cycle_failed": "The stock paper worker failed.",
    "stale_worker": "The worker has not reported a cycle for more than 15 minutes.",
    "order_rejected": "Alpaca reported a rejected paper order.",
    "quote_not_usable": "Three open-session cycles could not verify a usable quote.",
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


def publish(path: Path, payload: dict, *, mode=0o644) -> None:
    with NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as output:
        temporary = Path(output.name)
        json.dump(payload, output, allow_nan=False)
    try:
        # The parent is the private Docker bind mount, not a public web directory.
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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
    try:
        history = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        history = {}
    if not isinstance(history, dict):
        history = {}
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
    args = parser.parse_args()
    snapshot = build_snapshot(args.study)
    if args.stock_db:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from src.trading.stock_evidence import read_summary
        try:
            snapshot["stock_pilot"] = read_summary(args.stock_db)
        except (sqlite3.Error, ValueError, OSError):
            snapshot["stock_pilot"] = None
    publish(args.output, snapshot)
    if args.telegram_env and args.alert_state:
        delivery = maybe_alert(snapshot, args.telegram_env, args.alert_state, now=datetime.now(UTC))
        if delivery in {"sent", "delivery_unavailable"}:
            print("Stock Pilot alert:", delivery)


if __name__ == "__main__":
    main()
