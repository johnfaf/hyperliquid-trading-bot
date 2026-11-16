#!/usr/bin/env python3
"""Publish a small allowlisted host status snapshot to the dashboard bind mount."""
import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import subprocess
from tempfile import NamedTemporaryFile


UNITS = {
    "stock": "hyperliquid-stock-paper",
    "pilot_report": "hyperliquid-paper-pilot-report",
    "exit_scorecard": "hyperliquid-exit-scorecard",
    "study_backup": "hyperliquid-study-backup",
}
STOCK_STATES = {
    "waiting_for_paper_credentials", "no_order", "outside_entry_or_exit_window", "shadow_intent",
    "not_enough_completed_bars", "intent_sent", "order_pending", "submission_unknown",
    "broker_position_mismatch", "unmanaged_or_unconfirmed_fill", "unmanaged_open_order", "cycle_failed",
    "stale_broker_clock", "quote_not_usable", "unexpected_symbol", "broker_trading_blocked",
    "unsupported_asset", "not_a_trading_session", "insufficient_cash", "order_below_minimum",
    "intent_already_recorded_or_paused", "session_intent_already_recorded", "invalid_limit_price",
    "entries_paused", "daily_limit_reached",
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


def publish(path: Path, payload: dict) -> None:
    with NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as output:
        temporary = Path(output.name)
        json.dump(payload, output, allow_nan=False)
    try:
        # The parent is the private Docker bind mount, not a public web directory.
        temporary.chmod(0o644)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    publish(args.output, build_snapshot(args.study))


if __name__ == "__main__":
    main()
