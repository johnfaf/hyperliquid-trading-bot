#!/usr/bin/env python3
"""
Run rotation engine in 7-day shadow mode.

Behavior:
- Forces rotation engine ON
- Forces dry-run telemetry ON (replacement decisions simulated, not executed)
- Requires explicit threshold env vars
- Launches bot command and stops it after N days
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List


REQUIRED_THRESHOLD_ENVS = [
    "PORTFOLIO_REPLACEMENT_THRESHOLD",
    "PORTFOLIO_MAX_REPLACEMENTS_PER_CYCLE",
    "PORTFOLIO_MAX_REPLACEMENTS_PER_HOUR",
    "PORTFOLIO_MAX_REPLACEMENTS_PER_DAY",
    "PORTFOLIO_FORCED_EXIT_COOLDOWN_MINUTES",
    "PORTFOLIO_ROUND_TRIP_BLOCK_MINUTES",
    "PORTFOLIO_MAX_COIN_EXPOSURE_PCT",
    "PORTFOLIO_MAX_SIDE_EXPOSURE_PCT",
    "PORTFOLIO_MAX_CLUSTER_EXPOSURE_PCT",
    "PORTFOLIO_TRANSACTION_COST_WEIGHT",
    "PORTFOLIO_CHURN_PENALTY",
    "PORTFOLIO_EXPECTED_SLIPPAGE_BPS",
]


def _validate_threshold_envs(env: Dict[str, str]) -> List[str]:
    return [k for k in REQUIRED_THRESHOLD_ENVS if not str(env.get(k, "")).strip()]


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 7-day rotation shadow mode")
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Shadow-mode duration in days (default: 7)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use (default: current interpreter)",
    )
    parser.add_argument(
        "--command",
        nargs="+",
        default=["main.py"],
        help="Bot command to run after python executable (default: main.py)",
    )
    args = parser.parse_args()

    days = max(args.days, 1)
    env = os.environ.copy()
    env["ROTATION_ENGINE_ENABLED"] = "true"
    env["ROTATION_DRY_RUN_TELEMETRY"] = "true"
    env["ROTATION_REQUIRE_EXPLICIT_THRESHOLDS"] = "true"
    env["ROTATION_SHADOW_MODE_DAYS"] = str(days)

    missing = _validate_threshold_envs(env)
    if missing:
        print("Missing required explicit threshold env vars:", file=sys.stderr)
        for key in missing:
            print(f"  - {key}", file=sys.stderr)
        return 2

    start = datetime.now(timezone.utc)
    end = start + timedelta(days=days)

    cmd = [args.python] + args.command
    print("Starting rotation shadow mode")
    print(f"  start_utc: {start.isoformat()}")
    print(f"  end_utc:   {end.isoformat()}")
    print(f"  command:   {' '.join(cmd)}")
    print("  mode:      replacements simulated (dry-run telemetry)")

    proc = subprocess.Popen(cmd, env=env)
    try:
        while True:
            if proc.poll() is not None:
                return proc.returncode or 0
            if datetime.now(timezone.utc) >= end:
                print("Shadow window complete. Stopping bot process.")
                _terminate_process(proc)
                return 0
            time.sleep(15)
    except KeyboardInterrupt:
        print("Interrupted by operator. Stopping bot process.")
        _terminate_process(proc)
        return 130
    except Exception:
        _terminate_process(proc)
        raise


if __name__ == "__main__":
    # Ensure Ctrl+C cascades to child on Windows and POSIX.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    raise SystemExit(main())
