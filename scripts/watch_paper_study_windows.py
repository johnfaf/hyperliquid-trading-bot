#!/usr/bin/env python3
"""Check the VPS from Windows, alert locally, and pull only encrypted backups."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile


ARCHIVE_RE = re.compile(r"study-[0-9]{8}T[0-9]{6}Z\.tar\.gz\.gpg\Z")
MAX_BACKUP_BYTES = 2_000_000_000
LABELS = {
    "collector_stale": "No usable BTC quotes in the last 90 seconds.",
    "collector_unreadable": "The market recording is unavailable.",
    "paper_marks_stale": "The paper pilot has not recorded a recent mark.",
    "paper_ledger_unreadable": "The paper ledger is unavailable.",
    "scorecard_missing": "The expected UTC scorecard is missing.",
    "disk_low": "The VPS is running low on disk space.",
    "backup_overdue": "A scheduled encrypted backup is overdue.",
    "backup_unreadable": "The server backup index or archive is unreadable.",
    "download_failed": "Windows could not verify or download the encrypted backup.",
    "vps_check_failed": "Windows could not reach or verify the VPS.",
}
TOAST = r"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$n = New-Object System.Windows.Forms.NotifyIcon
try {
    $n.Icon = [System.Drawing.SystemIcons]::Warning
    $n.Visible = $true
    $n.BalloonTipIcon = [System.Windows.Forms.ToolTipIcon]::Warning
    $n.BalloonTipTitle = 'Hyperliquid paper study'
    $n.BalloonTipText = $env:HL_STUDY_ALERT_MESSAGE
    $n.ShowBalloonTip(7000)
    [System.Media.SystemSounds]::Exclamation.Play()
    Start-Sleep -Seconds 8
} finally {
    $n.Dispose()
}
"""


def _win_options() -> dict:
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def _call_ssh(prefix: list[str], command: str, *, output=None, timeout: int = 30):
    return subprocess.run(
        [*prefix, command], stdin=subprocess.DEVNULL, stdout=output if output is not None else subprocess.PIPE,
        stderr=subprocess.PIPE, check=True, timeout=timeout, **_win_options(),
    )


def _digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def pull_backup(prefix: list[str], metadata: dict, directory: Path) -> bool:
    name, size, sha = metadata["name"], metadata["bytes"], metadata["sha256"]
    if (not ARCHIVE_RE.fullmatch(name) or type(size) is not int or not 0 < size <= MAX_BACKUP_BYTES
            or not re.fullmatch(r"[0-9a-f]{64}", sha)):
        raise ValueError("invalid encrypted backup metadata")
    target = directory / name
    if target.is_file() and target.stat().st_size == size and _digest(target) == sha:
        return False
    with NamedTemporaryFile("wb", dir=directory, delete=False) as f:
        temp = Path(f.name)
        try:
            _call_ssh(prefix, f"fetch {name}", output=f, timeout=180)
        except BaseException:
            f.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        if temp.stat().st_size != size or _digest(temp) != sha:
            raise ValueError("downloaded archive digest or size mismatch")
        temp.replace(target)
        return True
    finally:
        temp.unlink(missing_ok=True)


def transition(previous: dict, issues: list[str], now: float) -> tuple[dict, bool]:
    failed = previous.get("failed_checks", 0) + 1 if issues else 0
    signature = ",".join(sorted(set(issues)))
    alert = bool(signature) and failed >= 2 and (
        signature != previous.get("last_alert") or now - previous.get("last_alert_at", 0) >= 12 * 3600
    )
    return {
        "failed_checks": failed, "current_issues": issues,
        "last_alert": signature if alert else previous.get("last_alert", "") if issues else "",
        "last_alert_at": now if alert else previous.get("last_alert_at", 0),
    }, alert


def _notify(message: str):
    ps = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    env = {**os.environ, "HL_STUDY_ALERT_MESSAGE": message}
    subprocess.run(
        [str(ps), "-NoLogo", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", TOAST],
        check=True, timeout=30, env=env, **_win_options(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--no-notify", action="store_true", help="Log only, useful for a smoke check")
    args = parser.parse_args()
    root = args.state_dir
    stash = root / "encrypted"
    stash.mkdir(exist_ok=True)
    logfile = root / "watch.log"

    def log(message):
        with logfile.open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")

    configured_ssh = str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/OpenSSH/ssh.exe")
    prefix = [
        configured_ssh, "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=yes",
        "-o", "IdentitiesOnly=yes", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
        "-i", str(args.key), "--", args.remote,
    ]
    try:
        report = json.loads(_call_ssh(prefix, "status").stdout)
        issues = report["issues"]
        if report["schema"] != 1 or not isinstance(issues, list) or any(issue not in LABELS for issue in issues):
            raise ValueError("unrecognized status response")
        backup = report.get("backup")
        if backup:
            try:
                if pull_backup(prefix, backup, stash):
                    log(f"verified encrypted download: {backup['name']}")
            except (OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError):
                issues.append("download_failed")
    except (OSError, subprocess.SubprocessError, TypeError, KeyError, ValueError):
        issues = ["vps_check_failed"]
    state_path = root / "watch-state.json"
    previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    state, alert = transition(previous, issues, time.time())
    state["checked_utc"] = datetime.now(timezone.utc).isoformat()
    with NamedTemporaryFile("w", encoding="utf-8", dir=root, delete=False) as f:
        temp = Path(f.name)
        json.dump(state, f, indent=2)
    try:
        temp.replace(state_path)
    finally:
        temp.unlink(missing_ok=True)
    if "checked_utc" not in previous or previous.get("current_issues", []) != issues:
        log("OK" if not issues else "; ".join(issues))
    if alert and not args.no_notify:
        try:
            _notify(" ".join(LABELS[issue] for issue in sorted(set(issues)))[:250])
            log("desktop alert requested")
        except (OSError, subprocess.SubprocessError):
            log("desktop alert failed; check Focus Assist and the signed-in session")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
