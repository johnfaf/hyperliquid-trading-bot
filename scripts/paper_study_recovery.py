#!/usr/bin/env python3
"""Encrypted online backups, restore verification, and a narrow SSH status endpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_exit_experiment import DB_NAME, SCHEMA_VERSION, _ms, _readonly, _utc, capture_status
from scripts.paper_exit_scorecard import DAY_MS, render, score_window
from scripts.verify_sqlite_backup import verify_backup
from src.data.paper_exit_feed import SAMPLE_MS


SITE = Path("/srv/hyperliquid-audit-pilot")
PILOT_NAME = "audit-paper-1000.db"
MANIFEST = "manifest.json"
REPORT = "reference-scorecard.md"
MEMBERS = {PILOT_NAME, DB_NAME, MANIFEST, REPORT}
ARCHIVE_RE = re.compile(r"study-[0-9]{8}T[0-9]{6}Z\.tar\.gz\.gpg\Z")
MAX_BACKUP_BYTES = 2_000_000_000


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _json(path: Path, payload: dict):
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def snapshot(source: Path, target: Path):
    if source.resolve() == target.resolve() or target.exists():
        raise ValueError("backup output must be a new file")
    with closing(_readonly(source)) as db, closing(sqlite3.connect(target)) as copy:
        deadline = time.monotonic() + 120

        def check_deadline(_status, _remaining, _total):
            if time.monotonic() > deadline:
                raise TimeoutError("timed out waiting for a consistent database snapshot")

        db.backup(copy, pages=512, sleep=0.1, progress=check_deadline)
        copy.execute("PRAGMA journal_mode=DELETE")
        copy.commit()


def _market_bounds(path: Path, *, integrity: bool = False) -> tuple[int, int]:
    with closing(_readonly(path)) as conn:
        if conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise ValueError("unexpected market recording version")
        if integrity and conn.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("market SQLite integrity check failed")
        first, last = conn.execute("SELECT min(at_ms), max(at_ms) FROM market_samples").fetchone()
        if first is None:
            raise ValueError("market recording has no observations")
        return first, last


def _last_mark(path: Path) -> int:
    with closing(_readonly(path)) as conn:
        stamp, = conn.execute("SELECT max(observed_at) FROM paper_pilot_marks").fetchone()
        if stamp is None:
            raise ValueError("pilot has not recorded any paper marks")
    return _ms(stamp)


def _write_reference(work: Path) -> tuple[int, int]:
    paper, market = work / PILOT_NAME, work / DB_NAME
    first, last = _market_bounds(market)
    cutoff = min(last, _last_mark(paper)) // SAMPLE_MS * SAMPLE_MS
    start = max(first // SAMPLE_MS * SAMPLE_MS, cutoff - 7 * DAY_MS)
    if cutoff <= start:
        raise ValueError("snapshots have no common market/pilot time window")
    text = render(score_window(market, paper, start, cutoff, cutoff), start, cutoff, cutoff)
    (work / REPORT).write_text(text, encoding="utf-8", newline="\n")
    return start, cutoff


def create_backup(site: Path, keyring: Path, recipient: str, revision: str):
    if not re.fullmatch(r"[A-F0-9]{40}", recipient):
        raise ValueError("backup recipient must be a pinned OpenPGP fingerprint")
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("backup verifier revision must be a git commit SHA")
    study = site / "exit-study"
    output = study / "recovery" / "encrypted"
    now = datetime.now(timezone.utc)
    name = now.strftime("study-%Y%m%dT%H%M%SZ.tar.gz.gpg")
    final = output / name
    if final.exists():
        raise FileExistsError(final)
    with TemporaryDirectory(prefix="hl-study-backup-") as scratch:
        work = Path(scratch)
        snapshot(site / "data" / PILOT_NAME, work / PILOT_NAME)
        snapshot(study / "market" / DB_NAME, work / DB_NAME)
        start, cutoff = _write_reference(work)
        _json(work / MANIFEST, {
            "schema": 1, "created_utc": now.isoformat(), "revision": revision,
            "start_ms": start, "cutoff_ms": cutoff,
            "hashes": {filename: sha256(work / filename) for filename in sorted(MEMBERS - {MANIFEST})},
        })
        plain = work / "payload.tar.gz"
        with tarfile.open(plain, mode="w:gz") as tar:
            for member in sorted(MEMBERS):
                tar.add(work / member, arcname=member, recursive=False)
        with NamedTemporaryFile(dir=output, delete=False) as f:
            encrypted = Path(f.name)
        try:
            subprocess.run(
                ["/usr/bin/gpg", "--homedir", str(keyring), "--batch", "--yes", "--no-tty", "--trust-model", "always",
                 "--recipient", recipient, "--output", str(encrypted), "--encrypt", str(plain)],
                check=True, timeout=120,
            )
            size = encrypted.stat().st_size
            if not 0 < size <= MAX_BACKUP_BYTES:
                raise ValueError("encrypted artifact outside expected size limit")
            metadata = {"name": name, "bytes": size, "sha256": sha256(encrypted),
                        "created_utc": now.isoformat(), "cutoff_utc": _utc(cutoff)}
            os.link(encrypted, final)
            _json(output / "latest.json", metadata)
        finally:
            encrypted.unlink(missing_ok=True)
    return metadata


def verify_restore(archive: Path) -> dict:
    with TemporaryDirectory(prefix="hl-study-restore-") as scratch, tarfile.open(archive, mode="r:gz") as tar:
        work = Path(scratch)
        members = tar.getmembers()
        if (len(members) != len(MEMBERS) or {m.name for m in members} != MEMBERS
                or not all(m.isfile() and m.size <= MAX_BACKUP_BYTES for m in members)):
            raise ValueError("invalid research backup members")
        for member in members:
            with tar.extractfile(member) as source, (work / member.name).open("xb") as target:
                shutil.copyfileobj(source, target)
        expected = json.loads((work / MANIFEST).read_text(encoding="utf-8"))
        if expected["schema"] != 1 or set(expected["hashes"]) != MEMBERS - {MANIFEST}:
            raise ValueError("invalid research backup manifest")
        for name, digest in expected["hashes"].items():
            if sha256(work / name) != digest:
                raise ValueError(f"hash mismatch for {name}")
        verify_backup(work / PILOT_NAME, expected["hashes"][PILOT_NAME])
        _market_bounds(work / DB_NAME, integrity=True)
        start, cutoff = _write_reference(work)
        if (start, cutoff) != (expected["start_ms"], expected["cutoff_ms"]):
            raise ValueError("restored cutoff differs from the source cutoff")
        if sha256(work / REPORT) != expected["hashes"][REPORT]:
            raise ValueError("replayed scorecard digest does not match the original")
    return {"verified": True, "revision": expected["revision"], "cutoff_utc": _utc(cutoff),
            "replayed_report_sha256": expected["hashes"][REPORT]}


def _latest(output: Path) -> dict | None:
    path = output / "latest.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    name = result["name"]
    if not ARCHIVE_RE.fullmatch(name) or not re.fullmatch(r"[0-9a-f]{64}", result["sha256"]):
        raise ValueError("invalid backup index")
    artifact = output / name
    if artifact.is_symlink() or not 0 < result["bytes"] <= MAX_BACKUP_BYTES or artifact.stat().st_size != result["bytes"]:
        raise ValueError("encrypted artifact missing or truncated")
    return result


def status(site: Path, *, now: datetime | None = None) -> dict:
    now = now if now is not None else datetime.now(timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    study = site / "exit-study"
    issues = []
    try:
        market = capture_status(study / "market" / DB_NAME, now_ms, max_age_s=90)
        if not market["healthy"]:
            issues.append("collector_stale")
    except (ValueError, OSError, sqlite3.Error):
        issues.append("collector_unreadable")
    try:
        age = now_ms - _last_mark(site / "data" / PILOT_NAME)
        if not -5_000 <= age <= 180_000:
            issues.append("paper_marks_stale")
    except (ValueError, OSError, sqlite3.Error):
        issues.append("paper_ledger_unreadable")

    scorecard_day = (midnight - timedelta(days=1 if now >= midnight + timedelta(minutes=35) else 2)).date()
    if not (study / "reports" / f"scorecard-{scorecard_day}.md").is_file():
        issues.append("scorecard_missing")
    usage = shutil.disk_usage(site)
    if usage.free < max(5 * 1024**3, usage.total // 10):
        issues.append("disk_low")
    try:
        latest = _latest(study / "recovery" / "encrypted")
        on_schedule = now >= midnight + timedelta(minutes=55)
        due = midnight if on_schedule else midnight - timedelta(days=1)
        if latest is None or _ms(latest["created_utc"]) < int(due.timestamp() * 1000):
            issues.append("backup_overdue")
    except (KeyError, TypeError, ValueError, OSError):
        issues.append("backup_unreadable")
        latest = None
    return {"schema": 1, "checked_utc": now.isoformat(), "issues": issues, "backup": latest,
            "disk_free_bytes": usage.free}


def serve(command: str):
    if command == "status":
        print(json.dumps(status(SITE), sort_keys=True))
        return
    name = command.removeprefix("fetch ") if command.startswith("fetch ") else ""
    if not ARCHIVE_RE.fullmatch(name):
        raise PermissionError("only status and an exact encrypted backup name are available")
    output = SITE / "exit-study/recovery/encrypted"
    path = output / name
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_BACKUP_BYTES:
        raise FileNotFoundError("encrypted backup not available")
    with path.open("rb") as f:
        shutil.copyfileobj(f, sys.stdout.buffer)


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    backup = commands.add_parser("backup")
    backup.add_argument("--keyring", type=Path, required=True)
    backup.add_argument("--recipient", required=True)
    backup.add_argument("--revision", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("archive", type=Path, help="Decrypted backup tarball, verified without touching the source files")
    commands.add_parser("serve")
    args = parser.parse_args()
    try:
        if args.command == "backup":
            print(json.dumps(create_backup(SITE, args.keyring, args.recipient, args.revision), sort_keys=True))
        elif args.command == "verify":
            print(json.dumps(verify_restore(args.archive), sort_keys=True))
        else:
            serve(os.environ.get("SSH_ORIGINAL_COMMAND", ""))
    except (OSError, sqlite3.Error, subprocess.SubprocessError, ValueError, KeyError, TypeError, tarfile.TarError) as exc:
        parser.exit(1, f"Study recovery failed: {exc}\n")


if __name__ == "__main__":
    main()
