"""Independent coverage and encrypted recovery checks for the pinned SPY study."""
import argparse
from contextlib import closing
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile
from zoneinfo import ZoneInfo

STATE = Path("/var/lib/hyperliquid-stock-forward/market.sqlite3")
PIN = Path("/opt/hyperliquid-stock-forward/current")
NY = ZoneInfo("America/New_York")
REVISION = re.compile(r"[a-f0-9]{40}")
DIGEST = re.compile(r"[a-f0-9]{64}")
SOURCES = ("scripts/run_stock_forward_shadow.py", "src/backtest/stock_shadow.py", "src/trading/stock_signals.py")
MANIFEST, LEDGER = "manifest.json", "market.sqlite3"
LIMIT = 4 * 1024**2
MEMBERS = {MANIFEST: 4096, LEDGER: LIMIT, **dict.fromkeys(SOURCES, 65536)}


def read_db(path):
    return sqlite3.connect(path.resolve(strict=True).as_uri() + "?mode=ro", uri=True, timeout=2)


def bounded_rows(db, query, limit):
    rows = db.execute(query).fetchmany(limit+1)
    if len(rows) > limit:
        raise ValueError("study row limit exceeded")
    return rows


def study(db, now):
    """Review only the frozen calendar, recorded inputs and captured quotes."""
    protocols = bounded_rows(db, "SELECT frozen_at,start_day,end_day,digest FROM protocol", 1)
    if len(protocols) != 1:
        raise ValueError("study is not registered")
    frozen_raw, start_raw, end_raw, digest = protocols[0]
    frozen = datetime.fromisoformat(frozen_raw)
    start, end = date.fromisoformat(start_raw), date.fromisoformat(end_raw)
    if (frozen.utcoffset() is None or frozen > now or not DIGEST.fullmatch(digest)
            or frozen.astimezone(NY).date()+timedelta(days=1) != start or end-start != timedelta(days=60)):
        raise ValueError("invalid frozen study")
    sessions = {}
    for raw, decision, fill, close in bounded_rows(
        db, "SELECT day,decision_at,fill_at,closes_at FROM sessions ORDER BY day", 112,
    ):
        day = date.fromisoformat(raw)
        expected = int(datetime.combine(day, time(9, 45), NY).timestamp())
        if (raw != day.isoformat() or raw in sessions or day.weekday() > 4 or not start-timedelta(days=50) <= day < end
                or any(type(x) is not int for x in (decision, fill, close))
                or decision != expected or fill != expected+300 or not fill+75 < close <= expected+22500):
            raise ValueError("calendar does not match the frozen slots")
        sessions[raw] = decision, fill
    forward = [day for day in sessions if start_raw <= day < end_raw]
    if not forward or len(sessions)-len(forward) < 21:
        raise ValueError("invalid frozen calendar")
    frames = {}
    for day, slot, quote, observed, bid, ask, enter, leave, inputs in bounded_rows(
        db, "SELECT day,slot,quote_at,observed_at,bid,ask,entry,exit,inputs FROM frames", 120,
    ):
        if day not in forward or type(slot) is not int or slot not in (0, 1) or (day, slot) in frames:
            raise ValueError("unexpected capture")
        at = sessions[day][slot]
        bid, ask = Decimal(bid), Decimal(ask)
        if (any(type(x) is not int for x in (quote, observed))
                or not at <= quote <= observed <= min(quote+30, at+75, now.timestamp())
                or not (bid.is_finite() and ask.is_finite() and Decimal(".01") <= bid <= ask <= 100_000)
                or (ask-bid)/bid > Decimal(".005")):
            raise ValueError("invalid captured quote")
        if slot == 1 or inputs is None:
            if any(x is not None for x in (enter, leave, inputs)):
                raise ValueError("unexpected signal inputs")
        else:
            closes = json.loads(inputs)
            if (not isinstance(closes, list) or len(closes) != 21
                    or any(not isinstance(row, list) or len(row) != 2 for row in closes)
                    or [row[0] for row in closes] != [value for value in sessions if value < day][-21:]):
                raise ValueError("signal inputs are not the prior 21 completed sessions")
            prices = [Decimal(row[1]) for row in closes]
            if not all(p.is_finite() and p > 0 for p in prices):
                raise ValueError("invalid completed closes")
            crossover = prices[-2]*20 <= sum(prices[:-1]) and prices[-1]*20 > sum(prices[1:])
            below = prices[-1]*20 < sum(prices[1:])
            if type(enter) is not int or type(leave) is not int or (enter, leave) != (crossover, below):
                raise ValueError("stored signal does not match its inputs")
        frames[day, slot] = quote, observed, enter
    missed = {}
    paired = 0
    for day in forward:
        first, second = frames.get((day, 0)), frames.get((day, 1))
        if first and second and first[2] is not None and first[1] < second[0]:
            paired += 1
        elif now.timestamp() > sessions[day][1]+135:
            reason = "both_quotes" if not first and not second else "decision_quote" if not first else (
                "fill_quote" if not second else "signal_inputs")
            missed[day] = reason
    return {"start": start_raw, "end": end_raw, "protocol_sha256": digest, "sessions": len(forward),
            "frames": len(frames), "paired_sessions": paired, "missed": missed}


def current(path, now):
    with closing(read_db(path)) as db:
        db.execute("BEGIN")
        return study(db, now)


def pack(path, pin, output, *, clock=lambda: datetime.now(UTC)):
    pin = pin.resolve(strict=True)
    if not REVISION.fullmatch(pin.name):
        raise ValueError("invalid pinned source revision")
    files = {name: (pin/name).read_bytes() for name in SOURCES}
    with TemporaryDirectory(prefix="stock-forward-pack-") as tmp:
        target = Path(tmp)/LEDGER
        with closing(read_db(path)) as source, closing(sqlite3.connect(target)) as copy:
            source.backup(copy)
        files[LEDGER] = target.read_bytes()
    manifest = {"version": 1, "revision": pin.name, "checkpoint_at": clock().isoformat(),
                "files": {k: sha256(v).hexdigest() for k, v in files.items()}}
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in {**files, MANIFEST: json.dumps(manifest).encode("ascii")}.items():
            if not 0 < len(content) <= MEMBERS[name]:
                raise ValueError("invalid checkpoint member size")
            archive.writestr(name, content)


def verify(stream, scratch, now):
    with ZipFile(stream) as archive:
        members = archive.infolist()
        if (len(members) != len(MEMBERS) or {m.filename for m in members} != MEMBERS.keys()
                or any(not 0 < m.file_size <= MEMBERS[m.filename] for m in members)):
            raise ValueError("unexpected checkpoint contents")
        payload = {m.filename: archive.read(m) for m in members}
    meta = json.loads(payload.pop(MANIFEST))
    checkpoint = datetime.fromisoformat(meta["checkpoint_at"])
    if (meta["version"] != 1 or not REVISION.fullmatch(meta["revision"])
            or checkpoint.utcoffset() is None or checkpoint > now
            or meta["files"] != {k: sha256(v).hexdigest() for k, v in payload.items()}):
        raise ValueError("checkpoint manifest mismatch")
    path = scratch/LEDGER
    with path.open("xb") as target:
        target.write(payload[LEDGER])
    with closing(read_db(path)) as db:
        if db.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("checkpoint ledger failed SQLite verification")
        actual = study(db, checkpoint)
    if actual["protocol_sha256"] != sha256(b"".join(payload[name] for name in SOURCES)).hexdigest():
        raise ValueError("checkpoint source does not match the frozen protocol")
    return {"verified": True, "revision": meta["revision"], "checkpoint_at": checkpoint.isoformat(), **actual}


def fetch(host, key, directory):
    recipient = (directory/"recipient.txt").read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[a-z][a-z0-9_-]*@[a-zA-Z0-9][a-zA-Z0-9.-]*", host) or not re.fullmatch(r"age1[a-z0-9]{58}", recipient):
        raise ValueError("invalid configured recovery host or age recipient")
    ssh = ["ssh", "-oBatchMode=yes", "-oStrictHostKeyChecking=yes", "-oIdentitiesOnly=yes",
           "-oConnectTimeout=8", "-i", str(key), host]
    command = ("bash -c 'set -o pipefail; sudo -n python3 -I -B /opt/hyperliquid-recovery/stock_forward_evidence.py pack"
               f" | /opt/hyperliquid-recovery/age/age -r {recipient}'")
    target = None
    try:
        with NamedTemporaryFile(dir=directory, prefix="stock-forward-", suffix=".partial.age", delete=False) as output:
            target = Path(output.name)
            subprocess.run(ssh+[command], stdout=output, stdin=subprocess.DEVNULL, check=True, timeout=45)
            output.flush()
            os.fsync(output.fileno())
        if not 128 < target.stat().st_size <= LIMIT+262144:
            raise ValueError("invalid encrypted checkpoint size")
        return target
    except BaseException:
        if target:
            target.unlink(missing_ok=True)
        raise


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    command = cli.add_subparsers(dest="action", required=True)
    command.add_parser("pack")
    puller = command.add_parser("fetch")
    puller.add_argument("--host", required=True)
    puller.add_argument("--key", required=True, type=Path)
    puller.add_argument("--directory", required=True, type=Path)
    check = command.add_parser("verify")
    check.add_argument("--age", type=Path, required=True)
    check.add_argument("--archive", type=Path, required=True)
    check.add_argument("--scratch", type=Path, required=True)
    args = cli.parse_args()
    if args.action == "pack":
        pack(STATE, PIN, sys.stdout.buffer)
    elif args.action == "fetch":
        print(fetch(args.host, args.key, args.directory.resolve(strict=True)))
    else:
        if not 128 < args.archive.stat().st_size <= LIMIT+262144:
            raise ValueError("invalid encrypted checkpoint size")
        pipe = subprocess.run([str(args.age), "-d", "-i", "-", str(args.archive)],
                              stdout=subprocess.PIPE, check=True, timeout=30)
        with TemporaryDirectory(prefix="stock-forward-verify-", dir=args.scratch) as tmp:
            result = verify(BytesIO(pipe.stdout), Path(tmp), datetime.now(UTC))
        print(json.dumps(result))


if __name__ == "__main__":
    main()
