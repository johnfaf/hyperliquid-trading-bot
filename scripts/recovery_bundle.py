"""Stream immutable rollout ledgers and verify a decrypted copy in scratch."""
from argparse import ArgumentParser
from contextlib import closing
from datetime import datetime, timezone
import gzip
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import monotonic


LEDGERS = {
    "bot.db": {"paper_account", "paper_trades", "decision_snapshots"},
    "decisions.db": {"evidence"},
    "orders.db": {"orders", "fills"},
    "risk.db": {"account_equity_samples"},
    "funding.db": {"paper_oracle_samples", "paper_funding_rates"},
    "stock.db": {"orders", "fills"},
}
REQUIRED = {"bot.db", "decisions.db", "risk.db", "funding.db", "stock.db"}
LIMIT = 20 * 1024**3
MANIFEST = "manifest.json"
REVISION = re.compile("[a-f0-9]{40}")


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = sha256()

    def read(self, size):
        data = self.stream.read(size)
        self.hash.update(data)
        return data


def signature(path):
    stamp = path.stat()
    return stamp.st_ino, stamp.st_size, stamp.st_mtime_ns


def immutable(path):
    return (path.is_file() and not path.is_symlink()
            and not any(Path(str(path)+suffix).exists() for suffix in ("-wal", "-journal", "-shm")))


def bundle_paths(root, revision):
    if not REVISION.fullmatch(revision):
        raise ValueError("a full rollout revision is required")
    paths = {name: root / f"{name[:-3]}-pre-{revision}.db" for name in LEDGERS}
    paths = {k: v for k, v in paths.items() if v.exists()}
    if not REQUIRED <= paths.keys() or not all(immutable(p) for p in paths.values()):
        raise ValueError("required immutable rollout ledgers are not available")
    if sum(p.stat().st_size for p in paths.values()) > LIMIT:
        raise ValueError("bundle exceeds the configured restore size limit")
    return paths


def pack(root, revision, stream):
    paths = bundle_paths(root, revision)
    snapshot = datetime.fromtimestamp(paths["bot.db"].stat().st_mtime, timezone.utc)
    manifest = {"version": 1, "revision": revision, "sealed_at": datetime.now(timezone.utc).isoformat(),
                "snapshot_at": snapshot.isoformat(), "files": {}}
    with gzip.GzipFile(fileobj=stream, mode="wb", compresslevel=1, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w|") as bundle:
            for name, path in sorted(paths.items()):
                before = signature(path)
                info = tarfile.TarInfo(name)
                info.size = before[1]
                with path.open("rb") as raw:
                    reader = HashReader(raw)
                    bundle.addfile(info, reader)
                if not immutable(path) or before != signature(path):
                    raise ValueError("a rollout source changed during capture")
                manifest["files"][name] = {"bytes": info.size, "sha256": reader.hash.hexdigest()}
            encoded = json.dumps(manifest, sort_keys=True).encode()
            info = tarfile.TarInfo(MANIFEST)
            info.size = len(encoded)
            bundle.addfile(info, BytesIO(encoded))


def restore(stream, scratch, revision):
    started = monotonic()
    observed, expected, used = {}, None, 0
    with tarfile.open(fileobj=stream, mode="r|gz") as bundle:
        for member in bundle:
            if not member.isfile() or expected is not None:
                raise ValueError("unexpected member in recovery bundle")
            with closing(bundle.extractfile(member)) as source:
                if member.name == MANIFEST and 0 < member.size <= 4096:
                    expected = json.load(source)
                    continue
                used += member.size
                if member.name not in LEDGERS or member.name in observed or member.size <= 0 or used > LIMIT:
                    raise ValueError("invalid or oversized ledger member")
                reader = HashReader(source)
                with (scratch/member.name).open("xb") as target:
                    shutil.copyfileobj(reader, target, length=1024*1024)
                observed[member.name] = {"bytes": member.size, "sha256": reader.hash.hexdigest()}
    if (not isinstance(expected, dict) or expected.get("version") != 1
            or not REVISION.fullmatch(expected.get("revision") or "") or expected["revision"] != revision
            or not REQUIRED <= observed.keys() or expected.get("files") != observed):
        raise ValueError("recovery manifest or hash mismatch")
    for name in observed:
        with closing(sqlite3.connect((scratch/name).resolve().as_uri()+"?mode=ro&immutable=1", uri=True)) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not LEDGERS[name] <= tables or conn.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValueError(f"restored {name} failed its SQLite checks")
    return {"verified": True, "revision": revision, "files": sorted(observed), "bytes_restored": used,
            "duration_seconds": round(monotonic()-started, 1), "snapshot_at": expected.get("snapshot_at")}


def latest(root):
    raw = sorted(root.glob("bot-pre-*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in raw:
        revision = path.stem.removeprefix("bot-pre-")
        try:
            bundle_paths(root, revision)
        except ValueError:
            continue
        return {"revision": revision, "snapshot_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
    raise ValueError("no complete rollout bundle found")


def pull(root, host, key):
    """Fetch an encrypted-only backup; this machine never sends the identity."""
    if not re.fullmatch(r"[a-z][a-z0-9_-]*@[a-zA-Z0-9][a-zA-Z0-9.-]*", host):
        raise ValueError("expected an explicit user@host")
    recipient = (root/"recipient.txt").read_text(encoding="ascii").strip()
    if not re.fullmatch(r"age1[a-z0-9]{58}", recipient):
        raise ValueError("invalid age recipient")
    ssh = ["ssh", "-oBatchMode=yes", "-oStrictHostKeyChecking=yes", "-oIdentitiesOnly=yes",
           "-oConnectTimeout=8", "-oServerAliveInterval=30", "-oServerAliveCountMax=3", "-i", str(key), host]
    remote = "sudo -n python3 /opt/hyperliquid-recovery/recovery_bundle.py"
    sources = "--rollbacks /srv/hyperliquid-bot/rollbacks"
    state = json.loads(subprocess.check_output(ssh+[f"{remote} latest {sources}"], timeout=25))
    revision = state["revision"]
    if not REVISION.fullmatch(revision):
        raise ValueError("invalid remote revision")
    target = root/f"paper-recovery-{revision}.tar.gz.age"
    if target.exists():
        return {**state, "downloaded": False}
    if shutil.disk_usage(root).free < 12 * 1024**3:
        raise ValueError("insufficient local free space for backup and restore scratch")
    publisher = "/opt/hyperliquid-recovery/age/age"
    action = f"bash -c 'set -o pipefail; {remote} pack {sources} --revision {revision} | {publisher} -r {recipient}'"
    temporary = None
    try:
        with NamedTemporaryFile(dir=root, prefix="encrypted-", suffix=".partial", delete=False) as output:
            temporary = Path(output.name)
            subprocess.run(ssh+[action], check=True, stdout=output, stdin=subprocess.DEVNULL, timeout=1500)
            output.flush()
            os.fsync(output.fileno())
        if not 128 < temporary.stat().st_size <= LIMIT:
            raise ValueError("invalid encrypted archive size")
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {**state, "downloaded": True, "bytes": target.stat().st_size}


def main():
    parser = ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="action", required=True)
    exporter = command.add_parser("pack")
    exporter.add_argument("--rollbacks", type=Path, required=True)
    exporter.add_argument("--revision", required=True)
    finder = command.add_parser("latest")
    finder.add_argument("--rollbacks", type=Path, required=True)
    puller = command.add_parser("pull")
    puller.add_argument("--directory", type=Path, required=True)
    puller.add_argument("--host", required=True)
    puller.add_argument("--key", required=True, type=Path)
    verifier = command.add_parser("verify")
    verifier.add_argument("--archive", type=Path, required=True)
    verifier.add_argument("--age", type=Path, required=True)
    verifier.add_argument("--scratch", type=Path, required=True)
    verifier.add_argument("--revision", required=True)
    args = parser.parse_args()
    if args.action == "pack":
        pack(args.rollbacks.resolve(strict=True), args.revision, sys.stdout.buffer)
        return
    if args.action == "latest":
        print(json.dumps(latest(args.rollbacks.resolve(strict=True))))
        return
    if args.action == "pull":
        print(json.dumps(pull(args.directory.resolve(strict=True), args.host, args.key)))
        return
    with TemporaryDirectory(prefix="restore-drill-", dir=args.scratch) as tmp:
        with subprocess.Popen([str(args.age), "-d", "-i", "-", str(args.archive)], stdout=subprocess.PIPE) as age:
            try:
                result = restore(age.stdout, Path(tmp), args.revision)
                while age.stdout.read(1024*1024):
                    pass  # Finish authentication before accepting the report.
                if age.wait(timeout=30):
                    raise ValueError("archive failed age authentication")
            finally:
                if age.poll() is None:
                    age.kill()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
