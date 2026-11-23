"""Compress a selected immutable rollout backup after round-trip verification."""
from argparse import ArgumentParser
from contextlib import closing
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile


def archive(root, name, *, allow_empty_wal=False):
    if not re.fullmatch(r"(?:bot|decisions|orders|risk|funding|stock)-pre-[a-f0-9]{40}\.db", name):
        raise ValueError("select an exact rollout backup filename")
    source = root.resolve(strict=True) / name
    target = source.with_suffix(".db.gz")
    if not source.is_file() or source.is_symlink() or target.exists() or target.is_symlink():
        raise ValueError("need a regular unarchived backup")
    def sidecars():
        result = []
        for suffix in ("-wal", "-journal", "-shm"):
            path = Path(str(source) + suffix)
            if path.is_symlink():
                raise ValueError("unexpected symbolic SQLite sidecar")
            if not path.exists():
                continue
            stamp = path.stat()
            if (not allow_empty_wal or not path.is_file() or suffix == "-journal"
                    or (suffix == "-wal" and stamp.st_size != 0)):
                raise ValueError("backup has SQLite sidecars; inspect it before archiving")
            result.append((suffix, stamp.st_ino, stamp.st_size, stamp.st_mtime_ns))
        return result

    untouched_sidecars = sidecars()
    original = source.stat()
    digest = sha256()
    temporary = None
    try:
        with NamedTemporaryFile(dir=source.parent, prefix=".archive-", delete=False) as output:
            temporary = Path(output.name)
            with source.open("rb") as raw, gzip.GzipFile(fileobj=output, mode="wb", compresslevel=1, mtime=0) as zipped:
                while chunk := raw.read(1024*1024):
                    digest.update(chunk)
                    zipped.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        copied = sha256()
        with closing(gzip.open(temporary, "rb")) as zipped:
            while chunk := zipped.read(1024*1024):
                copied.update(chunk)
        current = source.stat()
        if (copied.digest() != digest.digest() or sidecars() != untouched_sidecars
                or (current.st_ino, current.st_size, current.st_mtime_ns)
                != (original.st_ino, original.st_size, original.st_mtime_ns)):
            raise ValueError("archive verification or source stability check failed")
        # A hard link refuses to overwrite a concurrently created archive.
        os.link(temporary, target)
        _sync_dir(source.parent)
        source.unlink()
        _sync_dir(source.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"archive": target.name, "sha256_uncompressed": digest.hexdigest(),
            "bytes_saved": original.st_size - target.stat().st_size}


def _sync_dir(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("backup")
    parser.add_argument("--directory", default=Path("/srv/hyperliquid-bot/rollbacks"), type=Path)
    parser.add_argument("--allow-empty-wal", action="store_true",
                        help="For verified inactive snapshots only; preserve an empty WAL and its SHM unchanged")
    args = parser.parse_args()
    print(json.dumps(archive(args.directory, args.backup, allow_empty_wal=args.allow_empty_wal)))


if __name__ == "__main__":
    main()
