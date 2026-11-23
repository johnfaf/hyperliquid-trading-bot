"""Produce a paired read-only copy-on/copy-off contribution report."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path

from src.learning.copy_toggle_replay import compare


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--start", required=True, type=datetime.fromisoformat)
    parser.add_argument("--end", required=True, type=datetime.fromisoformat)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if any(t.utcoffset() is None for t in (args.start,args.end)) or args.end > datetime.now(timezone.utc):
        parser.error("use explicit offsets and an evaluation end in the past")
    result = compare(args.db, args.start.astimezone(timezone.utc), args.end.astimezone(timezone.utc))
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result,allow_nan=False))


if __name__ == "__main__":
    main()
