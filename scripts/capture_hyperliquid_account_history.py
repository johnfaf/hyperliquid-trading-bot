"""Archive account-scoped public Hyperliquid info responses; no order APIs."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re

from scripts.fetch_hyperliquid_replay_market import post
from scripts.import_hyperliquid_fills import import_fills
from src.backtest.history_archive import timestamp

PAGE_CAP = 2000
MONTH_MS = 30*86_400_000


def bounded(send, *, kind, account, start, end):
    result = []
    # Provider bounds are inclusive, while the audit window is half-open.
    pending = [(lo, min(end, lo+MONTH_MS)) for lo in range(start, end, MONTH_MS)][::-1]
    requests = 0
    while pending:
        lo, hi = pending.pop()
        requests += 1
        if requests > 400:
            raise ValueError("account capture request bound exceeded")
        rows = send({"type": kind, "user": account, "startTime": lo, "endTime": hi-1,
                     **({"aggregateByTime": False} if kind == "userFillsByTime" else {})})
        if not isinstance(rows, list) or any(not isinstance(r, dict) or type(r.get("time")) is not int
                                             or not lo <= r["time"] < hi for r in rows):
            raise ValueError("unexpected history response or ignored time bounds")
        if len(rows) >= PAGE_CAP:
            if hi-lo < 2:
                raise ValueError("one millisecond saturates the venue response cap")
            mid = (lo+hi)//2
            pending.extend(((mid, hi), (lo, mid)))
        else:
            result.extend(rows)
    return result


def capture(root, account, start, end, send=post):
    if not re.fullmatch("0x[0-9a-f]{40}", account):
        raise ValueError("public account address required")
    if not 0 < start < end or end-start > 740*86_400_000:
        raise ValueError("request a bounded window of at most 740 days")
    fills = bounded(send, kind="userFillsByTime", account=account, start=start, end=end)
    funding = bounded(send, kind="userFunding", account=account, start=start, end=end)
    orders = send({"type": "historicalOrders", "user": account})
    if not isinstance(orders, list) or any(not isinstance(row, dict) or not isinstance(row.get("order"), dict) for row in orders):
        raise ValueError("unexpected historicalOrders response")
    root.mkdir(mode=0o700)
    for name, rows in (("fills", fills), ("funding", funding), ("orders", orders)):
        fd = os.open(root / f"{name}.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(rows, f, sort_keys=True, allow_nan=False)
    added, repeats = import_fills(root/"fills.sqlite3", account, fills)
    manifest = {"captured_at": datetime.now(timezone.utc).isoformat(), "start_ms": start, "end_exclusive_ms": end,
                "account_sha256": sha256(account.encode()).hexdigest(),
                "fills": added, "exact_repeats": repeats, "funding_rows": len(funding), "historical_orders": len(orders),
                "historical_order_limit_reached": len(orders) >= PAGE_CAP, "statement_complete": False}
    with (root/"manifest.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--account", required=True, help="Public address only")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = capture(args.output, args.account.lower(), int(timestamp(args.start).timestamp()*1000),
                     int(timestamp(args.end).timestamp()*1000))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
