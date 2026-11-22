"""Read-only public Hyperliquid capture for the paper funding model."""
from argparse import ArgumentParser
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from scripts.fetch_hyperliquid_replay_market import post
from src.core.clock_provider import utc_now
from src.trading.paper_funding import FundingLedger, HOUR, MODEL, ORACLE_MAX_AGE, ledger_path, millis
from src.trading.paper_reporting import fields


def capture(path, *, send=post, clock=utc_now):
    ledger = FundingLedger(ledger_path(path))
    now = millis(clock())
    target = (now//HOUR+1)*HOUR
    oracle_count = 0
    if 0 < target-now <= ORACLE_MAX_AGE:
        meta, contexts = send({"type": "metaAndAssetCtxs"})
        universe = meta["universe"]
        if len(universe) != len(contexts):
            raise ValueError("asset contexts do not match the public universe")
        prices = {m["name"]: c.get("oraclePx") for m, c in zip(universe, contexts, strict=True)}
        received = millis(clock())
        if received < target:
            ledger.record(observed_ms=received, oracle_prices=prices)
            oracle_count = len(prices)
    with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True, timeout=2)) as conn:
        active = {}
        for coin, opened, meta in conn.execute("SELECT coin,opened_at,metadata FROM paper_trades WHERE status='open'"):
            start = millis(opened)
            if start is not None and fields(meta).get("paper_funding_model") == MODEL:
                active[coin] = min(start, active.get(coin, start))
    settled_hour = now//HOUR*HOUR
    count = 0
    for coin, opened in sorted(active.items()):
        first = max((opened//HOUR+1)*HOUR, settled_hour-47*HOUR)
        if first > settled_hour:
            continue
        if ledger.path.exists():
            with closing(sqlite3.connect(ledger.path.resolve().as_uri()+"?mode=ro", uri=True, timeout=2)) as conn:
                seen = conn.execute("SELECT count(*) FROM paper_funding_rates WHERE coin=? AND hour_ms BETWEEN ? AND ?",
                                    (coin, first, settled_hour)).fetchone()[0]
            if seen == (settled_hour-first)//HOUR+1:
                continue
        rows = send({"type": "fundingHistory", "coin": coin, "startTime": first, "endTime": now})
        if not isinstance(rows, list) or any(row.get("coin") != coin or type(row.get("time")) is not int
                                              or not first <= row["time"] <= now for row in rows):
            raise ValueError("settlement response exceeds requested coin or time range")
        ledger.record(observed_ms=millis(clock()), rates=rows)
        count += len(rows)
    return {"state": "ok", "oracle_assets": oracle_count, "published_rate_rows": count, "tracked_assets": len(active)}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--paper-db", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(capture(args.paper_db)))


if __name__ == "__main__":
    main()
