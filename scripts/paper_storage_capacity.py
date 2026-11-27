"""Read-only capacity check shared by VPS monitoring and the guarded rollout."""
import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from shutil import disk_usage

GIB = 1024**3
RUN_RESERVE = 2*GIB
SIDECARS = ("decisions", "orders", "risk", "funding")


def sources(primary, stock):
    found = {"bot":primary}
    for key in SIDECARS:
        path = primary.with_suffix(f".{key}.sqlite3")
        if path.exists():
            found[key] = path
    if stock is not None and stock.exists():
        found["stock"] = stock
    return found


def measure(primary, stock, rollback_dir):
    ledgers = sources(primary, stock)
    logical = ledgers["bot"].stat().st_size
    source_bytes = 0
    for path in ledgers.values():
        source_bytes += path.stat().st_size
        wal = path.with_name(path.name+"-wal")
        try:
            source_bytes += wal.stat().st_size
        except FileNotFoundError:
            pass
    active, rollback = disk_usage(primary.parent), disk_usage(rollback_dir)
    required = 2*source_bytes + RUN_RESERVE
    return {"paper_db_bytes":logical, "source_bytes":source_bytes, "active_free":active.free,
            "active_total":active.total, "rollback_free":rollback.free, "required":required,
            "shortfall":max(0,required-rollback.free,RUN_RESERVE-active.free)}


def track(sample, history, now):
    instant = int(now.timestamp())
    series = {}
    for row in history if isinstance(history,list) else []:
        if not isinstance(row,list) or len(row) != 2 or any(type(x) is not int for x in row):
            continue
        at, size = row
        if instant-48*3600 <= at <= instant and size > 0:
            series[at//3600] = row
    prior = [series[h] for h in sorted(series) if 4*3600 <= instant-series[h][0] <= 25*3600]
    growth = None
    if prior:
        at, size = prior[0]
        growth = (sample["paper_db_bytes"]-size)*86400/(instant-at)
    headroom = sample["active_free"]-RUN_RESERVE
    remaining = max(0,headroom)/growth if growth is not None and growth > 0 else None
    ratio = sample["active_free"]/sample["active_total"]
    severity = ("critical" if sample["active_free"] <= 4*GIB or ratio <= .05 else
                "low" if ratio <= .12 or (remaining is not None and remaining <= 3) else "normal")
    series.setdefault(instant//3600,[instant,sample["paper_db_bytes"]])
    return {**sample, "severity":severity, "observed_at":now.isoformat(),
            "growth_bytes_per_day":round(growth) if growth is not None else None,
            "days_to_reserve":round(remaining,2) if remaining is not None else None}, [series[h] for h in sorted(series)][-49:]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paper-db",type=Path,required=True)
    p.add_argument("--stock-db",type=Path)
    p.add_argument("--rollbacks",type=Path,required=True)
    p.add_argument("--require-rollout-space",action="store_true")
    args=p.parse_args()
    sample=measure(args.paper_db,args.stock_db,args.rollbacks)
    print(json.dumps(sample))
    if args.require_rollout_space and sample["shortfall"]:
        p.exit(1,"Guarded rollout blocked: not enough space for two copies and operating reserve.\n")


if __name__ == "__main__":
    main()
