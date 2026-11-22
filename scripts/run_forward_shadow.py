"""Frozen, prospective BTC shadow study using closed public candle observations."""
from argparse import ArgumentParser
from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile

from scripts.fetch_hyperliquid_replay_market import BAR, DAY, HOUR, fetch_candles, fetch_funding, iso
from src.backtest.candle_exit_replay import Bar, Decision, POSITION_FRACTION, aggregate, simulate

RULE = "btc_long_24h_channel_v1"
CAPITAL = 1000.
LOOKBACK = 96
DURATION = 28*DAY
COSTS = {"base": (5, 4.5), "stress": (12, 6)}
SCHEMA = """
CREATE TABLE IF NOT EXISTS protocol (id INTEGER PRIMARY KEY CHECK(id=1), frozen INTEGER, start INTEGER, end INTEGER, fingerprint TEXT);
CREATE TABLE IF NOT EXISTS bars (t INTEGER PRIMARY KEY, o REAL,h REAL,l REAL,c REAL,v REAL);
CREATE TABLE IF NOT EXISTS funding (t INTEGER PRIMARY KEY, published INTEGER, rate REAL);
"""


def decisions(bars):
    result = []
    for i, bar in enumerate(bars):
        prior = bars[max(0, i-LOOKBACK):i]
        breakout = len(prior) == LOOKBACK and bar.close > max(p.high for p in prior)
        result.append(Decision(int(breakout), 1. if breakout else 0., None, "channel" if prior else "warmup"))
    return result


def fingerprint():
    root = Path(__file__).resolve().parents[1]
    paths = ("scripts/run_forward_shadow.py", "src/backtest/candle_exit_replay.py")
    return sha256(b"\0".join((root/p).read_text(encoding="utf-8").encode() for p in paths)).hexdigest()


def freeze(db, now, digest):
    with db:
        start = (now//HOUR+1)*HOUR
        db.execute("INSERT OR IGNORE INTO protocol VALUES(1,?,?,?,?)", (now, start, start+DURATION, digest))
    frozen, start, end, prior = db.execute("SELECT frozen,start,end,fingerprint FROM protocol").fetchone()
    if prior != digest or not frozen < start < end:
        raise ValueError("frozen rule mismatch; use a separate study ledger")
    return frozen, start, end


def _append(db, table, records):
    columns = {"bars": 6, "funding": 3}[table]
    params = ",".join("?" for _ in range(columns))
    for row in records:
        before = db.execute(f"SELECT * FROM {table} WHERE t=?", (row[0],)).fetchone()
        if before is None:
            db.execute(f"INSERT INTO {table} VALUES({params})", row)
        elif before != row:
            raise ValueError("public history disagrees with the frozen capture")


def _missing_start(db, table, start, end, step):
    seen = {r[0] for r in db.execute(f"SELECT t FROM {table} WHERE t>=? AND t<?", (start, end))}
    return next((i for i in range(start, end, step) if i not in seen), end)


def compare(bars, rates, start, end, *, complete=False, trace=None):
    series = {b.t: b for b in bars}
    if any(t not in series for t in range(start-LOOKBACK*BAR, end, BAR)):
        raise ValueError("missing closed candle")
    if any(t not in rates for t in range((start//HOUR+1)*HOUR, end, HOUR)):
        raise ValueError("missing published funding hour")
    result = {}
    signals = decisions(bars)
    for name, (slip_bps, fee_bps) in COSTS.items():
        fee, slip = fee_bps/10_000, slip_bps/10_000
        sim = simulate(bars, signals, rates, start, end, policy="fixed", fee_rate=fee,
                       slip_bps=slip_bps, capital=CAPITAL, liquidate_at_end=complete,
                       trace=(lambda row: trace({"scenario": name, **row})) if trace else None)
        stats = aggregate([sim], CAPITAL)
        buy, sell = series[start].open*(1+slip), series[end-BAR].close*(1-slip)
        units = CAPITAL*POSITION_FRACTION/buy
        passive = (sell-buy)*units - (buy+sell)*units*fee
        passive -= sum(units*series[h].open*rates[h] for h in range((start//HOUR+1)*HOUR, end, HOUR))
        executed = [row for row in sim["trades"] if row["exit_reason"] != "window_end"]
        stats.update(cash=0., passive=passive, open=bool(sim["open"]), closed=len(executed),
                     net_without_largest=stats["net"]-max([0., *(v["pnl"] for v in sim["trades"])]))
        result[name] = stats
    return result


def atomic_output(path, data):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    temp.replace(path)


def cycle(root, *, now, fetch_bars=fetch_candles, fetch_rates=fetch_funding):
    root.mkdir(mode=0o700, exist_ok=True)
    with closing(sqlite3.connect(root/"market.sqlite3", timeout=3)) as db:
        db.execute("PRAGMA synchronous=FULL")
        db.executescript(SCHEMA)
        frozen, start, finish = freeze(db, now, fingerprint())
        end = min(finish, now//HOUR*HOUR)
        report = {"rule": RULE, "capital": CAPITAL, "frozen_at": iso(frozen), "start": iso(start),
                  "end": iso(finish), "evaluated_at": iso(end), "published_at": iso(now),
                  "state": "Waiting for forward candles", "results": None, "live_approval": False}
        first = start-LOOKBACK*BAR
        if end > first:
            lo = _missing_start(db, "bars", first, end, BAR)
            if lo < end:
                rows = fetch_bars("BTC", lo, end)
                with db:
                    _append(db, "bars", ((t, *row) for t, row in rows.items()))
        if end > start:
            lo = _missing_start(db, "funding", start+HOUR, end, HOUR)
            if lo < end:
                rows = fetch_rates("BTC", lo, end)
                with db:
                    _append(db, "funding", ((t, *row) for t, row in rows.items()))
            bars = [Bar(*r) for r in db.execute("SELECT * FROM bars WHERE t>=? AND t<? ORDER BY t", (first, end))]
            rates = dict(db.execute("SELECT t,rate FROM funding WHERE t>? AND t<?", (start, end)))
            trace = []
            try:
                report["results"] = compare(bars, rates, start, end, complete=end==finish, trace=trace.append)
            except ValueError:
                report["state"] = "Incomplete market data"
            else:
                report["state"] = "Window complete" if end == finish else "Observing"
                atomic_output(root/"trace.jsonl", "".join(json.dumps(row, allow_nan=False)+"\n" for row in trace))
        atomic_output(root/"latest.json", json.dumps(report, allow_nan=False))
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--state-dir", type=Path)
    modes.add_argument("--screen", type=Path, help="Retrospective screening on an existing immutable capture")
    parser.add_argument("--screen-out", type=Path)
    args = parser.parse_args()
    if args.screen:
        if not args.screen_out:
            parser.error("--screen requires --screen-out")
        from scripts.run_hyperliquid_exit_replay import load_market, offline_runtime
        from scripts.run_strategy_shadow import window_ms
        with offline_runtime():
            _, _, markets = load_market(args.screen.resolve(strict=True))
            bars, rates = markets["BTC"]
            result = {"rule": RULE, "capital": CAPITAL, "live_approval": False, "type": "retrospective_only",
                      "input_sha256": sha256(args.screen.read_bytes()).hexdigest(), "windows": {}}
            for name in ("screen", "date_holdout"):
                start, end = window_ms(name)
                visible = [bar for bar in bars if start-LOOKBACK*BAR <= bar.t < end]
                result["windows"][name] = {"start": iso(start), "end_exclusive": iso(end),
                                            **compare(visible, rates, start, end, complete=True)}
            with args.screen_out.open("x", encoding="utf-8") as f:
                json.dump(result, f, indent=2, allow_nan=False)
    else:
        if args.screen_out:
            parser.error("--screen-out is only for a retrospective run")
        result = cycle(args.state_dir, now=int(datetime.now(timezone.utc).timestamp()*1000))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
