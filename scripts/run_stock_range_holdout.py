#!/usr/bin/env python3
"""Capture and compare a fixed SPY range-strategy holdout. Research only."""
import argparse
from contextlib import closing
from datetime import UTC,date,datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts.run_stock_orb_replay import History, NY, calendar, publish, quote_window, utc
from src.backtest.stock_range_reversal import CASES, PROTOCOL, RULES, Sample, clears_screen, decisions, exit_decision, simulate
from src.trading.stock_orb import ENTRY_BARS,HOLD,STEP


def fingerprint():
    modules=("scripts/run_stock_range_holdout.py","scripts/run_stock_orb_replay.py",
             "src/backtest/stock_range_reversal.py","src/backtest/stock_entry_lab.py",
             "src/backtest/stock_orb.py","src/trading/stock_orb.py")
    return sha256(b"".join((ROOT/name).read_bytes() for name in modules)).hexdigest()


def preregister(db,start,end,now):
    if end>now.astimezone(NY).date() or not 56<=(end-start).days<=160:
        raise ValueError("request a bounded set of completed sessions")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS protocol (id INTEGER PRIMARY KEY CHECK(id=1), manifest TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sessions (day TEXT PRIMARY KEY, opened INTEGER, closed INTEGER);
        CREATE TABLE IF NOT EXISTS inputs (day TEXT PRIMARY KEY, receipt TEXT NOT NULL);
    """)
    proof={"rule":PROTOCOL,"start":str(start),"end":str(end),"source_sha256":fingerprint(),"feed":"iex"}
    encoded=json.dumps(proof,sort_keys=True)
    with db:
        db.execute("INSERT OR IGNORE INTO protocol VALUES(1,?)",(encoded,))
    if db.execute("SELECT manifest FROM protocol WHERE id=1").fetchone()[0]!=encoded:
        raise ValueError("capture folder already pinned to other rules or dates")
    return proof


def capture_day(feed,opened,closed):
    through=opened+ENTRY_BARS*STEP+HOLD
    if through+30>=closed:
        raise ValueError("cannot complete planned trade before regular close")
    reply=feed.get("bars",timeframe="5Min",start=utc(opened),end=utc(through-1),feed="iex",
                   sort="asc",adjustment="raw",limit=150)
    if reply.get("next_page_token"):
        raise ValueError("partial bar result")
    rows=reply["bars"]
    bars,plans=decisions(rows,opened,closed,through)
    stamps=set()
    for plan in plans.values():
        if plan:
            out=exit_decision(bars,plan,through)
            if not out or out[0]+30>closed:
                raise ValueError("exit cannot be verified")
            stamps.update((plan.at,out[0]))
    quotes={str(at):quote_window(feed,at) for at in sorted(stamps)}
    return {"bars":rows,"quotes":quotes}


def collect(db,feed,start,end):
    sessions=db.execute("SELECT day,opened,closed FROM sessions ORDER BY day").fetchall()
    if not sessions:
        sessions=calendar(feed,start,end)
        with db:
            db.executemany("INSERT INTO sessions VALUES(?,?,?)",sessions)
    for day,opened,closed in sessions:
        if db.execute("SELECT 1 FROM inputs WHERE day=?",(day,)).fetchone():
            continue
        try:
            record=capture_day(feed,opened,closed)
        except (OSError,KeyError,TypeError,ValueError,ArithmeticError) as error:
            print(f"{day} capture unavailable: {type(error).__name__}",file=sys.stderr,flush=True)
            continue
        with db:
            db.execute("INSERT INTO inputs VALUES(?,?)",(day,json.dumps(record,allow_nan=False)))


def evaluate(db,proof,now):
    rows=db.execute("""SELECT s.day,s.opened,s.closed,i.receipt FROM sessions s LEFT JOIN inputs i
                       ON i.day=s.day ORDER BY s.day LIMIT 101""").fetchall()
    if not 40<=len(rows)<=100:
        raise ValueError("invalid holdout calendar")
    missed=[]
    samples={rule:[] for rule in RULES}
    for day,opened,closed,raw in rows:
        if raw is None:
            missed.append(day)
            continue
        record=json.loads(raw)
        through=opened+ENTRY_BARS*STEP+HOLD
        bars,plans=decisions(record["bars"],opened,closed,through)
        for name,plan in plans.items():
            if plan is None:
                continue
            leave,reason=exit_decision(bars,plan,through)
            samples[name].append(Sample(day,plan,record["quotes"][str(plan.at)],leave,reason,record["quotes"][str(leave)]))
    covered=len(rows)-len(missed)
    cutoff=max(r[2] for r in rows)
    results,trace={},{}
    for rule in RULES:
        runs,trace[rule]={},{}
        for name in CASES:
            runs[name],trace[rule][name]=simulate(samples[rule],name,cutoff)
        results[rule]={"signals":len(samples[rule]),"results":runs,
                       "passes_economics_screen":clears_screen(len(rows),covered,runs)}
    report={**proof,"published_at":now.isoformat(),"sessions":len(rows),"covered":covered,"missing_sessions":missed,
            "capital":1000,"sleeve":75,"cash":0,"paper_approval":False,"live_approval":False,
            "short_borrow_verified":False,"rules":results}
    return report,trace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir",required=True,type=Path)
    parser.add_argument("--start",required=True,type=date.fromisoformat)
    parser.add_argument("--end",required=True,type=date.fromisoformat)
    args=parser.parse_args()
    key,secret=(os.getenv(k) for k in ("STOCK_PAPER_KEY","STOCK_PAPER_SECRET"))
    if os.getenv("STOCK_PAPER_SEND_ORDERS")!="false" or not key or not secret:
        parser.error("read-only Alpaca market configuration unavailable")
    with closing(sqlite3.connect(args.state_dir/"holdout.sqlite3",timeout=8)) as db:
        proof=preregister(db,args.start,args.end,datetime.now(UTC))
        collect(db,History(key,secret),args.start,args.end)
        report,trace=evaluate(db,proof,datetime.now(UTC))
    publish(args.state_dir/"trace.json",trace)
    publish(args.state_dir/"latest.json",report)
    print(json.dumps(report,allow_nan=False))


if __name__=="__main__":
    main()
