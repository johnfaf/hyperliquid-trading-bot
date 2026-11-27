#!/usr/bin/env python3
"""Replay stock entry alternatives at a fixed as-of time, without an order API."""
import argparse
from contextlib import closing
from datetime import UTC,date,datetime,time
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts.run_stock_orb_replay import History, NY, fingerprint, publish, quote_window, utc
from src.backtest.stock_entry_lab import LAB, RULES, compare_signals, stressed_replays
from src.trading.stock_orb import ENTRY_BARS,HOLD,RULE,STEP,bars_before,exit_decision,first_signal


def session(feed,day):
    values=feed.get("calendar",start=day.isoformat(),end=day.isoformat())
    if len(values)!=1 or values[0]["date"]!=day.isoformat():
        raise ValueError("regular US session is not verified")
    opening,closing=(time.fromisoformat(values[0][k]) for k in ("open","close"))
    if opening!=time(9,30) or not time(12)<=closing<=time(16) or closing.tzinfo is not None:
        raise ValueError("unexpected market hours")
    return tuple(int(datetime.combine(day,t,NY).timestamp()) for t in (opening,closing))


def capture(feed,day,as_of):
    opened,closed=session(feed,day)
    cutoff=min(int(as_of.timestamp()),closed,opened+STEP*ENTRY_BARS+HOLD+30)
    if cutoff<=opened:
        raise ValueError("cutoff precedes market open")
    reply=feed.get("bars",timeframe="5Min",start=utc(opened),end=utc(cutoff-1),
                   feed="iex",adjustment="raw",sort="asc",limit=150)
    if reply.get("next_page_token"):
        raise ValueError("incomplete bars window")
    rows=reply["bars"]
    bars,decisions=compare_signals(rows,opened,closed,cutoff)
    slots=set()
    for signal,_ in decisions.values():
        if signal is None:
            continue
        slots.add(signal.at)
        leave=exit_decision(bars,signal.at,signal.stop,cutoff)
        if leave:
            slots.add(leave[0])
    quotes={}
    for at in sorted(slots):
        if at+30<=cutoff:
            try:
                quotes[str(at)]=quote_window(feed,at)
            except (OSError,ValueError,KeyError,TypeError) as error:
                print(f"Quote window at {utc(at)} unavailable: {type(error).__name__}",file=sys.stderr)
                quotes[str(at)]=None
    return {"opened":opened,"closed":closed,"cutoff":cutoff,"bars":rows,"quotes":quotes}


def review_day(day,evidence):
    opened,closed,cutoff=(evidence[k] for k in ("opened","closed","cutoff"))
    bars,decisions=compare_signals(evidence["bars"],opened,closed,cutoff)
    plans,trace={},{}
    for name in RULES:
        signal,reason=decisions[name]
        state="No entry signal"
        if signal is None:
            if reason in ("missing_bars","missing_vwap_data"):
                state="Missing input"
            elif cutoff<opened+ENTRY_BARS*STEP:
                state="No signal yet"
            plans[name]={"state":state,"reason":reason,"signal_at":None,"results":None}
            continue
        plan={"signal_at":utc(signal.at),"signal_close":float(signal.price),"range_high":float(signal.range_high),
              "stop":float(signal.stop)}
        exits=exit_decision(bars,signal.at,signal.stop,cutoff)
        when,why=exits if exits else (None,None)
        entries_raw=evidence["quotes"].get(str(signal.at))
        if signal.at+30>cutoff:
            state="Awaiting entry quote window"
        elif entries_raw is None:
            state="Entry quote evidence unavailable"
        elif when is None or when+30>cutoff:
            state="Exit window pending"
        elif evidence["quotes"].get(str(when)) is None:
            state="Exit quote evidence unavailable"
        else:
            state="Scored in replay"
        results=None
        if entries_raw is not None:
            results,trace[name]=stressed_replays(
                [(day,signal,entries_raw,when,why,evidence["quotes"].get(str(when)) or [])],cutoff=cutoff,
            )
        plans[name]={**plan,"state":state,"exit_at":utc(when) if when else None,
                     "reason":why if state=="Scored in replay" else None,"results":results}
    first=[bars.get(opened+i*STEP) for i in range(6)]
    opening=None
    if None not in first:
        opening={"high":float(max(b.high for b in first)),"low":float(min(b.low for b in first))}
    return {"rule":LAB,"day":day,"as_of":utc(cutoff),"session_open":utc(opened),"session_close":utc(closed),
            "opening_range":opening,"book":1000,"sleeve":75,"cash":0,"live_approval":False,
            "paper_approval":False,"plans":plans},trace


def stress_holdout(path):
    uri=path.resolve(strict=True).as_uri()+"?mode=ro"
    with closing(sqlite3.connect(uri,uri=True,timeout=2)) as db:
        draft=json.loads(db.execute("SELECT definition FROM protocol WHERE id=1").fetchone()[0])
        if draft.get("rule")!=RULE or draft.get("replay_sha256")!=fingerprint():
            raise ValueError("this holdout does not match the original opening-range evaluator")
        rows=db.execute("""SELECT s.day,s.opened,s.closed,c.payload FROM sessions s LEFT JOIN captures c
                           ON c.day=s.day ORDER BY s.day LIMIT 101""").fetchall()
    if not 40<=len(rows)<=100:
        raise ValueError("original holdout is not bounded")
    records=[]
    missing=0
    for day,opened,closed,raw in rows:
        if raw is None:
            missing+=1
            continue
        through=opened+ENTRY_BARS*STEP+HOLD
        payload=json.loads(raw)
        bars=bars_before(payload["rows"],opened,closed,through)
        sig=first_signal(bars,opened,through)
        if sig is None:
            continue
        out,why=exit_decision(bars,sig.at,sig.stop,through)
        records.append((day,sig,payload["quotes"][str(sig.at)],out,why,payload["quotes"][str(out)]))
    cutoff=max(r[2] for r in rows)
    results,trace=stressed_replays(records,cutoff=cutoff)
    return {"start":draft["start"],"end":draft["end"],"sessions":len(rows),"missing":missing,
            "signals":len(records),"results":results},trace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--day",required=True,type=date.fromisoformat)
    p.add_argument("--as-of",required=True,type=datetime.fromisoformat)
    p.add_argument("--state-dir",required=True,type=Path)
    p.add_argument("--orb-holdout",type=Path)
    args=p.parse_args()
    cutoff=args.as_of
    if cutoff.utcoffset() is None or cutoff>datetime.now(UTC) or cutoff.astimezone(NY).date()!=args.day:
        p.error("cutoff must be historical, explicit, and on the requested New York date")
    key,secret=(os.environ.get(k) for k in ("STOCK_PAPER_KEY","STOCK_PAPER_SECRET"))
    if os.environ.get("STOCK_PAPER_SEND_ORDERS")!="false" or not key or not secret:
        p.error("read-only Alpaca market configuration unavailable")
    locked=args.state_dir/"protocol.json"
    proto={"rule":LAB,"day":args.day.isoformat(),"requested_cutoff":cutoff.isoformat(),
           "source_sha256":sha256(b"".join((ROOT/file).read_bytes() for file in (
               "scripts/run_stock_entry_lab.py","src/backtest/stock_entry_lab.py",
           ))).hexdigest(),"original_replay_sha256":fingerprint()}
    if locked.exists():
        if json.loads(locked.read_text(encoding="ascii"))!=proto:
            p.error("session, cutoff, or rule already pinned to this directory")
    else:
        with locked.open("x",encoding="ascii") as out:
            json.dump(proto,out)
            out.flush()
            os.fsync(out.fileno())
    input_file=args.state_dir/"inputs.json"
    if input_file.exists():
        inputs=json.loads(input_file.read_text(encoding="ascii"))
    else:
        inputs=capture(History(key,secret),args.day,cutoff)
        publish(input_file,inputs)
    report,trace=review_day(args.day.isoformat(),inputs)
    report["requested_cutoff"]=proto["requested_cutoff"]
    if args.orb_holdout:
        report["orb_holdout"],trace["orb_holdout"]=stress_holdout(args.orb_holdout)
    publish(args.state_dir/"trace.json",trace)
    publish(args.state_dir/"latest.json",report)
    print(json.dumps(report,allow_nan=False))


if __name__=="__main__":
    main()
