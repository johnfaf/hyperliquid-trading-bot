#!/usr/bin/env python3
"""Replay one pre-registered SPY opening-range holdout; no broker order API."""
import argparse
from contextlib import closing
from datetime import UTC,date,datetime,time,timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import sys
from tempfile import NamedTemporaryFile
from time import sleep
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler,Request,build_opener
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.backtest.stock_orb import COSTS,later_quotes,simulate
from src.trading.stock_orb import ENTRY_BARS,HOLD,RULE,STEP,bars_before,exit_decision,first_signal,qualifies,rule_digest

NY=ZoneInfo("America/New_York")
ENDPOINTS={"calendar":"https://paper-api.alpaca.markets/v2/calendar",
           "bars":"https://data.alpaca.markets/v2/stocks/SPY/bars",
           "quotes":"https://data.alpaca.markets/v2/stocks/SPY/quotes"}


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        return None


class History:
    def __init__(self,key,secret):
        self.headers={"APCA-API-KEY-ID":key,"APCA-API-SECRET-KEY":secret}
        self.client=build_opener(NoRedirects())

    def get(self,endpoint,**query):
        req=Request(ENDPOINTS[endpoint]+"?"+urlencode(query),headers=self.headers,method="GET")
        with self.client.open(req,timeout=8) as reply:
            content=reply.read(2_000_001)
        if len(content)>2_000_000:
            raise ValueError("historical response too large")
        sleep(.09)
        return json.loads(content)


def utc(sec):
    return datetime.fromtimestamp(sec,UTC).isoformat()


def calendar(feed,start,end):
    records=[]
    for row in feed.get("calendar", start=start.isoformat(), end=(end-timedelta(days=1)).isoformat()):
        day=date.fromisoformat(row["date"])
        op,cl=(time.fromisoformat(row[k]) for k in ("open","close"))
        if not start <= day < end or day.weekday()>4 or op!=time(9,30) or cl.tzinfo is not None or not time(12)<=cl<=time(16):
            raise ValueError("unverified regular session")
        opened,closed=(int(datetime.combine(day,x,NY).timestamp()) for x in (op,cl))
        records.append((day.isoformat(),opened,closed))
    if not 40<=len(records)<=100 or len({r[0] for r in records})!=len(records):
        raise ValueError("holdout calendar not complete or too large")
    return sorted(records)


def quote_window(feed,at):
    query={"start":utc(at+2),"end":utc(at+30),"feed":"iex","sort":"asc","limit":5000}
    quotes,seen=[],set()
    for _ in range(8):
        page=feed.get("quotes",**query)
        batch=page["quotes"]
        if not isinstance(batch,list):
            raise ValueError("malformed historical quotes")
        quotes.extend(batch)
        token=page.get("next_page_token")
        if not token:
            return quotes
        if not isinstance(token,str) or not batch or token in seen:
            break
        seen.add(token)
        query["page_token"]=token
    raise ValueError("partial historical quote window")


def capture(feed,opened,closed):
    through=opened+ENTRY_BARS*STEP+HOLD
    if through+30>=closed:
        raise ValueError("session too short for fixed exit")
    payload=feed.get("bars", timeframe="5Min", start=utc(opened), end=utc(through-1), feed="iex",
                     adjustment="raw", sort="asc", limit=150)
    if payload.get("next_page_token"):
        raise ValueError("partial minute bar response")
    rows=payload["bars"]
    bars=bars_before(rows,opened,closed,through)
    signal=first_signal(bars,opened,through)
    quotes={}
    if signal:
        decision=exit_decision(bars,signal.at,signal.stop,through)
        for at in (signal.at,decision[0]):
            quotes[str(at)]=quote_window(feed,at)
    return {"rows":rows,"quotes":quotes}


def fingerprint():
    return sha256(b"".join((ROOT/name).read_bytes() for name in (
        "scripts/run_stock_orb_replay.py","src/backtest/stock_orb.py","src/trading/stock_orb.py",
    ))).hexdigest()


def preregister(db,start,end,now):
    if end>now.astimezone(NY).date() or not 56<=(end-start).days<=160:
        raise ValueError("use one bounded window of already completed sessions")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS protocol (id INTEGER PRIMARY KEY CHECK(id=1), definition TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sessions (day TEXT PRIMARY KEY, opened INTEGER, closed INTEGER);
        CREATE TABLE IF NOT EXISTS captures (day TEXT PRIMARY KEY, payload TEXT NOT NULL);
    """)
    manifest={"rule":RULE,"start":str(start),"end":str(end),"replay_sha256":fingerprint()}
    encoded=json.dumps(manifest,sort_keys=True)
    with db:
        db.execute("INSERT OR IGNORE INTO protocol VALUES(1,?)",(encoded,))
    if db.execute("SELECT definition FROM protocol WHERE id=1").fetchone()[0]!=encoded:
        raise ValueError("holdout already registered with different code or dates")
    return manifest


def collect(db,feed,start,end):
    days=db.execute("SELECT day,opened,closed FROM sessions ORDER BY day").fetchall()
    if not days:
        days=calendar(feed,start,end)
        with db:
            db.executemany("INSERT INTO sessions VALUES(?,?,?)",days)
    for day,opened,closed in days:
        if db.execute("SELECT 1 FROM captures WHERE day=?",(day,)).fetchone():
            continue
        try:
            sample=capture(feed,opened,closed)
        except (OSError,ValueError,KeyError,TypeError,ArithmeticError) as error:
            print(f"{day} capture unavailable: {type(error).__name__}",file=sys.stderr)
            continue
        with db:
            db.execute("INSERT INTO captures VALUES(?,?)",(day,json.dumps(sample,allow_nan=False)))


def evaluate(db,definition,now):
    all_rows=db.execute("""SELECT s.day,s.opened,s.closed,c.payload FROM sessions s LEFT JOIN captures c
                           ON c.day=s.day ORDER BY s.day""").fetchall()
    records,missing=[],[]
    for day,opened,closed,payload in all_rows:
        try:
            if payload is None:
                raise ValueError("no market capture")
            data=json.loads(payload)
            through=opened+ENTRY_BARS*STEP+HOLD
            bars=bars_before(data["rows"],opened,closed,through)
            signal=first_signal(bars,opened,through)
            if not signal:
                records.append((day,None,(),None,None,()))
                continue
            when,reason=exit_decision(bars,signal.at,signal.stop,through)
            entries=later_quotes(data["quotes"][str(signal.at)],signal.at)
            exits=later_quotes(data["quotes"][str(when)],when)
            records.append((day,signal,entries,when,reason,exits))
            if reason=="missing_protection_data":
                missing.append(day)
        except (ValueError,KeyError,TypeError,ArithmeticError):
            missing.append(day)
    results,trace={},{}
    for scenario in COSTS:
        results[scenario],trace[scenario]=simulate(records,scenario)
    report={**definition,"published_at":now.isoformat(),"strategy_sha256":rule_digest(),"live_approval":False,
            "window_closed":True,"feed":"iex","capital":1000,"sleeve":75,"sessions":len(all_rows),
            "covered":len(all_rows)-len(missing),"signals":sum(row[1] is not None for row in records),
            "missing_sessions":missing,"results":results}
    report["state"]="Eligible for supervised paper pilot" if qualifies(report,digest=rule_digest()) else "Not qualified"
    return report,trace


def publish(path,value):
    temp=None
    try:
        with NamedTemporaryFile(dir=path.parent,mode="w",encoding="ascii",delete=False) as stream:
            temp=Path(stream.name)
            json.dump(value,stream,allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir",type=Path,required=True)
    parser.add_argument("--start",type=date.fromisoformat,required=True)
    parser.add_argument("--end",type=date.fromisoformat,required=True)
    args=parser.parse_args()
    key,secret=(os.getenv(k) for k in ("STOCK_PAPER_KEY","STOCK_PAPER_SECRET"))
    if os.getenv("STOCK_PAPER_SEND_ORDERS")!="false" or not key or not secret:
        parser.error("read-only Alpaca market configuration unavailable")
    with closing(sqlite3.connect(args.state_dir/"holdout.sqlite3",timeout=8)) as db:
        definition=preregister(db,args.start,args.end,datetime.now(UTC))
        collect(db,History(key,secret),args.start,args.end)
        report,trace=evaluate(db,definition,datetime.now(UTC))
    publish(args.state_dir/"trace.json",trace)
    publish(args.state_dir/"latest.json",report)
    print(json.dumps({k:report[k] for k in ("state","sessions","covered","signals","results")},allow_nan=False))


if __name__=="__main__":
    main()
