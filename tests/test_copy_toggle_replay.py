from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import sqlite3

import pytest

from src.learning.copy_toggle_replay import compare


def test_copy_toggle_carries_prior_positions_and_never_scores_future_close(tmp_path):
    start = datetime(2026,11,16,tzinfo=timezone.utc)
    end = start+timedelta(days=7)
    path = tmp_path/"bot.db"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("""
            CREATE TABLE paper_trades(id PRIMARY KEY,opened_at,closed_at,entry_price,exit_price,size,leverage,pnl,metadata)
        """)
        for n,(opened,closed,source,pnl) in enumerate((
            (start-timedelta(days=1),start+timedelta(hours=2),"copy_trade",-1),
            (start,start+timedelta(days=2),"copy_trade",-4),
            (start,start+timedelta(days=3),"strategy",6),
            (start,end,"copy_trade",3000),
            (start,None,"strategy",None),
        )):
            db.execute("INSERT INTO paper_trades VALUES (?,?,?,100,100,1,2,?,?)",
                       (n,opened.isoformat(),closed.isoformat() if closed else None,pnl,json.dumps({"source":source})))
    r = compare(path,start,end)
    assert r["copy_on"]["base"]["net"] == 1
    assert r["copy_off"]["base"]["net"] == 5
    assert r["carry_in_copy_closes"] == 1 and r["omitted_copy"]["base"]["n"] == 1
    assert r["open_at_end"] == 2 and r["omitted_open_at_end"] == 1
    assert r["delta_stress"] == pytest.approx(4.32)
    assert r["copy_on"]["reported_fees"] is None
    assert r["alternate_admissions_measured"] is r["live_approved"] is False
