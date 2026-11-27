"""An optional SPY pilot that routes through the existing paper order owner."""
from contextlib import closing
from datetime import UTC, datetime, time
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from src.trading.stock_orb import HOLD, RULE, STEP, bars_before, exit_decision, first_signal, qualifies, rule_digest

PURPOSE = "intraday"
NY = ZoneInfo("America/New_York")


def approved(path, now):
    try:
        info=path.lstat()
        if path.is_symlink() or info.st_uid!=0 or info.st_mode & 0o022 or not 0<info.st_size<=16384:
            return False
        data=json.loads(path.read_text(encoding="utf-8"))
        produced=datetime.fromisoformat(data["published_at"])
        return (produced.utcoffset() is not None and 0 <= (now-produced).total_seconds() <= 30*86400
                and qualifies(data,digest=rule_digest()))
    except (OSError,ValueError,TypeError,KeyError,AttributeError):
        return False


class Pilot:
    def __init__(self,path,*,enabled):
        self.path=Path(path)
        self.enabled=enabled

    def read_bars(self,broker,now,closed):
        opened=int(datetime.combine(now.astimezone(NY).date(),time(9,30),NY).timestamp())
        if now.timestamp()<opened+7*STEP:
            return opened,{}
        reply=broker.request("/v2/stocks/SPY/bars",market_data=True,query={
            "timeframe":"5Min","feed":"iex","adjustment":"raw","limit":100,
            "start":datetime.fromtimestamp(opened,UTC).isoformat(),"end":now.isoformat(),
        })
        if reply.get("next_page_token"):
            raise ValueError("incomplete five-minute bars")
        return opened,bars_before(reply["bars"],opened,int(closed.timestamp()),now.timestamp())

    def candidate(self,broker,now,closed):
        if not self.enabled or (closed-now).total_seconds() < HOLD+STEP:
            return None
        try:
            opened,bars=self.read_bars(broker,now,closed)
            signal=first_signal(bars,opened,now.timestamp())
            if signal and 0 <= now.timestamp()-signal.at <= 30:
                return signal
        except (OSError,ValueError,TypeError,KeyError,ArithmeticError):
            pass
        return None

    def protect(self,broker,journal,now,closed):
        """Losing permission for new entries must not strand existing inventory."""
        orders=[o for o in journal.orders() if o["entry"] and o["intent"].get("_purpose")==PURPOSE]
        plan=None
        if orders:
            try:
                with closing(sqlite3.connect(self.path.resolve().as_uri()+"?mode=ro",uri=True,timeout=2)) as db:
                    plan=db.execute("SELECT signal_at,stop FROM plans WHERE account=? AND client_id=?",
                                    (journal.account,orders[-1]["client_id"])).fetchone()
            except (OSError,sqlite3.Error):
                pass
        if not plan:
            return "intraday_plan_unavailable"
        at,stop=plan
        if now.timestamp()>=at+HOLD:
            return "time_limit"
        try:
            _,bars=self.read_bars(broker,now,closed)
            result=exit_decision(bars,at,Decimal(stop),now.timestamp())
            return result[1] if result else None
        except (OSError,ValueError,TypeError,KeyError,ArithmeticError):
            return "missing_protection_data"

    def reserve_plan(self,account,client,signal):
        with closing(sqlite3.connect(self.path,timeout=5)) as db, db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("""CREATE TABLE IF NOT EXISTS plans (
                account TEXT NOT NULL, client_id TEXT NOT NULL, protocol TEXT NOT NULL,
                signal_at INTEGER, stop TEXT, signal_price TEXT, range_high TEXT,
                PRIMARY KEY(account,client_id)
            )""")
            row=(account,client,RULE,signal.at,str(signal.stop),str(signal.price),str(signal.range_high))
            db.execute("INSERT OR IGNORE INTO plans VALUES (?,?,?,?,?,?,?)",row)
            if db.execute("SELECT * FROM plans WHERE account=? AND client_id=?",row[:2]).fetchone()!=row:
                raise ValueError("intraday plan changed after reservation")
