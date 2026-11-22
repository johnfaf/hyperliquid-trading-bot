"""Keep historical execution evidence separate from replayable paper sessions."""
from collections import Counter
from contextlib import closing
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import re
from zoneinfo import ZoneInfo

from src.data.decision_coverage import open_readonly
from src.trading.copy_shadow_report import clean_close, stress_cost, study_peers
from src.trading.trade_decision_memory import COPY_CONCENTRATION_STUDY, is_copy_metadata, number

FORMAT_VERSION = 1
CSV_FIELDS = ("time", "coin", "dir", "px", "sz", "ntl", "fee", "closedPnl")
PAPER_FIELDS = ("id", "opened_at", "closed_at", "coin", "side", "status", "pnl", "entry_price", "size", "metadata")
SYMBOL = re.compile(r"[A-Za-z0-9@][A-Za-z0-9.@:_/-]{0,49}")
SESSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}")
DIRECTIONS = {"Open Long", "Open Short", "Close Long", "Close Short", "Long > Short", "Short > Long"}
MAX_SOURCE_ROWS = 250_000

SCHEMA = """
    CREATE TABLE archive_format (version INTEGER NOT NULL);
    CREATE TABLE fill_sources (id INTEGER PRIMARY KEY, filename TEXT NOT NULL);
    CREATE TABLE fill_signatures (
        fingerprint TEXT PRIMARY KEY, export_time TEXT NOT NULL, utc_time TEXT,
        symbol TEXT NOT NULL, direction TEXT NOT NULL, price TEXT NOT NULL, quantity TEXT NOT NULL,
        notional TEXT NOT NULL, fee TEXT NOT NULL, closed_pnl TEXT NOT NULL
    );
    CREATE TABLE fill_occurrences (
        source_id INTEGER NOT NULL REFERENCES fill_sources(id), row_number INTEGER NOT NULL,
        fingerprint TEXT NOT NULL REFERENCES fill_signatures(fingerprint), PRIMARY KEY(source_id,row_number)
    );
    CREATE INDEX fill_occurrences_signature ON fill_occurrences(fingerprint);
    CREATE TABLE paper_sessions (
        name TEXT PRIMARY KEY, source_name TEXT NOT NULL, timeline_gaps INTEGER NOT NULL,
        metadata_json TEXT NOT NULL
    );
    CREATE TABLE archived_paper_trades (
        session TEXT NOT NULL REFERENCES paper_sessions(name), id INTEGER NOT NULL, opened_at TEXT NOT NULL,
        closed_at TEXT, coin TEXT NOT NULL, side TEXT NOT NULL, status TEXT NOT NULL, pnl REAL,
        entry_price REAL, size REAL, metadata TEXT NOT NULL, PRIMARY KEY(session,id)
    );
    CREATE INDEX archived_paper_entry_time ON archived_paper_trades(session,opened_at,id);
"""


def timestamp(value):
    value = datetime.fromisoformat(value)
    if value.utcoffset() is None:
        raise ValueError("timestamp missing timezone")
    return value.astimezone(timezone.utc)


def resolve_local_time(value, zone):
    if zone is None:
        return None
    choices = set()
    for fold in (0, 1):
        utc = value.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc)
        if utc.astimezone(zone).replace(tzinfo=None) == value:
            choices.add(utc)
    return choices.pop().isoformat() if len(choices) == 1 else None


def decimal_text(raw):
    text = raw.strip()
    if len(text) > 80:
        raise ValueError("numeric field invalid")
    value = Decimal(text)
    if not value.is_finite() or abs(value.as_tuple().exponent) > 40:
        raise ValueError("numeric field invalid")
    text = format(value, "f")
    return (text.rstrip("0").rstrip(".") if "." in text else text) if value else "0"


def fill_row(row, zone):
    observed = datetime.strptime(row["time"].strip(), "%d/%m/%Y - %H:%M:%S")
    symbol, direction = row["coin"].strip(), row["dir"].strip()
    if not SYMBOL.fullmatch(symbol) or direction not in DIRECTIONS:
        raise ValueError("unrecognized direction or symbol")
    numbers = tuple(decimal_text(row[key]) for key in CSV_FIELDS[3:])
    if any(Decimal(n) <= 0 for n in numbers[:2]) or Decimal(numbers[2]) < 0:
        raise ValueError("invalid size, notional or price")
    values = (observed.isoformat(), symbol, direction, *numbers)
    ident = sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()
    return ident, values[0], resolve_local_time(observed, zone), *values[1:]


def import_fills(db, sources, timezone_name):
    zone = ZoneInfo(timezone_name) if timezone_name else None
    rejected = rows = 0
    for source in sources:
        with source.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, strict=True)
            headers = reader.fieldnames or ()
            if len(headers) != len(CSV_FIELDS) or set(CSV_FIELDS) != set(headers):
                raise ValueError(f"unsupported trade CSV schema: {source.name}")
            source_id = db.execute("INSERT INTO fill_sources(filename) VALUES (?)", (source.name,)).lastrowid
            for record, row in enumerate(reader, 1):
                if record > MAX_SOURCE_ROWS:
                    raise ValueError(f"source exceeds row limit: {source.name}")
                rows += 1
                try:
                    if None in row or any(row[k] is None for k in CSV_FIELDS):
                        raise ValueError("malformed row")
                    normalized = fill_row(row, zone)
                except (ValueError, InvalidOperation):
                    rejected += 1
                    continue
                db.execute("INSERT OR IGNORE INTO fill_signatures VALUES (?,?,?,?,?,?,?,?,?,?)", normalized)
                db.execute("INSERT INTO fill_occurrences VALUES (?,?,?)", (source_id, record, normalized[0]))
    distinct, first, last, untimed = db.execute("""
        SELECT count(*), min(export_time), max(export_time), count(CASE WHEN utc_time IS NULL THEN 1 END)
        FROM fill_signatures
    """).fetchone()
    cross_exports = db.execute("""
        SELECT count(*) FROM (SELECT fingerprint FROM fill_occurrences GROUP BY fingerprint
                              HAVING count(DISTINCT source_id)>1)
    """).fetchone()[0]
    within_export = db.execute("""
        SELECT count(DISTINCT fingerprint) FROM (
            SELECT fingerprint FROM fill_occurrences GROUP BY source_id,fingerprint HAVING count(*)>1
        )
    """).fetchone()[0]
    return dict(sources=len(sources), total=rows, rejected=rejected, signatures=distinct, cross_exports=cross_exports,
                within_export=within_export, untimed=untimed, first=first, last=last, timezone=timezone_name)


def parse_paper(row):
    meta = json.loads(row["metadata"] if row["metadata"] is not None else "{}")
    if not isinstance(meta, dict):
        raise ValueError("metadata is not an object")
    copy = is_copy_metadata(meta)
    opened = timestamp(row["opened_at"])
    closed = timestamp(row["closed_at"]) if row["closed_at"] is not None else None
    ident, side, status = row["id"], str(row["side"]).lower(), row["status"]
    coin = str(row["coin"])
    if (type(ident) is not int or ident < 1 or not isinstance(row["coin"], str) or not SYMBOL.fullmatch(coin)
            or side not in {"long", "short"} or status not in {"open", "closed"}
            or (status == "closed") != (closed is not None) or (closed is not None and closed < opened)):
        raise ValueError("invalid paper trade identity or timeline")
    clean, peers = clean_close(meta), study_peers(meta)
    compact = {"is_copy_trade": copy}
    # Keep only the entry-time tag and the label-exclusion flag, not raw provider context.
    if not clean:
        compact["tainted"] = True
    if peers is not None:
        compact["entry_decision"] = {"copy_concentration_study": {"policy": COPY_CONCENTRATION_STUDY, "peers": peers}}
    values = (ident, opened.isoformat(), closed.isoformat() if closed is not None else None, coin, side, status,
              number(row["pnl"]), number(row["entry_price"]), number(row["size"]), json.dumps(compact))
    return values, copy, clean, peers


def import_paper_session(db, name, source):
    counts = Counter()
    start = latest = None
    with closing(open_readonly(source)) as ledger:
        ledger.execute("BEGIN")
        tables = {r[0] for r in ledger.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        columns = {r[1] for r in ledger.execute("PRAGMA table_info(paper_trades)")} if "paper_trades" in tables else set()
        if not set(PAPER_FIELDS).issubset(columns):
            raise ValueError(f"session {name}: missing required paper ledger columns")
        db.execute("INSERT INTO paper_sessions VALUES (?,?,0,'{}')", (name, source.name))
        seen = set()
        fields = ",".join(PAPER_FIELDS)
        for row in ledger.execute(f"SELECT {fields} FROM paper_trades"):
            counts["source_rows"] += 1
            if counts["source_rows"] > MAX_SOURCE_ROWS:
                raise ValueError(f"session {name}: paper ledger exceeds row limit")
            try:
                values, copy, clean, peers = parse_paper(row)
                if values[0] in seen:
                    raise ValueError("duplicate trade ID")
            except (ValueError, TypeError, OverflowError):
                # An invalid row could have occupied an overlap slot; never replay around it.
                counts["timeline_gaps"] += 1
                continue
            seen.add(values[0])
            db.execute("INSERT INTO archived_paper_trades VALUES (?,?,?,?,?,?,?,?,?,?,?)", (name, *values))
            counts["valid_rows"] += 1
            opened, closed = values[1:3]
            start = min(start, opened) if start else opened
            latest = max(latest, closed or opened) if latest else closed or opened
            if copy:
                counts["copy_entries"] += 1
                if closed is None:
                    counts["copy_open"] += 1
                elif not clean:
                    counts["copy_tainted"] += 1
                elif values[6] is None:
                    counts["copy_unpriced"] += 1
                else:
                    counts["copy_tagged" if peers is not None else "copy_reconstructed"] += 1
                    if stress_cost(row) is None:
                        counts["copy_unknown_cost"] += 1
        for table in ("decision_snapshots", "decision_stage_events"):
            if table in tables:
                counts[table] = ledger.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    summary = {**counts, "start": start, "latest": latest}
    db.execute("UPDATE paper_sessions SET timeline_gaps=?,metadata_json=? WHERE name=?",
               (counts["timeline_gaps"], json.dumps(summary, sort_keys=True), name))
    return summary


def archived_ledger(db, session):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='archive_format'").fetchone():
        if session is not None:
            raise ValueError("--session requires a historical archive")
        return "paper_trades", "", ()
    version = db.execute("SELECT version FROM archive_format").fetchall()
    if len(version) != 1 or version[0][0] != FORMAT_VERSION:
        raise ValueError("historical archive is incomplete or unsupported")
    if session is None:
        raise ValueError("historical archives require --session; CSV fills cannot be used as decisions")
    row = db.execute("SELECT timeline_gaps,metadata_json FROM paper_sessions WHERE name=?", (session,)).fetchone()
    if row is None:
        raise ValueError("unknown paper session; CSV fills are execution-only")
    if row[0]:
        raise ValueError("this session is blocked: invalid ledger rows could change historical exposure")
    if not json.loads(row[1]).get("copy_entries"):
        raise ValueError("this session has no copy decisions to replay")
    return "archived_paper_trades", "session=? AND ", (session,)


def quality_report(fills, sessions):
    lines = ["# Historical import: data quality", "", "Offline research only. No live database, trading settings, or orders changed.",
             "", "## Hyperliquid fill exports (execution evidence only)",
             f"- {fills['sources']:,} exports; {fills['total']:,} input rows; {fills['rejected']:,} rejected rows.",
             f"- {fills['signatures']:,} distinct normalized row signatures, **not** verified unique fills.",
             f"- {fills['cross_exports']:,} signatures repeat across different files; {fills['within_export']:,} repeat within at least one file. "
             "Every valid occurrence was retained with source and row number. The source exports have no fill IDs."]
    if fills["first"]:
        lines.append(f"- Export timestamp range: {fills['first']} to {fills['last']}. "
                     f"Timezone: {fills['timezone'] or 'unconfirmed'}. Signatures without a verified UTC timestamp: {fills['untimed']:,}.")
    lines += ["- Exported fills contribute **zero decision rows**. They can inform fee and execution analysis but cannot prove "
              "the copied source, historical candidate roster, or decision-time state.", "", "## Paper ledger sessions"]
    if not sessions:
        lines += ["No paper ledger supplied. Copy replay unavailable."]
    for name, row in sessions.items():
        def count(key):
            return row.get(key, 0)
        lines += [f"### {name}", f"- {count('valid_rows'):,} valid paper rows of {count('source_rows'):,}; "
                  f"{count('copy_entries'):,} identified copy entries and {count('copy_open'):,} still open in the snapshot."]
        if row["start"]:
            lines.append(f"- Recorded time range (UTC): {row['start']} to {row['latest']}.")
        if count("timeline_gaps"):
            lines.append(f"- **Replay blocked:** {count('timeline_gaps'):,} invalid ledger rows could affect exposure.")
        elif not count("copy_entries"):
            lines.append("- **No copy decisions to replay.**")
        lines += [f"- Clean priced copy closes: {count('copy_tagged'):,} with an entry-time overlap tag, "
                  f"{count('copy_reconstructed'):,} eligible for reconstructed overlap only. "
                  f"{count('copy_unknown_cost'):,} of these lack notional for the cost stress test.",
                  f"- Excluded labels: {count('copy_tainted'):,} tainted/reconciled; {count('copy_unpriced'):,} without finite P&L.",
                  f"- Source journal: {count('decision_snapshots'):,} decision snapshots and {count('decision_stage_events'):,} stage events found. "
                  "These are inventory counts only; the importer does not train on mutable snapshots or raw context.", ""]
    lines += ["Keep sessions separate: successive backups can describe the same trades. Legacy exposure is reconstructed "
              "from recorded opening/closing timestamps and cannot prove inventory or candidates missing from the source. "
              "Training and holdout cutoffs still apply; a replayable close is not proof of broader strategy performance.", ""]
    return "\n".join(lines)
