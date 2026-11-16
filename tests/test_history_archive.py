from contextlib import closing
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import sqlite3
from zoneinfo import ZoneInfo

import pytest

from scripts.import_trading_history import create_archive, session_arg
from scripts.run_copy_walk_forward import load, point_in_time
from src.backtest.history_archive import CSV_FIELDS, resolve_local_time
from src.trading.trade_decision_memory import COPY_CONCENTRATION_STUDY


def paper_db(path, rows):
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE paper_trades(id, opened_at, closed_at, coin, side, status, pnl, entry_price, size, metadata);
            CREATE TABLE decision_snapshots (id);
            INSERT INTO decision_snapshots VALUES ('inventory only');
        """)
        db.executemany("INSERT INTO paper_trades VALUES (?,?,?,?,?,?,?,?,?,?)", rows)


def row(ident, opened, closed, *, meta=None, pnl=5):
    return (ident, opened, closed, "BTC", "long", "closed" if closed is not None else "open", pnl, 100, 2,
            json.dumps({"source": "copy_trade"} if meta is None else meta))


def test_csv_multifile_overlap_preserves_provenance_and_never_creates_decision(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    fill = ("05/04/2026 - 14:30:58", "BTC", "Open Long", "100.50", "1", "100.5", "0.2", "0")
    variant = (*fill[:3], "100.50000", *fill[4:])
    invalid = (*fill[:4], "0", *fill[5:])
    for path, items in ((a, [fill, fill, invalid]), (b, [variant])):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(CSV_FIELDS)
            writer.writerows(items)
    out, report = tmp_path / "archive.db", tmp_path / "report.md"
    fills, _ = create_archive([a, b], {}, None, out, report)
    assert (fills["total"], fills["rejected"], fills["signatures"], fills["cross_exports"],
            fills["within_export"], fills["untimed"]) == (4, 1, 1, 1, 1, 1)
    with closing(sqlite3.connect(out)) as db:
        assert db.execute("SELECT source_id,row_number FROM fill_occurrences ORDER BY 1,2").fetchall() == [(1, 1), (1, 2), (2, 1)]
    assert "not** verified unique fills" in report.read_text(encoding="utf-8")
    lo, hi = datetime(2026, 4, 5, tzinfo=timezone.utc), datetime(2026, 4, 6, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="require --session"):
        load(out, lo, hi)
    with pytest.raises(ValueError, match="CSV fills are execution-only"):
        load(out, lo, hi, session="a")


def test_timezone_explicit_but_ambiguous_or_nonexistent_local_times_stay_unknown():
    zone = ZoneInfo("Europe/London")
    assert resolve_local_time(datetime(2026, 4, 5, 14, 30), zone) == "2026-04-05T13:30:00+00:00"
    assert resolve_local_time(datetime(2026, 10, 25, 1, 30), zone) is None
    assert resolve_local_time(datetime(2026, 3, 29, 1, 30), zone) is None
    assert resolve_local_time(datetime(2026, 4, 5, 14, 30), None) is None


def test_replay_named_session_keeps_earlier_active_inventory_and_masks_tainted_labels(tmp_path):
    alpha, beta = tmp_path / "input-alpha.db", tmp_path / "input-beta.db"
    paper_db(alpha, [
        row(1, "2026-04-05T02:00:00+01:00", "2026-04-05T05:00:00+01:00",
            meta={"is_copy_trade": True, "close_reason": "live_reconciled_closed", "raw_secret_test": "discard"}),
        row(2, "2026-04-05T03:00:00+01:00", "2026-04-05T06:00:00+01:00"),
        row(3, "2026-04-05T03:25:00+01:00", "2026-04-05T06:00:00+01:00", meta={"is_copy_trade": True,
            "entry_decision": {"copy_concentration_study": {"policy": COPY_CONCENTRATION_STUDY, "peers": 0}, "future_pnl": -500}}),
    ])
    paper_db(beta, [row(1, "2026-04-05T02:10:00+00:00", "2026-04-05T07:00:00+00:00", pnl=-20)])
    before = sha256(alpha.read_bytes()).digest()
    out, report = tmp_path / "archive.db", tmp_path / "report.md"
    _, sessions = create_archive([], {"alpha": alpha, "beta": beta}, None, out, report)
    assert sha256(alpha.read_bytes()).digest() == before
    assert sessions["alpha"]["copy_tainted"] == sessions["alpha"]["copy_tagged"] == sessions["alpha"]["copy_reconstructed"] == 1
    assert sessions["alpha"]["decision_snapshots"] == 1
    start = datetime(2026, 4, 5, 2, tzinfo=timezone.utc)
    end = datetime(2026, 4, 5, 6, tzinfo=timezone.utc)
    rows = load(out, start, end, session="alpha")
    assert "raw_secret_test" not in rows[0]["metadata"] and "future_pnl" not in rows[2]["metadata"]
    entries, labels = point_in_time(rows)
    assert [e.overlaps for e in entries] == [False, True, False]
    assert list(labels) == [2, 3]
    entries, labels = point_in_time(load(out, start, end, session="beta"))
    assert len(entries) == 1 and not entries[0].overlaps and labels[1].net == -20


def test_bad_inventory_is_reported_but_replay_is_blocked(tmp_path):
    source = tmp_path / "input.db"
    paper_db(source, [row(1, "2026-04-05T02:00:00", None), row(2, "2026-04-05T03:00:00Z", None)])
    out, report = tmp_path / "archive.db", tmp_path / "report.md"
    create_archive([], {"session": source}, None, out, report)
    assert "Replay blocked" in report.read_text(encoding="utf-8")
    cutoff = datetime(2026, 4, 5, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="invalid ledger rows could change historical exposure"):
        load(out, cutoff, cutoff, session="session")


def test_import_never_overwrites_source_and_cleans_up_on_error(tmp_path):
    source = tmp_path / "input.db"
    source.write_bytes(b"not sqlite")
    out, report = tmp_path / "archive.db", tmp_path / "report.md"
    with pytest.raises(ValueError, match="nothing is overwritten"):
        create_archive([], {"session": source}, None, source, report)
    with pytest.raises(sqlite3.DatabaseError):
        create_archive([], {"session": source}, None, out, report)
    assert not out.exists() and not report.exists() and source.read_bytes() == b"not sqlite"
    with pytest.raises(ValueError, match="NAME=PATH"):
        session_arg("bad|label=some.db")
