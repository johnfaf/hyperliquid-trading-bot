from contextlib import closing
from datetime import UTC, datetime, timedelta
from io import BytesIO
import json
import shutil
import sqlite3
from zipfile import ZipFile

import pytest

from scripts import publish_dashboard_status as publisher
from scripts import run_stock_forward_shadow as runner
from scripts import stock_forward_evidence as evidence
from tests.test_stock_forward_shadow import Feed


def registered(path, feed):
    with closing(sqlite3.connect(path)) as db:
        runner.cycle(db, feed, clock=feed.clock)


def capture(path, feed, minute, *, date=24):
    feed.now = datetime(2026, 11, date, 14, minute, 15, tzinfo=UTC)
    with closing(sqlite3.connect(path)) as db:
        runner.cycle(db, feed, clock=feed.clock)


def test_missing_quotes_and_completed_prices_are_not_hidden(tmp_path):
    path, feed = tmp_path/"market.sqlite3", Feed()
    registered(path, feed)
    capture(path, feed, 45)
    # A paused or slow evaluator is not a problem before the grace period passes.
    assert evidence.current(path, feed.now+timedelta(minutes=6))["missed"] == {}
    mature = feed.now+timedelta(minutes=8)
    assert evidence.current(path, mature)["missed"] == {"2026-11-24": "fill_quote"}
    feed.missing_bars = True
    capture(path, feed, 45, date=25)
    capture(path, feed, 50, date=25)
    mature = feed.now+timedelta(minutes=3)
    result = evidence.current(path, mature)
    assert result["frames"] == 3 and result["paired_sessions"] == 0
    assert result["missed"] == {"2026-11-24": "fill_quote", "2026-11-25": "signal_inputs"}
    saturday = datetime(2026, 11, 28, 17, tzinfo=UTC)
    assert "2026-11-28" not in evidence.current(path, saturday)["missed"]


def test_receipts_and_alerts_are_retryable_and_deduplicated(tmp_path, monkeypatch):
    path, feed = tmp_path/"market.sqlite3", Feed()
    registered(path, feed)
    now = datetime(2026, 11, 24, 15, tzinfo=UTC)
    found = evidence.current(path, now)
    sent = []

    def unreliable(*args):
        sent.append(args[-1])
        return len(sent) > 1

    monkeypatch.setattr(publisher, "send_telegram", unreliable)
    marker = tmp_path/"alerts.json"
    assert publisher.notify_forward_gaps(found, None, marker) == "delivery_unavailable"
    assert not marker.exists()
    assert publisher.notify_forward_gaps(found, None, marker) == "sent"
    assert publisher.notify_forward_gaps(found, None, marker) == "quiet"
    assert all("2026-11-24: both scheduled quotes missing" in m for m in sent)
    proof = {"verified": True, "encrypted_sha256": "f"*64, "revision": "a"*40, "checkpoint_at": now.isoformat(),
             "verified_at": now.isoformat(), **found}
    receipt = tmp_path/"proof.json"
    receipt.write_text(json.dumps(proof), encoding="utf-8")
    assert publisher.forward_restore_status(receipt, found, now)["current"] is True
    assert publisher.forward_restore_status(receipt, {**found, "frames": 1}, now)["current"] is False
    assert publisher.forward_restore_status(receipt, found, now-timedelta(seconds=1)) is None
    proof["protocol_sha256"] = "d"*64
    receipt.write_text(json.dumps(proof), encoding="utf-8")
    assert publisher.forward_restore_status(receipt, found, now) is None


def test_recovery_contains_exact_frozen_code_and_consistent_sqlite(tmp_path):
    path, feed = tmp_path/"market.sqlite3", Feed()
    registered(path, feed)
    capture(path, feed, 45)
    capture(path, feed, 50)
    pin = tmp_path/("a"*40)
    for name in evidence.SOURCES:
        target = pin/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(runner.ROOT/name, target)
    archive = BytesIO()
    evidence.pack(path, pin, archive, clock=feed.clock)
    scratch = tmp_path/"verify"
    scratch.mkdir()
    proof = evidence.verify(archive, scratch, feed.now)
    assert proof["paired_sessions"] == 1 and proof["frames"] == 2 and proof["verified"]
    with closing(evidence.read_db(scratch/evidence.LEDGER)) as db:
        assert db.execute("SELECT COUNT(*) FROM frames").fetchone() == (2,)
    with ZipFile(archive) as source:
        malformed = BytesIO()
        with ZipFile(malformed, "w") as target:
            for name in source.namelist():
                target.writestr(name, b"invalid" if name == evidence.LEDGER else source.read(name))
    with pytest.raises(ValueError, match="manifest"):
        evidence.verify(malformed, scratch, feed.now)
    with (pin/evidence.SOURCES[0]).open("ab") as wrong:
        wrong.write(b"\n# changed source\n")
    mismatch = BytesIO()
    evidence.pack(path, pin, mismatch, clock=feed.clock)
    extra = tmp_path/"bad"
    extra.mkdir()
    with pytest.raises(ValueError, match="frozen protocol"):
        evidence.verify(mismatch, extra, feed.now)


def test_audit_rejects_unreproducible_signal(tmp_path):
    path, feed = tmp_path/"market.sqlite3", Feed()
    registered(path, feed)
    capture(path, feed, 45)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE frames SET entry=0")
    assert publisher.stock_forward_integrity(path, feed.now) is None
