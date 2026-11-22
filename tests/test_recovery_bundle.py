from contextlib import closing
from io import BytesIO
import sqlite3

import pytest

from scripts import recovery_bundle
from src.learning.promotion_review import review


def test_recovery_bundle_round_trip_and_required_ledgers(tmp_path):
    revision = "a"*40
    scratch = tmp_path/"scratch"
    scratch.mkdir()
    for name in recovery_bundle.REQUIRED:
        file = tmp_path/f"{name[:-3]}-pre-{revision}.db"
        with closing(sqlite3.connect(file)) as conn, conn:
            for table in recovery_bundle.LEDGERS[name]:
                conn.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY)")
    archive = BytesIO()
    recovery_bundle.pack(tmp_path, revision, archive)
    assert recovery_bundle.latest(tmp_path)["revision"] == revision
    restored = recovery_bundle.restore(BytesIO(archive.getvalue()), scratch, revision)
    assert restored["verified"] is True
    assert set(restored["files"]) == recovery_bundle.REQUIRED
    assert restored["snapshot_at"] is not None
    wrong = tmp_path/"wrong-revision"
    wrong.mkdir()
    with pytest.raises(ValueError, match="manifest"):
        recovery_bundle.restore(BytesIO(archive.getvalue()), wrong, "b"*40)


def test_release_review_never_grants_live_permission():
    outcome = {"closed": 35, "net": 55., "net_without_largest": 35., "cash": 0.}
    host = {"forward_shadow": {"state": "Window complete", "stale": False, "results": {"base": outcome, "stress": outcome}}}
    evidence = {"available": True, "rows": [{"state": "Ready for human review"}]}
    report = {row["check"]: row["result"] for row in review(evidence, host)}
    assert report["Frozen BTC study"] == "Passed evidence screen only"
    assert report["Off-host recovery"] == "No verified snapshot"
    assert report["Live release approval"] == "Locked"
    outcome["closed"] = 0
    assert review(evidence, host)[1]["result"] == "Too few completed trades"
