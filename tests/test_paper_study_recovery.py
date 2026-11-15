import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from scripts import paper_study_recovery as recovery
from scripts import watch_paper_study_windows as watch


def test_online_copy_contains_wal_writes_and_refuses_overwrite(tmp_path):
    source, restored = tmp_path / "source.db", tmp_path / "snapshot.db"
    with closing(sqlite3.connect(source)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE evidence (n int)")
        conn.execute("INSERT INTO evidence VALUES (1)")
        conn.commit()
        recovery.snapshot(source, restored)
        with pytest.raises(ValueError, match="new file"):
            recovery.snapshot(source, restored)
        assert conn.execute("SELECT n FROM evidence").fetchall() == [(1,)]
    with closing(sqlite3.connect(restored)) as conn:
        assert conn.execute("SELECT n FROM evidence").fetchall() == [(1,)]
        assert conn.execute("PRAGMA journal_mode").fetchone() == ("delete",)


@pytest.mark.parametrize("command", ["id", "fetch ../../private", "fetch study-20261115T010101Z.tar.gz.gpg extra", ""])
def test_forced_command_refuses_shell_and_filename_injection(command):
    with pytest.raises(PermissionError):
        recovery.serve(command)


def test_backup_download_checks_hash_and_never_keeps_a_partial_file(tmp_path, monkeypatch):
    name = "study-20261115T010101Z.tar.gz.gpg"
    info = {"name": name, "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    calls = []

    def stub(_prefix, command, *, output, timeout):
        calls.append(command)
        output.write(b"bad" if len(calls) == 1 else b"abc")

    monkeypatch.setattr(watch, "_call_ssh", stub)
    with pytest.raises(ValueError, match="digest"):
        watch.pull_backup([], info, tmp_path)
    assert not list(tmp_path.iterdir())
    assert watch.pull_backup([], info, tmp_path)
    assert not watch.pull_backup([], info, tmp_path)
    assert calls == [f"fetch {name}"] * 2


def test_external_alert_requires_two_failures_and_can_fire_again_after_recovery():
    issues = ["vps_check_failed"]
    state, fire = watch.transition({}, issues, 100)
    assert not fire
    state, fire = watch.transition(state, issues, 400)
    assert fire
    state, fire = watch.transition(state, issues, 700)
    assert not fire
    state, fire = watch.transition(state, [], 1000)
    assert not fire and state["failed_checks"] == 0
    state, fire = watch.transition(state, issues, 1300)
    assert not fire
    _, fire = watch.transition(state, issues, 1600)
    assert fire


def test_truncated_published_archive_is_not_available(tmp_path):
    name = "study-20261115T010101Z.tar.gz.gpg"
    (tmp_path / name).write_bytes(b"a")
    (tmp_path / "latest.json").write_text(json.dumps({
        "name": name, "sha256": "0" * 64, "bytes": 2,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="truncated"):
        recovery._latest(Path(tmp_path))
