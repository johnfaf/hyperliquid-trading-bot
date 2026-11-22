import gzip
from hashlib import sha256

import pytest

from scripts import archive_sqlite_rollback as backups


def test_archive_replaces_only_the_verified_immutable_file(tmp_path, monkeypatch):
    monkeypatch.setattr(backups, "_sync_dir", lambda _: None)
    source = tmp_path / ("bot-pre-" + "a"*40 + ".db")
    data = b"immutable backup"*10_000
    source.write_bytes(data)
    summary = backups.archive(tmp_path, source.name)
    assert not source.exists() and summary["bytes_saved"] > 0
    assert sha256(data).hexdigest() == summary["sha256_uncompressed"]
    assert gzip.decompress((tmp_path/summary["archive"]).read_bytes()) == data


def test_archive_rejects_live_names_and_sqlite_sidecars(tmp_path):
    name = "bot-pre-" + "b"*40 + ".db"
    (tmp_path/name).write_bytes(b"backup")
    (tmp_path/(name + "-wal")).touch()
    with pytest.raises(ValueError, match="sidecars"):
        backups.archive(tmp_path, name)
    with pytest.raises(ValueError, match="exact rollout backup filename"):
        backups.archive(tmp_path, "bot.db")
    assert (tmp_path/name).read_bytes() == b"backup"
