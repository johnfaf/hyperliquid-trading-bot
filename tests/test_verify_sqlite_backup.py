import hashlib
import sqlite3
from contextlib import closing

import pytest

from scripts.verify_sqlite_backup import verify_backup


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "restored #snapshot.db"
    with closing(sqlite3.connect(path)) as conn:
        with conn:
            conn.execute("CREATE TABLE paper_account (id, total_trades, winning_trades, total_pnl)")
            conn.execute("INSERT INTO paper_account VALUES (1, 1, 1, 2.5)")
            conn.execute("CREATE TABLE paper_trades (status, pnl)")
            conn.execute("INSERT INTO paper_trades VALUES ('closed', 2.5), ('open', NULL)")
    return path


def test_restore_verification_is_read_only(snapshot):
    data = snapshot.read_bytes()
    verify_backup(snapshot, hashlib.sha256(data).hexdigest())
    assert snapshot.read_bytes() == data


def test_restore_verification_rejects_wrong_digest(snapshot):
    with pytest.raises(ValueError, match="SHA-256"):
        verify_backup(snapshot, "0" * 64)


def test_restore_verification_rejects_inconsistent_ledger(snapshot):
    with closing(sqlite3.connect(snapshot)) as conn:
        with conn:
            conn.execute("INSERT INTO paper_trades VALUES ('closed', -1)")
    with pytest.raises(ValueError, match="account totals"):
        verify_backup(snapshot, hashlib.sha256(snapshot.read_bytes()).hexdigest())
