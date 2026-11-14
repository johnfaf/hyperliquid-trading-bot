import sqlite3
from contextlib import closing

import pytest

from scripts.import_hyperliquid_fills import import_fills


ACCOUNT = "0x" + "1" * 40
FILL = {
    "coin": "BTC", "time": 1700000000000, "tid": 7, "oid": 2, "side": "B",
    "px": "40000", "sz": "0.01", "fee": "0.18", "closedPnl": "0", "startPosition": "0",
}


def test_fill_import_keeps_partials_and_deduplicates_repeated_downloads(tmp_path):
    path = tmp_path / "fills.db"
    partials = [FILL, {**FILL, "tid": 8}]
    assert import_fills(path, ACCOUNT, partials) == (2, 0)
    assert import_fills(path, ACCOUNT, partials) == (0, 2)
    assert import_fills(path, "0x" + "2" * 40, [FILL]) == (1, 0)


def test_conflicting_fill_rolls_back_entire_batch(tmp_path):
    path = tmp_path / "fills.db"
    import_fills(path, ACCOUNT, [FILL])
    with pytest.raises(ValueError, match="rolled back"):
        import_fills(path, ACCOUNT, [{**FILL, "tid": 8}, {**FILL, "sz": "0.02"}])
    assert import_fills(path, ACCOUNT, [{**FILL, "tid": 8}]) == (1, 0)


def test_no_guessed_fill_id_and_no_paper_ledger_writes(tmp_path):
    path = tmp_path / "bot.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE paper_trades (id INT)")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="dedicated"):
        import_fills(path, ACCOUNT, [FILL])
    assert path.read_bytes() == before
    without_id = {key: value for key, value in FILL.items() if key != "tid"}
    with pytest.raises(ValueError, match="tid"):
        import_fills(tmp_path / "fills.db", ACCOUNT, [without_id])
