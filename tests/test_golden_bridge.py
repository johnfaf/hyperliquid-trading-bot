import sqlite3
from contextlib import contextmanager

from src.discovery import golden_bridge


def _connection_ctx(conn):
    @contextmanager
    def _ctx(*, for_read: bool = False):
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    return _ctx


def test_golden_bridge_disconnects_malformed_connected_wallets(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE golden_wallets (
            address TEXT PRIMARY KEY,
            sharpe_ratio REAL,
            penalised_pnl REAL,
            win_rate REAL,
            penalised_max_drawdown_pct REAL,
            trades_per_day REAL,
            best_coin TEXT,
            coins_traded TEXT,
            is_golden INTEGER,
            connected_to_live INTEGER
        )
        """
    )
    valid = "0x" + "a" * 40
    malformed = "0xalpha_momentum_001"
    for address in (malformed, valid):
        conn.execute(
            """
            INSERT INTO golden_wallets (
                address, sharpe_ratio, penalised_pnl, win_rate,
                penalised_max_drawdown_pct, trades_per_day, best_coin,
                coins_traded, is_golden, connected_to_live
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (address, 1.2, 1000.0, 70.0, 5.0, 2.0, "BTC", "[]", 1, 1),
        )

    seen = []

    def _get_user_state(address):
        seen.append(address)
        return {
            "positions": [
                {
                    "coin": "BTC",
                    "size": 0.01,
                    "side": "long",
                    "entry_price": 50000.0,
                    "leverage": 2,
                }
            ]
        }

    monkeypatch.setattr(golden_bridge.db, "get_connection", _connection_ctx(conn))
    monkeypatch.setattr(golden_bridge.hl, "get_all_mids", lambda: {"BTC": 50000.0})
    monkeypatch.setattr(golden_bridge.hl, "get_user_state", _get_user_state)

    signals = golden_bridge.get_golden_copy_signals()

    assert seen == [valid]
    assert len(signals) == 1
    row = conn.execute(
        "SELECT connected_to_live FROM golden_wallets WHERE address = ?",
        (malformed,),
    ).fetchone()
    assert row["connected_to_live"] == 0
