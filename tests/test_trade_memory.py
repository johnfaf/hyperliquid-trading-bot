import contextlib
import sqlite3

from src.trading import trade_memory as trade_memory_module


@contextlib.contextmanager
def _sqlite_ctx(conn):
    yield conn
    conn.commit()


def _memory(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    monkeypatch.setattr(trade_memory_module.db, "get_connection", lambda for_read=False: _sqlite_ctx(conn))
    monkeypatch.setattr(trade_memory_module.db, "get_backend_name", lambda: "sqlite")
    return trade_memory_module.TradeMemory()


def test_find_similar_returns_caution_for_zero_feature_vector(monkeypatch):
    memory = _memory(monkeypatch)

    result = memory.find_similar(
        {
            "volatility": 0.0,
            "volume_ratio": 0.0,
            "rsi": 0.0,
        }
    )

    assert result.recommendation == "caution"
    assert result.reason == "No feature data available"


def test_find_similar_returns_caution_for_sparse_feature_overlap(monkeypatch):
    memory = _memory(monkeypatch)
    memory.record_trade(
        trade_id="t-sparse",
        coin="BTC",
        side="long",
        strategy_type="momentum",
        entry_price=100.0,
        exit_price=101.0,
        pnl=1.0,
        return_pct=0.01,
        opened_at="2026-04-21T10:00:00+00:00",
        closed_at="2026-04-21T11:00:00+00:00",
        features={"momentum_score": 1.0},
    )

    result = memory.find_similar(
        {
            "momentum_score": 1.0,
            "overall_score": 0.5,
            "volatility": 0.2,
        },
        coin="BTC",
        side="long",
        min_similarity=0.1,
    )

    assert result.recommendation == "caution"
    assert result.reason == "Insufficient overlapping feature data for similarity check"
