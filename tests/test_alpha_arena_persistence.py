import contextlib
import sqlite3

from src.data import database as db
from src.signals.alpha_arena import AlphaArena, AgentStatus


@contextlib.contextmanager
def _sqlite_ctx(conn):
    yield conn
    conn.commit()


def _memory_db(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    monkeypatch.setattr(db, "get_connection", lambda for_read=False: _sqlite_ctx(conn))
    return conn


def test_arena_records_trade_events_and_persists_agent_state(monkeypatch):
    conn = _memory_db(monkeypatch)
    arena = AlphaArena()

    arena.record_trade_for_strategy(
        "momentum_long",
        12.5,
        0.0125,
        metadata={"trade_id": "paper-1", "coin": "BTC"},
    )

    agent = arena.agents["seed_momentum_long"]
    assert agent.total_trades == 1
    assert agent.total_pnl == 12.5

    row = conn.execute(
        "SELECT total_trades, total_pnl FROM arena_agents WHERE agent_id = ?",
        ("seed_momentum_long",),
    ).fetchone()
    assert row["total_trades"] == 1
    assert row["total_pnl"] == 12.5

    event = conn.execute(
        "SELECT trade_id, agent_id, strategy_type, pnl FROM arena_trade_events"
    ).fetchone()
    assert event["trade_id"] == "paper-1"
    assert event["agent_id"] == "seed_momentum_long"
    assert event["strategy_type"] == "momentum_long"
    assert event["pnl"] == 12.5


def test_arena_refreshes_backtests_on_cold_start_and_persists_round(monkeypatch):
    conn = _memory_db(monkeypatch)
    arena = AlphaArena()
    calls = []

    def _fake_backtest(agent, candle_universe):
        calls.append(agent.agent_id)
        agent.backtest_trades = 4
        agent.backtest_pnl = 1.0
        return {"total_trades": 4, "total_pnl": 1.0}

    monkeypatch.setattr(arena.backtester, "backtest_agent", _fake_backtest)
    bars = [
        {"open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000}
        for _ in range(60)
    ]

    arena.run_cycle({"BTC": bars})
    assert len(calls) == len(arena.agents)

    for agent in arena.agents.values():
        agent.total_trades = 3
        agent.winning_trades = 2
        agent.win_rate = 2 / 3
        agent.sharpe_ratio = 1.0
        agent.status = AgentStatus.ACTIVE
    arena.cycle_count = arena.TOURNAMENT_INTERVAL - 1
    arena.run_cycle(None)

    assert conn.execute("SELECT COUNT(*) FROM arena_rounds").fetchone()[0] == 1
