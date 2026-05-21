"""``record_signal`` must persist the agent_scores row immediately.

Background
----------
The live-mirror promotion gate (``src/learning/promotion_gate.py``)
checks ``agent_scores.total_signals`` via a DB SELECT in the SAME
trading cycle that executed the paper trade:

  copy_trader.py        -> record_signal()   -> total_signals +=1 (mem)
  copy_trader.py        -> _open_copy_trade() -> paper row inserted
  live_execution.py     -> mirror_executed_trades_to_live(trade)
  live_execution.py     -> _rescale_size_for_live(trade)
  promotion_gate.py     -> is_live_promotable(trade)
  promotion_gate.py     -> SELECT ... FROM agent_scores WHERE source_key=?

Before this fix, ``_save_score`` was only called inside ``record_outcome``,
not ``record_signal``.  So the very first call to ``record_signal``
incremented the in-memory counter but wrote NOTHING to the DB.  The
live-mirror SELECT returned NULL and the gate rejected with
``no_agent_score_row`` -- even for sources that had legitimately just
emitted a signal.  Observed in production logs across multiple cycles
where the same source kept firing paper trades while the live-mirror
counter stayed at zero.

After the fix
-------------
``record_signal`` calls ``_save_score`` after incrementing the counter
and appending to history, so the row is durable by the time the
live-mirror code runs.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import pytest

import src.signals.agent_scoring as ag
from src.signals.agent_scoring import AgentScorer


class _FakeCursor:
    """Captures execute() params so tests can assert what was written."""

    def __init__(self, captures: List[Tuple[str, Tuple]]):
        self._captures = captures

    def execute(self, sql: str, params: Tuple = ()) -> "_FakeCursor":
        self._captures.append((sql.strip(), tuple(params)))
        return self

    def fetchone(self):
        return None  # tests don't care about reads here

    def fetchall(self):
        return []


class _FakeConn:
    def __init__(self, captures: List[Tuple[str, Tuple]]):
        self.captures = captures
        self._closed = False

    def execute(self, sql: str, params: Tuple = ()) -> _FakeCursor:
        return _FakeCursor(self.captures).execute(sql, params)

    def commit(self) -> None: ...

    def __enter__(self) -> "_FakeConn":
        return self

    def __exit__(self, *exc) -> bool:
        self._closed = True
        return False


@pytest.fixture
def captured_db(monkeypatch):
    """Stub ``db.get_connection`` so we capture every INSERT/UPDATE."""
    captures: List[Tuple[str, Tuple]] = []

    @contextmanager
    def fake_get_connection(*a, **k):
        yield _FakeConn(captures)

    monkeypatch.setattr(ag.db, "get_connection", fake_get_connection)
    return captures


def _agent_score_writes(captures):
    """Filter captured executes down to the agent_scores UPSERT only."""
    return [
        (sql, params)
        for sql, params in captures
        if "agent_scores" in sql and "INSERT" in sql
    ]


# ── The headline guarantee ───────────────────────────────────


def test_record_signal_persists_to_agent_scores(captured_db):
    """The first record_signal call must INSERT/UPDATE agent_scores so the
    live-mirror SELECT sees a row with total_signals >= 1."""
    scorer = AgentScorer()  # default cfg

    signal_id = scorer.record_signal(
        "copy_trade:0x932bdd2d5e21475e62d2fea8158fc5974507cb1a",
        {"coin": "DOGE", "side": "long", "confidence": 0.42},
    )

    # An agent_scores write happened exactly once for this signal.
    writes = _agent_score_writes(captured_db)
    assert len(writes) >= 1, (
        "record_signal must write to agent_scores so the live-mirror gate "
        "can see the row in the same cycle (was zero writes -> "
        "no_agent_score_row in prod)"
    )
    sql, params = writes[-1]
    # source_key (param 0) matches the live-mirror lookup key shape.
    assert params[0] == "copy_trade:0x932bdd2d5e21475e62d2fea8158fc5974507cb1a"
    # total_signals (param 1) is 1 after the first emit.
    assert params[1] == 1
    # signal_id is well-formed so record_outcome can match by id later.
    assert signal_id.startswith(
        "copy_trade:0x932bdd2d5e21475e62d2fea8158fc5974507cb1a:1:"
    )


def test_repeated_record_signal_upserts_increasing_total(captured_db):
    """Three record_signal calls for the same source produce three writes
    with monotonically increasing total_signals (1, 2, 3)."""
    scorer = AgentScorer()
    source = "copy_trade:0xabcdef" + "0" * 34

    for i in range(3):
        scorer.record_signal(
            source, {"coin": "ETH", "side": "long", "confidence": 0.50},
        )

    writes = _agent_score_writes(captured_db)
    assert len(writes) == 3, f"expected 3 UPSERTs, got {len(writes)}"
    totals = [params[1] for _, params in writes]
    assert totals == [1, 2, 3]


def test_record_signal_persistence_failure_is_swallowed(monkeypatch):
    """If the DB write fails, record_signal must NOT raise -- the bot
    keeps trading on paper, we just lose the live-mirror promotion
    signal until the next cycle.  Same fail-safe semantics
    ``record_outcome`` already had."""

    @contextmanager
    def boom(*a, **k):
        raise RuntimeError("DB unavailable")
        yield  # pragma: no cover

    monkeypatch.setattr(ag.db, "get_connection", boom)

    scorer = AgentScorer()
    # Should not raise even though the underlying save throws.
    signal_id = scorer.record_signal(
        "copy_trade:0x" + "f" * 40,
        {"coin": "BTC", "side": "short", "confidence": 0.40},
    )
    assert signal_id  # still returns a valid id for trade tracking
    # And the in-memory counter still incremented (paper bookkeeping intact).
    score = scorer.scores["copy_trade:0x" + "f" * 40]
    assert score.total_signals == 1


def test_other_signal_sources_persist_too(captured_db):
    """The fix must apply to every source kind, not just copy_trade -- the
    same gate runs against ``strategy:<type>`` and ``options_flow`` keys
    via Path 3 of ``is_live_promotable``."""
    scorer = AgentScorer()

    cases = [
        ("strategy:mean_reversion", {"coin": "BTC", "side": "long"}),
        ("strategy:momentum_short", {"coin": "ETH", "side": "short"}),
        ("options_flow", {"coin": "SOL", "side": "long"}),
    ]
    for source, data in cases:
        scorer.record_signal(source, {**data, "confidence": 0.50})

    writes = _agent_score_writes(captured_db)
    keys_written = [params[0] for _, params in writes]
    for expected_source, _ in cases:
        assert expected_source in keys_written, (
            f"record_signal for {expected_source!r} did not produce an "
            f"agent_scores write -- only {keys_written} were persisted"
        )
