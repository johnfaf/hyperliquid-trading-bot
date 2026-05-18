"""Per-source realized-PnL attribution (#8).

The evidence bar gates *entry*; this measures *outcome* so "which source
actually makes money?" is provable, not inferred. Pure aggregator is
unit-tested in isolation; the reader is best-effort.
"""
from __future__ import annotations

import src.data.decision_journal as dj
from src.data.decision_journal import _summarize_source_pnl_rows


def test_groups_by_source_key_with_pnl_and_winrate():
    rows = [
        {"source": "copy_trade", "source_key": "copy_trade:0xabc",
         "action_taken": True, "label_win": 1, "outcome_pnl": 12.0},
        {"source": "copy_trade", "source_key": "copy_trade:0xabc",
         "action_taken": True, "label_win": 0, "outcome_pnl": -4.0},
        {"source": "options_flow", "source_key": "options_flow",
         "action_taken": False, "label_win": None, "outcome_pnl": 0.0},
        {"source": "options_flow", "source_key": "options_flow",
         "action_taken": True, "label_win": 1, "outcome_pnl": 7.5},
    ]
    s = _summarize_source_pnl_rows(rows, window_days=7)
    assert s["available"] is True
    assert s["source_count"] == 2
    by = {g["source_key"]: g for g in s["sources"]}

    abc = by["copy_trade:0xabc"]
    assert abc["trades"] == 2 and abc["acted"] == 2
    assert abc["wins"] == 1 and abc["losses"] == 1
    assert abc["realized_pnl"] == 8.0
    assert abc["win_rate"] == 0.5

    of = by["options_flow"]
    assert of["trades"] == 2 and of["acted"] == 1
    assert of["realized_pnl"] == 7.5
    assert of["win_rate"] == 1.0  # only one labelled outcome, a win

    # sorted by realized pnl desc; totals
    assert s["sources"][0]["source_key"] == "copy_trade:0xabc"
    assert s["total_realized_pnl"] == 15.5
    assert s["net_positive_sources"] == 2
    assert s["net_negative_sources"] == 0


def test_unknown_source_bucket_and_no_labels():
    rows = [{"source": None, "source_key": None, "action_taken": False,
             "label_win": None, "outcome_pnl": None}]
    s = _summarize_source_pnl_rows(rows, window_days=1)
    g = s["sources"][0]
    assert g["source_key"] == "unknown"
    assert g["trades"] == 1 and g["win_rate"] is None
    assert g["realized_pnl"] == 0.0


def test_summarize_source_pnl_is_best_effort(monkeypatch):
    monkeypatch.setattr(dj, "_enabled", lambda: True)

    class _Boom:
        def __enter__(self):
            raise RuntimeError("no decision_outcomes table")

        def __exit__(self, *a):
            return False

    import src.data.database as db
    monkeypatch.setattr(db, "get_connection", lambda **k: _Boom())
    out = dj.summarize_source_pnl(days=7)
    assert out["available"] is False
