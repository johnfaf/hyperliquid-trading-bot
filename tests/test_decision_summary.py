"""Why-not-trading rollup over decision_snapshots.

This is the surface that turns "bullish shown but no orders / only
shorts" from hours of Railway log archaeology into one line.
"""
from __future__ import annotations

import src.data.decision_journal as dj
from src.data.decision_journal import _summarize_decision_rows, _summary_line


def test_counts_executed_rejected_other_and_top_reasons():
    rows = [
        {"final_status": "paper_opened", "paper_trade_id": 12,
         "rejection_reason": None, "source": "copy_trade", "coin": "BTC"},
        {"final_status": "candidate", "paper_trade_id": None,
         "rejection_reason": "rejected_exposure", "source": "options_flow", "coin": "BTC"},
        {"final_status": "rejected", "paper_trade_id": None,
         "rejection_reason": "rejected_exposure", "source": "options_flow", "coin": "ETH"},
        {"final_status": "rejected", "paper_trade_id": None,
         "rejection_reason": "rejected_source_policy", "source": "strategy", "coin": "SOL"},
        {"final_status": "candidate", "paper_trade_id": None,
         "rejection_reason": "", "firewall_decision": "approved",
         "source": "x", "coin": "X"},  # neither executed nor a real reject
    ]
    s = _summarize_decision_rows(rows, window_hours=6)
    assert s["available"] is True
    assert s["total"] == 5
    assert s["executed"] == 1
    assert s["rejected"] == 3
    assert s["other"] == 1
    tr = dict(s["top_reasons"])
    assert tr["rejected_exposure"] == 2
    assert tr["rejected_source_policy"] == 1
    assert ("options_flow", 2) in s["by_source"]


def test_message_reasons_collapse_so_they_aggregate():
    rows = [
        {"final_status": "rejected", "paper_trade_id": None,
         "rejection_reason": "Current market read blocks long: best read is short",
         "source": "of", "coin": "BTC"},
        {"final_status": "rejected", "paper_trade_id": None,
         "rejection_reason": "Current market read blocks long: different detail",
         "source": "of", "coin": "ETH"},
    ]
    s = _summarize_decision_rows(rows, window_hours=6)
    assert dict(s["top_reasons"]).get("Current market read blocks long") == 2


def test_summary_line_is_compact_and_safe():
    s = _summarize_decision_rows(
        [{"final_status": "rejected", "rejection_reason": "rejected_exposure",
          "source": "of", "coin": "BTC"}],
        window_hours=6,
    )
    line = _summary_line(s)
    assert "Decision summary (6h)" in line
    assert "rejected_exposure x1" in line
    assert "unavailable" in _summary_line({"available": False})


def test_summarize_recent_decisions_is_best_effort(monkeypatch):
    """A missing/broken decision_snapshots table must yield
    available=False, never raise into the trading cycle."""
    monkeypatch.setattr(dj, "_enabled", lambda: True)

    class _Boom:
        def __enter__(self):
            raise RuntimeError("no decision_snapshots table")

        def __exit__(self, *a):
            return False

    import src.data.database as db
    monkeypatch.setattr(db, "get_connection", lambda **k: _Boom())

    out = dj.summarize_recent_decisions(hours=6)
    assert out["available"] is False
