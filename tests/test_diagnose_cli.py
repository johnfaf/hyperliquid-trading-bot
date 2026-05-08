"""Smoke tests for the --diagnose CLI command."""
from __future__ import annotations

import json
from contextlib import contextmanager

import cli


def test_diagnose_cli_runs_with_empty_db(monkeypatch, capsys):
    """The diagnose command must NEVER crash, even when every table is empty
    or when the calibration tracker / shadow tracker fail to initialise.
    A diagnostic that throws is useless."""
    parser = cli.build_parser()
    args = parser.parse_args(["--diagnose"])

    class _EmptyConn:
        def execute(self, *_a, **_k):
            class _Cur:
                def fetchall(self_inner):
                    return []
            return _Cur()

    @contextmanager
    def _ctx(*, for_read=False):
        yield _EmptyConn()

    import src.data.database as db_mod
    monkeypatch.setattr(db_mod, "get_connection", _ctx)

    rc = cli.run_diagnose_cli(args)
    captured = capsys.readouterr().out
    assert rc == 0
    assert "DIAGNOSE @" in captured


def test_diagnose_json_output_is_valid_json(monkeypatch, capsys):
    parser = cli.build_parser()
    args = parser.parse_args(["--diagnose", "--diagnose-json"])

    class _EmptyConn:
        def execute(self, *_a, **_k):
            class _Cur:
                def fetchall(self_inner):
                    return []
            return _Cur()

    @contextmanager
    def _ctx(*, for_read=False):
        yield _EmptyConn()

    import src.data.database as db_mod
    monkeypatch.setattr(db_mod, "get_connection", _ctx)

    rc = cli.run_diagnose_cli(args)
    captured = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(captured)
    assert "generated_at" in payload
    # Every section key should be a list (or *_error string) — never a partial object.
    for key in ("paper_sources", "live_sources", "shadow_sources",
                "calibration_top", "active_strategies"):
        if key in payload:
            assert isinstance(payload[key], list)


def test_format_source_breakdown_sorts_by_trade_count():
    by_source = {
        "strategy:foo": {"trades": 3, "wins": 1, "pnl": -1.5},
        "copy_trade:0xabc": {"trades": 10, "wins": 6, "pnl": 4.0},
        "options_flow": {"trades": 1, "wins": 0, "pnl": -0.5},
    }
    out = cli._format_source_breakdown(by_source)
    assert [o["source"] for o in out] == [
        "copy_trade:0xabc",  # 10
        "strategy:foo",      # 3
        "options_flow",      # 1
    ]
    assert out[0]["win_rate"] == 0.6
    assert out[0]["losses"] == 4
