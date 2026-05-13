from __future__ import annotations

import json

from src.ui import replay_dashboard


def test_params_from_mapping_bounds_and_normalizes_inputs():
    params = replay_dashboard.params_from_mapping({
        "coins": "btc, eth,btc,, sol ",
        "window_days": "999",
        "min_rows": "-5",
        "min_live_match_rate": "1.5",
        "min_replay_match_rate": "-1",
        "allow_network": "yes",
        "lax_api": "false",
    })

    assert params.coins == "BTC,ETH,SOL"
    assert params.window_days == 30
    assert params.min_rows == 1
    assert params.min_live_match_rate == 1.0
    assert params.min_replay_match_rate == 0.0
    assert params.allow_network is True
    assert params.lax_api is False


def test_extract_summary_reads_replay_and_diff_reports(tmp_path):
    report = tmp_path / "report.json"
    diff = tmp_path / "diff.json"
    report.write_text(
        json.dumps({
            "config": {
                "run_id": "unit",
                "start": "2026-05-01T00:00:00+00:00",
                "end": "2026-05-04T00:00:00+00:00",
                "coins": ["BTC", "ETH"],
                "step": "1h",
            },
            "outputs": {"audit_trail_rows": 9},
            "execution": {"completed_ticks": 72, "failed_ticks": 0},
            "replay_db_path": "data/replay_unit.db",
        }),
        encoding="utf-8",
    )
    diff.write_text(
        json.dumps({
            "totals": {
                "live": 10,
                "replay": 9,
                "matched": 7,
                "live_only": 3,
                "replay_only": 2,
            },
            "diagnostics": {
                "status": "trusted",
                "trustworthy": True,
                "live_match_rate": 0.7,
                "replay_match_rate": 0.7778,
                "guidance": "ok",
            },
            "reject_reasons": {
                "live_only": {"polymarket_stubbed": 2},
                "replay_only": {"state_bleed": 1},
            },
        }),
        encoding="utf-8",
    )

    summary = replay_dashboard._extract_summary(
        report_path=report,
        diff_path=diff,
        returncode=0,
        started_at=10.0,
        completed_at=15.0,
        stdout="done",
        stderr="",
    )

    assert summary["status"] == "trusted"
    assert summary["trustworthy"] is True
    assert summary["duration_s"] == 5.0
    assert summary["totals"]["matched"] == 7
    assert summary["top_live_only_reasons"] == [("polymarket_stubbed", 2)]
    assert summary["window"]["start_iso"].startswith("2026-05-01")


def test_build_command_threads_dashboard_report_paths(tmp_path):
    params = replay_dashboard.params_from_mapping({
        "coins": "BTC,ETH",
        "window_days": 3,
        "step": "1h",
        "halt_on_error": True,
    })
    report = tmp_path / "report.json"
    diff = tmp_path / "diff.json"

    cmd = replay_dashboard._build_command(params, report, diff)

    assert any(part.endswith("run_replay_validation.py") for part in cmd)
    assert "--report-out" in cmd
    assert str(report) in cmd
    assert "--diff-report-out" in cmd
    assert str(diff) in cmd
    assert "--lax-api" in cmd
    assert "--halt-on-error" in cmd


def test_list_recent_replay_results_ignores_diff_files(tmp_path, monkeypatch):
    monkeypatch.setattr(replay_dashboard, "REPORTS_DIR", tmp_path)
    report = tmp_path / "dashboard_replay_validation_123.json"
    diff = tmp_path / "dashboard_replay_validation_diff_123.json"
    report.write_text(
        json.dumps({"config": {"run_id": "unit", "coins": ["BTC"]}}),
        encoding="utf-8",
    )
    diff.write_text(
        json.dumps({
            "totals": {"live": 1, "replay": 1, "matched": 1},
            "diagnostics": {"status": "trusted", "trustworthy": True},
        }),
        encoding="utf-8",
    )

    results = replay_dashboard.list_recent_replay_results()

    assert len(results) == 1
    assert results[0]["run_id"] == "unit"
    assert results[0]["totals"]["matched"] == 1
