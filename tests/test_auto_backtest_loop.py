from __future__ import annotations

import json
from dataclasses import replace

from src.learning.auto_backtest_loop import (
    AutoBacktestConfig,
    AutoBacktestLoop,
    get_auto_backtest_status,
    latest_auto_backtest_reports,
)


def test_auto_backtest_config_from_env_bounds_values(monkeypatch):
    monkeypatch.setenv("AUTO_BACKTEST_LOOP_ENABLED", "true")
    monkeypatch.setenv("AUTO_BACKTEST_INTERVAL_SECONDS", "1")
    monkeypatch.setenv("AUTO_BACKTEST_DATASET_LIMIT", "9999999")
    monkeypatch.setenv("AUTO_BACKTEST_COINS", "btc,eth,btc,,sol")
    monkeypatch.setenv("AUTO_BACKTEST_REPLAY_MIN_LIVE_MATCH_RATE", "2")

    cfg = AutoBacktestConfig.from_env()

    assert cfg.enabled is True
    assert cfg.interval_seconds == 900
    assert cfg.dataset_limit == 100_000
    assert cfg.coins == "BTC,ETH,SOL"
    assert cfg.replay_min_live_match_rate == 1.0


def test_auto_backtest_cycle_runs_enabled_steps_without_live_mutation(tmp_path, monkeypatch):
    cfg = AutoBacktestConfig(
        reports_dir=str(tmp_path),
        run_offline_learning=False,
        run_replay_validation=True,
        run_candle_research=True,
        live_db="missing.db",
    )
    loop = AutoBacktestLoop(cfg)
    calls = []

    def fake_run_command(*, step_name, cmd, report_path):
        calls.append((step_name, cmd, report_path))
        report_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
        return {"status": "completed", "returncode": 0, "report_path": str(report_path)}

    monkeypatch.setattr(loop, "_run_command", fake_run_command)

    result = loop.run_cycle()

    assert result.status == "completed_with_skips"
    assert result.steps["replay_validation"]["status"] == "skipped"
    assert result.steps["candle_research"]["status"] == "completed"
    assert result.config["safety"] == "offline_only_no_live_config_mutation"
    assert calls and calls[0][0] == "candle_research"
    reports = latest_auto_backtest_reports(str(tmp_path))
    assert len(reports) == 1
    assert reports[0]["cycle_id"] == result.cycle_id


def test_auto_backtest_status_reads_latest_reports(tmp_path, monkeypatch):
    cfg = replace(AutoBacktestConfig.from_env(), reports_dir=str(tmp_path), enabled=True)
    report = tmp_path / "auto_bt_20260513_120000.json"
    subreport = tmp_path / "auto_bt_20260513_120000_candle_research.json"
    report.write_text(
        json.dumps({
            "cycle_id": "auto_bt_20260513_120000",
            "status": "completed",
            "steps": {},
            "next_actions": [],
        }),
        encoding="utf-8",
    )
    subreport.write_text(json.dumps({"not": "a cycle report"}), encoding="utf-8")

    status = get_auto_backtest_status(cfg)

    assert status["enabled"] is True
    assert len(status["recent_results"]) == 1
    assert status["recent_results"][0]["cycle_id"] == "auto_bt_20260513_120000"
