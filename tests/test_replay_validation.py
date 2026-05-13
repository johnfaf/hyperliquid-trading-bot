import importlib.util
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_replay_validation.py"
spec = importlib.util.spec_from_file_location("run_replay_validation", SCRIPT_PATH)
validation = importlib.util.module_from_spec(spec)
sys.modules["run_replay_validation"] = validation
spec.loader.exec_module(validation)


def _make_audit_db(path: Path, timestamps):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE audit_trail (timestamp TEXT NOT NULL)")
    conn.executemany("INSERT INTO audit_trail (timestamp) VALUES (?)", [(ts,) for ts in timestamps])
    conn.commit()
    conn.close()


def test_find_best_audit_window_picks_densest_three_day_window(tmp_path):
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    sparse = [(base + timedelta(days=i)).isoformat() for i in range(3)]
    dense_start = base + timedelta(days=10)
    dense = [
        (dense_start + timedelta(hours=i * 6)).isoformat()
        for i in range(8)
    ]
    db = tmp_path / "bot.db"
    _make_audit_db(db, sparse + dense)

    window = validation.find_best_audit_window(str(db), days=3, min_rows=1)

    assert window.row_count == 8
    assert window.start_iso.startswith("2026-05-11")
    assert window.end_iso.startswith("2026-05-14")


def test_build_replay_command_threads_diff_gate(tmp_path):
    args = type("Args", (), {
        "run_id": "unit",
        "report_out": str(tmp_path / "report.json"),
        "diff_report_out": str(tmp_path / "diff.json"),
        "step": "1h",
        "coins": "BTC,ETH",
        "cache_db": "data/candle_cache.db",
        "live_db": "data/bot.db",
        "match_window": 300.0,
        "min_live_match_rate": 0.7,
        "min_replay_match_rate": 0.65,
        "strategy_snapshot": None,
        "frozen_xgb_model": None,
        "lax_api": True,
        "allow_network": False,
        "halt_on_error": True,
    })()
    window = validation.AuditWindow(
        "2026-05-01T00:00:00+00:00",
        "2026-05-04T00:00:00+00:00",
        42,
    )

    cmd = validation.build_replay_command(args, window)

    assert "--diff-live-db" in cmd
    assert "--diff-min-live-match-rate" in cmd
    assert "0.7" in cmd
    assert "--lax-api" in cmd
    assert "--halt-on-error" in cmd
