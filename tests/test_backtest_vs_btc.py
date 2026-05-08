"""Smoke test for scripts/backtest_vs_btc.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    here = Path(__file__).resolve().parent.parent
    path = here / "scripts" / "backtest_vs_btc.py"
    spec = importlib.util.spec_from_file_location("backtest_vs_btc", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_compute_sharpe_handles_short_series():
    mod = _load_module()
    assert mod.compute_sharpe([]) == 0.0
    assert mod.compute_sharpe([1.0]) == 0.0
    # All zero stdev -> 0
    assert mod.compute_sharpe([1.0, 1.0, 1.0]) == 0.0


def test_max_drawdown_is_zero_for_monotonic_curve():
    mod = _load_module()
    assert mod.max_drawdown([0.0, 1.0, 2.0, 3.0]) == 0.0


def test_max_drawdown_reports_peak_to_trough():
    mod = _load_module()
    # Up to 5, down to 1 -> drawdown -4.
    assert mod.max_drawdown([0.0, 5.0, 3.0, 1.0, 2.0]) == -4.0


def test_main_with_minimal_csv(tmp_path, capsys):
    mod = _load_module()
    csv_path = tmp_path / "trades.csv"
    csv_path.write_text(
        "time,coin,dir,px,sz,ntl,fee,closedPnl\n"
        "05/04/2026 - 14:30:00,BTC,Open Long,67000,0.001,67.0,0.03,-0.03\n"
        "06/05/2026 - 14:30:00,BTC,Close Long,81000,0.001,81.0,0.03,13.97\n"
    )
    rc = mod.main(["backtest_vs_btc.py", str(csv_path), "1000"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "BOT vs BTC" in captured
    assert "BTC BUY-AND-HOLD" in captured
    # Either bot or buy-and-hold may "win" depending on PnL signs; both
    # branches print a verdict line that mentions the comparison.
    assert "buy-and-hold" in captured.lower()
