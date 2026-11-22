from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import sqlite3
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import config
from src.core.cycles import trading_cycle as cycle
from src.data import database as db, decision_coverage
from src.signals.decision_firewall import DecisionFirewall
from src.signals.signal_schema import TradeSignal, SignalSide, SignalSource
from src.signals.agent_scoring import AgentScorer
from src.trading.paper_reporting import metrics, valuation, weekly_report
from scripts.publish_dashboard_status import absorption_status, fixed_weekly_report


def test_paper_admission_blocks_opposite_with_idempotent_replay():
    db.init_db()
    trade = db.open_paper_trade(None, "BTC", "long", 100, 1, idempotency_key="first")
    assert db.open_paper_trade(None, "BTC", "long", 100, 1, idempotency_key="first") == trade
    with pytest.raises(ValueError, match="Opposing"):
        db.open_paper_trade(None, " btc ", "SHORT", 100, 1)
    assert len(db.get_open_paper_trades()) == 1


def test_admission_is_serialized_across_processes(tmp_path):
    path = tmp_path / "race.db"
    env = {**os.environ, "HL_BOT_DB": str(path), "DB_BACKEND": "sqlite"}
    subprocess.run([sys.executable, "-c", "from src.data import database as db; db.init_db()"], env=env, check=True, timeout=20)
    code = """
import sys,time
from pathlib import Path
from src.data import database as db
gate = Path(sys.argv[2])
deadline = time.monotonic()+10
while not gate.exists() and time.monotonic()<deadline:
    time.sleep(.01)
try:
    db.open_paper_trade(None,'BTC',sys.argv[1],100,1)
    print('opened')
except ValueError:
    print('blocked')
"""
    gate = tmp_path / "go"
    processes = [subprocess.Popen([sys.executable, "-c", code, side, str(gate)], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for side in ("long", "short")]
    try:
        gate.touch()
        results = [proc.communicate(timeout=20) for proc in processes]
        assert all(proc.returncode == 0 for proc in processes), results
        assert sorted(out.strip() for out, _ in results) == ["blocked", "opened"]
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()


@pytest.mark.parametrize("policy", [None, {"size_multiplier": 0}, {"size_multiplier": float("nan")},
                                    {"size_multiplier": -1}, {"min_confidence": 2}])
def test_invalid_source_policy_rejects(policy):
    signal = TradeSignal(coin="BTC", side=SignalSide.LONG, source=SignalSource.STRATEGY,
                         strategy_type="scalping", confidence=.8, entry_price=100, position_pct=.03, reason="test")
    fw = DecisionFirewall({"agent_scorer": SimpleNamespace(get_source_policy=lambda _: policy)})
    assert fw._apply_source_policy(signal, "strategy:scalping")[0] is False


def test_source_lookup_failure_rejects():
    def broken(_):
        raise RuntimeError("unavailable")
    fw = DecisionFirewall({"agent_scorer": SimpleNamespace(get_source_policy=broken)})
    assert not fw._apply_source_policy(SimpleNamespace(), "arena_champion:scalping")[0]


def test_untracked_source_is_warmup_not_unlimited():
    scorer = AgentScorer()
    policy = scorer.get_source_policy("arena_champion:new_strategy")
    assert policy["status"] == "warmup"
    assert 0 < policy["size_multiplier"] < 1
    assert policy["max_signals_per_day"] > 0


@pytest.mark.parametrize("mode", ["missing", "rejected", "approved", "sizing", "live_no_balance", "invalid_quote"])
def test_arena_firewall_and_final_size(monkeypatch, mode):
    signal = {"coin": "BTC", "side": "long", "confidence": .8, "price": 100., "agent_id": "a",
              "agent_name": "alpha", "strategy_type": "scalping", "agent_fitness": 1., "agent_elo": 1100}
    if mode == "invalid_quote":
        signal["price"] = float("nan")
    arena = SimpleNamespace(run_cycle=lambda **k: None,
                            get_stats=lambda: {"active_agents": 1, "champions": 1, "total_arena_pnl": 0},
                            get_champion_signals=lambda **k: [signal])
    opened, calls, resolutions = [], [], []

    def validate(candidate, **kwargs):
        calls.append((candidate.source, kwargs))
        candidate.position_pct *= .25
        candidate.leverage = 1
        return mode != "rejected", "source paused"

    container = SimpleNamespace(arena=arena, live_trader=SimpleNamespace(get_free_margin=lambda: None),
                                firewall=None if mode == "missing" else SimpleNamespace(
                                    validate=validate, _source_key=DecisionFirewall._source_key))
    if mode == "sizing":
        container.kelly_sizer = object()
        monkeypatch.setattr(cycle, "_get_dynamic_sizing", lambda *a, **k: SimpleNamespace(position_pct=.02))
    monkeypatch.setattr(cycle, "_fetch_arena_candle_universe", lambda: {"BTC": [{}] * 40})
    monkeypatch.setattr(cycle, "is_live_trading_active", lambda _: mode == "live_no_balance")
    monkeypatch.setattr(cycle, "get_execution_open_positions", lambda _: [])
    monkeypatch.setattr(db, "get_paper_account", lambda: {"balance": 1000.})
    monkeypatch.setattr(db, "open_paper_trade", lambda **k: opened.append(k) or 1)
    monkeypatch.setattr(cycle.decision_journal, "record_paper_opened", lambda *a, **k: None)
    monkeypatch.setattr(decision_coverage, "expect", lambda *a: ["ref"])
    monkeypatch.setattr(decision_coverage, "bind_signal", lambda *a: None)
    monkeypatch.setattr(decision_coverage, "resolve", lambda *a, **k: resolutions.append((a, k)))
    cycle._run_alpha_arena(container, {})
    if mode in {"missing", "rejected", "live_no_balance", "invalid_quote"}:
        assert not opened and resolutions
    else:
        trade, = opened
        assert calls[0][0] == SignalSource.ARENA_CHAMPION
        assert calls[0][1]["account_balance"] == 1000
        assert trade["leverage"] == 1
        assert trade["size"] == pytest.approx(.05 if mode == "sizing" else .1)
        assert trade["metadata"]["entry_decision"]["firewall"] == "approved"
        assert trade["metadata"]["source_key"] == "arena_champion:scalping"


def test_metrics_expose_small_sample_and_winner_dependence():
    trades = [{"id": n, "status": "closed", "closed_at": f"2026-11-0{n+1}T00:00:00+00:00", "pnl": pnl}
              for n, pnl in enumerate([60., -10., -20.])]
    result = metrics(trades)
    assert result["profit_factor"] == 2
    assert result["realized_drawdown"] == 30
    assert result["net_without_largest"] == -30
    assert result["evidence"] == "Insufficient sample" and not result["live_approved"]


def test_marked_equity_is_unmeasured_when_quote_missing_or_stale(tmp_path):
    path = tmp_path / "book.db"
    now = datetime(2026, 11, 22, tzinfo=timezone.utc)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.executescript("""CREATE TABLE paper_account(id,balance); INSERT INTO paper_account VALUES(1,1000);
            CREATE TABLE paper_trades(coin,side,entry_price,size,leverage,status);
            INSERT INTO paper_trades VALUES('BTC','long',100,2,2,'open'); CREATE TABLE bot_state(key,value);""")
        conn.execute("INSERT INTO bot_state VALUES('paper_mark_prices',?)", (json.dumps({"observed_at": now.isoformat(), "mids": {"BTC": 110}}),))
    result = valuation(path, now)
    assert result["unrealized"] == 40 and result["equity"] == 1040
    assert valuation(path, now + timedelta(minutes=16))["equity"] is None
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("UPDATE bot_state SET value=?", (json.dumps({"observed_at": now.isoformat(), "mids": {}}),))
    assert valuation(path, now)["unrealized"] is None


def test_weekly_report_is_bounded_excludes_synthetic_and_archives_once(tmp_path):
    path = tmp_path / "book.db"
    now = datetime(2026, 11, 22, tzinfo=timezone.utc)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE paper_trades(id,status,closed_at,pnl,metadata)")
        conn.executemany("INSERT INTO paper_trades VALUES(?,'closed',?,?,?)", [
            (1, "2026-11-09T00:00:00+00:00", 10, '{"source":"arena_champion","total_fees_paid":1}'),
            (2, "2026-11-15T23:59:59+00:00", -2, '{}'),
            (3, "2026-11-16T00:00:00+00:00", 500, '{}'),
            (4, "2026-11-12T00:00:00+00:00", -100, '{"close_reason":"live_reconciled_closed"}'),
        ])
    report = weekly_report(path, now)
    assert report["summary"]["net"] == 8 and report["excluded"] == 1
    assert report["start"][:10] == "2026-11-09" and report["end_exclusive"][:10] == "2026-11-16"
    directory = tmp_path / "reports"
    archived = fixed_weekly_report(path, directory, now)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("UPDATE paper_trades SET pnl=99 WHERE id=1")
    assert fixed_weekly_report(path, directory, now) == archived


def test_absorption_staleness_cannot_grant_live_approval(tmp_path):
    path = tmp_path / "report.json"
    now = datetime.now(timezone.utc)
    path.write_text(json.dumps({"version": 1, "observed_at": (now - timedelta(hours=27)).isoformat(),
                                "live_approved": True, "secret": "not relayed"}))
    report = absorption_status(path, now)
    assert report["stale"] and not report["live_approved"] and "secret" not in report


def test_research_page_is_authenticated_and_renders_empty_evidence(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from src.ui.v2.app import create_app

    path = tmp_path / "book.db"
    monkeypatch.setattr(config, "DB_PATH", str(path))
    monkeypatch.setenv("DASHBOARD_AUTH_TOKEN", "test-research-token")
    monkeypatch.setenv("DASHBOARD_V2_PRIMARY", "true")
    host = {"observed_at": datetime.now(timezone.utc).isoformat(), "absorption": {
        "rule": "flow_absorption_v1", "state": "Insufficient forward evidence", "stale": False,
        "frames": 0, "ready": 0, "expected": 0, "healthy_at_scoring": False, "side_supported": False,
        "observed_at": datetime.now(timezone.utc).isoformat(), "last_frame": "none", "horizons": [],
    }}
    path.with_name("dashboard_ops.json").write_text(json.dumps(host))
    client = TestClient(create_app())
    assert client.get("/research", follow_redirects=False).status_code == 303
    client.post("/api/auth/login", data={"token": "test-research-token"})
    response = client.get("/research")
    assert response.status_code == 200
    assert "Insufficient forward evidence" in response.text and "No live approval" in response.text
    assert "test-research-token" not in response.text
