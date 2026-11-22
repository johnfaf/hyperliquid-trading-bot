from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import os
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import config
from src.core.cycles import trading_cycle as cycle
from src.data import database as db
from src.learning import promotion_gate
from src.signals.decision_firewall import DecisionFirewall
from src.signals.signal_schema import TradeSignal, SignalSide, SignalSource
from src.trading.equity_risk import EquityRiskLedger, PAPER_ACCOUNT, paper_ledger
from src.trading.hyperliquid_order_recovery import recover_submissions
from src.trading.order_journal import OrderJournal

NOW = datetime(2026, 11, 22, 10, tzinfo=timezone.utc)


@pytest.mark.parametrize("side,rate,blocked", [("long", .004, True), ("short", .004, False),
                                            ("short", -.002, True), ("long", -.002, False),
                                            ("long", 0., False), ("short", 0., False)])
def test_funding_blocks_the_paying_side(monkeypatch, side, rate, blocked):
    monkeypatch.setattr(config, "DATA_READINESS_GATE_ENABLED", False)
    monkeypatch.setattr(config, "EV_GATE_ENABLED", False)
    mock = MagicMock()
    mock.get_open_paper_trades.return_value = []
    mock.get_paper_trade_history.return_value = []
    mock.get_paper_account.return_value = {"balance": 1000.}
    monkeypatch.setattr("src.signals.decision_firewall.db", mock)
    firewall = DecisionFirewall({"event_risk_enabled": False, "enable_predictive_derisk": False,
                                 "short_hardening_enabled": False, "long_hardening_enabled": False})
    monkeypatch.setattr(firewall, "_get_funding_rate", lambda _: rate)
    signal = TradeSignal(coin="BTC", side=SignalSide(side), confidence=.8, source=SignalSource.STRATEGY,
                         strategy_type="scalping", reason="test", entry_price=100., leverage=1, position_pct=.01)
    passed, reason = firewall.validate(signal)
    assert passed is not blocked, reason
    if blocked:
        assert firewall.get_stats()["rejected_funding"] == 1


def test_funding_context_is_normalized_to_eight_hours(monkeypatch):
    manager = SimpleNamespace(post=lambda **k: [{"universe": [{"name": "BTC"}]}, [{"funding": ".0005"}]])
    monkeypatch.setattr("src.core.api_manager.get_manager", lambda: manager)
    assert DecisionFirewall()._get_funding_rate("BTC") == .004


@pytest.mark.parametrize("namespace", ["arena_champion", "options_flow", "strategy"])
def test_promotion_never_borrows_another_source_cohort(monkeypatch, namespace):
    seen = []
    def score(key, **kwargs):
        seen.append(key)
        return key == "strategy:scalping", "test_cohort"
    monkeypatch.setattr(promotion_gate, "_agent_score_promotion_ok", score)
    monkeypatch.setattr(promotion_gate, "_maybe_bootstrap_tier", lambda *a, **k: (False, "insufficient_evidence"))
    passed, _ = promotion_gate.is_live_promotable({"metadata": {
        "source": namespace, "strategy_type": "scalping", "source_key": f"{namespace}:scalping",
    }})
    assert seen == [f"{namespace}:scalping"]
    assert passed is (namespace == "strategy")


def test_promotion_rejects_conflicting_identity_before_database_lookup(monkeypatch):
    monkeypatch.setattr(db, "get_strategy", lambda _: pytest.fail("wrong strategy must not be queried"))
    assert promotion_gate.is_live_promotable({"strategy_id": 5, "metadata": {
        "source": "arena_champion", "strategy_type": "scalping", "source_key": "strategy:scalping",
    }}) == (False, "source_key_mismatch")


@pytest.mark.parametrize("route", ["liquidation", "options_flow", "arena"])
@pytest.mark.parametrize("policy", ["missing", "reject", "derisk"])
def test_every_direct_route_has_final_admission_and_stop_budget(monkeypatch, route, policy):
    opened, validations = [], []
    signal = {"coin": "BTC", "side": "long", "price": 100., "confidence": .8, "leverage": 2,
              "stop_loss": 97., "take_profit": 110., "features": {"setup_type": "reversal"},
              "agent_id": "a", "agent_name": "alpha", "agent_fitness": 1., "agent_elo": 1100, "strategy_type": "scalping"}
    def validate(candidate, **kwargs):
        validations.append(kwargs)
        candidate.position_pct *= .25
        candidate.size *= .25
        candidate.leverage = 1
        return policy != "reject", "test_rejection"
    arena = SimpleNamespace(run_cycle=lambda **k: None, get_champion_signals=lambda **k: [signal],
                            get_stats=lambda: {"active_agents": 1, "champions": 1, "total_arena_pnl": 0})
    container = SimpleNamespace(live_trader=None, firewall=None if policy == "missing" else SimpleNamespace(validate=validate),
                                arena=arena, agent_scorer=None, kelly_sizer=None, rl_sizer=None, trade_memory=None, llm_filter=None,
                                options_scanner=SimpleNamespace(top_convictions=[{
                                    "ticker": "BTC", "direction": "BULLISH", "conviction_pct": 80, "net_flow": 50000., "total_prints": 2}]))
    monkeypatch.setattr(cycle, "is_live_trading_active", lambda _: False)
    monkeypatch.setattr(cycle, "get_execution_open_positions", lambda _: [])
    monkeypatch.setattr(cycle, "_fetch_arena_candle_universe", lambda: {"BTC": [{}]*40})
    monkeypatch.setattr(db, "get_paper_account", lambda: {"balance": 1000.})
    monkeypatch.setattr(db, "open_paper_trade", lambda **k: opened.append(k) or 1)
    monkeypatch.setattr(cycle.decision_journal, "record_paper_opened", lambda *a, **k: None)
    monkeypatch.setattr("src.data.hyperliquid_client.get_all_mids", lambda: {"BTC": 100.})
    monkeypatch.setattr("src.notifications.telegram_bot.is_configured", lambda: False)
    if route == "arena":
        cycle._run_alpha_arena(container, {})
    elif route == "liquidation":
        cycle._execute_lcrs_signals(container, [signal], {})
    else:
        cycle._execute_options_flow_trades(container, {})
    if policy != "derisk":
        assert not opened
    else:
        trade, = opened
        assert trade["leverage"] == 1
        assert trade["metadata"]["entry_decision"]["firewall"] == "approved"
        assert trade["metadata"]["planned_stop"]["planned_stop_loss_usd"] <= 1000*config.PAPER_MAX_PLANNED_STOP_LOSS_PCT+1e-8
        assert validations[0]["account_balance"] == 1000


def test_equity_limits_survive_restart_and_accounts_are_separate(tmp_path):
    path = tmp_path / "risk.db"
    ledger = EquityRiskLedger(path)
    ledger.observe("crypto", now=NOW, equity=1000.)
    ledger.observe("stocks", now=NOW, equity=1000.)
    ledger.observe("crypto", now=NOW+timedelta(seconds=30), equity=960.)
    restarted = EquityRiskLedger(path)
    assert restarted.status("crypto", now=NOW+timedelta(seconds=31))["reason"] == "daily_equity_loss_limit"
    assert not restarted.status("stocks", now=NOW+timedelta(seconds=31))["blocked"]
    assert restarted.status("crypto", now=NOW)["equity"] == 1000.  # no future observation leaks
    assert restarted.status("crypto", now=NOW+timedelta(minutes=20))["reason"] == "equity_snapshot_stale"


def test_missing_quotes_are_not_zero_returns_or_a_reset_of_peak(tmp_path):
    ledger = EquityRiskLedger(tmp_path / "risk.db")
    ledger.observe("main", now=NOW, equity=1000.)
    ledger.observe("main", now=NOW+timedelta(seconds=1), equity=None, complete=False)
    assert ledger.status("main", now=NOW+timedelta(seconds=1))["reason"] == "equity_quotes_incomplete"
    ledger.observe("main", now=NOW+timedelta(seconds=2), equity=850.)
    state = ledger.status("main", now=NOW+timedelta(seconds=2), daily_loss_pct=.5)
    assert state["peak"] == 1000 and state["max_drawdown"] == 150
    assert state["reason"] == "peak_equity_drawdown_limit"
    with pytest.raises(ValueError, match="out-of-order"):
        ledger.observe("main", now=NOW, equity=2000.)


def test_missing_risk_storage_is_read_only_and_blocks_entries(tmp_path):
    path = tmp_path / "missing.db"
    assert EquityRiskLedger(path).status("main", now=NOW)["blocked"]
    assert not path.exists()


def test_daily_rollover_preserves_equity_peak(tmp_path):
    ledger = EquityRiskLedger(tmp_path / "risk.db")
    ledger.observe("main", now=NOW, equity=1000.)
    ledger.observe("main", now=NOW+timedelta(minutes=1), equity=960.)
    later = NOW+timedelta(days=1)
    ledger.observe("main", now=later, equity=960.)
    status = ledger.status("main", now=later)
    assert status["daily_loss"] == 0 and status["drawdown"] == 40
    assert not status["blocked"]


@pytest.fixture
def guarded_book(tmp_path, monkeypatch):
    path = tmp_path / "bot.db"
    monkeypatch.setattr(config, "DB_PATH", str(path))
    monkeypatch.setattr(config, "DB_BACKEND", "sqlite")
    monkeypatch.setattr(config, "PAPER_PORTFOLIO_GUARD_ENABLED", True)
    monkeypatch.setattr(config, "PAPER_MAX_OPEN_STOP_RISK_PCT", .003)
    monkeypatch.setattr(config, "PAPER_MAX_PLANNED_STOP_LOSS_PCT", .0025)
    monkeypatch.setattr("src.core.clock_provider.utc_now", lambda: NOW)
    db.init_db()
    db.init_paper_account(1000.)
    paper_ledger().observe(PAPER_ACCOUNT, now=NOW, equity=1000., cash=1000., exposure=0.)
    db.set_bot_state("paper_mark_prices", {"observed_at": NOW.isoformat(), "mids": {"BTC": 100., "ETH": 100.}})
    return path


def candidate(coin="BTC", key=None):
    return db.open_paper_trade(None, coin, "long", 100., .9, stop_loss=98., take_profit=105.,
                               idempotency_key=key, metadata={"entry_decision": {"firewall": "approved"}})


def test_whole_book_stop_risk_is_reserved_under_the_admission_lock(guarded_book):
    candidate()
    with pytest.raises(ValueError, match="portfolio stop-risk"):
        candidate("ETH")
    assert len(db.get_open_paper_trades()) == 1


def test_equity_stop_does_not_break_idempotent_replay(guarded_book):
    trade = candidate(key="once")
    paper_ledger().observe(PAPER_ACCOUNT, now=NOW, equity=900.)
    assert candidate(key="once") == trade
    with pytest.raises(ValueError, match="daily_equity_loss"):
        candidate("ETH")
    db.close_paper_trade(trade, 99., -.9)  # risk pause never prevents exits
    assert not db.get_open_paper_trades()


def test_guard_refuses_missing_firewall_proof(guarded_book):
    with pytest.raises(ValueError, match="firewall approval"):
        db.open_paper_trade(None, "BTC", "long", 100, .1, stop_loss=98)


def test_cross_process_portfolio_budget_cannot_be_double_spent(guarded_book, tmp_path):
    env = {**os.environ, "HL_BOT_DB": str(guarded_book), "DB_BACKEND": "sqlite", "PAPER_PORTFOLIO_GUARD_ENABLED": "true",
           "PAPER_MAX_OPEN_STOP_RISK_PCT": ".003", "PAPER_MAX_PLANNED_STOP_LOSS_PCT": ".0025"}
    # Use a current-time persisted observation in both independent worker processes.
    now = datetime.now(timezone.utc)
    with closing(sqlite3.connect(guarded_book.with_suffix(".risk.sqlite3"))) as conn, conn:
        conn.execute("UPDATE account_equity_samples SET observed_at=?,baseline_at=?", (now.isoformat(), now.isoformat()))
    db.set_bot_state("paper_mark_prices", {"observed_at": now.isoformat(), "mids": {"BTC": 100., "ETH": 100.}})
    gate = tmp_path / "go"
    code = """
import sys,time
from pathlib import Path
from src.data import database as db
while not Path(sys.argv[2]).exists(): time.sleep(.01)
try:
 db.open_paper_trade(None,sys.argv[1],'long',100,.9,stop_loss=98,take_profit=105,metadata={'entry_decision':{'firewall':'approved'}})
 print('opened')
except ValueError as exc:
 print('blocked:'+str(exc))
"""
    workers = [subprocess.Popen([sys.executable, "-c", code, coin, str(gate)], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True) for coin in ("BTC", "ETH")]
    try:
        gate.touch()
        outputs = [worker.communicate(timeout=25) for worker in workers]
        assert all(worker.returncode == 0 for worker in workers), outputs
        assert sum(out.strip() == "opened" for out, _ in outputs) == 1, outputs
        assert sum("portfolio stop-risk" in out for out, _ in outputs) == 1, outputs
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.communicate()


def test_recovery_after_partial_fill_does_not_resubmit_or_invent_remaining_size(tmp_path):
    path = tmp_path / "orders.db"
    cid = "0x"+"a"*32
    journal = OrderJournal(path, "test-only")
    assert journal.reserve(cid, {"symbol": "BTC", "side": "buy"}, entry=True)
    restarted = OrderJournal(path, "test-only")
    assert not restarted.reserve("another-entry", {}, entry=True)
    assert restarted.reserve("protective-stop", {}, entry=False)
    restarted.observe("protective-stop", "accepted", "20")
    def lookup(_):
        return {"status": "order", "order": {"status": "canceled", "order": {
            "cloid": cid, "oid": 12, "coin": "BTC", "side": "B", "origSz": ".2", "sz": ".1"}}}
    fill = {"tid": 2, "oid": 12, "coin": "BTC", "side": "B", "sz": ".1", "px": "100"}
    recover_submissions(restarted, lookup, lambda: [fill, fill])
    assert restarted.inventory() == {"BTC": Decimal(".1")}
    assert restarted.status(cid) == "canceled"
    assert restarted.status("protective-stop") == "accepted"
    assert not restarted.reserve(cid, {"symbol": "BTC", "side": "buy"}, entry=True)


def test_negative_venue_lookup_is_not_permission_to_resend(tmp_path):
    journal = OrderJournal(tmp_path / "orders.db", "test-only")
    journal.reserve("ambiguous", {}, entry=True)
    recover_submissions(journal, lambda _: {"status": "unknown"}, lambda: [])
    assert journal.unresolved() == ["ambiguous"]
    assert not journal.reserve("new", {}, entry=True)


def test_execution_mode_probe_is_fast_and_reports_actual_binding(monkeypatch):
    from fastapi.testclient import TestClient
    from src.ui.v2.app import create_app
    from src.ui.v2.state import get_components, set_components
    previous = get_components().live_trader
    monkeypatch.setenv("BOT_RUN_MODE", "paper")
    monkeypatch.setattr(config, "LIVE_TRADING_ENABLED", False)
    monkeypatch.setattr("src.core.readiness.evaluate_readiness", lambda **k: pytest.fail("must not run heavy readiness"))
    try:
        set_components(live_trader=SimpleNamespace(live_requested=False, signer=None))
        client = TestClient(create_app())
        assert client.get("/api/execution-mode").json()["paper_only"] is True
        set_components(live_trader=SimpleNamespace(live_requested=True, signer=object()))
        assert client.get("/api/execution-mode").json()["paper_only"] is False
        set_components(live_trader=None)
        assert client.get("/api/execution-mode").status_code == 503
    finally:
        set_components(live_trader=previous)


def test_live_missing_data_never_uses_paper_capital_or_an_empty_book(monkeypatch):
    from src.core import live_execution
    container = SimpleNamespace(live_trader=SimpleNamespace(get_account_value=lambda: None, get_positions=lambda **k: None))
    monkeypatch.setattr(live_execution, "is_live_trading_active", lambda _: True)
    monkeypatch.setattr(live_execution.db, "get_paper_account", lambda: pytest.fail("live must not read paper capital"))
    assert live_execution.get_execution_account_balance(container) is None
    with pytest.raises(RuntimeError, match="snapshot unavailable"):
        live_execution.get_execution_open_positions(container)


def test_direct_admission_uses_the_selected_execution_wallet(monkeypatch):
    monkeypatch.setenv("LIVE_EXECUTION_VENUE", "lighter")
    monkeypatch.setattr(cycle, "is_live_trading_active", lambda _: True)
    container = SimpleNamespace(live_trader=SimpleNamespace(get_account_value=lambda: 9999.),
                                lighter_live_trader=SimpleNamespace(get_account_value=lambda: 20.))
    assert cycle._direct_account_balance(container) == 20.


def test_strategy_id_cannot_borrow_a_different_strategy_type(monkeypatch):
    monkeypatch.setattr(db, "get_strategy", lambda _: {"strategy_type": "momentum_short", "trade_count": 100})
    assert promotion_gate.is_live_promotable({"strategy_id": 4, "metadata": {
        "source": "strategy", "strategy_type": "momentum_long", "source_key": "strategy:momentum_long",
    }}) == (False, "strategy_source_key_mismatch")


def test_public_collector_probe_requires_both_recent_coins_and_is_read_only(tmp_path):
    from scripts.check_absorption_health import healthy
    path = tmp_path / "feed.db"
    assert not healthy(path, now=NOW)
    assert not path.exists()
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE frames(coin TEXT, at INTEGER, state TEXT, PRIMARY KEY(coin,at)) WITHOUT ROWID")
        stamp = int(NOW.timestamp()*1000)
        conn.execute("INSERT INTO frames VALUES('BTC',?,'ready')", (stamp,))
    assert not healthy(path, now=NOW)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("INSERT INTO frames VALUES('ETH',?,'ready')", (stamp,))
    assert healthy(path, now=NOW)
    assert not healthy(path, now=NOW-timedelta(seconds=1))
    assert not healthy(path, now=NOW+timedelta(seconds=121))
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("UPDATE frames SET state='book_stale' WHERE coin='ETH'")
    assert not healthy(path, now=NOW)


def test_offline_restore_preserves_ambiguous_orders_and_protection(tmp_path):
    from scripts.verify_sqlite_recovery import verify_backup
    source = tmp_path / "before.db"
    journal = OrderJournal(source, "simulation")
    journal.reserve("unknown-entry", {"symbol": "BTC"}, entry=True)
    journal.reserve("existing-stop", {"stop": 98}, entry=False)
    journal.observe("existing-stop", "accepted", "23")
    restored = tmp_path / "restored.db"
    result = verify_backup(source, restored)
    assert result["orders"]["rows"] == 2
    recovered = OrderJournal(restored, "simulation")
    assert recovered.unresolved() == ["unknown-entry"]
    assert recovered.status("existing-stop") == "accepted"
    assert not recovered.reserve("another-entry", {}, entry=True)
    with pytest.raises(ValueError, match="new scratch"):
        verify_backup(source, source)
