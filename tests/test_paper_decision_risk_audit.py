import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import config
from src.data import database as db, decision_journal
from src.trading.copy_trader import CopyTrader
from src.trading.paper_risk import size_for_planned_stop
from src.trading.paper_trader import PaperTrader
from src.trading.portfolio_rotation import RotationDecision


def copy_candidate(**overrides):
    return {
        "type": "copy_open", "coin": "BTC", "side": "short", "price": 100.0,
        "leverage": 2, "confidence": 0.8,
        "source_trader": "0xabc0000000000000000000000000000000000000",
        **overrides,
    }


@pytest.fixture
def copy_trader(monkeypatch):
    monkeypatch.setenv("DECISION_JOURNAL_ENABLED", "true")
    monkeypatch.setattr("src.trading.copy_trader.hl.get_all_mids", lambda: {"BTC": 100.0})
    db.init_db()
    saved = db.get_paper_account()
    db.init_paper_account(10_000)
    monkeypatch.setattr(db, "get_paper_trade_history", lambda **kwargs: [])
    monkeypatch.setattr(config, "PAPER_MAX_PLANNED_STOP_LOSS_PCT", 0.0025)
    trader = CopyTrader(agent_scorer=SimpleNamespace(
        get_weight=lambda _: 0.8, record_signal=lambda *args: "separate-scorer-id",
        record_outcome=lambda *args: None,
    ))
    monkeypatch.setattr(trader.rotation_manager, "decide", lambda *a, **k: RotationDecision("open", "test", 0.8))
    monkeypatch.setattr(trader, "_refresh_copy_guardrail_status", lambda: {"status": "active"})
    yield trader
    if saved:
        db.update_paper_account(saved["balance"], saved["total_pnl"], saved["total_trades"], saved["winning_trades"])
    else:
        with db.get_connection() as conn:
            conn.execute("DELETE FROM paper_account WHERE id = 1")


@pytest.mark.parametrize("exit_path", ["source", "market"])
def test_copy_decision_links_to_open_and_closed_trade(copy_trader, exit_path):
    trade, = copy_trader.execute_copy_signals([copy_candidate()])
    decision_id = trade["metadata"]["decision_id"]
    assert decision_id != "separate-scorer-id"
    assert trade["metadata"]["signal_id"] == "separate-scorer-id"
    assert trade["metadata"]["initial_stop_loss"] == trade["stop_loss"]
    with db.get_connection() as conn:
        snap = conn.execute("SELECT * FROM decision_snapshots WHERE decision_id = ?", (decision_id,)).fetchone()
        opened = conn.execute("SELECT * FROM decision_outcomes WHERE decision_id = ?", (decision_id,)).fetchone()
        persisted = conn.execute("SELECT metadata FROM paper_trades WHERE id = ?", (trade["id"],)).fetchone()
    assert snap["source"] == "copy_trade"
    assert snap["final_status"] == "paper_opened"
    assert opened["paper_trade_id"] == trade["id"] and opened["action_taken"]
    assert opened["label_win"] is None
    assert json.loads(persisted["metadata"])["decision_id"] == decision_id

    if exit_path == "source":
        reason = "source_exit"
        event = copy_trader._close_trade(trade, 104.0, reason)
    else:
        reason = "stop_loss"
        trader = PaperTrader()
        event = trader._close_trade(trade, 104.0, reason)
    assert event["pnl"] < 0
    # A late duplicate open write must not regress the journal's closed state.
    decision_journal.record_paper_opened(decision_id, trade["id"])
    with db.get_connection() as conn:
        closed = conn.execute("SELECT * FROM decision_outcomes WHERE decision_id = ?", (decision_id,)).fetchone()
    assert closed["final_status"] == "paper_closed"
    assert closed["paper_trade_id"] == trade["id"]
    assert closed["exit_reason"] == reason
    assert closed["outcome_pnl"] == event["pnl"]
    assert db.get_paper_account()["balance"] == pytest.approx(10_000 + event["pnl"])


@pytest.mark.parametrize("modifier", ["regime_size_modifier", "copy_source_size_multiplier"])
def test_zero_copy_size_is_journaled_without_closing_incumbent(copy_trader, monkeypatch, modifier):
    open_trade = {"id": 123, "coin": "ETH", "side": "long", "entry_price": 100.0}
    monkeypatch.setattr(db, "get_open_paper_trades", lambda: [open_trade])
    monkeypatch.setattr(config, "ROTATION_ENGINE_ENABLED", True)
    monkeypatch.setattr(config, "ROTATION_DRY_RUN_TELEMETRY", False)
    monkeypatch.setattr(copy_trader.rotation_manager, "decide", lambda *a, **k: RotationDecision(
        action="replace", reason="test", candidate_score=0.8, replacement_trade_id=123,
    ))
    monkeypatch.setattr(copy_trader, "_close_trade", lambda *a, **k: pytest.fail("must not close incumbent"))
    assert copy_trader.execute_copy_signals([copy_candidate(**{modifier: 0.0})]) == []
    with db.get_connection() as conn:
        outcome = conn.execute(
            "SELECT * FROM decision_outcomes WHERE final_status = 'copy_entry_rejected'",
        ).fetchone()
    assert outcome is not None and not outcome["action_taken"]


def test_copy_approval_is_not_execution_when_duplicate_is_skipped(copy_trader):
    def validate(signal, *, dry_run=False, **kwargs):
        if not dry_run:
            decision_journal.update_decision_status(
                signal.signal_id, final_status="approved", firewall_decision="approved",
            )
        return True, ""

    copy_trader.firewall = SimpleNamespace(validate=validate)
    assert len(copy_trader.execute_copy_signals([copy_candidate()])) == 1
    assert copy_trader.execute_copy_signals([copy_candidate()]) == []
    with db.get_connection() as conn:
        outcome = conn.execute(
            "SELECT * FROM decision_outcomes WHERE final_status = 'paper_open_failed'",
        ).fetchone()
        snapshot = conn.execute(
            "SELECT * FROM decision_snapshots WHERE decision_id = ?", (outcome["decision_id"],),
        ).fetchone()
    assert snapshot["firewall_decision"] == "approved"
    assert snapshot["rejection_reason"] == "copy_already_open_from_source"
    assert not outcome["action_taken"] and outcome["paper_trade_id"] is None


def test_strategy_entry_persists_capped_size_without_changing_stop(copy_trader, monkeypatch):
    monkeypatch.setattr(PaperTrader, "_apply_slippage", staticmethod(lambda price, *a, **k: price))
    paper = PaperTrader()
    trade = paper._execute_paper_trade(db.get_paper_account(), {}, {
        "coin": "BTC", "side": "short", "price": 100.0, "size": 7.32, "leverage": 2,
        "stop_loss": 102.4, "take_profit": 96.0, "strategy_type": "scalping",
    })
    assert trade["size"] < 7.32
    assert trade["stop_loss"] == 102.4
    assert trade["take_profit"] == 96.0
    assert trade["metadata"]["planned_stop_loss_usd"] == pytest.approx(25)
    saved, = db.get_open_paper_trades()
    assert saved["size"] == pytest.approx(trade["size"])


def test_risk_budget_caps_size_using_leverage_and_costs(monkeypatch):
    monkeypatch.setattr(config, "PAPER_MAX_PLANNED_STOP_LOSS_PCT", 0.0025)
    args = dict(balance=10_000, side="short", entry=100.0, stop=102.5,
                execution_role="taker", size=7.3)
    size, details = size_for_planned_stop(leverage=2, **args)
    high_lev_size, _ = size_for_planned_stop(leverage=4, **args)
    assert size < 5.0  # Distance alone would allow 5; fees and modeled slippage count.
    assert high_lev_size == pytest.approx(size / 2)
    assert details["planned_stop_loss_usd"] == pytest.approx(25)
    assert details["entry_notional_usd"] == pytest.approx(200 * size)
    with pytest.raises(ValueError, match="wrong side"):
        size_for_planned_stop(leverage=2, **{**args, "stop": 99.0})


@pytest.mark.parametrize("side, original, tightened, mid", [
    ("long", 96.0, 100.5, 99.0), ("short", 104.0, 99.5, 101.0),
])
def test_saved_tightened_stop_retains_initial_risk_and_reports_trigger(monkeypatch, side, original, tightened, mid):
    opened_at = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    trade = {"id": 42, "coin": "BTC", "side": side, "entry_price": 100.0, "size": 1,
             "leverage": 2, "stop_loss": tightened, "take_profit": None, "opened_at": opened_at,
             "metadata": {"initial_stop_loss": original}}
    monkeypatch.setattr(db, "get_paper_account", lambda: {"balance": 10_000})
    monkeypatch.setattr(db, "get_open_paper_trades", lambda: [trade])
    monkeypatch.setattr("src.trading.paper_trader.hl.get_asset_contexts", lambda: {}, raising=False)
    trader = PaperTrader()
    monkeypatch.setattr(trader, "_update_trade_excursion", lambda *a, **k: None)
    monkeypatch.setattr(trader, "_close_trade", lambda trade, **kwargs: {**kwargs, "stop": trade["stop_trigger_price"]})
    assert trader._resolve_trade_risk_policy(trade)["stop_roe_pct"] == pytest.approx(0.08)
    assert trader.check_open_positions(mids={"BTC": mid}) == [
        {"current_price": mid, "close_reason": "trailing_stop", "stop": tightened},
    ]
