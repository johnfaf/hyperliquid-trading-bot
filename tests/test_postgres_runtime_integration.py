"""Real migrations and close/credit contracts against a disposable Postgres database."""
from datetime import datetime, timedelta, timezone
import os

import pytest

import config
from src.data import database as db
from src.data.db import migrations, postgres
from src.trading.paper_funding import FundingLedger, HOUR, ledger_path, millis, verified_cost
from src.trading.paper_trader import PaperTrader


def test_funded_close_is_atomic_in_dualwrite(tmp_path, monkeypatch):
    dsn = os.getenv("HL_AUDIT_POSTGRES_DSN")
    if not dsn:
        pytest.skip("requires the disposable Postgres integration service")
    from psycopg import connect
    from psycopg.conninfo import conninfo_to_dict

    if conninfo_to_dict(dsn).get("dbname") != "hl_audit_integration":
        pytest.fail("integration test only runs in the dedicated test database")
    postgres._close_pool()
    path = tmp_path / "bot.db"
    now = datetime(2026, 1, 1, 12, 5, tzinfo=timezone.utc)
    opened = now - timedelta(minutes=20)
    monkeypatch.setattr(config, "POSTGRES_DSN", dsn)
    monkeypatch.setattr(config, "DB_PATH", str(path))
    monkeypatch.setattr(config, "DB_BACKEND", "dualwrite")
    monkeypatch.setattr(config, "PAPER_FUNDING_LEDGER_ENABLED", True)
    monkeypatch.setattr(config, "PAPER_TRADING_FUNDING_ENABLED", True)
    monkeypatch.setattr(config, "PAPER_PORTFOLIO_GUARD_ENABLED", False)
    monkeypatch.setattr(config, "PAPER_TRADING_TAKER_FEE_BPS", 0)
    monkeypatch.setattr(PaperTrader, "_sync_fee_tier", lambda self: None)
    monkeypatch.setattr(PaperTrader, "_apply_slippage", staticmethod(lambda price, *a, **kw: price))
    monkeypatch.setattr("src.core.clock_provider.utc_now", lambda: opened)
    try:
        db.init_db()
        assert migrations.run_migrations() == 0
        db.init_paper_account(1000.)
        trade_id = db.open_paper_trade(None, "BTC", "long", 100., 1., leverage=2., stop_loss=95.)
        ledger = FundingLedger(ledger_path(path))
        hour = millis(now) // HOUR * HOUR
        ledger.record(observed_ms=hour-30_000, oracle_prices={"BTC": 100})
        ledger.record(observed_ms=hour+20_000, rates=[{"coin": "BTC", "time": hour, "fundingRate": ".001"}])
        monkeypatch.setattr("src.core.clock_provider.utc_now", lambda: now)
        trader = PaperTrader()
        trade, = db.get_open_paper_trades()
        assert trader._close_trade(trade, 105., "take_profit")["pnl"] == pytest.approx(9.8)
        assert trader._close_trade(trade, 105., "take_profit") is None
        with connect(dsn) as pg:
            assert pg.execute("SELECT balance,total_trades FROM paper_account").fetchone() == (1009.8, 1)
            status, pnl, meta = pg.execute("SELECT status,pnl,metadata FROM paper_trades WHERE id=%s", (trade_id,)).fetchone()
            assert status == "closed" and pnl == pytest.approx(9.8)
            assert verified_cost({**trade, "closed_at": now}, meta["funding_coverage"]) == pytest.approx(.2)

        second_id = db.open_paper_trade(None, "ETH", "long", 100., 1., stop_loss=95.)
        with connect(dsn) as pg:
            # Force a PostgreSQL-only rejection between trade and balance updates.
            pg.execute("ALTER TABLE paper_trades ADD CHECK (NOT (metadata ? 'integration_reject'))")
        assert db.close_paper_trade_and_credit_account(
            second_id, 101., 1., metadata_updates={"integration_reject": True}
        )
        assert db.get_paper_account()["balance"] == pytest.approx(1010.8)
        with connect(dsn) as pg:
            assert pg.execute("SELECT status FROM paper_trades WHERE id=%s", (second_id,)).fetchone() == ("open",)
            assert pg.execute("SELECT balance,total_trades FROM paper_account").fetchone() == (1009.8, 1)
    finally:
        postgres._close_pool()
