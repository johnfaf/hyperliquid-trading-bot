import logging
import os
import sqlite3
import sys
import tempfile
import types
from contextlib import contextmanager
from datetime import datetime, timezone

import config
import main
from src.core import boot
from src.core.cycles.trading_cycle import _process_closed_trades
from src.core.live_execution import (
    _rescale_size_for_live,
    get_execution_open_positions,
    sync_shadow_book_to_live,
)
from src.signals.decision_firewall import DecisionFirewall
from src.signals.signal_schema import TradeSignal, signal_from_execution_dict
from src.trading.live_trader import HyperliquidSigner, LiveTrader
from src.trading.portfolio_rotation import PortfolioRotationManager, RotationDecision


def _fake_live_credentials(self):
    self.signer = type("Signer", (), {"address": "0x1111111111111111111111111111111111111111"})()
    self.agent_wallet_address = self.signer.address
    self.public_address = "0x2222222222222222222222222222222222222222"
    self.status_reason = "credentials_loaded"


def test_shadow_mode_allows_missing_rotation_threshold_envs(monkeypatch):
    logger = logging.getLogger("test-shadow")
    monkeypatch.setattr(config, "ROTATION_ENGINE_ENABLED", True)
    monkeypatch.setattr(config, "ROTATION_REQUIRE_EXPLICIT_THRESHOLDS", True)
    monkeypatch.setattr(config, "ROTATION_DRY_RUN_TELEMETRY", True)
    monkeypatch.setattr(config, "ROTATION_SHADOW_MODE_DAYS", 7)

    required = [
        "PORTFOLIO_REPLACEMENT_THRESHOLD",
        "PORTFOLIO_MAX_REPLACEMENTS_PER_CYCLE",
        "PORTFOLIO_MAX_REPLACEMENTS_PER_HOUR",
        "PORTFOLIO_MAX_REPLACEMENTS_PER_DAY",
        "PORTFOLIO_FORCED_EXIT_COOLDOWN_MINUTES",
        "PORTFOLIO_ROUND_TRIP_BLOCK_MINUTES",
        "PORTFOLIO_MAX_COIN_EXPOSURE_PCT",
        "PORTFOLIO_MAX_SIDE_EXPOSURE_PCT",
        "PORTFOLIO_MAX_CLUSTER_EXPOSURE_PCT",
        "PORTFOLIO_TRANSACTION_COST_WEIGHT",
        "PORTFOLIO_CHURN_PENALTY",
        "PORTFOLIO_EXPECTED_SLIPPAGE_BPS",
    ]
    for env_name in required:
        monkeypatch.delenv(env_name, raising=False)

    boot.validate_operational_controls(logger)


def test_live_mode_still_requires_explicit_rotation_threshold_envs(monkeypatch):
    logger = logging.getLogger("test-live-thresholds")
    monkeypatch.setattr(config, "ROTATION_ENGINE_ENABLED", True)
    monkeypatch.setattr(config, "ROTATION_REQUIRE_EXPLICIT_THRESHOLDS", True)
    monkeypatch.setattr(config, "ROTATION_DRY_RUN_TELEMETRY", False)

    monkeypatch.delenv("PORTFOLIO_REPLACEMENT_THRESHOLD", raising=False)

    try:
        boot.validate_operational_controls(logger)
        assert False, "Expected RuntimeError when live mode lacks explicit thresholds"
    except RuntimeError as exc:
        assert "PORTFOLIO_REPLACEMENT_THRESHOLD" in str(exc)


def test_signal_from_execution_dict_preserves_trade_metadata():
    signal = signal_from_execution_dict(
        {
            "coin": "BTC",
            "side": "long",
            "confidence": 0.73,
            "entry_price": 101000,
            "strategy_id": 42,
            "strategy_type": "momentum_long",
            "position_pct": 0.06,
            "leverage": 3,
            "size": 0.15,
            "signal_id": "abc123",
            "source_accuracy": 0.61,
        }
    )

    assert signal.coin == "BTC"
    assert signal.side.value == "long"
    assert signal.strategy_id == 42
    assert signal.strategy_type == "momentum_long"
    assert signal.position_pct == 0.06
    assert signal.size == 0.15
    assert signal.signal_id == "abc123"
    assert signal.source_accuracy == 0.61


def test_shadow_mode_bypasses_rotation_reject_when_capacity_exists():
    manager = PortfolioRotationManager({"target_positions": 8})
    decision = RotationDecision(
        action="reject",
        reason="reserved slots held for stronger arrivals",
        candidate_score=0.55,
    )
    assert manager.should_bypass_reject_in_shadow_mode(decision, open_positions_count=6) is True
    assert manager.should_bypass_reject_in_shadow_mode(decision, open_positions_count=8) is False


def test_live_trader_coerces_execution_dict_and_uses_live_firewall_state(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            assert isinstance(signal, TradeSignal)
            assert signal.coin == "ETH"
            assert kwargs["open_positions"] == [{"coin": "BTC", "size": 0.1, "entry_price": 100000, "leverage": 2}]
            assert kwargs["account_balance"] == 2500.0
            return True, "ok"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [
        {"coin": "BTC", "size": 0.1, "entry_price": 100000, "leverage": 2}
    ])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(LiveTrader, "place_market_order", lambda self, *args, **kwargs: {"status": "success"})
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(LiveTrader, "verify_fill", lambda self, *args, **kwargs: {"status": "verified"})
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(LiveTrader, "place_trigger_order", lambda self, *args, **kwargs: {"status": "success"})

    # Disable the bootstrap $ cap so the test exercises the original sizing path.
    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
    result = trader.execute_signal(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.1,
            "strategy_type": "momentum_long",
        }
    )

    assert result is not None
    assert result["status"] == "success"


def test_live_trader_maps_long_to_buy_and_places_sell_protection(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    entry_calls = []
    trigger_calls = []

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(
        LiveTrader,
        "place_market_order",
        lambda self, coin, side, size, leverage=1, reduce_only=False: (
            entry_calls.append((coin, side, size, reduce_only)) or {"status": "success"}
        ),
    )
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(LiveTrader, "verify_fill", lambda self, *args, **kwargs: {"status": "verified"})
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(
        LiveTrader,
        "place_trigger_order",
        lambda self, coin, side, size, trigger_price, tp_or_sl="sl": (
            trigger_calls.append((coin, side, size, trigger_price, tp_or_sl)) or {"status": "success"}
        ),
    )

    # Disable the bootstrap $ cap so entry_calls reflects the requested 0.1 ETH.
    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
    result = trader.execute_signal(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.1,
            "strategy_type": "momentum_long",
        }
    )

    assert result is not None
    assert result["status"] == "success"
    assert entry_calls == [("ETH", "buy", 0.1, False)]
    assert [call[1] for call in trigger_calls] == ["sell", "sell"]
    assert [call[4] for call in trigger_calls] == ["sl", "tp"]


def test_live_trader_uses_verified_fill_size_for_protection(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    trigger_calls = []

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(LiveTrader, "place_market_order", lambda self, *args, **kwargs: {"status": "success"})
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(
        LiveTrader,
        "verify_fill",
        lambda self, *args, **kwargs: {"status": "verified", "size": 0.06, "partial_fill": True},
    )
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(
        LiveTrader,
        "place_trigger_order",
        lambda self, coin, side, size, trigger_price, tp_or_sl="sl": (
            trigger_calls.append((coin, side, size, trigger_price, tp_or_sl)) or {"status": "success"}
        ),
    )

    # Disable the bootstrap $ cap so fill verification preserves the requested 0.1 size.
    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
    result = trader.execute_signal(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.1,
            "strategy_type": "momentum_long",
        }
    )

    assert result is not None
    assert result["status"] == "success"
    assert abs(result["size"] - 0.06) < 1e-12
    assert abs(result["requested_size"] - 0.1) < 1e-12
    assert [call[2] for call in trigger_calls] == [0.06, 0.06]


def test_live_trader_closes_position_when_protection_fails(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    trigger_results = iter(
        [
            {"status": "success"},
            {"status": "error", "message": "tp rejected"},
        ]
    )
    close_calls = []

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(LiveTrader, "place_market_order", lambda self, *args, **kwargs: {"status": "success"})
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(LiveTrader, "verify_fill", lambda self, *args, **kwargs: {"status": "verified"})
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(LiveTrader, "place_trigger_order", lambda self, *args, **kwargs: next(trigger_results))
    monkeypatch.setattr(
        LiveTrader,
        "close_position",
        lambda self, coin: close_calls.append(coin) or {"status": "success", "coin": coin},
    )

    # Disable the bootstrap $ cap so the sizing reaches place_trigger_order unmodified.
    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)
    result = trader.execute_signal(
        {
            "coin": "ETH",
            "side": "short",
            "confidence": 0.75,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.1,
            "strategy_type": "momentum_short",
        }
    )

    assert result is not None
    assert result["status"] == "error"
    assert result["message"] == "protective_order_failed"
    assert close_calls == ["ETH"]


def test_verify_fill_waits_for_minimum_meaningful_fill(monkeypatch):
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    class FakeClock:
        def __init__(self):
            self.current = 0.0

        def time(self):
            return self.current

        def sleep(self, seconds):
            self.current += seconds

    positions = iter(
        [
            [{"coin": "ETH", "szi": 0.04}],
            [{"coin": "ETH", "szi": 0.06}],
        ]
    )
    clock = FakeClock()

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_positions", lambda self: next(positions))
    monkeypatch.setattr("src.trading.live_trader.time.time", clock.time)
    monkeypatch.setattr("src.trading.live_trader.time.sleep", clock.sleep)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False)
    result = trader.verify_fill("ETH", "buy", 0.1, timeout=1.0, poll_interval=0.1)

    assert result is not None
    assert result["status"] == "verified"
    assert result["partial_fill"] is True
    assert abs(result["size"] - 0.06) < 1e-12
    assert abs(result["position_size"] - 0.06) < 1e-12


def test_execution_open_positions_prefer_live_state_when_deployable():
    class FakeLiveTrader:
        def is_live_enabled(self):
            return True

        def is_deployable(self):
            return True

        def get_positions(self):
            return [{"coin": "BTC", "side": "long", "size": 0.25}]

    container = type("Container", (), {"live_trader": FakeLiveTrader()})()
    positions = get_execution_open_positions(container)
    assert positions == [{"coin": "BTC", "side": "long", "size": 0.25}]


def test_sync_shadow_book_closes_paper_trade_when_live_position_missing(monkeypatch):
    metadata_updates = []
    closed = []

    class FakeLiveTrader:
        def is_live_enabled(self):
            return True

        def is_deployable(self):
            return True

        def get_positions(self):
            return []

        def get_account_value(self):
            return 500.0  # Funded account — reconciliation should proceed

    container = type(
        "Container",
        (),
        {"live_trader": FakeLiveTrader(), "paper_trader": object()},
    )()

    monkeypatch.setattr(
        "src.core.live_execution.db.get_open_paper_trades",
        lambda: [{"id": 7, "coin": "ETH", "side": "long", "entry_price": 2000.0}],
    )
    monkeypatch.setattr(
        "src.core.live_execution.get_all_mids",
        lambda: {"ETH": 2100.0},
    )
    monkeypatch.setattr(
        "src.core.live_execution.db.update_paper_trade_metadata",
        lambda trade_id, meta: metadata_updates.append((trade_id, meta)),
    )
    monkeypatch.setattr(
        "src.core.live_execution.db.close_paper_trade",
        lambda trade_id, exit_price, pnl: closed.append((trade_id, exit_price, pnl)) or True,
    )

    reconciled = sync_shadow_book_to_live(container)
    assert len(reconciled) == 1
    assert reconciled[0]["trade_id"] == 7
    assert reconciled[0]["coin"] == "ETH"
    assert reconciled[0]["reason"] == "live_reconciled_closed"
    assert reconciled[0]["pnl"] == 0.0
    assert closed == [(7, 2100.0, 0.0)]
    assert metadata_updates[0][0] == 7
    assert metadata_updates[0][1]["synthetic_reconciliation"] is True
    assert metadata_updates[0][1]["reconciliation_reason"] == "live_reconciled_closed"


def test_process_closed_trades_skips_synthetic_reconciliation():
    class FakeArena:
        def __init__(self):
            self.calls = []

        def record_trade_for_strategy(self, *args):
            self.calls.append(args)

    class FakeKelly:
        def __init__(self):
            self.calls = []

        def record_outcome(self, **kwargs):
            self.calls.append(kwargs)

    class FakeAgentScorer:
        def __init__(self):
            self.calls = []

        def record_outcome(self, *args):
            self.calls.append(args)

    container = type(
        "Container",
        (),
        {
            "arena": FakeArena(),
            "kelly_sizer": FakeKelly(),
            "agent_scorer": FakeAgentScorer(),
            "shadow_tracker": None,
        },
    )()

    _process_closed_trades(
        container,
        [
            {
                "trade_id": 7,
                "coin": "ETH",
                "side": "long",
                "entry_price": 2000.0,
                "exit_price": 2100.0,
                "size": 0.1,
                "leverage": 2,
                "pnl": 0.0,
                "strategy_type": "momentum_long",
                "reason": "live_reconciled_closed",
                "metadata": {"synthetic_reconciliation": True},
            }
        ],
    )

    assert container.arena.calls == []
    assert container.kelly_sizer.calls == []
    assert container.agent_scorer.calls == []


def test_firewall_uses_explicit_live_positions_without_falling_back(monkeypatch):
    monkeypatch.setattr(
        "src.signals.decision_firewall.db.get_open_paper_trades",
        lambda: [{"coin": "BTC"}] * 10,
    )
    monkeypatch.setattr(
        "src.signals.decision_firewall.db.audit_log",
        lambda **kwargs: None,
    )

    firewall = DecisionFirewall(
        {
            "max_positions": 1,
            "funding_risk_enabled": False,
            "enable_predictive_derisk": False,
        }
    )
    signal = signal_from_execution_dict(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.1,
        }
    )

    passed, reason = firewall.validate(signal, open_positions=[], account_balance=5000.0)

    assert passed is True
    assert reason == "approved"


def test_firewall_daily_drawdown_uses_provided_account_balance(monkeypatch):
    monkeypatch.setattr(
        "src.signals.decision_firewall.db.audit_log",
        lambda **kwargs: None,
    )

    firewall = DecisionFirewall(
        {
            "daily_loss_limit_pct": 0.03,
            "funding_risk_enabled": False,
            "enable_predictive_derisk": False,
        }
    )
    firewall._daily_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    firewall.set_daily_losses(40.0)
    signal = signal_from_execution_dict(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.1,
        }
    )

    passed, reason = firewall.validate(signal, open_positions=[], account_balance=1000.0)

    assert passed is False
    assert "Daily loss limit hit" in reason


def test_positive_daily_pnl_does_not_trigger_kill_switch(monkeypatch):
    monkeypatch.setattr(LiveTrader, "_load_credentials", lambda self: None)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)

    trader = LiveTrader(
        firewall=DecisionFirewall(
            {
                "funding_risk_enabled": False,
                "enable_predictive_derisk": False,
            }
        ),
        dry_run=True,
    )
    trader.daily_pnl = 125.0

    assert trader.check_daily_loss() is False
    assert trader.kill_switch_active is False


def test_hyperliquid_signer_zero_pads_signature_components(monkeypatch):
    class FakeAccount:
        def sign_message(self, message):
            return type("Signed", (), {"r": 0x1234, "s": 0xABCD, "v": 27})()

    signer = HyperliquidSigner.__new__(HyperliquidSigner)
    signer.account = FakeAccount()

    # New signer uses eth_account.encode_typed_data (imported as
    # _encode_typed_data).  Replace it with a no-op so we do not need a
    # real eth_account install to verify zero-padding.
    monkeypatch.setattr(
        "src.trading.live_trader._encode_typed_data",
        lambda *args, **kwargs: object(),
        raising=False,
    )

    signature = signer.sign_action({"type": "noop"}, nonce=123)

    assert signature["r"].startswith("0x")
    assert signature["s"].startswith("0x")
    assert len(signature["r"]) == 66
    assert len(signature["s"]) == 66
    assert signature["r"].endswith("1234")
    assert signature["s"].endswith("abcd")


def test_hyperliquid_signer_includes_vault_address_in_hash():
    """Regression test: the old signer ignored the vault address when
    computing the signed hash, so agent-wallet orders signed one hash and
    Hyperliquid reconstructed a different one — ecrecover then returned a
    mystery address and the exchange replied "User or API Wallet 0x… does
    not exist".  The fix includes ``vault_address`` in the keccak
    preimage, so the same action + nonce with different vaults MUST
    produce different hashes.
    """
    action = {
        "type": "order",
        "orders": [{"a": 0, "b": True, "p": "50000", "s": "0.001", "r": False,
                    "t": {"limit": {"tif": "Ioc"}}}],
        "grouping": "na",
    }
    nonce = 1_700_000_000_000

    no_vault_hash = HyperliquidSigner._action_hash(action, None, nonce)
    with_vault_hash = HyperliquidSigner._action_hash(
        action, "0x1234567890abcdef1234567890abcdef12345678", nonce,
    )
    other_vault_hash = HyperliquidSigner._action_hash(
        action, "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd", nonce,
    )

    assert len(no_vault_hash) == 32
    assert len(with_vault_hash) == 32
    assert no_vault_hash != with_vault_hash
    assert with_vault_hash != other_vault_hash


def test_hyperliquid_signer_uses_agent_primary_type_and_exchange_domain():
    """Regression test: the signing payload must use ``primaryType='Agent'``
    with ``domain.name='Exchange'`` (Hyperliquid's real scheme), not the
    legacy ``HyperliquidTransaction`` struct that produced garbage
    ecrecover addresses.
    """
    captured = {}

    class FakeAccount:
        def sign_message(self, message):
            return type("Signed", (), {"r": 1, "s": 2, "v": 27})()

    def fake_encode(*args, **kwargs):
        captured["payload"] = kwargs.get("full_message") or (args[0] if args else None)
        return object()

    signer = HyperliquidSigner.__new__(HyperliquidSigner)
    signer.account = FakeAccount()

    import src.trading.live_trader as lt
    original = lt._encode_typed_data
    try:
        lt._encode_typed_data = fake_encode
        signer.sign_action({"type": "noop"}, nonce=1, vault_address=None)
    finally:
        lt._encode_typed_data = original

    payload = captured["payload"]
    assert payload is not None, "encode_typed_data was not called"
    assert payload["primaryType"] == "Agent"
    assert payload["domain"]["name"] == "Exchange"
    assert payload["domain"]["chainId"] == 1337
    assert payload["domain"]["verifyingContract"] == "0x" + "0" * 40
    assert "Agent" in payload["types"]
    assert {"name": "source", "type": "string"} in payload["types"]["Agent"]
    assert {"name": "connectionId", "type": "bytes32"} in payload["types"]["Agent"]
    assert payload["message"]["source"] == "a"
    assert isinstance(payload["message"]["connectionId"], (bytes, bytearray))
    assert len(payload["message"]["connectionId"]) == 32


def test_rescale_size_for_live_blocks_when_paper_balance_missing(monkeypatch):
    class FakeTrader:
        def get_account_value(self):
            return 2500.0

    monkeypatch.setattr("src.core.live_execution.db.get_paper_account", lambda: {"balance": 0})

    scaled = _rescale_size_for_live({"coin": "ETH", "size": 0.2}, FakeTrader())

    assert scaled is None


def test_rescale_size_for_live_enforces_max_order_usd_cap(monkeypatch):
    """After rescaling, notional must never exceed trader.max_order_usd."""
    class FakeTrader:
        max_order_usd = 3.0

        def get_account_value(self):
            return 10_000.0  # equal to paper — scale = 1.0

    monkeypatch.setattr("src.core.live_execution.db.get_paper_account", lambda: {"balance": 10_000.0})
    monkeypatch.setattr("src.core.live_execution.get_all_mids", lambda: {"ETH": 2000.0})

    # 0.5 ETH @ $2000 = $1000 notional, should be capped to $3 → 0.0015 ETH
    scaled = _rescale_size_for_live(
        {"coin": "ETH", "size": 0.5, "entry_price": 2000.0},
        FakeTrader(),
    )
    assert scaled is not None
    assert abs(scaled["size"] * 2000.0 - 3.0) < 1e-9


def test_rescale_size_for_live_floors_up_to_exchange_minimum(monkeypatch):
    """Small live wallet vs large paper book: proportional rescaling
    produces sub-minimum orders that Hyperliquid silently drops.  The
    rescaler should floor UP to the exchange minimum rather than skip,
    so the bot actually trades.  Regression test for the XRP copy-mirror
    scenario where 0.172 XRP @ $1.24 = $0.21 notional got skipped.
    """
    class FakeTrader:
        max_order_usd = 12.0
        min_order_usd = 11.0

        def get_account_value(self):
            return 12.44  # the real problem case

    monkeypatch.setattr("src.core.live_execution.db.get_paper_account",
                        lambda: {"balance": 10_041.0})
    monkeypatch.setattr("src.core.live_execution.get_all_mids",
                        lambda: {"XRP": 1.24})

    # 138.7 XRP × $1.24 = $172 paper notional, rescales to 0.172 XRP
    # (= $0.21) at scale factor 0.00124.  Should floor UP to ~$11 notional.
    scaled = _rescale_size_for_live(
        {"coin": "XRP", "size": 138.7, "entry_price": 1.24, "leverage": 5},
        FakeTrader(),
    )
    assert scaled is not None, "floor-up should succeed, not skip"
    notional = scaled["size"] * 1.24
    assert notional >= 11.0, f"notional ${notional:.2f} should be >= $11 min"
    assert notional <= 12.0, f"notional ${notional:.2f} should be <= $12 cap"


def test_rescale_size_for_live_skips_when_margin_exceeds_wallet_headroom(monkeypatch):
    """If even the minimum order's margin requirement exceeds 80% of the
    live wallet, skip instead of flooring up — the wallet is too small
    to safely carry any position at the configured leverage."""
    class FakeTrader:
        max_order_usd = 11.0
        min_order_usd = 11.0

        def get_account_value(self):
            return 5.0  # wallet too small for $11 notional at 1x

    monkeypatch.setattr("src.core.live_execution.db.get_paper_account",
                        lambda: {"balance": 10_000.0})
    monkeypatch.setattr("src.core.live_execution.get_all_mids",
                        lambda: {"BTC": 50_000.0})

    # 0.01 BTC × $50k = $500 paper, rescales to 0.00001 BTC = $0.25.
    # Min order $11 at 1x leverage = $11 margin > 80% of $5 wallet ($4).
    scaled = _rescale_size_for_live(
        {"coin": "BTC", "size": 0.01, "entry_price": 50_000.0, "leverage": 1},
        FakeTrader(),
    )
    assert scaled is None, "should skip: margin too large for wallet"


def test_execute_signal_bypass_firewall_skips_validation(monkeypatch):
    """When bypass_firewall=True, DecisionFirewall.validate must NOT be called."""
    validate_calls = []

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            validate_calls.append(signal.coin)
            return False, "Cooldown: ETH traded 58s ago"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(LiveTrader, "place_market_order", lambda self, *args, **kwargs: {"status": "success"})
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(LiveTrader, "verify_fill", lambda self, *args, **kwargs: {"status": "verified"})
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(LiveTrader, "place_trigger_order", lambda self, *args, **kwargs: {"status": "success"})

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=1_000_000)

    # Without bypass: firewall rejects, result is None
    r_strict = trader.execute_signal(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.1,
            "strategy_type": "momentum_long",
        }
    )
    assert r_strict is None
    assert validate_calls == ["ETH"]

    # With bypass: firewall is NOT called, trade proceeds
    validate_calls.clear()
    r_bypass = trader.execute_signal(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.1,
            "strategy_type": "momentum_long",
        },
        bypass_firewall=True,
    )
    assert r_bypass is not None
    assert r_bypass["status"] == "success"
    assert validate_calls == []


def test_execute_signal_caps_notional_at_max_order_usd(monkeypatch):
    """A large computed size must be shrunk so notional <= max_order_usd."""
    placed_sizes = []

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 2500.0)
    monkeypatch.setattr(
        LiveTrader,
        "place_market_order",
        lambda self, coin, side, size, leverage=1, reduce_only=False: (
            placed_sizes.append(size) or {"status": "success"}
        ),
    )
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(LiveTrader, "verify_fill", lambda self, *args, **kwargs: {"status": "verified"})
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2000.0)
    monkeypatch.setattr(LiveTrader, "place_trigger_order", lambda self, *args, **kwargs: {"status": "success"})

    # Use a cap above the $10 exchange minimum so the test is not fighting
    # LiveTrader's min-notional auto-raise.  $15 is comfortably above the
    # $11 default min and proves the cap logic still shrinks large orders.
    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=15.0)
    result = trader.execute_signal(
        {
            "coin": "ETH",
            "side": "long",
            "confidence": 0.8,
            "entry_price": 2000.0,
            "position_pct": 0.05,
            "leverage": 2,
            "size": 0.5,  # $1000 notional — should be capped to $15 → 0.0075 ETH
            "strategy_type": "momentum_long",
        }
    )

    assert result is not None
    assert result["status"] == "success"
    assert len(placed_sizes) == 1
    assert abs(placed_sizes[0] * 2000.0 - 15.0) < 1e-9


def test_live_trader_raises_cap_to_exchange_minimum(monkeypatch):
    """Regression test: LIVE_MAX_ORDER_USD below Hyperliquid's $10 minimum
    notional makes every live order silently fail (matching engine drops
    it, then fill verification times out as "FILL NOT VERIFIED").  The
    LiveTrader should raise the cap to the exchange minimum at startup
    with a warning rather than allow the impossible state."""
    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=3.0)

    # Cap was below the $10 min — should have been raised to min (default 11)
    assert trader.max_order_usd >= 10.0
    assert trader.max_order_usd == trader.min_order_usd


def test_execute_signal_rejects_below_exchange_minimum(monkeypatch):
    """An order whose rescaled notional is under the exchange minimum
    must be rejected up front, not sent and then fail fill verification."""
    placed_sizes = []

    class FakeFirewall:
        def validate(self, signal, **kwargs):
            return True, "ok"

    monkeypatch.setattr(LiveTrader, "_load_credentials", _fake_live_credentials)
    monkeypatch.setattr(LiveTrader, "_load_asset_index_map", lambda self: None)
    monkeypatch.setattr(LiveTrader, "reconcile_positions", lambda self: None)
    monkeypatch.setattr(LiveTrader, "get_firewall_positions", lambda self: [])
    monkeypatch.setattr(LiveTrader, "get_account_value", lambda self: 20.0)
    monkeypatch.setattr(
        LiveTrader,
        "place_market_order",
        lambda self, coin, side, size, leverage=1, reduce_only=False: (
            placed_sizes.append(size) or {"status": "success"}
        ),
    )
    monkeypatch.setattr(LiveTrader, "update_daily_pnl_from_fills", lambda self: None)
    monkeypatch.setattr(LiveTrader, "verify_fill", lambda self, *a, **kw: {"status": "verified"})
    monkeypatch.setattr(LiveTrader, "_get_mid_price", lambda self, coin: 2.5)

    trader = LiveTrader(firewall=FakeFirewall(), dry_run=False, max_order_usd=12.0)
    # Force a tiny order whose notional is well below $11: 0.172 * $2.50 = $0.43
    result = trader.execute_signal(
        {
            "coin": "XRP",
            "side": "short",
            "confidence": 0.8,
            "entry_price": 2.5,
            "position_pct": 0.01,
            "leverage": 5,
            "size": 0.172,
            "strategy_type": "momentum_short",
        }
    )

    # Should be rejected before hitting place_market_order
    assert result is None or result.get("status") in {"rejected", "error"}
    assert placed_sizes == []


def test_copy_trade_preserves_precise_stops_for_low_priced_assets(monkeypatch):
    opened = {}

    def fake_open_paper_trade(**kwargs):
        opened.update(kwargs)
        return 321

    monkeypatch.setitem(sys.modules, "numpy", types.SimpleNamespace(std=lambda values: 0.0))
    monkeypatch.setattr("src.trading.copy_trader.db.open_paper_trade", fake_open_paper_trade)

    from src.trading.copy_trader import CopyTrader

    trader = CopyTrader()
    price = 0.01234567
    leverage = 4
    trade = trader._open_copy_trade(
        {"balance": 10000.0},
        {
            "coin": "FART",
            "side": "long",
            "price": price,
            "leverage": leverage,
            "type": "copy_trade_signal",
            "confidence": 0.8,
            "source_trader": "0xabc",
        },
        [],
    )

    expected_stop_loss = price * (1 - 0.04 / leverage)
    expected_take_profit = price * (1 + 0.08 / leverage)

    assert trade is not None
    assert abs(opened["stop_loss"] - expected_stop_loss) < 1e-12
    assert abs(opened["take_profit"] - expected_take_profit) < 1e-12
    assert abs(trade["stop_loss"] - expected_stop_loss) < 1e-12
    assert abs(trade["take_profit"] - expected_take_profit) < 1e-12

    signal = signal_from_execution_dict(trade)
    assert abs(signal.risk.stop_loss_pct - (0.04 / leverage)) < 1e-12
    assert abs(signal.risk.take_profit_pct - (0.08 / leverage)) < 1e-12


def test_discovery_time_persistence_round_trips_through_db_context(monkeypatch):
    fd, raw_path = tempfile.mkstemp(prefix="bot_state_", suffix=".db", dir=os.getcwd())
    os.close(fd)
    db_path = os.path.abspath(raw_path)

    @contextmanager
    def fake_get_connection():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    monkeypatch.setattr(main.db, "get_connection", fake_get_connection)

    try:
        bot = main.HyperliquidResearchBot.__new__(main.HyperliquidResearchBot)
        bot._last_discovery = 12345.67

        bot._save_last_discovery_time()
        restored = bot._restore_last_discovery_time()

        assert restored == 12345.67
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
