"""
Unit tests for DecisionFirewall.
"""

import pytest
from unittest.mock import patch, MagicMock


class MockSignal:
    """Minimal mock of TradeSignal for testing."""
    def __init__(self, coin="BTC", side_val="long", confidence=0.5,
                 leverage=3, size=0.1, entry_price=50000,
                 position_pct=0.05, strategy_type="momentum",
                 source_accuracy=0.0):
        self.coin = coin
        self.side = MagicMock()
        self.side.value = side_val
        self.confidence = confidence
        self.leverage = leverage
        self.size = size
        self.entry_price = entry_price
        self.position_pct = position_pct
        self.strategy_type = strategy_type
        self.source_accuracy = source_accuracy
        self.regime_size_modifier = 1.0
        self.source = "test"

    def validate(self):
        return True


class _FakeScorer:
    def __init__(self, policy):
        self._policy = policy

    def get_source_policy(self, source_key):
        return {
            "source_key": source_key,
            "status": "active",
            "rank": 1,
            "blocked": False,
            "max_signals_per_day": 0,
            "size_multiplier": 1.0,
            "min_confidence": 0.0,
            "dynamic_weight": 0.6,
            "weighted_accuracy": 0.6,
            "completed_trades": 10,
            "recent_pnl": 0.0,
            **self._policy,
        }

    def get_scorecard(self):
        return [self.get_source_policy("test:momentum")]


class _FakeEventScanner:
    def __init__(self, payload):
        self.payload = payload

    def get_risk_state(self, asset=None):
        return dict(self.payload)


@patch("src.signals.decision_firewall.db")
def test_firewall_passes_valid_signal(mock_db):
    """Valid signal should pass all checks."""
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 1000000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall
    fw = DecisionFirewall({"enable_predictive_derisk": False, "funding_risk_enabled": False})
    signal = MockSignal(confidence=0.5)
    passed, reason = fw.validate(signal)
    assert passed is True
    assert reason == "approved"


@patch("src.signals.decision_firewall.db")
def test_firewall_rejects_low_confidence(mock_db):
    """Signals below min_confidence should be rejected."""
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 1000000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall
    fw = DecisionFirewall({"min_confidence": 0.3, "enable_predictive_derisk": False,
                           "funding_risk_enabled": False})
    signal = MockSignal(confidence=0.1)
    passed, reason = fw.validate(signal)
    assert passed is False
    assert "confidence" in reason.lower()


@patch("src.signals.decision_firewall.db")
def test_regime_size_modifier_scales_size_and_position_pct(mock_db):
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 1000000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall

    fw = DecisionFirewall({
        "enable_predictive_derisk": False,
        "funding_risk_enabled": False,
        "cooldown_seconds": 0,
        "same_side_cooldown_seconds": 0,
    })
    signal = MockSignal(confidence=0.8, size=2.0, position_pct=0.10)

    passed, reason = fw.validate(
        signal,
        regime_data={"strategy_guidance": {"size_modifier": 0.4}},
        open_positions=[],
        account_balance=1000000,
    )

    assert passed is True
    assert reason == "approved"
    assert signal.regime_size_modifier == 0.4
    assert signal.size == 0.8
    assert signal.position_pct == pytest.approx(0.04)


@patch("src.signals.decision_firewall.db")
def test_regime_disagreement_blocks_countertrend_side(mock_db):
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 1000000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall

    fw = DecisionFirewall({
        "enable_predictive_derisk": False,
        "funding_risk_enabled": False,
        "cooldown_seconds": 0,
        "same_side_cooldown_seconds": 0,
    })
    signal = MockSignal(side_val="short", confidence=0.8, strategy_type="options_momentum")

    passed, reason = fw.validate(
        signal,
        regime_data={
            "overall_regime": "volatile",
            "detector_regime": "trending_up",
            "forecaster_regime": "crash",
            "countertrend_block_side": "short",
            "strategy_guidance": {"size_modifier": 1.0, "pause": [], "activate": []},
        },
        open_positions=[],
        account_balance=1000000,
    )

    assert passed is False
    assert "countertrend" in reason.lower()


@patch("src.signals.decision_firewall.db")
def test_firewall_rejects_max_positions(mock_db):
    """Should reject when max positions reached."""
    mock_db.get_open_paper_trades.return_value = [
        {"coin": "ETH", "side": "long", "size": 1, "entry_price": 3000, "leverage": 1},
        {"coin": "SOL", "side": "long", "size": 1, "entry_price": 100, "leverage": 1},
        {"coin": "ARB", "side": "long", "size": 1, "entry_price": 1, "leverage": 1},
        {"coin": "DOGE", "side": "long", "size": 1, "entry_price": 0.1, "leverage": 1},
        {"coin": "AVAX", "side": "long", "size": 1, "entry_price": 30, "leverage": 1},
    ]
    mock_db.get_paper_account.return_value = {"balance": 1000000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall
    fw = DecisionFirewall({"max_positions": 5, "enable_predictive_derisk": False,
                           "funding_risk_enabled": False})
    signal = MockSignal(coin="BTC")
    passed, reason = fw.validate(signal)
    assert passed is False
    assert "max positions" in reason.lower()


@patch("src.signals.decision_firewall.db")
def test_firewall_can_preview_past_position_cap(mock_db):
    """Preview checks should ignore the hard cap without mutating stats or cooldowns."""
    mock_db.get_open_paper_trades.return_value = [
        {"coin": "ETH", "side": "long", "size": 1, "entry_price": 3000, "leverage": 1},
        {"coin": "SOL", "side": "long", "size": 1, "entry_price": 100, "leverage": 1},
        {"coin": "ARB", "side": "long", "size": 1, "entry_price": 1, "leverage": 1},
        {"coin": "DOGE", "side": "long", "size": 1, "entry_price": 0.1, "leverage": 1},
        {"coin": "AVAX", "side": "long", "size": 1, "entry_price": 30, "leverage": 1},
    ]
    mock_db.get_paper_account.return_value = {"balance": 1000000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall
    fw = DecisionFirewall({"max_positions": 5, "enable_predictive_derisk": False,
                           "funding_risk_enabled": False})
    signal = MockSignal(coin="BTC")

    passed, reason = fw.validate(signal, ignore_position_limit=True, dry_run=True)
    assert passed is True
    assert reason == "approved"
    assert fw.get_stats()["total_signals"] == 0


@patch("src.signals.decision_firewall.db")
def test_firewall_rejects_conflict(mock_db):
    """Should reject opposing positions on same coin."""
    mock_db.get_open_paper_trades.return_value = [
        {"coin": "BTC", "side": "short", "size": 1, "entry_price": 50000, "leverage": 1},
    ]
    mock_db.get_paper_account.return_value = {"balance": 1000000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall
    fw = DecisionFirewall({"enable_predictive_derisk": False, "funding_risk_enabled": False})
    signal = MockSignal(coin="BTC", side_val="long")
    passed, reason = fw.validate(signal)
    assert passed is False
    assert "conflict" in reason.lower()


@patch("src.signals.decision_firewall.db")
def test_firewall_clamps_leverage(mock_db):
    """Leverage above max should be clamped, not rejected."""
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 1000000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall
    fw = DecisionFirewall({"max_leverage": 5, "enable_predictive_derisk": False,
                           "funding_risk_enabled": False})
    signal = MockSignal(leverage=10)
    passed, reason = fw.validate(signal)
    assert passed is True
    assert signal.leverage == 5


@patch("src.signals.decision_firewall.db")
def test_firewall_cooldown(mock_db):
    """Should enforce cooldown between trades on same coin."""
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 10000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall
    fw = DecisionFirewall({"cooldown_seconds": 300, "enable_predictive_derisk": False,
                           "funding_risk_enabled": False})

    signal1 = MockSignal(coin="ETH")
    passed1, _ = fw.validate(signal1)
    assert passed1 is True

    signal2 = MockSignal(coin="ETH")
    passed2, reason2 = fw.validate(signal2)
    assert passed2 is False
    assert "cooldown" in reason2.lower()


@patch("src.signals.decision_firewall.db")
def test_firewall_same_side_cooldown_blocks_pyramiding(mock_db):
    """Same-side re-entries on the same coin should respect the anti-pyramiding cooldown."""
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 10000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall

    fw = DecisionFirewall(
        {
            "cooldown_seconds": 0,
            "same_side_cooldown_seconds": 900,
            "enable_predictive_derisk": False,
            "funding_risk_enabled": False,
        }
    )

    passed1, _ = fw.validate(MockSignal(coin="BTC", side_val="short", confidence=0.7))
    passed2, reason2 = fw.validate(MockSignal(coin="BTC", side_val="short", confidence=0.72))

    assert passed1 is True
    assert passed2 is False
    assert "pyramiding" in reason2.lower()
    assert fw.get_stats()["rejected_pyramiding"] == 1


@patch("src.signals.decision_firewall.db")
def test_firewall_same_side_position_limit_blocks_stacking(mock_db):
    """The firewall should stop stacking beyond the same-side per-coin limit."""
    mock_db.get_open_paper_trades.return_value = [
        {"coin": "ETH", "side": "short", "size": 1, "entry_price": 2000, "leverage": 2},
        {"coin": "ETH", "side": "short", "size": 1, "entry_price": 1990, "leverage": 2},
    ]
    mock_db.get_paper_account.return_value = {"balance": 10000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall

    fw = DecisionFirewall(
        {
            "max_same_side_positions_per_coin": 2,
            "enable_predictive_derisk": False,
            "funding_risk_enabled": False,
        }
    )

    passed, reason = fw.validate(MockSignal(coin="ETH", side_val="short", confidence=0.8))

    assert passed is False
    assert "pyramiding" in reason.lower()


@patch("src.signals.decision_firewall.db")
def test_firewall_stats(mock_db):
    """Stats should track rejections accurately."""
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 10000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall
    fw = DecisionFirewall({"enable_predictive_derisk": False, "funding_risk_enabled": False})

    fw.validate(MockSignal(confidence=0.5))
    fw.validate(MockSignal(coin="ETH", confidence=0.05))

    stats = fw.get_stats()
    assert stats["total_signals"] == 2
    assert stats["passed"] == 1
    assert stats["rejected_confidence"] == 1


@patch("src.signals.decision_firewall.db")
def test_firewall_enforces_per_source_day_cap(mock_db):
    """Approved signals per source should stop at configured daily cap."""
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 10000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall

    fw = DecisionFirewall(
        {
            "max_signals_per_source_per_day": 1,
            "enable_predictive_derisk": False,
            "funding_risk_enabled": False,
        }
    )
    first = MockSignal(coin="BTC", confidence=0.6)
    first.source = "copy_trade"
    second = MockSignal(coin="ETH", confidence=0.6)
    second.source = "copy_trade"

    passed1, _ = fw.validate(first)
    passed2, reason2 = fw.validate(second)

    assert passed1 is True
    assert passed2 is False
    assert "source/day cap" in reason2.lower()
    stats = fw.get_stats()
    assert stats["rejected_source_cap"] == 1
    assert stats["source_signal_counts"]["copy_trade"] == 1


@patch("src.signals.decision_firewall.db")
def test_firewall_canary_mode_tightens_position_limit(mock_db):
    """Canary mode should force max_positions down to canary max."""
    mock_db.get_open_paper_trades.return_value = [
        {"coin": "ETH", "side": "long", "size": 1, "entry_price": 3000, "leverage": 1},
        {"coin": "SOL", "side": "long", "size": 1, "entry_price": 100, "leverage": 1},
    ]
    mock_db.get_paper_account.return_value = {"balance": 10000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall

    fw = DecisionFirewall(
        {
            "max_positions": 8,
            "canary_mode": True,
            "canary_max_positions": 2,
            "enable_predictive_derisk": False,
            "funding_risk_enabled": False,
        }
    )
    passed, reason = fw.validate(MockSignal(coin="BTC", confidence=0.8))
    assert passed is False
    assert "max positions" in reason.lower()


@patch("src.signals.decision_firewall.db")
def test_firewall_rejects_paused_source_policy(mock_db):
    """Allocator-paused sources should be blocked before consuming capacity."""
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 10000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall

    fw = DecisionFirewall(
        {
            "agent_scorer": _FakeScorer({"status": "paused", "blocked": True}),
            "enable_predictive_derisk": False,
            "funding_risk_enabled": False,
        }
    )
    signal = MockSignal(confidence=0.9)
    signal.source = "strategy"

    passed, reason = fw.validate(signal)

    assert passed is False
    assert "allocator paused" in reason.lower()
    assert fw.get_stats()["rejected_source_policy"] == 1


@patch("src.signals.decision_firewall.db")
def test_firewall_derisks_degraded_sources_and_uses_policy_cap(mock_db):
    """Degraded sources should trade smaller and respect the tighter policy cap."""
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 10000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall

    fw = DecisionFirewall(
        {
            "max_signals_per_source_per_day": 2,
            "agent_scorer": _FakeScorer(
                {
                    "status": "degraded",
                    "max_signals_per_day": 1,
                    "size_multiplier": 0.5,
                    "min_confidence": 0.55,
                    "dynamic_weight": 0.18,
                    "recent_pnl": -25.0,
                }
            ),
            "enable_predictive_derisk": False,
            "funding_risk_enabled": False,
        }
    )

    first = MockSignal(coin="BTC", confidence=0.60, size=0.2, position_pct=0.10)
    first.source = "strategy"
    second = MockSignal(coin="ETH", confidence=0.60, size=0.2, position_pct=0.10)
    second.source = "strategy"

    passed1, _ = fw.validate(first)
    passed2, reason2 = fw.validate(second)

    assert passed1 is True
    assert first.size == 0.1
    assert first.position_pct == 0.05
    assert passed2 is False
    assert "source/day cap" in reason2.lower()


@patch("src.signals.decision_firewall.db")
def test_firewall_halves_confidence_when_predictive_inputs_are_partial(mock_db):
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 10000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall

    class _PartialForecaster:
        def predict_regime(self, coin):
            return {
                "signal": 0.0,
                "regime": "neutral",
                "confidence": 0.1,
                "components": {},
                "active_inputs": ["funding_slope", "imbalance"],
                "active_input_count": 2,
                "partial_signal": True,
                "partial_inputs": ["polymarket"],
            }

    fw = DecisionFirewall(
        {
            "min_confidence": 0.3,
            "forecaster": _PartialForecaster(),
            "enable_predictive_derisk": True,
            "funding_risk_enabled": False,
        }
    )
    signal = MockSignal(confidence=0.5)

    passed, reason = fw.validate(signal)

    assert passed is False
    assert signal.confidence == 0.25
    assert "low confidence" in reason.lower()


@patch("src.signals.decision_firewall.db")
def test_firewall_blocks_on_active_event_risk(mock_db):
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 10000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall

    fw = DecisionFirewall(
        {
            "event_scanner": _FakeEventScanner(
                {
                    "enabled": True,
                    "block_new_entries": True,
                    "degrade": False,
                    "confidence_multiplier": 1.0,
                    "size_multiplier": 1.0,
                    "reasons": ["Active critical incident: Coinbase Status API outage"],
                }
            ),
            "enable_predictive_derisk": False,
            "funding_risk_enabled": False,
        }
    )

    passed, reason = fw.validate(MockSignal(confidence=0.8))

    assert passed is False
    assert "incident" in reason.lower()
    assert fw.get_stats()["rejected_event_risk"] == 1


@patch("src.signals.decision_firewall.db")
def test_firewall_derisks_on_upcoming_event_window(mock_db):
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 10000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall

    fw = DecisionFirewall(
        {
            "min_confidence": 0.3,
            "event_scanner": _FakeEventScanner(
                {
                    "enabled": True,
                    "block_new_entries": False,
                    "degrade": True,
                    "confidence_multiplier": 0.6,
                    "size_multiplier": 0.5,
                    "reasons": ["High-impact release ahead: CPI (20m)"],
                }
            ),
            "enable_predictive_derisk": False,
            "funding_risk_enabled": False,
        }
    )
    signal = MockSignal(confidence=0.8, size=0.2, position_pct=0.1)

    passed, reason = fw.validate(signal)

    assert passed is True
    assert reason == "approved"
    assert signal.confidence == 0.48
    assert signal.size == 0.1
    assert signal.position_pct == 0.05


@patch("src.signals.decision_firewall.db")
def test_firewall_blocks_shorts_when_recent_shorts_are_bad(mock_db):
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 10000}
    mock_db.get_paper_trade_history.return_value = [
        {"side": "short", "pnl": -0.7, "metadata": "{}"},
        {"side": "short", "pnl": -0.4, "metadata": "{}"},
        {"side": "short", "pnl": -0.2, "metadata": "{}"},
        {"side": "short", "pnl": 0.1, "metadata": "{}"},
    ]
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall

    fw = DecisionFirewall(
        {
            "short_hardening_enabled": True,
            "short_hardening_min_closed_trades": 4,
            "short_hardening_block_win_rate": 0.35,
            "short_hardening_block_net_pnl": -1.0,
            "enable_predictive_derisk": False,
            "funding_risk_enabled": False,
        }
    )
    signal = MockSignal(side_val="short", confidence=0.8)

    passed, reason = fw.validate(signal)

    assert passed is False
    assert "underperforming" in reason.lower()
    assert fw.get_stats()["rejected_side_policy"] == 1


@patch("src.signals.decision_firewall.db")
def test_firewall_derisks_shorts_when_recent_shorts_need_caution(mock_db):
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 10000}
    mock_db.get_paper_trade_history.return_value = [
        {"side": "short", "pnl": -0.3, "metadata": "{}"},
        {"side": "short", "pnl": -0.2, "metadata": "{}"},
        {"side": "short", "pnl": 0.4, "metadata": "{}"},
        {"side": "short", "pnl": -0.1, "metadata": "{}"},
    ]
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall

    fw = DecisionFirewall(
        {
            "short_hardening_enabled": True,
            "short_hardening_min_closed_trades": 4,
            "short_hardening_degrade_win_rate": 0.45,
            "short_hardening_block_win_rate": 0.20,
            "short_hardening_block_net_pnl": -5.0,
            "short_hardening_confidence_multiplier": 0.75,
            "short_hardening_size_multiplier": 0.5,
            "min_confidence": 0.3,
            "enable_predictive_derisk": False,
            "funding_risk_enabled": False,
        }
    )
    signal = MockSignal(side_val="short", confidence=0.8, size=0.2, position_pct=0.1)

    passed, reason = fw.validate(signal)

    assert passed is True
    assert reason == "approved"
    assert signal.confidence == pytest.approx(0.6)
    assert signal.size == 0.1
    assert signal.position_pct == 0.05


@patch("src.signals.decision_firewall.db")
def test_validate_batch_counts_projected_notional_for_signals_without_size(mock_db):
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 1000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall

    fw = DecisionFirewall(
        {
            "max_aggregate_exposure": 1.0,
            "enable_predictive_derisk": False,
            "funding_risk_enabled": False,
        }
    )

    first = MockSignal(
        coin="BTC",
        confidence=0.8,
        size=0.0,
        entry_price=100.0,
        leverage=2,
        position_pct=0.30,
    )
    second = MockSignal(
        coin="ETH",
        confidence=0.7,
        size=0.0,
        entry_price=100.0,
        leverage=2,
        position_pct=0.30,
    )

    results = fw.validate_batch([first, second])

    assert results[0][1] is True
    assert first.size == pytest.approx(3.0)
    assert results[1][1] is False
    assert "aggregate exposure" in results[1][2].lower()


# ─────────────────────────────────────────────────────────────────
# AUDIT M1 — aggregate margin cap is an independent, leverage-agnostic
# capital-at-risk check that runs alongside the legacy leveraged-notional
# exposure cap.  These tests cover:
#   - margin cap rejects when exposure cap would allow
#   - exposure cap rejects when margin cap would allow
#   - disabling margin cap (<=0) reverts to legacy-only behavior
#   - both caps approved = signal approved
# ─────────────────────────────────────────────────────────────────


@patch("src.signals.decision_firewall.db")
def test_firewall_margin_cap_rejects_when_notional_cap_allows(mock_db):
    """High-leverage signal: notional cap passes, margin cap blocks."""
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 1000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall

    # Generous notional cap (200%), strict margin cap (10%).
    # Signal with size=7, price=$100, leverage=5:
    #   - leveraged notional = 7 × 100 × 5 = 3500 → 350% → blows notional cap
    # So instead pick size=3 at 5x leverage:
    #   - leveraged notional = 3 × 100 × 5 = 1500 → 150% (fits 200% cap)
    #   - margin            = 3 × 100 / 5 = 60    →  6% (fits 10% cap)
    # Then size=6 at 5x leverage:
    #   - leveraged notional = 6 × 100 × 5 = 3000 → 300% (fails 200%)
    # To isolate margin cap we need exposure cap forgiving AND margin cap tight.
    fw = DecisionFirewall(
        {
            "max_aggregate_exposure": 10.0,      # effectively disabled
            "max_aggregate_margin_pct": 0.10,    # 10% margin cap
            "enable_predictive_derisk": False,
            "funding_risk_enabled": False,
            "min_confidence": 0.2,
        }
    )

    # Size=3 @ $100 @ 5x lev → notional=300, margin=60 → 60/1000 = 6% (passes)
    ok_signal = MockSignal(
        coin="BTC", confidence=0.8,
        size=3.0, entry_price=100.0, leverage=5, position_pct=0.0,
    )
    passed, reason = fw.validate(ok_signal)
    assert passed is True, f"Expected approval, got: {reason}"

    # Size=6 @ $100 @ 5x lev → notional=600, margin=120 → 120/1000 = 12% (fails 10%)
    # With NO existing positions to add to.  Exposure = 600/1000 = 60% (fits 1000% cap).
    mock_db.get_open_paper_trades.return_value = []  # reset — no persisted positions
    big_signal = MockSignal(
        coin="ETH", confidence=0.8,
        size=6.0, entry_price=100.0, leverage=5, position_pct=0.0,
    )
    passed, reason = fw.validate(big_signal)
    assert passed is False
    assert "aggregate margin" in reason.lower()


@patch("src.signals.decision_firewall.db")
def test_firewall_notional_cap_still_rejects_when_margin_cap_allows(mock_db):
    """Low-leverage high-notional signal: margin cap passes, notional cap blocks."""
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 1000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall

    # Tight notional cap (50% leveraged), generous margin cap (80%).
    # Size=7 @ $100 @ 1x lev → notional (leveraged) = 700, margin = 700
    #   700/1000 = 70% → fails 50% exposure cap
    #   700/1000 = 70% → fits 80% margin cap
    fw = DecisionFirewall(
        {
            "max_aggregate_exposure": 0.50,
            "max_aggregate_margin_pct": 0.80,
            "enable_predictive_derisk": False,
            "funding_risk_enabled": False,
            "min_confidence": 0.2,
        }
    )
    signal = MockSignal(
        coin="BTC", confidence=0.8,
        size=7.0, entry_price=100.0, leverage=1, position_pct=0.0,
    )
    passed, reason = fw.validate(signal)
    assert passed is False
    assert "aggregate exposure" in reason.lower()
    # The margin-cap branch should NOT appear in the rejection reason.
    assert "aggregate margin" not in reason.lower()


@patch("src.signals.decision_firewall.db")
def test_firewall_margin_cap_disabled_when_nonpositive(mock_db):
    """max_aggregate_margin_pct <= 0 turns off the margin cap entirely."""
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 1000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall

    fw = DecisionFirewall(
        {
            "max_aggregate_exposure": 10.0,      # effectively disabled
            "max_aggregate_margin_pct": 0.0,     # disabled via 0
            "enable_predictive_derisk": False,
            "funding_risk_enabled": False,
            "min_confidence": 0.2,
        }
    )
    # size=9 @ $100 @ 5x lev → margin 180 (18% of balance).  With cap=0, allowed.
    signal = MockSignal(
        coin="BTC", confidence=0.8,
        size=9.0, entry_price=100.0, leverage=5, position_pct=0.0,
    )
    passed, reason = fw.validate(signal)
    assert passed is True, f"Expected approval with margin cap disabled, got: {reason}"


@patch("src.signals.decision_firewall.db")
def test_firewall_both_caps_approve_signal(mock_db):
    """Both caps satisfied → signal is approved."""
    mock_db.get_open_paper_trades.return_value = []
    mock_db.get_paper_account.return_value = {"balance": 1000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall

    fw = DecisionFirewall(
        {
            "max_aggregate_exposure": 1.50,
            "max_aggregate_margin_pct": 0.60,
            "enable_predictive_derisk": False,
            "funding_risk_enabled": False,
            "min_confidence": 0.2,
        }
    )
    # size=2 @ $100 @ 3x lev → notional 600 (60% of balance), margin 67 (6.7%)
    signal = MockSignal(
        coin="BTC", confidence=0.8,
        size=2.0, entry_price=100.0, leverage=3, position_pct=0.0,
    )
    passed, reason = fw.validate(signal)
    assert passed is True, f"Expected approval, got: {reason}"


@patch("src.signals.decision_firewall.db")
def test_firewall_margin_cap_aggregates_over_existing_positions(mock_db):
    """Margin cap must sum across already-open positions (per-position margin)."""
    mock_db.get_paper_account.return_value = {"balance": 1000}
    mock_db.audit_log = MagicMock()

    from src.signals.decision_firewall import DecisionFirewall

    # Three existing 5x-leveraged positions, each with margin = 20 (= 2%),
    # leveraged notional = 500 each → total leveraged notional = 1500.
    # Margin cap = 40% → total margin budget = 400.
    mock_db.get_open_paper_trades.return_value = [
        {"coin": "BTC", "side": "long", "size": 1, "entry_price": 100, "leverage": 5},
        {"coin": "ETH", "side": "long", "size": 1, "entry_price": 100, "leverage": 5},
        {"coin": "SOL", "side": "long", "size": 1, "entry_price": 100, "leverage": 5},
    ]

    # Exposure cap set very loose (100.0 = 10000% of balance) so the leveraged-
    # notional guard won't fire — we isolate the margin-cap behavior.
    fw = DecisionFirewall(
        {
            "max_aggregate_exposure": 100.0,      # effectively disabled
            "max_aggregate_margin_pct": 0.40,
            "max_positions": 8,
            "enable_predictive_derisk": False,
            "funding_risk_enabled": False,
            "min_confidence": 0.2,
        }
    )
    # Candidate 1: size=2.5 @ 5x → margin 50 (5%).  Total projected = 60 + 50 = 110 (11%) → passes 40%.
    small = MockSignal(
        coin="DOGE", confidence=0.8,
        size=2.5, entry_price=100.0, leverage=5, position_pct=0.0,
    )
    passed, reason = fw.validate(small)
    assert passed is True, f"Expected approval, got: {reason}"

    # Candidate 2: size=20 @ 5x → margin 400 (40%).  Total projected = 60 + 400 = 460 (46%) → fails 40%.
    big = MockSignal(
        coin="AVAX", confidence=0.8,
        size=20.0, entry_price=100.0, leverage=5, position_pct=0.0,
    )
    passed, reason = fw.validate(big)
    assert passed is False
    assert "aggregate margin" in reason.lower()
