"""
Unit tests for DecisionFirewall.
"""

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
