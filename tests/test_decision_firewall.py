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
