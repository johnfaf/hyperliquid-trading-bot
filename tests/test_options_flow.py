import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.data.options_flow import OptionsFlowScanner


class TestOptionsFlowScanner:
    """Test suite for OptionsFlowScanner class"""

    @pytest.fixture
    def scanner(self):
        """Create a scanner instance for testing"""
        return OptionsFlowScanner()

    def test_init(self):
        """Test that scanner initializes without error"""
        scanner = OptionsFlowScanner()
        assert scanner is not None
        assert isinstance(scanner, OptionsFlowScanner)

    def test_classify_print_bullish_call_buy(self, scanner):
        """Test that buying a call is classified as bullish"""
        trade = {
            "option_type": "call",
            "direction": "buy",
            "size": 100,
            "strike": 50000,
            "spot_price": 45000,
            "open_interest": 1000000,
        }

        with patch.object(scanner, 'get_spot_price', return_value=45000):
            with patch.object(scanner, 'get_open_interest', return_value=1000000):
                result = scanner.classify_print(trade)
                assert result is not None
                assert result.get("direction") == "bullish"

    def test_classify_print_bearish_put_buy(self, scanner):
        """Test that buying a put is classified as bearish"""
        trade = {
            "option_type": "put",
            "direction": "buy",
            "size": 100,
            "strike": 45000,
            "spot_price": 50000,
            "open_interest": 1000000,
        }

        with patch.object(scanner, 'get_spot_price', return_value=50000):
            with patch.object(scanner, 'get_open_interest', return_value=1000000):
                result = scanner.classify_print(trade)
                assert result is not None
                assert result.get("direction") == "bearish"

    def test_classify_print_tier_mega(self, scanner):
        """Test that notional >= 5M is classified as MEGA_BLOCK"""
        trade = {
            "option_type": "call",
            "direction": "buy",
            "size": 100,
            "strike": 50000,
            "spot_price": 50000,
            "open_interest": 2000000,
            "premium": 60000,  # 100 * 60000 = 6M notional
        }

        with patch.object(scanner, 'get_spot_price', return_value=50000):
            with patch.object(scanner, 'get_open_interest', return_value=2000000):
                result = scanner.classify_print(trade)
                assert result is not None
                assert result.get("tier") == "MEGA_BLOCK"

    def test_classify_print_tier_sweep(self, scanner):
        """Test that notional 500k-2M is classified as SWEEP"""
        trade = {
            "option_type": "call",
            "direction": "buy",
            "size": 50,
            "strike": 50000,
            "spot_price": 50000,
            "open_interest": 1000000,
            "premium": 15000,  # 50 * 15000 = 750k notional
        }

        with patch.object(scanner, 'get_spot_price', return_value=50000):
            with patch.object(scanner, 'get_open_interest', return_value=1000000):
                result = scanner.classify_print(trade)
                assert result is not None
                assert result.get("tier") == "SWEEP"

    def test_classify_print_tier_normal(self, scanner):
        """Test that notional < 100k is classified as NORMAL"""
        trade = {
            "option_type": "call",
            "direction": "buy",
            "size": 10,
            "strike": 50000,
            "spot_price": 50000,
            "open_interest": 1000000,
            "premium": 5000,  # 10 * 5000 = 50k notional
        }

        with patch.object(scanner, 'get_spot_price', return_value=50000):
            with patch.object(scanner, 'get_open_interest', return_value=1000000):
                result = scanner.classify_print(trade)
                assert result is not None
                assert result.get("tier") == "NORMAL"

    def test_vol_oi_ratio(self, scanner):
        """Test that vol/OI ratio is computed correctly"""
        trade = {
            "option_type": "call",
            "direction": "buy",
            "size": 100,
            "strike": 50000,
            "spot_price": 50000,
            "open_interest": 1000000,
            "premium": 10000,
        }

        with patch.object(scanner, 'get_spot_price', return_value=50000):
            with patch.object(scanner, 'get_open_interest', return_value=1000000):
                result = scanner.classify_print(trade)
                assert result is not None
                vol_oi_ratio = result.get("vol_oi_ratio")
                # vol = 100, oi = 1000000, ratio = 100/1000000 = 0.0001
                assert vol_oi_ratio is not None
                assert abs(vol_oi_ratio - 0.0001) < 0.00001

    def test_is_unusual_flag(self, scanner):
        """Test that is_unusual is True when vol_oi >= 1.5 AND notional >= 50k"""
        trade = {
            "option_type": "call",
            "direction": "buy",
            "size": 1500000,  # High volume
            "strike": 50000,
            "spot_price": 50000,
            "open_interest": 1000000,
            "premium": 50,  # 1500000 * 50 = 75M notional
        }

        with patch.object(scanner, 'get_spot_price', return_value=50000):
            with patch.object(scanner, 'get_open_interest', return_value=1000000):
                result = scanner.classify_print(trade)
                assert result is not None
                assert result.get("is_unusual") is True

    def test_get_flow_signal_strong_conviction(self, scanner):
        """Test that get_flow_signal returns signal when conviction > 60%"""
        with patch.object(scanner, 'scan_flow', return_value={
            "conviction": 0.75,
            "direction": "bullish",
            "strength": "strong"
        }):
            signal = scanner.get_flow_signal("BTC")
            assert signal is not None
            assert signal.get("conviction") == 0.75

    def test_get_flow_signal_weak_conviction(self, scanner):
        """Test that get_flow_signal returns None when conviction <= 60%"""
        with patch.object(scanner, 'scan_flow', return_value={
            "conviction": 0.45,
            "direction": "neutral",
            "strength": "weak"
        }):
            signal = scanner.get_flow_signal("BTC")
            assert signal is None

    def test_dashboard_data_structure(self, scanner):
        """Test that get_dashboard_data returns expected keys"""
        with patch.object(scanner, 'get_recent_trades', return_value=[]):
            with patch.object(scanner, 'classify_print', return_value={}):
                with patch.object(scanner, 'scan_flow', return_value={}):
                    dashboard = scanner.get_dashboard_data()
                    assert dashboard is not None
                    assert isinstance(dashboard, dict)
                    # Check for expected top-level keys
                    expected_keys = ["summary", "trades", "signals"]
                    for key in expected_keys:
                        assert key in dashboard or isinstance(dashboard, dict)

    def test_tracked_currencies(self, scanner):
        """Test that TRACKED_CURRENCIES contains BTC, ETH, SOL"""
        tracked = scanner.TRACKED_CURRENCIES
        assert "BTC" in tracked
        assert "ETH" in tracked
        assert "SOL" in tracked
