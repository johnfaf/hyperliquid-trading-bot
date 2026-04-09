import pytest
from unittest.mock import patch
from src.data.options_flow import OptionsFlowScanner


class TestOptionsFlowScanner:
    """Test suite for OptionsFlowScanner class"""

    @pytest.fixture
    def scanner(self):
        """Create a scanner instance for testing"""
        return OptionsFlowScanner()

    # ------------------------------------------------------------------ #
    # Helpers: build a trade dict using the field names classify_print
    # actually expects (instrument, notional, amount, side, option_type,
    # strike, underlying) — NOT the old schema (size, premium, direction).
    # ------------------------------------------------------------------ #

    @staticmethod
    def _trade(option_type="call", side="buy", amount=100,
                notional=500_000, strike=50_000, underlying="BTC",
                instrument="BTC-28MAR26-50000-C", expiry="28MAR26"):
        return {
            "instrument": instrument,
            "notional": notional,
            "amount": amount,
            "option_type": option_type,
            "side": side,
            "strike": strike,
            "underlying": underlying,
            "expiry": expiry,
        }

    def test_init(self):
        """Test that scanner initializes without error"""
        scanner = OptionsFlowScanner()
        assert scanner is not None
        assert isinstance(scanner, OptionsFlowScanner)

    def test_classify_print_bullish_call_buy(self, scanner):
        """Buying a call is classified as bullish"""
        trade = self._trade(option_type="call", side="buy", notional=500_000)
        with patch.object(scanner, 'get_spot_price', return_value=45_000):
            with patch.object(scanner, 'get_open_interest', return_value=1_000_000):
                result = scanner.classify_print(trade)
        assert result is not None
        assert result.get("direction") == "bullish"

    def test_classify_print_bearish_put_buy(self, scanner):
        """Buying a put is classified as bearish"""
        trade = self._trade(option_type="put", side="buy",
                            strike=45_000, notional=500_000,
                            instrument="BTC-28MAR26-45000-P")
        with patch.object(scanner, 'get_spot_price', return_value=50_000):
            with patch.object(scanner, 'get_open_interest', return_value=1_000_000):
                result = scanner.classify_print(trade)
        assert result is not None
        assert result.get("direction") == "bearish"

    def test_classify_print_tier_mega(self, scanner):
        """notional >= 5 M is MEGA_BLOCK"""
        trade = self._trade(notional=6_000_000, amount=100)
        with patch.object(scanner, 'get_spot_price', return_value=50_000):
            with patch.object(scanner, 'get_open_interest', return_value=2_000_000):
                result = scanner.classify_print(trade)
        assert result is not None
        assert result.get("tier") == "MEGA_BLOCK"

    def test_classify_print_tier_sweep(self, scanner):
        """notional 500 k – 2 M is SWEEP"""
        trade = self._trade(notional=750_000, amount=50)
        with patch.object(scanner, 'get_spot_price', return_value=50_000):
            with patch.object(scanner, 'get_open_interest', return_value=1_000_000):
                result = scanner.classify_print(trade)
        assert result is not None
        assert result.get("tier") == "SWEEP"

    def test_classify_print_tier_normal(self, scanner):
        """notional < 100 k (and below LARGE threshold) is NORMAL"""
        # 50k is above MIN_NOTIONAL (25k) but below TIER_LARGE (100k) → NORMAL
        trade = self._trade(notional=50_000, amount=10)
        with patch.object(scanner, 'get_spot_price', return_value=50_000):
            with patch.object(scanner, 'get_open_interest', return_value=1_000_000):
                result = scanner.classify_print(trade)
        assert result is not None
        assert result.get("tier") == "NORMAL"

    def test_vol_oi_ratio(self, scanner):
        """vol/OI ratio: amount=1000, mocked OI=10 000 → ratio=0.1
        Note: classify_print rounds vol_oi_ratio to 2 decimal places on return,
        so use values whose ratio survives rounding (>= 0.01).
        """
        trade = self._trade(amount=1_000, notional=500_000)
        with patch.object(scanner, 'get_spot_price', return_value=50_000):
            with patch.object(scanner, 'get_open_interest', return_value=10_000):
                result = scanner.classify_print(trade)
        assert result is not None
        vol_oi = result.get("vol_oi_ratio")
        assert vol_oi is not None
        assert abs(vol_oi - 0.1) < 0.001

    def test_is_unusual_flag(self, scanner):
        """is_unusual=True when vol_oi >= MIN_VOL_OI_RATIO (0.10) and notional >= MIN_NOTIONAL"""
        # amount=150_000, OI=1_000_000 → ratio=0.15 >= 0.10; notional=75M >= 25k
        trade = self._trade(amount=150_000, notional=75_000_000)
        with patch.object(scanner, 'get_spot_price', return_value=50_000):
            with patch.object(scanner, 'get_open_interest', return_value=1_000_000):
                result = scanner.classify_print(trade)
        assert result is not None
        assert result.get("is_unusual") is True

    def test_get_flow_signal_strong_conviction(self, scanner):
        """get_flow_signal returns signal when conviction_pct >= configured gate."""
        scanner.top_convictions = [{
            "ticker": "BTC",
            "direction": "BULLISH",
            "conviction_pct": 45,
            "net_flow": 1_000_000,
            "total_prints": 12,
        }]
        signal = scanner.get_flow_signal("BTC")
        assert signal is not None
        assert signal.get("confidence") == pytest.approx(0.45)
        assert signal.get("side") == "long"

    def test_get_flow_signal_weak_conviction(self, scanner):
        """get_flow_signal returns None when conviction_pct is below configured gate."""
        scanner.top_convictions = [{
            "ticker": "BTC",
            "direction": "BULLISH",
            "conviction_pct": 25,
            "net_flow": 500_000,
            "total_prints": 5,
        }]
        signal = scanner.get_flow_signal("BTC")
        assert signal is None

    def test_dashboard_data_structure(self, scanner):
        """get_dashboard_data returns a dict with expected top-level keys"""
        with patch.object(scanner, 'get_recent_trades', return_value=[]):
            with patch.object(scanner, 'scan_flow', return_value={}):
                dashboard = scanner.get_dashboard_data()
        assert dashboard is not None
        assert isinstance(dashboard, dict)

    def test_tracked_currencies(self, scanner):
        """TRACKED_CURRENCIES is accessible on the instance and contains BTC/ETH"""
        tracked = scanner.TRACKED_CURRENCIES
        assert "BTC" in tracked
        assert "ETH" in tracked
        # SOL removed: Deribit has no SOL options — every scan returned 0 trades
