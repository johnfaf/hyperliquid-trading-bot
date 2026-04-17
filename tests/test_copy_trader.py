"""
Tests for CopyTrader — V2 copy trading engine.
Covers:
  - Confidence model (_calculate_signal_confidence)
  - Regime weighting (_apply_regime_weight)
  - Metadata deduplication (_build_trade_metadata)
  - Position cache TTL eviction (scan_top_traders)
  - Silent risk policy failure handling (_resolve_copy_trade_risk)
"""
import time
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.trading.copy_trader import (
    CopyTrader,
    _SIGNAL_CONFIDENCE_MODEL,
    _CRASH_COPY_CONFIDENCE_MULTIPLIER,
    _NEUTRAL_COPY_CONFIDENCE_MULTIPLIER,
    _BULLISH_COPY_CONFIDENCE_MULTIPLIER,
    _POSITION_CACHE_TTL_SECONDS,
)


# ─── Confidence model ────────────────────────────────────────────

class TestCalculateSignalConfidence:
    def test_copy_open_perfect_trader(self):
        """100% win rate brings open confidence up to cap."""
        conf = CopyTrader._calculate_signal_confidence("copy_open", 1.0)
        expected = min(_SIGNAL_CONFIDENCE_MODEL["copy_open"]["max"], 0.50 + 0.50)
        assert abs(conf - expected) < 0.001

    def test_copy_open_zero_win_rate(self):
        """0% win rate returns base confidence."""
        conf = CopyTrader._calculate_signal_confidence("copy_open", 0.0)
        assert abs(conf - 0.50) < 0.001

    def test_copy_flip_higher_base_than_open(self):
        """Flip signals have higher base than open signals."""
        flip = CopyTrader._calculate_signal_confidence("copy_flip", 0.5)
        open_ = CopyTrader._calculate_signal_confidence("copy_open", 0.5)
        assert flip > open_, "Flip should have higher confidence than open at same win rate"

    def test_copy_scale_in_lower_base_than_open(self):
        """Scale-in signals have lower base confidence."""
        scale = CopyTrader._calculate_signal_confidence("copy_scale_in", 0.5)
        open_ = CopyTrader._calculate_signal_confidence("copy_open", 0.5)
        assert scale < open_, "Scale-in should have lower confidence than open"

    def test_caps_are_respected(self):
        """Even a perfect trader cannot exceed the max cap."""
        for sig_type, model in _SIGNAL_CONFIDENCE_MODEL.items():
            conf = CopyTrader._calculate_signal_confidence(sig_type, 1.0)
            assert conf <= model["max"] + 1e-6, f"{sig_type} exceeded its cap"

    def test_unknown_signal_type_defaults_safely(self):
        """Unknown signal types fall back to base 0.50 / cap 0.90."""
        conf = CopyTrader._calculate_signal_confidence("unknown_type", 0.5)
        assert 0.0 < conf <= 1.0

    def test_win_rate_capped_at_1(self):
        """Win rate > 1.0 (bad data) doesn't push confidence over cap."""
        conf = CopyTrader._calculate_signal_confidence("copy_open", 5.0)
        assert conf <= _SIGNAL_CONFIDENCE_MODEL["copy_open"]["max"] + 1e-6


# ─── Regime weighting ────────────────────────────────────────────

class TestApplyRegimeWeight:
    def _make_trader(self):
        ct = CopyTrader()
        ct.regime_forecaster = MagicMock()
        return ct

    def test_crash_reduces_confidence(self):
        ct = self._make_trader()
        ct.regime_forecaster.predict_regime.return_value = {"regime": "crash"}
        signal = {"confidence": 0.80}
        result = ct._apply_regime_weight(signal, "BTC")
        expected = 0.80 * _CRASH_COPY_CONFIDENCE_MULTIPLIER
        assert abs(result["confidence"] - expected) < 0.001

    def test_neutral_reduces_confidence(self):
        ct = self._make_trader()
        ct.regime_forecaster.predict_regime.return_value = {"regime": "neutral"}
        signal = {"confidence": 0.80}
        result = ct._apply_regime_weight(signal, "BTC")
        expected = 0.80 * _NEUTRAL_COPY_CONFIDENCE_MULTIPLIER
        assert abs(result["confidence"] - expected) < 0.001

    def test_bullish_boosts_confidence(self):
        ct = self._make_trader()
        ct.regime_forecaster.predict_regime.return_value = {"regime": "bullish"}
        signal = {"confidence": 0.80}
        result = ct._apply_regime_weight(signal, "BTC")
        expected = min(0.80 * _BULLISH_COPY_CONFIDENCE_MULTIPLIER, 1.0)
        assert abs(result["confidence"] - expected) < 0.001

    def test_bullish_capped_at_1(self):
        ct = self._make_trader()
        ct.regime_forecaster.predict_regime.return_value = {"regime": "bullish"}
        signal = {"confidence": 0.95}  # high enough that boost would exceed 1.0
        result = ct._apply_regime_weight(signal, "BTC")
        assert result["confidence"] <= 1.0

    def test_unknown_regime_unchanged(self):
        ct = self._make_trader()
        ct.regime_forecaster.predict_regime.return_value = {"regime": "sideways"}
        signal = {"confidence": 0.70}
        result = ct._apply_regime_weight(signal, "BTC")
        assert abs(result["confidence"] - 0.70) < 0.001

    def test_no_forecaster_returns_signal_unchanged(self):
        ct = CopyTrader()  # No regime_forecaster
        signal = {"confidence": 0.70}
        result = ct._apply_regime_weight(signal, "BTC")
        assert abs(result["confidence"] - 0.70) < 0.001

    def test_forecaster_exception_returns_signal_unchanged(self):
        ct = self._make_trader()
        ct.regime_forecaster.predict_regime.side_effect = RuntimeError("network error")
        signal = {"confidence": 0.70}
        result = ct._apply_regime_weight(signal, "BTC")
        assert abs(result["confidence"] - 0.70) < 0.001, \
            "Exception in forecaster should not modify signal"


# ─── Metadata helper ─────────────────────────────────────────────

class TestBuildTradeMetadata:
    def test_all_required_fields_present(self):
        signal = {
            "type": "copy_open",
            "source_trader": "0xabc123",
            "confidence": 0.75,
            "_signal_id": "sig-1",
            "_source_key": "copy_trade:0xabc123",
            "source_accuracy": 0.8,
            "regime": "bullish",
            "is_golden": True,
            "price": 50000.0,
        }
        meta = CopyTrader._build_trade_metadata(signal, {"stop_loss_pct": 0.04}, "taker")
        assert meta["type"] == "copy_open"
        assert meta["source_trader"] == "0xabc123"
        assert meta["confidence"] == 0.75
        assert meta["is_copy_trade"] is True
        assert meta["is_golden"] is True
        assert meta["golden_wallet"] is True
        assert meta["source"] == "copy_trade"
        assert meta["execution_role"] == "taker"
        assert meta["risk_policy"] == {"stop_loss_pct": 0.04}

    def test_missing_optional_fields_default_safely(self):
        signal = {"type": "copy_open", "price": 1000.0}
        meta = CopyTrader._build_trade_metadata(signal, {}, "maker")
        assert meta["source_trader"] == ""
        assert meta["confidence"] == 0
        assert meta["is_golden"] is False
        assert meta["regime"] == ""

    def test_intended_entry_price_from_signal(self):
        signal = {"type": "copy_open", "price": 42000.0}
        meta = CopyTrader._build_trade_metadata(signal, {}, "taker")
        assert meta["intended_entry_price"] == 42000.0
        assert meta["entry_slipped_price"] == 42000.0


# ─── Position cache TTL eviction ─────────────────────────────────

class TestPositionCacheTTL:
    def test_stale_entries_evicted_on_scan(self):
        """Entries older than TTL and not in current top-N are evicted."""
        ct = CopyTrader()

        # Inject a stale entry for an address NOT in the active top-N
        old_addr = "0xstale_address"
        ct._position_cache[old_addr] = {"BTC": {"size": 1.0}}
        ct._position_cache_ts[old_addr] = time.time() - (_POSITION_CACHE_TTL_SECONDS + 1)

        # We need at least one active trader so scan_top_traders doesn't return early.
        # The active trader is a DIFFERENT address so old_addr should still be evicted.
        active_trader = {
            "address": "0xactive_trader",
            "win_rate": 0.6,
            "total_pnl": 5000,
        }
        mock_state = {"positions": []}

        with patch("src.trading.copy_trader.db.get_active_traders", return_value=[active_trader]), \
             patch("src.trading.copy_trader.hl.get_all_mids", return_value={}), \
             patch("src.trading.copy_trader.hl.get_user_state", return_value=mock_state):
            ct.scan_top_traders(top_n=5)

        assert old_addr not in ct._position_cache, "Stale entry should be evicted"
        assert old_addr not in ct._position_cache_ts

    def test_fresh_entries_retained_on_scan(self):
        """Entries within TTL are kept even if trader dropped from top-N."""
        ct = CopyTrader()

        recent_addr = "0xrecent_address"
        ct._position_cache[recent_addr] = {"ETH": {"size": 0.5}}
        ct._position_cache_ts[recent_addr] = time.time() - 60  # 1 minute ago

        with patch("src.trading.copy_trader.db.get_active_traders", return_value=[]), \
             patch("src.trading.copy_trader.hl.get_all_mids", return_value={}):
            ct.scan_top_traders(top_n=5)

        assert recent_addr in ct._position_cache, "Recent entry should NOT be evicted"

    def test_cache_timestamp_updated_on_scan(self):
        """After scanning a trader, their cache timestamp is updated."""
        trader = {
            "address": "0x" + "5" * 40,
            "win_rate": 0.6,
            "total_pnl": 10000,
        }
        ct = CopyTrader()

        mock_state = {
            "positions": [{"coin": "BTC", "size": 1.0, "side": "long",
                           "entry_price": 50000.0, "leverage": 2}]
        }

        before = time.time()
        with patch("src.trading.copy_trader.db.get_active_traders", return_value=[trader]), \
             patch("src.trading.copy_trader.hl.get_all_mids", return_value={"BTC": 50000.0}), \
             patch("src.trading.copy_trader.hl.get_user_state", return_value=mock_state):
            ct.scan_top_traders(top_n=1)

        ts = ct._position_cache_ts.get(trader["address"], 0)
        assert ts >= before, "Cache timestamp should be updated after scan"

    def test_scan_top_traders_skips_invalid_addresses(self):
        """Malformed trader rows should never be queried against Hyperliquid."""
        ct = CopyTrader()
        valid = "0x" + "4" * 40
        invalid = "0xalpha_momentum_001"
        traders = [
            {"address": invalid, "win_rate": 0.8, "total_pnl": 1000},
            {"address": valid, "win_rate": 0.6, "total_pnl": 5000},
        ]
        seen = []
        mock_state = {
            "positions": [
                {
                    "coin": "BTC",
                    "size": 1.0,
                    "side": "long",
                    "entry_price": 50000.0,
                    "leverage": 2,
                }
            ]
        }

        def _get_user_state(address):
            seen.append(address)
            return mock_state

        with patch("src.trading.copy_trader.db.get_active_traders", return_value=traders), \
             patch("src.trading.copy_trader.hl.get_all_mids", return_value={"BTC": 50000.0}), \
             patch("src.trading.copy_trader.hl.get_user_state", side_effect=_get_user_state):
            ct.scan_top_traders(top_n=5)

        assert seen == [valid]


# ─── Risk policy silent failure handling ─────────────────────────

class TestRiskPolicyResilience:
    def test_none_return_does_not_crash(self):
        """risk_policy_engine.apply() returning None should not crash."""
        ct = CopyTrader()
        ct.risk_policy_engine = MagicMock()
        ct.risk_policy_engine.apply.return_value = None

        signal = {
            "coin": "BTC",
            "side": "long",
            "price": 50000.0,
            "confidence": 0.7,
            "source_trader": "0xtest",
        }
        # Should not raise
        stop, tp, policy = ct._resolve_copy_trade_risk(signal, leverage=2, regime_data=None)
        assert isinstance(stop, float)
        assert isinstance(tp, float)

    def test_exception_in_apply_does_not_crash(self):
        """Exception in risk_policy_engine.apply() should be caught gracefully."""
        ct = CopyTrader()
        ct.risk_policy_engine = MagicMock()
        ct.risk_policy_engine.apply.side_effect = RuntimeError("policy engine unavailable")

        signal = {
            "coin": "ETH",
            "side": "short",
            "price": 3000.0,
            "confidence": 0.6,
            "source_trader": "0xtest",
        }
        # Should not raise
        stop, tp, policy = ct._resolve_copy_trade_risk(signal, leverage=3, regime_data=None)
        assert isinstance(stop, float)
        assert isinstance(tp, float)


# ─── Detect position changes ──────────────────────────────────────

class TestDetectPositionChanges:
    def _make_trader(self):
        return {"address": "0xtest123456", "win_rate": 0.65, "total_pnl": 50000}

    def test_new_position_generates_copy_open(self):
        ct = CopyTrader()
        trader = self._make_trader()
        mids = {"BTC": 50000.0}
        old = {}
        new = {"BTC": {"side": "long", "size": 1.0, "entry_price": 50000.0, "leverage": 2}}

        signals = ct._detect_position_changes("0xtest", old, new, trader, mids)
        opens = [s for s in signals if s["type"] == "copy_open"]
        assert len(opens) == 1
        assert opens[0]["coin"] == "BTC"
        assert opens[0]["side"] == "long"

    def test_full_source_trader_address_is_preserved(self):
        ct = CopyTrader()
        trader = self._make_trader()
        mids = {"BTC": 50000.0}
        address = "0x1234567890abcdef1234567890abcdef12345678"
        signals = ct._detect_position_changes(
            address,
            {},
            {"BTC": {"side": "long", "size": 1.0, "entry_price": 50000.0, "leverage": 2}},
            trader,
            mids,
        )
        assert signals[0]["source_trader"] == address

    def test_closed_position_generates_copy_close(self):
        ct = CopyTrader()
        trader = self._make_trader()
        mids = {}
        old = {"BTC": {"side": "long", "size": 1.0}}
        new = {}  # BTC position gone

        signals = ct._detect_position_changes("0xtest", old, new, trader, mids)
        closes = [s for s in signals if s["type"] == "copy_close"]
        assert len(closes) == 1
        assert closes[0]["coin"] == "BTC"

    def test_scale_in_generates_copy_scale_in(self):
        ct = CopyTrader()
        trader = self._make_trader()
        mids = {"ETH": 3000.0}
        old = {"ETH": {"side": "long", "size": 1.0, "entry_price": 3000.0, "leverage": 2}}
        new = {"ETH": {"side": "long", "size": 2.0, "entry_price": 3000.0, "leverage": 2}}

        signals = ct._detect_position_changes("0xtest", old, new, trader, mids)
        scales = [s for s in signals if s["type"] == "copy_scale_in"]
        assert len(scales) == 1

    def test_side_flip_generates_copy_flip(self):
        ct = CopyTrader()
        trader = self._make_trader()
        mids = {"SOL": 100.0}
        old = {"SOL": {"side": "long", "size": 5.0, "entry_price": 100.0, "leverage": 3}}
        new = {"SOL": {"side": "short", "size": 5.0, "entry_price": 100.0, "leverage": 3}}

        signals = ct._detect_position_changes("0xtest", old, new, trader, mids)
        flips = [s for s in signals if s["type"] == "copy_flip"]
        assert len(flips) == 1
        assert flips[0]["side"] == "short"

    def test_confidence_uses_model(self):
        """Open signal confidence is derived from the model, not hardcoded."""
        ct = CopyTrader()
        trader = {"address": "0xtest", "win_rate": 0.8, "total_pnl": 10000}
        mids = {"BTC": 50000.0}
        old = {}
        new = {"BTC": {"side": "long", "size": 1.0, "entry_price": 50000.0, "leverage": 2}}

        signals = ct._detect_position_changes("0xtest", old, new, trader, mids)
        expected = CopyTrader._calculate_signal_confidence("copy_open", 0.8)
        assert abs(signals[0]["confidence"] - expected) < 0.001
