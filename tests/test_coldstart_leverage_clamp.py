"""Cold-start leverage clamp.

When a (source|side|regime) calibration bucket is below the sample
threshold the EV gate runs on the assumed p_win=0.50 prior. An
unproven bucket must not run high leverage -- it saturates the
leveraged-notional aggregate-exposure cap (one 8x cold-start short
locked out 23 signals / 6h in prod). Clamp until the bucket earns it.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

import config
from src.signals.decision_firewall import DecisionFirewall
from tests.test_decision_firewall import MockSignal  # reuse fixture-style mock


@pytest.fixture(autouse=True)
def _bypass_unrelated_gates(monkeypatch):
    # Isolate the clamp: the clamp runs after EV-accept, and a bare
    # MockSignal has no risk policy so the EV gate would reject it
    # (cold-start 200/200 - cost < 0) before the clamp block. Disable
    # data-readiness + EV gate so the signal reaches the clamp; bucket_n
    # is still computed regardless of EV enabled state.
    monkeypatch.setattr(config, "DATA_READINESS_GATE_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "EV_GATE_ENABLED", False, raising=False)


class _StubCalibration:
    """Minimal calibration tracker: configurable bucket sample size."""
    def __init__(self, n):
        self._n = n

    def get_sample_size(self, key):
        return self._n

    def get_adjustment_factor(self, *a, **k):
        return 0.50  # cold-start prior

    def get_bucketed_min_confidence(self, *a, **k):
        return (0.0, "test_open")  # don't block on bucketed threshold


def _firewall(bucket_n, *, clamp=True, max_lev=3.0, min_samples=30):
    fw = DecisionFirewall({
        "enable_predictive_derisk": False,
        "funding_risk_enabled": False,
        "min_confidence": 0.0,
        # High so the firewall's section-3 generic leverage clamp
        # (max_leverage) doesn't pre-clamp 8x and mask the cold-start
        # clamp under test.
        "max_leverage": 25,
        "block_unknown_sources": False,
        "use_bucketed_thresholds": False,
        "coldstart_leverage_clamp_enabled": clamp,
        "coldstart_max_leverage": max_lev,
        "coldstart_calibration_min_samples": min_samples,
        "cooldown_seconds": 0,
        "same_side_cooldown_seconds": 0,
    })
    fw.calibration = _StubCalibration(bucket_n)
    return fw


@pytest.fixture
def _mock_db(monkeypatch):
    import src.signals.decision_firewall as fwmod
    m = MagicMock()
    m.get_open_paper_trades.return_value = []
    m.get_paper_account.return_value = {"balance": 10_000}
    m.audit_log = MagicMock()
    monkeypatch.setattr(fwmod, "db", m)
    return m


def test_clamps_high_leverage_when_bucket_coldstart(_mock_db):
    fw = _firewall(bucket_n=0)
    sig = MockSignal(coin="SOL", side_val="short", confidence=0.6, leverage=8.0)
    sig.source = "strategy"
    sig.strategy_type = "momentum_short"
    fw.validate(sig, regime_data={"overall_regime": "trending_down"})
    assert sig.leverage == 3.0, "8x cold-start leverage must be clamped to 3x"


def test_does_not_clamp_when_bucket_has_history(_mock_db):
    fw = _firewall(bucket_n=50)  # >= min_samples 30
    sig = MockSignal(coin="SOL", side_val="short", confidence=0.6, leverage=8.0)
    sig.source = "strategy"
    sig.strategy_type = "momentum_short"
    fw.validate(sig, regime_data={"overall_regime": "trending_down"})
    assert sig.leverage == 8.0, "proven bucket keeps full leverage"


def test_low_leverage_untouched_during_coldstart(_mock_db):
    fw = _firewall(bucket_n=0)
    sig = MockSignal(coin="SOL", side_val="short", confidence=0.6, leverage=2.0)
    sig.source = "strategy"
    sig.strategy_type = "momentum_short"
    fw.validate(sig, regime_data={"overall_regime": "trending_down"})
    assert sig.leverage == 2.0, "already-conservative leverage is left as-is"


def test_clamp_disabled_keeps_high_leverage(_mock_db):
    fw = _firewall(bucket_n=0, clamp=False)
    sig = MockSignal(coin="SOL", side_val="short", confidence=0.6, leverage=8.0)
    sig.source = "strategy"
    sig.strategy_type = "momentum_short"
    fw.validate(sig, regime_data={"overall_regime": "trending_down"})
    assert sig.leverage == 8.0, "knob off -> no clamp"


def test_clamp_records_context_breadcrumb(_mock_db):
    fw = _firewall(bucket_n=1)
    sig = MockSignal(coin="SOL", side_val="short", confidence=0.6, leverage=8.0)
    sig.source = "strategy"
    sig.strategy_type = "momentum_short"
    fw.validate(sig, regime_data={"overall_regime": "trending_down"})
    cs = sig.context.get("coldstart_leverage_clamped")
    assert cs and cs["from"] == 8.0 and cs["to"] == 3.0 and cs["bucket_n"] == 1
