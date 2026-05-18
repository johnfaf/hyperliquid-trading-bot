"""Regime hysteresis (#7): pure debounce + flag-gated wiring.

The label flipped bullish/crash/neutral cycle-to-cycle and re-poisoned
the ~12 gates. Hysteresis debounces a *changed* label; a genuine
high-confidence change still passes instantly. Default OFF -> byte
identical, proven here too.
"""
from __future__ import annotations

import config
from src.analysis.regime_detector import Regime, RegimeDetector
from src.analysis.regime_stability import apply_regime_hysteresis, empty_state


def _step(state, regime, conf, *, min_streak=2, override=0.85):
    return apply_regime_hysteresis(
        state, new_regime=regime, new_confidence=conf,
        min_streak=min_streak, override_confidence=override,
    )


def test_first_read_is_adopted():
    eff, st = _step(empty_state(), "trending_up", 0.7)
    assert eff == "trending_up"
    assert st["effective"] == "trending_up" and st["streak"] == 1


def test_reaffirm_increments_streak_and_clears_pending():
    eff, st = _step({"effective": "trending_up", "streak": 3,
                     "pending": "ranging", "pending_count": 1}, "trending_up", 0.6)
    assert eff == "trending_up"
    assert st["streak"] == 4 and st["pending"] is None


def test_low_conf_challenger_is_held_until_min_streak():
    st = empty_state()
    eff, st = _step(st, "trending_up", 0.7)          # adopt up
    eff, st = _step(st, "trending_down", 0.70)        # challenger #1 -> held
    assert eff == "trending_up"
    assert st["pending"] == "trending_down" and st["pending_count"] == 1
    eff, st = _step(st, "trending_down", 0.70)        # challenger #2 -> flips
    assert eff == "trending_down"
    assert st["effective"] == "trending_down" and st["pending"] is None


def test_high_confidence_change_flips_immediately():
    """The crash escape hatch: a real high-confidence regime change must
    NOT be debounced."""
    eff, st = _step({"effective": "trending_up", "streak": 5,
                     "pending": None, "pending_count": 0}, "crash", 0.92)
    assert eff == "crash"
    assert st["effective"] == "crash" and st["streak"] == 1


def test_interrupted_challenger_resets_streak():
    st = empty_state()
    eff, st = _step(st, "trending_up", 0.7)
    eff, st = _step(st, "trending_down", 0.7)   # pending down x1
    eff, st = _step(st, "ranging", 0.7)         # different challenger -> pending resets
    assert eff == "trending_up"
    assert st["pending"] == "ranging" and st["pending_count"] == 1


def test_nan_and_garbage_confidence_safe():
    eff, st = _step({"effective": "trending_up", "streak": 1,
                     "pending": None, "pending_count": 0},
                    "trending_down", float("nan"))
    # NaN conf < override -> debounced, not an immediate flip, no crash
    assert eff == "trending_up"


# ── flag-gated wiring (default OFF == byte identical) ──

class _StubState:
    def __init__(self, regime, confidence):
        self.regime = regime
        self.confidence = confidence

    def to_dict(self):
        return {"regime": self.regime.value, "confidence": self.confidence}


def _detector(monkeypatch, regime: Regime, conf=0.7):
    d = RegimeDetector()
    monkeypatch.setattr(d, "detect_regime",
                        lambda coin, candles=None: _StubState(regime, conf))
    import src.analysis.regime_detector as rd
    monkeypatch.setattr(rd.time, "sleep", lambda *_a, **_k: None)
    return d


def test_flag_off_is_passthrough(monkeypatch):
    monkeypatch.setattr(config, "REGIME_HYSTERESIS_ENABLED", False, raising=False)
    d = _detector(monkeypatch, Regime.TRENDING_UP)
    r1 = d.get_market_regime(coins=["BTC"])
    assert r1["overall_regime"] == "trending_up"
    assert r1["raw_overall_regime"] == "trending_up"
    # flip the raw read; with flag off the change passes straight through
    monkeypatch.setattr(d, "detect_regime",
                        lambda coin, candles=None: _StubState(Regime.TRENDING_DOWN, 0.7))
    r2 = d.get_market_regime(coins=["BTC"])
    assert r2["overall_regime"] == "trending_down"


def test_flag_on_debounces_one_cycle_flip(monkeypatch):
    monkeypatch.setattr(config, "REGIME_HYSTERESIS_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "REGIME_HYSTERESIS_MIN_STREAK", 2, raising=False)
    monkeypatch.setattr(config, "REGIME_HYSTERESIS_OVERRIDE_CONF", 0.85, raising=False)
    d = _detector(monkeypatch, Regime.TRENDING_UP, conf=0.7)
    assert d.get_market_regime(coins=["BTC"])["overall_regime"] == "trending_up"
    # one-cycle low-conf flip -> effective HELD at trending_up, raw shows down
    monkeypatch.setattr(d, "detect_regime",
                        lambda coin, candles=None: _StubState(Regime.TRENDING_DOWN, 0.7))
    r = d.get_market_regime(coins=["BTC"])
    assert r["overall_regime"] == "trending_up"
    assert r["raw_overall_regime"] == "trending_down"
    # persists a 2nd cycle -> now flips
    r = d.get_market_regime(coins=["BTC"])
    assert r["overall_regime"] == "trending_down"


def test_flag_on_high_conf_change_flips_immediately(monkeypatch):
    monkeypatch.setattr(config, "REGIME_HYSTERESIS_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "REGIME_HYSTERESIS_MIN_STREAK", 3, raising=False)
    monkeypatch.setattr(config, "REGIME_HYSTERESIS_OVERRIDE_CONF", 0.85, raising=False)
    d = _detector(monkeypatch, Regime.TRENDING_UP, conf=0.7)
    d.get_market_regime(coins=["BTC"])
    # A genuine high-confidence regime change must bypass the debounce
    # even with min_streak=3 (the crash escape hatch).
    monkeypatch.setattr(d, "detect_regime",
                        lambda coin, candles=None: _StubState(Regime.TRENDING_DOWN, 0.95))
    r = d.get_market_regime(coins=["BTC"])
    assert r["overall_regime"] == "trending_down"  # bypassed the debounce
