"""A2 Thompson allocator wired into AgentScorer.get_weight (default OFF).

Default == byte-identical legacy weight + lazy (no allocator object).
When enabled it blends a posterior sample by AGENT_BANDIT_BLEND and is
fed the same win/loss outcomes; any allocator error falls back to legacy.
"""
from __future__ import annotations

import pytest

from src.signals.agent_scoring import AgentScorer


class _FakeAlloc:
    def __init__(self, sample_val=0.8):
        self.sample_val = sample_val
        self.updates = []
        self.raise_on_sample = False

    def sample(self, source, now_ts=None):
        if self.raise_on_sample:
            raise RuntimeError("boom")
        return self.sample_val

    def update(self, source, won, now_ts=None):
        self.updates.append((source, won))


def test_default_off_is_lazy_and_byte_identical():
    s = AgentScorer()  # no cfg -> default OFF
    assert s._bandit_enabled is False
    assert s._bandit is None
    legacy = s._legacy_get_weight("brand_new_src")
    assert s.get_weight("brand_new_src") == legacy  # identical
    assert s._bandit is None  # never instantiated when off


def test_enabled_blends_sample_with_legacy():
    s = AgentScorer({"bandit_allocator_enabled": True, "bandit_blend": 1.0})
    s._bandit = _FakeAlloc(sample_val=0.8)  # inject deterministic alloc
    # unknown source -> legacy weight is 0.5; blend=1.0 -> pure sample
    assert s._legacy_get_weight("u") == pytest.approx(0.5)
    assert s.get_weight("u") == pytest.approx(0.8)


def test_blend_fraction_is_respected():
    s = AgentScorer({"bandit_allocator_enabled": True, "bandit_blend": 0.5})
    s._bandit = _FakeAlloc(sample_val=0.9)
    # 0.5*legacy(0.5) + 0.5*sample(0.9) = 0.7
    assert s.get_weight("u") == pytest.approx(0.7)


def test_blend_zero_returns_legacy_even_when_enabled():
    s = AgentScorer({"bandit_allocator_enabled": True, "bandit_blend": 0.0})
    s._bandit = _FakeAlloc(sample_val=0.99)
    assert s.get_weight("u") == pytest.approx(s._legacy_get_weight("u"))


def test_allocator_error_falls_back_to_legacy():
    s = AgentScorer({"bandit_allocator_enabled": True, "bandit_blend": 1.0})
    fa = _FakeAlloc()
    fa.raise_on_sample = True
    s._bandit = fa
    assert s.get_weight("u") == pytest.approx(s._legacy_get_weight("u"))  # no raise


def test_record_outcome_feeds_allocator_when_enabled(monkeypatch):
    s = AgentScorer({"bandit_allocator_enabled": True})
    fa = _FakeAlloc()
    s._bandit = fa
    monkeypatch.setattr(s, "_recalculate", lambda *_a, **_k: None)
    monkeypatch.setattr(s, "_save_score", lambda *_a, **_k: None)
    s.record_outcome("src:x", "sig1", pnl=1.5, return_pct=0.02)   # win
    s.record_outcome("src:x", "sig2", pnl=-0.4, return_pct=-0.01)  # loss
    assert ("src:x", True) in fa.updates
    assert ("src:x", False) in fa.updates


def test_record_outcome_does_not_touch_allocator_when_off(monkeypatch):
    s = AgentScorer()  # default OFF
    monkeypatch.setattr(s, "_recalculate", lambda *_a, **_k: None)
    monkeypatch.setattr(s, "_save_score", lambda *_a, **_k: None)
    s.record_outcome("src:y", "sig1", pnl=1.0)
    assert s._bandit is None  # lazy: never created when disabled


def test_real_allocator_lazy_instantiation_when_enabled():
    s = AgentScorer({"bandit_allocator_enabled": True, "bandit_blend": 1.0})
    assert s._bandit is None                 # not built at construction
    w = s.get_weight("real_src")
    assert s._bandit is not None             # built on first use
    assert 0.0 <= w <= 1.0                    # valid posterior-blended weight
