"""Require-proven live-mirror gate (algo #1).

A source must accumulate real calibration evidence (enough samples AND a
positive edge) before it risks live capital; unproven sources keep paper-
trading. Flag-gated default OFF (fail-open), so nothing changes until the
operator enables it.
"""
from __future__ import annotations

from src.core import live_execution as le


class _Sig:
    def __init__(self, source="copy_trade", side="long", regime="trend", strategy_type=""):
        self.source = source
        self.side = side
        self.regime = regime
        self.strategy_type = strategy_type


class _Cal:
    """Stub calibration tracker exposing proven_evidence -> (edge, n)."""
    def __init__(self, edge, n):
        self._e, self._n = edge, n

    def proven_evidence(self, source, side=None, regime=None):
        return (self._e, self._n)


def _enable(monkeypatch, samples=30, edge=0.50):
    monkeypatch.setenv("LIVE_MIRROR_REQUIRE_PROVEN_ENABLED", "true")
    monkeypatch.setenv("LIVE_MIRROR_PROVEN_MIN_SAMPLES", str(samples))
    monkeypatch.setenv("LIVE_MIRROR_PROVEN_MIN_EDGE", str(edge))


def test_off_by_default_allows(monkeypatch):
    monkeypatch.delenv("LIVE_MIRROR_REQUIRE_PROVEN_ENABLED", raising=False)
    allow, _ = le._live_mirror_require_proven(_Sig(), _Cal(0.20, 0))
    assert allow, "gate must be OFF by default (fail-open)"


def test_blocks_thin_sample(monkeypatch):
    _enable(monkeypatch)
    allow, reason = le._live_mirror_require_proven(_Sig(), _Cal(0.70, 5))
    assert not allow and "samples" in reason


def test_blocks_low_edge(monkeypatch):
    _enable(monkeypatch)
    allow, reason = le._live_mirror_require_proven(_Sig(), _Cal(0.45, 100))
    assert not allow and "edge" in reason


def test_allows_proven(monkeypatch):
    _enable(monkeypatch)
    allow, _ = le._live_mirror_require_proven(_Sig(), _Cal(0.62, 100))
    assert allow


def test_fail_open_without_tracker(monkeypatch):
    _enable(monkeypatch)
    allow, _ = le._live_mirror_require_proven(_Sig(), None)
    assert allow, "no calibration tracker -> fail-open (don't freeze live)"


def test_conviction_gate_composes_require_proven(monkeypatch):
    """The combined gate blocks an unproven source when require-proven is on."""
    _enable(monkeypatch)
    allow, reason = le._live_mirror_conviction_gate(_Sig(), calibration=_Cal(0.20, 100))
    assert not allow and "edge" in reason
