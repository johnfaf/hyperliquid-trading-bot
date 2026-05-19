"""A5 wired into the promotion gate (default OFF, fail-open, only blocks).

The Deflated-Sharpe gate may only downgrade a promotion the base gate
already approved; it never unblocks, and any missing/insufficient/error
data path fails OPEN so it can tighten promotion but never break it.
"""
from __future__ import annotations

from contextlib import contextmanager

import config
import src.learning.promotion_gate as pg
from src.learning.promotion_gate import _dsr_promotion_ok, is_live_promotable

# A *moderate* edge over many observations is DSR-significant (an
# extreme Sharpe over few obs deflates to non-significant by design --
# Bailey-LdP: the Sharpe estimator's SE grows with SR^2). This series:
# mean~0.20, std~0.20, n=120 -> deflated Sharpe ~9.9, p~0, sig95=True
# (verified empirically against promotion_stats.deflated_sharpe).
_STRONG = [0.2 + (((i * 3 % 7) - 3) / 10.0) for i in range(120)]
# Flat zero-mean -> Sharpe ~0 -> not significant.
_FLAT = [0.1, -0.1] * 30


def _conn(rows):
    @contextmanager
    def _ctx(*a, **k):
        class _C:
            def execute(self, *_a, **_k):
                class _Cur:
                    def fetchall(self_):
                        return [{"pnl": v} for v in rows]
                return _Cur()
        yield _C()
    return _ctx


def test_no_strategy_id_fails_open():
    ok, why = _dsr_promotion_ok(None, num_trials=50, min_obs=20)
    assert ok is True and "no_strategy_id" in why


def test_insufficient_history_fails_open(monkeypatch):
    monkeypatch.setattr(pg.db, "get_connection", _conn([0.3] * 5))
    ok, why = _dsr_promotion_ok(7, num_trials=50, min_obs=20)
    assert ok is True and why.startswith("dsr_insufficient")


def test_db_error_fails_open(monkeypatch):
    @contextmanager
    def _boom(*a, **k):
        raise RuntimeError("db down")
        yield  # pragma: no cover
    monkeypatch.setattr(pg.db, "get_connection", _boom)
    ok, why = _dsr_promotion_ok(7, num_trials=50, min_obs=20)
    assert ok is True and why == "dsr_lookup_failed"


def test_significant_edge_passes(monkeypatch):
    monkeypatch.setattr(pg.db, "get_connection", _conn(_STRONG))
    ok, why = _dsr_promotion_ok(7, num_trials=1, min_obs=20)
    assert ok is True and why.startswith("ok_dsr")


def test_no_edge_blocks(monkeypatch):
    monkeypatch.setattr(pg.db, "get_connection", _conn(_FLAT))
    ok, why = _dsr_promotion_ok(7, num_trials=50, min_obs=20)
    assert ok is False and why.startswith("dsr_not_significant")


# ── flag gating through is_live_promotable (Path 1) ──

def _wire(monkeypatch, *, base_ok, dsr_rows, require_dsr):
    monkeypatch.setattr(config, "LIVE_PROMOTION_GATE_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "PROMOTION_REQUIRE_DSR", require_dsr, raising=False)
    monkeypatch.setattr(pg.db, "get_strategy", lambda sid: {"id": sid})
    monkeypatch.setattr(
        pg, "_strategy_promotion_ok",
        lambda strat, **kw: (base_ok, "ok" if base_ok else "score_too_low:0.1<0.2"),
    )
    monkeypatch.setattr(pg.db, "get_connection", _conn(dsr_rows))


def test_flag_off_is_zero_change(monkeypatch):
    # base approves; DSR would block (flat), but flag OFF -> still promoted
    _wire(monkeypatch, base_ok=True, dsr_rows=_FLAT, require_dsr=False)
    ok, why = is_live_promotable({"strategy_id": 7})
    assert ok is True and why == "ok"


def test_flag_on_blocks_insignificant(monkeypatch):
    _wire(monkeypatch, base_ok=True, dsr_rows=_FLAT, require_dsr=True)
    ok, why = is_live_promotable({"strategy_id": 7})
    assert ok is False and why.startswith("dsr_not_significant")


def test_flag_on_keeps_significant(monkeypatch):
    _wire(monkeypatch, base_ok=True, dsr_rows=_STRONG, require_dsr=True)
    monkeypatch.setattr(config, "PROMOTION_DSR_NUM_TRIALS", 1, raising=False)
    ok, why = is_live_promotable({"strategy_id": 7})
    assert ok is True and why == "ok"  # base reason preserved


def test_dsr_never_consulted_when_base_rejects(monkeypatch):
    # base already False -> DSR must not run / not change the reason
    _wire(monkeypatch, base_ok=False, dsr_rows=_STRONG, require_dsr=True)
    ok, why = is_live_promotable({"strategy_id": 7})
    assert ok is False and why.startswith("score_too_low")
