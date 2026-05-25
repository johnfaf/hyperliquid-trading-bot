"""Live-mirror promotion gate must fail-CLOSED on exceptions.

Background
----------
``src.core.live_execution._rescale_size_for_live`` wraps the live-mirror
promotion gate check in a try/except.  Before this fix, the except
branch silently set ``promotable=True`` and ``promotion_scale=1.0``,
silently bypassing the entire promotion safety system on:

  * a transient Postgres outage during ``is_live_promotable``
  * any ImportError on the gate module (e.g. mid-refactor)
  * any internal gate exception (e.g. schema migration window)

The wrapper's fail-OPEN-to-live default was the inverse of the gate's
internal fail-CLOSED design (``is_live_promotable`` returns False on
exceptions).  The wrapper was the weak link.

After this fix
--------------
The default is fail-CLOSED for live: an exception during gate
evaluation skips the live mirror entirely (returns None).  The paper
trade is unaffected; the source keeps accumulating outcomes and can
promote on the next cycle.

The legacy fail-open behaviour is still reachable via the
``LIVE_PROMOTION_GATE_FAIL_OPEN=1`` env var for the rare bootstrap case
where the operator explicitly wants to bypass.
"""
from __future__ import annotations

from src.core import live_execution


class _StubTrader:
    """Minimal trader stub so ``_rescale_size_for_live`` can run."""

    max_order_usd = 100_000.0
    min_order_usd = 11.0

    def get_account_value(self):
        return 10_000.0

    def get_free_margin(self):
        return 10_000.0


_TRADE = {
    "coin": "ETH",
    "side": "short",
    "size": 0.25,
    "entry_price": 2000.0,
    "leverage": 1.0,
    "source": "copy_trade",
    "source_trader": "0xABCDEF" + "0" * 34,
}


def _patch_rescale_deps(monkeypatch):
    """Stub paper account + mids so rescale runs without DBs/HTTP."""
    monkeypatch.setattr(
        live_execution.db, "get_paper_account",
        lambda: {"balance": 10_000.0},
    )
    monkeypatch.setattr(live_execution, "_paper_open_margin_used", lambda: 0.0)
    monkeypatch.setattr(live_execution, "get_all_mids", lambda: {"ETH": 2000.0})


# ── Fail-closed default ──────────────────────────────────────


def test_gate_exception_defaults_to_fail_closed(monkeypatch):
    """An exception in is_live_promotable returns None (skip live mirror)."""
    _patch_rescale_deps(monkeypatch)
    monkeypatch.delenv("LIVE_PROMOTION_GATE_FAIL_OPEN", raising=False)

    def _boom(*a, **k):
        raise RuntimeError("DB pool exhausted")

    # Patch BOTH the originating module AND the live_execution-local import
    # site so the late ``from src.learning.promotion_gate import ...`` inside
    # _rescale_size_for_live picks up the stub.
    import src.learning.promotion_gate as pg
    monkeypatch.setattr(pg, "is_live_promotable", _boom)

    result = live_execution._rescale_size_for_live(dict(_TRADE), _StubTrader())
    assert result is None, (
        "Live mirror must be skipped when the promotion gate raises; "
        "previously this returned a full-size live trade with "
        "reason='gate_error_fail_open'"
    )


def test_gate_import_error_defaults_to_fail_closed(monkeypatch):
    """Even an ImportError on the gate module must fail-closed."""
    _patch_rescale_deps(monkeypatch)
    monkeypatch.delenv("LIVE_PROMOTION_GATE_FAIL_OPEN", raising=False)

    import builtins
    real_import = builtins.__import__

    def _broken_import(name, *args, **kwargs):
        if name == "src.learning.promotion_gate":
            raise ImportError("simulated mid-refactor breakage")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _broken_import)

    result = live_execution._rescale_size_for_live(dict(_TRADE), _StubTrader())
    assert result is None


# ── Explicit fail-open opt-in ────────────────────────────────


def test_fail_open_opt_in_proceeds_at_full_size(monkeypatch):
    """LIVE_PROMOTION_GATE_FAIL_OPEN=1 restores legacy fail-open behaviour."""
    _patch_rescale_deps(monkeypatch)
    monkeypatch.setenv("LIVE_PROMOTION_GATE_FAIL_OPEN", "1")

    def _boom(*a, **k):
        raise RuntimeError("DB pool exhausted")

    import src.learning.promotion_gate as pg
    monkeypatch.setattr(pg, "is_live_promotable", _boom)

    result = live_execution._rescale_size_for_live(dict(_TRADE), _StubTrader())
    assert result is not None, (
        "With LIVE_PROMOTION_GATE_FAIL_OPEN=1 the trade should proceed "
        "(legacy behaviour) so the bot doesn't grind to a halt during a "
        "known temporary outage."
    )
    assert float(result["size"]) > 0


def test_fail_open_disabled_by_explicit_zero(monkeypatch):
    """LIVE_PROMOTION_GATE_FAIL_OPEN=0 (explicit) is still fail-closed."""
    _patch_rescale_deps(monkeypatch)
    monkeypatch.setenv("LIVE_PROMOTION_GATE_FAIL_OPEN", "0")

    def _boom(*a, **k):
        raise RuntimeError("DB pool exhausted")

    import src.learning.promotion_gate as pg
    monkeypatch.setattr(pg, "is_live_promotable", _boom)

    result = live_execution._rescale_size_for_live(dict(_TRADE), _StubTrader())
    assert result is None


# ── Happy path unaffected ────────────────────────────────────


def test_normal_pass_through_unchanged(monkeypatch):
    """When the gate works and approves the trade, the function returns a
    live-scaled trade as before."""
    _patch_rescale_deps(monkeypatch)
    monkeypatch.delenv("LIVE_PROMOTION_GATE_FAIL_OPEN", raising=False)

    import src.learning.promotion_gate as pg
    monkeypatch.setattr(pg, "is_live_promotable", lambda trade: (True, "ok"))

    result = live_execution._rescale_size_for_live(dict(_TRADE), _StubTrader())
    assert result is not None
    assert float(result["size"]) > 0


def test_normal_block_pass_through_unchanged(monkeypatch):
    """When the gate works and rejects the trade, the function returns None
    with the standard 'blocked' log path (NOT the new fail-closed path)."""
    _patch_rescale_deps(monkeypatch)
    monkeypatch.delenv("LIVE_PROMOTION_GATE_FAIL_OPEN", raising=False)

    import src.learning.promotion_gate as pg
    monkeypatch.setattr(
        pg, "is_live_promotable",
        lambda trade: (False, "insufficient_signals:n=3<30"),
    )

    result = live_execution._rescale_size_for_live(dict(_TRADE), _StubTrader())
    assert result is None
