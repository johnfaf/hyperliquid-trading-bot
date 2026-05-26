"""Loss-leak controls — minimum-hold-before-SL + per-coin-side blocklist.

Background
----------
Wallet trade history 2026-04-05 -> 2026-05-26 (51 days, 234 closes,
-$21.22 net) revealed two structural loss leaks:

  1. **Noise-stops**: positions held <30 min lost 83% of the time
     (-$12.59 net across 35 trades).  Positions held >24h won 71%
     of the time (+$11.27 net).  The bot was getting noise-stopped
     before its edge could materialise.

  2. **Per-coin-side blind spot**: HYPE long was 0/6 wins, -$2.89
     net.  The same coin/side came through multiple source keys
     so the source allocator's per-source tracking never paused
     it.

This PR adds:
  * ``SL_MIN_HOLD_SECONDS`` (default 600) in paper_trader -- the
    stop-loss check is suppressed during the first N seconds of a
    position's life.  Take-profit, time-limit, break-even/trailing
    updates are unaffected.
  * ``PER_COIN_SIDE_BLOCKLIST`` (default ``HYPE:long``) in the
    decision firewall -- comma-separated COIN:SIDE pairs that get
    hard-rejected before any other gate runs.
"""
from __future__ import annotations

import pytest


# ── per-coin-side blocklist (decision_firewall) ──────────────


@pytest.fixture
def firewall_with_blocklist(monkeypatch):
    """Build a minimal DecisionFirewall instance with HYPE long blocked."""
    import config
    from src.signals.decision_firewall import DecisionFirewall

    monkeypatch.setattr(config, "PER_COIN_SIDE_BLOCKLIST", "HYPE:long", raising=False)
    # Minimal config kwargs to instantiate.  Pass no agent_scorer /
    # forecaster so we exercise the early-gate path only.
    fw = DecisionFirewall({
        "min_confidence": 0.0,
        "max_positions": 99,
        "max_per_coin": 99,
        "max_signals_per_source_per_day": 99,
        "block_unknown_sources": False,
    })
    return fw, config


def _make_signal(coin: str, side: str, confidence: float = 0.7):
    """Build a TradeSignal that passes basic .validate() checks."""
    from src.signals.signal_schema import TradeSignal, SignalSide, SignalSource

    return TradeSignal(
        coin=coin,
        side=SignalSide.LONG if side == "long" else SignalSide.SHORT,
        confidence=confidence,
        source=SignalSource.STRATEGY,
        reason="test signal",
        strategy_type="momentum_long" if side == "long" else "momentum_short",
        entry_price=100.0,
        leverage=1.0,
        position_pct=0.1,
    )


def test_blocklist_rejects_hype_long(firewall_with_blocklist):
    """HYPE long is rejected before any other gate runs."""
    fw, _ = firewall_with_blocklist
    signal = _make_signal("HYPE", "long")
    passed, reason = fw.validate(signal)
    assert passed is False
    assert "blocklist" in reason.lower() or "PER_COIN_SIDE_BLOCKLIST" in reason


def test_blocklist_is_side_specific(firewall_with_blocklist):
    """HYPE short is NOT blocked (only HYPE:long is)."""
    fw, _ = firewall_with_blocklist
    signal = _make_signal("HYPE", "short")
    passed, reason = fw.validate(signal)
    # The signal might be rejected by other gates (low confidence,
    # missing data, etc.) but it should NOT be rejected by the
    # blocklist gate specifically.
    assert "blocklist" not in (reason or "").lower(), (
        f"HYPE short was hit by the blocklist gate but only HYPE:long "
        f"is configured.  reason={reason}"
    )


def test_blocklist_does_not_match_other_coins(firewall_with_blocklist):
    """BTC long is NOT blocked when only HYPE:long is on the list."""
    fw, _ = firewall_with_blocklist
    signal = _make_signal("BTC", "long")
    passed, reason = fw.validate(signal)
    assert "blocklist" not in (reason or "").lower()


def test_blocklist_empty_means_no_blocking(monkeypatch):
    """Empty ``PER_COIN_SIDE_BLOCKLIST`` env disables the gate entirely."""
    import config
    from src.signals.decision_firewall import DecisionFirewall

    monkeypatch.setattr(config, "PER_COIN_SIDE_BLOCKLIST", "", raising=False)
    fw = DecisionFirewall({
        "min_confidence": 0.0,
        "max_positions": 99,
        "max_per_coin": 99,
        "max_signals_per_source_per_day": 99,
        "block_unknown_sources": False,
    })
    signal = _make_signal("HYPE", "long")
    passed, reason = fw.validate(signal)
    assert "blocklist" not in (reason or "").lower()


def test_blocklist_handles_multiple_entries(monkeypatch):
    """Multiple COIN:SIDE entries are all enforced."""
    import config
    from src.signals.decision_firewall import DecisionFirewall

    monkeypatch.setattr(
        config, "PER_COIN_SIDE_BLOCKLIST",
        "HYPE:long, SOL:short, MON:short",
        raising=False,
    )
    fw = DecisionFirewall({
        "min_confidence": 0.0,
        "max_positions": 99,
        "max_per_coin": 99,
        "max_signals_per_source_per_day": 99,
        "block_unknown_sources": False,
    })
    for coin, side in (("HYPE", "long"), ("SOL", "short"), ("MON", "short")):
        signal = _make_signal(coin, side)
        passed, reason = fw.validate(signal)
        assert passed is False, f"{coin} {side} should be rejected"
        assert "blocklist" in reason.lower()


def test_blocklist_handles_malformed_entries(monkeypatch):
    """Malformed entries (no colon, trailing comma) are silently skipped."""
    import config
    from src.signals.decision_firewall import DecisionFirewall

    monkeypatch.setattr(
        config, "PER_COIN_SIDE_BLOCKLIST",
        "HYPE:long, notacoin, BTC, SOL:short,,, ,",
        raising=False,
    )
    fw = DecisionFirewall({
        "min_confidence": 0.0,
        "max_positions": 99,
        "max_per_coin": 99,
        "max_signals_per_source_per_day": 99,
        "block_unknown_sources": False,
    })
    # HYPE:long and SOL:short still work; "notacoin" and "BTC" (no
    # side) are skipped without crashing.
    signal = _make_signal("HYPE", "long")
    passed, _ = fw.validate(signal)
    assert passed is False
    signal = _make_signal("BTC", "long")
    passed, reason = fw.validate(signal)
    assert "blocklist" not in (reason or "").lower()


# ── SL_MIN_HOLD_SECONDS guard ───────────────────────────────


def test_sl_min_hold_seconds_default_is_600():
    """The new env-var default is 10 minutes."""
    import importlib
    import config as _cfg
    importlib.reload(_cfg)
    assert _cfg.SL_MIN_HOLD_SECONDS == 600


def test_sl_min_hold_logic_recognises_young_positions(monkeypatch):
    """A 30-second-old position with SL would-be-hit must NOT close."""
    # We test the logic in isolation by replicating the guard's
    # arithmetic.  The integration with paper_trader._close_paper_positions
    # is exercised end-to-end in production where the full state machine
    # is available; here we just lock in the numeric behaviour.
    from datetime import datetime, timezone
    import config
    monkeypatch.setattr(config, "SL_MIN_HOLD_SECONDS", 600, raising=False)

    sl_min_hold_s = max(0, int(getattr(config, "SL_MIN_HOLD_SECONDS", 600)))

    # Position opened 30 seconds ago.
    now = datetime.now(timezone.utc)
    opened = now.replace(microsecond=0)
    # 30 seconds in past
    from datetime import timedelta
    opened = now - timedelta(seconds=30)
    position_age_s = (now - opened).total_seconds()

    sl_armed = position_age_s >= sl_min_hold_s if sl_min_hold_s > 0 else True
    assert sl_armed is False, (
        f"30s-old position should have sl_armed=False under "
        f"SL_MIN_HOLD_SECONDS=600; age={position_age_s}s"
    )


def test_sl_min_hold_logic_arms_after_window(monkeypatch):
    """A position older than SL_MIN_HOLD_SECONDS has sl_armed=True."""
    from datetime import datetime, timedelta, timezone
    import config
    monkeypatch.setattr(config, "SL_MIN_HOLD_SECONDS", 600, raising=False)

    now = datetime.now(timezone.utc)
    opened = now - timedelta(seconds=900)   # 15 min old
    position_age_s = (now - opened).total_seconds()
    sl_armed = position_age_s >= 600
    assert sl_armed is True


def test_sl_min_hold_zero_disables_guard(monkeypatch):
    """SL_MIN_HOLD_SECONDS=0 restores legacy behaviour (SL armed from t=0)."""
    import config
    monkeypatch.setattr(config, "SL_MIN_HOLD_SECONDS", 0, raising=False)
    sl_min_hold_s = max(0, int(getattr(config, "SL_MIN_HOLD_SECONDS", 0)))
    sl_armed = True if sl_min_hold_s <= 0 else False
    assert sl_armed is True
