"""Full-pipeline (firewall) integration harness + gate-cascade scenarios.

Almost every incident this session was a *gate interaction* — cold-start
x bucket-floor x exposure-denominator x market-side-guard x regime —
each found one-at-a-time from prod logs. Unit tests cover gates in
isolation; this runs a signal through the REAL ``DecisionFirewall.
validate()`` against seeded regime/exposure/balance state and asserts the
end-to-end verdict, so the cascade can't silently regress.

``run_through_firewall`` is the reusable harness; other suites can import
it. Assertions key off deterministic rejection-reason substrings so a
scenario isolates the gate-interaction under test without being flaky if
an unrelated downstream gate also has an opinion.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import config


@pytest.fixture(autouse=True)
def _isolate_new_gates(monkeypatch):
    # Mirror test_decision_firewall: the data-readiness / EV gates need
    # populated signal context; neutralize them so scenarios isolate the
    # cascade gates they target. Dedicated coverage lives elsewhere.
    monkeypatch.setattr(config, "DATA_READINESS_GATE_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "EV_GATE_ENABLED", False, raising=False)


class _HarnessSignal:
    """Minimal TradeSignal-like, matching the firewall's read surface."""

    def __init__(self, *, coin="BTC", side="long", confidence=0.7,
                 leverage=2, size=0.001, entry_price=100_000.0,
                 position_pct=0.05, strategy_type="momentum",
                 source="test", context=None):
        self.coin = coin
        self.side = MagicMock()
        self.side.value = side
        self.confidence = confidence
        self.leverage = leverage
        self.size = size
        self.entry_price = entry_price
        self.position_pct = position_pct
        self.strategy_type = strategy_type
        self.source_accuracy = 0.0
        self.regime_size_modifier = 1.0
        self.source = source
        self.context = dict(context or {})

    def validate(self):
        return True


def run_through_firewall(
    *,
    cfg=None,
    regime_data=None,
    open_positions=None,
    account_balance=None,
    require_live_balance=False,
    **signal_kwargs,
):
    """Build a real firewall, push one signal through full validate().

    Returns ``(passed: bool, reason: str)``.
    """
    from src.signals.decision_firewall import DecisionFirewall

    base_cfg = {
        "enable_predictive_derisk": False,
        "funding_risk_enabled": False,
        "cooldown_seconds": 0,
        "same_side_cooldown_seconds": 0,
    }
    base_cfg.update(cfg or {})
    fw = DecisionFirewall(base_cfg)
    sig = _HarnessSignal(**signal_kwargs)
    with patch("src.signals.decision_firewall.db") as mock_db:
        mock_db.get_open_paper_trades.return_value = open_positions or []
        mock_db.get_paper_account.return_value = {"balance": 1_000_000}
        mock_db.audit_log = MagicMock()
        passed, reason = fw.validate(
            sig,
            regime_data=regime_data,
            open_positions=open_positions if open_positions is not None else [],
            account_balance=account_balance,
            require_live_balance=require_live_balance,
        )
    return passed, reason


# ── Exposure floor x tiny live wallet (the $2,629/$102 = 2570% bug) ──

def test_exposure_floor_unblocks_tiny_live_wallet():
    """$5k floor: a small position on a ~$102 live wallet must NOT be
    rejected by the leveraged-notional aggregate cap."""
    passed, reason = run_through_firewall(
        cfg={"max_aggregate_exposure": 1.50, "aggregate_exposure_floor_usd": 5000.0,
             "max_aggregate_margin_pct": 0.0},
        account_balance=102.0,
        size=0.001, entry_price=100_000.0, leverage=2,  # ~$200 leveraged notional
    )
    assert "exposure" not in reason.lower(), reason


def test_exposure_without_floor_still_caps_tiny_wallet():
    """Floor disabled -> the percentage cap must still bite (no
    regression): a large notional on $102 is rejected for exposure."""
    passed, reason = run_through_firewall(
        cfg={"max_aggregate_exposure": 1.50, "aggregate_exposure_floor_usd": 0.0,
             "max_aggregate_margin_pct": 0.0},
        account_balance=102.0,
        size=0.05, entry_price=100_000.0, leverage=2,  # ~$10k leveraged notional
    )
    assert passed is False
    assert "exposure" in reason.lower()


# ── Live-only balance: never silently fall back to the $10k paper basis ──

def test_require_live_balance_rejects_when_live_value_missing():
    passed, reason = run_through_firewall(
        cfg={"max_aggregate_exposure": 1.50},
        account_balance=None,
        require_live_balance=True,
    )
    assert passed is False
    assert "live balance" in reason.lower() or "paper-account fallback" in reason.lower()


# ── Market-side guard x regime (BULLISH shown but LONG vetoed) ──

def test_market_guard_allows_high_conviction_options_long_vs_downtrend():
    """The fix: a 100%-conviction options-flow LONG must NOT be vetoed
    by a lone trending_down@74% regime read."""
    passed, reason = run_through_firewall(
        cfg={"market_side_guard_enabled": True, "market_side_guard_min_confidence": 0.60},
        regime_data={"overall_regime": "trending_down", "overall_confidence": 0.74},
        account_balance=1_000_000,
        source="options_flow", side="long", confidence=1.0,
    )
    assert "blocks long" not in reason.lower(), reason


def test_market_guard_still_blocks_long_into_confirmed_crash():
    """Crash carve-out intact: one bullish print cannot buy a confirmed
    high-confidence crash."""
    passed, reason = run_through_firewall(
        cfg={"market_side_guard_enabled": True, "market_side_guard_min_confidence": 0.60},
        regime_data={"overall_regime": "crash", "overall_confidence": 0.92},
        account_balance=1_000_000,
        source="options_flow", side="long", confidence=1.0,
    )
    assert passed is False
    assert "blocks long" in reason.lower()
