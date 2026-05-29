"""Verify the feature-precompute cap covers a typical-prod fan-out.

Phase 5 root-cause: production audit found 35% of decision_outcomes
rows had empty ``features`` because the previous cap of 12 cut
alphabetically-late coins (VVV, ZEC, ...) from the precompute set.
Each one then reached the journal with no features attached, which
in turn drove the learning quality auditor's
``missing_feature_ratio`` failure (0.49 vs strict 0.15).

These tests pin the new default cap (32) and validate the resolve
loop's behaviour on a realistic strategy fan-out.
"""
from __future__ import annotations

from typing import Dict, List


def _resolve_feature_coins(strategies: List[Dict], *, cap: int = 32) -> List[str]:
    """Mirror the in-loop code at src/trading/paper_trader.py:746 so
    we can assert on the resolved set without needing a full
    PaperTrader fixture (which pulls in DB / API surfaces).

    Keep this in sync with the production loop.
    """
    _core = {"BTC", "ETH", "SOL"}
    _extra: set = set()
    for _strat in strategies or []:
        _params = _strat.get("parameters", {}) or {}
        _pre = str(_strat.get("_decision_coin", "") or "").strip().upper()
        if _pre and _pre != "UNKNOWN":
            _extra.add(_pre)
        _coins = (
            _params.get("coins")
            or _params.get("coins_traded")
            or _params.get("coin")
            or []
        )
        if isinstance(_coins, str):
            _coins = [_coins]
        for _c in _coins:
            _cu = str(_c or "").strip().upper()
            if _cu and _cu != "UNKNOWN":
                _extra.add(_cu)
    cap_eff = max(3, int(cap))
    feature_coins = ["BTC", "ETH", "SOL"] + sorted(_extra - _core)
    return feature_coins[:cap_eff]


def _strategy(coin: str) -> Dict:
    return {"_decision_coin": coin, "parameters": {}}


# ── Default cap ────────────────────────────────────────────────


def test_default_cap_is_32():
    """Sanity: the production default after Phase 5 is 32."""
    from src.trading import paper_trader  # noqa: F401
    # Read straight from the config module (no env override).
    import config
    cap = int(getattr(config, "PAPER_FEATURE_PRECOMPUTE_MAX_COINS", -1))
    assert cap >= 32, f"default cap regressed to {cap}; Phase 5 expected >= 32"


# ── Realistic fan-out cases ────────────────────────────────────


_LATE_ALPHA_COINS = ["VVV", "ZEC", "ZRX", "ZK"]


def test_late_alphabet_coin_survives_under_typical_fanout():
    """Bot's typical 25-strategy fan-out used to cut VVV / ZEC with
    cap=12.  Cap=32 must keep them."""
    # Simulate 25 strategies across an alphabetic spread + a few late
    # ones we explicitly care about.
    coins = sorted([
        "AAVE", "ARB", "AVAX", "BCH", "BIO", "BNB", "DOGE", "DYDX",
        "HYPE", "LIT", "LINK", "MON", "OP", "PURR", "SUI", "TAO",
        "TRUMP", "TST", "XRP",
    ] + _LATE_ALPHA_COINS)
    strategies = [_strategy(c) for c in coins]

    out = _resolve_feature_coins(strategies, cap=32)

    # All four late-alpha coins must be present.
    for c in _LATE_ALPHA_COINS:
        assert c in out, f"{c} was dropped from feature_coins (cap=32)"


def test_late_alphabet_dropped_under_legacy_cap_12():
    """Regression coverage: under the old cap (12), VVV/ZEC are
    silently cut.  This is the exact bug PR #42 fixes."""
    coins = sorted([
        "AAVE", "ARB", "AVAX", "BCH", "BIO", "BNB", "DOGE", "DYDX",
        "HYPE", "LIT", "LINK", "MON", "OP", "PURR", "SUI", "TAO",
        "TRUMP", "TST", "XRP",
    ] + _LATE_ALPHA_COINS)
    out = _resolve_feature_coins([_strategy(c) for c in coins], cap=12)
    # With cap=12, only 9 non-core coins survive after alphabetic sort.
    assert len(out) == 12
    assert "VVV" not in out
    assert "ZEC" not in out


def test_cap_keeps_core_plus_extras_in_order():
    """Core coins always come first; the extras fill the remaining slots."""
    strategies = [_strategy(c) for c in ["XRP", "ARB", "DOGE"]]
    out = _resolve_feature_coins(strategies, cap=10)
    assert out[:3] == ["BTC", "ETH", "SOL"]
    assert set(out[3:]) == {"ARB", "DOGE", "XRP"}


def test_cap_minimum_three():
    """Cap floor: even cap=1 keeps at least the 3 core coins."""
    strategies = [_strategy("HYPE")]
    out = _resolve_feature_coins(strategies, cap=1)
    assert out == ["BTC", "ETH", "SOL"]


def test_extras_via_parameters_coins_field():
    """Strategy 'parameters.coins' list contributes coins too."""
    strategies = [{
        "_decision_coin": "HYPE",
        "parameters": {"coins": ["ARB", "OP"]},
    }]
    out = _resolve_feature_coins(strategies, cap=10)
    assert {"ARB", "HYPE", "OP"} <= set(out)


def test_extras_via_parameters_coin_singular():
    """Singular 'parameters.coin' field also contributes."""
    strategies = [{"_decision_coin": "", "parameters": {"coin": "VVV"}}]
    out = _resolve_feature_coins(strategies, cap=10)
    assert "VVV" in out


def test_unknown_coins_filtered():
    """'UNKNOWN' string never enters the precompute set."""
    strategies = [
        _strategy("UNKNOWN"),
        {"_decision_coin": "", "parameters": {"coins": ["UNKNOWN", "DYDX"]}},
    ]
    out = _resolve_feature_coins(strategies, cap=10)
    assert "UNKNOWN" not in out
    assert "DYDX" in out


def test_no_strategies_returns_core_only():
    assert _resolve_feature_coins([], cap=10) == ["BTC", "ETH", "SOL"]
