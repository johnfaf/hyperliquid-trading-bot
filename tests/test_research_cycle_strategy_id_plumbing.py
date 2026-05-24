"""Discovery -> research_cycle -> strategy_identifier metadata plumbing.

Locks the fix for a silent bug observed in production: every newly
identified strategy was getting classified as ``momentum_*`` regardless
of the actual trader's behaviour, because ``research_cycle.py``
hardcoded ``trading_frequency = "unknown"`` when building the
StrategyIdentifier profile.

The detectors in ``strategy_identifier.py`` gate on
``trade_analysis["trading_frequency"]``:

  * ``_detect_scalping``       requires ``frequency == "scalper"``
  * ``_detect_swing_trading``  requires ``frequency in {"swing_trader",
                                                         "position_trader"}``

With the field permanently "unknown", those branches were structurally
unreachable -- half the strategy pool was silently disabled.

This test verifies:
  1. The discovery cycle persists ``trading_frequency`` (plus other
     trade-analysis characteristics) to the trader's metadata dict.
  2. The research cycle reads those fields from metadata when building
     the StrategyIdentifier profile (instead of injecting "unknown").
  3. The detector receives the real frequency value through the chain.
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def test_discovery_persists_trading_frequency_to_metadata():
    """trader_discovery's deep-analysis path must persist the computed
    ``trading_frequency`` (and related fields) into the metadata dict
    it hands to ``db.upsert_trader``.  Without this, research_cycle
    has nothing to read and falls back to ``unknown``.

    Verifies by checking the persisted-metadata fields appear in the
    same vicinity as the ``db.upsert_trader`` call (within ~80 lines
    after).  Looser than a full AST parse but catches a missing field
    without breaking on whitespace / comment churn.
    """
    src = _read("src/discovery/trader_discovery.py")
    upsert_idx = src.find("db.upsert_trader(")
    assert upsert_idx >= 0, (
        "Could not find db.upsert_trader call in trader_discovery.py"
    )
    # Look ahead a generous window to catch the metadata block.
    window = src[upsert_idx:upsert_idx + 6000]
    required_fields = [
        "trading_frequency",
        "avg_trade_size",
        "trades_per_day",
        "liquidations",
        "avg_win",
        "avg_loss",
        "raw_fill_count",
        "closed_trade_count",
        "sample_is_capped",
    ]
    missing = [k for k in required_fields if f'"{k}"' not in window]
    assert not missing, (
        f"trader_discovery.py db.upsert_trader's metadata block is "
        f"missing: {missing}.  Without these fields, research_cycle's "
        f"StrategyIdentifier profile falls back to placeholder values "
        f"and the corresponding detector branches become unreachable."
    )


def test_research_cycle_reads_trading_frequency_from_metadata():
    """research_cycle.py's Phase 2 must read ``trading_frequency`` from
    trader_meta, NOT hardcode it to ``"unknown"``.  The hardcoded
    string was the visible symptom of the bug -- it made every
    detected strategy fall through to ``momentum_*``."""
    src = _read("src/core/cycles/research_cycle.py")

    # 1. The hardcoded string literal must NOT appear as a value in the
    #    trade_analysis dict.
    hardcoded = re.search(
        r'"trading_frequency"\s*:\s*"unknown"',
        src,
    )
    assert hardcoded is None, (
        'research_cycle.py still hardcodes "trading_frequency": "unknown" '
        "-- the StrategyIdentifier scalper / swing_trader / "
        "position_trader detectors will remain unreachable.  Read from "
        "trader_meta.get(\"trading_frequency\", \"unknown\") instead."
    )

    # 2. The trader_meta read for trading_frequency must be present.
    fetched = re.search(
        r'trader_meta\.get\(\s*"trading_frequency"',
        src,
    )
    assert fetched is not None, (
        "research_cycle.py must read trading_frequency from trader_meta "
        "so the persisted discovery output reaches strategy_identifier."
    )


def test_strategy_identifier_detectors_gate_on_trading_frequency():
    """Confirm the bug's root condition: scalping / swing_trading
    detectors require specific ``trading_frequency`` values.  If this
    contract changes, the plumbing tests above need to be revisited."""
    src = _read("src/analysis/strategy_identifier.py")
    # Scalping requires == "scalper"
    assert 'frequency == "scalper"' in src, (
        "strategy_identifier._detect_scalping no longer gates on "
        'frequency == "scalper".  Update this test + recheck the '
        "metadata plumbing is still necessary."
    )
    # Swing trading requires the {swing_trader, position_trader} set
    assert 'frequency in ("swing_trader", "position_trader")' in src or \
        'frequency in (\'swing_trader\', \'position_trader\')' in src, (
        "strategy_identifier._detect_swing_trading no longer gates on "
        "frequency in (swing_trader, position_trader)."
    )


def test_h27_profit_factor_fix_preserved():
    """Regression guard for the H27 fix: ``profit_factor`` must stay
    ``None`` in research_cycle's Phase 2 (don't reintroduce the
    synthetic PF=1.5 default that gave every trader a free score
    boost).  This test fails if someone "helpfully" sets a numeric
    default while editing the trade_analysis dict for the new
    trading_frequency plumbing."""
    src = _read("src/core/cycles/research_cycle.py")
    bad_default = re.search(
        r'"profit_factor"\s*:\s*\d+(?:\.\d+)?',
        src,
    )
    assert bad_default is None, (
        f"research_cycle.py reintroduced a hardcoded numeric "
        f"profit_factor at: {bad_default.group(0)}.  The H27 fix "
        f"requires this to stay None so the strategy scorer can tell "
        f"'no measurement' apart from a real positive PF."
    )
