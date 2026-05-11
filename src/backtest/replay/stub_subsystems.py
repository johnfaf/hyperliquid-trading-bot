"""Neutral stand-ins for subsystems whose live data sources we cannot replay.

Each stub implements ONLY the methods that trading_cycle / decision_firewall /
paper_trader actually call. They return values that would produce the same
behaviour as if the live subsystem had simply observed nothing interesting:
no signals, neutral sentiment, "normal" macro risk, no event blocks, healthy
venues.

Crucially, each stub records every call into a `STUBBED` counter so the
harness operator can audit which code paths the replay actually exercised.
This is the difference between "the bot wasn't affected by polymarket" and
"the bot never even asked polymarket about anything" -- the second is what's
true here.

If we later want to replay one of these (e.g. polymarket history), the
replacement just needs to implement the same surface.
"""
from __future__ import annotations

import logging
import threading
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class _StubBase:
    """Thread-safe call counter shared by every stub."""

    def __init__(self, name: str):
        self._stub_name = name
        self._lock = threading.Lock()
        self._calls: Counter[str] = Counter()

    def _record(self, method: str) -> None:
        with self._lock:
            self._calls[method] += 1

    def get_stub_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {"name": self._stub_name, "calls": dict(self._calls)}


# --- polymarket ----------------------------------------------------

class StubPolymarketScanner(_StubBase):
    """Polymarket prediction-market signals + sentiment.

    Methods exercised by trading_cycle:
      - get_market_sentiment() -> dict
      - generate_signals(hl_regime=...) -> list[dict]
    """

    def __init__(self):
        super().__init__("polymarket")

    def get_market_sentiment(self) -> Dict[str, Any]:
        self._record("get_market_sentiment")
        return {
            "stubbed": True,
            "overall_bias": "neutral",
            "btc_up_prob": 0.5,
            "eth_up_prob": 0.5,
            "confidence": 0.0,
            "markets_used": 0,
        }

    def generate_signals(self, hl_regime: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        self._record("generate_signals")
        return []


# --- options flow --------------------------------------------------

class StubOptionsFlowScanner(_StubBase):
    """Options conviction scanner. Exposes `top_convictions` list + `scan_flow`."""

    def __init__(self):
        super().__init__("options_scanner")
        self.top_convictions: List[Any] = []

    def scan_flow(self) -> Dict[str, Any]:
        self._record("scan_flow")
        return {"stubbed": True, "convictions": []}


# --- macro regime --------------------------------------------------

class StubMacroRegime(_StubBase):
    """Web-scraped macro sentiment. Stubbed to NEUTRAL.

    `_apply_macro_regime_overlay` in trading_cycle.py short-circuits when the
    posture is `normal`, so this is effectively a no-op overlay. Any real
    macro signal during the replay window is invisible -- callers should
    note that in the harness report.
    """

    def __init__(self):
        super().__init__("macro_regime")

    def get_risk_posture(self) -> Dict[str, Any]:
        self._record("get_risk_posture")
        return {
            "stubbed": True,
            "macro_risk_level": "normal",
            "macro_score": 0.0,
            "size_modifier": 1.0,
            "confidence_drag": 0.0,
            "block_new_entries": False,
            "reasons": [],
        }


# --- event scanner -------------------------------------------------

class StubEventScanner(_StubBase):
    """Macro event / exchange status scanner. Stubbed to NEVER BLOCK.

    The firewall's `_apply_event_risk` reads `get_risk_state(coin)` and either
    blocks the trade, applies size/confidence multipliers, or passes through.
    A stub that returns no block + no degrade is the most permissive option;
    we surface in the report that no event filter was actually applied.
    """

    def __init__(self):
        super().__init__("event_scanner")

    def get_risk_state(self, coin: str) -> Dict[str, Any]:
        self._record("get_risk_state")
        return {
            "stubbed": True,
            "block_new_entries": False,
            "degrade": False,
            "confidence_multiplier": 1.0,
            "size_multiplier": 1.0,
            "reasons": [],
        }

    def has_blocking_event(self, *args: Any, **kwargs: Any) -> bool:
        self._record("has_blocking_event")
        return False


# --- exchange aggregator -------------------------------------------

class StubExchangeAggregator(_StubBase):
    """Cross-venue volume/funding/orderbook aggregator. Returns neutral overview."""

    def __init__(self):
        super().__init__("exchange_agg")

    def get_market_overview(self) -> Dict[str, Any]:
        self._record("get_market_overview")
        return {
            "stubbed": True,
            "venues_healthy": [],
            "venues_degraded": [],
            "global_bias": "neutral",
            "by_coin": {},
        }


# --- multi-exchange scanner ----------------------------------------

class StubMultiScanner(_StubBase):
    """Cross-venue scanner -- arb, funding spread, signal confirmation.

    Methods exercised by trading_cycle:
      - check_health() -> dict
      - get_common_markets() -> list
      - scan_funding_arb() -> list
      - inject_lighter_strategies(strategies) -> list
      - confirm_signals(signals) -> list (just pass-through; "confirmed" means
        the multi-scanner saw the same signal at another venue; we have no
        other venue, so we pass through unchanged with a STUBBED tag)
    """

    def __init__(self):
        super().__init__("multi_scanner")

    def check_health(self) -> Dict[str, Any]:
        self._record("check_health")
        return {"stubbed": True, "healthy": True, "venues": {}}

    def get_common_markets(self) -> List[str]:
        self._record("get_common_markets")
        return []

    def scan_funding_arb(self) -> List[Dict[str, Any]]:
        self._record("scan_funding_arb")
        return []

    def inject_lighter_strategies(self, strategies: Any) -> List[Any]:
        self._record("inject_lighter_strategies")
        return []

    def confirm_signals(self, signals: List[Any]) -> List[Any]:
        self._record("confirm_signals")
        # Pass through; we have no other venue to confirm with.
        return signals or []


# --- predictive forecaster ----------------------------------------

class StubPredictiveForecaster(_StubBase):
    """Predictive regime forecaster -- rule-based + (optionally) ML.

    In replay v1 we stub it to "no prediction" which makes the cycle's
    `_reconcile_regimes` fall through to using the rule-based regime
    detector's output as-is.
    """

    def __init__(self):
        super().__init__("predictive_forecaster")

    def predict_regime(self, coin: str) -> Dict[str, Any]:
        self._record("predict_regime")
        return {
            "stubbed": True,
            "coin": coin,
            "predicted_regime": None,
            "confidence": 0.0,
            "horizon_minutes": 0,
        }

    def update_options_flow(self, *args: Any, **kwargs: Any) -> None:
        self._record("update_options_flow")

    def update_polymarket_sentiment(self, *args: Any, **kwargs: Any) -> None:
        self._record("update_polymarket_sentiment")


# --- whale scanner -------------------------------------------------

class StubWhaleScanner(_StubBase):
    """Crypto.com whale scanner. Runs on the fast-cycle, which we don't simulate
    in v1 -- but if anything probes the scanner directly we return empty."""

    def __init__(self):
        super().__init__("whale_scanner")

    def get_pending(self) -> List[Any]:
        self._record("get_pending")
        return []

    def scan(self) -> List[Any]:
        self._record("scan")
        return []


# --- cross-venue hedger --------------------------------------------

class StubCrossVenueHedger(_StubBase):
    """Issues live Kraken hedges in production. Forced dry_run + no-op here."""

    def __init__(self):
        super().__init__("cross_venue_hedger")
        self.dry_run = True

    def evaluate(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self._record("evaluate")
        return {"stubbed": True, "action": "noop"}

    def get_state(self) -> Dict[str, Any]:
        self._record("get_state")
        return {"stubbed": True, "open_hedges": []}


def all_stubs() -> Dict[str, _StubBase]:
    """Convenience factory: return a fresh stub for each subsystem the
    REPLAY profile expects to swap. Used by subsystem_registry."""
    return {
        "polymarket": StubPolymarketScanner(),
        "options_scanner": StubOptionsFlowScanner(),
        "macro_regime": StubMacroRegime(),
        "event_scanner": StubEventScanner(),
        "exchange_agg": StubExchangeAggregator(),
        "multi_scanner": StubMultiScanner(),
        "predictive_forecaster": StubPredictiveForecaster(),
        "whale_scanner": StubWhaleScanner(),
        "cross_venue_hedger": StubCrossVenueHedger(),
    }
