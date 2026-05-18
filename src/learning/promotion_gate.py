"""Walk-forward promotion gate for live trades.

Before a paper trade is mirrored to live, this gate checks that its
underlying strategy/source has accumulated enough out-of-sample
evidence to justify real money. Trades whose source hasn't earned
live status stay paper-only.

This is a *defense-in-depth* layer on top of the existing strategy
quarantine (``is_strategy_live_eligible`` / ``strategy_quarantine_reason``
in ``src.data.database``). Quarantine catches *broken* or *synthetic*
strategies; this gate catches strategies that are well-formed but
simply haven't proven themselves yet.

Design notes:
  - Fails CLOSED for live by default. If we can't resolve the source's
    track record, the trade stays paper. The "if it can't be measured
    it can't be sized" principle.
  - Two source paths supported in MVP: strategy-id-backed (look up the
    DB strategy row) and copy-trade (look up the agent_scorer entry).
  - Thresholds are all env-configurable. Disable the whole gate with
    ``LIVE_PROMOTION_GATE_ENABLED=false`` if you need to bypass during
    bootstrap.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Tuple

import config
from src.data import database as db

logger = logging.getLogger(__name__)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


def _strategy_promotion_ok(
    strategy: Dict[str, Any],
    *,
    min_trades: int,
    min_win_rate: float,
    min_score: float,
) -> Tuple[bool, str]:
    if not strategy:
        return False, "strategy_not_found"
    # Quarantine check first — synthetic / placeholder / bot-source
    # strategies never reach live regardless of trade history.
    quarantine_reason = db.strategy_quarantine_reason(strategy)
    if quarantine_reason:
        return False, f"quarantined:{quarantine_reason}"

    trade_count = int(_float(strategy.get("trade_count"), 0.0))
    win_rate = _float(strategy.get("win_rate"), 0.0)
    if win_rate > 1.5:
        win_rate /= 100.0  # tolerate stored-as-percent rows
    score = _float(strategy.get("current_score"), 0.0)

    if trade_count < min_trades:
        return False, f"insufficient_trades:{trade_count}/{min_trades}"
    if win_rate < min_win_rate:
        return False, f"win_rate_too_low:{win_rate:.2%}<{min_win_rate:.2%}"
    if score < min_score:
        return False, f"score_too_low:{score:.3f}<{min_score:.3f}"
    return True, "ok"


def _agent_score_promotion_ok(
    source_key: str,
    *,
    min_trades: int,
    min_win_rate: float,
) -> Tuple[bool, str]:
    """Look up the agent_scorer row directly from DB.

    Loaded ad-hoc (not via the in-memory scorer) so the gate works even
    when the live-mirror path doesn't have a scorer instance handy.
    """
    if not source_key:
        return False, "missing_source_key"
    try:
        with db.get_connection(for_read=True) as conn:
            row = conn.execute(
                "SELECT total_signals, correct_signals, accuracy "
                "FROM agent_scores WHERE source_key = ?",
                (source_key,),
            ).fetchone()
    except Exception as exc:
        logger.debug("agent_scores lookup failed for %s: %s", source_key, exc)
        return False, "agent_score_lookup_failed"
    if not row:
        return False, "no_agent_score_row"
    row_dict = dict(row)
    total = int(_float(row_dict.get("total_signals"), 0.0))
    accuracy = _float(row_dict.get("accuracy"), 0.0)
    if total < min_trades:
        return False, f"insufficient_signals:{total}/{min_trades}"
    if accuracy < min_win_rate:
        return False, f"accuracy_too_low:{accuracy:.2%}<{min_win_rate:.2%}"
    return True, "ok"


def is_live_promotable(trade: Dict[str, Any]) -> Tuple[bool, str]:
    """Return ``(promotable, reason)`` for a paper trade about to mirror live.

    Resolution order:
      1. Direct ``strategy_id`` on the trade — most reliable.
      2. Copy-trade ``source_trader`` in metadata — agent_scorer lookup.
      3. ``strategy_type`` in metadata — best-effort agent_scorer lookup
         under ``strategy:<type>`` key.

    Returns ``(False, "gate_disabled")`` and lets the caller through when
    ``config.LIVE_PROMOTION_GATE_ENABLED`` is false.
    """
    if not bool(getattr(config, "LIVE_PROMOTION_GATE_ENABLED", True)):
        return True, "gate_disabled"

    min_trades = int(getattr(config, "LIVE_PROMOTION_MIN_TRADES", 30))
    min_win_rate = float(getattr(config, "LIVE_PROMOTION_MIN_WIN_RATE", 0.45))
    min_score = float(getattr(config, "LIVE_PROMOTION_MIN_SCORE", 0.20))

    metadata = _parse_metadata(trade.get("metadata", {}))

    # Path 1: strategy_id on trade
    strategy_id = trade.get("strategy_id") or metadata.get("strategy_id")
    if strategy_id:
        try:
            strategy = db.get_strategy(strategy_id)
        except Exception as exc:
            logger.debug("get_strategy(%s) failed: %s", strategy_id, exc)
            strategy = None
        if strategy:
            return _strategy_promotion_ok(
                strategy,
                min_trades=min_trades,
                min_win_rate=min_win_rate,
                min_score=min_score,
            )

    # Path 2: copy-trade source_trader (agent_scorer)
    source_trader = (
        metadata.get("source_trader")
        or trade.get("source_trader")
        or ""
    )
    source = str(metadata.get("source") or trade.get("source") or "").strip().lower()
    if source == "copy_trade" and source_trader:
        return _agent_score_promotion_ok(
            f"copy_trade:{str(source_trader).strip().lower()}",
            min_trades=min_trades,
            min_win_rate=min_win_rate,
        )

    # Path 3: strategy_type tag (agent_scorer under strategy:<type>)
    strategy_type = str(
        metadata.get("strategy_type") or trade.get("strategy_type") or ""
    ).strip().lower()
    if strategy_type and strategy_type not in {"unknown", "untagged"}:
        return _agent_score_promotion_ok(
            f"strategy:{strategy_type}",
            min_trades=min_trades,
            min_win_rate=min_win_rate,
        )

    # No promotion data resolvable — fail closed for live.
    return False, "no_promotion_data"
