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


# ── Bootstrap tier ────────────────────────────────────────────
# Sources that don't yet meet the standard 30-trades / 45%-WR bar can
# earn a small live foothold by passing a SMALLER sample at a HIGHER
# accuracy bar (defaults: 5 trades, 60% WR -> 0.25x size). This is
# guarded by PROMOTION_BOOTSTRAP_TIER_ENABLED (default False); when
# off the helpers below are never called.
#
# Reason format: ``ok_bootstrap:scale=<float>`` -- callers (currently
# only ``src/core/live_execution.py``) parse the scale from the reason
# and multiply the rescaled trade size by it before sending to live.
# Using the reason string as the side-channel keeps the public
# ``is_live_promotable`` API at ``(bool, str)`` so existing call sites
# and tests stay unchanged.

_BOOTSTRAP_PREFIX = "ok_bootstrap:scale="


def _bootstrap_agent_score_ok(
    source_key: str,
    *,
    min_trades: int,
    min_win_rate: float,
    scale: float,
) -> Tuple[bool, str]:
    """Lower-sample / higher-accuracy variant of ``_agent_score_promotion_ok``.

    Returns ``(True, "ok_bootstrap:scale=X")`` on pass so the caller can
    apply the size scale modifier.  All other paths reuse the standard
    rejection reasons so logs stay consistent between tiers.
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
        return False, f"insufficient_signals_bootstrap:{total}/{min_trades}"
    if accuracy < min_win_rate:
        return False, (
            f"bootstrap_accuracy_too_low:{accuracy:.2%}<{min_win_rate:.2%}"
        )
    return True, f"{_BOOTSTRAP_PREFIX}{scale:.4f}"


def get_bootstrap_scale(reason: str) -> float:
    """Extract the size scale modifier from an ``ok_bootstrap:scale=`` reason.

    Returns the parsed scale clamped to ``(0, 1]``.  Any parse error
    yields 1.0 (full size) -- this is the conservative default for
    *standard* promotions; bootstrap-promoted trades should already
    have passed parse-time validation in :func:`_bootstrap_agent_score_ok`.
    Callers that need to fail-closed should check whether the reason
    starts with :data:`_BOOTSTRAP_PREFIX` before applying the scale.
    """
    if not isinstance(reason, str) or not reason.startswith(_BOOTSTRAP_PREFIX):
        return 1.0
    raw = reason[len(_BOOTSTRAP_PREFIX):]
    # Strip anything after the first comma in case future extensions
    # append additional key=value pairs to the reason.
    raw = raw.split(",", 1)[0]
    try:
        scale = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if scale <= 0.0:
        return 1.0
    if scale > 1.0:
        return 1.0
    return scale


def _dsr_promotion_ok(
    strategy_id: Any, *, num_trials: int, min_obs: int
) -> Tuple[bool, str]:
    """A5: require the strategy's recent paper-trade P&L Sharpe to be
    statistically significant after deflating for selection bias.

    Strictly conservative: returns ``(True, ...)`` on any missing /
    insufficient data or computation error (fail OPEN -- defer to the
    base gate) and only returns ``(False, ...)`` with positive evidence
    the edge is NOT significant. Raw P&L is a valid input: Sharpe is
    scale-invariant so the significance test is unaffected by notional.
    """
    if strategy_id in (None, ""):
        return True, "dsr_skip_no_strategy_id"
    try:
        with db.get_connection(for_read=True) as conn:
            rows = conn.execute(
                "SELECT pnl FROM paper_trades "
                "WHERE strategy_id = ? AND status != 'open' "
                "ORDER BY id DESC LIMIT 500",
                (strategy_id,),
            ).fetchall()
    except Exception as exc:
        logger.debug("DSR paper_trades lookup failed for %s: %s", strategy_id, exc)
        return True, "dsr_lookup_failed"
    returns: list = []
    for r in rows or []:
        try:
            val = r["pnl"] if isinstance(r, dict) else (
                r[0] if not hasattr(r, "keys") else r["pnl"]
            )
            returns.append(float(val or 0.0))
        except (TypeError, ValueError, KeyError, IndexError):
            continue
    if len(returns) < max(2, int(min_obs)):
        return True, f"dsr_insufficient:{len(returns)}/{min_obs}"
    try:
        from src.learning.promotion_stats import deflated_sharpe

        res = deflated_sharpe(returns, num_trials=max(1, int(num_trials)))
    except Exception as exc:
        logger.debug("DSR computation failed for %s: %s", strategy_id, exc)
        return True, "dsr_compute_failed"
    if not getattr(res, "significant_at_95", False):
        return False, (
            f"dsr_not_significant:dsr={res.deflated_sharpe:.2f},"
            f"p={res.p_value:.3f},n={res.num_observations},"
            f"trials={res.num_trials}"
        )
    return True, f"ok_dsr:dsr={res.deflated_sharpe:.2f},p={res.p_value:.3f}"


def _drift_promotion_ok(*, max_age_hours: float) -> Tuple[bool, str]:
    """Consult FeatureDriftMonitor reports: any recent block kills promotion.

    The drift_monitor module computes a ``blocks_promotion`` flag on every
    DriftReport and persists it to ``learning_drift_reports`` -- but until
    this hook landed, NO code consulted that flag. Reports were emitted
    into a void.

    This helper queries the most recent persisted DriftReport. If it has
    ``blocks_promotion = TRUE`` and was generated within ``max_age_hours``,
    we block the promotion. Otherwise we fail OPEN (no recent block, or
    a stale block that is presumably no longer applicable).

    Strictly conservative on the wiring side: ANY exception (DB error,
    schema mismatch, parse failure) returns ``(True, "drift_check_skipped")``
    so a broken drift query never breaks the promotion path.
    """
    try:
        with db.get_connection(for_read=True) as conn:
            # Look for any recent BLOCKING report, not just the latest
            # report. The pre-fix query took the most recent row only --
            # if dataset A's latest was non-blocking but dataset B had a
            # fresh blocks_promotion=TRUE row, the gate would miss it.
            row = conn.execute(
                """
                SELECT created_at, blocks_promotion, status, summary,
                       current_dataset_id, baseline_dataset_id
                  FROM learning_drift_reports
                 WHERE blocks_promotion = TRUE
                 ORDER BY created_at DESC
                 LIMIT 1
                """,
            ).fetchone()
    except Exception as exc:
        logger.debug("drift_check lookup failed: %s", exc)
        return True, "drift_check_skipped:lookup_failed"

    if row is None:
        return True, "drift_check_skipped:no_reports"

    try:
        # Row may be a Mapping or a tuple depending on the DB adapter.
        if hasattr(row, "keys"):
            created_at = row["created_at"]
            blocks = bool(row["blocks_promotion"])
            status = str(row["status"] or "")
            summary = row["summary"]
        else:
            created_at, blocks, status, summary, *_ = row
            blocks = bool(blocks)
            status = str(status or "")
    except Exception as exc:
        logger.debug("drift_check row parse failed: %s", exc)
        return True, "drift_check_skipped:parse_failed"

    if not blocks:
        return True, f"ok_drift:status={status}"

    # blocks_promotion = TRUE. Check the window.
    try:
        from datetime import datetime, timezone
        if isinstance(created_at, str):
            created_dt = datetime.fromisoformat(
                created_at.replace("Z", "+00:00"),
            )
        else:
            created_dt = created_at
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - created_dt).total_seconds() / 3600.0
    except Exception as exc:
        logger.debug("drift_check age computation failed: %s", exc)
        # Conservative on age failure with a blocking flag set: BLOCK,
        # because we'd rather hold a promotion than auto-promote past
        # an unparseable drift block.
        return False, "drift_blocked:age_unparseable"

    if age_h <= float(max_age_hours):
        return False, (
            f"drift_blocked:status={status},age={age_h:.1f}h"
            f"<={max_age_hours:.0f}h"
        )
    return True, f"ok_drift:stale_block_age={age_h:.1f}h>{max_age_hours:.0f}h"


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
            ok, reason = _strategy_promotion_ok(
                strategy,
                min_trades=min_trades,
                min_win_rate=min_win_rate,
                min_score=min_score,
            )
            # A5 (default OFF): only ever downgrade an approved promotion.
            if ok and bool(getattr(config, "PROMOTION_REQUIRE_DSR", False)):
                dsr_ok, dsr_reason = _dsr_promotion_ok(
                    strategy_id,
                    num_trials=int(getattr(config, "PROMOTION_DSR_NUM_TRIALS", 50)),
                    min_obs=int(getattr(config, "PROMOTION_DSR_MIN_OBS", 20)),
                )
                if not dsr_ok:
                    return False, dsr_reason
            # Drift gate (default OFF): consult learning_drift_reports.
            # Only downgrades an approved promotion; never upgrades. The
            # check fails OPEN on any error so a broken drift query
            # cannot block all promotions.
            if ok and bool(getattr(config, "PROMOTION_REQUIRE_DRIFT_OK", False)):
                drift_ok, drift_reason = _drift_promotion_ok(
                    max_age_hours=float(
                        getattr(config, "PROMOTION_DRIFT_MAX_AGE_HOURS", 24.0)
                    ),
                )
                if not drift_ok:
                    return False, drift_reason
            return ok, reason

    # Path 2: copy-trade source_trader (agent_scorer)
    source_trader = (
        metadata.get("source_trader")
        or trade.get("source_trader")
        or ""
    )
    source = str(metadata.get("source") or trade.get("source") or "").strip().lower()
    if source == "copy_trade" and source_trader:
        source_key = f"copy_trade:{str(source_trader).strip().lower()}"
        ok, reason = _agent_score_promotion_ok(
            source_key,
            min_trades=min_trades,
            min_win_rate=min_win_rate,
        )
        if not ok:
            ok, reason = _maybe_bootstrap_tier(source_key, base_reason=reason)
        return ok, reason

    # Path 3: strategy_type tag (agent_scorer under strategy:<type>)
    strategy_type = str(
        metadata.get("strategy_type") or trade.get("strategy_type") or ""
    ).strip().lower()
    if strategy_type and strategy_type not in {"unknown", "untagged"}:
        source_key = f"strategy:{strategy_type}"
        ok, reason = _agent_score_promotion_ok(
            source_key,
            min_trades=min_trades,
            min_win_rate=min_win_rate,
        )
        if not ok:
            ok, reason = _maybe_bootstrap_tier(source_key, base_reason=reason)
        return ok, reason

    # No promotion data resolvable — fail closed for live.
    return False, "no_promotion_data"


def _maybe_bootstrap_tier(
    source_key: str, *, base_reason: str,
) -> Tuple[bool, str]:
    """Try the bootstrap tier when the standard tier rejected the source.

    Returns the standard ``base_reason`` unchanged when the bootstrap
    tier is off OR when the bootstrap tier also rejects.  Only returns
    ``(True, "ok_bootstrap:scale=X")`` when the bootstrap tier passes.

    Sentinel reasons that mean "data missing, not 'fails by policy'" --
    ``agent_score_lookup_failed``, ``missing_source_key`` -- are passed
    through without retrying so a transient DB error never escalates
    into a bootstrap promotion.
    """
    if not bool(getattr(config, "PROMOTION_BOOTSTRAP_TIER_ENABLED", False)):
        return False, base_reason
    if base_reason in ("agent_score_lookup_failed", "missing_source_key"):
        return False, base_reason

    boot_min_trades = int(
        getattr(config, "PROMOTION_BOOTSTRAP_MIN_TRADES", 5)
    )
    boot_min_wr = float(
        getattr(config, "PROMOTION_BOOTSTRAP_MIN_WIN_RATE", 0.60)
    )
    boot_scale = float(
        getattr(config, "PROMOTION_BOOTSTRAP_SIZE_SCALE", 0.25)
    )
    # Defense in depth: the env loader clamps these, but a direct
    # ``setattr(config, ...)`` (tests, hotfix consoles) can bypass the
    # clamp.  Re-clamp here so we never send full-size when the operator
    # meant a fraction.
    boot_scale = max(0.01, min(1.0, boot_scale))
    boot_min_wr = max(0.0, min(1.0, boot_min_wr))
    boot_min_trades = max(1, boot_min_trades)

    return _bootstrap_agent_score_ok(
        source_key,
        min_trades=boot_min_trades,
        min_win_rate=boot_min_wr,
        scale=boot_scale,
    )
