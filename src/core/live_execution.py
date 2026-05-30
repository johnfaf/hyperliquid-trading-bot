"""
Helpers for live-vs-paper execution state.

These utilities keep the trading cycles anchored to exchange truth whenever
live trading is actually enabled and deployable.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.data import database as db
from src.data.hyperliquid_client import get_all_mids
from src.signals.signal_schema import signal_from_execution_dict

logger = logging.getLogger(__name__)


def _trade_metadata(trade: Dict) -> Dict:
    try:
        existing_meta = trade.get("metadata", {})
        if isinstance(existing_meta, str):
            existing_meta = json.loads(existing_meta or "{}")
        return dict(existing_meta or {})
    except Exception:
        return {}


def _notify_manual_close_detected(trade: Dict, exit_price: float) -> None:
    try:
        from src.notifications import telegram_bot as tg
    except Exception:
        return

    try:
        tg.notify_manual_close_detected(trade, exit_price=exit_price)
    except Exception as exc:
        logger.debug("Manual close notification failed for %s: %s", trade.get("coin", "?"), exc)


def _is_insufficient_margin_rejection(result) -> bool:
    """Return True when a rejection payload indicates insufficient margin."""
    if not isinstance(result, dict):
        return False

    reason = str(result.get("reason", "") or "").strip().lower()
    if reason == "insufficient_margin":
        return True

    messages: List[str] = []
    errors = result.get("errors")
    if isinstance(errors, list):
        messages.extend(str(item) for item in errors)
    elif errors:
        messages.append(str(errors))

    message = result.get("message")
    if message:
        messages.append(str(message))

    return any("insufficient margin" in msg.lower() for msg in messages)


# ── Live-only conviction + re-entry-cooldown gate ────────────────
#
# The CSV audit of the live wallet (May 2026) found ~70% of the loss
# was FEES from over-trading: 277 live opens, mostly small positions
# churned faster than they could clear the ~5bps round-trip taker fee,
# which bled the wallet ~$10-16 and tripped the drawdown kill switch.
#
# These two gates throttle the LIVE mirror path ONLY -- paper trades
# (and therefore the learning/calibration loop) keep running at full
# rate.  Live becomes selective: it only mirrors higher-conviction
# signals, and never re-enters the same coin within a cooldown.  Both
# default OFF so behaviour is unchanged unless an operator opts in.
import time as _time

_LAST_LIVE_MIRROR_TS: Dict[str, float] = {}


def _live_mirror_min_confidence() -> float:
    try:
        return max(0.0, min(1.0, float(os.environ.get("LIVE_MIRROR_MIN_CONFIDENCE", "0") or 0)))
    except (TypeError, ValueError):
        return 0.0


def _live_mirror_reentry_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("LIVE_MIRROR_MIN_REENTRY_SECONDS", "0") or 0))
    except (TypeError, ValueError):
        return 0.0


def _signal_bucket(live_signal) -> tuple[str, str, str]:
    """Derive (source, side, regime) for a live signal, namespaced like the
    edge-analysis buckets (e.g. ``strategy:momentum_long`` / ``copy_trade``).
    """
    src = getattr(live_signal, "source", None)
    src = getattr(src, "value", src)
    src = str(src or "").strip().lower()
    stype = str(getattr(live_signal, "strategy_type", "") or "").strip().lower()
    if src == "strategy" and stype:
        src = f"strategy:{stype}"
    side = getattr(live_signal, "side", "")
    side = str(getattr(side, "value", side) or "").strip().lower()
    regime = str(getattr(live_signal, "regime", "") or "").strip().lower()
    return src, side, regime


def _src_matches(pattern: str, src: str) -> bool:
    """A blocklist source matches the signal source either exactly, or as a
    namespace prefix (``copy_trade`` matches ``copy_trade:0xabc...``)."""
    if pattern == src:
        return True
    return ":" not in pattern and src.startswith(pattern + ":")


def _live_mirror_bucket_blocked(live_signal) -> tuple[bool, str]:
    """Block live mirroring for (source|side[|regime]) buckets the operator
    has flagged via LIVE_MIRROR_BUCKET_BLOCKLIST -- e.g. proven-loser buckets
    from scripts/analyze_edge.py (``copy_trade|short``, ``strategy:momentum_long|long``).
    Empty env => OFF.  ``*`` is a wildcard for side/regime.  Fail-open.
    """
    raw = str(os.environ.get("LIVE_MIRROR_BUCKET_BLOCKLIST", "") or "").strip()
    if not raw:
        return False, ""
    try:
        src, side, regime = _signal_bucket(live_signal)
        for entry in (e.strip().lower() for e in raw.split(",") if e.strip()):
            parts = entry.split("|")
            b_src = parts[0]
            b_side = parts[1] if len(parts) > 1 else "*"
            b_reg = parts[2] if len(parts) > 2 else "*"
            if (
                _src_matches(b_src, src)
                and b_side in ("*", side)
                and b_reg in ("*", regime)
            ):
                return True, f"bucket blocklisted ({entry})"
    except Exception:
        return False, ""  # fail-open
    return False, ""


def _live_mirror_conviction_gate(live_signal) -> tuple[bool, str]:
    """Return (allow, reason) for the live-only mirror gate: edge-bucket
    blocklist + conviction floor + re-entry cooldown.

    All components default OFF.  Never raises.
    """
    blocked, breason = _live_mirror_bucket_blocked(live_signal)
    if blocked:
        return False, breason
    min_conf = _live_mirror_min_confidence()
    reentry_s = _live_mirror_reentry_seconds()
    if min_conf <= 0.0 and reentry_s <= 0.0:
        return True, ""
    try:
        coin = str(getattr(live_signal, "coin", "") or "").upper()
        if min_conf > 0.0:
            conf = float(getattr(live_signal, "confidence", 0.0) or 0.0)
            if conf < min_conf:
                return False, f"conviction {conf:.2f} < live floor {min_conf:.2f}"
        if reentry_s > 0.0 and coin:
            last = _LAST_LIVE_MIRROR_TS.get(coin)
            if last is not None and (_time.time() - last) < reentry_s:
                wait = reentry_s - (_time.time() - last)
                return False, f"re-entry cooldown {wait:.0f}s left for {coin}"
    except Exception:
        return True, ""  # fail-open: gate must never block by erroring
    return True, ""


def _mark_live_mirror_time(live_signal) -> None:
    try:
        coin = str(getattr(live_signal, "coin", "") or "").upper()
        if coin:
            _LAST_LIVE_MIRROR_TS[coin] = _time.time()
    except Exception:
        pass


def _paper_trade_id_from_live_mirror(execution, live_signal=None) -> Optional[int]:
    """Extract the paper trade id that produced a live mirror attempt."""
    candidates = []
    if isinstance(execution, dict):
        candidates.extend(
            execution.get(key)
            for key in ("id", "paper_trade_id", "trade_id")
        )
    context = getattr(live_signal, "context", None)
    if isinstance(context, dict):
        candidates.extend(
            context.get(key)
            for key in ("paper_trade_id", "trade_id", "id")
        )

    for value in candidates:
        try:
            trade_id = int(value)
        except (TypeError, ValueError):
            continue
        if trade_id > 0:
            return trade_id
    return None


def _execution_entry_price(execution=None, live_signal=None) -> float:
    """Best-effort entry price for closing an unmirrored shadow row flat."""
    candidates = []
    if isinstance(execution, dict):
        candidates.extend(
            execution.get(key)
            for key in ("entry_price", "price", "fill_price")
        )
    for attr in ("entry_price", "price", "fill_price"):
        candidates.append(getattr(live_signal, attr, None))

    for value in candidates:
        try:
            price = float(value or 0.0)
        except (TypeError, ValueError):
            continue
        if price > 0:
            return price
    return 0.0


def is_confirmed_live_execution_result(live_result: Dict) -> bool:
    """Return True only when the live entry is durable enough to shadow."""
    if not isinstance(live_result, dict):
        return False
    status = str(live_result.get("status", "") or "").strip().lower()
    if status == "success":
        return True
    if status != "submitted":
        return False
    if str(live_result.get("venue", "") or "").lower() != "lighter":
        return False
    for leg_key in ("entry", "stop_loss", "take_profit"):
        leg = live_result.get(leg_key)
        if not isinstance(leg, dict):
            return False
        leg_status = str(leg.get("status", "") or "").strip().lower()
        if leg_status != "submitted":
            return False
    return True


def _is_confirmed_live_mirror_result(live_result: Dict) -> bool:
    """Backward-compatible alias for older tests/imports."""
    return is_confirmed_live_execution_result(live_result)


def _close_unmirrored_paper_trade(
    trade_id: Optional[int],
    live_signal=None,
    reason: str = "live_mirror_not_opened",
    live_result: Optional[Dict] = None,
    entry_price: Optional[float] = None,
) -> None:
    """Close a paper trade flat when its live mirror never became real."""
    if not trade_id:
        return

    flat_price = float(entry_price or 0.0)
    if flat_price <= 0:
        flat_price = _execution_entry_price(live_signal=live_signal)
    if flat_price <= 0:
        logger.warning(
            "Could not close unmirrored paper trade %s flat: missing entry price",
            trade_id,
        )
        return

    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "live_mirror": False,
        "live_mirror_attempted": True,
        "live_mirror_failed_at": now,
        "live_mirror_blocked_reason": reason,
        "live_mirror_closed_flat": True,
        "close_reason": reason,
        "gross_pnl_before_fees": 0.0,
        "net_pnl_after_fees": 0.0,
    }
    if live_signal is not None:
        metadata.update({
            "live_mirror_coin": getattr(live_signal, "coin", None),
            "live_mirror_side": (
                getattr(getattr(live_signal, "side", None), "value", None)
                or str(getattr(live_signal, "side", "") or "")
            ),
        })
    if isinstance(live_result, dict):
        metadata["live_mirror_status"] = live_result.get("status")
        for key in ("reason", "message", "entry_order_type"):
            value = live_result.get(key)
            if value not in (None, ""):
                metadata[f"live_mirror_{key}"] = value
        errors = live_result.get("errors")
        if errors:
            metadata["live_mirror_errors"] = errors
    else:
        metadata["live_mirror_status"] = "no_result"

    try:
        db.update_paper_trade_metadata(int(trade_id), metadata)
        if db.close_paper_trade(int(trade_id), flat_price, 0.0):
            logger.info(
                "Closed unmirrored paper trade %s flat (%s) so it cannot "
                "be reconciled as live PnL later",
                trade_id,
                reason,
            )
    except Exception as exc:
        logger.warning(
            "Could not close unmirrored paper trade %s flat (%s): %s",
            trade_id,
            reason,
            exc,
        )


def _mark_paper_trade_live_mirrored(
    trade_id: Optional[int],
    live_signal,
    live_result: Dict,
) -> None:
    """Persist live-mirror metadata so live-mode history filters see the trade."""
    if not trade_id:
        return
    if not isinstance(live_result, dict):
        return

    metadata = {
        "live_mirror": True,
        "live_mirror_attempted": True,
        "live_mirror_marked_at": datetime.now(timezone.utc).isoformat(),
        "live_mirror_status": live_result.get("status"),
        "live_mirror_coin": getattr(live_signal, "coin", None),
        "live_mirror_side": getattr(getattr(live_signal, "side", None), "value", None)
        or str(getattr(live_signal, "side", "") or ""),
    }
    for key in ("order_id", "oid", "cloid", "client_order_id", "live_order_id"):
        value = live_result.get(key)
        if value not in (None, ""):
            metadata[f"live_mirror_{key}"] = value

    try:
        db.update_paper_trade_metadata(int(trade_id), metadata)
    except Exception as exc:
        logger.warning(
            "Could not mark paper trade %s as live_mirror for live history: %s",
            trade_id,
            exc,
        )


def _paper_trade_id_for_client_order_id(client_order_id: str) -> Optional[int]:
    """Return an existing paper trade id for an idempotency key, if present."""
    if not client_order_id:
        return None
    try:
        with db.get_connection(for_read=True) as conn:
            row = conn.execute(
                "SELECT id FROM paper_trades WHERE client_order_id = ?",
                (client_order_id,),
            ).fetchone()
    except Exception as exc:
        logger.debug("paper trade idempotency lookup failed: %s", exc)
        return None
    if not row:
        return None
    try:
        return int(row["id"] if hasattr(row, "keys") else row[0])
    except Exception:
        return None


def get_live_trader(container):
    """Return the attached live trader, if any."""
    venue = os.environ.get("LIVE_EXECUTION_VENUE", "hyperliquid").strip().lower()
    if venue == "lighter":
        lighter = getattr(container, "lighter_live_trader", None)
        if lighter is not None:
            return lighter
    return getattr(container, "live_trader", None)


def is_live_trading_requested(container) -> bool:
    """True when the operator explicitly enabled live trading."""
    trader = get_live_trader(container)
    return bool(trader and trader.is_live_enabled())


def is_live_trading_active(container) -> bool:
    """True when the operator enabled live trading and the trader is deployable."""
    trader = get_live_trader(container)
    return bool(trader and trader.is_live_enabled() and trader.is_deployable())


def get_execution_open_positions(container) -> List[Dict]:
    """Use exchange positions as the source of truth when live trading is active."""
    trader = get_live_trader(container)
    if trader and is_live_trading_active(container):
        try:
            return trader.get_positions(force_fresh=True) or []
        except TypeError:
            return trader.get_positions() or []
    return db.get_open_paper_trades()


def get_execution_account_balance(container) -> Optional[float]:
    """Use live account value when available, otherwise fall back to paper balance."""
    trader = get_live_trader(container)
    if trader and is_live_trading_active(container):
        get_value = getattr(trader, "get_account_value", None)
        if callable(get_value):
            value = get_value()
            if value is not None:
                return float(value)

    account = db.get_paper_account()
    if not account:
        return None
    try:
        return float(account.get("balance", 0))
    except (TypeError, ValueError):
        return None


def sync_shadow_book_to_live(container) -> List[Dict]:
    """
    Close shadow paper trades that no longer exist on the exchange.

    In live mode, exchange state is the authority. This keeps the paper book as
    a reporting shadow instead of letting it drive runtime decisions.

    SAFETY: If the live account has $0 perps margin, skip reconciliation
    entirely.  Otherwise we'd close every paper trade because the exchange
    shows 0 positions (the money is in spot or hasn't been deposited).
    """
    if not is_live_trading_active(container) or not getattr(container, "paper_trader", None):
        return []

    # Guard: don't reconcile paper trades against an unfunded exchange account.
    # With $0 perps margin the exchange will always show 0 positions, which
    # would cause this function to close EVERY paper trade immediately.
    trader = get_live_trader(container)
    account_value = trader.get_account_value() if trader else None
    if account_value is None or account_value <= 0:
        return []

    try:
        fetched_positions = trader.get_positions(force_fresh=True) if trader else None
    except TypeError:
        fetched_positions = trader.get_positions() if trader else None
    if fetched_positions is None:
        logger.warning("Skipping shadow/live reconciliation: exchange positions unavailable")
        return []

    live_positions = {
        pos.get("coin", ""): pos
        for pos in fetched_positions
        if pos.get("coin") and abs(float(pos.get("szi", pos.get("size", 0)) or 0)) > 0
    }
    open_trades = db.get_open_paper_trades()
    if not open_trades:
        open_trades = []

    mids = get_all_mids() or {}
    closed = []
    matched_live_keys = set()
    for trade in open_trades:
        existing_meta = _trade_metadata(trade)
        # ★ RECONCILER FIX (origin/main 3d3e493): skip paper trades that
        # were never attempted on live (e.g. promotion-gate-blocked
        # sources, untagged sources, bootstrap-tier-deferred copies).
        # For these the absence of a live position is expected by
        # design, not an anomaly to close out.  Without this guard the
        # reconciler force-closes ~89% of paper trades at adverse
        # mid-prices ~3 minutes after open, poisoning the firewall's
        # recent-loss windows and the agent_scorer's outcome history
        # with -$735 of fake losses.
        is_live_shadow = bool(
            existing_meta.get("live_mirror") or existing_meta.get("orphan_found")
        )
        if not is_live_shadow:
            logger.debug(
                "Skipping shadow/live reconciliation for unmirrored paper "
                "trade_id=%s coin=%s side=%s",
                trade.get("id"),
                trade.get("coin", "?"),
                trade.get("side", "?"),
            )
            continue

        live_pos = live_positions.get(trade.get("coin", ""))
        if live_pos and live_pos.get("side") == trade.get("side"):
            matched_live_keys.add(
                (
                    str(trade.get("coin", "") or "").upper(),
                    str(trade.get("side", "") or "").lower(),
                )
            )
            continue

        current_price = float(
            mids.get(trade.get("coin", ""), trade.get("entry_price", 0))
            or trade.get("entry_price", 0)
            or 0
        )
        if current_price <= 0:
            continue

        trade_id = trade.get("id")
        if trade_id is None:
            continue

        # Compute reconciliation PnL FIRST so we can stamp the analytics
        # fields into metadata in the same write as the reconciliation
        # markers. Previously metadata was updated before reconciled_pnl
        # was computed, so net_pnl_after_fees / close_reason were never
        # written and any analytics reading metadata->>'net_pnl_after_fees'
        # saw $0 for ~80% of copy_trade closes (the reconciliation path
        # dominates copy_trade close volume).
        entry_price = float(trade.get("entry_price", 0) or 0)
        trade_size = float(trade.get("size", 0) or 0)
        trade_leverage = float(trade.get("leverage", 1) or 1)
        if trade.get("side") == "long":
            gross_reconciled_pnl = (current_price - entry_price) * trade_size * trade_leverage
        else:
            gross_reconciled_pnl = (entry_price - current_price) * trade_size * trade_leverage
        gross_reconciled_pnl = round(gross_reconciled_pnl, 2)

        # Estimate the round-trip live fee so net_pnl_after_fees in
        # metadata reflects actual fee drag, not just gross PnL at mid.
        # Hyperliquid live closes are typically taker on the exit leg
        # (forced reduce-only) so we use TAKER_FEE_BPS for both legs
        # as a conservative upper bound.
        try:
            import config as _cfg
            taker_bps = float(getattr(_cfg, "PAPER_TRADING_TAKER_FEE_BPS", 4.5))
        except Exception:
            taker_bps = 4.5
        fee_rate = taker_bps / 10_000.0
        entry_notional = entry_price * trade_size * trade_leverage
        exit_notional = current_price * trade_size * trade_leverage
        reconciled_fees = round(
            (max(entry_notional, 0.0) + max(exit_notional, 0.0)) * fee_rate, 4,
        )
        net_reconciled_pnl = round(gross_reconciled_pnl - reconciled_fees, 2)
        reconciled_pnl = net_reconciled_pnl  # legacy alias used below

        existing_meta.update({
            "synthetic_reconciliation": True,
            "reconciliation_reason": "live_reconciled_closed",
            "reconciliation_exit_price": current_price,
            # Mirror the reconciled PnL into the analytics fields so any
            # consumer reading metadata (dashboards, scorecards, the
            # decision-replay diff tool, ad-hoc SQL like
            # `details->>'net_pnl_after_fees'`) sees the same number that
            # the `pnl` DB column already has. We estimate the live
            # round-trip fee from PAPER_TRADING_TAKER_FEE_BPS so
            # net_pnl_after_fees reflects real fee drag and gross/net
            # differ correctly. Prior to this fix gross == net silently
            # under-reported fee cost in analytics.
            "close_reason": "live_reconciled_closed",
            "net_pnl_after_fees": net_reconciled_pnl,
            "gross_pnl_before_fees": gross_reconciled_pnl,
            "reconciled_fees_estimated": reconciled_fees,
        })
        db.update_paper_trade_metadata(trade_id, existing_meta)
        if not db.close_paper_trade(trade_id, current_price, reconciled_pnl):
            continue
        _notify_manual_close_detected(
            {
                **trade,
                "metadata": existing_meta,
            },
            exit_price=current_price,
        )

        # ★ H8 FIX: previously hardcoded "pnl": 0.0 in the returned dict
        # despite computing reconciled_pnl correctly above and writing it to
        # the DB. Callers consuming the returned dict (shadow tracker, kelly
        # outcome feed, telegram notifier) saw zero PnL on every reconciled
        # trade, poisoning their stats. Mirror the DB value into the dict.
        closed_trade = {
            "trade_id": trade_id,
            "entry_price": trade.get("entry_price", 0),
            "size": trade.get("size", 0),
            "leverage": trade.get("leverage", 1),
            "coin": trade.get("coin", ""),
            "side": trade.get("side", ""),
            "pnl": reconciled_pnl,
            "gross_pnl": reconciled_pnl,
            "fees_paid": 0.0,
            "slippage_cost": 0.0,
            "reason": "live_reconciled_closed",
            "strategy_type": existing_meta.get("strategy_type", "unknown"),
            "signal_id": existing_meta.get("signal_id", ""),
            "exit_price": current_price,
            "metadata": existing_meta,
            "opened_at": trade.get("opened_at", ""),
            "closed_at": trade.get("closed_at", ""),
        }
        closed.append(closed_trade)
        logger.info(
            "Shadow paper trade reconciled to exchange truth: %s %s",
            trade.get("side", "?").upper(),
            trade.get("coin", "?"),
        )

    for live_pos in live_positions.values():
        coin = str(live_pos.get("coin", "") or "").upper()
        side = str(live_pos.get("side", "") or "").lower()
        if not coin or side not in {"long", "short"}:
            continue
        if (coin, side) in matched_live_keys:
            continue

        entry_price = float(
            live_pos.get("entry_price", live_pos.get("entryPx", 0))
            or 0
        )
        size = abs(float(live_pos.get("size", live_pos.get("szi", 0)) or 0))
        leverage = float(live_pos.get("leverage", 1) or 1)
        if entry_price <= 0 or size <= 0:
            continue

        metadata = {
            "synthetic_reconciliation": True,
            "reconciliation_reason": "orphan_found",
            "orphan_found": True,
            "orphan_found_at": datetime.now(timezone.utc).isoformat(),
            "source": "live_orphan",
            "source_key": "live_orphan",
            "strategy_type": "orphan_found",
            "live_snapshot": {
                "coin": coin,
                "side": side,
                "entry_price": entry_price,
                "size": size,
                "leverage": leverage,
            },
        }
        # H5 (audit): a crash mid-reconciliation must not spawn a second
        # shadow record for the *same* orphan.  The exchange-level
        # identity (coin + side + entry + size + leverage) is stable
        # enough to serve as the idempotency key here — a repeated
        # reconciliation run sees the existing shadow row instead of
        # inserting a duplicate.
        orphan_key = (
            f"orphan:{coin}:{side}:{entry_price:.10g}:{size:.10g}:{leverage:.4g}"
        )
        existing_orphan_trade_id = _paper_trade_id_for_client_order_id(orphan_key)
        trade_id = db.open_paper_trade(
            None,
            coin,
            side,
            entry_price,
            size,
            leverage=leverage,
            metadata=metadata,
            idempotency_key=orphan_key,
        )
        db.audit_log(
            action="orphan_found",
            coin=coin,
            side=side,
            price=entry_price,
            size=size,
            source="live_execution",
            details={"trade_id": trade_id, "metadata": metadata},
        )
        if existing_orphan_trade_id is not None:
            logger.info(
                "Synthetic paper trade already tracks orphan live position: %s %s "
                "trade_id=%s size=%.6f entry=%.6f",
                side.upper(),
                coin,
                trade_id,
                size,
                entry_price,
            )
        else:
            logger.warning(
                "Created synthetic paper trade for orphan live position: %s %s "
                "trade_id=%s size=%.6f entry=%.6f",
                side.upper(),
                coin,
                trade_id,
                size,
                entry_price,
            )

    return closed


def _paper_open_margin_used() -> float:
    """Return the total notional margin the paper book currently has locked.

    H6 (audit): paper-to-live mirror sizing uses ``paper_balance`` (the
    paper account's *cash* balance) as the denominator while the live
    side correctly deducts ``totalMarginUsed`` to compute
    ``live_free_margin``.  That asymmetry silently oversizes live when
    the paper book has open exposure, because the denominator stays at
    the full cash balance even though paper would block a new trade
    with the same constraint.  This helper produces the paper-side
    equivalent so the scale ratio compares "new-trade capacity" on
    both sides.  Open paper trades without a ``leverage`` field fall
    back to 1x — the most conservative assumption.
    """
    try:
        open_trades = db.get_open_paper_trades() or []
    except Exception as exc:
        logger.debug("Cannot compute paper open margin used: %s", exc)
        return 0.0
    total = 0.0
    for t in open_trades:
        try:
            size = float(t.get("size", 0) or 0)
            entry_price = float(t.get("entry_price", 0) or 0)
            lev = float(t.get("leverage", 0) or 0) or 1.0
            if size <= 0 or entry_price <= 0:
                continue
            notional = size * entry_price
            total += notional / max(lev, 1.0)
        except Exception:
            continue
    return max(0.0, total)


def _rescale_size_for_live(trade: Dict, trader) -> Optional[Dict]:
    """
    Rescale paper trade size proportionally to the live account balance.

    Paper sizes are computed from the paper account (default $10k).  When
    mirroring to live, the coin quantity must be adjusted so the same
    *percentage* of the live account is risked, not the same absolute size.

    After rescaling, the final notional is clamped to trader.max_order_usd
    (if set) so nothing above the bootstrap cap ever hits the exchange.
    """
    # Walk-forward promotion gate: block live mirror if the strategy /
    # source hasn't earned live status yet (insufficient trade history,
    # poor win rate, or no resolvable track record). Paper continues
    # so the source keeps accumulating outcomes; only the live mirror
    # is gated. Bypass via ``LIVE_PROMOTION_GATE_ENABLED=false`` if you
    # need to bootstrap a new source.
    promotion_scale = 1.0  # bootstrap-tier size modifier; 1.0 = full
    # ★ AUDIT FIX (live-mirror fail-closed): the previous behaviour
    # caught ANY exception from ``is_live_promotable`` and forced
    # ``promotable=True`` with reason ``gate_error_fail_open``, silently
    # bypassing the entire live-promotion safety system on transient DB
    # errors, import failures, or any other gate-internal exception.
    # ``is_live_promotable`` is *itself* fail-closed (returns False on
    # exceptions); the wrapper here was overriding that into a
    # fail-OPEN-to-live default, which is exactly the failure mode the
    # gate exists to prevent.
    #
    # New default: fail-CLOSED for live.  An exception while evaluating
    # the gate skips the live mirror; the paper trade continues so the
    # source keeps accumulating outcomes and can promote on the next
    # cycle.  Operators can opt back into the legacy fail-open behaviour
    # via ``LIVE_PROMOTION_GATE_FAIL_OPEN=1`` (e.g. for a one-off
    # bootstrap of a fresh source while the gate's underlying DB schema
    # is being migrated).
    try:
        from src.learning.promotion_gate import is_live_promotable, get_bootstrap_scale
        promotable, reason = is_live_promotable(trade)
        promotion_scale = get_bootstrap_scale(reason)
    except Exception as exc:
        fail_open = str(
            os.environ.get("LIVE_PROMOTION_GATE_FAIL_OPEN", "0")
        ).strip().lower() in {"1", "true", "yes"}
        if fail_open:
            logger.warning(
                "Promotion gate check failed for %s but LIVE_PROMOTION_GATE_FAIL_OPEN=1 "
                "is set -- proceeding to live mirror at full size (legacy "
                "fail-open behaviour, %s)",
                trade.get("coin", "?"),
                exc,
            )
            promotable, reason = True, "gate_error_fail_open"
        else:
            logger.error(
                "Promotion gate check failed for %s -- skipping live mirror "
                "(fail-closed for live safety; set LIVE_PROMOTION_GATE_FAIL_OPEN=1 "
                "to override): %s",
                trade.get("coin", "?"),
                exc,
            )
            return None
    if not promotable:
        logger.info(
            "Skipping live mirror for %s: promotion gate blocked (%s). "
            "Paper trade continues; live stays gated until source meets "
            "promotion thresholds.",
            trade.get("coin", "?"),
            reason,
        )
        return None
    if promotion_scale < 1.0:
        logger.info(
            "Bootstrap-tier promotion for %s: applying %.2fx size scale (%s). "
            "Full-size promotion unlocks once the source meets the standard "
            "30-trade / 45%% win-rate bar.",
            trade.get("coin", "?"), promotion_scale, reason,
        )

    paper_account = db.get_paper_account()
    paper_balance = float((paper_account or {}).get("balance", 0) or 0)
    # H6 (audit): compute a *free* paper balance that deducts margin
    # already locked up by open paper trades.  Without this deduction
    # the scale ratio ``live_free_margin / paper_balance`` is
    # asymmetric: live correctly excludes locked margin while paper
    # effectively double-counts any open-position margin as available,
    # which scales the live order larger than intended.
    paper_margin_used = _paper_open_margin_used()
    paper_free_balance = max(0.0, paper_balance - paper_margin_used)
    live_equity = trader.get_account_value()
    live_free_margin = None
    if hasattr(trader, "get_free_margin"):
        try:
            live_free_margin = trader.get_free_margin()
        except Exception as exc:
            logger.warning(
                "Cannot rescale %s: live free margin API call failed (%s). "
                "Blocking trade to prevent oversizing.",
                trade.get("coin", "?"),
                exc,
            )
            return None
    if not paper_balance or paper_balance <= 0:
        logger.error(
            "Cannot rescale %s: paper account balance unavailable (%s). "
            "Blocking trade to prevent wrong sizing.",
            trade.get("coin", "?"),
            paper_balance,
        )
        return None
    if live_equity is None:
        logger.error(
            "Cannot rescale %s: live account balance API call failed. "
            "Blocking trade to prevent wrong sizing.",
            trade.get("coin", "?"),
        )
        return None
    if live_free_margin is None:
        live_free_margin = live_equity
    live_free_margin = float(live_free_margin or 0.0)
    if live_free_margin <= 0:
        logger.warning(
            "Skipping live mirror for %s: free perps margin is $%.2f "
            "(equity=$%.2f). Transfer USDC from Spot to Perps or free margin "
            "before opening new positions.",
            trade.get("coin", "?"), live_free_margin, float(live_equity or 0.0),
        )
        return None
    if paper_free_balance <= 0:
        logger.warning(
            "Skipping live mirror for %s: paper account has no free balance "
            "(balance=$%.2f, margin_used=$%.2f).  Close or shrink paper "
            "exposure before mirroring new live trades.",
            trade.get("coin", "?"),
            paper_balance,
            paper_margin_used,
        )
        return None

    # H6: scale on symmetric "new-trade capacity" on both sides.
    scale = live_free_margin / paper_free_balance
    # Bootstrap-tier promotion: multiply the rescale by the bootstrap scale
    # so the live mirror size becomes (paper_scale * bootstrap_fraction).
    # Floor-up below may still raise this back to min_order_usd if the
    # result lands under the exchange minimum -- by design, since refusing
    # every bootstrap mirror on a small wallet would defeat the tier.
    if promotion_scale < 1.0:
        scale *= promotion_scale
    # H7: Clamp scale to 1.0 when live free margin exceeds the paper
    # reference unless the operator explicitly opts in via
    # LIVE_ALLOW_SCALE_ABOVE_PAPER=1.  The paper book is the source of our
    # sizing edge (it's where Kelly/regime/confidence multipliers were
    # calibrated); letting a larger live balance push scale > 1 would
    # quietly 10x the position when operators topped-up their wallet
    # without also bumping PAPER_TRADING_INITIAL_BALANCE.
    allow_above = str(
        os.environ.get("LIVE_ALLOW_SCALE_ABOVE_PAPER", "0")
    ).strip().lower() in {"1", "true", "yes"}
    if scale > 1.0 and not allow_above:
        logger.info(
            "Clamping live mirror scale for %s from %.4f to 1.0 "
            "(live_free_margin=$%.2f > paper_free_balance=$%.2f, "
            "paper_balance=$%.0f, paper_margin_used=$%.2f). "
            "Set LIVE_ALLOW_SCALE_ABOVE_PAPER=1 to opt out.",
            trade.get("coin", "?"),
            scale,
            live_free_margin,
            paper_free_balance,
            paper_balance,
            paper_margin_used,
        )
        scale = 1.0
    original_size = float(trade.get("size", 0) or 0)
    if original_size <= 0:
        return trade

    scaled_trade = dict(trade)  # shallow copy to avoid mutating caller's dict
    if abs(scale - 1.0) >= 0.01:
        scaled_trade["size"] = original_size * scale
        logger.info(
            "Rescaled %s size for live: %.6f -> %.6f "
            "(paper=$%.0f [free=$%.2f, margin_used=$%.2f], "
            "free_margin=$%.2f, equity=$%.2f, scale=%.4f)",
            trade.get("coin", "?"),
            original_size,
            scaled_trade["size"],
            paper_balance,
            paper_free_balance,
            paper_margin_used,
            live_free_margin,
            float(live_equity or 0.0),
            scale,
        )

    # Enforce per-order $ cap (bootstrap safety net).  Applied on top of the
    # proportional rescale so nothing above LIVE_MAX_ORDER_USD hits the exchange.
    max_order_usd = getattr(trader, "max_order_usd", None)
    min_order_usd = getattr(trader, "min_order_usd", None) or 0.0

    # IMPORTANT: prefer the *live mid price* over the signal's entry_price
    # for cap/floor calculations.  place_market_order/place_limit_order use
    # mid price when they execute, so we must size using the same reference
    # — otherwise a 2-5% price drift between signal generation and order
    # placement flips our "$11.55 target notional" into "$10.97 actual
    # notional" and the exchange rejects with below_exchange_minimum_notional.
    coin = scaled_trade.get("coin", "") or ""
    mids = get_all_mids() or {}
    mid_price = 0.0
    try:
        mid_price = float(mids.get(coin, 0) or 0)
    except (TypeError, ValueError):
        mid_price = 0.0
    entry_price = mid_price
    if entry_price <= 0:
        entry_price = float(
            scaled_trade.get("entry_price")
            or trade.get("entry_price")
            or 0
        )

    # E6: cap using the *slipped* price that market orders actually trade at,
    # not the raw mid.  Hyperliquid's SDK default slippage is 5%, so a buy
    # order sized at mid against a $35 cap can blow past the cap to ~$36.75
    # when the fill prints at mid*1.05.  Use the slipped price for the cap
    # check and the floor-up check so the reference used for sizing matches
    # the price the order actually executes at.
    # C6: safe parse with range clamp — slippage > 50% would effectively
    # disable the cap, and negative input would cap *above* the raw mid.
    raw_slip = os.environ.get("LIVE_MARKET_SLIPPAGE_PCT", "0.05")
    try:
        slippage_pct = float(raw_slip)
    except (TypeError, ValueError):
        logger.warning(
            "LIVE_MARKET_SLIPPAGE_PCT=%r not numeric; using default 0.05",
            raw_slip,
        )
        slippage_pct = 0.05
    if slippage_pct < 0.0:
        logger.warning(
            "LIVE_MARKET_SLIPPAGE_PCT=%s is negative; clamping to 0.0",
            slippage_pct,
        )
        slippage_pct = 0.0
    elif slippage_pct > 0.5:
        logger.warning(
            "LIVE_MARKET_SLIPPAGE_PCT=%s > 0.5 (50%%); clamping to 0.5",
            slippage_pct,
        )
        slippage_pct = 0.5
    side_raw = str(scaled_trade.get("side", "") or trade.get("side", "") or "").strip().lower()
    if side_raw in {"buy", "long"}:
        slipped_price = entry_price * (1.0 + slippage_pct) if entry_price > 0 else 0.0
    elif side_raw in {"sell", "short"}:
        slipped_price = entry_price * (1.0 - slippage_pct) if entry_price > 0 else 0.0
    else:
        # Unknown side — fall back to the more conservative (higher) buy-side
        # slipped price so the cap still holds.
        slipped_price = entry_price * (1.0 + slippage_pct) if entry_price > 0 else 0.0

    # Reference price used for cap enforcement.  For BUYS the slipped price
    # is higher than mid (shrinks size), for SELLS it is lower (also shrinks
    # size when the proceeds must stay below the cap).
    cap_reference_price = (
        max(slipped_price, entry_price) if slipped_price > 0 else entry_price
    )

    if max_order_usd and max_order_usd > 0 and cap_reference_price > 0:
        current_size = float(scaled_trade.get("size", 0) or 0)
        notional = current_size * cap_reference_price
        if notional > max_order_usd:
            capped_size = max_order_usd / cap_reference_price
            logger.info(
                "Capping %s live mirror to $%.2f (slipped ref=%.6f): %.6f -> %.6f "
                "(notional $%.2f -> $%.2f)",
                coin or "?",
                max_order_usd,
                cap_reference_price,
                current_size,
                capped_size,
                notional,
                capped_size * cap_reference_price,
            )
            scaled_trade["size"] = capped_size

    # Floor-up to the exchange minimum when proportional rescaling would
    # otherwise produce an un-fillable order.  A small live wallet relative
    # to the paper book (e.g. $12 live vs $10k paper = 0.0012 scale factor)
    # makes every proportional rescale tiny, but Hyperliquid drops anything
    # under $10.  Rather than skipping every mirror, we floor up to the
    # exchange minimum — the LiveTrader's max_order_usd cap still bounds the
    # absolute max, so at most we send an $11 notional order.
    #
    # Safety: check that the *margin* required (notional / leverage) fits
    # in the live wallet with headroom for other positions.  Checking
    # notional would be wrong — a $11 notional at 5x leverage only ties up
    # $2.20 of margin, so a $12 wallet can comfortably hold 4+ of them.
    if min_order_usd > 0 and entry_price > 0:
        current_size = float(scaled_trade.get("size", 0) or 0)
        notional = current_size * entry_price
        if notional < min_order_usd:
            # Margin required = notional / leverage.  Fall back to 1x (the
            # most conservative assumption) if leverage is not specified.
            leverage = float(
                scaled_trade.get("leverage")
                or trade.get("leverage")
                or 1
            )
            if leverage <= 0:
                leverage = 1.0

            # Headroom budget: leave 5% of the wallet untouched for fees,
            # slippage, and funding.  Leverage multiplies the notional
            # each dollar of margin can support, so a $12 wallet at 5x
            # can carry up to $57 notional, while at 1x it caps out at
            # $11.40 notional (barely clearing the $11 minimum).  This
            # is the cap the check uses — NOT 80% of wallet, which
            # incorrectly blocked 1x trades on small wallets.
            wallet_notional_budget = max(0.0, live_free_margin) * 0.95 * leverage

            # Target 1.10x the minimum so normal price drift, slippage,
            # and size rounding (szDecimals) don't slip us back under
            # the floor.  Cap at max_order_usd AND at the wallet budget.
            headroom_target = min_order_usd * 1.10
            target_notional = headroom_target
            if max_order_usd and max_order_usd > 0:
                target_notional = min(target_notional, max_order_usd)
            target_notional = min(target_notional, wallet_notional_budget)

            # Only reject if even the bare minimum cannot fit in the
            # wallet at this leverage.  This is the true physical limit:
            # no amount of floor-up can make a $11 notional fit in a $10
            # wallet at 1x leverage.
            if target_notional < min_order_usd:
                logger.warning(
                    "Skipping %s live mirror: wallet $%.2f at %.1fx "
                    "leverage supports max $%.2f notional (95%% headroom), "
                    "which is below the $%.2f exchange minimum.  Fund "
                    "the wallet or raise leverage for this asset.",
                    coin or "?",
                    live_free_margin,
                    leverage,
                    wallet_notional_budget,
                    min_order_usd,
                )
                return None

            required_margin = target_notional / leverage
            floored_size = target_notional / entry_price
            logger.info(
                "Flooring %s live mirror UP to exchange minimum: "
                "%.6f -> %.6f (notional $%.2f -> $%.2f, margin @ %.1fx = "
                "$%.2f, ref_price=$%.4f %s).  Proportional rescale from "
                "paper was below $%.2f; this departs from strict "
                "paper-proportional sizing, unavoidable given live "
                "wallet $%.2f vs paper $%.0f.",
                coin or "?",
                current_size,
                floored_size,
                notional,
                floored_size * entry_price,
                leverage,
                required_margin,
                entry_price,
                "mid" if mid_price > 0 else "signal",
                min_order_usd,
                live_free_margin,
                paper_balance,
            )
            scaled_trade["size"] = floored_size

    return scaled_trade


def mirror_executed_trades_to_live(
    container,
    executed: List[Dict],
    success_label: str,
    skip_label: str,
) -> None:
    """Submit executed shadow trades to the live trader when live mode is active."""
    trader = get_live_trader(container)
    if not trader or not executed:
        return

    if is_live_trading_active(container):
        candidates = []
        for trade in executed:
            try:
                # Rescale size from paper balance to live balance
                scaled_trade = _rescale_size_for_live(trade, trader) if isinstance(trade, dict) else trade
                if scaled_trade is None:
                    continue  # blocked by rescale — already logged
                live_signal = signal_from_execution_dict(scaled_trade) if isinstance(scaled_trade, dict) else scaled_trade
                if hasattr(live_signal, "context"):
                    live_context = dict(getattr(live_signal, "context", {}) or {})
                    live_context.update({
                        "live_mirror": True,
                        "live_mirror_rescaled": isinstance(scaled_trade, dict),
                    })
                    live_signal.context = live_context
                entry_price = float(
                    getattr(live_signal, "entry_price", 0)
                    or (scaled_trade.get("entry_price", scaled_trade.get("price", 0)) if isinstance(scaled_trade, dict) else 0)
                    or 0
                )
                # Live-only conviction / re-entry-cooldown gate (default
                # OFF).  Skips mirroring low-conviction or churned-coin
                # signals to cut the fee bleed; the paper trade stands.
                allow, gate_reason = _live_mirror_conviction_gate(live_signal)
                if not allow:
                    logger.info(
                        "%s skipped (live conviction gate): %s %s -- %s",
                        success_label,
                        getattr(live_signal, "coin", "?"),
                        getattr(getattr(live_signal, "side", None), "value", live_signal.side)
                        if hasattr(live_signal, "side") else "?",
                        gate_reason,
                    )
                    continue
                size = abs(float(getattr(live_signal, "size", 0) or 0))
                leverage = max(1.0, float(getattr(live_signal, "leverage", 1.0) or 1.0))
                notional = max(0.0, size * entry_price)
                margin = notional / leverage if leverage > 0 else notional
                candidates.append({
                    "signal": live_signal,
                    "paper_trade_id": _paper_trade_id_from_live_mirror(trade, live_signal),
                    "notional": notional,
                    "margin": margin,
                })
            except Exception as exc:
                logger.error("%s live execution prep error: %s", success_label, exc)

        if not candidates:
            return

        # Prefer free/available margin (accountValue - totalMarginUsed) so the
        # batch budget doesn't double-count margin already locked by open
        # positions.  Falls back to total account value only if the trader
        # doesn't expose a free-margin helper.
        free_margin: Optional[float] = None
        if hasattr(trader, "get_free_margin"):
            try:
                fm = trader.get_free_margin()
                free_margin = float(fm) if fm is not None else None
            except Exception as exc:
                logger.debug("%s get_free_margin failed: %s", success_label, exc)
                free_margin = None
        if free_margin is None:
            try:
                free_margin = float(trader.get_account_value() or 0.0)
            except Exception:
                free_margin = 0.0

        margin_budget = max(0.0, free_margin) * 0.95
        selected = []
        used_margin = 0.0

        # Zero/negative budget means "no room to mirror anything" — reject
        # all candidates rather than (accidentally) admitting them all.
        if margin_budget <= 0.0:
            logger.warning(
                "%s skipped %d candidate(s): no free margin available "
                "(free=$%.2f, budget=$%.2f)",
                success_label,
                len(candidates),
                free_margin or 0.0,
                margin_budget,
            )
            return

        # Keep the paper execution order to maximize live-vs-paper parity when
        # margin/canary caps force us to drop some mirrors.
        for item in candidates:
            projected = used_margin + item["margin"]
            if projected > margin_budget:
                logger.warning(
                    "%s skipped %s %s: batch margin budget exceeded "
                    "(need $%.2f, used $%.2f, budget $%.2f)",
                    success_label,
                    item["signal"].coin,
                    item["signal"].side.value,
                    item["margin"],
                    used_margin,
                    margin_budget,
                )
                continue
            selected.append(item)
            used_margin = projected

        for item in selected:
            live_signal = item["signal"]
            try:
                # Mirror path: the paper trade has already passed the firewall
                # (cooldown, risk checks, etc.), so bypass firewall validation
                # here.  Otherwise the firewall's cooldown check rejects every
                # mirrored trade as "COIN traded Ns ago" — the paper trade that
                # triggered the mirror.  Kill-switch and daily loss still apply.
                live_result = trader.execute_signal(live_signal, bypass_firewall=True)
                if is_confirmed_live_execution_result(live_result):
                    _mark_live_mirror_time(live_signal)  # start re-entry cooldown
                    _mark_paper_trade_live_mirrored(
                        item.get("paper_trade_id"),
                        live_signal,
                        live_result,
                    )
                    logger.info(
                        "%s: %s %s %s",
                        success_label,
                        live_result.get("status", "?"),
                        live_signal.coin,
                        live_signal.side.value,
                    )
                else:
                    _close_unmirrored_paper_trade(
                        item.get("paper_trade_id"),
                        live_signal=live_signal,
                        reason="live_mirror_not_opened",
                        live_result=live_result if isinstance(live_result, dict) else None,
                    )
                    if live_result is None:
                        logger.info(
                            "%s skipped: %s %s blocked by live guardrails (no execution result)",
                            success_label,
                            live_signal.coin,
                            live_signal.side.value,
                        )
                    elif _is_insufficient_margin_rejection(live_result):
                        logger.warning(
                            "%s skipped due to insufficient margin: %s %s -> %s",
                            success_label,
                            live_signal.coin,
                            live_signal.side.value,
                            live_result,
                        )
                    else:
                        logger.error(
                            "%s FAILED: %s %s %s -- result: %s",
                            success_label,
                            live_signal.coin,
                            live_signal.side.value,
                            live_signal.confidence,
                            live_result,
                        )
            except Exception as exc:
                logger.error("%s live execution error: %s", success_label, exc)
    elif trader.is_live_enabled():
        logger.warning("%s", skip_label)
