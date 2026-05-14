"""Expected-value gate.

Replaces confidence-only thresholds with a discipline that asks the
right question: "after fees + slippage + funding, is this trade
positive-EV?" A signal that clears the firewall's bare min-confidence
check can still be net-negative once you subtract realistic costs.

Formula::

    EV_bps = p_win * avg_win_bps - (1 - p_win) * avg_loss_bps - cost_bps

Where:
  * ``p_win`` is the calibrated win probability for the
    (source, side, regime) bucket, with cold-start fallback to a
    conservative prior when the bucket is thin.
  * ``avg_win_bps`` / ``avg_loss_bps`` come from the strategy's
    risk-policy snapshot (TP/SL ROE) -- these are the *intended* gains
    and losses, scaled by the conditional realisation rate observed in
    history when available.
  * ``cost_bps`` from ``trade_costs.estimate_signal_costs_bps`` (round-
    trip fees + slippage + holding-period funding).

Approve rule::

    EV_bps > max(MIN_EV_BPS, MIN_EV_COST_RATIO * cost_bps)

Live-tightening (when ``signal.context["live_mirror"]`` is true)::

    EV_bps - LIVE_EV_SIGMA_MULTIPLIER * sigma_EV_bps > 0

The sigma is a cheap normal-approximation, not a true bootstrap LCB.
That's deliberate: bootstrap CIs require per-signal resamples which
are expensive at signal-rate. The mean-minus-2-sigma approximation is
~equivalent to a 95% LCB and is computed in microseconds.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import config

logger = logging.getLogger(__name__)


# Cold-start defaults. When the bucket has no calibration history we
# assume a small positive edge so the gate doesn't reject every
# bootstrap trade -- the cost gate still applies, just with priors.
_COLDSTART_P_WIN = 0.50
_COLDSTART_AVG_WIN_BPS = 200.0   # conservative 2% win
_COLDSTART_AVG_LOSS_BPS = 200.0  # conservative 2% loss


def _config_float(name: str, default: float) -> float:
    try:
        return float(getattr(config, name, default) or default)
    except (TypeError, ValueError):
        return default


def _min_ev_bps() -> float:
    return _config_float("EV_GATE_MIN_BPS", 10.0)


def _min_ev_cost_ratio() -> float:
    return _config_float("EV_GATE_MIN_COST_RATIO", 1.5)


def _live_sigma_multiplier() -> float:
    return _config_float("EV_GATE_LIVE_SIGMA_MULT", 2.0)


def _is_live_mirror(signal: Any) -> bool:
    ctx = getattr(signal, "context", None)
    if isinstance(ctx, dict) and ctx.get("live_mirror"):
        return True
    return False


def _signal_attr(signal: Any, name: str, default: Any = None) -> Any:
    if isinstance(signal, dict):
        return signal.get(name, default)
    return getattr(signal, name, default)


def _resolve_win_loss_bps(signal: Any) -> Tuple[float, float]:
    """Derive expected win/loss magnitudes (in bps) from the signal's risk policy.

    Uses ROE-space TP/SL because that's how the firewall and risk
    policy already think; converts to bps for parity with cost units.
    """
    risk = _signal_attr(signal, "risk", None)
    leverage = float(_signal_attr(signal, "leverage", 1.0) or 1.0) or 1.0
    tp_roe = sl_roe = None
    if risk is not None:
        try:
            if hasattr(risk, "resolve_roe_take_profit_pct"):
                tp_roe = float(risk.resolve_roe_take_profit_pct(leverage))
            elif hasattr(risk, "take_profit_pct"):
                tp_roe = float(getattr(risk, "take_profit_pct", 0.0) or 0.0)
            if hasattr(risk, "resolve_roe_stop_loss_pct"):
                sl_roe = float(risk.resolve_roe_stop_loss_pct(leverage))
            elif hasattr(risk, "stop_loss_pct"):
                sl_roe = float(getattr(risk, "stop_loss_pct", 0.0) or 0.0)
        except Exception:
            pass
    avg_win_bps = max(_COLDSTART_AVG_WIN_BPS, (tp_roe or 0.0) * 10_000.0)
    avg_loss_bps = max(_COLDSTART_AVG_LOSS_BPS, abs((sl_roe or 0.0) * 10_000.0))
    return avg_win_bps, avg_loss_bps


def _resolve_calibrated_p_win(
    signal: Any,
    *,
    calibration: Optional[Any] = None,
    source_key: Optional[str] = None,
    regime: Optional[str] = None,
) -> Tuple[float, str]:
    """Return ``(p_win, source)`` for the calibrated win probability.

    Prefers the calibration tracker's bucketed adjustment. Falls back
    to the signal's raw confidence (with a coldstart cap), and finally
    to the coldstart prior.
    """
    raw_conf = float(_signal_attr(signal, "confidence", 0.5) or 0.5)
    if calibration is not None and source_key:
        side_obj = _signal_attr(signal, "side", "")
        side_val = side_obj.value if hasattr(side_obj, "value") else str(side_obj)
        try:
            adjusted = float(calibration.get_adjustment_factor(
                source_key, raw_conf, side=side_val, regime=regime,
            ))
            return max(0.05, min(adjusted, 0.95)), "calibration_tracker"
        except Exception as exc:
            logger.debug("ev_gate: calibration adjustment failed: %s", exc)
    return max(0.05, min(raw_conf, 0.95)), "raw_confidence"


def _sigma_ev_bps(p_win: float, avg_win_bps: float, avg_loss_bps: float, n_eff: float) -> float:
    """Cheap normal-approximation sigma on EV.

    Treats each trade outcome as Bernoulli(p) with payoffs (avg_win, -avg_loss).
    sigma of the *single-trade* return is sqrt(p*(1-p)*(win+loss)^2). Per the
    central limit theorem the sigma on the MEAN over n trades is that
    over sqrt(n_eff). When sample size is unknown, n_eff=1 yields the
    per-trade sigma (most conservative).
    """
    spread = max(avg_win_bps + avg_loss_bps, 0.0)
    if spread <= 0:
        return 0.0
    var = p_win * (1.0 - p_win) * (spread * spread)
    sigma_single = math.sqrt(max(var, 0.0))
    return sigma_single / max(math.sqrt(max(n_eff, 1.0)), 1.0)


def compute_expected_value(
    signal: Any,
    *,
    costs: Optional[Dict[str, float]] = None,
    calibration: Optional[Any] = None,
    source_key: Optional[str] = None,
    regime: Optional[str] = None,
    bucket_n: float = 1.0,
) -> Dict[str, Any]:
    """Return EV breakdown for a signal.

    ``bucket_n`` is the effective sample size for the calibration
    bucket; pass ``CalibrationTracker.get_sample_size(resolved_key)``
    for a real estimate. Defaults to 1.0 (per-trade sigma).
    """
    if costs is None:
        try:
            from src.signals.trade_costs import estimate_signal_costs_bps
            costs = estimate_signal_costs_bps(signal)
        except Exception as exc:
            logger.debug("ev_gate: cost estimate failed: %s", exc)
            costs = {"total_bps": 0.0}

    avg_win_bps, avg_loss_bps = _resolve_win_loss_bps(signal)
    p_win, p_source = _resolve_calibrated_p_win(
        signal,
        calibration=calibration,
        source_key=source_key,
        regime=regime,
    )
    cost_bps = float(costs.get("total_bps", 0.0) or 0.0)

    ev_bps = p_win * avg_win_bps - (1.0 - p_win) * avg_loss_bps - cost_bps
    sigma_bps = _sigma_ev_bps(p_win, avg_win_bps, avg_loss_bps, bucket_n)
    return {
        "ev_bps": round(ev_bps, 3),
        "sigma_bps": round(sigma_bps, 3),
        "p_win": round(p_win, 4),
        "p_win_source": p_source,
        "avg_win_bps": round(avg_win_bps, 3),
        "avg_loss_bps": round(avg_loss_bps, 3),
        "cost_bps": round(cost_bps, 3),
        "costs": costs,
        "bucket_n": float(bucket_n),
    }


def evaluate_signal_ev(
    signal: Any,
    *,
    costs: Optional[Dict[str, float]] = None,
    calibration: Optional[Any] = None,
    source_key: Optional[str] = None,
    regime: Optional[str] = None,
    bucket_n: float = 1.0,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Apply the EV gate and return ``(accept, reason, breakdown)``.

    Returns ``(True, "gate_disabled", {})`` when
    ``EV_GATE_ENABLED=false`` so callers don't have to short-circuit.
    """
    if not bool(getattr(config, "EV_GATE_ENABLED", True)):
        return True, "gate_disabled", {}

    breakdown = compute_expected_value(
        signal,
        costs=costs,
        calibration=calibration,
        source_key=source_key,
        regime=regime,
        bucket_n=bucket_n,
    )
    ev_bps = breakdown["ev_bps"]
    sigma_bps = breakdown["sigma_bps"]
    cost_bps = breakdown["cost_bps"]

    min_bps = _min_ev_bps()
    ratio = _min_ev_cost_ratio()
    threshold = max(min_bps, ratio * cost_bps)
    if ev_bps <= threshold:
        return (
            False,
            f"ev_below_threshold:ev={ev_bps:.1f}bps<=thr={threshold:.1f}bps "
            f"(cost={cost_bps:.1f}bps, p_win={breakdown['p_win']:.2f})",
            breakdown,
        )

    if _is_live_mirror(signal):
        sigma_mult = _live_sigma_multiplier()
        lower_bound = ev_bps - sigma_mult * sigma_bps
        if lower_bound <= 0:
            return (
                False,
                f"ev_lcb_negative_for_live:ev={ev_bps:.1f}bps "
                f"-{sigma_mult:.1f}*sigma({sigma_bps:.1f}bps)={lower_bound:.1f}bps",
                breakdown,
            )

    return True, "ok", breakdown
