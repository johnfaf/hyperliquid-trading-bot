"""
Live-trading scaling tiers.

Rather than a single binary ``LIVE_CANARY_MODE`` flag (canary OR full), this
module defines a graduated ladder the operator can climb one checkpoint at
a time.  Each tier hard-caps four things:

  * ``max_order_usd``          — per-order notional ceiling
  * ``max_daily_loss_usd``     — daily realized-loss kill-switch threshold
  * ``max_position_size_usd``  — per-coin notional ceiling
  * ``max_signals_per_day``    — entry-signal throughput cap

and softly caps sizing aggressiveness via ``kelly_multiplier``.

**Safety invariant (downward-only).**  Tiers can only *tighten* the limits
an operator already configured via env (``LIVE_MAX_ORDER_USD`` etc.).  They
never silently expand capital exposure beyond what the operator set.  If
you want more headroom than the tier allows, you must raise the tier AND
the underlying env var — the tier alone will not do it.

**Advancement is operator-driven.**  The bot does not auto-advance from
T0 → T1 → T2 based on its own judgment about P&L.  The operator flips
``LIVE_TIER`` manually after confirming the advance gates documented in
``docs/SCALING_RUNBOOK.md``.  This is a financial-judgment call that does
not belong in the bot itself.

Typical rollout::

    T0 canary  →  T1 seed  →  T2 growth  →  T3 scale  →  T4 full

Each step should run for at least a handful of trading sessions before
advancing.  See the runbook for explicit checkpoints.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScalingTier:
    """Hard-cap envelope for one rung of the scaling ladder."""

    name: str
    label: str
    #: Per-order notional ceiling in USD.  Floor the operator's env cap to
    #: this value.  ``None`` = no tier override (use operator config).
    max_order_usd: Optional[float]
    #: Daily realized-loss ceiling in USD that triggers the kill switch.
    #: ``None`` = no tier override.
    max_daily_loss_usd: Optional[float]
    #: Per-coin notional ceiling in USD.  ``None`` = no tier override.
    max_position_size_usd: Optional[float]
    #: Max entry signals per calendar day.  ``0``/``None`` = no cap.
    max_signals_per_day: Optional[int]
    #: Kelly aggressiveness multiplier (applied on top of config KELLY_MULTIPLIER).
    #: 1.0 = no dampening; <1.0 = shrink sizes.
    kelly_dampen: float = 1.0
    #: One-line human description, shown in /healthz and logs.
    description: str = ""
    #: Suggested minimum days before considering advancement.  Informational only.
    min_days_recommended: int = 1


# ─────────────────────────────────────────────────────────────────────
# Preset ladder
#
# Tune the envelopes to your own risk tolerance — these defaults are the
# conservative path I'd recommend for a retail book starting from zero.
# A production operator should review and tweak before first live trade.
# ─────────────────────────────────────────────────────────────────────

_TIER_DEFS: Dict[str, ScalingTier] = {
    "T0": ScalingTier(
        name="T0",
        label="canary",
        max_order_usd=25.0,
        max_daily_loss_usd=50.0,
        max_position_size_usd=50.0,
        max_signals_per_day=25,
        kelly_dampen=0.5,
        description="Canary -- smallest viable size, heavy Kelly dampening, tight loss cap.",
        min_days_recommended=2,
    ),
    "T1": ScalingTier(
        name="T1",
        label="seed",
        max_order_usd=50.0,
        max_daily_loss_usd=100.0,
        max_position_size_usd=150.0,
        max_signals_per_day=40,
        kelly_dampen=0.65,
        description="Seed -- 2x canary, still defensive on Kelly, small loss envelope.",
        min_days_recommended=3,
    ),
    "T2": ScalingTier(
        name="T2",
        label="growth",
        max_order_usd=100.0,
        max_daily_loss_usd=250.0,
        max_position_size_usd=500.0,
        max_signals_per_day=60,
        kelly_dampen=0.8,
        description="Growth -- 4x canary, moderate Kelly, meaningful-but-recoverable daily cap.",
        min_days_recommended=5,
    ),
    "T3": ScalingTier(
        name="T3",
        label="scale",
        max_order_usd=250.0,
        max_daily_loss_usd=600.0,
        max_position_size_usd=1500.0,
        max_signals_per_day=100,
        kelly_dampen=0.9,
        description="Scale -- 10x canary, near-full Kelly, operator already saw multi-day stability.",
        min_days_recommended=7,
    ),
    "T4": ScalingTier(
        name="T4",
        label="full",
        max_order_usd=None,
        max_daily_loss_usd=None,
        max_position_size_usd=None,
        max_signals_per_day=None,
        kelly_dampen=1.0,
        description="Full -- tier overrides off; limits come from LIVE_MAX_* env vars only.",
        min_days_recommended=0,
    ),
}


def available_tiers() -> Dict[str, ScalingTier]:
    """Return a copy of the preset tier ladder (T0..T4)."""
    return dict(_TIER_DEFS)


def resolve_tier(name: Optional[str]) -> Optional[ScalingTier]:
    """Normalize a tier name and return the matching ``ScalingTier``.

    Accepts ``"T0"``/``"t0"``/``"canary"``/``"CANARY"`` etc.  Returns
    ``None`` if ``name`` is empty or unknown (caller falls back to
    pre-tier behaviour in that case).
    """
    if name is None:
        return None
    raw = str(name).strip()
    if not raw:
        return None
    upper = raw.upper()
    if upper in _TIER_DEFS:
        return _TIER_DEFS[upper]
    lower = raw.lower()
    for tier in _TIER_DEFS.values():
        if tier.label.lower() == lower:
            return tier
    logger.warning(
        "LIVE_TIER=%r is not a recognized tier name (expected one of %s or %s). "
        "Ignoring -- falling back to env-var limits.",
        name,
        sorted(_TIER_DEFS.keys()),
        sorted(t.label for t in _TIER_DEFS.values()),
    )
    return None


def apply_tier_caps(
    tier: ScalingTier,
    *,
    max_order_usd: float,
    max_daily_loss: float,
    max_position_size: float,
    max_signals_per_day: int,
    min_order_usd: float,
) -> Dict[str, float]:
    """Compute the post-tier effective caps.

    **Downward-only**: each tier field either lowers the matching env cap
    or is ignored (when ``None``).  The result is a dict mapping
    ``max_order_usd`` / ``max_daily_loss`` / ``max_position_size`` /
    ``max_signals_per_day`` to their tier-adjusted values.

    Respects ``min_order_usd`` as a hard floor on ``max_order_usd`` —
    the tier cannot push per-order notional below the exchange minimum.
    """
    out: Dict[str, float] = {
        "max_order_usd": float(max_order_usd),
        "max_daily_loss": float(max_daily_loss),
        "max_position_size": float(max_position_size),
        "max_signals_per_day": int(max_signals_per_day),
    }

    if tier.max_order_usd is not None and tier.max_order_usd > 0:
        tier_cap = max(float(min_order_usd), float(tier.max_order_usd))
        out["max_order_usd"] = min(float(max_order_usd), tier_cap)

    if tier.max_daily_loss_usd is not None and tier.max_daily_loss_usd > 0:
        out["max_daily_loss"] = min(float(max_daily_loss), float(tier.max_daily_loss_usd))

    if tier.max_position_size_usd is not None and tier.max_position_size_usd > 0:
        out["max_position_size"] = min(
            float(max_position_size), float(tier.max_position_size_usd)
        )

    if tier.max_signals_per_day is not None and tier.max_signals_per_day > 0:
        current = int(max_signals_per_day) if max_signals_per_day and max_signals_per_day > 0 else None
        if current is None:
            out["max_signals_per_day"] = int(tier.max_signals_per_day)
        else:
            out["max_signals_per_day"] = min(current, int(tier.max_signals_per_day))

    return out


def summarize_tier(tier: ScalingTier) -> Dict[str, object]:
    """Return a JSON-friendly dict describing the tier (for /healthz)."""
    return {
        "name": tier.name,
        "label": tier.label,
        "description": tier.description,
        "max_order_usd": tier.max_order_usd,
        "max_daily_loss_usd": tier.max_daily_loss_usd,
        "max_position_size_usd": tier.max_position_size_usd,
        "max_signals_per_day": tier.max_signals_per_day,
        "kelly_dampen": tier.kelly_dampen,
        "min_days_recommended": tier.min_days_recommended,
    }


__all__ = [
    "ScalingTier",
    "available_tiers",
    "resolve_tier",
    "apply_tier_caps",
    "summarize_tier",
]
