"""Canonical account-basis resolution.

The ``$102 live vs $10k paper`` denominator mismatch was a whole *class*
of bugs (exposure cap, copy validation, options-flow sizing) because the
"what balance does this path size / validate against?" decision was an
ad-hoc ``live_value if live_active and live_value else paper`` ternary
duplicated across paths — easy to get inconsistent.

This is the single source of truth for that decision plus the invariant
that a genuinely-live execution must never silently size/validate against
the paper balance. Pure and dependency-free so it is trivially testable
and safe to call from any path.

Scope note: this centralizes the logic and adds the invariant + tests.
Migrating every individual sizing call site onto it is a deliberately
deferred, separately-reviewed follow-up (those paths are already
per-path-correct; the live-money risk of a sweeping sizing rewrite is not
worth bundling here). The firewall side of the bug class is already
closed via ``require_live_balance``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

LIVE = "live"
PAPER = "paper"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class AccountBasis:
    """Resolved basis: the USD figure and where it came from."""

    usd: float
    source: str  # LIVE | PAPER | UNKNOWN

    @property
    def is_known(self) -> bool:
        return self.source in (LIVE, PAPER) and self.usd > 0


def _coerce(value: Optional[float]) -> float:
    try:
        if value is None:
            return 0.0
        out = float(value)
        return out if out == out else 0.0  # NaN guard
    except (TypeError, ValueError):
        return 0.0


def resolve_account_basis(
    *,
    live_active: bool,
    live_value: Optional[float],
    paper_value: Optional[float],
) -> AccountBasis:
    """Resolve the canonical account basis.

    Precedence (identical to the logic already in use across paths):
      1. live trading active AND a positive live wallet value -> LIVE
      2. otherwise a positive paper balance                   -> PAPER
      3. otherwise                                            -> UNKNOWN (0.0)
    """
    lv = _coerce(live_value)
    if live_active and lv > 0:
        return AccountBasis(usd=lv, source=LIVE)
    pv = _coerce(paper_value)
    if pv > 0:
        return AccountBasis(usd=pv, source=PAPER)
    return AccountBasis(usd=0.0, source=UNKNOWN)


def is_basis_mismatch(*, executing_live: bool, basis_source: str) -> bool:
    """The bug-class invariant: a genuinely-live execution sizing /
    validating against the *paper* basis. ``True`` == the dangerous
    condition that produced the ``$2,629/$102 = 2570%`` rejections.
    Callers should reject or alert (never silently proceed) when True.
    """
    return bool(executing_live and basis_source == PAPER)
