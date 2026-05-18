"""Canonical account-basis resolver + the $102-vs-$10k invariant.

Includes an explicit equivalence proof against the exact legacy ternary
that was wired into the options-flow sizing path, so the centralization
is provably behavior-preserving (no live-money sizing change).
"""
from __future__ import annotations

import pytest

from src.core.account_basis import (
    AccountBasis,
    is_basis_mismatch,
    resolve_account_basis,
)


def _legacy_sizing_balance(live_active, live_value, paper_value):
    """The EXACT pre-refactor logic from trading_cycle options-flow."""
    sizing_balance = live_value if live_active and live_value else None
    if sizing_balance is None:
        sizing_balance = float(paper_value) if paper_value else None
    return sizing_balance


def _resolved_sizing_balance(live_active, live_value, paper_value):
    b = resolve_account_basis(
        live_active=live_active, live_value=live_value, paper_value=paper_value
    )
    return b.usd if b.source != "unknown" else None


@pytest.mark.parametrize(
    "live_active,live_value,paper_value",
    [
        (True, 102.0, 10000.0),    # live wins
        (True, 0.0, 10000.0),      # live unfunded -> paper
        (True, None, 10000.0),     # live unknown -> paper
        (False, 102.0, 10000.0),   # not live -> paper
        (False, None, 9947.5),     # paper only
        (True, 102.5, None),       # live, no paper
        (False, None, None),       # nothing -> None
        (True, None, None),        # nothing -> None
        (False, 0.0, 0.0),         # all zero -> None
    ],
)
def test_resolver_is_behaviorally_identical_to_legacy(live_active, live_value, paper_value):
    legacy = _legacy_sizing_balance(live_active, live_value, paper_value)
    resolved = _resolved_sizing_balance(live_active, live_value, paper_value)
    assert resolved == pytest.approx(legacy) if legacy is not None else resolved is None


def test_live_basis_takes_precedence_when_funded():
    b = resolve_account_basis(live_active=True, live_value=102.0, paper_value=10000.0)
    assert b == AccountBasis(usd=102.0, source="live")
    assert b.is_known


def test_unfunded_live_falls_back_to_paper():
    b = resolve_account_basis(live_active=True, live_value=0.0, paper_value=9947.0)
    assert b.source == "paper"
    assert b.usd == pytest.approx(9947.0)


def test_no_basis_is_unknown_zero():
    b = resolve_account_basis(live_active=True, live_value=None, paper_value=None)
    assert b.source == "unknown"
    assert b.usd == 0.0
    assert not b.is_known


def test_nan_and_garbage_coerce_safely():
    b = resolve_account_basis(
        live_active=True, live_value=float("nan"), paper_value="not-a-number"
    )
    assert b.source == "unknown"
    assert b.usd == 0.0


def test_bug_class_invariant():
    """A genuinely-live execution sizing against the paper basis is the
    exact condition that produced the $2,629/$102 = 2570% rejections."""
    assert is_basis_mismatch(executing_live=True, basis_source="paper") is True
    assert is_basis_mismatch(executing_live=True, basis_source="live") is False
    assert is_basis_mismatch(executing_live=False, basis_source="paper") is False
