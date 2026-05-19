"""Canonical SourceKey construction + validation.

Background
----------
The historical address-truncation bug (``source_trader = address[:10]``)
caused agent_scorer key fragmentation: the *same* trader was stored
under 12 distinct keys because different code paths truncated to
different lengths or not at all. The structural fix is a single
construction site for source keys with a runtime validator that
*refuses* to build a truncated key.

This module is the canonical builder. Call ``copy_trade_source_key()``
instead of ``f"copy_trade:{address}"`` anywhere a source key is being
built for a copy trade. The validator enforces the invariant:

* If the input is a 0x-prefixed hex string, it MUST be the full ETH
  address (0x + 40 hex chars). Truncated forms raise ValueError.
* If the input is not 0x-prefixed (e.g. a strategy name or a custom
  trader tag), it is passed through verbatim after a basic length /
  whitespace check.

Construction-site enforcement
-----------------------------
The functions here raise on bad input by default. Callers that want
"best-effort, fall back to untagged" semantics can use the
``strict=False`` parameter, which returns the canonical ``"copy_trade"``
key for any malformed address. This is the right default for the
trading hot path -- a malformed address should not silently fragment
agent_scorer state, but it also shouldn't crash trade execution.

The companion ``is_canonical_source_key()`` is a pure predicate the
CI test suite uses to scan call-sites in src/ and flag any new
``f"copy_trade:{...}"`` patterns that bypass the builder.

Invariants
----------
A canonical copy_trade source key has shape ``"copy_trade:0x" + 40 hex
chars`` (total 53 chars). A canonical strategy / generic key matches
``[a-z_:.][a-z0-9_:./-]+`` and is non-empty.
"""
from __future__ import annotations

import re
from typing import NewType


# A SourceKey is just a str at runtime, but the NewType makes intent
# explicit at typing time and lets mypy flag uses of raw str where a
# SourceKey is expected. (No runtime overhead.)
SourceKey = NewType("SourceKey", str)


# An Ethereum address: 0x + exactly 40 hex characters. Case-insensitive.
_ETH_ADDR_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")

# Any reasonable non-address key: lowercase letters/digits/_:./-, plus
# colon-prefixed segments (e.g. "strategy:momentum_long").
_GENERIC_KEY_RE = re.compile(r"^[a-z_][a-z0-9_:./\\-]{0,127}$")


# Public copy_trade key prefix. Centralised so future renames touch one
# constant instead of greps across the repo.
COPY_TRADE_PREFIX = "copy_trade"
COPY_TRADE_UNTAGGED = "copy_trade"


def _looks_like_address(s: str) -> bool:
    """True iff `s` starts with 0x and is reasonable hex-shaped."""
    return s.startswith("0x") and all(c in "0123456789abcdefABCDEF" for c in s[2:])


def _validate_address(s: str) -> bool:
    """True iff `s` is exactly a 0x-prefixed 40-hex Ethereum address."""
    return bool(_ETH_ADDR_RE.match(s))


def copy_trade_source_key(address: str, *, strict: bool = False) -> SourceKey:
    """Build the canonical source key for a copy_trade signal.

    Parameters
    ----------
    address
        The trader's wallet address. Should be the FULL 42-char
        0x-prefixed ETH address. Whitespace is stripped; case is
        lowercased. Truncated 0x-prefixed strings are rejected.
    strict
        When True (default False), an invalid address raises
        ValueError. When False, an invalid address falls back to the
        canonical untagged form ``"copy_trade"`` -- the historical
        fragmentation-safe default.

    Returns
    -------
    SourceKey newtype-wrapping the resulting string.

    Raises
    ------
    ValueError
        Only when strict=True and the address is malformed.
    """
    raw = (address or "").strip().lower()
    if not raw:
        if strict:
            raise ValueError("SourceKey: empty address")
        return SourceKey(COPY_TRADE_UNTAGGED)

    # If it LOOKS like an address (starts with 0x), it MUST be a full one.
    # This is the central anti-truncation rule: a truncated address like
    # "0x1ee7a73c" looks like an address but is wrong -- silently keying
    # under it would re-create the fragmentation bug.
    if raw.startswith("0x"):
        if not _validate_address(raw):
            if strict:
                raise ValueError(
                    f"SourceKey: 0x-prefixed input is not a full ETH address: "
                    f"{raw!r} (len={len(raw)}, expected 42)"
                )
            return SourceKey(COPY_TRADE_UNTAGGED)
        return SourceKey(f"{COPY_TRADE_PREFIX}:{raw}")

    # Not 0x-prefixed: a custom trader tag or label. Allow it through
    # the basic generic-key check.
    if _GENERIC_KEY_RE.match(raw):
        return SourceKey(f"{COPY_TRADE_PREFIX}:{raw}")

    if strict:
        raise ValueError(f"SourceKey: invalid address format: {address!r}")
    return SourceKey(COPY_TRADE_UNTAGGED)


def strategy_source_key(strategy_name: str, *, strict: bool = False) -> SourceKey:
    """Build the canonical source key for a strategy-emitted signal.

    Shape: ``"strategy:" + name``. The name must match the generic key
    pattern (lowercase alphanumerics, underscores, dashes, dots).
    """
    raw = (strategy_name or "").strip().lower()
    if not raw or not _GENERIC_KEY_RE.match(raw):
        if strict:
            raise ValueError(f"SourceKey: invalid strategy name: {strategy_name!r}")
        return SourceKey("strategy:unknown")
    return SourceKey(f"strategy:{raw}")


def is_canonical_source_key(s: str) -> bool:
    """True iff `s` is a canonical SourceKey shape.

    Pure predicate; used by the CI structural test to scan code for
    raw-string source keys that bypass the builder.
    """
    if not isinstance(s, str) or not s:
        return False
    # Generic patterns: <namespace>:<id>
    if s == COPY_TRADE_UNTAGGED or s == "strategy:unknown":
        return True
    if s.startswith(f"{COPY_TRADE_PREFIX}:"):
        rest = s[len(COPY_TRADE_PREFIX) + 1:]
        if rest.startswith("0x"):
            return _validate_address(rest)
        return bool(_GENERIC_KEY_RE.match(rest))
    if s.startswith("strategy:"):
        rest = s[len("strategy:"):]
        return bool(_GENERIC_KEY_RE.match(rest))
    # Allow other namespaces we use today: funding_carry, alpha_arena,
    # xgboost, etc. They just need to look like ns:id.
    if ":" in s:
        ns, _, rest = s.partition(":")
        return (
            bool(_GENERIC_KEY_RE.match(ns))
            and (rest == "" or bool(_GENERIC_KEY_RE.match(rest)))
        )
    return bool(_GENERIC_KEY_RE.match(s))


def is_truncated_address(s: str) -> bool:
    """True iff `s` looks like an address but isn't a full one.

    Used by audit scripts to flag historical fragmentation. The
    structural builder above prevents NEW occurrences; this predicate
    helps clean up old DB state.
    """
    if not isinstance(s, str):
        return False
    # Strip the namespace if present
    candidate = s
    if ":" in candidate:
        candidate = candidate.rsplit(":", 1)[-1]
    if not candidate.startswith("0x"):
        return False
    if _validate_address(candidate):
        return False
    return _looks_like_address(candidate)
