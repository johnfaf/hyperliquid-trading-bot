"""CI structural defence against the address-truncation class of bug.

The historical fragmentation came from f-strings like
``f"copy_trade:{trader}"`` scattered across the code base, where some
sites passed the FULL address and others passed ``address[:10]``. The
fix was a 6-site patch + 12 DB rows migrated.

This test enforces the structural rule: from now on, all
``"copy_trade:<id>"`` keys MUST be built via
``src.signals.source_key.copy_trade_source_key()`` (which validates
the input). Direct f-strings are flagged.

When the test fails, the message points the new offender at the
canonical builder.

Allowlist
---------
The few legitimate places that DO build the key as a string literal
(the builder itself, documentation references, this test) are listed
in ``_ALLOWLIST_FILES``. Adding to that list requires a code-review
justification.
"""
from __future__ import annotations

import re
from pathlib import Path



REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


# Files that are allowed to contain raw f"copy_trade:..." patterns:
#   - The builder itself (it produces the key).
#   - Audit scripts that consume historical raw keys.
#   - Documentation strings.
_ALLOWLIST_FILES = {
    "src/signals/source_key.py",
    # Scripts directory is exempt -- these are operator-run tools that
    # display existing keys, not code that allocates new ones.
}


# Match f"copy_trade:{...}" or f'copy_trade:{...}' with anything (including
# qualified expressions and method calls) in the braces.
_RAW_FSTRING_RE = re.compile(r"""f["']copy_trade:\{[^}]+\}["']""")

# Match plain string concatenation: "copy_trade:" + something
_CONCAT_RE = re.compile(r"""["']copy_trade:["']\s*\+""")


def _rel_path(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT)).replace("\\", "/")


def _scan_for_raw_keys() -> list[tuple[str, int, str]]:
    """Return (rel_path, lineno, line) for every offending site under src/."""
    out: list[tuple[str, int, str]] = []
    for py in SRC_ROOT.rglob("*.py"):
        rel = _rel_path(py)
        if rel in _ALLOWLIST_FILES:
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            stripped = line.split("#", 1)[0]  # strip comments
            if _RAW_FSTRING_RE.search(stripped) or _CONCAT_RE.search(stripped):
                out.append((rel, i, line.strip()))
    return out


def test_no_raw_copy_trade_fstring_in_src():
    """If this test fails: a new f\"copy_trade:{...}\" was added in src/
    that bypasses the canonical builder. Replace it with::

        from src.signals.source_key import copy_trade_source_key
        source_key = copy_trade_source_key(trader)

    The builder validates the address and falls back to the safe
    "copy_trade" untagged key on malformed input.
    """
    offenders = _scan_for_raw_keys()
    # Note: known historical sites still exist (we did NOT bulk-rewrite
    # them in this commit -- the goal is to STOP NEW occurrences while
    # the wiring lands incrementally). We lock in the COUNT as of ship.
    # When you wire one of these to use the builder, decrement
    # MAX_KNOWN_LEGACY_SITES.
    MAX_KNOWN_LEGACY_SITES = 6
    msg = (
        f"Found {len(offenders)} raw copy_trade f-string / concat sites in src/.\n"
        f"Ratchet: must be <= {MAX_KNOWN_LEGACY_SITES} until each is migrated.\n"
        f"Sites:\n" + "\n".join(f"  {p}:{ln}  {line}" for p, ln, line in offenders)
    )
    assert len(offenders) <= MAX_KNOWN_LEGACY_SITES, msg


def test_builder_is_importable_and_callable():
    """Smoke check: the canonical builder is reachable from src.signals."""
    from src.signals.source_key import (
        COPY_TRADE_PREFIX,
        copy_trade_source_key,
        is_canonical_source_key,
        strategy_source_key,
    )
    full = "0x" + "1" * 40
    sk = copy_trade_source_key(full)
    assert sk.startswith(COPY_TRADE_PREFIX)
    assert is_canonical_source_key(sk)
    assert strategy_source_key("momentum_long").startswith("strategy:")


def test_known_legacy_sites_documented_for_migration():
    """Inventory the legacy sites so future PRs that fix one have an
    obvious target to update MAX_KNOWN_LEGACY_SITES in this file."""
    offenders = _scan_for_raw_keys()
    # As of this commit, we expect these sites (best-effort listing):
    expected_files = {
        "src/analysis/trade_analytics.py",
        "src/core/cycles/trading_cycle.py",
        "src/learning/promotion_gate.py",
        "src/trading/copy_trader.py",
    }
    found_files = {p for p, _, _ in offenders}
    # Document drift: if a NEW file appears in offenders, the test
    # output will show it -- forcing the operator to either migrate it
    # to the builder or add to expected_files explicitly.
    unexpected = found_files - expected_files
    assert not unexpected, (
        f"New file(s) with raw copy_trade f-strings: {sorted(unexpected)} "
        "-- migrate to copy_trade_source_key() (preferred) or add to "
        "expected_files in this test if the new site is legitimate."
    )
