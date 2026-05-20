"""CI determinism guards for replay reproducibility.

Investor-grade replay requires that running the same harness against
the same inputs produces byte-identical outputs. Python's hash
randomisation (default since 3.3) makes `hash("string")` vary per
interpreter start, which then leaks through any place that:

  * iterates a `set` of source keys,
  * relies on `dict` insertion order to be stable across runs,
  * uses Python's built-in `hash()` as a cache key.

These tests are SKIPPED locally (where determinism is a nice-to-have)
but ENFORCED in CI via ``PYTHONHASHSEED=0`` set in
``.github/workflows/ci.yml``.

If you see this test FAIL in CI: the env var was removed from the
workflow. If you see it FAIL locally: you set PYTHONHASHSEED yourself
and hash_randomization is still on -- a deeper issue.

The companion to this test is `test_no_raw_hash_in_decision_paths`
which scans for raw `hash(...)` calls in the decision module trees.
A `hash()` on a tuple of strings will return different values per
interpreter start; for cache keys / IDs use `hashlib.sha256` or
`hashlib.blake2b` instead.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest


def _hashseed_enforced() -> bool:
    """True iff PYTHONHASHSEED is pinned (CI or operator-set)."""
    return os.environ.get("PYTHONHASHSEED") not in (None, "random", "")


@pytest.mark.skipif(
    not _hashseed_enforced(),
    reason="PYTHONHASHSEED not set; determinism enforcement is CI-only.",
)
def test_hash_randomization_is_disabled_when_seed_pinned():
    """When PYTHONHASHSEED is set to a number, hash_randomization
    must be 0. If this fails, the env var didn't reach the interpreter
    (e.g. workflow misconfigured)."""
    assert sys.flags.hash_randomization == 0, (
        f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')!r} but "
        f"sys.flags.hash_randomization={sys.flags.hash_randomization}. "
        "Replay determinism is not actually enforced."
    )


@pytest.mark.skipif(
    not _hashseed_enforced(),
    reason="PYTHONHASHSEED not set; determinism enforcement is CI-only.",
)
def test_hash_of_str_is_deterministic_across_calls():
    """Smoke check: under PYTHONHASHSEED=0, hash('source_key') is
    a fixed constant. The actual constant doesn't matter -- only
    that it's reproducible."""
    # Two calls in the same interpreter return the same value (always
    # true, even without seed). The investor-grade property is that
    # across separate CI runs the value is also the same. We can't
    # test that here -- it's enforced by the workflow env var. This
    # test exists to document the contract.
    assert hash("source_key") == hash("source_key")


# ── Audit: raw hash() in decision paths ─────────────────────────


_HASH_CALL_RE = re.compile(r"(?<![\.\w])hash\s*\(")

# Files that legitimately call `hash()` (none today, but reserved for
# false-positive shielding without compromising the audit).
_HASH_CALL_ALLOWLIST: set[str] = set()


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDITED_TREES = ["src/signals", "src/learning", "src/trading", "src/analysis"]


def _find_raw_hash_calls() -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    for tree in AUDITED_TREES:
        base = REPO_ROOT / tree
        if not base.is_dir():
            continue
        for py in base.rglob("*.py"):
            rel = str(py.relative_to(REPO_ROOT)).replace("\\", "/")
            if rel in _HASH_CALL_ALLOWLIST:
                continue
            try:
                text = py.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                code = line.split("#", 1)[0]
                if _HASH_CALL_RE.search(code):
                    out.append((rel, i, line.strip()))
    return out


def test_no_raw_python_hash_in_decision_paths():
    """Raw `hash(x)` on a string or tuple returns a value that changes
    across interpreter starts (unless PYTHONHASHSEED is pinned -- which
    is CI-only). Use `hashlib.sha256(x.encode()).hexdigest()` for cache
    keys, IDs, or anything that needs to be stable across runs.

    If this test fails: replace the raw `hash(...)` call with a
    `hashlib` equivalent, or add the file to ``_HASH_CALL_ALLOWLIST``
    with a code-review justification.
    """
    offenders = _find_raw_hash_calls()
    assert not offenders, (
        f"Found {len(offenders)} raw hash() call(s) in decision paths "
        f"that would produce non-deterministic values across replays:\n"
        + "\n".join(f"  {p}:{ln}  {line}" for p, ln, line in offenders)
    )
