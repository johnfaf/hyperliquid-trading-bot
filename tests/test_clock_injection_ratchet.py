"""CI ratchet: enforce clock-injection discipline on decision paths.

Background
----------
`src/core/clock_provider.py` already provides a swappable time source
(`utc_now()`, `unix_now()`, `unix_ms()`, `iso_now()`) that the replay
harness uses to slide all time reads back to a deterministic window.
But ~149 sites in src/signals/, src/learning/, src/trading/, and
src/analysis/ still call `time.time()` / `datetime.now()` directly,
bypassing the injection point. Those calls return the *current* wall
clock during replay -- not the replay clock -- which is the
"silent replay-vs-prod skew" structural risk in the audit roadmap.

This test enforces a one-way ratchet: the snapshot count is recorded,
and any PR that *adds* a new raw time-read in the four decision module
trees fails CI with a pointer at the canonical helper. Each migration
PR (replacing a raw call with `clock_provider.utc_now()` etc.) is
expected to *decrement* the snapshot.

We do not aim for zero immediately. Bulk-replacing 149 sites in one
commit is the kind of "big risky change" the audit roadmap explicitly
calls out as harmful. The ratchet gives us incremental rollout with
no possibility of regression.

Allowlist
---------
Files that are LEGITIMATE consumers of raw stdlib time:
  * src/core/clock_provider.py and src/backtest/replay/clock.py
    (these IMPLEMENT the provider; they must call the stdlib).
  * Test fixtures and scripts (we only audit src/).
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

# The 4 decision-relevant trees the replay harness drives. We do NOT
# audit src/core/, src/data/, src/exchanges/, src/notifications/,
# src/ui/, src/discovery/ because:
#   - They contain a lot of logging / persistence / HTTP code where
#     wall-clock is intentional.
#   - The replay harness drives the four trees below; those are the
#     paths where skew matters.
AUDITED_TREES = ["signals", "learning", "trading", "analysis"]

# Files that legitimately call stdlib time (they IMPLEMENT the provider).
_ALLOWLIST_FILES = {
    "src/core/clock_provider.py",
    "src/backtest/replay/clock.py",
}


_TIME_RE = re.compile(r"\btime\.time\s*\(")
_DT_NOW_RE = re.compile(r"\bdatetime\.(?:now|utcnow)\s*\(")


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT)).replace("\\", "/")


def _scan_audited_trees() -> list[tuple[str, int, str]]:
    """Return (rel_path, lineno, line) for every raw time-read site
    in the audited trees, excluding the allowlist."""
    out: list[tuple[str, int, str]] = []
    for tree in AUDITED_TREES:
        base = SRC_ROOT / tree
        if not base.is_dir():
            continue
        for py in base.rglob("*.py"):
            rel = _rel(py)
            if rel in _ALLOWLIST_FILES:
                continue
            try:
                text = py.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                # Skip lines that are entirely commented out.
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                # Strip trailing comments before testing.
                code = line.split("#", 1)[0]
                if _TIME_RE.search(code) or _DT_NOW_RE.search(code):
                    out.append((rel, i, line.strip()))
    return out


# Snapshot from the commit landing this test. New occurrences fail CI;
# each migration PR (replacing a raw call with clock_provider) is
# expected to decrement this number.
#
# When you migrate one or more sites:
#   1. Replace `datetime.now(timezone.utc)` with
#      `clock_provider.utc_now()`.
#   2. Replace `time.time()` with `clock_provider.unix_now()`.
#   3. Replace `int(time.time() * 1000)` with `clock_provider.unix_ms()`.
#   4. Replace `datetime.now(timezone.utc).isoformat()` with
#      `clock_provider.iso_now()`.
#   5. Decrement SNAPSHOT_TOTAL by the number of migrated occurrences
#      and run pytest -q tests/test_clock_injection_ratchet.py to confirm.
SNAPSHOT_TOTAL = 140


def test_no_new_raw_time_reads_on_decision_paths():
    """If this test fails: a new raw `time.time()` / `datetime.now()` was
    added in src/{signals,learning,trading,analysis}/ that bypasses the
    canonical clock_provider. Replace it with the injection helper::

        from src.core import clock_provider
        now = clock_provider.utc_now()        # datetime
        ts  = clock_provider.unix_now()       # float seconds
        ms  = clock_provider.unix_ms()        # int ms
        iso = clock_provider.iso_now()        # ISO string

    The provider falls back to wall-clock in production (zero behavior
    change) and is swapped to the deterministic replay clock when the
    OOS replay harness boots subsystems.
    """
    offenders = _scan_audited_trees()
    msg = (
        f"Found {len(offenders)} raw time-read sites in the audited "
        f"trees ({', '.join(AUDITED_TREES)}). Snapshot ratchet: "
        f"must be <= {SNAPSHOT_TOTAL} until each is migrated.\n"
        f"First 20 sites:\n"
        + "\n".join(f"  {p}:{ln}  {line}" for p, ln, line in offenders[:20])
    )
    assert len(offenders) <= SNAPSHOT_TOTAL, msg


def test_clock_provider_is_canonical_helper():
    """Smoke: the provider's public surface is callable and behaves
    correctly in production (LiveClock backend)."""
    from src.core import clock_provider
    # Wall clock backend by default; numbers should be positive and
    # increase monotonically (no exact value asserted).
    t1 = clock_provider.unix_now()
    t2 = clock_provider.unix_ms()
    iso = clock_provider.iso_now()
    dt = clock_provider.utc_now()
    assert t1 > 0
    assert t2 > 0
    assert isinstance(iso, str) and iso.endswith("Z")
    assert dt is not None and dt.tzinfo is not None


def test_allowlist_files_exist():
    """Sanity: the files we exempt from the audit must actually exist,
    otherwise the allowlist is misconfigured and we have a false-pass."""
    for rel in _ALLOWLIST_FILES:
        path = REPO_ROOT / rel
        assert path.exists(), (
            f"Allowlisted file {rel} does not exist -- update _ALLOWLIST_FILES"
        )


def test_snapshot_is_within_expected_range():
    """Locked-in invariant: the snapshot must be > 100 (we have a
    real legacy backlog) and <= 200 (we haven't regressed
    catastrophically). If you see this fail, either the ratchet was
    broken by a bulk addition (>50 new sites) OR a successful migration
    has shrunk the backlog enough that the lower-bound needs updating."""
    offenders = _scan_audited_trees()
    assert 100 <= len(offenders) <= 200, (
        f"Sanity range broken: {len(offenders)} raw time-reads found. "
        "Was a bulk migration just landed? Update the range here."
    )
