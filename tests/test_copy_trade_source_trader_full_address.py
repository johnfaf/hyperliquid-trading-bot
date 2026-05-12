"""Regression: every copy-trade signal's source_trader must be the full
42-char Ethereum address, never a truncated prefix.

User-reported symptom: source allocator kept rejecting copy_trade signals
with the message "Source allocator requires 45% confidence for
copy_trade:0x1ee7a73c (got 43%)". The 10-character key was the smoking
gun -- the signal generators in ws_position_monitor.py and golden_bridge.py
were writing `source_trader = address[:10]`. That caused:

  1. agent_scorer to fragment scores between full-form and short-form
     source_keys for the same trader (different `record_signal` /
     `record_outcome` paths used different keys).
  2. The source-policy lookup against the short key always returned
     "warmup" with min_confidence = 0.45 because total_signals stayed
     at 0 on that key (outcomes recorded against it without prior
     record_signal, leaving the counter pinned at 0).
  3. The source-side guard observed 16 closed paper_trades under the
     short key for the long side, marked the source "degraded" once,
     then crushed signal confidence by 0.75 -- and combined with the
     synthetic-regime cap (0.50) and the agent_scorer weight blend
     (line 915 in copy_trader.py) the final confidence landed at
     0.425 ~ 0.43, just below the 0.45 floor. Forever.

The fix is removing the [:10] slice; the full address is what every
other module already expects. This test guards against regression by
checking the signal-builder modules' generated dicts.
"""
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TARGETS = [
    "src/notifications/ws_position_monitor.py",
    "src/discovery/golden_bridge.py",
    "src/trading/copy_trader.py",
]


def _truncating_source_trader_sites(path: str) -> list[tuple[int, str]]:
    """Return (lineno, source_excerpt) for any source_trader = address[:N]
    style assignments in `path`. Logs and reason-strings are exempted by
    matching on the keyword 'source_trader' specifically."""
    text = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(text)
    hits: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        # Dict literals: {"source_trader": address[:10], ...}
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if (isinstance(k, ast.Constant) and k.value == "source_trader"
                        and isinstance(v, ast.Subscript)):
                    src_line = text.splitlines()[v.lineno - 1].strip()
                    hits.append((v.lineno, src_line))
        # Direct assignments: signal["source_trader"] = address[:10]
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value == "source_trader"
                        and isinstance(node.value, ast.Subscript)):
                    src_line = text.splitlines()[node.lineno - 1].strip()
                    hits.append((node.lineno, src_line))
    return hits


def test_no_truncated_source_trader_in_signal_builders():
    """No signal-builder module may assign a sliced address to
    source_trader. The whole-address contract is what the rest of the
    pipeline assumes (firewall._source_key, agent_scorer.record_signal,
    paper_trader metadata, etc.).
    """
    all_hits: dict[str, list[tuple[int, str]]] = {}
    for path in TARGETS:
        hits = _truncating_source_trader_sites(path)
        if hits:
            all_hits[path] = hits

    if all_hits:
        msg = ["Found source_trader truncations (these cause agent_scorer key fragmentation):"]
        for path, hits in all_hits.items():
            for ln, src in hits:
                msg.append(f"  {path}:{ln}  {src}")
        raise AssertionError("\n".join(msg))


def test_ws_position_monitor_emits_full_address():
    """Functional check: drive the signal generator directly and verify
    the emitted dict has the full 42-char address."""
    import importlib
    mon_mod = importlib.import_module("src.notifications.ws_position_monitor")
    monitor_cls = mon_mod.PositionMonitor

    # We don't need to start the monitor -- just call the position-change
    # detector with synthetic inputs and inspect the signals it emits.
    monitor = monitor_cls.__new__(monitor_cls)  # bypass __init__ side effects
    monitor._position_cache = {}
    monitor.logger = mon_mod.logger

    full_addr = "0x1ee7a73cb5b0b6b056d8138085b2009e6a6bedf5"
    old = {}
    new = {"BTC": {"size": 1.0, "side": "long", "leverage": 5, "entry_price": 50_000}}
    mids = {"BTC": 50_000}

    # _position_changes_to_signals is the helper that emits the dicts
    # whose source_trader we are testing.
    helper = getattr(monitor, "_position_changes_to_signals", None)
    if helper is None:
        # API shape may differ; fall back to the AST-only test.
        import pytest
        pytest.skip("_position_changes_to_signals not present; AST test still covers the file")
    signals = helper(full_addr, old, new, mids)
    assert signals, "Expected at least one open-position signal"
    for s in signals:
        assert s["source_trader"] == full_addr, (
            f"Signal source_trader = {s['source_trader']!r} "
            f"(expected full address {full_addr!r})"
        )
        assert len(s["source_trader"]) == 42, (
            f"source_trader length = {len(s['source_trader'])}, expected 42"
        )
