"""Build a SubsystemContainer wired for replay.

`subsystem_registry.build_subsystems(REPLAY_PROFILE)` instantiates the real
post-signal decision pipeline (scorer + firewall + paper_trader + regime +
sizing + calibration). After that returns, we overlay the stub subsystems
on the container so the live data sources (polymarket, macro_regime, etc.)
are inert. The harness uses this instead of building a container itself --
keeping all subsystem construction in one place.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from src.core.subsystem_registry import (
    REPLAY_PROFILE, SubsystemContainer, build_subsystems,
)
from src.core.health_registry import SubsystemHealthRegistry
from src.backtest.replay.stub_subsystems import _StubBase, all_stubs

logger = logging.getLogger(__name__)


# Mapping from container attribute -> stub factory name. The container
# attribute is the field on SubsystemContainer; the stub key is what
# `all_stubs()` returns.
_STUB_OVERLAY: Dict[str, str] = {
    "polymarket": "polymarket",
    "options_scanner": "options_scanner",
    "macro_regime": "macro_regime",
    "event_scanner": "event_scanner",
    "exchange_agg": "exchange_agg",
    "multi_scanner": "multi_scanner",
    "predictive_forecaster": "predictive_forecaster",
    "cross_venue_hedger": "cross_venue_hedger",
}


def build_replay_container(
    health: Optional[SubsystemHealthRegistry] = None,
) -> tuple[SubsystemContainer, Dict[str, _StubBase]]:
    """Build the container for replay + return the stub bag.

    Returns:
        container: a SubsystemContainer with real decision-pipeline
            subsystems and stub overlays for data sources we can't replay.
        stubs: dict of stub_name -> stub instance for telemetry / asserts.
    """
    health = health or SubsystemHealthRegistry()
    container = build_subsystems(health=health, profile=REPLAY_PROFILE)
    stubs = all_stubs()
    overlay_stubs(container, stubs)
    _rewire_firewall_event_scanner(container)
    return container, stubs


def overlay_stubs(
    container: SubsystemContainer, stubs: Dict[str, _StubBase],
) -> None:
    """Replace data-source subsystem slots on the container with stubs.

    `build_subsystems` may already have populated these slots (the REPLAY
    profile includes the names so the registry attempts construction).
    We overwrite unconditionally so the harness has a consistent
    overlay even if real construction succeeded.
    """
    for container_attr, stub_key in _STUB_OVERLAY.items():
        stub = stubs.get(stub_key)
        if stub is None:
            logger.warning("No stub for slot %s (key=%s)", container_attr, stub_key)
            continue
        prev = getattr(container, container_attr, None)
        setattr(container, container_attr, stub)
        if prev is not None:
            logger.debug("Overlayed stub onto container.%s (replaced %r)",
                         container_attr, type(prev).__name__)
        else:
            logger.debug("Installed stub at container.%s", container_attr)


def _rewire_firewall_event_scanner(container: SubsystemContainer) -> None:
    """The registry's _wire_event_scanner ran before our overlay. Redo it
    with the stub instance so the firewall actually consults the stub
    rather than whatever instance (or None) was there before."""
    fw = getattr(container, "firewall", None)
    es = getattr(container, "event_scanner", None)
    if fw is None or es is None:
        return
    if not hasattr(fw, "set_event_scanner"):
        return
    try:
        fw.set_event_scanner(es)
        logger.debug("Rewired firewall.event_scanner -> stub")
    except Exception as e:
        logger.warning("Could not rewire firewall event_scanner: %s", e)
