"""Shared component registry for v2 dashboard.

The bot constructs all its subsystems in
:mod:`src.core.subsystem_registry`. The v2 dashboard reads from those
live objects (it never reaches around them into the DB unless the
component is unavailable). We keep references in a module-level
dataclass and let the FastAPI routes pull from it through a single
``get_components()`` accessor.

This is deliberately not a global ``app.state`` — multiple test
clients can spin up the app independently, and we want the
production binding to survive across reloads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class V2Components:
    firewall: Any = None
    calibration: Any = None
    agent_scorer: Any = None
    kelly_sizer: Any = None
    regime_detector: Any = None
    arena: Any = None
    arena_incubator: Any = None
    decision_engine: Any = None
    signal_processor: Any = None
    multi_scanner: Any = None
    event_scanner: Any = None
    shadow_tracker: Any = None
    trade_memory: Any = None
    llm_filter: Any = None
    liquidation_strategy: Any = None
    options_scanner: Any = None
    copy_trader: Any = None
    live_trader: Any = None
    health_registry: Any = None
    extras: dict = field(default_factory=dict)


_components = V2Components()


def set_components(**kwargs: Any) -> None:
    """Replace registered components. Unknown keys land in ``extras``.

    Called from the boot sequence right after the subsystem registry
    has finished initializing. Callable repeatedly — a re-bind during
    a hot reload should just update references.
    """
    extras = {}
    for key, value in kwargs.items():
        if hasattr(_components, key):
            setattr(_components, key, value)
        else:
            extras[key] = value
    if extras:
        _components.extras.update(extras)


def get_components() -> V2Components:
    return _components


def reset_components() -> None:
    """Test-only: clear registered components."""
    global _components
    _components = V2Components()
