"""
Subsystem health registry for tracking and monitoring trading bot components.

Provides centralized health monitoring with states (HEALTHY, DEGRADED, DISABLED, FAILED)
and trading safety checks. Thread-safe with automatic stale heartbeat detection.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)
# Prevent "No handler found" warnings and stderr lastResort output
# when the library is used before the application configures logging.
logger.addHandler(logging.NullHandler())


class SubsystemState(Enum):
    """Health states for registered subsystems."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    DISABLED = "DISABLED"
    FAILED = "FAILED"


@dataclass
class SubsystemStatus:
    """Status information for a registered subsystem."""
    name: str
    state: SubsystemState
    reason: str = ""
    last_heartbeat: Optional[datetime] = None
    startup_status: str = "PENDING"
    dependency_ready: bool = False
    affects_trading: bool = True
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_trading_safe(self) -> bool:
        """Check if this subsystem is safe for trading operations."""
        return (
            self.state in (SubsystemState.HEALTHY, SubsystemState.DEGRADED)
            and self.affects_trading
            and self.dependency_ready
        )

    def __str__(self) -> str:
        """Human-readable status representation."""
        status_str = f"{self.name}: {self.state.value}"
        if self.reason:
            status_str += f" ({self.reason})"
        if self.last_heartbeat:
            status_str += f" [last heartbeat: {self.last_heartbeat.isoformat()}]"
        return status_str


class SubsystemHealthRegistry:
    """
    Thread-safe registry for tracking health status of trading bot subsystems.

    Manages registration, state tracking, heartbeat monitoring, and trading safety checks
    for all system components. Supports automatic degradation of stale subsystems.
    """

    def __init__(self):
        """Initialize the health registry with thread safety."""
        self._subsystems: Dict[str, SubsystemStatus] = {}
        self._lock = threading.RLock()
        # Optional callback invoked when any subsystem transitions to FAILED.
        # Signature: callback(subsystem_name: str, reason: str) -> None
        # Set via set_failure_callback() to avoid circular imports.
        self._on_failure_callback: Optional[Callable[[str, str], None]] = None
        logger.debug("SubsystemHealthRegistry initialized")

    def set_failure_callback(self, callback: Callable[[str, str], None]) -> None:
        """Register a callback for FAILED transitions (e.g. Telegram alert).

        Args:
            callback: Called with (subsystem_name, reason) when a subsystem
                      transitions to FAILED.  Must be non-blocking.
        """
        self._on_failure_callback = callback

    def register(self, name: str, affects_trading: bool = True) -> SubsystemStatus:
        """
        Register a subsystem.  Idempotent — re-registering the same name
        updates ``affects_trading`` but does not raise.

        Args:
            name: Unique identifier for the subsystem
            affects_trading: Whether this subsystem's outputs can affect trading decisions

        Returns:
            SubsystemStatus: The created or updated status object
        """
        with self._lock:
            if name in self._subsystems:
                # Idempotent update — no error on re-registration
                self._subsystems[name].affects_trading = affects_trading
                logger.debug("Re-registered subsystem '%s' (affects_trading=%s)", name, affects_trading)
                return self._subsystems[name]

            status = SubsystemStatus(
                name=name,
                state=SubsystemState.HEALTHY,
                startup_status="PENDING",
                dependency_ready=False,
                affects_trading=affects_trading,
                registered_at=datetime.now(timezone.utc)
            )
            self._subsystems[name] = status
            logger.debug("Registered subsystem '%s' (affects_trading=%s)", name, affects_trading)
            return status

    def heartbeat(self, name: str) -> None:
        """
        Update the last heartbeat timestamp for a subsystem.
        Silently ignores unknown names (non-throwing) so callers don't
        need try/except around every heartbeat.
        """
        with self._lock:
            sub = self._subsystems.get(name)
            if sub is None:
                logger.debug("heartbeat() for unknown subsystem '%s' — ignored", name)
                return
            sub.last_heartbeat = datetime.now(timezone.utc)
            if sub.state == SubsystemState.DEGRADED and (
                sub.reason.startswith("No heartbeat")
                or sub.reason.startswith("Stale heartbeat")
            ):
                sub.state = SubsystemState.HEALTHY
                sub.reason = ""

    def set_status(
        self,
        name: str,
        state: SubsystemState,
        reason: str = "",
        dependency_ready: Optional[bool] = None,
        startup_status: Optional[str] = None,
    ) -> None:
        """
        Update the state of a registered subsystem.
        Silently ignores unknown names (non-throwing).

        Logs at WARNING only for transitions *into* FAILED.  All other
        transitions are logged at INFO to avoid polluting production logs
        with expected state changes.
        """
        if not isinstance(state, SubsystemState):
            raise TypeError(f"state must be SubsystemState, got {type(state)}")

        with self._lock:
            sub = self._subsystems.get(name)
            if sub is None:
                logger.debug("set_status() for unknown subsystem '%s' — ignored", name)
                return

            old_state = sub.state
            sub.state = state
            sub.reason = reason
            if dependency_ready is not None:
                sub.dependency_ready = dependency_ready
            if startup_status is not None:
                sub.startup_status = startup_status

            if old_state != state:
                # Only WARN on transitions into FAILED — everything else is INFO
                level = logging.WARNING if state == SubsystemState.FAILED else logging.INFO
                logger.log(
                    level,
                    "Subsystem '%s' %s -> %s%s",
                    name, old_state.value, state.value,
                    f" ({reason})" if reason else "",
                )
                # Fire alert callback on transitions INTO FAILED so the
                # operator knows immediately (e.g. Telegram critical alert).
                if state == SubsystemState.FAILED and self._on_failure_callback:
                    try:
                        self._on_failure_callback(name, reason)
                    except Exception as cb_err:
                        logger.warning(
                            "Failure callback for '%s' raised: %s", name, cb_err,
                        )

    def get_status(self, name: str) -> Optional[SubsystemStatus]:
        """
        Retrieve the current status of a subsystem.

        Args:
            name: Subsystem identifier

        Returns:
            SubsystemStatus object or None if not registered
        """
        with self._lock:
            return self._subsystems.get(name)

    def get_all(self) -> Dict[str, SubsystemStatus]:
        """
        Get a snapshot of all registered subsystem statuses.

        Returns:
            Dictionary mapping subsystem names to SubsystemStatus objects
        """
        with self._lock:
            return dict(self._subsystems)

    def is_trading_safe(self, name: str) -> bool:
        """
        Check if a specific subsystem is safe for trading operations.

        A subsystem is trading-safe if:
        - It is registered
        - Its state is HEALTHY or DEGRADED
        - affects_trading is True

        Args:
            name: Subsystem identifier

        Returns:
            True if safe for trading, False otherwise
        """
        with self._lock:
            if name not in self._subsystems:
                return False
            return self._subsystems[name].is_trading_safe()

    def is_all_trading_safe(self) -> bool:
        """
        Check if all trading-affecting subsystems are safe for trading.

        Returns:
            True if all trading-affecting subsystems are HEALTHY or DEGRADED, False otherwise
        """
        with self._lock:
            trading_subsystems = [
                s for s in self._subsystems.values()
                if s.affects_trading
            ]
            if not trading_subsystems:
                return True
            return all(
                s.state in (SubsystemState.HEALTHY, SubsystemState.DEGRADED) and s.dependency_ready
                for s in trading_subsystems
            )

    def check_stale(self, timeout_seconds: int = 300) -> Dict[str, bool]:
        """
        Check for stale heartbeats and auto-degrade subsystems that exceed timeout.

        This method should be called periodically (e.g., every 60 seconds) to detect
        subsystems that have stopped sending heartbeats.

        Args:
            timeout_seconds: Heartbeat timeout threshold (default 300 seconds)

        Returns:
            Dictionary mapping subsystem names to whether they were auto-degraded
        """
        now = datetime.now(timezone.utc)
        degraded = {}

        with self._lock:
            for name, status in self._subsystems.items():
                # Skip subsystems that are already in a terminal state
                if status.state == SubsystemState.DISABLED:
                    degraded[name] = False
                    continue

                # Check if heartbeat is missing or stale
                if status.last_heartbeat is None:
                    # Never received a heartbeat
                    time_since_registration = (
                        now - status.registered_at
                    ).total_seconds()
                    if time_since_registration > timeout_seconds:
                        stale_reason = f"No heartbeat for {timeout_seconds}s"
                        already_stale = (
                            status.state == SubsystemState.DEGRADED
                            and status.reason == stale_reason
                        )
                        self._subsystems[name].state = SubsystemState.DEGRADED
                        self._subsystems[name].reason = stale_reason
                        degraded[name] = True
                        if not already_stale:
                            logger.warning(
                                f"Auto-degraded '{name}': no heartbeat received since registration"
                            )
                    else:
                        degraded[name] = False
                else:
                    # Check for stale heartbeat
                    time_since_heartbeat = (
                        now - status.last_heartbeat
                    ).total_seconds()
                    if time_since_heartbeat > timeout_seconds:
                        stale_reason = f"Stale heartbeat ({time_since_heartbeat:.0f}s ago)"
                        already_stale = (
                            status.state == SubsystemState.DEGRADED
                            and status.reason.startswith("Stale heartbeat")
                        )
                        self._subsystems[name].state = SubsystemState.DEGRADED
                        self._subsystems[name].reason = stale_reason
                        degraded[name] = True
                        if not already_stale:
                            logger.warning(
                                f"Auto-degraded '{name}': stale heartbeat ({time_since_heartbeat:.0f}s ago)"
                            )
                    else:
                        degraded[name] = False

        return degraded

    def get_health_report(self) -> str:
        """
        Generate a formatted health report of all registered subsystems.

        Returns:
            Multi-line string summarizing the health status of all subsystems
        """
        with self._lock:
            if not self._subsystems:
                return "No subsystems registered"

            lines = ["Health Report:"]
            lines.append("-" * 80)

            # Group by state for better readability
            by_state = {}
            for status in self._subsystems.values():
                state = status.state.value
                if state not in by_state:
                    by_state[state] = []
                by_state[state].append(status)

            # Print in priority order
            state_order = [
                SubsystemState.FAILED,
                SubsystemState.DISABLED,
                SubsystemState.DEGRADED,
                SubsystemState.HEALTHY
            ]

            for state in state_order:
                state_val = state.value
                if state_val in by_state:
                    lines.append(f"\n{state_val}:")
                    for status in by_state[state_val]:
                        uptime = self._format_uptime(status.registered_at)
                        lines.append(f"  - {status.name}")
                        lines.append(f"      Registered: {status.registered_at.isoformat()} ({uptime})")
                        lines.append(f"      Affects Trading: {status.affects_trading}")
                        lines.append(f"      Dependencies Ready: {status.dependency_ready}")
                        if status.reason:
                            lines.append(f"      Reason: {status.reason}")
                        if status.last_heartbeat:
                            time_since = (
                                datetime.now(timezone.utc) - status.last_heartbeat
                            ).total_seconds()
                            lines.append(f"      Last Heartbeat: {time_since:.1f}s ago")

            lines.append("-" * 80)

            # Summary line
            all_safe = self.is_all_trading_safe()
            trading_status = "SAFE" if all_safe else "AT RISK"
            lines.append(f"Trading Status: {trading_status}")

            return "\n".join(lines)

    @staticmethod
    def _format_uptime(registered_at: datetime) -> str:
        """Format uptime duration as human-readable string."""
        now = datetime.now(timezone.utc)
        delta = now - registered_at
        seconds = int(delta.total_seconds())

        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes}m"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"

    def reset(self) -> None:
        """
        Clear all registered subsystems.

        WARNING: This is primarily for testing purposes. Use with caution in production.
        """
        with self._lock:
            count = len(self._subsystems)
            self._subsystems.clear()
            logger.warning(f"Health registry reset: {count} subsystems cleared")


# Module-level singleton for easy import
registry = SubsystemHealthRegistry()
