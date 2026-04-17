from src.core.health_registry import SubsystemHealthRegistry, SubsystemState
from src.core.subsystem_registry import (
    SubsystemContainer,
    _load_json_override,
    _wire_event_scanner,
    _wire_kelly_sizer_from_agent_scorer,
)


def test_load_json_override_raises_on_invalid_json():
    try:
        _load_json_override("RISK_POLICY_SOURCE_PROFILES_JSON", "{bad json")
    except ValueError as exc:
        assert "Invalid RISK_POLICY_SOURCE_PROFILES_JSON" in str(exc)
    else:
        raise AssertionError("Expected invalid JSON override to raise ValueError")


def test_kelly_wiring_failure_marks_subsystem_degraded():
    registry = SubsystemHealthRegistry()
    registry.register("kelly_sizer")
    registry.set_status("kelly_sizer", SubsystemState.HEALTHY, dependency_ready=True)

    class _BrokenKelly:
        def load_from_agent_scorer(self, _scorer):
            raise RuntimeError("load failed")

    container = SubsystemContainer(kelly_sizer=_BrokenKelly(), agent_scorer=object())

    _wire_kelly_sizer_from_agent_scorer(container, registry)

    status = registry.get_status("kelly_sizer")
    assert status is not None
    assert status.state == SubsystemState.DEGRADED
    assert "load failed" in status.reason


def test_event_scanner_wiring_failure_marks_firewall_and_scanner_degraded():
    registry = SubsystemHealthRegistry()
    registry.register("event_scanner", affects_trading=False)
    registry.register("decision_firewall")
    registry.set_status("event_scanner", SubsystemState.HEALTHY, dependency_ready=True)
    registry.set_status("decision_firewall", SubsystemState.HEALTHY, dependency_ready=True)

    class _BrokenFirewall:
        def set_event_scanner(self, _scanner):
            raise RuntimeError("boom")

    container = SubsystemContainer(firewall=_BrokenFirewall(), event_scanner=object())

    _wire_event_scanner(container, registry)

    assert registry.get_status("event_scanner").state == SubsystemState.DEGRADED
    assert registry.get_status("decision_firewall").state == SubsystemState.DEGRADED
