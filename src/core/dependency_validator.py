"""
Boot-Time Dependency Validator
==============================
Validates that all Python packages required by enabled features are actually
importable before the bot starts.  This eliminates the single biggest source
of silent degradation: a feature flag is True but the package behind it was
never installed, so the subsystem silently catches ImportError and runs in a
crippled no-op mode that *looks* healthy to the scheduler.

Usage (in boot.py or main.py)::

    from src.core.dependency_validator import validate_or_fail, get_boot_report
    validate_or_fail()        # raises RuntimeError if a required dep is missing
    logger.info(get_boot_report())   # pretty summary for the startup log
"""
import importlib.util
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Feature → packages required for that feature to function
# ---------------------------------------------------------------------------
FEATURE_DEPENDENCIES: Dict[str, List[str]] = {
    # Always required — bot will not start without these
    "core": [
        "requests",
        "pandas",
        "numpy",
        "websocket",       # websocket-client
        "eth_account",
    ],
    # Optional overlays — only validated when their config flag is True
    "ml_regime": [
        "xgboost",
        "sklearn",         # scikit-learn
    ],
    "polymarket": [
        "requests",        # already in core but explicit for clarity
    ],
    "options_flow": [
        "requests",
    ],
    "cross_venue_hedger": [
        "requests",
    ],
    "telegram": [
        "requests",
    ],
    "dashboard": [
        "flask",
    ],
    "backtester": [
        "pandas",
        "numpy",
    ],
}

# ---------------------------------------------------------------------------
# Feature → config attribute that gates it.  None = always checked if present.
# "core" is implicitly always required and is not listed here.
# ---------------------------------------------------------------------------
FEATURE_CONFIG_FLAGS: Dict[str, Optional[str]] = {
    "ml_regime":           "ENABLE_XGBOOST_FORECASTER",
    "polymarket":          "POLYMARKET_ENABLED",
    "options_flow":        "OPTIONS_FLOW_ENABLED",
    "cross_venue_hedger":  None,   # available if imports work
    "telegram":            None,   # enabled by env vars at runtime
    "dashboard":           None,
    "backtester":          None,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _can_import(package_name: str) -> bool:
    """Return True if *package_name* is importable on this Python install."""
    try:
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    except (ModuleNotFoundError, ValueError):
        return False


def validate_feature(feature_name: str) -> Tuple[bool, List[str]]:
    """
    Check whether every package required by *feature_name* is importable.

    Returns
    -------
    (all_ok, missing)
        *all_ok* is True when every dependency is present.
        *missing* is the list of packages that could not be found.
    """
    deps = FEATURE_DEPENDENCIES.get(feature_name, [])
    missing = [pkg for pkg in deps if not _can_import(pkg)]
    return (len(missing) == 0, missing)


def _is_feature_enabled(feature_name: str, config_module=None) -> bool:
    """Decide whether a feature should be validated based on config flags."""
    if feature_name == "core":
        return True  # always required

    flag = FEATURE_CONFIG_FLAGS.get(feature_name)
    if flag is None:
        return True  # no config flag → always check

    if config_module is None:
        # Try to import the project config
        try:
            import config as config_module  # noqa: F811
        except ImportError:
            return True  # can't read config → assume enabled, validate anyway

    return bool(getattr(config_module, flag, False))


def validate_all(config_module=None) -> Dict[str, Dict]:
    """
    Check every registered feature, respecting config flags.

    Returns
    -------
    dict keyed by feature name::

        {
            "core": {"enabled": True, "available": True, "missing": []},
            "ml_regime": {"enabled": False, "available": False, "missing": ["xgboost", "sklearn"]},
            ...
        }
    """
    results: Dict[str, Dict] = {}

    for feature in FEATURE_DEPENDENCIES:
        enabled = _is_feature_enabled(feature, config_module)
        all_ok, missing = validate_feature(feature)
        results[feature] = {
            "enabled": enabled,
            "available": all_ok,
            "missing": missing,
        }

    return results


def validate_or_fail(features: Optional[List[str]] = None,
                     config_module=None) -> None:
    """
    Fail fast if any *enabled* feature is missing required packages.

    Parameters
    ----------
    features : list[str] | None
        Specific features to check.  ``None`` means check everything.
    config_module
        Optional reference to your ``config`` module.  If omitted the
        validator tries ``import config`` from the project root.

    Raises
    ------
    RuntimeError
        Human-readable message listing every enabled feature whose
        dependencies are not satisfied.
    """
    report = validate_all(config_module)

    if features is not None:
        report = {k: v for k, v in report.items() if k in features}

    failures: List[str] = []
    for feature, info in report.items():
        if info["enabled"] and not info["available"]:
            failures.append(
                f"  {feature}: missing {', '.join(info['missing'])}"
            )

    if failures:
        msg = (
            "Boot-time dependency check FAILED.\n"
            "The following enabled features have missing packages:\n"
            + "\n".join(failures)
            + "\n\nInstall them or disable the feature in config / env vars."
        )
        logger.critical(msg)
        raise RuntimeError(msg)

    logger.info("Dependency validation passed — all enabled features have their packages.")


def get_boot_report(config_module=None) -> str:
    """
    Return a human-readable summary suitable for the startup log.

    Example output::

        ┌─ Dependency Boot Report ────────────────────┐
        │ ✓ core              READY                   │
        │ ✓ polymarket        READY                   │
        │ ✗ ml_regime         MISSING xgboost, sklearn│
        │ — dashboard         DISABLED                │
        └─────────────────────────────────────────────┘
    """
    report = validate_all(config_module)

    lines = ["", "┌─ Dependency Boot Report ─────────────────────────┐"]

    for feature, info in sorted(report.items()):
        enabled = info["enabled"]
        available = info["available"]
        missing = info["missing"]

        if not enabled:
            icon = "—"
            status = "DISABLED"
        elif available:
            icon = "✓"
            status = "READY"
        else:
            icon = "✗"
            status = f"MISSING {', '.join(missing)}"

        line = f"│ {icon} {feature:<22s} {status:<25s} │"
        lines.append(line)

    lines.append("└───────────────────────────────────────────────────┘")
    return "\n".join(lines)
