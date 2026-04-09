"""
Shared pytest configuration.

Adds the project root to sys.path once so individual test files
don't need their own sys.path.insert hacks.
"""
import sys
import os

# Add project root to path so `import config` and `from src.xxx` work
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
