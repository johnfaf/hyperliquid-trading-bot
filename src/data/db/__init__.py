"""
Database backend abstraction layer.

Provides a unified interface over SQLite and Postgres so the rest of the
codebase doesn't need to care which storage engine is active.

Usage — callers continue to use ``database.py`` public functions as before.
This package is an internal implementation detail.
"""
