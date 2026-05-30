"""Per-run replay database management.

Each replay run gets its own SQLite DB at `data/replay_<run_id>.db` so that:

  1. paper_trades / audit_trail / calibration / agent_scorer state from one
     run can't leak into another.
  2. The production `data/bot.db` is never touched by a replay -- you can
     run analyses while the live bot is trading.
  3. Forensic state is preserved across runs (the DB stays on disk after
     teardown for diffing / re-analysis).

The production code reads its DB path from `config.DB_PATH`, which is fixed
at import time. We set the path via the `HL_BOT_DB` env var BEFORE the
database module is imported -- this is the only escape hatch the current
code exposes. The harness orchestrator must therefore construct the
ReplayDB before any module that imports `src.data.database`.

Usage:
    with ReplayDB(run_id="experiment_001") as db:
        db.init_schema()
        db.reset_runtime_state()
        # ... harness runs trading_cycle against db ...
"""
from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# Tables that carry RUNTIME STATE we want fresh per phase.
# Reference data (strategies, traders) is preserved across resets.
_RUNTIME_TABLES = (
    "paper_trades",
    "audit_trail",
    "research_logs",
    "strategy_scores",
    "position_snapshots",
)


class ReplayDBError(RuntimeError):
    pass


class ReplayDB:
    """Lifecycle manager for a per-run SQLite DB.

    Owns: the file at `data/replay_<run_id>.db`, the env-var swap that
    points the production database module at it, and reset semantics on
    runtime tables.

    Does NOT own: the actual schema DDL -- delegates to `init_db` from
    `src.data.database` so the replay schema is byte-identical to live.
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        *,
        data_dir: str = "data",
        keep_on_exit: bool = True,
        clobber: bool = False,
    ):
        self.run_id = run_id or self._mint_run_id()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / f"replay_{self.run_id}.db"
        self.keep_on_exit = keep_on_exit
        self._prev_env: Optional[str] = None
        self._installed = False

        if clobber and self.db_path.exists():
            self.db_path.unlink()

    @staticmethod
    def _mint_run_id() -> str:
        # Short uuid suffix; readable enough for logs but unique per invocation.
        return f"r{uuid.uuid4().hex[:10]}"

    # ---- lifecycle ----------------------------------------------------

    def __enter__(self) -> "ReplayDB":
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.uninstall()

    def install(self) -> None:
        """Set HL_BOT_DB so subsequent database module imports pick this up.

        Idempotent. Records the previous value so uninstall restores it.
        """
        if self._installed:
            return
        self._prev_env = os.environ.get("HL_BOT_DB")
        os.environ["HL_BOT_DB"] = str(self.db_path)

        # ISOLATION: force sqlite-only so replay writes never dualwrite into the
        # production Postgres mirror.  Without this, running a replay in an
        # environment configured for dualwrite (e.g. in-container) mirrors every
        # replay INSERT to the live Postgres -- which crashes the run on primary-
        # key collisions ("strategy_scores_pkey ... already exists") and pollutes
        # prod.  The router reads config.DB_BACKEND live, and install() runs
        # before the pg pool opens, so pinning it here keeps the pool from ever
        # opening.
        self._prev_backend_env = os.environ.get("DB_BACKEND")
        os.environ["DB_BACKEND"] = "sqlite"
        try:
            import config as _cfg
            self._prev_backend_cfg = getattr(_cfg, "DB_BACKEND", None)
            _cfg.DB_BACKEND = "sqlite"
        except Exception:
            self._prev_backend_cfg = None

        self._installed = True
        logger.info(
            "ReplayDB installed: HL_BOT_DB=%s DB_BACKEND=sqlite (run_id=%s)",
            self.db_path, self.run_id,
        )

        # If `src.data.database` was already imported in this process, its
        # captured `_DB_PATH` is now stale. We patch it in place so the
        # active module observes our path. This is hacky -- but the
        # alternative is forcing every replay to be a fresh subprocess.
        self._retarget_database_module()

    def uninstall(self) -> None:
        """Restore the previous HL_BOT_DB. Optionally delete the file.

        On Windows, sqlite connections opened by production code (the
        router caches none but garbage-collected ones may still hold the
        file briefly) can leave the .db locked. We make a best-effort
        attempt to nudge GC + close any tracked connections before unlink,
        then fall back to silently leaving the file if deletion still
        fails.
        """
        if not self._installed:
            return
        try:
            if self._prev_env is None:
                os.environ.pop("HL_BOT_DB", None)
            else:
                os.environ["HL_BOT_DB"] = self._prev_env
            # Restore the dualwrite backend pinned in install().
            if getattr(self, "_prev_backend_env", None) is None:
                os.environ.pop("DB_BACKEND", None)
            else:
                os.environ["DB_BACKEND"] = self._prev_backend_env
            if getattr(self, "_prev_backend_cfg", None) is not None:
                try:
                    import config as _cfg
                    _cfg.DB_BACKEND = self._prev_backend_cfg
                except Exception:
                    pass
        finally:
            self._installed = False
            if not self.keep_on_exit and self.db_path.exists():
                self._delete_db_file()
            logger.info("ReplayDB uninstalled (run_id=%s)", self.run_id)

    def _delete_db_file(self) -> None:
        """Tear down sqlite handles + unlink. Best-effort; tolerates Windows
        WAL-mode lockholding by retrying briefly."""
        import gc
        import time as _time
        # Nudge any lingering sqlite handles to close. Production code
        # uses `with sqlite3.connect(...)` blocks so they should already
        # be released, but Windows + WAL can hold the .db-wal sidecar
        # until the connection is GC'd.
        gc.collect()
        for attempt in range(5):
            try:
                self.db_path.unlink()
                # Also remove WAL sidecars if present.
                for suffix in ("-wal", "-shm", "-journal"):
                    side = self.db_path.with_suffix(self.db_path.suffix + suffix)
                    if side.exists():
                        try:
                            side.unlink()
                        except OSError:
                            pass
                return
            except OSError as e:
                if attempt == 4:
                    logger.warning("Could not delete %s after 5 attempts: %s "
                                   "(left on disk; safe to remove manually)",
                                   self.db_path, e)
                    return
                _time.sleep(0.1)

    def _retarget_database_module(self) -> None:
        """Patch the active DB path everywhere it's already cached.

        The production code caches the DB path in three places at module
        import time:
          - config.DB_PATH         (set by config._resolve_db_path() on import)
          - src.data.database._DB_PATH  (captured from config.DB_PATH)
          - src.data.db.router    (reads config.DB_PATH lazily, no cache)

        We need to patch the first two so any module that's already loaded
        sees our replay path. The env var alone isn't enough -- it only
        affects FUTURE imports of config.py, but config is usually loaded
        very early in the process.
        """
        import sys
        targets = [
            ("config", "DB_PATH"),
            ("src.data.database", "_DB_PATH"),
        ]
        for module_name, attr in targets:
            mod = sys.modules.get(module_name)
            if mod is None:
                continue
            try:
                setattr(mod, attr, str(self.db_path))
                logger.debug("Patched %s.%s -> %s", module_name, attr, self.db_path)
            except Exception as e:
                logger.warning("Could not patch %s.%s: %s", module_name, attr, e)

    # ---- schema + reset -----------------------------------------------

    def init_schema(self) -> None:
        """Build the production schema in this DB.

        Imports `src.data.database` and calls its `init_db()`. Because we've
        already set HL_BOT_DB, the production init writes here, not to bot.db.
        Also runs `init_golden_tables()` -- the golden_wallets DDL lives in
        src/discovery/golden_wallet.py rather than database.init_db, and the
        copy_trader code path reads from it during the trading cycle.
        Without this the cycle emits a 'no such table: golden_wallets'
        SQLite warning on every run.
        """
        if not self._installed:
            raise ReplayDBError("ReplayDB.install() must be called before init_schema()")

        # Local imports: deferred so any prior imports in the calling code
        # don't capture the wrong _DB_PATH.
        from src.data import database as db
        db.init_db()

        try:
            from src.discovery.golden_wallet import init_golden_tables
            init_golden_tables()
        except Exception as e:
            logger.warning("Could not init golden_wallets tables: %s", e)

        logger.info("Schema initialised at %s", self.db_path)

    def reset_runtime_state(self) -> None:
        """Truncate runtime tables to clear bleed from any prior phase.

        Reference data (strategies, traders, bot_state) is preserved -- a
        fresh paper_trades / audit_trail / calibration set is what makes
        train/test phases independent.
        """
        if not self.db_path.exists():
            raise ReplayDBError(f"ReplayDB file does not exist yet: {self.db_path}")
        with sqlite3.connect(str(self.db_path)) as conn:
            for table in _RUNTIME_TABLES:
                try:
                    conn.execute(f"DELETE FROM {table}")
                except sqlite3.OperationalError as e:
                    # Table may not exist yet on first phase -- that's fine.
                    logger.debug("reset_runtime_state: skipping %s (%s)", table, e)
            conn.commit()
        logger.info("Runtime state reset at %s (tables: %s)", self.db_path, _RUNTIME_TABLES)

    def truncate_all(self) -> None:
        """Drop ALL data (runtime + reference). For starting a clean run."""
        if not self.db_path.exists():
            return
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            for (name,) in rows:
                try:
                    conn.execute(f"DELETE FROM {name}")
                except sqlite3.OperationalError as e:
                    logger.debug("truncate_all: skipping %s (%s)", name, e)
            conn.commit()

    def snapshot_to(self, dest_path: str) -> str:
        """Copy this DB to `dest_path` for archival / diff. Returns dest path."""
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.db_path, dest)
        logger.info("Snapshotted %s -> %s", self.db_path, dest)
        return str(dest)

    # ---- introspection ------------------------------------------------

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        """Convenience read/write cursor on the replay DB."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn.cursor()
            conn.commit()
        finally:
            conn.close()

    def table_count(self, table: str) -> int:
        with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            return int(row[0]) if row else 0

    def list_tables(self) -> list[str]:
        with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()
            return [r[0] for r in rows]

    def __repr__(self) -> str:
        state = "installed" if self._installed else "unused"
        return f"<ReplayDB run_id={self.run_id} path={self.db_path} {state}>"
