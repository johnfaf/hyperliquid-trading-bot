"""Replay harness orchestrator.

Boots a causal, deterministic version of the bot's data plane:
  - swaps the clock provider's backend to a ReplayClock
  - installs the api_manager shim so all Hyperliquid traffic is intercepted
  - engages the network sandbox so any direct HTTP request raises
  - exposes a `tick` API that advances the clock one step

What's NOT in this v1:
  - Full trading_cycle invocation. The cycle pulls scorer/discovery/firewall/
    paper_trader, all of which need their own state setup (strategy pool,
    fresh paper_trades DB, calibration reset). v1 wires the data plane and
    proves causality; v2 wires the decision plane on top.
  - Multi-coin universe; the regime detector takes a 10-coin weighted vote
    in production but we only have BTC in cache today.
  - ML model freezing; alpha_pipeline and xgboost_forecaster are disabled.

Usage:
    with ReplayHarness(start_ts_ms=..., end_ts_ms=..., cache_db=...) as h:
        for tick in h.iter_ticks(step_ms=3_600_000):
            mids = h.api.post({"type": "allMids"})
            ...
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from src.backtest.replay.clock import ReplayClock
from src.backtest.replay.candle_oracle import CandleOracle
from src.backtest.replay.api_manager_shim import (
    ReplayAPIManager,
    install_replay_manager,
    uninstall_replay_manager,
)
from src.backtest.replay.network_sandbox import engage as engage_sandbox
from src.backtest.replay.network_sandbox import disengage as disengage_sandbox
from src.backtest.replay.stub_subsystems import all_stubs
from src.backtest.replay.replay_db import ReplayDB
from src.core import clock_provider

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DB = "data/candle_cache.db"


@dataclass
class ReplayTick:
    """One step of the replay. The clock has been advanced; subsystems may
    now read at this t."""
    index: int
    ts_ms: int

    def __repr__(self) -> str:
        return f"<Tick #{self.index} @ {self.ts_ms}>"


@dataclass
class ReplayReport:
    """Telemetry collected during a replay run. Surfaces what was actually
    exercised + what was stubbed so the operator can audit the run."""
    start_ts_ms: int
    end_ts_ms: int
    step_ms: int
    tick_count: int = 0
    api_calls_by_type: Dict[str, int] = field(default_factory=dict)
    api_coin_cache_misses: Dict[str, int] = field(default_factory=dict)
    stub_calls: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Replay window: ms[{self.start_ts_ms}, {self.end_ts_ms}], step={self.step_ms} ms",
            f"Ticks: {self.tick_count}",
            f"API calls by type: {self.api_calls_by_type or 'none'}",
        ]
        if self.api_coin_cache_misses:
            lines.append(f"Coin cache misses (coin not in oracle): {self.api_coin_cache_misses}")
        if self.stub_calls:
            lines.append("Stubbed subsystem calls:")
            for name, calls in self.stub_calls.items():
                if calls:
                    lines.append(f"  {name}: {calls}")
        return "\n".join(lines)


class ReplayHarness:
    """Context-manager-style replay environment.

    Owns the lifecycle of the swappable globals: clock provider, api_manager
    singleton, network sandbox. Always use as a `with` block so teardown is
    guaranteed -- otherwise production code in the same process will keep
    seeing the replay clock.
    """

    def __init__(
        self,
        start_ts_ms: int,
        end_ts_ms: int,
        *,
        cache_db: str = DEFAULT_CACHE_DB,
        coins: Optional[List[str]] = None,
        funding_rate_8h: float = 0.0,
        strict_api: bool = True,
        engage_network_sandbox: bool = True,
        sandbox_allow_loopback: bool = True,
        build_container: bool = False,
        run_id: Optional[str] = None,
        keep_replay_db: bool = True,
        frozen_xgb_model: Optional[str] = None,
        fills_db: Optional[str] = None,
    ):
        if end_ts_ms <= start_ts_ms:
            raise ValueError(f"end_ts_ms ({end_ts_ms}) must be > start_ts_ms ({start_ts_ms})")
        self.start_ts_ms = int(start_ts_ms)
        self.end_ts_ms = int(end_ts_ms)
        self.cache_db = cache_db
        self.coins = coins
        self.funding_rate_8h = float(funding_rate_8h)
        self.strict_api = bool(strict_api)
        self._engage_network = bool(engage_network_sandbox)
        self._sandbox_loopback = bool(sandbox_allow_loopback)
        self._build_container = bool(build_container)
        self._run_id = run_id
        self._keep_replay_db = bool(keep_replay_db)
        self._frozen_xgb_model = frozen_xgb_model
        self._fills_db = fills_db

        # Lazy-built in __enter__
        self.clock: Optional[ReplayClock] = None
        self.oracle: Optional[CandleOracle] = None
        self.api: Optional[ReplayAPIManager] = None
        self.stubs: Dict[str, Any] = {}
        self.replay_db: Optional[ReplayDB] = None
        self.container: Any = None  # SubsystemContainer if build_container=True
        self.health: Any = None

        self._prev_clock_backend = None
        self._engaged = False

    # ---- lifecycle ----------------------------------------------------

    def __enter__(self) -> "ReplayHarness":
        self._build()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.teardown()

    def _build(self) -> None:
        # 1. clock
        self.clock = ReplayClock(start_ts_ms=self.start_ts_ms, label="harness")

        # 2. install clock as the process-wide time backend
        self._prev_clock_backend = clock_provider.install(self.clock)

        # 3. candle oracle
        self.oracle = CandleOracle(self.cache_db, self.clock)

        # 4. api shim + singleton swap.  When a fills DB is supplied, build a
        # trader-position oracle so clearinghouseState serves the source
        # traders' reconstructed historical positions -> copy trades fire.
        position_oracle = None
        if self._fills_db:
            from src.backtest.replay.position_oracle import TraderPositionOracle
            position_oracle = TraderPositionOracle.from_db(self._fills_db)
            logger.info(
                "Replay position oracle: loaded %d trader(s) from %s",
                len(position_oracle.addresses()), self._fills_db,
            )
        self.api = ReplayAPIManager(
            self.oracle, self.clock,
            known_coins=self.coins,
            funding_rate_8h=self.funding_rate_8h,
            strict=self.strict_api,
            position_oracle=position_oracle,
        )
        install_replay_manager(self.api)

        # 5. stubs (caller can read them via self.stubs)
        self.stubs = all_stubs()

        # 6. network sandbox -- engage BEFORE building subsystems so any
        # subsystem __init__ that tries to phone home raises loudly.
        if self._engage_network:
            engage_sandbox(allow_loopback=self._sandbox_loopback)

        # 7. Replay DB + subsystem container (opt-in).
        if self._build_container:
            self.replay_db = ReplayDB(run_id=self._run_id, keep_on_exit=self._keep_replay_db)
            self.replay_db.install()
            self.replay_db.init_schema()
            self.replay_db.reset_runtime_state()

            from src.backtest.replay.subsystem_assembly import build_replay_container
            self.container, container_stubs = build_replay_container(
                enable_xgboost=bool(self._frozen_xgb_model),
                xgboost_model_path=self._frozen_xgb_model,
            )
            # The stubs created during assembly are the ones actually on the
            # container; use them for telemetry instead of the bag we created
            # at step 5.
            self.stubs = container_stubs
            # Make the clock available on the container for code that
            # otherwise can't find it (subsystems that don't import
            # clock_provider directly can reach .clock here).
            if hasattr(self.container, "__dict__"):
                self.container.clock = self.clock

        self._engaged = True
        logger.info(
            "ReplayHarness engaged: window=[%d, %d], cache=%s, coins=%s, "
            "sandbox=%s, container=%s",
            self.start_ts_ms, self.end_ts_ms, self.cache_db,
            self.api._known_coins, self._engage_network, self._build_container,
        )

    def teardown(self) -> None:
        if not self._engaged:
            return
        try:
            if self._engage_network:
                disengage_sandbox()
        finally:
            try:
                uninstall_replay_manager()
            finally:
                try:
                    if self.replay_db is not None:
                        self.replay_db.uninstall()
                finally:
                    clock_provider.restore(self._prev_clock_backend)
                    self._engaged = False
                    self.container = None
                    logger.info("ReplayHarness torn down")

    # ---- ticking ------------------------------------------------------

    def iter_ticks(self, step_ms: int) -> Iterator[ReplayTick]:
        """Yield ticks across the configured window. First tick is at
        `start_ts_ms`; subsequent ticks advance by `step_ms` until `end_ts_ms`
        is reached (exclusive)."""
        if step_ms <= 0:
            raise ValueError(f"step_ms must be > 0, got {step_ms}")
        if not self._engaged:
            raise RuntimeError("ReplayHarness must be used as `with ReplayHarness(...) as h:`")
        index = 0
        t = self.start_ts_ms
        while t < self.end_ts_ms:
            self.clock.set(t)
            yield ReplayTick(index=index, ts_ms=t)
            t += step_ms
            index += 1

    def advance_to(self, ts_ms: int) -> None:
        """Set the clock to an exact timestamp. Useful for jumping to a
        specific event boundary rather than iterating."""
        if not self._engaged:
            raise RuntimeError("Harness not engaged")
        self.clock.set(int(ts_ms))

    # ---- reporting ----------------------------------------------------

    def build_report(self, tick_count: int, step_ms: int) -> ReplayReport:
        api_stats = self.api.get_stats() if self.api else {}
        return ReplayReport(
            start_ts_ms=self.start_ts_ms,
            end_ts_ms=self.end_ts_ms,
            step_ms=step_ms,
            tick_count=tick_count,
            api_calls_by_type=api_stats.get("calls_by_type", {}),
            api_coin_cache_misses=api_stats.get("coin_cache_misses", {}),
            stub_calls={name: stub.get_stub_stats()["calls"]
                        for name, stub in self.stubs.items()},
        )
