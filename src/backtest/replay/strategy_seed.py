"""Seed a frozen strategy + trader pool into the replay DB.

The production bot's scorer + decision engine operate on whatever's in the
`strategies` and `traders` tables -- a pool that's continuously refreshed by
the discovery cycle scanning Hyperliquid's leaderboard. For a replay we want
that pool to be FROZEN to a specific historical state so:

  - the decision engine always sees the same options on every tick
  - results are deterministic and reproducible
  - "discovery contamination" (a new trader added mid-replay) can't happen

Two seeding modes:

1. **From a snapshot JSON** -- the most reproducible. The JSON contains
   `traders` and `strategies` rows captured as-of some date. v1 ships with
   a small synthetic snapshot for smoke tests; an operator can dump real
   live state with `--export-snapshot`.

2. **From a live DB snapshot** -- copy the `traders` and `strategies` tables
   from a backup of `data/bot.db` taken on the replay-start date. This is
   the "real" path once you have a backup at the right date.

The seeder writes into whatever DB the database module is currently pointed
at -- in practice that's the ReplayDB the orchestrator just installed.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_FIXTURE_PATH = "fixtures/replay_strategy_pool.json"


@dataclass
class SeedTrader:
    address: str
    first_seen: str = "2025-04-01T00:00:00Z"
    last_updated: str = "2025-04-01T00:00:00Z"
    total_pnl: float = 0.0
    roi_pct: float = 0.0
    account_value: float = 100_000.0
    win_rate: float = 0.55
    trade_count: int = 100
    active: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SeedStrategy:
    name: str
    strategy_type: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    discovered_at: str = "2025-04-01T00:00:00Z"
    last_scored: Optional[str] = None
    current_score: float = 0.6
    total_pnl: float = 0.0
    trade_count: int = 100
    win_rate: float = 0.55
    sharpe_ratio: float = 1.0
    active: int = 1


@dataclass
class SeedSnapshot:
    """A frozen pool of traders + strategies."""
    snapshot_date: str
    description: str
    traders: List[SeedTrader] = field(default_factory=list)
    strategies: List[SeedStrategy] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_date": self.snapshot_date,
            "description": self.description,
            "traders": [asdict(t) for t in self.traders],
            "strategies": [asdict(s) for s in self.strategies],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeedSnapshot":
        return cls(
            snapshot_date=data["snapshot_date"],
            description=data.get("description", ""),
            traders=[SeedTrader(**t) for t in data.get("traders", [])],
            strategies=[SeedStrategy(**s) for s in data.get("strategies", [])],
        )


def build_default_smoke_snapshot() -> SeedSnapshot:
    """Tiny snapshot used by the smoke test and as the default fallback.

    Three traders, ten strategies covering momentum + mean reversion +
    breakout in both directions. Just enough to exercise the scorer +
    decision engine without depending on any live data dump.
    """
    traders = [
        SeedTrader(
            address=f"0x{'a' * 38}{i:02d}",
            total_pnl=100_000.0 * (i + 1),
            roi_pct=0.5 + 0.1 * i,
            win_rate=0.55 + 0.02 * i,
            trade_count=200 + 50 * i,
            account_value=500_000.0,
            metadata={"stub": True, "seed_idx": i},
        )
        for i in range(3)
    ]

    strategy_specs = [
        ("synthetic_momentum_long_BTC", "momentum", {"side": "long", "coin": "BTC", "fast": 10, "slow": 30}),
        ("synthetic_momentum_short_BTC", "momentum", {"side": "short", "coin": "BTC", "fast": 10, "slow": 30}),
        ("synthetic_mean_reversion_BTC", "mean_reversion", {"coin": "BTC", "lookback": 14}),
        ("synthetic_breakout_BTC", "breakout", {"coin": "BTC", "window": 20}),
        ("synthetic_rsi_BTC", "rsi", {"coin": "BTC", "period": 14, "ob": 70, "os": 30}),
        ("synthetic_momentum_long_ETH", "momentum", {"side": "long", "coin": "ETH", "fast": 10, "slow": 30}),
        ("synthetic_momentum_short_ETH", "momentum", {"side": "short", "coin": "ETH", "fast": 10, "slow": 30}),
        ("synthetic_mean_reversion_ETH", "mean_reversion", {"coin": "ETH", "lookback": 14}),
        ("synthetic_breakout_ETH", "breakout", {"coin": "ETH", "window": 20}),
        ("synthetic_rsi_ETH", "rsi", {"coin": "ETH", "period": 14, "ob": 70, "os": 30}),
    ]
    strategies = [
        SeedStrategy(
            name=name,
            strategy_type=stype,
            description=f"Synthetic {stype} strategy for replay smoke test",
            parameters=params,
            current_score=0.6 + 0.02 * i,
            total_pnl=1000.0 * (i + 1),
            sharpe_ratio=0.8 + 0.05 * i,
        )
        for i, (name, stype, params) in enumerate(strategy_specs)
    ]
    return SeedSnapshot(
        snapshot_date="2025-04-01",
        description="Default replay smoke snapshot — 3 traders, 10 strategies (BTC + ETH)",
        traders=traders,
        strategies=strategies,
    )


def load_snapshot(path: str) -> SeedSnapshot:
    """Load a snapshot JSON from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return SeedSnapshot.from_dict(json.load(f))


def save_snapshot(snapshot: SeedSnapshot, path: str) -> str:
    """Persist a snapshot to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(snapshot.to_dict(), f, indent=2, sort_keys=True)
    logger.info("Wrote snapshot to %s", p)
    return str(p)


def build_traders_from_fills(fills_db_path: str) -> List[SeedTrader]:
    """Build copyable SeedTraders from the wallets in a fills DB's wallet_fills.

    For a COPY backtest the trader pool must BE the wallets the position oracle
    can serve.  Otherwise copy_trader scans the discovery top-N (ranked by PnL),
    which barely overlaps the tracked-copy wallets, so it never queries the
    oracle's wallets and no copy signals fire.  Each wallet is emitted as an
    active, evidence-bar-clearing trader ranked by fill activity (so the most
    active copy sources sort into copy_trader's top-N scan).
    """
    out: List[SeedTrader] = []
    p = Path(fills_db_path)
    if not p.exists():
        return out
    try:
        with sqlite3.connect(f"file:{p}?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "SELECT wallet_address, COUNT(*) AS n FROM wallet_fills "
                "WHERE wallet_address IS NOT NULL AND wallet_address <> '' "
                "GROUP BY wallet_address"
            ).fetchall()
    except Exception:
        return out
    for addr, n in rows:
        a = str(addr or "").strip()
        if not a:
            continue
        cnt = int(n or 0)
        out.append(SeedTrader(
            address=a,
            total_pnl=float(max(1, cnt)),     # positive => clears the evidence bar; ranks by activity
            roi_pct=0.10,
            win_rate=0.55,
            trade_count=max(10, cnt),          # >= TRADER_MIN_CLOSED_TRADES (default 10)
            active=1,
            metadata={"source": "replay_fills"},
        ))
    return out


def export_from_live_db(live_db_path: str, snapshot_date: str, description: str = "") -> SeedSnapshot:
    """Dump the live bot's current strategy + trader pool into a snapshot.

    Use this on a known-good live state to capture "what the bot knew on day X".
    """
    p = Path(live_db_path)
    if not p.exists():
        raise FileNotFoundError(f"Live DB not found: {live_db_path}")

    with sqlite3.connect(f"file:{p}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        traders = [_row_to_seed_trader(r) for r in conn.execute("SELECT * FROM traders WHERE active = 1")]
        strategies = [_row_to_seed_strategy(r) for r in conn.execute("SELECT * FROM strategies WHERE active = 1")]

    snap = SeedSnapshot(
        snapshot_date=snapshot_date,
        description=description or f"Exported from {live_db_path}",
        traders=traders,
        strategies=strategies,
    )
    logger.info("Exported %d traders + %d strategies from %s", len(traders), len(strategies), live_db_path)
    return snap


def _row_to_seed_trader(row: sqlite3.Row) -> SeedTrader:
    try:
        metadata = json.loads(row["metadata"] or "{}")
    except (KeyError, json.JSONDecodeError):
        metadata = {}
    return SeedTrader(
        address=row["address"],
        first_seen=row["first_seen"],
        last_updated=row["last_updated"],
        total_pnl=float(row["total_pnl"] or 0),
        roi_pct=float(row["roi_pct"] or 0),
        account_value=float(row["account_value"] or 0),
        win_rate=float(row["win_rate"] or 0),
        trade_count=int(row["trade_count"] or 0),
        active=int(row["active"] or 0),
        metadata=metadata,
    )


def _row_to_seed_strategy(row: sqlite3.Row) -> SeedStrategy:
    try:
        params = json.loads(row["parameters"] or "{}")
    except json.JSONDecodeError:
        params = {}
    return SeedStrategy(
        name=row["name"],
        strategy_type=row["strategy_type"],
        description=row["description"] or "",
        parameters=params,
        discovered_at=row["discovered_at"],
        last_scored=row["last_scored"],
        current_score=float(row["current_score"] or 0),
        total_pnl=float(row["total_pnl"] or 0),
        trade_count=int(row["trade_count"] or 0),
        win_rate=float(row["win_rate"] or 0),
        sharpe_ratio=float(row["sharpe_ratio"] or 0),
        active=int(row["active"] or 0),
    )


def seed_into(db_path: str, snapshot: SeedSnapshot, *, replace: bool = True) -> Dict[str, int]:
    """Insert a snapshot's traders + strategies into the given SQLite DB.

    If `replace=True`, DELETE existing rows first so the pool is exactly the
    snapshot. Otherwise insert with IGNORE so existing keys are kept.

    Returns a dict with the number of rows inserted.
    """
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Target DB does not exist: {db_path}. "
            "Run ReplayDB.init_schema() first to create tables."
        )

    inserted = {"traders": 0, "strategies": 0}
    with sqlite3.connect(str(p)) as conn:
        if replace:
            conn.execute("DELETE FROM traders")
            conn.execute("DELETE FROM strategies")

        for t in snapshot.traders:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO traders
                       (address, first_seen, last_updated, total_pnl, roi_pct,
                        account_value, win_rate, trade_count, active, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (t.address, t.first_seen, t.last_updated, t.total_pnl, t.roi_pct,
                     t.account_value, t.win_rate, t.trade_count, t.active,
                     json.dumps(t.metadata)),
                )
                if conn.total_changes:
                    inserted["traders"] += 1
            except sqlite3.IntegrityError as e:
                logger.debug("Skipping trader %s: %s", t.address, e)

        for s in snapshot.strategies:
            try:
                conn.execute(
                    """INSERT INTO strategies
                       (name, description, strategy_type, parameters, discovered_at,
                        last_scored, current_score, total_pnl, trade_count, win_rate,
                        sharpe_ratio, active)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (s.name, s.description, s.strategy_type, json.dumps(s.parameters),
                     s.discovered_at, s.last_scored, s.current_score, s.total_pnl,
                     s.trade_count, s.win_rate, s.sharpe_ratio, s.active),
                )
                inserted["strategies"] += 1
            except sqlite3.IntegrityError as e:
                logger.debug("Skipping strategy %s: %s", s.name, e)

        conn.commit()

    logger.info("Seeded %d traders, %d strategies into %s",
                inserted["traders"], inserted["strategies"], db_path)
    return inserted
