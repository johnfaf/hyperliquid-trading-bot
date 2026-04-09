"""
Vectorized Backtester
=====================
Phase 1 of the institutional roadmap: event-driven backtester that replays
golden wallet fills through the copy-trading logic with configurable parameters.

Architecture:
  1. Load historical fills from the `wallet_fills` table (already penalised)
  2. Generate copy-trade signals from the fill stream (lag-adjusted)
  3. Run signals through a simulated DecisionFirewall + position manager
  4. Compute strategy-level metrics: Sharpe, max DD, profit factor, win rate
  5. Store each run in the `experiments` table for comparison

Usage:
    python -m src.backtester --wallet 0xabc... --initial-balance 10000
    python -m src.backtester --all-golden --sweep         # parameter sweep
    python -m src.backtester --experiment exp_001 --show   # view past run

Design decisions:
  - Vectorised where possible (numpy for equity curves, drawdown, returns)
  - No live API calls — everything from DB
  - Deterministic: same params + same data = same result (seeded RNG)
  - Slippage model matches golden_wallet.py (configurable bps)
"""
import logging
import json
import time
import statistics
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Dict, Optional
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger("backtester")


# ─── Configuration ──────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """All tunable parameters for a single backtest run."""
    # Capital
    initial_balance: float = 10_000.0

    # Position sizing
    max_position_pct: float = 0.08       # 8% of balance per trade
    max_positions: int = 5
    max_per_coin: int = 2
    max_leverage: float = 5.0
    max_aggregate_exposure_pct: float = 0.30  # 30% of balance

    # Risk management
    stop_loss_pct: float = 0.05          # 5%
    take_profit_pct: float = 0.10        # 10%
    trailing_stop: bool = True
    trailing_pct: float = 0.025          # 2.5%

    # Execution model
    copy_delay_ms: int = 2000            # 2s delay to copy (realistic)
    slippage_bps: float = 4.5            # 0.045% per leg
    min_confidence: float = 0.30         # firewall minimum
    cooldown_ms: int = 300_000           # 5 min cooldown per coin

    # Funding rate simulation
    funding_rate_8h: float = 0.0001      # 0.01% per 8h (HL average)
    funding_enabled: bool = True         # charge funding on open positions

    # Partial fill simulation
    partial_fill_prob: float = 0.10      # 10% chance of partial fill
    partial_fill_min_pct: float = 0.40   # minimum 40% of order filled

    # Liquidation simulation
    liquidation_enabled: bool = True
    maintenance_margin_pct: float = 0.03 # 3% maintenance margin (HL standard)

    # Filters
    min_trade_size_usd: float = 100.0    # ignore dust
    max_hold_hours: float = 72.0         # force-close after 72h

    # Seed for reproducibility
    seed: int = 42

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "BacktestConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─── Data structures ────────────────────────────────────────────────

@dataclass
class BacktestFill:
    """A single fill event in the backtest timeline."""
    wallet_address: str
    coin: str
    side: str               # "buy" / "sell"
    price: float            # penalised price from DB
    original_price: float
    size: float
    time_ms: int
    closed_pnl: float       # penalised PnL
    direction: str           # "Open Long", "Close Long", etc.
    is_liquidation: bool


@dataclass
class SimPosition:
    """An open simulated position."""
    coin: str
    side: str                # "long" / "short"
    entry_price: float
    size: float
    leverage: float
    entry_time_ms: int
    stop_loss: float
    take_profit: float
    trailing_high: float     # for trailing stop
    trailing_low: float      # for trailing stop (shorts)
    source_wallet: str
    notional: float = 0.0

    def __post_init__(self):
        self.notional = self.entry_price * self.size * self.leverage


@dataclass
class SimTrade:
    """A completed (closed) simulated trade."""
    coin: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    leverage: float
    entry_time_ms: int
    exit_time_ms: int
    pnl: float
    pnl_pct: float
    exit_reason: str         # "signal", "stop_loss", "take_profit", "trailing_stop", "time_limit", "force_close", "liquidation"
    source_wallet: str
    hold_time_hours: float


@dataclass
class BacktestResult:
    """Complete results from a single backtest run."""
    experiment_id: str
    config: BacktestConfig
    wallets_used: List[str]

    # Metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_hold_hours: float = 0.0
    max_consecutive_losses: int = 0
    calmar_ratio: float = 0.0
    expectancy: float = 0.0

    # Curves
    equity_curve: List[float] = field(default_factory=list)
    equity_timestamps: List[int] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)

    # Breakdown
    trades: List[SimTrade] = field(default_factory=list)
    coin_breakdown: Dict[str, Dict] = field(default_factory=dict)

    # Meta
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0

    def summary_dict(self) -> Dict:
        """Compact summary for display / DB storage."""
        return {
            "experiment_id": self.experiment_id,
            "wallets": len(self.wallets_used),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 2),
            "total_pnl": round(self.total_pnl, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "profit_factor": round(self.profit_factor, 3),
            "calmar_ratio": round(self.calmar_ratio, 3),
            "expectancy": round(self.expectancy, 4),
            "avg_hold_hours": round(self.avg_hold_hours, 1),
            "max_consecutive_losses": self.max_consecutive_losses,
            "duration_seconds": round(self.duration_seconds, 1),
        }


# ─── Core Engine ────────────────────────────────────────────────────

class BacktestEngine:
    """
    Event-driven backtester that replays historical fills as copy-trade signals.

    Flow:
      1. Load fills from DB (already penalised by golden_wallet.py)
      2. Sort into a unified timeline
      3. For each fill that represents a position open/close, generate a copy signal
      4. Apply the copy delay + additional slippage
      5. Simulate position management (stops, take-profits, trailing)
      6. Track equity curve and compute metrics
    """

    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self._rng_seed = cfg.seed

        # State
        self._balance = cfg.initial_balance
        self._positions: List[SimPosition] = []
        self._completed_trades: List[SimTrade] = []
        self._equity_curve: List[float] = [cfg.initial_balance]
        self._equity_timestamps: List[int] = [0]
        self._cooldowns: Dict[str, int] = {}  # coin → last trade time_ms

        # Price cache: latest known price per coin
        self._latest_prices: Dict[str, float] = {}

        # Stats
        self._total_signals = 0
        self._signals_taken = 0
        self._signals_rejected = 0
        self._rejection_reasons: Dict[str, int] = defaultdict(int)

    def run(self, fills: List[BacktestFill], experiment_id: str = None) -> BacktestResult:
        """
        Run the backtest on a list of fills.

        Args:
            fills: Sorted list of BacktestFill events (ascending by time_ms)
            experiment_id: Optional ID; auto-generated if not provided

        Returns:
            BacktestResult with all metrics and trade log
        """
        start_time = time.time()

        if experiment_id is None:
            experiment_id = f"bt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{self._rng_seed}"

        if not fills:
            logger.warning("No fills to backtest")
            return BacktestResult(
                experiment_id=experiment_id,
                config=self.cfg,
                wallets_used=[],
            )

        # Reset state
        self._balance = self.cfg.initial_balance
        self._positions = []
        self._completed_trades = []
        self._equity_curve = [self.cfg.initial_balance]
        self._equity_timestamps = [fills[0].time_ms]
        self._cooldowns = {}
        self._latest_prices = {}
        self._total_signals = 0
        self._signals_taken = 0
        self._signals_rejected = 0
        self._rejection_reasons = defaultdict(int)

        wallets_seen = set()

        # Process each fill as a potential copy-trade signal
        for fill in fills:
            wallets_seen.add(fill.wallet_address)
            self._latest_prices[fill.coin] = fill.price

            # Check stops/take-profits against current price
            self._check_exits(fill.time_ms)

            # Force-close positions exceeding time limit
            self._check_time_limits(fill.time_ms)

            # Generate copy signal from this fill
            self._process_fill(fill)

        # Force-close all remaining positions at last known prices
        if fills:
            self._close_all_positions(fills[-1].time_ms, "force_close")

        # Compute metrics
        elapsed = time.time() - start_time
        result = self._compute_result(
            experiment_id=experiment_id,
            wallets_used=list(wallets_seen),
            elapsed=elapsed,
        )

        logger.info(
            f"Backtest {experiment_id}: {result.total_trades} trades, "
            f"PnL=${result.total_pnl:+,.2f}, Sharpe={result.sharpe_ratio:.3f}, "
            f"MaxDD={result.max_drawdown_pct:.1f}%, WR={result.win_rate:.1f}% "
            f"({elapsed:.1f}s)"
        )

        return result

    # ─── Signal Processing ──────────────────────────────────────

    def _process_fill(self, fill: BacktestFill):
        """Decide whether to copy this fill as a new trade or position close."""
        direction = fill.direction.lower() if fill.direction else ""

        # We only care about position opens and closes
        is_open = "open" in direction
        is_close = "close" in direction

        if not is_open and not is_close:
            # Ambiguous fill (could be add to position, partial close, etc.)
            # If it has closed_pnl != 0, treat as close; otherwise as open
            if fill.closed_pnl != 0:
                is_close = True
            else:
                is_open = True

        if is_open:
            self._handle_open_signal(fill)
        elif is_close:
            self._handle_close_signal(fill)

    def _handle_open_signal(self, fill: BacktestFill):
        """Attempt to open a copy position."""
        self._total_signals += 1

        # Determine side
        side = "long" if fill.side == "buy" else "short"

        # Apply copy delay: shift price adversely
        delay_slippage = self.cfg.copy_delay_ms / 1000.0 * 0.001  # rough: 0.1% per second
        if side == "long":
            entry_price = fill.price * (1 + delay_slippage + self.cfg.slippage_bps / 10_000)
        else:
            entry_price = fill.price * (1 - delay_slippage - self.cfg.slippage_bps / 10_000)

        # Validation checks (simulated firewall)
        reject_reason = self._validate_signal(fill.coin, side, entry_price, fill.time_ms)
        if reject_reason:
            self._signals_rejected += 1
            self._rejection_reasons[reject_reason] += 1
            return

        # Position sizing
        position_value = self._balance * self.cfg.max_position_pct
        if position_value < self.cfg.min_trade_size_usd:
            self._signals_rejected += 1
            self._rejection_reasons["insufficient_balance"] += 1
            return

        size = position_value / entry_price if entry_price > 0 else 0
        if size <= 0:
            return

        # Partial fill simulation: randomly reduce fill size
        import random as _rng
        _rng.seed(self._rng_seed + self._total_signals)  # deterministic
        if self.cfg.partial_fill_prob > 0 and _rng.random() < self.cfg.partial_fill_prob:
            fill_pct = _rng.uniform(self.cfg.partial_fill_min_pct, 1.0)
            size *= fill_pct

        # Calculate stop/take-profit levels
        if side == "long":
            stop_loss = entry_price * (1 - self.cfg.stop_loss_pct)
            take_profit = entry_price * (1 + self.cfg.take_profit_pct)
        else:
            stop_loss = entry_price * (1 + self.cfg.stop_loss_pct)
            take_profit = entry_price * (1 - self.cfg.take_profit_pct)

        position = SimPosition(
            coin=fill.coin,
            side=side,
            entry_price=entry_price,
            size=size,
            leverage=min(self.cfg.max_leverage, 1.0),  # conservative default
            entry_time_ms=fill.time_ms,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_high=entry_price,
            trailing_low=entry_price,
            source_wallet=fill.wallet_address,
        )

        self._positions.append(position)
        self._signals_taken += 1
        self._cooldowns[fill.coin] = fill.time_ms

        # Update equity snapshot
        self._snapshot_equity(fill.time_ms)

    def _handle_close_signal(self, fill: BacktestFill):
        """Close matching positions when the source wallet closes."""
        coin = fill.coin
        matching = [p for p in self._positions
                    if p.coin == coin and p.source_wallet == fill.wallet_address]

        for pos in matching:
            # Apply exit slippage
            if pos.side == "long":
                exit_price = fill.price * (1 - self.cfg.slippage_bps / 10_000)
            else:
                exit_price = fill.price * (1 + self.cfg.slippage_bps / 10_000)

            self._close_position(pos, exit_price, fill.time_ms, "signal")

    def _validate_signal(self, coin: str, side: str, price: float,
                          time_ms: int) -> Optional[str]:
        """
        Simulated decision firewall. Returns None if valid, or rejection reason string.
        """
        # Max positions
        if len(self._positions) >= self.cfg.max_positions:
            return "max_positions"

        # Per-coin limit
        coin_positions = [p for p in self._positions if p.coin == coin]
        if len(coin_positions) >= self.cfg.max_per_coin:
            return "max_per_coin"

        # Conflict: opposite side on same coin
        for p in coin_positions:
            if p.side != side:
                return "conflict"

        # Cooldown
        last_trade = self._cooldowns.get(coin, 0)
        if time_ms - last_trade < self.cfg.cooldown_ms:
            return "cooldown"

        # Aggregate exposure (use 1x for sizing — matches position creation)
        total_exposure = sum(p.notional for p in self._positions)
        new_notional = self._balance * self.cfg.max_position_pct
        if (total_exposure + new_notional) / self._balance > self.cfg.max_aggregate_exposure_pct:
            return "aggregate_exposure"

        # Balance check
        if self._balance < self.cfg.min_trade_size_usd:
            return "insufficient_balance"

        return None

    # ─── Position Management ────────────────────────────────────

    def _charge_funding(self, current_time_ms: int):
        """Charge funding rate on all open positions (every 8h in simulation)."""
        if not self.cfg.funding_enabled:
            return
        funding_interval_ms = 8 * 3600 * 1000  # 8 hours
        for pos in self._positions:
            elapsed = current_time_ms - pos.entry_time_ms
            # Number of 8h periods since entry
            periods = elapsed // funding_interval_ms
            if periods < 1:
                continue
            # Charge once per snapshot — we approximate by checking if this is
            # a funding boundary (within one fill of a period boundary)
            if elapsed % funding_interval_ms < 60_000:  # within 1 minute of boundary
                funding_cost = pos.notional * self.cfg.funding_rate_8h
                # Longs pay funding when positive, shorts receive (simplified)
                if pos.side == "long":
                    self._balance -= funding_cost
                else:
                    self._balance += funding_cost

    def _check_liquidations(self, current_time_ms: int):
        """Check if any position has been liquidated (margin call)."""
        if not self.cfg.liquidation_enabled:
            return
        to_liquidate = []
        for pos in self._positions:
            price = self._latest_prices.get(pos.coin)
            if price is None:
                continue
            # Unrealized PnL
            if pos.side == "long":
                unrealized = (price - pos.entry_price) * pos.size * pos.leverage
            else:
                unrealized = (pos.entry_price - price) * pos.size * pos.leverage
            # Margin = notional / leverage
            margin = pos.notional / pos.leverage if pos.leverage > 0 else pos.notional
            # Liquidated when unrealized loss exceeds (1 - maintenance_margin) * margin
            if unrealized < -(margin * (1 - self.cfg.maintenance_margin_pct)):
                to_liquidate.append((pos, price))

        for pos, price in to_liquidate:
            self._close_position(pos, price, current_time_ms, "liquidation")

    def _check_exits(self, current_time_ms: int):
        """Check all open positions against their stop/take-profit levels."""
        # Charge funding before exit checks
        self._charge_funding(current_time_ms)
        # Check liquidations
        self._check_liquidations(current_time_ms)

        to_close = []

        for pos in self._positions:
            price = self._latest_prices.get(pos.coin)
            if price is None:
                continue

            # Update trailing stop
            if self.cfg.trailing_stop:
                if pos.side == "long":
                    if price > pos.trailing_high:
                        pos.trailing_high = price
                        pos.stop_loss = max(
                            pos.stop_loss,
                            price * (1 - self.cfg.trailing_pct)
                        )
                else:
                    if price < pos.trailing_low:
                        pos.trailing_low = price
                        pos.stop_loss = min(
                            pos.stop_loss,
                            price * (1 + self.cfg.trailing_pct)
                        )

            # Check stop loss
            if pos.side == "long" and price <= pos.stop_loss:
                to_close.append((pos, pos.stop_loss, "stop_loss"))
            elif pos.side == "short" and price >= pos.stop_loss:
                to_close.append((pos, pos.stop_loss, "stop_loss"))

            # Check take profit
            if pos.side == "long" and price >= pos.take_profit:
                to_close.append((pos, pos.take_profit, "take_profit"))
            elif pos.side == "short" and price <= pos.take_profit:
                to_close.append((pos, pos.take_profit, "take_profit"))

        for pos, exit_price, reason in to_close:
            self._close_position(pos, exit_price, current_time_ms, reason)

    def _check_time_limits(self, current_time_ms: int):
        """Force-close positions exceeding max hold time."""
        max_hold_ms = self.cfg.max_hold_hours * 3600 * 1000
        to_close = []

        for pos in self._positions:
            if current_time_ms - pos.entry_time_ms > max_hold_ms:
                price = self._latest_prices.get(pos.coin, pos.entry_price)
                to_close.append((pos, price))

        for pos, price in to_close:
            self._close_position(pos, price, current_time_ms, "time_limit")

    def _close_position(self, pos: SimPosition, exit_price: float,
                         time_ms: int, reason: str):
        """Close a position and record the trade."""
        if pos not in self._positions:
            return

        self._positions.remove(pos)

        # Calculate PnL
        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * pos.size * pos.leverage
        else:
            pnl = (pos.entry_price - exit_price) * pos.size * pos.leverage

        pnl_pct = pnl / (pos.entry_price * pos.size) if pos.entry_price * pos.size > 0 else 0
        hold_hours = (time_ms - pos.entry_time_ms) / (3600 * 1000)

        trade = SimTrade(
            coin=pos.coin,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            leverage=pos.leverage,
            entry_time_ms=pos.entry_time_ms,
            exit_time_ms=time_ms,
            pnl=round(pnl, 4),
            pnl_pct=round(pnl_pct, 6),
            exit_reason=reason,
            source_wallet=pos.source_wallet,
            hold_time_hours=round(hold_hours, 2),
        )

        self._completed_trades.append(trade)
        self._balance += pnl
        self._snapshot_equity(time_ms)

    def _close_all_positions(self, time_ms: int, reason: str):
        """Close all open positions at last known prices."""
        for pos in list(self._positions):
            price = self._latest_prices.get(pos.coin, pos.entry_price)
            self._close_position(pos, price, time_ms, reason)

    def _snapshot_equity(self, time_ms: int):
        """Record current equity (balance + unrealized PnL)."""
        unrealized = 0.0
        for pos in self._positions:
            price = self._latest_prices.get(pos.coin, pos.entry_price)
            if pos.side == "long":
                unrealized += (price - pos.entry_price) * pos.size * pos.leverage
            else:
                unrealized += (pos.entry_price - price) * pos.size * pos.leverage

        equity = self._balance + unrealized
        self._equity_curve.append(equity)
        self._equity_timestamps.append(time_ms)

    # ─── Metrics Computation ────────────────────────────────────

    def _compute_result(self, experiment_id: str, wallets_used: List[str],
                         elapsed: float) -> BacktestResult:
        """Compute all strategy metrics from completed trades."""
        trades = self._completed_trades
        n = len(trades)

        if n == 0:
            return BacktestResult(
                experiment_id=experiment_id,
                config=self.cfg,
                wallets_used=wallets_used,
                equity_curve=self._equity_curve,
                equity_timestamps=self._equity_timestamps,
                started_at=datetime.now(timezone.utc).isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_seconds=elapsed,
            )

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        n_wins = len(wins)
        n_losses = len(losses)

        total_pnl = sum(t.pnl for t in trades)
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0

        win_rate = (n_wins / n * 100) if n > 0 else 0
        avg_win = (gross_profit / n_wins) if n_wins > 0 else 0
        avg_loss = (gross_loss / n_losses) if n_losses > 0 else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        expectancy = ((win_rate / 100) * avg_win - (1 - win_rate / 100) * avg_loss) / self.cfg.initial_balance

        avg_hold = statistics.mean([t.hold_time_hours for t in trades]) if trades else 0

        # Max consecutive losses
        max_consec = 0
        current_consec = 0
        for t in trades:
            if t.pnl <= 0:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0

        # Max drawdown from equity curve
        max_dd = self._compute_max_drawdown(self._equity_curve)

        # Daily returns for Sharpe/Sortino
        daily_returns = self._compute_daily_returns(
            self._equity_curve, self._equity_timestamps
        )

        sharpe = self._compute_sharpe(daily_returns)
        sortino = self._compute_sortino(daily_returns)

        # Calmar = annualized return / max drawdown
        total_days = 0
        if len(self._equity_timestamps) >= 2:
            total_days = (self._equity_timestamps[-1] - self._equity_timestamps[0]) / (86400 * 1000)
        if total_days > 0 and max_dd > 0:
            annual_return = (total_pnl / self.cfg.initial_balance) * (365 / total_days)
            calmar = annual_return / (max_dd / 100)
        else:
            calmar = 0.0

        # Coin breakdown
        coin_breakdown = defaultdict(lambda: {"trades": 0, "pnl": 0, "wins": 0})
        for t in trades:
            coin_breakdown[t.coin]["trades"] += 1
            coin_breakdown[t.coin]["pnl"] += t.pnl
            if t.pnl > 0:
                coin_breakdown[t.coin]["wins"] += 1

        for coin_data in coin_breakdown.values():
            coin_data["win_rate"] = round(
                coin_data["wins"] / coin_data["trades"] * 100, 1
            ) if coin_data["trades"] > 0 else 0

        return BacktestResult(
            experiment_id=experiment_id,
            config=self.cfg,
            wallets_used=wallets_used,
            total_trades=n,
            winning_trades=n_wins,
            losing_trades=n_losses,
            win_rate=round(win_rate, 2),
            total_pnl=round(total_pnl, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            profit_factor=round(profit_factor, 3),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            avg_hold_hours=round(avg_hold, 1),
            max_consecutive_losses=max_consec,
            calmar_ratio=round(calmar, 3),
            expectancy=round(expectancy, 6),
            equity_curve=self._equity_curve,
            equity_timestamps=self._equity_timestamps,
            daily_returns=daily_returns,
            trades=trades,
            coin_breakdown=dict(coin_breakdown),
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_seconds=elapsed,
        )

    @staticmethod
    def _compute_max_drawdown(curve: List[float]) -> float:
        """Max drawdown as a percentage."""
        if len(curve) < 2:
            return 0.0
        peak = curve[0]
        max_dd = 0.0
        for val in curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd

    @staticmethod
    def _compute_daily_returns(equity_curve: List[float],
                                 timestamps: List[int]) -> List[float]:
        """Resample equity to daily and compute returns."""
        if len(equity_curve) < 2:
            return []

        daily_equity = {}
        for eq, ts in zip(equity_curve, timestamps):
            day = ts // (86400 * 1000)
            daily_equity[day] = eq

        sorted_days = sorted(daily_equity.keys())
        if len(sorted_days) < 2:
            return []

        returns = []
        for i in range(1, len(sorted_days)):
            prev = daily_equity[sorted_days[i - 1]]
            curr = daily_equity[sorted_days[i]]
            if prev > 0:
                returns.append((curr - prev) / prev)
        return returns

    @staticmethod
    def _compute_sharpe(daily_returns: List[float], periods: float = 365.0) -> float:
        """Annualized Sharpe ratio."""
        if len(daily_returns) < 5:
            return 0.0
        mean_r = statistics.mean(daily_returns)
        std_r = statistics.stdev(daily_returns)
        if std_r < 1e-10:
            return 0.0
        return (mean_r / std_r) * (periods ** 0.5)

    @staticmethod
    def _compute_sortino(daily_returns: List[float], periods: float = 365.0) -> float:
        """Annualized Sortino ratio (downside deviation only)."""
        if len(daily_returns) < 5:
            return 0.0
        mean_r = statistics.mean(daily_returns)
        downside = [r for r in daily_returns if r < 0]
        if len(downside) < 2:
            return 0.0
        downside_std = statistics.stdev(downside)
        if downside_std < 1e-10:
            return 0.0
        return (mean_r / downside_std) * (periods ** 0.5)


# ─── Data Loading ───────────────────────────────────────────────────

def load_fills_from_db(wallet_address: str = None,
                        golden_only: bool = False) -> List[BacktestFill]:
    """
    Load penalised fills from the wallet_fills table.

    Args:
        wallet_address: Specific wallet to load. If None, loads all.
        golden_only: If True, only load fills from golden wallets.
    """
    import sqlite3

    db_path = config.DB_PATH
    if not os.path.exists(db_path):
        logger.error(f"Database not found at {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if wallet_address:
            rows = conn.execute(
                "SELECT * FROM wallet_fills WHERE wallet_address = ? ORDER BY time_ms",
                (wallet_address,)
            ).fetchall()
        elif golden_only:
            rows = conn.execute("""
                SELECT wf.* FROM wallet_fills wf
                JOIN golden_wallets gw ON wf.wallet_address = gw.address
                WHERE gw.is_golden = 1
                ORDER BY wf.time_ms
            """).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM wallet_fills ORDER BY time_ms"
            ).fetchall()

        fills = []
        for r in rows:
            fills.append(BacktestFill(
                wallet_address=r["wallet_address"],
                coin=r["coin"],
                side=r["side"],
                price=r["penalised_price"],
                original_price=r["original_price"],
                size=r["size"],
                time_ms=r["time_ms"],
                closed_pnl=r["penalised_pnl"],
                direction=r["direction"] or "",
                is_liquidation=bool(r["is_liquidation"]),
            ))

        logger.info(f"Loaded {len(fills)} fills"
                     f"{f' for {wallet_address[:10]}' if wallet_address else ''}"
                     f"{' (golden only)' if golden_only else ''}")
        return fills

    finally:
        conn.close()


def list_golden_wallets() -> List[Dict]:
    """List all golden wallets with summary stats."""
    import sqlite3

    db_path = config.DB_PATH
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT address, penalised_pnl, sharpe_ratio, win_rate,
                   max_drawdown_pct, total_fills, trades_per_day
            FROM golden_wallets
            WHERE is_golden = 1
            ORDER BY penalised_pnl DESC
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ─── Experiment Persistence ─────────────────────────────────────────

def init_experiments_table():
    """Create the experiments table if it doesn't exist."""
    import sqlite3

    conn = sqlite3.connect(config.DB_PATH)
    try:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            config TEXT NOT NULL,
            wallets TEXT NOT NULL,
            total_trades INTEGER DEFAULT 0,
            win_rate REAL DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            max_drawdown_pct REAL DEFAULT 0,
            sharpe_ratio REAL DEFAULT 0,
            sortino_ratio REAL DEFAULT 0,
            profit_factor REAL DEFAULT 0,
            calmar_ratio REAL DEFAULT 0,
            expectancy REAL DEFAULT 0,
            avg_hold_hours REAL DEFAULT 0,
            max_consecutive_losses INTEGER DEFAULT 0,
            equity_curve TEXT DEFAULT '[]',
            equity_timestamps TEXT DEFAULT '[]',
            coin_breakdown TEXT DEFAULT '{}',
            duration_seconds REAL DEFAULT 0,
            notes TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_exp_created ON experiments(created_at);
        CREATE INDEX IF NOT EXISTS idx_exp_sharpe ON experiments(sharpe_ratio);
        CREATE INDEX IF NOT EXISTS idx_exp_pnl ON experiments(total_pnl);
        """)
        conn.commit()
    finally:
        conn.close()


def save_experiment(result: BacktestResult, notes: str = ""):
    """Save a backtest result to the experiments table."""
    import sqlite3

    init_experiments_table()

    conn = sqlite3.connect(config.DB_PATH)
    try:
        conn.execute("""
            INSERT OR REPLACE INTO experiments
            (id, created_at, config, wallets, total_trades, win_rate, total_pnl,
             max_drawdown_pct, sharpe_ratio, sortino_ratio, profit_factor,
             calmar_ratio, expectancy, avg_hold_hours, max_consecutive_losses,
             equity_curve, equity_timestamps, coin_breakdown, duration_seconds, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.experiment_id,
            result.started_at,
            json.dumps(result.config.to_dict()),
            json.dumps(result.wallets_used),
            result.total_trades,
            result.win_rate,
            result.total_pnl,
            result.max_drawdown_pct,
            result.sharpe_ratio,
            result.sortino_ratio,
            result.profit_factor,
            result.calmar_ratio,
            result.expectancy,
            result.avg_hold_hours,
            result.max_consecutive_losses,
            json.dumps(result.equity_curve[-1000:]),  # cap storage
            json.dumps(result.equity_timestamps[-1000:]),
            json.dumps(result.coin_breakdown),
            result.duration_seconds,
            notes,
        ))
        conn.commit()
        logger.info(f"Saved experiment {result.experiment_id}")
    finally:
        conn.close()


def load_experiment(experiment_id: str) -> Optional[Dict]:
    """Load a past experiment result."""
    import sqlite3

    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_experiments(limit: int = 20) -> List[Dict]:
    """List recent experiments, sorted by creation time."""
    import sqlite3

    db_path = config.DB_PATH
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT id, created_at, total_trades, win_rate, total_pnl,
                   max_drawdown_pct, sharpe_ratio, sortino_ratio,
                   profit_factor, calmar_ratio, duration_seconds, notes
            FROM experiments
            ORDER BY created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ─── Parameter Sweep ────────────────────────────────────────────────

def parameter_sweep(fills: List[BacktestFill],
                      sweep_params: Dict[str, List] = None) -> List[BacktestResult]:
    """
    Run backtests across multiple parameter combinations.

    Args:
        fills: The fill data to backtest against (loaded once, reused)
        sweep_params: Dict of param_name → [values_to_try]
                      e.g. {"stop_loss_pct": [0.03, 0.05, 0.07],
                            "max_position_pct": [0.05, 0.08, 0.10]}

    Returns:
        List of BacktestResult sorted by Sharpe ratio descending
    """
    if sweep_params is None:
        sweep_params = {
            "stop_loss_pct": [0.03, 0.05, 0.07, 0.10],
            "take_profit_pct": [0.05, 0.10, 0.15, 0.20],
            "max_position_pct": [0.05, 0.08, 0.10, 0.15],
        }

    # Generate all combinations
    import itertools
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    combos = list(itertools.product(*param_values))

    logger.info(f"Parameter sweep: {len(combos)} combinations across {param_names}")

    results = []
    for i, combo in enumerate(combos):
        cfg_dict = {}
        for name, val in zip(param_names, combo):
            cfg_dict[name] = val

        cfg = BacktestConfig(**cfg_dict)
        engine = BacktestEngine(cfg)

        experiment_id = f"sweep_{i:04d}_{'_'.join(f'{n}={v}' for n, v in zip(param_names, combo))}"
        result = engine.run(fills, experiment_id=experiment_id)
        results.append(result)

        if (i + 1) % 10 == 0:
            logger.info(f"  Sweep progress: {i+1}/{len(combos)}")

    # Sort by Sharpe
    results.sort(key=lambda r: r.sharpe_ratio, reverse=True)

    logger.info(f"Sweep complete. Best: Sharpe={results[0].sharpe_ratio:.3f}, "
                f"PnL=${results[0].total_pnl:+,.2f} "
                f"({results[0].experiment_id})")

    return results


# ─── CLI Entry Point ────────────────────────────────────────────────

def main():
    """Command-line interface for the backtester."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hyperliquid Copy-Trade Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest a specific golden wallet
  python -m src.backtester --wallet 0xabc123...

  # Backtest all golden wallets
  python -m src.backtester --all-golden

  # Parameter sweep across stop-loss values
  python -m src.backtester --all-golden --sweep

  # Custom config
  python -m src.backtester --all-golden --stop-loss 0.03 --take-profit 0.15

  # List past experiments
  python -m src.backtester --list-experiments

  # Show a specific experiment
  python -m src.backtester --show-experiment bt_20260324_120000_42
        """
    )

    # Data selection
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--wallet", type=str, help="Wallet address to backtest")
    data_group.add_argument("--all-golden", action="store_true",
                            help="Backtest all golden wallets")
    data_group.add_argument("--all", action="store_true",
                            help="Backtest all wallets with stored fills")

    # Config overrides
    parser.add_argument("--initial-balance", type=float, default=10000,
                        help="Starting balance (default: 10000)")
    parser.add_argument("--stop-loss", type=float, default=0.05,
                        help="Stop loss %% (default: 0.05)")
    parser.add_argument("--take-profit", type=float, default=0.10,
                        help="Take profit %% (default: 0.10)")
    parser.add_argument("--max-position", type=float, default=0.08,
                        help="Max position %% of balance (default: 0.08)")
    parser.add_argument("--max-positions", type=int, default=5,
                        help="Max concurrent positions (default: 5)")
    parser.add_argument("--slippage", type=float, default=4.5,
                        help="Slippage in bps (default: 4.5)")
    parser.add_argument("--copy-delay", type=int, default=2000,
                        help="Copy delay in ms (default: 2000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # Sweep mode
    parser.add_argument("--sweep", action="store_true",
                        help="Run parameter sweep")

    # Experiment management
    parser.add_argument("--list-experiments", action="store_true",
                        help="List past experiments")
    parser.add_argument("--show-experiment", type=str,
                        help="Show details of a specific experiment")
    parser.add_argument("--notes", type=str, default="",
                        help="Notes to attach to this experiment")

    # Output
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save results to experiments table (default: True)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Handle experiment listing
    if args.list_experiments:
        init_experiments_table()
        experiments = list_experiments(limit=20)
        if not experiments:
            print("No experiments found.")
            return

        print(f"\n{'ID':<45} {'Trades':>7} {'WR%':>6} {'PnL':>12} {'Sharpe':>8} {'MaxDD':>7} {'PF':>6}")
        print("─" * 95)
        for exp in experiments:
            print(f"{exp['id']:<45} {exp['total_trades']:>7} "
                  f"{exp['win_rate']:>5.1f}% ${exp['total_pnl']:>10,.2f} "
                  f"{exp['sharpe_ratio']:>7.3f} {exp['max_drawdown_pct']:>6.1f}% "
                  f"{exp['profit_factor']:>5.2f}")
        print()
        return

    if args.show_experiment:
        exp = load_experiment(args.show_experiment)
        if not exp:
            print(f"Experiment '{args.show_experiment}' not found.")
            return

        print(f"\n{'='*60}")
        print(f"Experiment: {exp['id']}")
        print(f"{'='*60}")
        print(f"Created:              {exp['created_at']}")
        print(f"Wallets:              {exp['wallets']}")
        print(f"Total Trades:         {exp['total_trades']}")
        print(f"Win Rate:             {exp['win_rate']:.1f}%")
        print(f"Total PnL:            ${exp['total_pnl']:+,.2f}")
        print(f"Max Drawdown:         {exp['max_drawdown_pct']:.1f}%")
        print(f"Sharpe Ratio:         {exp['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio:        {exp['sortino_ratio']:.3f}")
        print(f"Profit Factor:        {exp['profit_factor']:.3f}")
        print(f"Calmar Ratio:         {exp['calmar_ratio']:.3f}")
        print(f"Expectancy:           {exp['expectancy']:.4f}")
        print(f"Avg Hold Time:        {exp['avg_hold_hours']:.1f}h")
        print(f"Max Consec Losses:    {exp['max_consecutive_losses']}")
        print(f"Duration:             {exp['duration_seconds']:.1f}s")

        if exp.get("notes"):
            print(f"Notes:                {exp['notes']}")

        cfg = json.loads(exp["config"]) if isinstance(exp["config"], str) else exp["config"]
        print(f"\nConfig: {json.dumps(cfg, indent=2)}")

        coin_data = json.loads(exp["coin_breakdown"]) if isinstance(exp["coin_breakdown"], str) else exp["coin_breakdown"]
        if coin_data:
            print("\nCoin Breakdown:")
            print(f"  {'Coin':<10} {'Trades':>7} {'WR%':>6} {'PnL':>12}")
            for coin, data in sorted(coin_data.items(), key=lambda x: x[1].get("pnl", 0), reverse=True):
                wr = data.get("win_rate", 0)
                print(f"  {coin:<10} {data['trades']:>7} {wr:>5.1f}% ${data['pnl']:>10,.2f}")

        print()
        return

    # Load fills
    if args.wallet:
        fills = load_fills_from_db(wallet_address=args.wallet)
    elif args.all_golden:
        fills = load_fills_from_db(golden_only=True)
    elif args.all:
        fills = load_fills_from_db()
    else:
        # Default: golden wallets
        fills = load_fills_from_db(golden_only=True)

    if not fills:
        print("No fills found. Run the golden wallet scan first:")
        print("  python main.py  (let it complete a discovery + golden scan cycle)")
        return

    print(f"Loaded {len(fills)} fills from {len(set(f.wallet_address for f in fills))} wallets")

    # Build config
    cfg = BacktestConfig(
        initial_balance=args.initial_balance,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        max_position_pct=args.max_position,
        max_positions=args.max_positions,
        slippage_bps=args.slippage,
        copy_delay_ms=args.copy_delay,
        seed=args.seed,
    )

    if args.sweep:
        # Parameter sweep
        results = parameter_sweep(fills)

        print(f"\n{'='*100}")
        print(f"Parameter Sweep Results ({len(results)} runs)")
        print(f"{'='*100}")
        print(f"{'Rank':>4} {'ID':<55} {'Sharpe':>8} {'PnL':>12} {'WR%':>6} {'MaxDD':>7} {'PF':>6}")
        print("─" * 100)

        for i, r in enumerate(results[:20]):
            print(f"{i+1:>4} {r.experiment_id:<55} "
                  f"{r.sharpe_ratio:>7.3f} ${r.total_pnl:>10,.2f} "
                  f"{r.win_rate:>5.1f}% {r.max_drawdown_pct:>6.1f}% "
                  f"{r.profit_factor:>5.2f}")

        # Save top results
        if not args.no_save:
            init_experiments_table()
            for r in results[:10]:
                save_experiment(r, notes=f"sweep: {args.notes}" if args.notes else "parameter sweep")
            print("\nSaved top 10 experiments to DB")

    else:
        # Single run
        engine = BacktestEngine(cfg)
        result = engine.run(fills)

        print(f"\n{'='*60}")
        print(f"Backtest Results: {result.experiment_id}")
        print(f"{'='*60}")
        print(f"Wallets:              {len(result.wallets_used)}")
        print(f"Total Trades:         {result.total_trades}")
        print(f"Winning:              {result.winning_trades}")
        print(f"Losing:               {result.losing_trades}")
        print(f"Win Rate:             {result.win_rate:.1f}%")
        print(f"Total PnL:            ${result.total_pnl:+,.2f}")
        print(f"Max Drawdown:         {result.max_drawdown_pct:.1f}%")
        print(f"Sharpe Ratio:         {result.sharpe_ratio:.3f}")
        print(f"Sortino Ratio:        {result.sortino_ratio:.3f}")
        print(f"Profit Factor:        {result.profit_factor:.3f}")
        print(f"Calmar Ratio:         {result.calmar_ratio:.3f}")
        print(f"Expectancy:           {result.expectancy:.4f}")
        print(f"Avg Hold Time:        {result.avg_hold_hours:.1f}h")
        print(f"Max Consec Losses:    {result.max_consecutive_losses}")
        print(f"Duration:             {result.duration_seconds:.1f}s")

        if result.coin_breakdown:
            print("\nCoin Breakdown:")
            print(f"  {'Coin':<10} {'Trades':>7} {'WR%':>6} {'PnL':>12}")
            for coin, data in sorted(result.coin_breakdown.items(),
                                      key=lambda x: x[1]["pnl"], reverse=True)[:15]:
                print(f"  {coin:<10} {data['trades']:>7} {data['win_rate']:>5.1f}% "
                      f"${data['pnl']:>10,.2f}")

        # Rejection stats
        if engine._rejection_reasons:
            print(f"\nSignal Rejections ({engine._signals_rejected} total):")
            for reason, count in sorted(engine._rejection_reasons.items(),
                                          key=lambda x: x[1], reverse=True):
                print(f"  {reason:<25} {count:>5}")

        # Save
        if not args.no_save:
            init_experiments_table()
            save_experiment(result, notes=args.notes)
            print(f"\nSaved as experiment: {result.experiment_id}")

    print()


if __name__ == "__main__":
    main()
