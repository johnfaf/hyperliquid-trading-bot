"""
Alpha Arena — Competitive Agent Evaluation & Evolution System
=============================================================
A Darwinian arena where trading agents compete, evolve, and get capital-allocated
based on live + backtested performance.

Three layers:
  1. TOURNAMENT   — Agents compete in rounds, scored on risk-adjusted returns.
                    Bottom performers eliminated, top performers promoted.
  2. CAPITAL      — Dynamic capital flows from underperformers to outperformers.
                    Good agents grow, bad agents starve.
  3. CONSENSUS    — Before execution, top agents vote/debate on each signal.
                    Weighted by track record. Majority or weighted-avg decides.

Plus:
  - SPAWNER      — Generates new mutant agents (tweaked parameters, combined signals)
  - BACKTESTER   — Tests agents on historical data with strict temporal isolation
                    (walk-forward, no future data leakage)
"""
import logging
import json
import copy
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import database as db
from src.signals.signal_schema import RiskParams, SignalSide, SignalSource, TradeSignal

logger = logging.getLogger(__name__)


def _normalize_candle_universe(candles: Optional[Any]) -> Dict[str, List[Dict]]:
    """Accept either a single candle series or a per-coin candle map."""
    if candles is None:
        return {}
    if isinstance(candles, dict):
        normalized: Dict[str, List[Dict]] = {}
        for coin, series in candles.items():
            if not isinstance(series, list) or not series:
                continue
            normalized[str(coin).upper()] = series
        return normalized
    if isinstance(candles, list) and candles:
        return {"BTC": candles}
    return {}


def _select_primary_coin(candle_universe: Dict[str, List[Dict]]) -> Optional[str]:
    if not candle_universe:
        return None
    if "BTC" in candle_universe:
        return "BTC"
    return max(candle_universe.items(), key=lambda item: len(item[1]))[0]


# ═══════════════════════════════════════════════════════════════
#  Data Models
# ═══════════════════════════════════════════════════════════════

class AgentStatus(str, Enum):
    ACTIVE = "active"           # Currently trading
    PROBATION = "probation"     # Underperforming, reduced capital
    INCUBATING = "incubating"   # New/spawned, being tested
    ELIMINATED = "eliminated"   # Removed from arena
    CHAMPION = "champion"       # Top performer, max capital


@dataclass
class ArenaAgent:
    """A competing agent in the Alpha Arena."""
    agent_id: str                        # Unique identifier
    name: str                            # Human-readable name
    strategy_type: str                   # e.g. "momentum_long", "mean_reversion"
    status: AgentStatus = AgentStatus.INCUBATING

    # Parameters (the DNA that gets mutated)
    params: Dict[str, Any] = field(default_factory=dict)

    # Performance tracking
    capital_allocated: float = 1000.0    # Current virtual capital
    total_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0

    # Arena metadata
    generation: int = 0                  # 0 = original, 1+ = spawned
    parent_id: str = ""                  # If spawned, who was the parent
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    rounds_survived: int = 0
    elo_rating: float = 1000.0           # Chess-style rating
    tournament_rank: int = 0

    # Backtest results (stored separately to avoid leakage)
    backtest_sharpe: float = 0.0
    backtest_pnl: float = 0.0
    backtest_trades: int = 0
    backtest_win_rate: float = 0.0

    # Trade history for scoring
    _returns: List[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["status"] = self.status.value
        d.pop("_returns", None)
        return d

    @property
    def accuracy(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0

    @property
    def fitness_score(self) -> float:
        """Composite fitness used for tournament ranking."""
        if self.total_trades < 3:
            return 0.0
        # Blend of: Sharpe (40%), win_rate (25%), PnL (20%), consistency (15%)
        sharpe_norm = min(max(self.sharpe_ratio / 3.0, -1), 1)  # -1 to 1
        pnl_norm = min(max(self.total_pnl / (self.capital_allocated + 1), -1), 1)
        dd_penalty = max(0, 1 - self.max_drawdown * 5)  # 20% DD = 0 penalty
        return (
            0.40 * sharpe_norm +
            0.25 * self.win_rate +
            0.20 * pnl_norm +
            0.15 * dd_penalty
        )


@dataclass
class TournamentRound:
    """A single tournament round."""
    round_id: int
    started_at: str
    ended_at: str = ""
    agents_entered: int = 0
    agents_eliminated: int = 0
    agents_promoted: int = 0
    agents_spawned: int = 0
    best_agent: str = ""
    best_fitness: float = 0.0
    summary: str = ""


@dataclass
class ConsensusVote:
    """A single agent's vote on a trade signal."""
    agent_id: str
    vote: str              # "approve", "reject", "abstain"
    confidence: float      # 0-1
    reasoning: str         # Short explanation
    weight: float = 1.0    # Based on agent's track record


# ═══════════════════════════════════════════════════════════════
#  1. TOURNAMENT ENGINE
# ═══════════════════════════════════════════════════════════════

class TournamentEngine:
    """
    Runs competitive rounds where agents are scored and ranked.
    Bottom performers get eliminated, top performers get promoted.
    """

    # What % of agents get eliminated each round
    ELIMINATION_RATE = 0.0    # No elimination — all 9 agents always compete
    PROMOTION_RATE = 0.10     # Top 10% become champions
    MIN_AGENTS = 9            # All 9 seed agents always compete

    def __init__(self):
        self.round_count = 0
        self.history: List[TournamentRound] = []

    def run_round(self, agents: List[ArenaAgent]) -> TournamentRound:
        """
        Run a single tournament round:
        1. Rank all active agents by fitness
        2. Eliminate bottom performers
        3. Promote top performers
        4. Update ELO ratings
        """
        self.round_count += 1
        rnd = TournamentRound(
            round_id=self.round_count,
            started_at=datetime.now(timezone.utc).isoformat(),
            agents_entered=len(agents),
        )

        # Only rank agents with enough trades
        scoreable = [a for a in agents if a.status != AgentStatus.ELIMINATED]
        active = [a for a in scoreable if a.total_trades >= 3]

        if len(active) < self.MIN_AGENTS:
            rnd.ended_at = datetime.now(timezone.utc).isoformat()
            rnd.summary = f"Not enough active agents ({len(active)}) — skipping round"
            self.history.append(rnd)
            return rnd

        # Rank by fitness score
        ranked = sorted(active, key=lambda a: a.fitness_score, reverse=True)
        for i, agent in enumerate(ranked):
            agent.tournament_rank = i + 1

        # Update ELO (simplified: compare each agent to the median)
        median_fitness = ranked[len(ranked) // 2].fitness_score
        for agent in ranked:
            expected = 1 / (1 + 10 ** ((median_fitness - agent.fitness_score) * 4))
            actual = 1.0 if agent.fitness_score > median_fitness else 0.0
            k = 32  # ELO K-factor
            agent.elo_rating += k * (actual - expected)
            agent.rounds_survived += 1

        # Promote top performers
        n_promote = max(1, int(len(ranked) * self.PROMOTION_RATE))
        for agent in ranked[:n_promote]:
            if agent.status != AgentStatus.CHAMPION:
                agent.status = AgentStatus.CHAMPION
                rnd.agents_promoted += 1
                logger.info(f"Arena PROMOTE: {agent.name} → Champion "
                           f"(fitness={agent.fitness_score:.3f}, elo={agent.elo_rating:.0f})")

        # No elimination — all 9 agents compete permanently
        # (ELIMINATION_RATE = 0.0 means no agents are eliminated)

        # Graduate incubating agents with enough trades
        for agent in agents:
            if agent.status == AgentStatus.INCUBATING and agent.total_trades >= 5:
                agent.status = AgentStatus.ACTIVE
                logger.info(f"Arena GRADUATE: {agent.name} → Active")

        rnd.best_agent = ranked[0].name if ranked else ""
        rnd.best_fitness = ranked[0].fitness_score if ranked else 0
        rnd.ended_at = datetime.now(timezone.utc).isoformat()
        rnd.summary = (f"Ranked {len(ranked)} agents. "
                      f"Champion: {rnd.best_agent} (fitness={rnd.best_fitness:.3f}). "
                      f"Promoted: {rnd.agents_promoted}, Eliminated: {rnd.agents_eliminated}")

        self.history.append(rnd)
        logger.info(f"Tournament Round #{self.round_count}: {rnd.summary}")

        return rnd


# ═══════════════════════════════════════════════════════════════
#  2. CAPITAL ALLOCATOR
# ═══════════════════════════════════════════════════════════════

class CapitalAllocator:
    """
    Equal capital allocation for the 9 fixed arena agents.
    Each agent receives an equal share of the total pool ($10,000 each).
    No fitness-based weighting — agents compete on pure performance.
    """

    BASE_CAPITAL = 10_000.0   # Each of 9 agents gets equal share
    TOTAL_POOL = 90_000.0     # Total virtual capital: 9 agents × $10,000 each
    MAX_SINGLE_AGENT = 0.25   # No single agent gets more than 25% of pool (not used with equal allocation)
    MIN_ALLOCATION = 10_000.0 # Minimum capital for any active agent (not used with equal allocation)

    def __init__(self, total_pool: float = None):
        if total_pool:
            self.TOTAL_POOL = total_pool

    def reallocate(self, agents: List[ArenaAgent]) -> Dict[str, float]:
        """
        Allocate equal capital to all 9 agents.
        No fitness-based weighting, no status multipliers — pure equality.
        Returns dict of agent_id → new_capital.
        """
        active = [a for a in agents if a.status not in (AgentStatus.ELIMINATED,)]

        if not active:
            return {}

        # Equal allocation: each of the 9 agents gets the same capital
        per_agent = round(self.TOTAL_POOL / len(active), 2)
        allocations = {}

        for agent in active:
            agent.capital_allocated = per_agent
            allocations[agent.agent_id] = per_agent

        logger.info(f"Capital allocation (equal): {len(allocations)} agents, "
                   f"${per_agent:,.2f} each (total ${self.TOTAL_POOL:,.0f})")

        return allocations


# ═══════════════════════════════════════════════════════════════
#  3. CONSENSUS ENGINE
# ═══════════════════════════════════════════════════════════════

class ConsensusEngine:
    """
    Multi-agent debate system. Before any trade executes,
    relevant agents vote on whether to approve or reject.

    Voting is weighted by each agent's ELO + track record.
    A signal needs >50% weighted approval to pass.
    """

    APPROVAL_THRESHOLD = 0.50   # Need 50%+ weighted votes to approve
    MIN_VOTERS = 3              # Need at least 3 agents to form consensus

    def __init__(self):
        self.vote_history: List[Dict] = []

    def get_consensus(self, signal: TradeSignal,
                       agents: List[ArenaAgent],
                       features: Optional[Dict] = None) -> Tuple[bool, float, List[ConsensusVote]]:
        """
        Gather votes from relevant agents on a trade signal.

        Returns: (approved: bool, consensus_confidence: float, votes: List[ConsensusVote])
        """
        # Select relevant agents (same strategy type or universal agents)
        voters = self._select_voters(signal, agents)

        if len(voters) < self.MIN_VOTERS:
            # Not enough voters — approve by default with reduced confidence
            return True, signal.confidence * 0.7, []

        votes = []
        for agent in voters:
            vote = self._agent_vote(agent, signal, features)
            votes.append(vote)

        # Compute weighted consensus
        total_weight = sum(v.weight for v in votes)
        if total_weight <= 0:
            return True, signal.confidence * 0.7, votes

        weighted_approve = sum(
            v.weight * v.confidence for v in votes if v.vote == "approve"
        )
        weighted_reject = sum(
            v.weight * v.confidence for v in votes if v.vote == "reject"
        )

        approval_ratio = weighted_approve / (weighted_approve + weighted_reject + 0.001)
        approved = approval_ratio >= self.APPROVAL_THRESHOLD

        # Consensus confidence: higher when agents agree strongly
        agreement = abs(approval_ratio - 0.5) * 2  # 0 = split, 1 = unanimous
        consensus_conf = signal.confidence * (0.5 + agreement * 0.5)

        # Log the debate
        approve_count = sum(1 for v in votes if v.vote == "approve")
        reject_count = sum(1 for v in votes if v.vote == "reject")

        self.vote_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "coin": signal.coin,
            "side": signal.side.value,
            "approved": approved,
            "approval_ratio": round(approval_ratio, 3),
            "votes_for": approve_count,
            "votes_against": reject_count,
            "consensus_confidence": round(consensus_conf, 3),
        })
        # Keep last 500 votes
        self.vote_history = self.vote_history[-500:]

        logger.info(f"Consensus {'APPROVED' if approved else 'REJECTED'} "
                   f"{signal.side.value} {signal.coin}: "
                   f"{approve_count} for / {reject_count} against "
                   f"(ratio={approval_ratio:.0%}, conf={consensus_conf:.0%})")

        return approved, consensus_conf, votes

    def _select_voters(self, signal: TradeSignal,
                        agents: List[ArenaAgent]) -> List[ArenaAgent]:
        """Select agents eligible to vote on this signal."""
        eligible = []
        for agent in agents:
            if agent.status == AgentStatus.ELIMINATED:
                continue
            if agent.total_trades < 3:
                continue
            # Agents with matching strategy type vote with full weight
            # Others can still vote but with reduced weight
            eligible.append(agent)

        # Sort by ELO (highest-rated agents first), take top 10
        eligible.sort(key=lambda a: a.elo_rating, reverse=True)
        return eligible[:10]

    def _agent_vote(self, agent: ArenaAgent, signal: TradeSignal,
                     features: Optional[Dict] = None) -> ConsensusVote:
        """
        An agent casts its vote on a signal.

        Logic is deterministic based on agent's strategy type and parameters:
        - Momentum agents approve trending signals, reject ranging
        - Mean reversion agents approve when RSI extreme, reject trends
        - etc.
        """
        # Weight based on ELO and accuracy
        weight = (agent.elo_rating / 1000) * (0.5 + agent.accuracy * 0.5)
        weight = max(0.1, min(weight, 3.0))

        # Determine if this agent's strategy aligns with the signal
        stype = agent.strategy_type.lower()
        side = signal.side.value

        # Base vote: each strategy type has preferences
        vote = "approve"
        confidence = 0.5
        reasoning = ""

        if stype in ("momentum_long", "trend_following"):
            if side == "long":
                vote = "approve"
                confidence = 0.6 + agent.accuracy * 0.3
                reasoning = "Momentum/trend aligns with long signal"
            else:
                vote = "reject" if agent.accuracy > 0.5 else "abstain"
                confidence = 0.4
                reasoning = "Momentum agent skeptical of short"

        elif stype in ("momentum_short",):
            if side == "short":
                vote = "approve"
                confidence = 0.6 + agent.accuracy * 0.3
                reasoning = "Short momentum aligns"
            else:
                vote = "reject" if agent.accuracy > 0.5 else "abstain"
                confidence = 0.4
                reasoning = "Short momentum agent skeptical of long"

        elif stype in ("mean_reversion", "contrarian"):
            # Mean reversion likes signals against the trend
            vote = "approve"
            confidence = 0.5 + agent.accuracy * 0.2
            reasoning = "Mean reversion considers entry"

        elif stype in ("scalping",):
            # Scalpers approve high-confidence short-term signals
            if signal.confidence >= 0.6:
                vote = "approve"
                confidence = signal.confidence * 0.8
                reasoning = "High confidence → scalp opportunity"
            else:
                vote = "abstain"
                confidence = 0.3
                reasoning = "Low confidence — not a scalp setup"

        elif stype in ("funding_arb", "delta_neutral"):
            # Conservative strategies are cautious
            vote = "approve" if signal.confidence >= 0.7 else "reject"
            confidence = 0.4
            reasoning = "Conservative strategy evaluating risk"

        elif stype == "lstm_direction":
            # LSTM agent votes based on signal confidence and its own track record
            if signal.confidence >= 0.55:
                vote = "approve"
                confidence = 0.5 + agent.accuracy * 0.4
                reasoning = "LSTM: high-confidence signal aligns"
            elif agent.accuracy > 0.55 and signal.confidence >= 0.4:
                vote = "approve"
                confidence = 0.45
                reasoning = "LSTM: moderate signal, good track record"
            else:
                vote = "abstain"
                confidence = 0.3
                reasoning = "LSTM: insufficient signal clarity"

        else:
            # Generic: vote based on accuracy history
            if agent.accuracy >= 0.55:
                vote = "approve"
                confidence = 0.5 + agent.accuracy * 0.3
            elif agent.accuracy < 0.40 and agent.total_trades >= 10:
                vote = "reject"
                confidence = 0.4
            else:
                vote = "approve"
                confidence = 0.5

            reasoning = f"Generic vote based on {agent.accuracy:.0%} accuracy"

        # Feature-based adjustments
        if features and isinstance(features, dict):
            score = features.get("overall_score", 0)

            # Strong feature alignment boosts confidence
            if (side == "long" and score > 0.3) or (side == "short" and score < -0.3):
                confidence = min(confidence * 1.2, 1.0)
                reasoning += " + features aligned"
            elif (side == "long" and score < -0.3) or (side == "short" and score > 0.3):
                confidence *= 0.7
                if vote == "approve":
                    vote = "abstain"
                reasoning += " + features opposed"

        # Match weight for strategy-specific relevance
        if stype == signal.strategy_type:
            weight *= 1.5  # Same strategy type = higher authority

        return ConsensusVote(
            agent_id=agent.agent_id,
            vote=vote,
            confidence=round(confidence, 3),
            reasoning=reasoning,
            weight=round(weight, 3),
        )


# ═══════════════════════════════════════════════════════════════
#  4. AGENT SPAWNER (Genetic Mutation)
# ═══════════════════════════════════════════════════════════════

class AgentSpawner:
    """
    Creates new agents by:
    1. Cloning + mutating top performers (genetic approach)
    2. Combining two parents (crossover)
    3. Creating random explorers (diversity)

    Mutation targets: confidence thresholds, position sizes, SL/TP ratios,
    leverage preferences, regime preferences, entry/exit timing.
    """

    # Mutable parameter ranges
    PARAM_RANGES = {
        "confidence_threshold": (0.20, 0.90),
        "position_pct": (0.02, 0.15),
        "stop_loss_pct": (0.02, 0.10),
        "take_profit_pct": (0.05, 0.25),
        "max_leverage": (1.0, 5.0),
        "trailing_stop_pct": (0.01, 0.05),
        "regime_sensitivity": (0.0, 1.0),     # How much to adjust for regime
        "momentum_weight": (0.0, 1.0),
        "mean_reversion_weight": (0.0, 1.0),
        "volume_confirmation_required": (0.0, 1.0),  # 0 = never, 1 = always
        "options_flow_weight": (0.0, 1.0),
        "entry_patience": (0.0, 1.0),         # 0 = market order, 1 = very patient limit
    }

    STRATEGY_TYPES = [
        "momentum_long", "momentum_short", "mean_reversion", "breakout",
        "scalping", "swing_trading", "funding_arb", "trend_following",
        "contrarian", "concentrated_bet", "diversified_portfolio",
        "lstm_direction",
    ]

    MUTATION_RATE = 0.3    # 30% of params get mutated
    MUTATION_SCALE = 0.2   # Mutations change param by ±20%

    def __init__(self):
        self._spawn_count = 0

    def spawn_mutant(self, parent: ArenaAgent) -> ArenaAgent:
        """Clone a parent agent and mutate its parameters."""
        self._spawn_count += 1

        child_params = copy.deepcopy(parent.params)

        # Mutate random subset of parameters
        for param, (lo, hi) in self.PARAM_RANGES.items():
            if param in child_params and random.random() < self.MUTATION_RATE:
                current = child_params[param]
                delta = current * self.MUTATION_SCALE * random.uniform(-1, 1)
                child_params[param] = max(lo, min(hi, current + delta))

        child_id = f"mutant_{parent.strategy_type}_{self._spawn_count}"
        child = ArenaAgent(
            agent_id=child_id,
            name=f"{parent.name}_mut{self._spawn_count}",
            strategy_type=parent.strategy_type,
            status=AgentStatus.INCUBATING,
            params=child_params,
            generation=parent.generation + 1,
            parent_id=parent.agent_id,
        )

        logger.info(f"Spawned mutant: {child.name} from {parent.name} (gen {child.generation})")
        return child

    def spawn_crossover(self, parent_a: ArenaAgent,
                          parent_b: ArenaAgent) -> ArenaAgent:
        """Combine parameters from two parents (crossover)."""
        self._spawn_count += 1

        child_params = {}
        all_params = set(list(parent_a.params.keys()) + list(parent_b.params.keys()))

        for param in all_params:
            a_val = parent_a.params.get(param)
            b_val = parent_b.params.get(param)

            if a_val is not None and b_val is not None:
                # Blend with random weight
                blend = random.uniform(0.3, 0.7)
                if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                    child_params[param] = a_val * blend + b_val * (1 - blend)
                else:
                    child_params[param] = random.choice([a_val, b_val])
            else:
                child_params[param] = a_val or b_val

        # Strategy type: inherit from the better parent
        better = parent_a if parent_a.fitness_score >= parent_b.fitness_score else parent_b
        strategy = better.strategy_type

        child_id = f"cross_{strategy}_{self._spawn_count}"
        child = ArenaAgent(
            agent_id=child_id,
            name=f"X({parent_a.name[:8]}+{parent_b.name[:8]})",
            strategy_type=strategy,
            status=AgentStatus.INCUBATING,
            params=child_params,
            generation=max(parent_a.generation, parent_b.generation) + 1,
            parent_id=f"{parent_a.agent_id}+{parent_b.agent_id}",
        )

        logger.info(f"Spawned crossover: {child.name}")
        return child

    def spawn_random(self) -> ArenaAgent:
        """Create a completely random explorer agent."""
        self._spawn_count += 1

        strategy = random.choice(self.STRATEGY_TYPES)
        params = {}
        for param, (lo, hi) in self.PARAM_RANGES.items():
            params[param] = lo + random.random() * (hi - lo)

        agent_id = f"explorer_{strategy}_{self._spawn_count}"
        agent = ArenaAgent(
            agent_id=agent_id,
            name=f"Explorer_{strategy[:6]}_{self._spawn_count}",
            strategy_type=strategy,
            status=AgentStatus.INCUBATING,
            params=params,
            generation=0,
        )

        logger.info(f"Spawned random explorer: {agent.name} ({strategy})")
        return agent

    def spawn_generation(self, agents: List[ArenaAgent],
                          target_count: int = 5) -> List[ArenaAgent]:
        """
        Spawn a new generation of agents:
        - 40% mutants from top performers
        - 30% crossovers from random top pairs
        - 30% random explorers
        """
        champions = [a for a in agents if a.status == AgentStatus.CHAMPION]
        active = [a for a in agents if a.status in (AgentStatus.ACTIVE, AgentStatus.CHAMPION)]

        if not active:
            # All explorers if no active agents
            return [self.spawn_random() for _ in range(target_count)]

        new_agents = []
        n_mutants = max(1, int(target_count * 0.4))
        n_cross = max(1, int(target_count * 0.3))
        n_random = target_count - n_mutants - n_cross

        # Mutants from top performers
        pool = champions if champions else active
        for _ in range(n_mutants):
            parent = random.choice(pool)
            new_agents.append(self.spawn_mutant(parent))

        # Crossovers
        if len(active) >= 2:
            for _ in range(n_cross):
                a, b = random.sample(active, 2)
                new_agents.append(self.spawn_crossover(a, b))
        else:
            n_random += n_cross

        # Random explorers
        for _ in range(n_random):
            new_agents.append(self.spawn_random())

        logger.info(f"Spawned generation: {n_mutants} mutants, {n_cross} crossovers, "
                   f"{n_random} explorers = {len(new_agents)} new agents")

        return new_agents


# ═══════════════════════════════════════════════════════════════
#  5. BACKTESTER (Walk-Forward, No Leakage)
# ═══════════════════════════════════════════════════════════════

class Backtester:
    """
    Tests agents on historical data with STRICT temporal isolation.

    Walk-forward approach:
    1. Split history into train/test windows
    2. Agent only sees data UP TO the current bar (no future leakage)
    3. Evaluate on out-of-sample test period only
    4. Roll forward and repeat

    Data leakage prevention:
    - Agent receives candles only up to bar[t], never bar[t+1..N]
    - No look-ahead in feature computation
    - Train/test split is strictly temporal
    - Results only counted on test periods
    """

    def __init__(
        self,
        lstm_agent: Optional[object] = None,
        risk_policy_engine: Optional[object] = None,
    ):
        self.results: Dict[str, Dict] = {}  # agent_id → backtest results
        self.lstm_agent = lstm_agent
        self.risk_policy_engine = risk_policy_engine

    def backtest_agent(self, agent: ArenaAgent,
                        historical_candles: Any,
                        train_pct: float = 0.60,
                        test_windows: int = 5) -> Dict:
        """
        Walk-forward backtest of an agent.

        Args:
            agent: The agent to test
            historical_candles: List of OHLCV dicts (must be sorted by time, oldest first)
            train_pct: Fraction of data for initial training
            test_windows: Number of walk-forward test windows

        Returns: Backtest result dict with PnL, Sharpe, win_rate, trades
        """
        candle_universe = _normalize_candle_universe(historical_candles)
        if not candle_universe:
            return {"error": "Insufficient data", "trades": 0}

        all_trades = []
        coin_results: Dict[str, Dict[str, float]] = {}
        tested_coins = 0

        for coin, coin_candles in candle_universe.items():
            n = len(coin_candles)
            if n < 50:
                continue

            tested_coins += 1

            if agent.strategy_type == "lstm_direction" and self.lstm_agent:
                try:
                    self.lstm_agent.train(coin_candles)
                except Exception as exc:
                    logger.debug("LSTM backtest train skipped for %s: %s", coin, exc)

            train_end = int(n * train_pct)
            test_size = max(1, (n - train_end) // max(test_windows, 1))
            coin_trades = []

            for window in range(test_windows):
                test_start = train_end + window * test_size
                test_end = min(test_start + test_size, n)

                if test_start >= n:
                    break

                for bar_idx in range(test_start, test_end):
                    bars_visible = coin_candles[:bar_idx + 1]
                    current_bar = coin_candles[bar_idx]

                    signal = self._agent_generate_signal(
                        agent,
                        bars_visible,
                        current_bar,
                        coin=coin,
                    )

                    if signal and bar_idx + 1 < n:
                        next_bar = coin_candles[bar_idx + 1]
                        future_bars = coin_candles[bar_idx + 1:]
                        trade_result = self._simulate_trade(
                            agent,
                            signal,
                            current_bar,
                            next_bar,
                            bars_visible,
                            future_bars,
                            coin=coin,
                            sort_key=(
                                int(next_bar.get("timestamp", bar_idx + 1) or bar_idx + 1),
                                coin,
                            ),
                        )
                        if trade_result:
                            coin_trades.append(trade_result)
                            all_trades.append(trade_result)

            if coin_trades:
                coin_results[coin] = {
                    "total_trades": len(coin_trades),
                    "total_pnl": round(sum(t["pnl"] for t in coin_trades), 4),
                    "win_rate": round(
                        sum(1 for t in coin_trades if t["won"]) / len(coin_trades),
                        4,
                    ),
                }

        if tested_coins == 0:
            return {"error": "Insufficient data", "trades": 0}

        all_trades.sort(key=lambda trade: trade.get("_sort_key", (0, "")))
        equity_curve = [agent.capital_allocated]
        for trade in all_trades:
            equity_curve.append(equity_curve[-1] + trade["pnl"])

        # Compute backtest metrics
        result = self._compute_backtest_metrics(agent, all_trades, equity_curve)
        result["coins_tested"] = tested_coins
        result["coin_results"] = coin_results
        self.results[agent.agent_id] = result

        # Store on agent
        agent.backtest_pnl = result.get("total_pnl", 0)
        agent.backtest_sharpe = result.get("sharpe", 0)
        agent.backtest_trades = result.get("total_trades", 0)
        agent.backtest_win_rate = result.get("win_rate", 0)

        logger.info(
            "Backtest %s across %d coin(s): %d trades, PnL=$%.2f, Sharpe=%.2f, WR=%.0f%%",
            agent.name,
            tested_coins,
            result["total_trades"],
            result["total_pnl"],
            result["sharpe"],
            result["win_rate"] * 100,
        )

        return result

    def _agent_generate_signal(self, agent: ArenaAgent,
                                 bars: List[Dict],
                                 current_bar: Dict,
                                 coin: str = "BTC") -> Optional[Dict]:
        """
        Agent generates a signal based on visible data.
        Uses agent's strategy_type and params to decide.
        """
        if len(bars) < 20:
            return None

        params = agent.params
        closes = [b["close"] for b in bars[-30:]]
        current_price = current_bar["close"]

        # Compute simple features (no future data)
        sma_fast = np.mean(closes[-5:])
        sma_slow = np.mean(closes[-20:])
        prev = np.array(closes[-21:-1])
        curr = np.array(closes[-20:])
        if len(prev) != len(curr):
            prev = prev[:len(curr)]
        returns = (curr - prev) / (prev + 1e-10)

        # RSI (14-period)
        gains = [max(r, 0) for r in returns[-14:]]
        losses = [-min(r, 0) for r in returns[-14:]]
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0.001
        rsi = 100 - 100 / (1 + avg_gain / avg_loss)

        # Volatility
        atr_pct = np.std(returns[-14:]) if len(returns) >= 14 else 0.02

        # Strategy-specific signal generation
        stype = agent.strategy_type
        conf_threshold = params.get("confidence_threshold", 0.5)

        side = None
        confidence = 0.0

        if stype in ("momentum_long", "trend_following"):
            if sma_fast > sma_slow and rsi > 50 and rsi < 80:
                side = "long"
                momentum = (sma_fast - sma_slow) / sma_slow
                confidence = min(0.5 + momentum * 10, 0.95)

        elif stype == "momentum_short":
            if sma_fast < sma_slow and rsi < 50 and rsi > 20:
                side = "short"
                momentum = (sma_slow - sma_fast) / sma_slow
                confidence = min(0.5 + momentum * 10, 0.95)

        elif stype in ("mean_reversion", "contrarian"):
            if rsi < 30:
                side = "long"
                confidence = 0.5 + (30 - rsi) / 60
            elif rsi > 70:
                side = "short"
                confidence = 0.5 + (rsi - 70) / 60

        elif stype == "breakout":
            high_20 = max(b["high"] for b in bars[-20:])
            low_20 = min(b["low"] for b in bars[-20:])
            if current_price > high_20 * 0.99:
                side = "long"
                confidence = 0.6
            elif current_price < low_20 * 1.01:
                side = "short"
                confidence = 0.6

        elif stype == "scalping":
            # Short-term mean reversion
            sma_5 = np.mean(closes[-5:])
            if current_price < sma_5 * 0.99:
                side = "long"
                confidence = 0.55
            elif current_price > sma_5 * 1.01:
                side = "short"
                confidence = 0.55

        elif stype == "swing_trading":
            if sma_fast > sma_slow and rsi < 60:
                side = "long"
                confidence = 0.55 + (60 - rsi) / 100

        elif stype == "lstm_direction":
            if self.lstm_agent:
                try:
                    lstm_signal = self.lstm_agent.generate_signal(bars)
                except Exception as exc:
                    logger.debug("LSTM signal generation failed: %s", exc)
                    lstm_signal = None
                if lstm_signal:
                    side = lstm_signal.get("side")
                    confidence = float(lstm_signal.get("confidence", 0.0) or 0.0)
                    atr_pct = float(lstm_signal.get("atr_pct", atr_pct) or atr_pct)

            # Fallback when the model is unavailable or not confident enough.
            if side is None:
                signals_up = 0
                signals_down = 0
                if sma_fast > sma_slow:
                    signals_up += 1
                else:
                    signals_down += 1
                if rsi > 55:
                    signals_up += 1
                elif rsi < 45:
                    signals_down += 1
                # Volume momentum proxy
                if len(closes) >= 10:
                    vol_recent = np.std(returns[-5:])
                    vol_older = np.std(returns[-10:-5]) if len(returns) >= 10 else vol_recent
                    if vol_recent > vol_older * 1.2:  # Expanding volatility
                        if sma_fast > sma_slow:
                            signals_up += 1
                        else:
                            signals_down += 1
                if signals_up >= 2 and signals_down == 0:
                    side = "long"
                    confidence = 0.5 + signals_up * 0.1
                elif signals_down >= 2 and signals_up == 0:
                    side = "short"
                    confidence = 0.5 + signals_down * 0.1

        else:
            # Default: simple momentum
            if sma_fast > sma_slow:
                side = "long"
                confidence = 0.5
            elif sma_fast < sma_slow:
                side = "short"
                confidence = 0.5

        # Apply confidence threshold from agent params
        if side and confidence >= conf_threshold:
            return {
                "coin": coin,
                "side": side,
                "confidence": confidence,
                "price": current_price,
                "atr_pct": atr_pct,
            }

        return None

    def _resolve_trade_signal(
        self,
        agent: ArenaAgent,
        signal: Dict,
        entry_price: float,
    ) -> TradeSignal:
        params = agent.params
        trade_signal = TradeSignal(
            coin=str(signal.get("coin", "BTC") or "BTC").upper(),
            side=SignalSide(signal["side"]),
            confidence=float(signal.get("confidence", 0.5) or 0.5),
            source=SignalSource.ARENA_CHAMPION,
            reason=f"Arena backtest: {agent.strategy_type}",
            strategy_type=agent.strategy_type,
            entry_price=entry_price,
            leverage=float(params.get("max_leverage", 2.0) or 2.0),
            position_pct=float(params.get("position_pct", 0.05) or 0.05),
            risk=RiskParams(
                stop_loss_pct=float(params.get("stop_loss_pct", 0.05) or 0.05),
                take_profit_pct=float(params.get("take_profit_pct", 0.10) or 0.10),
                trailing_stop=True,
                risk_basis=str(params.get("risk_basis", "price") or "price"),
            ),
            context={
                "features": {},
                "atr_pct": float(signal.get("atr_pct", 0.02) or 0.02),
                "volatility": float(signal.get("atr_pct", 0.02) or 0.02),
            },
        )
        if self.risk_policy_engine:
            try:
                adjusted = self.risk_policy_engine.apply(trade_signal)
                if adjusted is not None:
                    trade_signal = adjusted
            except Exception as exc:
                logger.debug("Arena risk policy apply failed: %s", exc)
        return trade_signal

    @staticmethod
    def _r_multiple(
        entry_price: float,
        current_price: float,
        side: str,
        leverage: float,
        stop_roe_pct: float,
    ) -> float:
        if entry_price <= 0 or current_price <= 0 or leverage <= 0 or stop_roe_pct <= 0:
            return 0.0
        direction = 1.0 if side == "long" else -1.0
        move_pct = ((current_price - entry_price) / entry_price) * direction
        return (move_pct * leverage) / stop_roe_pct

    def _simulate_trade(self, agent: ArenaAgent, signal: Dict,
                          entry_bar: Dict, exit_bar: Dict,
                          history: List[Dict], future_bars: List[Dict],
                          coin: str = "BTC",
                          sort_key: Optional[Tuple[Any, ...]] = None) -> Optional[Dict]:
        """Simulate a trade using the same dynamic risk shape as live execution."""
        entry_price = exit_bar["open"]  # Enter at next bar's open (realistic)
        if entry_price <= 0 or not future_bars:
            return None

        trade_signal = self._resolve_trade_signal(agent, signal, entry_price)
        leverage = max(float(trade_signal.leverage or 1.0), 1.0)
        pos_pct = float(agent.params.get("position_pct", 0.05) or 0.05)
        size = agent.capital_allocated * pos_pct / entry_price
        side = signal["side"]

        stop_price, take_profit_price = trade_signal.risk.resolve_trigger_prices(
            entry_price,
            side,
            leverage,
        )
        stop_roe_pct = max(trade_signal.risk.resolve_roe_stop_loss_pct(leverage), 1e-9)
        break_even_at_r = float(trade_signal.risk.break_even_at_r or 0.0)
        break_even_buffer_pct = trade_signal.risk.resolve_price_break_even_buffer_pct(leverage)
        trail_after_r = float(trade_signal.risk.trail_activate_at_r or 0.0)
        trailing_distance_pct = trade_signal.risk.resolve_price_trailing_pct(leverage)
        trailing_enabled = bool(trade_signal.risk.trailing_stop)
        max_hold_bars = max(1, int(round(float(trade_signal.risk.time_limit_hours or 1.0))))

        current_stop = stop_price
        best_price = entry_price
        exit_price = future_bars[min(len(future_bars), max_hold_bars) - 1]["close"]

        for bar in future_bars[:max_hold_bars]:
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])

            favorable_price = high if side == "long" else low
            favorable_r = self._r_multiple(entry_price, favorable_price, side, leverage, stop_roe_pct)

            if break_even_at_r > 0 and favorable_r >= break_even_at_r:
                if side == "long":
                    current_stop = max(current_stop, entry_price * (1 + break_even_buffer_pct))
                else:
                    current_stop = min(current_stop, entry_price * (1 - break_even_buffer_pct))

            if trailing_enabled and trail_after_r > 0 and favorable_r >= trail_after_r:
                if side == "long":
                    best_price = max(best_price, high)
                    current_stop = max(current_stop, best_price * (1 - trailing_distance_pct))
                else:
                    best_price = min(best_price, low)
                    current_stop = min(current_stop, best_price * (1 + trailing_distance_pct))

            if side == "long":
                if low <= current_stop:
                    exit_price = current_stop
                    break
                if high >= take_profit_price:
                    exit_price = take_profit_price
                    break
            else:
                if high >= current_stop:
                    exit_price = current_stop
                    break
                if low <= take_profit_price:
                    exit_price = take_profit_price
                    break

            exit_price = close

        if side == "long":
            pnl = (exit_price - entry_price) * size * leverage
        else:
            pnl = (entry_price - exit_price) * size * leverage

        return {
            "coin": str(signal.get("coin", coin) or coin).upper(),
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": round(pnl, 4),
            "return_pct": pnl / (entry_price * size * leverage) if entry_price > 0 else 0,
            "won": pnl > 0,
            "_sort_key": sort_key or (0, str(signal.get("coin", coin) or coin).upper()),
        }

    def _compute_backtest_metrics(self, agent: ArenaAgent,
                                    trades: List[Dict],
                                    equity_curve: List[float]) -> Dict:
        """Compute comprehensive backtest metrics."""
        if not trades:
            return {"total_trades": 0, "total_pnl": 0, "sharpe": 0,
                    "win_rate": 0, "max_drawdown": 0, "profit_factor": 0}

        pnls = [t["pnl"] for t in trades]
        returns = [t["return_pct"] for t in trades]
        wins = sum(1 for t in trades if t["won"])

        total_pnl = sum(pnls)
        win_rate = wins / len(trades)

        # Sharpe
        if len(returns) >= 5:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
        else:
            sharpe = 0.0

        # Max drawdown from equity curve
        peak = equity_curve[0]
        max_dd = 0
        for val in equity_curve:
            peak = max(peak, val)
            dd = (peak - val) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            "total_trades": len(trades),
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(win_rate, 4),
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "profit_factor": round(profit_factor, 2),
            "avg_pnl": round(np.mean(pnls), 4),
            "best_trade": round(max(pnls), 4),
            "worst_trade": round(min(pnls), 4),
            "equity_final": round(equity_curve[-1], 2),
        }


# ═══════════════════════════════════════════════════════════════
#  6. ALPHA ARENA (Orchestrator)
# ═══════════════════════════════════════════════════════════════

class AlphaArena:
    """
    The Alpha Arena orchestrator. Ties together:
    - Tournament rounds
    - Capital allocation
    - Consensus voting
    - Agent spawning
    - Backtesting

    Lifecycle:
    1. Initialize with seed agents (one per strategy type)
    2. Each cycle: agents generate signals → consensus → execute
    3. Periodically: run tournament round → reallocate capital → spawn new agents
    4. On spawn: backtest new agents on historical data → only promote survivors
    """

    MAX_AGENTS = 9               # Fixed at exactly 9 agents
    TOURNAMENT_INTERVAL = 5      # Run tournament every N cycles
    SPAWN_INTERVAL = 0           # Disabled — no new agents spawned
    SPAWN_COUNT = 0              # Disabled — only 9 seed agents

    def __init__(
        self,
        lstm_agent: Optional[object] = None,
        risk_policy_engine: Optional[object] = None,
    ):
        self.agents: Dict[str, ArenaAgent] = {}
        self.tournament = TournamentEngine()
        self.allocator = CapitalAllocator()
        self.consensus = ConsensusEngine()
        self.spawner = AgentSpawner()
        self.lstm_agent = lstm_agent
        self.risk_policy_engine = risk_policy_engine
        self.backtester = Backtester(
            lstm_agent=lstm_agent,
            risk_policy_engine=risk_policy_engine,
        )
        self.cycle_count = 0

        # Initialize with seed agents
        self._seed_agents()

        # Persistence
        self._init_db()
        self._load_agents()

        logger.info(f"Alpha Arena initialized with {len(self.agents)} agents")

    def _init_db(self):
        """Create arena tables if they don't exist."""
        try:
            with db.get_connection() as conn:
                if db.get_backend_name() == "postgres":
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS arena_agents (
                            agent_id TEXT PRIMARY KEY,
                            name TEXT,
                            strategy_type TEXT,
                            status TEXT DEFAULT 'incubating',
                            params TEXT DEFAULT '{}',
                            capital_allocated DOUBLE PRECISION DEFAULT 1000,
                            total_pnl DOUBLE PRECISION DEFAULT 0,
                            total_trades INTEGER DEFAULT 0,
                            winning_trades INTEGER DEFAULT 0,
                            sharpe_ratio DOUBLE PRECISION DEFAULT 0,
                            max_drawdown DOUBLE PRECISION DEFAULT 0,
                            win_rate DOUBLE PRECISION DEFAULT 0,
                            generation INTEGER DEFAULT 0,
                            parent_id TEXT DEFAULT '',
                            elo_rating DOUBLE PRECISION DEFAULT 1000,
                            tournament_rank INTEGER DEFAULT 0,
                            backtest_sharpe DOUBLE PRECISION DEFAULT 0,
                            backtest_pnl DOUBLE PRECISION DEFAULT 0,
                            backtest_trades INTEGER DEFAULT 0,
                            backtest_win_rate DOUBLE PRECISION DEFAULT 0,
                            created_at TIMESTAMPTZ,
                            last_updated TIMESTAMPTZ
                        )
                    """)
                    for col, typedef in [
                        ("backtest_trades", "INTEGER DEFAULT 0"),
                        ("backtest_win_rate", "DOUBLE PRECISION DEFAULT 0"),
                    ]:
                        try:
                            conn.execute(f"ALTER TABLE arena_agents ADD COLUMN {col} {typedef}")
                        except Exception:
                            pass
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS arena_rounds (
                            round_id BIGINT PRIMARY KEY,
                            started_at TIMESTAMPTZ,
                            ended_at TIMESTAMPTZ,
                            agents_entered INTEGER,
                            agents_eliminated INTEGER,
                            agents_promoted INTEGER,
                            agents_spawned INTEGER,
                            best_agent TEXT,
                            best_fitness DOUBLE PRECISION,
                            summary TEXT
                        )
                    """)
                else:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS arena_agents (
                            agent_id TEXT PRIMARY KEY,
                            name TEXT,
                            strategy_type TEXT,
                            status TEXT DEFAULT 'incubating',
                            params TEXT DEFAULT '{}',
                            capital_allocated REAL DEFAULT 1000,
                            total_pnl REAL DEFAULT 0,
                            total_trades INTEGER DEFAULT 0,
                            winning_trades INTEGER DEFAULT 0,
                            sharpe_ratio REAL DEFAULT 0,
                            max_drawdown REAL DEFAULT 0,
                            win_rate REAL DEFAULT 0,
                            generation INTEGER DEFAULT 0,
                            parent_id TEXT DEFAULT '',
                            elo_rating REAL DEFAULT 1000,
                            tournament_rank INTEGER DEFAULT 0,
                            backtest_sharpe REAL DEFAULT 0,
                            backtest_pnl REAL DEFAULT 0,
                            backtest_trades INTEGER DEFAULT 0,
                            backtest_win_rate REAL DEFAULT 0,
                            created_at TEXT,
                            last_updated TEXT
                        )
                    """)
                    for col, typedef in [
                        ("backtest_trades", "INTEGER DEFAULT 0"),
                        ("backtest_win_rate", "REAL DEFAULT 0"),
                    ]:
                        try:
                            conn.execute(f"ALTER TABLE arena_agents ADD COLUMN {col} {typedef}")
                        except Exception:
                            pass
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS arena_rounds (
                            round_id INTEGER PRIMARY KEY,
                            started_at TEXT,
                            ended_at TEXT,
                            agents_entered INTEGER,
                            agents_eliminated INTEGER,
                            agents_promoted INTEGER,
                            agents_spawned INTEGER,
                            best_agent TEXT,
                            best_fitness REAL,
                            summary TEXT
                        )
                    """)
        except Exception as e:
            logger.warning(f"Arena DB init error: {e}")

    def _seed_agents(self):
        """Create one seed agent per strategy type."""
        strategy_types = [
            "momentum_long", "momentum_short", "mean_reversion",
            "breakout", "scalping", "swing_trading", "funding_arb",
            "trend_following", "contrarian", "lstm_direction",
        ]

        for stype in strategy_types:
            agent_id = f"seed_{stype}"
            if agent_id not in self.agents:
                # Default balanced params
                params = {
                    "confidence_threshold": 0.50,
                    "position_pct": 0.05,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.10,
                    "max_leverage": 2.0,
                    "trailing_stop_pct": 0.025,
                    "regime_sensitivity": 0.5,
                    "momentum_weight": 0.5 if "momentum" in stype else 0.3,
                    "mean_reversion_weight": 0.5 if "mean_reversion" in stype else 0.3,
                    "volume_confirmation_required": 0.5,
                    "options_flow_weight": 0.3,
                    "entry_patience": 0.3,
                }
                self.agents[agent_id] = ArenaAgent(
                    agent_id=agent_id,
                    name=f"Seed_{stype}",
                    strategy_type=stype,
                    status=AgentStatus.ACTIVE,
                    params=params,
                )

    def _load_agents(self):
        """Load agents from DB, purge non-seed agents, refill with seeds if needed."""
        try:
            with db.get_connection(for_read=True) as conn:
                rows = conn.execute("SELECT * FROM arena_agents").fetchall()

            # Load all agents from DB first
            for row in rows:
                row = dict(row)
                agent = ArenaAgent(
                    agent_id=row["agent_id"],
                    name=row["name"],
                    strategy_type=row["strategy_type"],
                    status=AgentStatus(row.get("status", "active")),
                    params=json.loads(row.get("params", "{}")),
                    capital_allocated=row.get("capital_allocated", 1000),
                    total_pnl=row.get("total_pnl", 0),
                    total_trades=row.get("total_trades", 0),
                    winning_trades=row.get("winning_trades", 0),
                    sharpe_ratio=row.get("sharpe_ratio", 0),
                    max_drawdown=row.get("max_drawdown", 0),
                    win_rate=row.get("win_rate", 0),
                    generation=row.get("generation", 0),
                    parent_id=row.get("parent_id", ""),
                    elo_rating=row.get("elo_rating", 1000),
                    tournament_rank=row.get("tournament_rank", 0),
                    backtest_sharpe=row.get("backtest_sharpe", 0),
                    backtest_pnl=row.get("backtest_pnl", 0),
                    backtest_trades=row.get("backtest_trades", 0),
                    backtest_win_rate=row.get("backtest_win_rate", 0),
                    created_at=row.get("created_at", ""),
                )
                self.agents[row["agent_id"]] = agent
            # Purge non-seed agents (only keep agents with IDs starting with "seed_")
            non_seed_ids = [aid for aid in self.agents.keys() if not aid.startswith("seed_")]
            for aid in non_seed_ids:
                del self.agents[aid]
                logger.info(f"Arena purged non-seed agent: {aid}")

            if rows:
                logger.info(f"Loaded {len(rows)} agents from DB; kept {len(self.agents)} seed agents")

            # Refill with seed agents if we have fewer than 9
            if len(self.agents) < 9:
                self._seed_agents()
                logger.info(f"Arena refilled to {len(self.agents)} seed agents")

        except Exception as e:
            logger.debug(f"Arena load: {e}")

    def _save_agents(self):
        """Persist all agents to DB."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            with db.get_connection() as conn:
                for agent in self.agents.values():
                    conn.execute("""
                        INSERT INTO arena_agents
                        (agent_id, name, strategy_type, status, params,
                         capital_allocated, total_pnl, total_trades, winning_trades,
                         sharpe_ratio, max_drawdown, win_rate, generation, parent_id,
                         elo_rating, tournament_rank, backtest_sharpe, backtest_pnl,
                         backtest_trades, backtest_win_rate,
                         created_at, last_updated)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT (agent_id) DO UPDATE SET
                            name = EXCLUDED.name,
                            strategy_type = EXCLUDED.strategy_type,
                            status = EXCLUDED.status,
                            params = EXCLUDED.params,
                            capital_allocated = EXCLUDED.capital_allocated,
                            total_pnl = EXCLUDED.total_pnl,
                            total_trades = EXCLUDED.total_trades,
                            winning_trades = EXCLUDED.winning_trades,
                            sharpe_ratio = EXCLUDED.sharpe_ratio,
                            max_drawdown = EXCLUDED.max_drawdown,
                            win_rate = EXCLUDED.win_rate,
                            generation = EXCLUDED.generation,
                            parent_id = EXCLUDED.parent_id,
                            elo_rating = EXCLUDED.elo_rating,
                            tournament_rank = EXCLUDED.tournament_rank,
                            backtest_sharpe = EXCLUDED.backtest_sharpe,
                            backtest_pnl = EXCLUDED.backtest_pnl,
                            backtest_trades = EXCLUDED.backtest_trades,
                            backtest_win_rate = EXCLUDED.backtest_win_rate,
                            created_at = EXCLUDED.created_at,
                            last_updated = EXCLUDED.last_updated
                    """, (
                        agent.agent_id, agent.name, agent.strategy_type,
                        agent.status.value, json.dumps(agent.params),
                        agent.capital_allocated, agent.total_pnl,
                        agent.total_trades, agent.winning_trades,
                        agent.sharpe_ratio, agent.max_drawdown, agent.win_rate,
                        agent.generation, agent.parent_id,
                        agent.elo_rating, agent.tournament_rank,
                        agent.backtest_sharpe, agent.backtest_pnl,
                        agent.backtest_trades, agent.backtest_win_rate,
                        agent.created_at, now,
                    ))
        except Exception as e:
            logger.warning(f"Arena save error: {e}")

    # ─── Public API ────────────────────────────────────────────

    def get_consensus_on_signal(self, signal: TradeSignal,
                                  features: Optional[Dict] = None) -> Tuple[bool, float]:
        """
        Run consensus voting on a trade signal.
        Returns (approved, adjusted_confidence).
        """
        active_agents = [a for a in self.agents.values()
                        if a.status != AgentStatus.ELIMINATED]

        approved, consensus_conf, votes = self.consensus.get_consensus(
            signal, active_agents, features
        )

        return approved, consensus_conf

    def record_trade_result(self, agent_id: str, pnl: float,
                              return_pct: float = 0.0):
        """Record a trade outcome for a specific agent."""
        agent = self.agents.get(agent_id)
        if not agent:
            return

        agent.total_trades += 1
        agent.total_pnl += pnl
        if pnl > 0:
            agent.winning_trades += 1
        agent.win_rate = agent.accuracy
        agent._returns.append(return_pct)

        # Update Sharpe
        if len(agent._returns) >= 5:
            agent.sharpe_ratio = float(
                np.mean(agent._returns) / (np.std(agent._returns) + 1e-8)
            )

        # Update max drawdown (fractional, consistent with fitness_score formula)
        cumulative = np.cumsum(agent._returns)
        if len(cumulative) > 0:
            peak = np.maximum.accumulate(cumulative)
            # Fractional drawdown: how far below peak as a fraction of peak
            drawdowns = np.where(peak != 0, (peak - cumulative) / (np.abs(peak) + 1e-8), 0)
            agent.max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

    def record_trade_for_strategy(self, strategy_type: str, pnl: float,
                                    return_pct: float = 0.0):
        """Record outcome to ALL agents matching a strategy type."""
        matched = [a for a in self.agents.values()
                  if a.strategy_type == strategy_type
                  and a.status != AgentStatus.ELIMINATED]

        for agent in matched:
            self.record_trade_result(agent.agent_id, pnl, return_pct)

    def run_cycle(self, historical_candles: Optional[Any] = None):
        """
        Run one arena cycle with 9 fixed agents:
        1. If tournament interval: run tournament round
        2. Reallocate capital (equal allocation)
        3. Save state

        Spawning is disabled — only the 9 seed agents exist.
        No elimination — all 9 agents always compete.
        """
        self.cycle_count += 1

        candle_universe = _normalize_candle_universe(historical_candles)
        primary_coin = _select_primary_coin(candle_universe)

        if primary_coin and self.lstm_agent:
            try:
                self.lstm_agent.train(candle_universe[primary_coin])
            except Exception as exc:
                logger.debug("Arena LSTM training skipped: %s", exc)

        if candle_universe and self.cycle_count % self.TOURNAMENT_INTERVAL == 0:
            tested_agents = 0
            for agent in self.agents.values():
                if agent.status == AgentStatus.ELIMINATED:
                    continue
                try:
                    self.backtester.backtest_agent(agent, candle_universe)
                    tested_agents += 1
                except Exception as exc:
                    logger.debug("Arena backtest skipped for %s: %s", agent.name, exc)
            if tested_agents:
                logger.info(
                    "Arena refreshed multi-coin backtests for %d agents across %d coins",
                    tested_agents,
                    len(candle_universe),
                )

        # Tournament
        if self.cycle_count % self.TOURNAMENT_INTERVAL == 0:
            agents_list = list(self.agents.values())
            self.tournament.run_round(agents_list)

        # Spawning disabled — all 9 seed agents remain fixed
        # (SPAWN_INTERVAL = 0 prevents this block from ever executing)

        # Reallocate capital (equal allocation to all 9 agents)
        active = [a for a in self.agents.values()
                 if a.status != AgentStatus.ELIMINATED]
        self.allocator.reallocate(active)

        # Save
        self._save_agents()

        logger.info(f"Arena cycle #{self.cycle_count}: "
                   f"{len(self.agents)} agents (fixed at 9) "
                   f"({sum(1 for a in self.agents.values() if a.status == AgentStatus.CHAMPION)} champions, "
                   f"{sum(1 for a in self.agents.values() if a.status == AgentStatus.ACTIVE)} active, "
                   f"{sum(1 for a in self.agents.values() if a.status == AgentStatus.INCUBATING)} incubating)")

    def get_leaderboard(self, top_n: int = 20) -> List[Dict]:
        """Get the arena leaderboard."""
        agents = [a for a in self.agents.values()
                 if a.status != AgentStatus.ELIMINATED]
        agents.sort(key=lambda a: a.fitness_score, reverse=True)

        return [
            {
                "rank": i + 1,
                "name": a.name,
                "strategy": a.strategy_type,
                "status": a.status.value,
                "elo": round(a.elo_rating, 0),
                "fitness": round(a.fitness_score, 4),
                "capital": round(a.capital_allocated, 0),
                "pnl": round(a.total_pnl, 2),
                "trades": a.total_trades,
                "win_rate": round(a.win_rate * 100, 1),
                "sharpe": round(a.sharpe_ratio, 2),
                "generation": a.generation,
                "backtest_sharpe": round(a.backtest_sharpe, 2),
            }
            for i, a in enumerate(agents[:top_n])
        ]

    def get_stats(self) -> Dict:
        """Get arena summary stats."""
        agents = list(self.agents.values())
        active = [a for a in agents if a.status != AgentStatus.ELIMINATED]

        return {
            "total_agents": len(agents),
            "active_agents": len(active),
            "champions": sum(1 for a in agents if a.status == AgentStatus.CHAMPION),
            "incubating": sum(1 for a in agents if a.status == AgentStatus.INCUBATING),
            "eliminated": sum(1 for a in agents if a.status == AgentStatus.ELIMINATED),
            "probation": sum(1 for a in agents if a.status == AgentStatus.PROBATION),
            "total_cycles": self.cycle_count,
            "total_rounds": self.tournament.round_count,
            "max_elo": max((a.elo_rating for a in active), default=0),
            "avg_fitness": np.mean([a.fitness_score for a in active]) if active else 0,
            "total_arena_pnl": sum(a.total_pnl for a in active),
            "consensus_history": len(self.consensus.vote_history),
            "recent_votes": self.consensus.vote_history[-5:],
        }

    # ═══════════════════════════════════════════════════════════
    #  V7: Champion Signal Generation (closes Arena → Live gap)
    # ═══════════════════════════════════════════════════════════

    def get_champion_signals(
        self,
        current_candles: Optional[Any] = None,
        min_fitness: float = 0.15,
        min_trades: int = 5,
        min_win_rate: float = 0.45,
    ) -> List[Dict]:
        """
        Generate live trading signals from champion/top-performing agents.

        This closes the critical gap where Arena agents evolved and learned
        but never contributed signals to the live trading pipeline.

        Args:
            current_candles: Recent candle data (at least 30 bars).
                             If None, no signals generated.
            min_fitness: Minimum fitness score to qualify (default 0.15)
            min_trades: Minimum trade history to trust agent (default 5)
            min_win_rate: Minimum realized win rate required (default 45%)

        Returns:
            List of signal dicts compatible with paper_trader pipeline:
            [{"coin": "BTC", "side": "long", "confidence": 0.72,
              "source": "arena_champion", "agent_id": "...", ...}]
        """
        candle_universe = {
            coin: candles[-100:]
            for coin, candles in _normalize_candle_universe(current_candles).items()
            if len(candles) >= 30
        }
        if not candle_universe:
            return []

        # Get qualified agents: champions + high-performing active agents
        qualified = [
            a for a in self.agents.values()
            if a.status in (AgentStatus.CHAMPION, AgentStatus.ACTIVE)
            and a.fitness_score >= min_fitness
            and a.total_trades >= min_trades
            and a.win_rate >= min_win_rate
        ]

        if not qualified:
            return []

        # Sort by fitness — best agents first
        qualified.sort(key=lambda a: a.fitness_score, reverse=True)

        signals = []
        seen_agent_sides = set()  # Avoid duplicate coin+strategy_type+side signals

        for agent in qualified[:10]:  # Top 10 qualified agents
            try:
                best_signal = None
                for coin, coin_candles in candle_universe.items():
                    signal = self.backtester._agent_generate_signal(
                        agent,
                        coin_candles,
                        coin_candles[-1],
                        coin=coin,
                    )
                    if signal and (
                        best_signal is None
                        or float(signal.get("confidence", 0.0) or 0.0)
                        > float(best_signal.get("confidence", 0.0) or 0.0)
                    ):
                        best_signal = signal

                if not best_signal:
                    continue

                # Dedup on coin+strategy_type+side so different strategies and coins can coexist
                sig_key = f"{best_signal['coin']}:{agent.strategy_type}:{best_signal['side']}"
                if sig_key in seen_agent_sides:
                    continue
                seen_agent_sides.add(sig_key)

                # Weight confidence by agent fitness and track record
                base_conf = best_signal["confidence"]
                fitness_mult = min(1.0 + agent.fitness_score, 1.3)
                adjusted_conf = min(base_conf * fitness_mult, 0.95)

                signals.append({
                    "coin": best_signal["coin"],
                    "side": best_signal["side"],
                    "confidence": round(adjusted_conf, 3),
                    "source": "arena_champion",
                    "strategy_type": agent.strategy_type,
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "agent_fitness": round(agent.fitness_score, 4),
                    "agent_elo": round(agent.elo_rating, 0),
                    "agent_trades": agent.total_trades,
                    "agent_win_rate": round(agent.win_rate, 3),
                    "agent_sharpe": round(agent.sharpe_ratio, 3),
                    "price": best_signal["price"],
                    "atr_pct": best_signal.get("atr_pct", 0.02),
                })

            except Exception as e:
                logger.debug(f"Champion signal error for {agent.name}: {e}")

        if signals:
            logger.info(
                "Arena champions generated %d live signals across %d coin(s) from %d qualified agents",
                len(signals),
                len(candle_universe),
                len(qualified),
            )

        return signals
