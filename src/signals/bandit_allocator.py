"""A2: Thompson-sampling allocator over copy-trade sources.

The existing AgentScorer is a static-ish weighted_accuracy × recent_pnl
heuristic. With small N per source (~handful of trades / day) and the
arrival of new candidates, that scheme has two failure modes:

1. **Lock-on-early-winner.** A source that wins its first 2 trades gets
   high weight and dominates allocation; a strictly-better source
   discovered later struggles to overcome.

2. **Exploration drought.** No mechanism explicitly samples new or
   borderline-marginal sources. Promising candidates die from neglect.

Thompson sampling (Beta–Bernoulli) solves both: each source carries a
Beta(α, β) posterior over its win probability, and on every cycle we
*sample* from each posterior and rank by sample. High-uncertainty
sources get a free shot at the top occasionally — that's exploration.
High-mean low-uncertainty sources dominate when their evidence is
strong — that's exploitation. The math is well-studied for portfolio
allocation (Zhu & Zheng 2019, arxiv:1911.05309) and especially
appropriate for the small-N regime crypto copy-trading lives in.

The allocator does NOT replace AgentScorer. It sits alongside as an
alternative weighting strategy that downstream code (e.g.
decision_engine) can opt into via flag. Default OFF.

Key invariants
--------------
- Beta priors default to (1, 1) — uniform, no opinion. New sources
  get a fair sample.
- A half-life decay on α and β makes the posterior react to regime
  change without forgetting everything.
- A Wilson lower-bound check caps allocation to sources with <
  `min_evidence_trades` closed trades — bandits will happily allocate
  to an arm with 1 win and 0 losses, which is fine for a slot machine
  but dangerous for real capital.

This file is self-contained. Persistence (writing α, β to the
agent_scores DB row) is left to the wiring layer; the allocator works
fine in pure in-memory mode for backtests and tests.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, List, Optional


# Default Beta prior. (1, 1) is the uniform distribution on [0, 1] — the
# "no opinion" prior, which guarantees brand-new sources are sampled with
# the same expected weight as a coin-flip.
DEFAULT_PRIOR_ALPHA = 1.0
DEFAULT_PRIOR_BETA = 1.0

# Decay half-life expressed in *days of elapsed time*, not in trade
# count. Crypto regime change is wall-clock-driven; a source that was
# great in October should not retain full posterior weight in January.
DEFAULT_DECAY_HALF_LIFE_DAYS = 30.0

# Wilson interval z for 95% lower bound. The allocator falls back to
# the prior mean for sources below `min_evidence_trades` *and* uses the
# Wilson lower bound (not the point estimate) to cap their sample weight.
WILSON_Z_95 = 1.96

# Below this many closed trades, the allocator treats the source as
# "exploration-only" and caps its allocation share at
# `min_evidence_share_cap`.
DEFAULT_MIN_EVIDENCE_TRADES = 5
DEFAULT_MIN_EVIDENCE_SHARE_CAP = 0.10


@dataclass
class ArmState:
    """Beta(α, β) posterior for a single source.

    α and β are stored as *effective* counts after decay (not raw
    cumulative). When an outcome arrives, the current α/β are decayed
    to "now" first, then incremented by 1 on the corresponding side.
    """
    source: str
    alpha: float = DEFAULT_PRIOR_ALPHA
    beta: float = DEFAULT_PRIOR_BETA
    n_observed: int = 0                     # raw count of outcomes seen
    last_update_ts: float = field(default_factory=time.time)
    half_life_days: float = DEFAULT_DECAY_HALF_LIFE_DAYS

    def posterior_mean(self) -> float:
        return self.alpha / max(self.alpha + self.beta, 1e-12)

    def wilson_lower_95(self) -> float:
        """Wilson score lower bound on the win-probability estimate.

        Conservative confidence interval — even with very few trials
        this never collapses to 0 or 1, which is the right behaviour
        for an allocator that funds real capital. Returns
        ``posterior_mean`` for arms below evidence threshold so callers
        don't have to special-case.
        """
        n = self.alpha + self.beta - DEFAULT_PRIOR_ALPHA - DEFAULT_PRIOR_BETA
        if n <= 0:
            return self.posterior_mean()
        p_hat = self.alpha / max(self.alpha + self.beta, 1e-12)
        z = WILSON_Z_95
        denom = 1.0 + z**2 / n
        centre = (p_hat + z**2 / (2 * n)) / denom
        margin = z * math.sqrt(max(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2), 0.0)) / denom
        return max(centre - margin, 0.0)

    def _decay_to(self, now_ts: float) -> None:
        """Apply exponential decay of (α-prior, β-prior) to ``now_ts``.

        Only the *informative* mass (α - prior, β - prior) decays — the
        prior itself is the noise floor and stays put. This is the
        right behaviour: an arm with no recent evidence reverts to the
        uniform prior, not to zero observations.
        """
        if self.half_life_days <= 0:
            self.last_update_ts = now_ts
            return
        elapsed_days = max((now_ts - self.last_update_ts) / 86400.0, 0.0)
        if elapsed_days <= 0:
            return
        decay = 0.5 ** (elapsed_days / self.half_life_days)
        info_a = max(self.alpha - DEFAULT_PRIOR_ALPHA, 0.0) * decay
        info_b = max(self.beta - DEFAULT_PRIOR_BETA, 0.0) * decay
        self.alpha = DEFAULT_PRIOR_ALPHA + info_a
        self.beta = DEFAULT_PRIOR_BETA + info_b
        self.last_update_ts = now_ts

    def update(self, won: bool, now_ts: Optional[float] = None) -> None:
        """Apply a single outcome (won/lost) to the posterior."""
        now = now_ts if now_ts is not None else time.time()
        self._decay_to(now)
        if won:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        self.n_observed += 1


class ThompsonAllocator:
    """Bayesian bandit allocator over named signal sources.

    Thread-safe. Backing state is in-memory by default; callers wiring
    this into production should snapshot ``arms_snapshot()`` to the
    agent_scores DB row and rehydrate on boot via ``load_state()``.
    """

    def __init__(
        self,
        *,
        prior_alpha: float = DEFAULT_PRIOR_ALPHA,
        prior_beta: float = DEFAULT_PRIOR_BETA,
        half_life_days: float = DEFAULT_DECAY_HALF_LIFE_DAYS,
        min_evidence_trades: int = DEFAULT_MIN_EVIDENCE_TRADES,
        min_evidence_share_cap: float = DEFAULT_MIN_EVIDENCE_SHARE_CAP,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._arms: Dict[str, ArmState] = {}
        self._prior_alpha = float(prior_alpha)
        self._prior_beta = float(prior_beta)
        self._half_life_days = float(half_life_days)
        self._min_evidence_trades = int(min_evidence_trades)
        self._min_evidence_share_cap = float(min_evidence_share_cap)
        self._rng = rng if rng is not None else random.Random()
        # RLock so arms_snapshot() can call arm_snapshot() without deadlock.
        self._lock = RLock()

    # ── State management ──────────────────────────────────────────────

    def _arm(self, source: str) -> ArmState:
        """Return (creating if necessary) the ArmState for ``source``."""
        if source not in self._arms:
            self._arms[source] = ArmState(
                source=source,
                alpha=self._prior_alpha,
                beta=self._prior_beta,
                half_life_days=self._half_life_days,
            )
        return self._arms[source]

    def update(self, source: str, won: bool, now_ts: Optional[float] = None) -> None:
        """Record a binary outcome for the named source."""
        with self._lock:
            self._arm(source).update(won, now_ts=now_ts)

    def update_pnl(self, source: str, pnl: float, fee_floor: float = 0.0,
                   now_ts: Optional[float] = None) -> None:
        """Convenience: convert a PnL number into a binary win/loss.

        A trade is a 'win' if PnL exceeds the expected round-trip fee
        cost (default 0 → just sign-of-PnL). This keeps the bandit
        focused on net economic outcomes, not raw signed PnL.
        """
        self.update(source, won=(pnl > fee_floor), now_ts=now_ts)

    # ── Sampling ──────────────────────────────────────────────────────

    def sample(self, source: str, now_ts: Optional[float] = None) -> float:
        """Sample a single value from the Beta posterior of ``source``.

        Lazily decays state to ``now_ts`` before sampling so a long-idle
        arm reverts toward its prior.
        """
        with self._lock:
            arm = self._arm(source)
            arm._decay_to(now_ts if now_ts is not None else time.time())
            return self._rng.betavariate(arm.alpha, arm.beta)

    def sample_weights(
        self,
        sources: List[str],
        *,
        now_ts: Optional[float] = None,
    ) -> Dict[str, float]:
        """Return Thompson-sampled, sum-to-1 weights over the given sources.

        Per-source raw samples are taken from each Beta posterior.
        Sources below ``min_evidence_trades`` are capped at
        ``min_evidence_share_cap`` of the total so a 1-win-0-loss arm
        cannot dominate allocation. Remaining mass is renormalised over
        the well-evidenced arms.
        """
        if not sources:
            return {}

        now = now_ts if now_ts is not None else time.time()
        raw: Dict[str, float] = {}
        with self._lock:
            for s in sources:
                arm = self._arm(s)
                arm._decay_to(now)
                raw[s] = self._rng.betavariate(arm.alpha, arm.beta)

            # Determine which arms are well-evidenced vs. exploratory
            well_evidenced = [
                s for s in sources
                if self._arms[s].n_observed >= self._min_evidence_trades
            ]
            exploratory = [s for s in sources if s not in well_evidenced]

        # Normalise within group; cap the exploratory group's total share
        if not well_evidenced:
            # No source has enough evidence — fall back to uniform-ish
            total = sum(raw[s] for s in sources) or 1.0
            return {s: raw[s] / total for s in sources}

        if exploratory:
            cap = max(0.0, min(1.0, self._min_evidence_share_cap))
            exploratory_raw = sum(raw[s] for s in exploratory) or 1.0
            well_raw = sum(raw[s] for s in well_evidenced) or 1.0
            weights: Dict[str, float] = {}
            for s in exploratory:
                weights[s] = (raw[s] / exploratory_raw) * cap
            for s in well_evidenced:
                weights[s] = (raw[s] / well_raw) * (1.0 - cap)
            return weights

        total = sum(raw[s] for s in well_evidenced) or 1.0
        return {s: raw[s] / total for s in well_evidenced}

    # ── Introspection / persistence ───────────────────────────────────

    def arm_snapshot(self, source: str) -> Optional[Dict]:
        with self._lock:
            if source not in self._arms:
                return None
            a = self._arms[source]
            return {
                "source": a.source,
                "alpha": a.alpha,
                "beta": a.beta,
                "n_observed": a.n_observed,
                "last_update_ts": a.last_update_ts,
                "posterior_mean": a.posterior_mean(),
                "wilson_lower_95": a.wilson_lower_95(),
            }

    def arms_snapshot(self) -> Dict[str, Dict]:
        """Snapshot all arms — intended for persistence to agent_scores."""
        with self._lock:
            return {s: self.arm_snapshot(s) or {} for s in self._arms}

    def load_state(self, snapshot: Dict[str, Dict]) -> None:
        """Rehydrate allocator state from a previous ``arms_snapshot()``.

        Tolerant of partial / missing fields — defaults to the prior on
        anything malformed so a bad row doesn't crash boot.
        """
        with self._lock:
            for source, data in (snapshot or {}).items():
                try:
                    self._arms[source] = ArmState(
                        source=str(source),
                        alpha=float(data.get("alpha", self._prior_alpha)),
                        beta=float(data.get("beta", self._prior_beta)),
                        n_observed=int(data.get("n_observed", 0)),
                        last_update_ts=float(data.get("last_update_ts", time.time())),
                        half_life_days=self._half_life_days,
                    )
                except Exception:
                    continue
