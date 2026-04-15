"""
Reinforcement Learning Position Sizer
======================================
Replaces static fractional Kelly with a DQN agent that learns when to
size up (high-edge regimes) vs shrink (turbulence/uncertainty).

The RL agent observes:
  - Kelly fraction suggestion (what pure math says)
  - Win rate, reward/risk ratio
  - Current regime (one-hot encoded)
  - Recent volatility
  - Signal confidence
  - Account drawdown from peak

And outputs a sizing multiplier (0.25x to 2.0x) applied to Kelly's suggestion.

Training Environment:
  Uses the existing Monte Carlo simulator to generate realistic trade sequences.
  The agent learns a policy that maximizes risk-adjusted returns (Sharpe) over
  episodes of N trades.

Integration:
  RLSizer wraps KellySizer — it calls Kelly first, then applies the learned
  multiplier.  Falls back to pure Kelly when the RL model hasn't been trained.

Requirements:
  pip install torch  (same as LSTM agent — CPU-only is fine)
"""
import logging
import os
import threading
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy torch import
_torch = None
_nn = None
_F = None


def _ensure_torch():
    global _torch, _nn, _F
    if _torch is not None:
        return True
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        _torch = torch
        _nn = nn
        _F = F
        return True
    except ImportError:
        logger.warning("RLSizer: PyTorch not installed. pip install torch")
        return False


# ─── State & Action Spaces ───────────────────────────────────

# State vector (11 features):
#   [0] kelly_fraction        — raw Kelly suggestion (0-1)
#   [1] win_rate              — historical win rate (0-1)
#   [2] reward_risk_ratio     — avg_win/avg_loss (capped at 5)
#   [3] signal_confidence     — current signal confidence (0-1)
#   [4] regime_trending_up    — one-hot
#   [5] regime_trending_down  — one-hot
#   [6] regime_volatile       — one-hot
#   [7] regime_ranging        — one-hot
#   [8] recent_volatility     — normalized (0-1)
#   [9] drawdown_from_peak    — 0 to 1 (0=at peak, 1=100% drawdown)
#  [10] trades_used_norm      — min(trades_used/100, 1.0)
STATE_DIM = 11

# Discrete action space: sizing multipliers
# Index → multiplier applied to Kelly fraction
ACTION_MULTIPLIERS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
N_ACTIONS = len(ACTION_MULTIPLIERS)


# ─── DQN Network ─────────────────────────────────────────────

class DQNetwork:
    """Simple DQN for position sizing."""

    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS,
                 hidden: int = 64):
        if not _ensure_torch():
            raise ImportError("PyTorch required for RL sizer")

        self.device = _torch.device("cpu")
        self.n_actions = n_actions

        # Q-network
        self.q_net = _nn.Sequential(
            _nn.Linear(state_dim, hidden),
            _nn.ReLU(),
            _nn.Linear(hidden, hidden),
            _nn.ReLU(),
            _nn.Linear(hidden, n_actions),
        ).to(self.device)

        # Target network (for stable training)
        self.target_net = _nn.Sequential(
            _nn.Linear(state_dim, hidden),
            _nn.ReLU(),
            _nn.Linear(hidden, hidden),
            _nn.ReLU(),
            _nn.Linear(hidden, n_actions),
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = _torch.optim.Adam(self.q_net.parameters(), lr=5e-4)

        # Replay buffer
        self._buffer: List[Tuple] = []
        self._buffer_max = 10000

        # Training params
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.target_update_freq = 100
        self._step_count = 0
        self._trained = False

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        with _torch.no_grad():
            s = _torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(s)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self._buffer.append((state, action, reward, next_state, done))
        if len(self._buffer) > self._buffer_max:
            self._buffer.pop(0)

    def train_step(self) -> Optional[float]:
        """One training step from replay buffer."""
        if len(self._buffer) < self.batch_size:
            return None

        # Sample mini-batch
        indices = np.random.choice(len(self._buffer), self.batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]

        states = _torch.FloatTensor([b[0] for b in batch]).to(self.device)
        actions = _torch.LongTensor([b[1] for b in batch]).to(self.device)
        rewards = _torch.FloatTensor([b[2] for b in batch]).to(self.device)
        next_states = _torch.FloatTensor([b[3] for b in batch]).to(self.device)
        dones = _torch.FloatTensor([b[4] for b in batch]).to(self.device)

        # Current Q values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q values (Double DQN)
        with _torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = _F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        _torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self._step_count,
            "trained": self._trained,
        }, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            ckpt = _torch.load(path, map_location=self.device, weights_only=False)
            self.q_net.load_state_dict(ckpt["q_net"])
            self.target_net.load_state_dict(ckpt["target_net"])
            self.epsilon = ckpt.get("epsilon", self.epsilon_min)
            self._step_count = ckpt.get("step_count", 0)
            self._trained = ckpt.get("trained", True)
            return True
        except Exception as e:
            logger.warning("RLSizer model load failed: %s", e)
            return False


# ─── Training Environment ────────────────────────────────────

class SizingEnvironment:
    """
    Simulated environment for training the RL sizer.

    Uses historical trade returns to create episodes where the agent
    chooses sizing multipliers and receives rewards based on risk-adjusted PnL.
    """

    def __init__(self, trade_returns: np.ndarray, episode_length: int = 50):
        self.trade_returns = trade_returns
        self.episode_length = min(episode_length, len(trade_returns) - 1)
        self._reset()

    def _reset(self) -> np.ndarray:
        """Reset environment for new episode."""
        # Pick a random starting point (walk-forward safe)
        max_start = len(self.trade_returns) - self.episode_length - 1
        self._start = np.random.randint(0, max(1, max_start))
        self._step = 0
        self._equity = 1.0
        self._peak_equity = 1.0
        self._returns_so_far = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Construct state vector from current episode position."""
        idx = self._start + self._step
        recent = self.trade_returns[max(0, idx - 20):idx + 1]

        # Compute pseudo-Kelly from recent data
        wins = recent[recent > 0]
        losses = recent[recent < 0]
        win_rate = len(wins) / max(len(recent), 1)
        avg_win = wins.mean() if len(wins) > 0 else 0.01
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.01
        rr_ratio = min(avg_win / (avg_loss + 1e-10), 5.0)

        kelly = max(0, win_rate - (1 - win_rate) / (rr_ratio + 1e-10))
        vol = recent.std() if len(recent) > 1 else 0.02
        drawdown = max(0, (self._peak_equity - self._equity) / (self._peak_equity + 1e-10))

        # Simulate regime from recent returns trend
        trend = recent[-5:].mean() if len(recent) >= 5 else 0
        regime_up = 1.0 if trend > 0.005 else 0.0
        regime_down = 1.0 if trend < -0.005 else 0.0
        regime_volatile = 1.0 if vol > 0.03 else 0.0
        regime_ranging = 1.0 if abs(trend) < 0.002 and vol < 0.02 else 0.0

        return np.array([
            kelly,
            win_rate,
            rr_ratio / 5.0,  # normalized
            0.6,  # simulated signal confidence
            regime_up,
            regime_down,
            regime_volatile,
            regime_ranging,
            min(vol / 0.05, 1.0),  # normalized vol
            min(drawdown, 1.0),
            min(len(recent) / 100.0, 1.0),
        ], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute one trade with the chosen sizing multiplier.

        Returns:
            (next_state, reward, done)
        """
        multiplier = ACTION_MULTIPLIERS[action]
        idx = self._start + self._step
        trade_return = self.trade_returns[idx]

        # Apply sizing: position_return = base_return * multiplier
        sized_return = trade_return * multiplier
        self._equity *= (1.0 + sized_return)
        self._peak_equity = max(self._peak_equity, self._equity)
        self._returns_so_far.append(sized_return)

        self._step += 1
        done = self._step >= self.episode_length

        # Reward: risk-adjusted return (per-step Sharpe-like)
        # Positive return is good, but penalize variance
        reward = sized_return
        if len(self._returns_so_far) > 5:
            ret_arr = np.array(self._returns_so_far[-10:])
            # Sharpe component: mean / std
            sharpe_component = ret_arr.mean() / (ret_arr.std() + 1e-6)
            reward = 0.5 * sized_return + 0.5 * sharpe_component * 0.01

        # Penalize deep drawdowns
        dd = (self._peak_equity - self._equity) / (self._peak_equity + 1e-10)
        if dd > 0.15:
            reward -= dd * 0.5

        next_state = self._get_state() if not done else np.zeros(STATE_DIM)
        return next_state, float(reward), done

    def reset(self) -> np.ndarray:
        return self._reset()


# ─── RL Position Sizer (wraps KellySizer) ─────────────────────

class RLPositionSizer:
    """
    RL-augmented Kelly sizer.

    Wraps the existing KellySizer and applies a learned multiplier
    based on market conditions.  Falls back to pure Kelly when RL
    model is not trained.
    """

    def __init__(self, kelly_sizer, config: Optional[Dict] = None):
        cfg = config or {}
        self.kelly = kelly_sizer
        self.model_dir = cfg.get("model_dir", "models/rl_sizer")
        self.retrain_interval = cfg.get("retrain_interval", 43200)  # 12 hours
        self.training_episodes = cfg.get("training_episodes", 500)
        self.episode_length = cfg.get("episode_length", 50)
        self.min_training_trades = cfg.get("min_training_trades", 100)

        self._dqn: Optional[DQNetwork] = None
        self._last_train_time: float = 0
        self._train_metrics: Dict = {}
        self._initialized = False
        self._total_adjustments = 0
        self._fallback_count = 0
        # Training runs in a daemon thread so the trading cycle is never
        # blocked by the ~25k tensor ops per training run.
        self._training_thread: Optional[threading.Thread] = None

        if _ensure_torch():
            self._dqn = DQNetwork()
            model_path = os.path.join(self.model_dir, "rl_sizer.pt")
            if self._dqn.load(model_path):
                self._initialized = True
                logger.info("RLSizer: loaded pre-trained model (epsilon=%.2f)", self._dqn.epsilon)
            else:
                logger.info("RLSizer: no pre-trained model, will train on first data")
        else:
            logger.warning("RLSizer: PyTorch not available, pure Kelly fallback")

    def get_sizing(self, strategy_key: str, account_balance: float,
                   signal_confidence: float = 0.5,
                   regime: str = "unknown",
                   recent_volatility: float = 0.02,
                   drawdown_from_peak: float = 0.0):
        """
        Get position size with RL-adjusted Kelly.

        Same interface as KellySizer.get_sizing() plus extra context params.
        Returns a SizingResult (from Kelly) with position_pct adjusted by RL.
        """
        # Step 1: Get base Kelly sizing
        result = self.kelly.get_sizing(strategy_key, account_balance, signal_confidence)

        # Step 2: Apply RL multiplier if trained
        if not self._initialized or self._dqn is None:
            self._fallback_count += 1
            return result

        state = self._build_state(result, signal_confidence, regime,
                                  recent_volatility, drawdown_from_peak)
        action = self._dqn.select_action(state, explore=False)
        multiplier = ACTION_MULTIPLIERS[action]

        # Apply multiplier to position_pct
        adjusted_pct = result.position_pct * multiplier
        # Respect Kelly's caps
        adjusted_pct = max(self.kelly.min_position_pct,
                          min(adjusted_pct, self.kelly.max_position_pct))

        result.position_pct = adjusted_pct
        result.position_usd = account_balance * adjusted_pct
        self._total_adjustments += 1

        logger.debug("RLSizer [%s]: Kelly=%.1f%% x %.2f = %.1f%% (regime=%s, dd=%.1f%%)",
                    strategy_key, result.position_pct / multiplier * 100,
                    multiplier, adjusted_pct * 100, regime, drawdown_from_peak * 100)

        return result

    def _build_state(self, sizing_result, signal_confidence: float,
                     regime: str, volatility: float,
                     drawdown: float) -> np.ndarray:
        """Build state vector for the DQN."""
        regime = regime.upper()
        return np.array([
            sizing_result.kelly_fraction,
            sizing_result.win_rate,
            min(sizing_result.reward_risk_ratio / 5.0, 1.0),
            signal_confidence,
            1.0 if regime == "TRENDING_UP" else 0.0,
            1.0 if regime == "TRENDING_DOWN" else 0.0,
            1.0 if regime == "VOLATILE" else 0.0,
            1.0 if regime == "RANGING" else 0.0,
            min(volatility / 0.05, 1.0),
            min(drawdown, 1.0),
            min(sizing_result.trades_used / 100.0, 1.0),
        ], dtype=np.float32)

    def train(self, trade_returns: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        Kick off RL sizer training in a background thread.

        Returns None immediately (never blocks the trading cycle). The
        daemon thread logs its own completion metrics. If a training run
        is already in flight, this is a no-op.

        Args:
            trade_returns: Array of per-trade return fractions. If None,
                the background thread will extract from Kelly's internal
                data at the time it runs.
        """
        if self._dqn is None:
            return None

        # Interval gate — avoid re-training too often even if called frequently.
        now = time.time()
        if self._initialized and (now - self._last_train_time) < self.retrain_interval:
            return None

        # Re-entrancy guard: if a previous training thread is still running,
        # skip. retrain_interval+alive-check together cap concurrency at 1.
        if self._training_thread is not None and self._training_thread.is_alive():
            return None

        # Mark the start time now so rapid back-to-back calls don't all spawn
        # threads before the first one updates _last_train_time on completion.
        self._last_train_time = now

        self._training_thread = threading.Thread(
            target=self._train_blocking,
            args=(trade_returns,),
            daemon=True,
            name="RLSizer-train",
        )
        self._training_thread.start()
        return None

    def _train_blocking(self, trade_returns: Optional[np.ndarray]) -> None:
        """Synchronous training body — runs inside the daemon thread."""
        try:
            # Extract returns from Kelly's data if not provided
            if trade_returns is None:
                all_returns = []
                for outcomes in self.kelly._strategy_outcomes.values():
                    all_returns.extend([o["return_pct"] for o in outcomes])
                if len(all_returns) < self.min_training_trades:
                    logger.info(
                        "RLSizer: not enough trade data (%d < %d), training skipped",
                        len(all_returns), self.min_training_trades,
                    )
                    return
                trade_returns = np.array(all_returns, dtype=np.float32)

            logger.info("RLSizer: training on %d trade returns, %d episodes (background)...",
                        len(trade_returns), self.training_episodes)

            env = SizingEnvironment(trade_returns, self.episode_length)
            total_rewards = []
            losses = []

            for ep in range(self.training_episodes):
                state = env.reset()
                ep_reward = 0

                for _ in range(env.episode_length):
                    action = self._dqn.select_action(state, explore=True)
                    next_state, reward, done = env.step(action)

                    self._dqn.store_transition(state, action, reward, next_state, float(done))
                    loss = self._dqn.train_step()
                    if loss is not None:
                        losses.append(loss)

                    state = next_state
                    ep_reward += reward

                    if done:
                        break

                total_rewards.append(ep_reward)

            self._dqn._trained = True
            self._initialized = True

            # Save model
            model_path = os.path.join(self.model_dir, "rl_sizer.pt")
            self._dqn.save(model_path)

            metrics = {
                "episodes": self.training_episodes,
                "mean_reward": float(np.mean(total_rewards[-50:])),
                "mean_loss": float(np.mean(losses[-100:])) if losses else 0,
                "epsilon": self._dqn.epsilon,
                "trade_returns_used": len(trade_returns),
            }
            self._train_metrics = metrics

            logger.info(
                "RLSizer trained: mean_reward=%.4f, epsilon=%.3f, trades=%d",
                metrics["mean_reward"], metrics["epsilon"], len(trade_returns),
            )
        except Exception as exc:  # noqa: BLE001 — background thread must never escape
            logger.warning("RLSizer background training failed: %s", exc)

    def get_stats(self) -> Dict:
        return {
            "initialized": self._initialized,
            "model_available": self._dqn is not None,
            "total_adjustments": self._total_adjustments,
            "fallback_count": self._fallback_count,
            "epsilon": self._dqn.epsilon if self._dqn else None,
            "train_metrics": self._train_metrics,
        }
