"""
Agent / Signal Source Scoring System
=====================================
Tracks prediction accuracy per signal source and dynamically weights them.

Every signal source (strategy types, copy traders, options flow) gets scored
based on its historical accuracy. Better sources get higher weight,
bad sources fade out via time decay.

This is what turns the system from "equal voting" into "meritocratic voting."
"""
import logging
import math
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict

from src.data import database as db

logger = logging.getLogger(__name__)

# Time decay: how quickly old performance fades
DECAY_HALF_LIFE_TRADES = 50   # After 50 trades, weight of old data is halved


@dataclass
class SourceScore:
    """Performance score for a signal source."""
    source_key: str          # e.g. "strategy:momentum_long", "copy:0xabc...", "options_flow"
    total_signals: int = 0
    correct_signals: int = 0
    total_pnl: float = 0.0
    total_return: float = 0.0
    accuracy: float = 0.0    # Correct / total
    sharpe: float = 0.0
    avg_return: float = 0.0
    weighted_accuracy: float = 0.0  # Time-decay weighted
    dynamic_weight: float = 0.5     # Current weight (0-1)
    last_updated: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class AgentScorer:
    """
    Tracks and scores signal sources based on historical outcomes.

    Usage:
    1. When a signal is generated: scorer.record_signal(source_key, signal_data)
    2. When a trade closes: scorer.record_outcome(signal_id, pnl, return_pct)
    3. Before execution: weight = scorer.get_weight(source_key)
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        # In-memory score tracking
        self.scores: Dict[str, SourceScore] = {}

        # Trade-level tracking for time decay
        self._trade_history: Dict[str, List[Dict]] = defaultdict(list)

        # Source policy thresholds. These gate weak sources before they keep
        # consuming paper/live capacity.
        self.policy_enabled = bool(cfg.get("policy_enabled", True))
        self.policy_min_closed_trades = int(cfg.get("policy_min_closed_trades", 3))
        self.policy_keep_top_n = int(cfg.get("policy_keep_top_n", 5))
        self.policy_pause_weight = float(cfg.get("policy_pause_weight", 0.12))
        self.policy_degrade_weight = float(cfg.get("policy_degrade_weight", 0.32))
        self.policy_warmup_max_signals_per_day = int(
            cfg.get("policy_warmup_max_signals_per_day", 1)
        )
        self.policy_degraded_max_signals_per_day = int(
            cfg.get("policy_degraded_max_signals_per_day", 1)
        )
        self.policy_warmup_size_multiplier = float(
            cfg.get("policy_warmup_size_multiplier", 0.75)
        )
        self.policy_degraded_size_multiplier = float(
            cfg.get("policy_degraded_size_multiplier", 0.60)
        )
        self.policy_warmup_min_confidence = float(
            cfg.get("policy_warmup_min_confidence", 0.45)
        )
        self.policy_degraded_min_confidence = float(
            cfg.get("policy_degraded_min_confidence", 0.55)
        )

        # Load existing scores from DB
        self._load_scores()

        logger.info(f"AgentScorer initialized with {len(self.scores)} tracked sources")

    def _load_scores(self):
        """Load existing scores from database.

        LOW-FIX LOW-7: use db.get_connection() instead of raw sqlite3.connect
        so all writes share the same WAL journal and avoid checkpoint stalls
        when concurrent writes are in flight from the main trading loop.
        """
        try:
            with db.get_connection() as conn:
                # Create table if not exists
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_scores (
                        source_key TEXT PRIMARY KEY,
                        total_signals INTEGER DEFAULT 0,
                        correct_signals INTEGER DEFAULT 0,
                        total_pnl REAL DEFAULT 0,
                        total_return REAL DEFAULT 0,
                        accuracy REAL DEFAULT 0,
                        sharpe REAL DEFAULT 0,
                        dynamic_weight REAL DEFAULT 0.5,
                        trade_history TEXT DEFAULT '[]',
                        last_updated TEXT
                    )
                """)
                # Safe migration: add total_return column if missing
                try:
                    conn.execute("ALTER TABLE agent_scores ADD COLUMN total_return REAL DEFAULT 0")
                except Exception:
                    pass  # Column already exists

                rows = conn.execute("SELECT * FROM agent_scores").fetchall()
                for row in rows:
                    row = dict(row)
                    score = SourceScore(
                        source_key=row["source_key"],
                        total_signals=row["total_signals"],
                        correct_signals=row["correct_signals"],
                        total_pnl=row["total_pnl"],
                        total_return=row.get("total_return", 0) or 0,
                        accuracy=row["accuracy"],
                        sharpe=row["sharpe"],
                        dynamic_weight=row["dynamic_weight"],
                        last_updated=row["last_updated"] or "",
                    )
                    self.scores[row["source_key"]] = score

                    # Load trade history for time decay
                    try:
                        history = json.loads(row.get("trade_history", "[]"))
                        self._trade_history[row["source_key"]] = history[-200:]
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Could not load agent scores: {e}")

    def _save_score(self, source_key: str):
        """Persist a single source's score to DB."""
        try:
            score = self.scores.get(source_key)
            if not score:
                return

            history_json = json.dumps(self._trade_history.get(source_key, [])[-200:])

            # LOW-FIX LOW-7 (continued): use shared WAL connection pool
            with db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO agent_scores
                    (source_key, total_signals, correct_signals, total_pnl,
                     total_return, accuracy, sharpe, dynamic_weight, trade_history, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (source_key) DO UPDATE SET
                        total_signals = EXCLUDED.total_signals,
                        correct_signals = EXCLUDED.correct_signals,
                        total_pnl = EXCLUDED.total_pnl,
                        total_return = EXCLUDED.total_return,
                        accuracy = EXCLUDED.accuracy,
                        sharpe = EXCLUDED.sharpe,
                        dynamic_weight = EXCLUDED.dynamic_weight,
                        trade_history = EXCLUDED.trade_history,
                        last_updated = EXCLUDED.last_updated
                """, (source_key, score.total_signals, score.correct_signals,
                      score.total_pnl, score.total_return, score.accuracy, score.sharpe,
                      score.dynamic_weight, history_json,
                      datetime.now(timezone.utc).isoformat()))
        except Exception as e:
            logger.warning(f"Could not save agent score for {source_key}: {e}")

    # ─── Recording ────────────────────────────────────────────

    def record_signal(self, source_key: str, signal_data: Dict) -> str:
        """
        Record that a signal was generated. Returns a signal_id for tracking.
        Call this when a signal is about to be executed.
        """
        if source_key not in self.scores:
            self.scores[source_key] = SourceScore(source_key=source_key)

        score = self.scores[source_key]
        score.total_signals += 1
        score.last_updated = datetime.now(timezone.utc).isoformat()

        signal_id = f"{source_key}:{score.total_signals}:{int(time.time())}"

        self._trade_history[source_key].append({
            "signal_id": signal_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "coin": signal_data.get("coin", ""),
            "side": signal_data.get("side", ""),
            "confidence": signal_data.get("confidence", 0),
            "pnl": None,  # Filled when outcome is recorded
            "correct": None,
        })

        return signal_id

    def record_outcome(self, source_key: str, signal_id: str,
                        pnl: float, return_pct: float = 0.0):
        """
        Record the outcome of a trade.
        Call this when a paper trade closes.
        """
        if source_key not in self.scores:
            self.scores[source_key] = SourceScore(source_key=source_key)

        score = self.scores[source_key]
        correct = pnl > 0

        # Update running totals
        if correct:
            score.correct_signals += 1
        score.total_pnl += pnl
        score.total_return += return_pct

        # Update trade history
        history = self._trade_history[source_key]
        for entry in reversed(history):
            if entry["signal_id"] == signal_id:
                entry["pnl"] = pnl
                entry["correct"] = correct
                entry["return_pct"] = return_pct
                break

        # Recalculate scores with time decay
        self._recalculate(source_key)
        self._save_score(source_key)

        logger.info(f"Agent score update [{source_key}]: pnl=${pnl:.2f}, "
                    f"accuracy={score.accuracy:.0%}, weight={score.dynamic_weight:.2f}")

    # ─── Scoring ──────────────────────────────────────────────

    def _recalculate(self, source_key: str):
        """Recalculate scores with exponential time decay."""
        score = self.scores.get(source_key)
        if not score:
            return

        history = self._trade_history.get(source_key, [])
        completed = [t for t in history if t.get("pnl") is not None]

        if not completed:
            score.accuracy = 0.0
            score.dynamic_weight = 0.5
            return

        # Raw accuracy
        total = len(completed)
        correct = sum(1 for t in completed if t.get("correct"))
        score.accuracy = correct / total if total > 0 else 0

        # Time-decay weighted accuracy
        # More recent trades have exponentially higher weight
        weighted_correct = 0.0
        weighted_total = 0.0
        for i, trade in enumerate(completed):
            # Weight: more recent = higher (exponential decay)
            age = total - i - 1  # 0 = most recent
            weight = math.exp(-0.693 * age / DECAY_HALF_LIFE_TRADES)  # 0.693 = ln(2)
            weighted_total += weight
            if trade.get("correct"):
                weighted_correct += weight

        score.weighted_accuracy = weighted_correct / weighted_total if weighted_total > 0 else 0

        # Sharpe-like score: mean return / std return
        returns = [t.get("return_pct", 0) for t in completed if t.get("return_pct") is not None]
        if len(returns) >= 5:
            import numpy as np
            mean_ret = np.mean(returns)
            # MED-FIX MED-6: use ddof=1 (sample std) to match statistics.stdev
            # used by shadow_tracker.compute_sharpe_proxy — previously population
            # std (ddof=0) underestimated variance vs sample std, making Sharpe
            # values incomparable across the two subsystems.
            std_ret = np.std(returns, ddof=1)
            score.sharpe = mean_ret / std_ret if std_ret > 0 else 0
            score.avg_return = mean_ret
        else:
            score.sharpe = 0
            score.avg_return = sum(returns) / len(returns) if returns else 0

        # Dynamic weight: blend of accuracy and recency
        # Formula: 0.5 * weighted_accuracy + 0.3 * raw_accuracy + 0.2 * sharpe_normalized
        sharpe_norm = min(max(score.sharpe / 2.0, 0), 1)  # Normalize sharpe to 0-1
        score.dynamic_weight = (
            0.5 * score.weighted_accuracy +
            0.3 * score.accuracy +
            0.2 * sharpe_norm
        )
        score.dynamic_weight = max(0.05, min(score.dynamic_weight, 1.0))  # Clamp to 0.05-1.0

    # ─── Query ────────────────────────────────────────────────

    def get_weight(self, source_key: str) -> float:
        """Get the current dynamic weight for a signal source."""
        score = self.scores.get(source_key)
        if not score or score.total_signals < 5:
            return 0.5  # Default weight for new/unknown sources
        return score.dynamic_weight

    def get_accuracy(self, source_key: str) -> float:
        """Get the time-decay weighted accuracy for a source."""
        score = self.scores.get(source_key)
        if not score:
            return 0.0
        return score.weighted_accuracy

    def get_source_key(self, signal) -> str:
        """Generate a source_key from a TradeSignal."""
        if hasattr(signal, 'source'):
            source = signal.source.value if hasattr(signal.source, 'value') else str(signal.source)
        else:
            source = signal.get("source", "unknown")
        source = str(source or "unknown").strip().lower() or "unknown"

        stype = ""
        if hasattr(signal, 'strategy_type'):
            stype = signal.strategy_type
        elif isinstance(signal, dict):
            stype = signal.get("strategy_type", "")

        trader_address = ""
        if hasattr(signal, "trader_address"):
            trader_address = getattr(signal, "trader_address", "")
        elif isinstance(signal, dict):
            trader_address = signal.get("trader_address", "") or signal.get("source_trader", "")
        trader_address = str(trader_address or "").strip().lower()

        if source == "copy_trade":
            if trader_address:
                return f"{source}:{trader_address}"
            return source
        if stype:
            return f"{source}:{stype}"
        return source

    def get_all_scores(self) -> List[Dict]:
        """Get scores for all tracked sources, sorted by weight."""
        result = []
        scorecard_map = {row["source_key"]: row for row in self.get_scorecard()}
        for key, score in self.scores.items():
            source_row = scorecard_map.get(key, {})
            result.append({
                "source_key": key,
                "total_signals": score.total_signals,
                "accuracy": round(score.accuracy, 3),
                "weighted_accuracy": round(score.weighted_accuracy, 3),
                "sharpe": round(score.sharpe, 3),
                "dynamic_weight": round(score.dynamic_weight, 3),
                "total_pnl": round(score.total_pnl, 2),
                "completed_trades": int(source_row.get("completed_trades", 0)),
                "status": source_row.get("status", "unknown"),
                "recent_pnl": round(float(source_row.get("recent_pnl", 0.0) or 0.0), 2),
            })
        result.sort(key=lambda x: x["dynamic_weight"], reverse=True)
        return result

    def _completed_history(self, source_key: str) -> List[Dict]:
        history = self._trade_history.get(source_key, []) or []
        return [entry for entry in history if entry.get("pnl") is not None]

    def _eligible_ranks(self) -> Dict[str, int]:
        eligible = []
        for source_key in set(self.scores) | set(self._trade_history):
            completed = self._completed_history(source_key)
            if len(completed) < self.policy_min_closed_trades:
                continue
            score = self.scores.get(source_key) or SourceScore(source_key=source_key)
            eligible.append((source_key, score.dynamic_weight, score.total_pnl, len(completed)))
        eligible.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
        return {source_key: rank + 1 for rank, (source_key, *_rest) in enumerate(eligible)}

    def get_scorecard(self) -> List[Dict]:
        """Return per-source scorecards used by the allocator/dashboard."""
        rank_map = self._eligible_ranks()
        rows = []
        for source_key in set(self.scores) | set(self._trade_history):
            score = self.scores.get(source_key) or SourceScore(source_key=source_key)
            completed = self._completed_history(source_key)
            completed_count = len(completed)
            wins = sum(1 for item in completed if item.get("correct"))
            recent = completed[-10:]
            recent_pnl = sum(float(item.get("pnl", 0.0) or 0.0) for item in recent)
            recent_avg_return = (
                sum(float(item.get("return_pct", 0.0) or 0.0) for item in recent) / len(recent)
                if recent
                else 0.0
            )
            last_trade_at = completed[-1].get("timestamp") if completed else score.last_updated
            win_rate = wins / completed_count if completed_count > 0 else 0.0
            rank = rank_map.get(source_key)

            if not self.policy_enabled:
                status = "active"
            elif completed_count < self.policy_min_closed_trades:
                status = "warmup"
            elif (
                score.dynamic_weight <= self.policy_pause_weight
                or (
                    completed_count >= max(self.policy_min_closed_trades, 5)
                    and recent_pnl < 0
                    and score.weighted_accuracy <= (self.policy_pause_weight + 0.08)
                )
            ):
                status = "paused"
            elif (
                score.dynamic_weight <= self.policy_degrade_weight
                or (rank is not None and rank > self.policy_keep_top_n)
                or (completed_count >= max(self.policy_min_closed_trades, 3) and recent_pnl < 0)
            ):
                status = "degraded"
            else:
                status = "active"

            rows.append(
                {
                    "source_key": source_key,
                    "rank": rank,
                    "status": status,
                    "completed_trades": completed_count,
                    "win_rate": round(win_rate, 3),
                    "accuracy": round(score.accuracy, 3),
                    "weighted_accuracy": round(score.weighted_accuracy, 3),
                    "dynamic_weight": round(score.dynamic_weight, 3),
                    "sharpe": round(score.sharpe, 3),
                    "avg_return": round(score.avg_return, 4),
                    "recent_avg_return": round(recent_avg_return, 4),
                    "total_pnl": round(score.total_pnl, 2),
                    "recent_pnl": round(recent_pnl, 2),
                    "last_trade_at": last_trade_at,
                }
            )
        rows.sort(
            key=lambda row: (
                {"active": 0, "warmup": 1, "degraded": 2, "paused": 3}.get(row["status"], 9),
                row["rank"] if row["rank"] is not None else 999999,
                -row["dynamic_weight"],
            )
        )
        return rows

    def get_source_policy(self, source_key: str) -> Dict:
        """Return the current allocator policy for a source."""
        default_policy = {
            "source_key": source_key,
            "status": "unknown",
            "rank": None,
            "blocked": False,
            "max_signals_per_day": 0,
            "size_multiplier": 1.0,
            "min_confidence": 0.0,
            "dynamic_weight": round(self.get_weight(source_key), 3),
            "weighted_accuracy": round(self.get_accuracy(source_key), 3),
            "completed_trades": len(self._completed_history(source_key)),
            "recent_pnl": 0.0,
        }
        if not self.policy_enabled:
            default_policy["status"] = "active"
            return default_policy

        row = next((item for item in self.get_scorecard() if item["source_key"] == source_key), None)
        if not row:
            return default_policy

        policy = {
            **default_policy,
            "status": row["status"],
            "rank": row["rank"],
            "dynamic_weight": row["dynamic_weight"],
            "weighted_accuracy": row["weighted_accuracy"],
            "completed_trades": row["completed_trades"],
            "recent_pnl": row["recent_pnl"],
        }
        if row["status"] == "paused":
            policy.update(
                {
                    "blocked": True,
                    "max_signals_per_day": 0,
                    "size_multiplier": 0.0,
                    "min_confidence": 1.0,
                }
            )
        elif row["status"] == "warmup":
            policy.update(
                {
                    "max_signals_per_day": self.policy_warmup_max_signals_per_day,
                    "size_multiplier": self.policy_warmup_size_multiplier,
                    "min_confidence": self.policy_warmup_min_confidence,
                }
            )
        elif row["status"] == "degraded":
            policy.update(
                {
                    "max_signals_per_day": self.policy_degraded_max_signals_per_day,
                    "size_multiplier": self.policy_degraded_size_multiplier,
                    "min_confidence": self.policy_degraded_min_confidence,
                }
            )
        else:
            policy["status"] = "active"
        return policy

    def apply_weights_to_signals(self, signals: List) -> List:
        """
        Adjust signal confidence by source weight.
        Higher-performing sources get boosted, lower ones reduced.
        """
        adjusted = []
        for signal in signals:
            source_key = self.get_source_key(signal)
            weight = self.get_weight(source_key)

            if hasattr(signal, 'confidence'):
                # Blend original confidence with source weight
                original = signal.confidence
                signal.confidence = original * 0.6 + weight * 0.4
                signal.source_accuracy = self.get_accuracy(source_key)
                logger.debug(f"Weight adjust [{source_key}]: {original:.2f} → {signal.confidence:.2f} "
                            f"(weight={weight:.2f})")
            elif isinstance(signal, dict):
                original = signal.get("confidence", 0.5)
                signal["confidence"] = original * 0.6 + weight * 0.4
                signal["source_accuracy"] = self.get_accuracy(source_key)

            adjusted.append(signal)

        return adjusted
