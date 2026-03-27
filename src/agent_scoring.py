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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict

from src import database as db
from src.signal_schema import SignalSource

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

    def __init__(self):
        # In-memory score tracking
        self.scores: Dict[str, SourceScore] = {}

        # Trade-level tracking for time decay
        self._trade_history: Dict[str, List[Dict]] = defaultdict(list)

        # Load existing scores from DB
        self._load_scores()

        logger.info(f"AgentScorer initialized with {len(self.scores)} tracked sources")

    def _load_scores(self):
        """Load existing scores from database."""
        try:
            # Check if agent_scores table exists
            import sqlite3
            import config
            conn = sqlite3.connect(config.DB_PATH)
            conn.row_factory = sqlite3.Row

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
            conn.commit()

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
                    self._trade_history[row["source_key"]] = history[-200:]  # Keep last 200
                except Exception:
                    pass

            conn.close()
        except Exception as e:
            logger.warning(f"Could not load agent scores: {e}")

    def _save_score(self, source_key: str):
        """Persist a single source's score to DB."""
        try:
            import sqlite3
            import config
            score = self.scores.get(source_key)
            if not score:
                return

            history_json = json.dumps(self._trade_history.get(source_key, [])[-200:])

            conn = sqlite3.connect(config.DB_PATH)
            conn.execute("""
                INSERT OR REPLACE INTO agent_scores
                (source_key, total_signals, correct_signals, total_pnl,
                 total_return, accuracy, sharpe, dynamic_weight, trade_history, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (source_key, score.total_signals, score.correct_signals,
                  score.total_pnl, score.total_return, score.accuracy, score.sharpe,
                  score.dynamic_weight, history_json,
                  datetime.utcnow().isoformat()))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Could not save agent score for {source_key}: {e}")

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
        score.last_updated = datetime.utcnow().isoformat()

        signal_id = f"{source_key}:{score.total_signals}:{int(time.time())}"

        self._trade_history[source_key].append({
            "signal_id": signal_id,
            "timestamp": datetime.utcnow().isoformat(),
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
            std_ret = np.std(returns)
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

        stype = ""
        if hasattr(signal, 'strategy_type'):
            stype = signal.strategy_type
        elif isinstance(signal, dict):
            stype = signal.get("strategy_type", "")

        if stype:
            return f"{source}:{stype}"
        return source

    def get_all_scores(self) -> List[Dict]:
        """Get scores for all tracked sources, sorted by weight."""
        result = []
        for key, score in self.scores.items():
            result.append({
                "source_key": key,
                "total_signals": score.total_signals,
                "accuracy": round(score.accuracy, 3),
                "weighted_accuracy": round(score.weighted_accuracy, 3),
                "sharpe": round(score.sharpe, 3),
                "dynamic_weight": round(score.dynamic_weight, 3),
                "total_pnl": round(score.total_pnl, 2),
            })
        result.sort(key=lambda x: x["dynamic_weight"], reverse=True)
        return result

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
