"""ML deployment discipline: purged walk-forward + beat-the-baseline OOS gate
(signal #8, PR-C).

A model (XGBoost regime forecaster, alpha pipeline) should only be trusted if it
demonstrably beats a naive baseline OUT OF SAMPLE -- otherwise it's noise dressed
as signal. This provides:

  * purged_walk_forward_splits -- expanding-window train/test splits with an
    embargo gap so labels that span a horizon can't leak from train into test.
  * majority_baseline_accuracy / oos_beats_baseline -- compare a model's OOS
    accuracy to the majority-class baseline, with a required margin.

Pure + dependency-free so the gate is unit-testable without the model. The
forecaster consults oos_beats_baseline behind ML_REQUIRE_OOS_BEATS_BASELINE and
falls back to neutral when the model fails to clear the bar.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Sequence, Tuple


def purged_walk_forward_splits(n: int, n_splits: int = 4, embargo: int = 0
                               ) -> List[Tuple[List[int], List[int]]]:
    """Expanding-window walk-forward splits over ``n`` time-ordered samples.

    Fold ``i`` trains on ``[0, test_start - embargo)`` and tests on
    ``[test_start, test_start + fold)``. The ``embargo`` gap is purged between
    train and test so a label whose outcome realizes over the next few bars
    can't leak backward into training. Returns only non-empty splits.
    """
    if n <= 0 or n_splits <= 0:
        return []
    fold = n // (n_splits + 1)
    if fold <= 0:
        return []
    out: List[Tuple[List[int], List[int]]] = []
    for i in range(1, n_splits + 1):
        test_start = i * fold
        train_end = max(0, test_start - int(embargo))
        test_end = min(n, test_start + fold)
        train_idx = list(range(0, train_end))
        test_idx = list(range(test_start, test_end))
        if train_idx and test_idx:
            out.append((train_idx, test_idx))
    return out


def majority_baseline_accuracy(y_true: Sequence) -> float:
    """Accuracy of always predicting the most common class -- the bar any real
    model must clear."""
    if not y_true:
        return 0.0
    counts = Counter(y_true)
    return max(counts.values()) / len(y_true)


def oos_beats_baseline(y_true: Sequence, y_pred: Sequence, *,
                       min_margin: float = 0.0) -> Tuple[bool, float]:
    """Return ``(passed, margin)`` where ``margin = model_acc - baseline_acc``.
    ``passed`` iff the model beats the majority-class baseline by at least
    ``min_margin``. Mismatched/empty inputs -> ``(False, 0.0)``."""
    n = len(y_true)
    if n == 0 or len(y_pred) != n:
        return (False, 0.0)
    acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / n
    margin = acc - majority_baseline_accuracy(y_true)
    return (margin >= float(min_margin), round(margin, 4))
