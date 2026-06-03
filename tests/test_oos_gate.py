"""ML deployment discipline: purged walk-forward + beat-baseline gate (signal #8, PR-C)."""
from __future__ import annotations

from src.learning.oos_gate import (
    majority_baseline_accuracy,
    oos_beats_baseline,
    purged_walk_forward_splits,
)


def test_purged_splits_are_causal_and_embargoed():
    splits = purged_walk_forward_splits(100, n_splits=4, embargo=5)
    assert len(splits) == 4
    for train_idx, test_idx in splits:
        assert train_idx and test_idx
        # train strictly precedes test, with at least `embargo` purged in between
        assert max(train_idx) < min(test_idx)
        assert min(test_idx) - max(train_idx) - 1 >= 5
    # expanding window: each fold's train set grows
    assert all(len(splits[i][0]) < len(splits[i + 1][0]) for i in range(len(splits) - 1))


def test_purged_splits_degenerate_inputs():
    assert purged_walk_forward_splits(0) == []
    assert purged_walk_forward_splits(3, n_splits=10) == []   # fold size 0


def test_majority_baseline_accuracy():
    assert majority_baseline_accuracy([1, 1, 1, 0]) == 0.75
    assert majority_baseline_accuracy([]) == 0.0


def test_oos_beats_baseline():
    y = [0, 0, 0, 1, 1, 1]                      # baseline (majority) = 0.5
    perfect = [0, 0, 0, 1, 1, 1]
    passed, margin = oos_beats_baseline(y, perfect, min_margin=0.1)
    assert passed and abs(margin - 0.5) < 1e-9

    always_majority = [0, 0, 0, 0, 0, 0]        # == baseline -> margin 0
    passed2, margin2 = oos_beats_baseline(y, always_majority, min_margin=0.05)
    assert not passed2 and abs(margin2) < 1e-9

    assert oos_beats_baseline([1, 0], [1], min_margin=0.0) == (False, 0.0)  # mismatch
