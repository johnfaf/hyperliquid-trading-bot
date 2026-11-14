import numpy as np
import pytest

from src.signals.lstm_agent import _validation_boundaries, create_sequences, label_direction


def test_unknown_lstm_outcomes_are_not_training_rows():
    labels = label_direction([{"close": x} for x in [100, 101, 102, 103, 104, 105]], lookahead=2)
    assert labels.tolist() == [2, 2, 2, 2, -1, -1]
    features = np.arange(6).reshape(-1, 1)
    X, y = create_sequences(features, labels, seq_len=2)
    assert X[:, :, 0].tolist() == [[0, 1], [1, 2]]
    assert y.tolist() == [2, 2]


def test_lstm_training_labels_cannot_reach_validation():
    train_end, val_start = _validation_boundaries(100, 0.15, 6)
    assert train_end - 1 + 6 < val_start
    with pytest.raises(ValueError):
        _validation_boundaries(8, 0.5, 6)
