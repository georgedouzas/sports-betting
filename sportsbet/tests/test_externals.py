"""
Test the externals module.
"""

import numpy as np
from sklearn.datasets import make_classification
import pytest

from sportsbet.externals import TimeSeriesSplit, MultiOutputClassifiers

@pytest.mark.parametrize('n_splits, min_train_size', [
    (1, 0.4),
    (1, 0.9),
    (1, 1.5),
    (2, 1.4),
    (3, 0.00)
])
def test_time_series_split_errors(n_splits, min_train_size):
    """Test time series cross validator errors."""
    with pytest.raises(ValueError):
        TimeSeriesSplit(n_splits, min_train_size)


@pytest.mark.parametrize('n_splits, min_train_size', [
    (2, 0.4),
    (3, 0.4),
    (4, 0.4),
    (2, 0.9),
    (3, 0.95),
    (5, 0.9),
    (10, 0.1)
])
def test_time_series_split(n_splits, min_train_size):
    """Test time series cross validator."""
    X, _ = make_classification(random_state=0)
    tscv = TimeSeriesSplit(n_splits, min_train_size)
    indices = list(tscv.split(X))
    assert len(indices) == n_splits == tscv.get_n_splits(X)
    for train_indices, test_indices in indices:
        assert train_indices[0] == 0
        assert len(train_indices) >= int(min_train_size * len(X))
        assert not set(train_indices).intersection(test_indices)
        assert max(train_indices) + 1 == min(test_indices)
    assert test_indices[-1] == len(X) - 1