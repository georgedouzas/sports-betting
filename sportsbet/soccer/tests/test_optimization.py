"""
Test the optimization module.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import pytest

from sportsbet.soccer import TARGETS
from sportsbet.soccer.optimization import (
        extract_multi_labels, 
        extract_class_labels,
        calculate_yields,
        extract_yields_stats,
        BettorMixin
)


def test_extract_multi_labels():
    """Test the extraction of multi-labels."""
    score1, score2 = [0, 2, 3], [1, 1, 3]
    targets = ['D', 'H', 'over_2.5']
    multi_labels = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1]])
    np.testing.assert_array_equal(extract_multi_labels(score1, score2, targets), multi_labels)


def test_extract_class_labels():
    """Test the extraction of class labels."""
    score1, score2 = [0, 2, 3], [1, 1, 3]
    odds = [[3.0, 1.5, 2.0], [4.0, 2.0, 3.0], [2.5, 2.5, 2.0]]
    targets = ['D', 'H', 'over_2.5']
    class_labels = np.array(['-', 'over_2.5', 'D'])
    np.testing.assert_array_equal(extract_class_labels(score1, score2, odds, targets), class_labels)


def test_calculate_yields():
    """Test the calculation of yields."""
    score1, score2 = [0, 2, 3], [1, 1, 3]
    bets = ['-', 'D', 'over_2.5']
    odds = [[3.0, 1.5, 2.0], [4.0, 2.0, 3.0], [2.5, 2.5, 2.5]]
    targets = ['D', 'H', 'over_2.5']
    yields = np.array([0.0, -1.0, 1.5])
    np.testing.assert_array_equal(calculate_yields(score1, score2, bets, odds, targets), yields)


def test_extract_yields_stats():
    """Test the calculation of yields."""
    yields = np.array([0.0, -1.0, 2.0, 0.0])
    np.testing.assert_array_equal(extract_yields_stats(yields), (0.5, 0.5, 1.5))


def test_none_targets_bettor_mixin():
    """Test bettor mixin with none targets."""
    base_bettor = BettorMixin(None).fit()
    np.testing.assert_array_equal(base_bettor.targets_, np.array(TARGETS.keys()))


def test_invalid_targets_bettor_mixin():
    """Test bettor mixin with invalid targets."""
    with pytest.raises(ValueError):
        BettorMixin(['Away', 'H']).fit()


def test_valid_targets_bettor_mixin():
    """Test bettor mixin with valid targets."""
    base_bettor = BettorMixin(['A', 'H']).fit()
    np.testing.assert_array_equal(base_bettor.targets_, np.array(['A', 'H']))
    
