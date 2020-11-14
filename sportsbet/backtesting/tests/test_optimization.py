"""
Test the _optimization module.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit

from sportsbet.datasets._soccer._utils import TARGETS_MAPPING 
from sportsbet.backtesting._optimization import (
        calculate_yields,
        extract_yields_stats,
        fit_bet,
        apply_backtesting,
)
from sportsbet.backtesting._betting import Bettor

X = [[0], [1], [2]] * 10
score1, score2 = [0, 2, 3] * 10, [1, 1, 3] * 10
odds = [[3.0, 1.5], [4.0, 2.0], [2.5, 2.5]] * 10


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


def test_fit_bet():
    """Test fit and bet function."""

    # Input data
    bettor = Bettor(classifier=DummyClassifier(), targets=['D', 'H'])
    params = {'classifier__strategy': 'constant', 'classifier__constant': 'H'}
    risk_factors = [0.0]
    random_state = 0
    X = np.random.uniform(size=(100, 2))
    scores = np.repeat([1, 0], 50), np.repeat([0, 1], 50), np.repeat([1, 0], 50), np.repeat([0, 1], 50)
    train_indices, test_indices = np.arange(0, 25), np.arange(25, 100)
    odds = np.repeat([2.0, 2.0], 100).reshape(-1, 2)

    # Output
    data = fit_bet(bettor, params, risk_factors, random_state, X, scores, odds, train_indices, test_indices)
    
    # Expected output
    expected_yields = np.concatenate([np.repeat(1.0, 25), np.repeat(-1.0, 50)])
    expected_data = pd.DataFrame([[str(params), random_state, risk_factors[0], expected_yields]], columns=['parameters', 'experiment', 'risk_factor', 'yields'])
    
    pd.testing.assert_frame_equal(expected_data, data)


def test_apply_backtesting():
    """Test backtesting function."""

    # Input data
    bettor = Bettor(classifier=DummyClassifier(), targets=['D', 'H'])
    param_grid = {'classifier__strategy': ['uniform', 'stratified']}
    risk_factors = [0.0, 0.2, 0.4]
    random_state = 0
    X = np.random.uniform(size=(100, 2))
    scores = np.repeat([1, 0], 50), np.repeat([0, 1], 50), np.repeat([1, 0], 50), np.repeat([0, 1], 50)
    odds = np.repeat([2.0, 2.0], 100).reshape(-1, 2)
    cv = TimeSeriesSplit()
    n_runs = 3
    n_jobs = -1

    # Output
    results = apply_backtesting(bettor, param_grid, risk_factors, X, scores, odds, cv, random_state, n_runs, n_jobs)

    assert list(results.columns) == ['parameters', 'risk_factor', 'coverage', 'mean_yield', 'std_yield', 'std_mean_yield']
    assert len(results) == len(risk_factors) * len(ParameterGrid(param_grid))
