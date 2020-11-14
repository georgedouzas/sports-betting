"""
Includes classes and functions to test and select the optimal 
betting strategy.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from itertools import product
from abc import abstractmethod
from joblib import delayed, Parallel

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state, check_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ..datasets._soccer._utils import TARGETS_MAPPING


def calculate_yields(score1, score2, bets, odds, targets):
    """Calculate the yields."""

    # Check odds
    odds = check_array(odds)
    targets = check_array(targets, dtype=object, ensure_2d=False)

    # Generate yields
    bets = MultiLabelBinarizer(classes=['-'] + targets.tolist()).fit_transform([[bet] for bet in bets])[:, 1:]
    yields = ((extract_multi_labels(score1, score2, targets) * odds - 1.0) * bets).sum(axis=1)

    return yields


def extract_yields_stats(yields):
    """Extract coverage, mean and std of yields."""
    coverage_mask = (yields != 0.0)
    return coverage_mask.mean(), yields[coverage_mask].mean(), yields[coverage_mask].std()
    

def check_random_states(random_state, repetitions):
    """Create random states for experiments."""
    random_state = check_random_state(random_state)
    return [random_state.randint(0, 2 ** 32 - 1, dtype='uint32') for _ in range(repetitions)]


def fit_bet(bettor, params, risk_factors, random_state, X, scores, odds, train_indices, test_indices):
    """Parallel fit and bet"""

    # Unpack scores
    avg_score1, avg_score2, score1, score2 = scores

    # Set random state
    for param_name in bettor.get_params():
        if 'random_state' in param_name:
            bettor.set_params(**{param_name: random_state})

    # Fit better
    bettor.set_params(**params).fit(X[train_indices], avg_score1[train_indices], avg_score2[train_indices], odds[train_indices])

    # Generate data
    data = []
    for risk_factor in risk_factors:
        bets = bettor.bet(X[test_indices], risk_factor)
        yields = calculate_yields(score1[test_indices], score2[test_indices], bets, odds[test_indices], bettor.targets_)
        data.append((str(params), random_state, risk_factor, yields))
    data = pd.DataFrame(data, columns=['parameters', 'experiment', 'risk_factor', 'yields'])
    
    return data


def apply_backtesting(bettor, param_grid, risk_factors, X, scores, odds, cv, random_state, n_runs, n_jobs):
    """Apply backtesting to evaluate bettor."""
    
    # Check random states
    random_states = check_random_states(random_state, n_runs)

    # Check arrays
    X = check_array(X, dtype=None, force_all_finite=False)
    normalized_scores = []
    for score in scores:
        normalized_scores.append(check_array(score, dtype=None, ensure_2d=False))
    odds = check_array(odds, dtype=None)

    # Extract parameters
    parameters = ParameterGrid(param_grid)

    # Run backtesting
    data = Parallel(n_jobs=n_jobs)(delayed(fit_bet)(bettor, params, risk_factors, random_state, X, normalized_scores, odds, train_indices, test_indices) 
           for params, random_state, (train_indices, test_indices) in tqdm(list(product(parameters, random_states, cv.split(X))), desc='Tasks'))
    
    # Combine data
    data = pd.concat(data, ignore_index=True)
    data = data.groupby(['parameters', 'risk_factor', 'experiment']).apply(lambda df: np.concatenate(df.yields.values)).reset_index()
    data[['coverage', 'mean_yield', 'std_yield']] = pd.DataFrame(data[0].apply(lambda yields: extract_yields_stats(yields)).values.tolist())
    
    # Calculate results
    results = data.drop(columns=['experiment', 0]).groupby(['parameters', 'risk_factor']).mean().reset_index()
    results['std_mean_yield'] = data.groupby(['parameters', 'risk_factor'])['mean_yield'].std().values
    results = results.sort_values('mean_yield', ascending=False).reset_index(drop=True)

    return results

