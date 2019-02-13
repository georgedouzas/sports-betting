"""
Includes classes and functions to test and select the optimal 
betting strategy on historical and current data.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from argparse import ArgumentParser
from os import listdir
from os.path import join
from pickle import dump, load
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.utils import Parallel, delayed, check_X_y
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.metrics import log_loss
from tqdm import tqdm

TARGET_TYPES_MAPPING = {
    'H': lambda y: (y[:, 0] > y[:, 1]).astype(int),
    'D': lambda y: (y[:, 0] == y[:, 1]).astype(int),
    'A': lambda y: (y[:, 0] < y[:, 1]).astype(int),
    'both_score': lambda y: (y[:, 0] * y[:, 1] > 0).astype(int),
    'no_nil_score': lambda y: (y[:, 0] + y[:, 1] > 0).astype(int)
}


def combine_odds(odds, betting_types):
    """Combine odds of different betting types."""
    return 1 / np.concatenate([1 / odds[betting_type].values.reshape(-1, 1) for betting_type in betting_types], axis=1).sum(axis=1)


class SeasonSplit(BaseCrossValidator):
    """Split time-series data based on a test season."""

    def __init__(self, n_splits, seasons, test_season):
        self.n_splits = n_splits
        self.seasons = seasons
        self.test_season = test_season

    def split(self, X=None, y=None, groups=None):
        """Generate indices to split data into training and test set."""
        start_index, end_index = (self.seasons != self.test_season).sum(), len(self.seasons)
        step = (end_index - start_index) // self.n_splits
        breakpoints = list(range(start_index, end_index, step)) + [end_index]
        for start, end in zip(breakpoints[:-1], breakpoints[1:]):
            yield np.arange(0, start), np.arange(start, end)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits


def mean_yields(y_true, y_pred, target_types, strategy):

    # Extract targets
    y_true = np.concatenate([TARGET_TYPES_MAPPING[target_type](y_true).reshape(-1, 1) for target_type in target_types], axis=1)

    # Extract predictions
    y_pred, odds = y_pred

    # Calculate yields and values
    yields = y_true * odds - 1.0
    values = y_pred * odds

    # Apply strategy
    yields[(y_pred if strategy[0] == 'probs' else values) <= strategy[1]] = 0.0
    yields = yields[range(len(yields)), (y_pred if 'probs' in strategy[0] else values).argmax(axis=1)]

    # Exclude no bets
    yields = yields[yields != 0.0]

    return yields.mean()


class BettingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, X, y, target_type):
        """Fit betting classifier."""

        # Check target type
        if target_type not in TARGET_TYPES_MAPPING.keys():
            raise ValueError('Wrong target type.')

        # Extract target
        y = TARGET_TYPES_MAPPING[target_type](y)

        # Fit classifier
        self.classifier_ = clone(self.classifier).fit(X[:, :-1], y)

        return self

    def predict_proba(self, X):
        """"Predict probability of betting classifier."""

        # Predict probability
        y_pred = self.classifier_.predict_proba(X[:, :-1])[:, 1:]

        # Extract odds
        odds = X[:, -1:]

        return y_pred, odds


class MetaBettingClassifier(_BaseComposition):

    def __init__(self, betting_classifiers):
        self.betting_classifiers = betting_classifiers

    def fit(self, X, y):
        """Fit betting classifiers."""
        
        # Placeholder
        self.betting_classifiers_ = []

        # Number of classifiers
        self.n_clfs_ = len(self.betting_classifiers)

        # Fit betting classifiers
        for ind, (target_type, clf) in enumerate(self.betting_classifiers):
            X_clf = np.concatenate([X[:, :-self.n_clfs_], X[:, ind - self.n_clfs_].reshape(-1, 1)], axis=1)
            self.betting_classifiers_.append(clone(clf).fit(X_clf, y, target_type))

        return self
    
    def predict_proba(self, X):
        """Predict probabilities of betting classifiers."""

        # Placeholders
        y_pred, odds = [], []

        # Predict probabilities
        for ind, clf in enumerate(self.betting_classifiers_):
            X_clf = np.concatenate([X[:, :-self.n_clfs_], X[:, ind - self.n_clfs_].reshape(-1, 1)], axis=1)
            proba, odd = clf.predict_proba(X_clf)
            y_pred.append(proba)
            odds.append(odd)
        
        return np.concatenate(y_pred, axis=1), np.concatenate(odds, axis=1)


if __name__ == '__main__':

    from sklearn.linear_model import LogisticRegression
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import make_pipeline
    from sportsbet.soccer.optimization import SeasonSplit, BettingClassifier, MetaBettingClassifier, mean_yields
    from sportsbet.soccer.data import SoccerDataLoader
    import numpy as np
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import cross_val_score
    from sklearn.neural_network import MLPClassifier

    # Load data
    sdl = SoccerDataLoader('all')
    X, y, odds, matches = sdl.training_data

    # Betting classifiers
    bclf_H = BettingClassifier(LogisticRegression(solver='lbfgs', max_iter=10000))
    X_H = np.concatenate([X, odds[['H']]], axis=1)

    bclf_A = BettingClassifier(make_pipeline(SMOTE(sampling_strategy=0.60), LogisticRegression(solver='lbfgs', max_iter=5000, C=1e3)))
    X_A = np.concatenate([X, odds[['A']]], axis=1)

    bclf_D = BettingClassifier(make_pipeline(SMOTE(sampling_strategy=0.95, random_state=0), MLPClassifier(activation='logistic', hidden_layer_sizes=(50, 50), validation_fraction=0.3, early_stopping=True)))
    X_D = np.concatenate([X, odds[['D']]], axis=1)

    mbclf = MetaBettingClassifier([('A', bclf_H), ('D', bclf_D)])
    X_AD = np.concatenate([X, odds[['A', 'D']]], axis=1)

    # Cross validation scores
    cv = SeasonSplit(5, matches.Season.values, '1819')

    scorer_H = make_scorer(mean_yields, needs_proba=True, target_types=['H'], strategy=('probs', 0.5))
    cross_val_score(bclf_H, X_H, y, scoring=scorer_H, cv=cv, fit_params={'target_type': 'H'})

    scorer_A = make_scorer(mean_yields, needs_proba=True, target_types=['A'], strategy=('probs', 0.5))
    cross_val_score(bclf_A, X_A, y, scoring=scorer_A, cv=cv, fit_params={'target_type': 'A'})

    scorer_D = make_scorer(mean_yields, needs_proba=True, target_types=['D'], strategy=('probs', 0.5))
    cross_val_score(bclf_D, X_D, y, scoring=scorer_D, cv=cv, fit_params={'target_type': 'D'})

    scorer_AD = make_scorer(mean_yields, needs_proba=True, target_types=['A', 'D'], strategy=('probs', 0.5))
    cross_val_score(mbclf, X_AD, y, scoring=scorer_AD, cv=cv)
    
    # Create parser
    parser = ArgumentParser('Models evaluation using backtesting.')
    
    # Add arguments
    parser.add_argument('--clfs-name', default='random', help='The name of classifiers to predict the results.')
    parser.add_argument('--max-odds', default=['pinnacle', 'bet365', 'bwin'], nargs='*', help='Maximum odds to use for evaluation.')
    parser.add_argument('--test-season', default=1819, type=int, help='Test season.')
    parser.add_argument('--random-states', default=[0, 1, 2], type=int, nargs='*', help='The random states of estimators.')
    parser.add_argument('--save-results', default=True, type=bool, help='Save backtesting results to csv.')

