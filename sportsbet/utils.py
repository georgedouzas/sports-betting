"""
Defines helper functions and classes.
"""

from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import SCORERS, make_scorer


class SamplingStrategy:
    """Helper class for the sampling strategy parameter of oversamplers."""

    def __init__(self, ratio):
        self.ratio = ratio
    
    def __call__(self, y):
        return {1: int(self.ratio * Counter(y)[0])}

    def __repr__(self):
        return str(self.ratio)


class ProfitEstimator(BaseEstimator, RegressorMixin):
    """Wrapper class of an estimator."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):

        # Clone estimator
        self.estimator_ = clone(self.estimator)

        # Exclude time index and maximum odds
        self.estimator_.fit(X[:, 1:-1], y, **fit_params)
        
        return self
        
    def predict(self, X):

        # Exclude time index and maximum odds
        profit = self.estimator_.predict(X[:, 1:-1]) * (X[:, -1] - 1)
        
        return profit


def _profit_score(y_true, y_pred):

    # Filter positive class preditions
    mask = y_pred > 0
    y_true_sel, y_pred_sel = y_true[mask], y_pred[mask]
    
    # No preidictions case
    if y_pred_sel.size == 0:
        return np.array([0.0])
    
    # Calculate profit
    profit = y_true_sel * y_pred_sel
    profit[profit == 0.0] = -1.0

    return profit


def mean_profit_score(y_true, y_pred):
    """Calculate total profit for a profit estimator."""
    
    profit = _profit_score(y_true, y_pred)

    return profit.mean()


def total_profit_score(y_true, y_pred):
    """Calculate total profit for a profit estimator."""
    
    profit = _profit_score(y_true, y_pred)

    return profit.sum()


def set_random_state(classifier, random_state):
    """Set the random state of all estimators."""
    for param in classifier.get_params():
        if 'random_state' in param:
            classifier.set_params(**{param: random_state})


def _fit_predict(classifier, X, y, train_indices, test_indices, **fit_params):
    """Fit estimator and predict for a set of train and test indices."""

    # Fit classifier
    classifier.fit(X[train_indices], y[train_indices], **fit_params)

    # Filter test samples
    y_test = y[test_indices]

    # Predict on test set
    y_pred = classifier.predict(X[test_indices])
        
    return y_test, y_pred


def import_custom_classifiers(default_classifiers):
    """Try to import custom classifiers."""
    try:
        from extra_config import CLASSIFIERS
        CLASSIFIERS.update(default_classifiers)
    except ImportError:
        CLASSIFIERS = default_classifiers
    return CLASSIFIERS
