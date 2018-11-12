"""
Defines helper functions and classes.
"""

from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import SCORERS, make_scorer, f1_score


class SamplingStrategy:
    """Helper class for the sampling strategy parameter of oversamplers."""

    def __init__(self, ratio):
        self.ratio = ratio
    
    def __call__(self, y):
        counter = Counter(y)
        majority_label, majority_n_samples = counter.most_common()[0]
        sampling_strategy = {label: max(int(self.ratio * majority_n_samples), n_samples) 
                                    if label != majority_label else majority_n_samples 
                             for label, n_samples in counter.items()}
        return sampling_strategy

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
        self.estimator_.fit(X[:, 1:-3], y, **fit_params)
        
        return self
        
    def predict(self, X):

        # Generate class labels predictions
        y_pred = self.estimator_.predict(X[:, 1:-3])

        # Generate class propabilities predictions
        y_pred_proba = self.estimator_.predict_proba(X[:, 1:-3])
        
        # Get odds
        odds = X[:, -3:]
        
        return (y_pred, y_pred_proba), odds


def mean_profit_score(y_true, y_pred_odds):
    """Calculate mean profit for a profit estimator."""

    # Define results
    results = ('H', 'A', 'D')

    # Get predictions and odds
    (y_pred, y_pred_proba), odds = y_pred_odds

    # Filter predictions
    mask = (y_pred != '-')
    y_true_sel, y_pred_sel, y_pred_proba_sel, odds_sel = y_true[mask], y_pred[mask], y_pred_proba[mask], odds[mask]

    # Get predicted classes probabilities
    pred_proba_sel = y_pred_proba_sel.max(axis=1)

    # Convert odds to probabilities
    odds_proba_sel = 1 / odds_sel
    odds_proba_sel = odds_proba_sel / odds_proba_sel.sum(axis=1)[:, None]

    # Generate column indices for predicted classes
    indices = [results.index(result) for result in y_pred_sel]
    
    # Select odds and probabilities
    odds_sel = odds_sel[np.arange(len(odds_sel)), indices]
    odds_proba_sel = odds_proba_sel[np.arange(len(odds_proba_sel)), indices]
    
    # Filter predictions
    mask = pred_proba_sel > odds_proba_sel
    y_true_sel, y_pred_sel, odds_sel = y_true_sel[mask], y_pred_sel[mask], odds_sel[mask]
    
    # No predictions case
    if y_pred_sel.size == 0:
        return 0.0
    
    # Calculate profit
    profit = (y_true_sel == y_pred_sel).astype(int) * (odds_sel - 1)
    profit[profit == 0.0] = -1.0

    return profit.mean()


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
