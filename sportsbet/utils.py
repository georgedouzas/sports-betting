"""
Defines helper functions and classes.
"""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

##############
#Optimization#
##############

class BookmakerEstimator(BaseEstimator, ClassifierMixin):
    """Estimator that uses the average odds to generate predictions."""

    def fit(self, X, y):

        # Define predicted labels
        self.labels_ = [label if label in np.unique(y) else '-' for label in ['H', 'A', 'D']]
        
        return self

    def predict(self, X):
        
        # Generate indices from minimum odds
        min_odds_indices = np.argmin(X[:, 0:3], axis=1)

        # Get predictions
        y_pred = np.array(self.labels_)[min_odds_indices]

        return y_pred

    def predict_proba(self, X):

        # Generate predicted probabilities
        y_pred_proba = 1 / X[:, 0:3]
        y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1)[:, None]

        # Sort predicted labels
        y_pred_proba = y_pred_proba[:, np.argsort(self.labels_)]
        
        # Add probabilities
        if len(np.unique(self.labels_)) < 3:
            y_pred_proba = np.apply_along_axis(arr=y_pred_proba, func1d=lambda probs: [probs[0:2].sum(), probs[2]], axis=1)

        return y_pred_proba


def fit_predict(classifier, params, X, y, odds, train_indices, test_indices, random_state):
    """Fit estimator and predict for a set of train and test indices."""

    # Set param_grid
    label, ind, params = params
    classifier.set_params(**params)
    
    # Set random state
    for param in classifier.get_params():
        if 'random_state' in param:
            classifier.set_params(**{param: random_state})

    # Filter test samples
    X_test, odds = X[test_indices], odds.iloc[test_indices].reset_index(drop=True)
    results = pd.DataFrame(y[test_indices], columns=['Result'])

    # Binarize labels
    y_bin = y.copy()
    y_bin[y_bin != label] = '-'

    # Fit classifier
    classifier.fit(X[train_indices], y_bin[train_indices])
    
    # Get test set predictions
    probabilities = pd.DataFrame(classifier.predict_proba(X_test)[:, -1], columns=[label])

    # Combine data
    data = pd.concat([results, probabilities, odds], axis=1)

    return test_indices[0], random_state, (label, ind), data


def generate_month_indices(matches, test_season):
    """Generate train and test monthly indices for a test season."""
    
    # Test indices
    test_indices = matches.loc[matches['Season'] == test_season].groupby('Month', sort=False).apply(lambda row: np.array(row.index)).values
    
    # Train indices
    train_indices = [np.arange(0, test_indices[0][0])]
    train_indices += [np.arange(0, test_ind[-1] + 1) for test_ind in test_indices]
    
    # Combine indices
    indices = list(zip(train_indices, test_indices))
    
    return indices

###########################
#Classifiers configuration#
###########################

def import_custom_classifiers(default_classifiers):
    """Try to import custom classifiers."""
    try:
        from config import CLASSIFIERS
        CLASSIFIERS = {'default': CLASSIFIERS}
        CLASSIFIERS.update(default_classifiers)
    except ImportError:
        CLASSIFIERS = default_classifiers
    return CLASSIFIERS


DEFAULT_CLASSIFIERS = {
    'random': {
        'H': (DummyClassifier(random_state=0), {}),
        'A': (DummyClassifier(random_state=1), {}),
        'D': (DummyClassifier(random_state=2), {})
    },
    'bookmaker': {
        'H': (BookmakerEstimator(), {}),
        'A': (BookmakerEstimator(), {}),
        'D': (BookmakerEstimator(), {})
    }
}
CLASSIFIERS = import_custom_classifiers(DEFAULT_CLASSIFIERS)