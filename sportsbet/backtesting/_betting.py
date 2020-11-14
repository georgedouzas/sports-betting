"""
Includes classes and functions to test and select the optimal 
betting strategy.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import check_array, check_scalar
from sklearn.model_selection import train_test_split


def extract_class_labels(Y, O, targets):
    """Extract class labels for multi-class classification."""
    Y = check_array(Y, ensure_2d=False, dtype=None)
    O = check_array(O, ensure_2d=False, dtype=None)
    indices = (Y * O).argmax(axis=1)
    class_labels = np.array([targets[ind] for ind in indices])
    class_labels[Y.sum(axis=1) == 0] = '-'
    return class_labels


class BettorMixin:
    """Mixin class for all bettors."""

    _estimator_type = 'bettor'

    def fit(self, X, Y, O):
        """Fit base bettor."""

        # Check input
        if not isinstance(Y, pd.DataFrame):
            raise TypeError(f'Targets Y should be a pandas dataframe. Instead got {type(Y)} class.')
        if O is not None:
            if not isinstance(O, pd.DataFrame):
                raise TypeError(f'Odds O should be a pandas dataframe. Instead got {type(O)} class.')
            for target_col, odds_col in zip(Y.columns, O.columns):
                if not odds_col.endswith(f'{target_col}_odds'):
                    raise ValueError('Columns of target and odds dataframes are not compatible.')
        
        # Check targets
        self.targets_ = check_array(Y.columns, dtype=None, ensure_2d=False)
        
        return self
    
    def bet(self, X, O, risk_factor=0.0):
        """Generate bets."""

        # Check risk factor
        check_scalar(risk_factor, 'risk_factor', target_type=float, min_val=0.0)
        
        # Generate bets
        bets = self.predict(X)

        # Apply no bets
        probs = self.predict_proba(X)
        start_ind = int((len(self.targets_) + 1) == probs.shape[1])
        bets[(probs[:, start_ind:] * O).max(axis=1) <= risk_factor] = '-'

        return bets


class MultiClassBettor(BettorMixin, ClassifierMixin, BaseEstimator):
    """Bettor class that uses a multi-class classifier."""

    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, X, Y, O):
        """Fit the classifier."""

        super(MultiClassBettor, self).fit(X, Y, O)

        # Extract targets
        y = extract_class_labels(Y, O, self.targets_)

        # Fit classifier
        self.classifier_ = clone(self.classifier).fit(X, y)

        return self

    def predict(self, X):
        """Predict class labels."""
        return self.classifier_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.classifier_.predict_proba(X)


class MultiOutputMetaBettor(BettorMixin, ClassifierMixin, BaseEstimator):
    """Bettor class that uses a multi-output classifier and a meta-classifier."""

    def __init__(self, multi_classifier, meta_classifier, test_size=0.5, random_state=None):
        self.multi_classifier = multi_classifier
        self.meta_classifier = meta_classifier
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self, X, Y, O):
        """Fit the multi-output classifier and the meta-classifier."""

        super(MultiOutputMetaBettor, self).fit(X, Y, O)

        # Split data
        X_multi, X_meta, Y_multi, Y_meta, _, O_meta  = train_test_split(
            X, Y, O, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # Extract targets
        y_meta = extract_class_labels(Y_meta, O_meta, self.targets_)

        # Fit multi-classifier
        self.multi_classifier_ = clone(self.multi_classifier).fit(X_multi, Y_multi)

        # Fit meta-classifier
        X_meta = np.column_stack([probs[:, 0] for probs in self.multi_classifier_.predict_proba(X_meta)])
        self.meta_classifier_ = clone(self.meta_classifier).fit(X_meta, y_meta)
        
        return self

    def predict(self, X):
        """Predict class labels."""
        X_meta = np.column_stack([probs[:, 0] for probs in self.multi_classifier_.predict_proba(X)])
        return self.meta_classifier_.predict(X_meta)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X_meta = np.column_stack([probs[:, 0] for probs in self.multi_classifier_.predict_proba(X)])
        return self.meta_classifier_.predict_proba(X_meta)
