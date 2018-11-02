"""
Defines constants and helper functions/classes.
"""

from collections import Counter
from pathlib import Path
from os.path import join
from sys import path

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import SCORERS, make_scorer

# Define default path
PATH = join(str(Path.home()), '.sports-betting')

# Append default path
path.append(PATH)

# Class for sampling strategy
class SamplingStrategy:
    """Helper class for the sampling strategy parameter of oversamplers."""

    def __init__(self, ratio):
        self.ratio = ratio
    
    def __call__(self, y):
        return {1: int(self.ratio * Counter(y)[0])}

    def __repr__(self):
        return str(self.ratio)

# Class for the profit estimation
class ProfitEstimator(BaseEstimator, RegressorMixin):
    """Wrapper class of an estimator."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, :-1], y, **fit_params)
        return self
        
    def predict(self, X):
        profit = self.estimator_.predict(X[:, :-1]) * (X[:, -1] - 1)
        return profit

# Function for total profit score
def total_profit_score(y_true, y_pred):
    """Calculate total profit for a profit estimator."""
    
    mask = y_pred > 0
    y_true_sel, y_pred_sel = y_true[mask], y_pred[mask]
    
    if y_pred_sel.size == 0:
        return 0.0
    
    profit = y_true_sel * y_pred_sel
    profit[profit == 0.0] = -1.0

    return profit.sum()

# Append total profit to scorers
SCORERS['total_profit_score'] = make_scorer(total_profit_score)