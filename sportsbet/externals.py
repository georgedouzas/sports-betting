"""
Includes extensions of external packages.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from math import ceil

from joblib import delayed, Parallel
import numpy as np
from sklearn.base import is_classifier
from sklearn.model_selection import BaseCrossValidator
from sklearn.multiclass import check_classification_targets
from sklearn.multioutput import MultiOutputClassifier, _fit_estimator
from sklearn.utils import check_X_y, check_array
from sklearn.utils.fixes import parallel_helper
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import has_fit_parameter, check_is_fitted


class TimeSeriesSplit(BaseCrossValidator):
    """Split time-series data."""

    def __init__(self, n_splits=5, min_train_size=0.5):
        if n_splits < 2:
            raise ValueError('K-fold cross-validation requires at least one train/test split. Got n_splits < 2.')
        if min_train_size <= 0.0 or min_train_size >= 1.0:
            raise ValueError(f'Minimum training size should be in the (0.0, 1.0) interval. Got {min_train_size}.')
        self.n_splits = n_splits
        self.min_train_size = min_train_size

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""
        start_index, end_index = int(self.min_train_size * len(X)), len(X)
        step = ceil((end_index - start_index) / self.n_splits)
        breakpoints = list(range(start_index, end_index, step)) + [end_index]
        for start, end in zip(breakpoints[:-1], breakpoints[1:]):
            yield np.arange(0, start), np.arange(start, end)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits


class MultiOutputClassifiers(_BaseComposition, MultiOutputClassifier):

    def __init__(self, classifiers, n_jobs=None):
        self.classifiers = classifiers
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """Fit a separate classifier for each output variable."""

        for _, clf in self.classifiers:
            if not hasattr(clf, 'fit'):
                raise ValueError('Every base classifier should implement a fit method.')

        X, y = check_X_y(X, y, multi_output=True, accept_sparse=True)

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            raise ValueError('Output y must have at least two dimensions for multi-output classification but has only one.')

        if sample_weight is not None and any([not has_fit_parameter(clf, 'sample_weight') for _, clf in self.classifiers]):
            raise ValueError('One of base classifiers does not support sample weights.')

        self.classifiers_ = Parallel(n_jobs=self.n_jobs)(delayed(_fit_estimator)(clf, X, y[:, i], sample_weight) 
                                                        for i, (_, clf) in zip(range(y.shape[1]), self.classifiers))
        
        return self

    def predict(self, X):
        """Predict multi-output target."""
        
        check_is_fitted(self, 'classifiers_')

        for _, clf in self.classifiers:
            if not hasattr(clf, 'predict'):
                raise ValueError('Every base classifier should implement a predict method')

        X = check_array(X, accept_sparse=True)

        y_pred = Parallel(n_jobs=self.n_jobs)(delayed(parallel_helper)(clf, 'predict', X) for clf in self.classifiers_)

        return np.asarray(y_pred).T

    def predict_proba(self, X):
        """Predict multi-output probabilities."""
        
        check_is_fitted(self, 'classifiers_')

        for _, clf in self.classifiers:
            if not hasattr(clf, 'predict_proba'):
                raise ValueError('Every base should implement predict_proba method')

        X = check_array(X, accept_sparse=True)

        y_pred = Parallel(n_jobs=self.n_jobs)(delayed(parallel_helper)(clf, 'predict_proba', X) for clf in self.classifiers_)

        return y_pred