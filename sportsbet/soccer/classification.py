"""
Includes a baseline estimator for the classification of
results and a time series cross validator.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import NotFittedError, check_array
from sklearn.model_selection._split import _BaseKFold, _num_samples
from sklearn.utils import indexable
from . import ODDS_FEATURES, TOTAL_ODDS_FEATURES, BETTING_INTERVAL


class OddsEstimator(BaseEstimator, ClassifierMixin):
    """Predict the result based on the odds given by betting agents."""

    def __init__(self, odd_features=None):
        self.odds_features = odd_features

    def fit(self, X, y):
        """No actual fitting occurs."""
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.odds_features_ = range(len(ODDS_FEATURES)) if self.odds_features is None else self.odds_features
        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        C : array, shape = [n_samples]
            Predicted class label per sample.
        """

        predictions = self.predict_proba(X).argmax(axis=1)
        return predictions

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        if not hasattr(self, "classes_"):
            raise NotFittedError("Call fit before prediction")
        X = check_array(X)
        odds = X[:, self.odds_features_].reshape(X.shape[0], -1, 3)
        probabilities = 1 / odds
        probabilities = (probabilities / probabilities.sum(axis=2)[:, :, None]).mean(axis=1)
        return probabilities


class SeasonTimeSeriesSplit(_BaseKFold):
    """Season time series cross-validator.
    Parameters
    ----------
    n_splits : int, default=4
        Number of splits. Must be at least 1.
    test_percentage : float, default=0.1
        The percentage of test samples. The values should be
        in the [0.0, 1.0] range.
    """

    def __init__(self, time_index=None, season_index=None, betting_interval=None):
        self.time_index = time_index
        self.season_index = season_index
        self.betting_interval = betting_interval

    def split(self, X, y=None, groups=None):
        """Generates indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like, with shape (n_samples,), optional
            Always ignored, exists for compatibility.
        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """

        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)

        self.time_index_ = len(TOTAL_ODDS_FEATURES) if self.time_index is None else self.time_index
        self.season_index_ = len(TOTAL_ODDS_FEATURES) + 3 if self.season_index is None else self.season_index
        self.betting_interval_ = BETTING_INTERVAL if self.betting_interval is None else self.betting_interval

        try:
            time, seasons = X[:, self.time_index_], X[:, self.season_index_]
        except TypeError:
            time, seasons = X.iloc[:, self.time_index_].values, X.iloc[:, self.season_index_].values
        time_intervals = pd.cut(time, range(0, max(time) + self.betting_interval_, self.betting_interval_), False)
        season_time_intervals = time_intervals[seasons == seasons[-1]].remove_unused_categories().categories
        self.n_splits = len(season_time_intervals)

        for time_interval in season_time_intervals:
            train_indices = indices[time_intervals < time_interval]
            test_indices = indices[time_intervals == time_interval]
            yield (train_indices, test_indices)


