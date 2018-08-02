"""
Includes a baseline estimator for the classification of
results and a time series cross validator.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import NotFittedError, check_array
from sklearn.model_selection._split import _BaseKFold, _num_samples
from sklearn.utils import indexable
from .. import ODDS_FEATURES


class OddsEstimator(BaseEstimator, ClassifierMixin):
    """Predict the result based on the odds given by betting agents."""

    def fit(self, X, y):
        """No actual fitting occurs."""
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.odds_features_ = range(len(ODDS_FEATURES))
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

    def __init__(self, n_splits=5, time_index=None, season_index):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.test_sizes = test_sizes

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
        test_size = int(n_samples * self.test_percentage)
        test_starts = [n_samples - test_size * ind for ind in range(1, self.n_splits + 1)]
        test_ends = [n_samples - test_size * ind for ind in range(0, self.n_splits)]
        for test_start, test_end in zip(test_starts, test_ends):
            train_indices = indices[0:test_start]
            test_indices = indices[test_start:test_end]
            yield (train_indices, test_indices)

import pandas as pd
from os.path import join
from sportsbet.soccer import BETTING_INTERVAL, TEST_SEASON
training_odds_data = pd.read_csv(join('data', 'training_odds_data.csv'))
X, y = training_odds_data.iloc[:, :-1], training_odds_data.iloc[:, -1]
weeks = pd.cut(training_odds_data.TimeIndex, range(0, max(training_odds_data.TimeIndex) + BETTING_INTERVAL, BETTING_INTERVAL), False)
weeks_test_season = weeks[training_odds_data.Season == TEST_SEASON].cat.remove_unused_categories()
X, y, groups = indexable(X, y, None)
