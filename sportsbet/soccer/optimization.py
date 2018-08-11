"""
Includes classes and functions to select the optimal betting strategy.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array
from sklearn.model_selection._split import _BaseKFold, _num_samples
from sklearn.utils import indexable
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
import progressbar
from . import (
    MIN_N_MATCHES,
    SPI_FEATURES,
    PROB_SPI_FEATURES,
    PROB_FD_FEATURES,
    SEASON_STARTING_DAY,
    RESULTS_MAPPING,
    FD_MAX_ODDS
)

RESULTS_COLUMNS = ['Days', 'Profit', 'Total profit', 'Precision', 'Bets precision',
                   'Bets precision ratio', 'Predictions ratio', 'Threshold']
ODDS_THRESHOLD = 3.75


class SeasonTimeSeriesSplit(_BaseKFold):
    """Season time series cross-validator.
    Parameters
    ----------
    day_index : int
        The index of the day feature.
    starting_day : int
        The starting day of the first test set.
    min_n_matches: int
        The minimum number of matches to include in each test set.
    """

    def __init__(self, day_index=None, starting_day=None, min_n_matches=None):
        self.day_index = day_index
        self.starting_day = starting_day
        self.min_n_matches = min_n_matches

    def _generate_breakpoints(self, X, y=None, groups=None):
        """Generates breakpoints to split data into training and test sets."""
        X, y, groups = indexable(X, y, groups)
        length_num_features = len(SPI_FEATURES + PROB_SPI_FEATURES + PROB_FD_FEATURES)

        self.n_samples_ = _num_samples(X)
        self.day_index_ = length_num_features if self.day_index is None else self.day_index
        self.days_ = pd.Series(check_array(X, dtype=None)[:, self.day_index_])
        self.starting_day_ = int(np.mean([min(self.days_), max(self.days_)])) if self.starting_day is None else self.starting_day
        self.min_n_matches_ = MIN_N_MATCHES if self.min_n_matches is None else self.min_n_matches

        test_days = self.days_[self.days_ >= self.starting_day_]
        count_matches = test_days.groupby(test_days).size()
        total_matches, self.breakpoints_ = 0, []
        for day, n_matches in count_matches.items():
            total_matches += n_matches
            if total_matches >= self.min_n_matches_:
                total_matches = 0
                self.breakpoints_.append(day)
        if self.days_.values[-1] not in self.breakpoints_:
            self.breakpoints_.append(self.days_.values[-1])
        self.n_splits = len(self.breakpoints_)

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
        self._generate_breakpoints(X, y, groups)
        indices = np.arange(self.n_samples_)

        yield (indices[self.days_ < self.starting_day_],
               indices[(self.starting_day_ <= self.days_) & (self.days_ <= self.breakpoints_[0])])

        intervals = list(zip(self.breakpoints_[:-1], self.breakpoints_[1:]))
        for start, end in intervals:
            train_indices = indices[self.days_ <= start]
            test_indices = indices[(start < self.days_) & (self.days_ <= end)]
            yield (train_indices, test_indices)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        self._generate_breakpoints(X, y, groups)
        return len(self.breakpoints_)


class Betting:

    def __init__(self, training_data, odds_data):
        self.training_data = training_data
        self.odds_data = odds_data

    @staticmethod
    def _calculate_profit(y_true, y_pred, odds, use_weights):
        """Calculate mean profit."""
        correct_bets = (y_true == y_pred)
        profit = correct_bets * (odds - 1)
        profit[profit == 0] = -1
        if use_weights:
            profit = np.average(profit, weights=np.exp(odds))
        else:
            profit = profit.mean()
        return profit

    def simulate(self, estimator, param_grid, min_n_matches, predicted_result, fix_threshold=True, use_weights=True):
        """Evaluate the profit by nested cross-validation."""
        pass

        # Define grid search object
        gscv = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring='precision',
            cv=SeasonTimeSeriesSplit(min_n_matches=min_n_matches),
            refit=True,
            n_jobs=-1
        )

        # Extract training data and binarize target
        X, y = self.training_data.iloc[:, :-1], self.training_data.iloc[:, -1]
        y = y.apply(lambda result: 1 if RESULTS_MAPPING[predicted_result] == result else 0)

        # Extract odds data
        odds = self.odds_data[FD_MAX_ODDS + predicted_result]

        # Define parameters
        total_profit = 0
        starting_day = SEASON_STARTING_DAY['17-18']

        # Main results placeholder
        results = []

        # Define time series cross-validator
        scv = SeasonTimeSeriesSplit(starting_day=starting_day, min_n_matches=min_n_matches)

        # Define progress bar
        bar = progressbar.ProgressBar(max_value=scv.get_n_splits(X, y) - 1)

        for ind, (train_indices, test_indices) in enumerate(scv.split(X, y)):

            # Split train and test data
            X_train, X_test, y_train, y_test = X.iloc[train_indices, :], X.iloc[test_indices, :], y[train_indices], y[test_indices]
            odds_test = odds[test_indices]

            # Perform nested cross-validation and make predictions
            if len(param_grid) == 0 and fix_threshold:
                y_pred = estimator.fit(X_train, y_train).predict(X_test)
            else:
                y_pred = gscv.fit(X_train, y_train).predict(X_test)
            odds_threshold = ODDS_THRESHOLD if fix_threshold else 1 / gscv.best_score_

            # Filter bets
            boolean_mask = y_pred.astype(bool) & (odds_test > odds_threshold)
            y_test_sel, y_pred_sel, odds_test_sel = y_test[boolean_mask], y_pred[boolean_mask], odds_test[boolean_mask]

            # Calculate main results
            days = (X.Day[test_indices[0]] - starting_day, X.Day[test_indices[-1]] - starting_day)
            profit = self._calculate_profit(y_test_sel, y_pred_sel, odds_test_sel, use_weights)
            total_profit += profit
            precision = precision_score(y_test, y_pred)
            bets_precision = precision_score(y_test_sel, y_pred_sel)
            n_correct_bets = (y_test_sel == y_pred_sel).sum()
            n_bets = y_pred_sel.size
            n_predictions = y_pred.sum()
            n_matches = y_pred.size

            result = (
                days, profit, total_profit,
                precision, bets_precision,
                '%s / %s' % (n_correct_bets, n_bets),
                '%s / %s' % (n_predictions, n_matches),
                odds_threshold
            )

            # Append results
            results.append(result)

            # Update progress bar
            bar.update(ind)

        results = pd.DataFrame(results, columns=RESULTS_COLUMNS)

        return results

