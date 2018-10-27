"""
Includes classes and functions to test and select the optimal 
betting strategy on historical and current data.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from os.path import join, dirname
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.utils import check_array, check_X_y
from sklearn.model_selection._split import BaseCrossValidator, _num_samples
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
import progressbar

from .data import _fetch_spi_data, _fetch_fd_data, _match_teams_names, LEAGUES_MAPPING


class Ratio:
    """Return dictionary for the ratio parameter of oversamplers."""

    def __init__(self, ratio):
        self.ratio = ratio
    
    def __call__(self, y):
        return {1: int(self.ratio * Counter(y)[0])}

    def __repr__(self):
        return str(self.ratio)


class SeasonTimeSeriesSplit(BaseCrossValidator):
    """Season time series cross-validator.
    Parameters
    ----------
    test_season : str, default='17-18'
        The testing season.
    max_day_range: int
        The maximum day range of each test fold.
    """

    def __init__(self, test_year=2, max_day_range=6):
        self.test_year = test_year
        self.max_day_range = max_day_range

    def _generate_season_indices(self, X):
        """Generate season indices to use in test set."""

        # Check input array
        X = check_array(X, dtype=None)
        
        # Define days
        self.days_ = X[:, 0]

        # Define all and season indices
        indices = np.arange(_num_samples(X))
        start_day, end_day = 365 * (self.test_year - 1), 365 * self.test_year
        season_indices = indices[(self.days_ >= start_day) & (self.days_ < end_day)]

        return season_indices


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
        train_indices : ndarray
            The training set indices for that split.
        test_indices : ndarray
            The testing set indices for that split.
        """

        # Generate season indices
        season_indices = self._generate_season_indices(X)

        # Yield train and test indices
        start_ind = season_indices[0]
        for ind in season_indices:
            if self.days_[ind] - self.days_[start_ind] >= self.max_day_range:
                train_indices = np.arange(0, start_ind)
                test_indices = np.arange(start_ind, ind)
                start_ind = ind
                yield (train_indices, test_indices)

    def get_n_splits(self, X, y=None, groups=None):
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

        # Generate season indices
        season_indices = self._generate_season_indices(X)

        # Calculate number of splits
        start_ind, n_splits = season_indices[0], 0
        for ind in season_indices:
            if self.days_[ind] - self.days_[start_ind] >= self.max_day_range:
                n_splits += 1
                start_ind = ind
        
        return n_splits


class BettingAgent:

    data_path = join(dirname(__file__), '..', '..', 'data', 'training_data.csv')

    def __init__(self, leagues='all'):
        self.leagues = leagues

    def fetch_training_data(self, load=True, save=True, only_numerical=True, return_df=True):
        """Fetch the training data."""

        # Define parameters
        X_numerical_features = ['HomeSPI', 'AwaySPI', 'HomeSPIProb', 'AwaySPIProb', 'DrawSPIProb', 'HomeFDProb', 'AwayFDProb', 'DrawFDProb', 'DiffSPI', 'DiffSPIProb', 'DiffFDProb']
        X_categorical_features = ['Season', 'League', 'HomeTeam', 'AwayTeam']
        odds_features = ['HomeMaximumOdd', 'AwayMaximumOdd', 'DrawMaximumOdd']

        if load:

            # Load data
            training_data = pd.read_csv(self.data_path)
        
            # Check consistency
            if self.leagues == 'all':
                leagues, _ = zip(*LEAGUES_MAPPING.values())
            elif leagues == 'main':
                leagues = [league_id for league_id, league_type in LEAGUES_MAPPING.values() if league_type == 'main']
            else:
                leagues = self.leagues
            if set(training_data.League.unique()) != set(leagues):
                raise RuntimeError('Saved data do not correspond to selected leagues. Set `load` parameter to False.')

        else:
    
            # Define parameters
            avg_odds_features = ['HomeAverageOdd', 'AwayAverageOdd', 'DrawAverageOdd']
            keys = ['Date', 'League', 'HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals']

            # Fetch data
            spi_data = _fetch_spi_data(self.leagues)
            fd_data = _fetch_fd_data(self.leagues)

            # Teams names matching
            mapping = _match_teams_names(spi_data, fd_data)
            spi_data['HomeTeam'] = spi_data['HomeTeam'].apply(lambda team: mapping[team])
            spi_data['AwayTeam'] = spi_data['AwayTeam'].apply(lambda team: mapping[team])

            # Probabilities data
            probs = 1 / fd_data.loc[:, avg_odds_features].values
            probs = pd.DataFrame(probs / probs.sum(axis=1)[:, None], columns=['HomeFDProb', 'AwayFDProb', 'DrawFDProb'])
            probs_data = pd.concat([probs, fd_data], axis=1)
            probs_data.drop(columns=avg_odds_features, inplace=True)

            # Combine data
            training_data = pd.merge(spi_data, probs_data, on=keys)
            training_data['Day'] = (training_data.Date - min(training_data.Date)).dt.days

            # Create features
            training_data['DiffSPI'] = training_data['HomeSPI'] - training_data['AwaySPI']
            training_data['DiffSPIProb'] = training_data['HomeSPIProb'] - training_data['AwaySPIProb']
            training_data['DiffFDProb'] = training_data['HomeFDProb'] - training_data['AwayFDProb']

            # Sort data
            training_data = training_data.sort_values(keys[:-2]).reset_index(drop=True)

        # Save data
        if save:
            training_data.to_csv(self.data_path, index=False)

        # Split data
        X = training_data.loc[:, ['Day'] + X_numerical_features if only_numerical else ['Day'] + X_categorical_features + X_numerical_features]
        y = (training_data['HomeGoals'] - training_data['AwayGoals']).apply(lambda sign: 'H' if sign > 0 else 'D' if sign == 0 else 'A')
        odds = training_data.loc[:, odds_features]

        # Check arrays
        if not return_df:
            X, y = check_X_y(X, y, dtype=None)
            odds = check_array(odds, dtype=None)

        return X, y, odds    
    
    @staticmethod
    def _calculate_profit(y_true, y_pred, odds, weights_func):
        """Calculate mean profit."""
        correct_bets = (y_true == y_pred)
        if correct_bets.size == 0:
            return 0.0
        profit = correct_bets * (odds - 1)
        profit[profit == 0] = -1
        if weights_func is not None:
            profit = np.average(profit, weights=weights_func(odds))
        else:
            profit = profit.mean()
        return profit

    def backtest(self, predicted_result=None, test_year=2, max_day_range=6, estimator=None, load=True, only_numerical=True, **fit_params):
        """Apply backtesting to betting agent."""
        
        # Load data
        X, y, odds = self.fetch_training_data(load=load, save=not load, only_numerical=only_numerical, return_df=False)

        # Define label encoder
        if predicted_result is None:
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
        else:
            y = (y == predicted_result).astype(int)

        # Define parameters
        validation_size = fit_params.pop('validation_size') if 'validation_size' in fit_params else None
        results_mapping = {'H': 0, 'A': 1, 'D': 2}

        # Define train and test indices
        indices = list(SeasonTimeSeriesSplit(test_year=test_year, max_day_range=max_day_range).split(X, y))

        # Define progress bar
        bar = progressbar.ProgressBar(min_value=1, max_value=len(indices))
        
        # Define results placeholder
        self.results = []

        # Run cross-validation
        for ind, (train_indices, test_indices) in enumerate(indices):
            
            # Split to train and test data
            X_train, X_test, y_train, y_test = X[train_indices, 1:], X[test_indices, 1:], y[train_indices], y[test_indices]
            if predicted_result is not None:
                odds_test = odds[test_indices, results_mapping[predicted_result]]

            # Append validation data
            if validation_size is not None:
                
                # Split to train and validation data
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, shuffle=False)
                
                # Apply transformation to validation data
                if isinstance(estimator, Pipeline):
                    X_val = estimator.steps[0][1].fit_transform(X_val, y_val)
                
                # Define parameter name
                param = '%s__eval_set' % estimator.steps[-1][0] if isinstance(estimator, Pipeline) else 'eval_set'
                
                # Update fitting parameters
                fit_params[param] = [(X_val, y_val)]

            # Get test predictions
            y_pred = estimator.fit(X_train, y_train, **fit_params).predict(X_test)

            # Append results
            self.results.append((y_test, y_pred, odds_test))

            # Update progress bar
            bar.update(ind + 1)

    def calculate_statistics(self, odds_threshold=1.0, weights_func=None, factor=1.5, credit_limit=5.0):

        # Initialize parameters
        statistics = []
        capital = 1.0
        bet_amount = 1.0

        for y_test, y_pred, odds_test in self.results:
            
            # Select predictions
            mask = y_pred.astype(bool)
            y_test_sel, y_pred_sel, odds_test_sel = y_test[mask], y_pred[mask], odds_test[mask]

            # Select bets above threshold
            if odds_threshold > 1.0:
                mask = (odds_test_sel > odds_threshold)
                y_test_sel, y_pred_sel, odds_test_sel = y_test_sel[mask], y_pred_sel[mask], odds_test_sel[mask]
            
            # Selects proportion of top odds
            else:
                indices = np.argsort(odds_test_sel)[::-1]
                indices = indices[:int(len(indices) * odds_threshold)]
                y_test_sel, y_pred_sel, odds_test_sel = y_test_sel[indices], y_pred_sel[indices], odds_test_sel[indices]

            # Calculate profit
            profit = bet_amount * self._calculate_profit(y_test_sel, y_pred_sel, odds_test_sel, weights_func)
            
            # Calculate capital
            capital += profit

            # Adjust bet amount
            bet_amount = bet_amount * factor if profit < 0.0 else 1.0

            # Calculate credit
            max_credit = capital + credit_limit
            if bet_amount > max_credit:
                bet_amount = max_credit
                
            # Calculate precision
            precision = precision_score(y_test, y_pred)
            bets_precision = precision_score(y_test_sel, y_pred_sel)
                
            # Calculate number of bets, predictions and matches
            n_bets = y_pred_sel.size
            n_predictions = y_pred.sum()
            n_matches = y_pred.size
                
            # Generate statistic
            statistic = (capital, profit, bets_precision, precision, n_bets, n_predictions, n_matches)

            # Append results
            statistics.append(statistic)

            if bet_amount == 0:
                break

        # Define statistics dataframe
        statistics = pd.DataFrame(statistics, columns=['Capital', 'Profit', 'Bets precision', 'Precision', 'Bets', 'Predictions', 'Matches'])

        return statistics
