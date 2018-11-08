"""
Includes classes and functions to test and select the optimal 
betting strategy on historical and current data.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from pathlib import Path
from os.path import join
from pickle import dump

import numpy as np
import pandas as pd
from sklearn.utils import check_array, check_X_y
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection._split import BaseCrossValidator, _num_samples
from sklearn.metrics import precision_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import Parallel, delayed
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

from .. import PATH
from ..utils import ProfitEstimator, total_profit_score, mean_profit_score, set_random_state, _fit_predict
from .data import (
    _fetch_historical_spi_data, 
    _fetch_historical_fd_data, 
    _fetch_future_spi_data,
    _scrape_op_data,
    _scrape_bb_data,
    _match_teams_names_historical, 
    _match_teams_names_future,
    LEAGUES_MAPPING
)
from ..config import DEFAULT_CLASSIFIERS

TRAINING_DATA_PATH = join(PATH, 'training_data.csv')
ODDS_DATA_PATH = join(PATH, 'odds_data.csv')
CLF_PATH = join(PATH, 'classifier.pkl')


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

    @staticmethod
    def _validate_leagues(leagues):
        """Validate leagues input."""
        valid_leagues = [league_id for _, (league_id, _), _ in LEAGUES_MAPPING]
        if leagues not in ('all', 'main') and not set(leagues).issubset(valid_leagues):
            msg = "The `leagues` parameter should be either equal to 'all' or 'main' or a list of valid league ids. Got {} instead."
            raise ValueError(msg.format(leagues))

    def fetch_training_data(self, leagues='all'):
        """Fetch the training data."""

        # Validate leagues
        self._validate_leagues(leagues)

        # Define parameters 
        avg_odds_features = ['HomeAverageOdd', 'AwayAverageOdd', 'DrawAverageOdd']
        keys = ['Date', 'League', 'HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals']

        # Fetch data
        spi_data = _fetch_historical_spi_data(leagues)
        fd_data = _fetch_historical_fd_data(leagues)

        # Teams names matching
        mapping = _match_teams_names_historical(spi_data, fd_data)
        spi_data['HomeTeam'] = spi_data['HomeTeam'].apply(lambda team: mapping[team])
        spi_data['AwayTeam'] = spi_data['AwayTeam'].apply(lambda team: mapping[team])

        # Probabilities data
        probs = 1 / fd_data.loc[:, avg_odds_features].values
        probs = pd.DataFrame(probs / probs.sum(axis=1)[:, None], columns=['HomeFDProb', 'AwayFDProb', 'DrawFDProb'])
        probs_data = pd.concat([probs, fd_data], axis=1)
        probs_data.drop(columns=avg_odds_features, inplace=True)

        # Combine data
        training_data = pd.merge(spi_data, probs_data, on=keys)

        # Create features
        training_data['DiffSPIGoals'] = training_data['HomeSPIGoals'] - training_data['AwaySPIGoals']
        training_data['DiffSPI'] = training_data['HomeSPI'] - training_data['AwaySPI']
        training_data['DiffSPIProb'] = training_data['HomeSPIProb'] - training_data['AwaySPIProb']
        training_data['DiffFDProb'] = training_data['HomeFDProb'] - training_data['AwayFDProb']

        # Create day index
        training_data['Day'] = (training_data.Date - min(training_data.Date)).dt.days

        # Sort data
        training_data = training_data.sort_values(keys[:-2]).reset_index(drop=True)

        # Drop features
        training_data.drop(columns=['Date', 'Season', 'League', 'HomeTeam', 'AwayTeam'], inplace=True)

        # Save data
        Path(PATH).mkdir(exist_ok=True)
        training_data.to_csv(TRAINING_DATA_PATH, index=False)

    def fetch_odds_data(self, leagues='all'):
        """Fetch odds data."""

        # Validate leagues
        self._validate_leagues(leagues)

        # Get data
        odds_data = pd.concat([_scrape_op_data(leagues), _scrape_bb_data(leagues)]).reset_index(drop=True)

        # Expand columns
        odds_data[['HomeAverageOdd', 'AwayAverageOdd', 'DrawAverageOdd']] = pd.DataFrame(odds_data.AverageOdd.tolist())
        odds_data[['HomeMaximumOdd', 'AwayMaximumOdd', 'DrawMaximumOdd']] = pd.DataFrame(odds_data.MaximumOdd.tolist())

        # Drop columns
        odds_data.drop(columns=['AverageOdd', 'MaximumOdd'], inplace=True)

        # Save data
        Path(PATH).mkdir(exist_ok=True)
        odds_data.to_csv(ODDS_DATA_PATH, index=False)

    def load_modeling_data(self, predicted_result='A'):
        """Load the data used for modeling."""

        # Load data
        try:
            training_data = pd.read_csv(TRAINING_DATA_PATH)
        except FileNotFoundError:
            raise FileNotFoundError('Training data do not exist. Fetch training data before loading modeling data.')

        # Split and prepare data
        X = training_data.drop(columns=['HomeMaximumOdd', 'AwayMaximumOdd', 'DrawMaximumOdd', 'HomeGoals', 'AwayGoals'])
        X = X[['Day'] + X.columns[:-1].tolist()]
        y = (training_data['HomeGoals'] - training_data['AwayGoals']).apply(lambda sign: 'H' if sign > 0 else 'D' if sign == 0 else 'A')
        y = (y == predicted_result).astype(int)
        odds = training_data.loc[:, {'H': 'HomeMaximumOdd', 'A': 'AwayMaximumOdd', 'D': 'DrawMaximumOdd'}[predicted_result]] 

        # Check arrays
        X, y = check_X_y(X, y, dtype=None)
        odds = check_array(odds, dtype=None, ensure_2d=False)

        return X, y, odds


    def load_predictions_data(self):

        # Load data
        try:
            odds_data = pd.read_csv(ODDS_DATA_PATH)
        except FileNotFoundError:
            raise FileNotFoundError('Odds data do not exist. Fetch odds data before loading predictions data.')

        # Define parameters 
        avg_odds_features = ['HomeAverageOdd', 'AwayAverageOdd', 'DrawAverageOdd']

        # Fetch spi data for future matches
        leagues = odds_data.League.unique().tolist()
        spi_future_data = _fetch_future_spi_data(leagues)

        # Team names matching
        mapping = _match_teams_names_future(spi_future_data, odds_data)
        spi_future_data['HomeTeam'] = spi_future_data['HomeTeam'].apply(lambda team: mapping[team] if team in mapping.keys() else team)
        spi_future_data['AwayTeam'] = spi_future_data['AwayTeam'].apply(lambda team: mapping[team] if team in mapping.keys() else team)

        # Combine data
        predictions_data = pd.merge(odds_data, spi_future_data).drop(columns=['HomeGoals', 'AwayGoals'])

        # Sort values by date
        predictions_data['Date'] = pd.to_datetime(predictions_data['Date'])
        predictions_data = predictions_data.sort_values('Date').reset_index(drop=True)

        # Split data
        X = predictions_data[['HomeSPI', 'AwaySPI', 'HomeSPIProb', 'AwaySPIProb',
                              'DrawSPIProb', 'HomeSPIGoals', 'AwaySPIGoals', 
                              'HomeAverageOdd', 'AwayAverageOdd', 'DrawAverageOdd']]
        odds = predictions_data[['Date', 'League', 'HomeTeam', 'AwayTeam', 'HomeMaximumOdd', 'AwayMaximumOdd', 'DrawMaximumOdd']]

        # Probabilities data
        probs = 1 / X.loc[:, avg_odds_features].values
        probs = pd.DataFrame(probs / probs.sum(axis=1)[:, None], columns=['HomeFDProb', 'AwayFDProb', 'DrawFDProb'])
        X = pd.concat([X, probs], axis=1).drop(columns=avg_odds_features)

        # Create features
        X['DiffSPIGoals'] = X['HomeSPIGoals'] - X['AwaySPIGoals']
        X['DiffSPI'] = X['HomeSPI'] - X['AwaySPI']
        X['DiffSPIProb'] = X['HomeSPIProb'] - X['AwaySPIProb']
        X['DiffFDProb'] = X['HomeFDProb'] - X['AwayFDProb']

        # Check array
        X = check_array(X, dtype=None)

        return X, odds

    def _load_prepare_data(self, predicted_result):

        # Load modelling data
        X, y, odds = self.load_modeling_data(predicted_result)

        # Prepare data
        X = np.hstack((X, odds.reshape(-1, 1)))

        return X, y

    @staticmethod
    def _check_classifier(classifier, fit_params):

        # Check classifier and its fitting parameters
        classifier = ProfitEstimator(classifier) if classifier is not None else ProfitEstimator(DEFAULT_CLASSIFIERS['trivial'][0])
        fit_params = fit_params.copy() if fit_params is not None else DEFAULT_CLASSIFIERS['trivial'][1]

        return classifier, fit_params

    def evaluate_classifier(self, classifier=None, fit_params=None, predicted_result='A', n_splits=5, random_state=None):
        """Evaluate classifier performance using the profit scores."""

        # Load and prepare data
        X, y = self._load_prepare_data(predicted_result)

        # Check classifier and its fitting parameters
        classifier, fit_params = self._check_classifier(classifier, fit_params)

        # Set random state
        set_random_state(classifier, random_state)

        # Run cross-validation
        gscv = GridSearchCV(estimator=classifier, param_grid={}, scoring=['total_profit', 'mean_profit'], 
                            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), 
                            n_jobs=-1, refit=False, iid=True, return_train_score=False)
        gscv.fit(X, y, **fit_params)

        # Extract results
        columns = ['mean_test_total_profit', 'std_test_total_profit', 'mean_test_mean_profit', 'std_test_mean_profit']
        results = pd.DataFrame(gscv.cv_results_)[columns].values.reshape(-1)
        total_profit, mean_profit = results[0:2].tolist(), results[2:].tolist()

        return total_profit, mean_profit

    def fit_dump_classifier(self, predicted_result, classifier, fit_params):
        """Fit and dump a classifier."""

        # Load modelling data
        X, y, _ = self.load_modeling_data(predicted_result)

        # Remove time index
        X = X[:, 1:]

        # Fit classifier
        classifier.fit(X, y, **fit_params)

        # Dump classifier
        with open(CLF_PATH, 'wb') as file:
            dump(classifier, file)

    def backtest(self, classifier=None, fit_params=None, predicted_result='A', test_year=2, max_day_range=6):
        """Apply backtesting to betting agent."""

        # Load and prepare data
        X, y = self._load_prepare_data(predicted_result)

        # Check classifier and its fitting parameters
        classifier, fit_params = self._check_classifier(classifier, fit_params)

        # Define train and test indices
        indices = list(SeasonTimeSeriesSplit(test_year=test_year, max_day_range=max_day_range).split(X, y))

        # Run cross-validation
        self.backtest_results_ = Parallel(n_jobs=-1)(delayed(_fit_predict)(classifier, 
                                                                           X, y, 
                                                                           train_indices, test_indices, 
                                                                           **fit_params)
                                          for train_indices, test_indices in tqdm(indices, desc='Backtesting: '))

    def calculate_backtest_results(self, bet_factor=2, credit_exponent=3):

        # Initialize parameters
        statistics, precisions = [], []
        capital, bet_amount = 1.0, 1.0
        y_test_all, y_pred_all = np.array([]), np.array([])
        

        for y_test, y_pred in self.backtest_results_:

            # Append results and predictions
            y_test_all, y_pred_all = np.hstack((y_test, y_test_all)), np.hstack((y_pred, y_pred_all))
            
            # Convert to binary
            y_pred_bin = (y_pred > 0).astype(int)

            # Calculate number of bets and matches
            n_bets = y_pred_bin.sum()
            n_matches = y_pred.size

            # Calculate precision
            precision = precision_score(y_test, y_pred_bin) if n_bets > 0 else np.nan
            precisions.append(precision)

            # Calculate profit
            profit = bet_amount * total_profit_score(y_test, y_pred) / n_bets if n_bets > 0 else 0.0
            
            # Calculate capital
            capital += profit

            # Adjust bet amount
            bet_amount = bet_amount * bet_factor if profit < 0.0 else 1.0

            # Calculate credit
            max_credit = capital + bet_factor ** credit_exponent
            if bet_amount > max_credit:
                bet_amount = max_credit
                
            # Generate statistic
            statistic = (capital, profit, bet_amount, n_bets, n_matches, precision)

            # Append statistic
            statistics.append(statistic)

            if bet_amount == 0:
                break

        # Define statistics dataframe
        statistics = pd.DataFrame(statistics, columns=['Capital', 'Profit', 'Bet amount', 'Bets', 'Matches', 'Precision'])

        # Define attributes
        mean_precision = np.nanmean(precisions)
        profit_per_bet = mean_profit_score(y_test_all, y_pred_all)

        return statistics, mean_precision, profit_per_bet
